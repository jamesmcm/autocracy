"""Best-case beam search against the installed Democracy 3 executable.

Unlike the simulator oracle, this agent does not forecast a native turn from
Python state.  Every branch is sent through :mod:`gamedrive.inject_drive`, its
fresh XML output is parsed, and the parsed save is the branch state used by the
next beam layer.  This is intentionally expensive but gives the search the
real game's answer for each evaluated action.
"""

from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, replace
import io
from itertools import combinations
import os
from pathlib import Path
import random
import re
import shutil
import time
from typing import Callable, Sequence

from autocracy import simulator
from autocracy.models import PolicyAction, PolicyActionOption, SimulationState
from autocracy.oracle import (
    OracleElectionLoss,
    OracleSearchResult,
    score_savegame,
    score_savegame_election,
    validate_score,
)
from autocracy.savegame import (
    ENCODING,
    SaveGame,
    load_state_from_savegame,
    parse_savegame,
)

from .inject_drive import GAME, PROBE, SAVE_ROOT, run as native_run
from .order_plan import NativeOrder, encode_orders


NativeRunner = Callable[..., int]


class _NativeElectionLoss(RuntimeError):
    """Internal marker used to discard one native branch after a loss."""

    def __init__(self, state: SimulationState) -> None:
        self.state = state
        super().__init__("native branch lost its election")


_ELECTION_BLOCK_RE = re.compile(r"<election>.*?</election>", re.DOTALL)
_VOTER_BLOCK_RE = re.compile(r"<voter>.*?</voter>", re.DOTALL)


def _replace_native_tag(block: str, tag: str, value: int) -> str:
    pattern = re.compile(rf"(<{tag}>)[^<]*(</{tag}>)")
    updated, count = pattern.subn(rf"\g<1>{value}\g<2>", block, count=1)
    if count != 1:
        raise ValueError(f"native save has no single <{tag}> field")
    return updated


def _write_resolved_election_save(path: Path, state: SimulationState) -> None:
    """Persist the simulator's result into a temporary native checkpoint.

    Headless GameDrive stops immediately at ``turnsuntilelection == 0`` and
    has no result-screen entrypoint.  The next native branch therefore needs a
    small, isolated checkpoint edit: the new term/countdown and the per-voter
    vote enums.  The original source save is never edited.
    """
    raw = path.read_text(encoding=ENCODING)
    election_match = _ELECTION_BLOCK_RE.search(raw)
    if election_match is None:
        raise ValueError(f"native save has no election block: {path}")
    election = election_match.group(0)
    election = _replace_native_tag(
        election, "turnsuntilelection", state.election_turns_until
    )
    election = _replace_native_tag(election, "currentterm", state.election_current_term)
    raw = raw[: election_match.start()] + election + raw[election_match.end() :]

    voter_matches = list(_VOTER_BLOCK_RE.finditer(raw))
    if len(voter_matches) < len(state.voters):
        raise ValueError(
            "native save has fewer voter records than the resolved simulator state"
        )
    chunks: list[str] = []
    cursor = 0
    for index, match in enumerate(voter_matches):
        chunks.append(raw[cursor : match.start()])
        voter_block = match.group(0)
        if index < len(state.voters):
            voter_block = _replace_native_tag(
                voter_block, "lastvote", state.voters[index].last_vote
            )
        chunks.append(voter_block)
        cursor = match.end()
    chunks.append(raw[cursor:])
    path.write_text("".join(chunks), encoding=ENCODING)


def native_order_for_option(option: PolicyActionOption) -> NativeOrder:
    """Translate one simulator action option to the native order protocol."""

    if option.action_type == "introduce":
        return NativeOrder("implement", option.policy_name, option.resulting_level)
    if option.action_type == "cancel":
        return NativeOrder("cancel", option.policy_name)
    return NativeOrder("slider", option.policy_name, option.resulting_level)


@dataclass(frozen=True, slots=True)
class _NativeBeamEntry:
    save_name: str
    save: SaveGame
    state: SimulationState
    plan: tuple[tuple[PolicyAction, ...], ...]
    score: float
    artifacts: tuple[str, ...]
    first_save: SaveGame | None = None
    first_state: SimulationState | None = None


class GameDriveOracleAgent:
    """Beam-search agent whose transition function is the native game.

    ``load_name`` must identify a save in ``save_root`` and the native probe
    must be built for the installed game.  The default ``candidate_limit`` is
    conservative because one evaluated branch launches a fresh gdb/Xvfb game
    process.  Set it to ``None`` for exhaustive enumeration of every legal
    action returned by ``list_available_actions``.
    """

    def __init__(
        self,
        load_name: str,
        *,
        save_root: str | Path = SAVE_ROOT,
        gamedata_root: str | Path | None = None,
        game: str | Path = GAME,
        probe: str | Path = PROBE,
        beam_width: int | None = 2,
        search_horizon: int | None = 2,
        candidate_limit: int | None = 16,
        max_actions_per_turn: int = 1,
        batch_candidate_limit: int | None = None,
        time_budget_seconds: float | None = None,
        random_seed: int | None = None,
        timeout: int = 120,
        turn_mode: str = "sync",
        objective: Callable[[SaveGame], float] = score_savegame,
        runner: NativeRunner | None = None,
    ) -> None:
        self.save_root = Path(save_root)
        self.current_save_name = str(load_name)
        self.current_save_path = self.save_root / f"{self.current_save_name}.xml"
        if not self.current_save_path.is_file():
            raise FileNotFoundError(self.current_save_path)
        self.data = simulator.load_simulation_data(
            str(gamedata_root) if gamedata_root is not None else None
        )
        self.game = Path(game)
        self.probe = Path(probe)
        self.save = parse_savegame(self.current_save_path)
        self.state, self.graph = load_state_from_savegame(
            self.current_save_path,
            data=self.data,
        )
        if beam_width is not None and beam_width < 1:
            raise ValueError("beam_width must be at least 1 or None")
        if search_horizon is not None and search_horizon < 1:
            raise ValueError("search_horizon must be at least 1 or None")
        if candidate_limit is not None and candidate_limit < 1:
            raise ValueError("candidate_limit must be at least 1 or None")
        if max_actions_per_turn < 1:
            raise ValueError("max_actions_per_turn must be at least 1")
        if batch_candidate_limit is not None and batch_candidate_limit < 1:
            raise ValueError("batch_candidate_limit must be at least 1 or None")
        if time_budget_seconds is not None and time_budget_seconds <= 0.0:
            raise ValueError("time_budget_seconds must be greater than zero or None")
        if timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        if turn_mode not in {"sync", "direct", "async"}:
            raise ValueError(f"unknown turn mode: {turn_mode}")
        self.beam_width = beam_width
        self.search_horizon = search_horizon
        self.candidate_limit = candidate_limit
        self.max_actions_per_turn = max_actions_per_turn
        self.batch_candidate_limit = batch_candidate_limit
        self.time_budget_seconds = time_budget_seconds
        self.random_seed = random_seed
        self._random = random.Random(random_seed) if random_seed is not None else None
        self.timeout = timeout
        self.turn_mode = turn_mode
        self.objective = objective
        self.runner = runner or native_run
        self.last_search: OracleSearchResult[SaveGame] | None = None
        self._counter = 0
        self._owned_artifacts: set[str] = set()

    def available_actions(self) -> list[PolicyActionOption]:
        """List legal moves from the native save's current slider targets."""

        # Native saves carry both the current implementation value and the
        # requested slider target.  Orders are issued against the target the
        # player sees, so use that map while asking the common action enumerator
        # for costs and action types.
        target_levels = self.state.policy_desired_throttles
        action_state = (
            replace(self.state, policies=target_levels.copy())
            if target_levels
            else self.state
        )
        return simulator.list_available_actions(action_state, data=self.data)

    @staticmethod
    def _action(option: PolicyActionOption) -> PolicyAction:
        return PolicyAction(
            policy_name=option.policy_name,
            delta=option.delta,
            action_type=option.action_type,
        )

    def _options(
        self,
        state: SimulationState,
        supplied: Sequence[PolicyActionOption] | None = None,
    ) -> list[PolicyActionOption]:
        target_levels = state.policy_desired_throttles
        action_state = (
            replace(state, policies=target_levels.copy())
            if target_levels
            else state
        )
        options = list(
            supplied
            if supplied is not None
            else simulator.list_available_actions(action_state, data=self.data)
        )
        if self.candidate_limit is None or len(options) <= self.candidate_limit:
            return options
        if self._random is not None:
            return self._random.sample(options, self.candidate_limit)
        return sorted(
            options,
            key=lambda option: (
                option.policy_name,
                option.action_type,
                option.resulting_level,
            ),
        )[: self.candidate_limit]

    def _action_batches(
        self,
        state: SimulationState,
        options: Sequence[PolicyActionOption],
    ) -> list[tuple[tuple[PolicyAction, ...], tuple[PolicyActionOption, ...]]]:
        """Enumerate native order batches legal from the same turn state."""

        batches: list[
            tuple[tuple[PolicyAction, ...], tuple[PolicyActionOption, ...]]
        ] = [((), ())]
        max_size = min(self.max_actions_per_turn, len(options))
        single_batches = [
            ((self._action(option),), (option,))
            for option in options
            if option.cost <= state.political_capital + simulator.EPSILON
        ]
        batches.extend(single_batches)
        combination_batches: list[
            tuple[tuple[PolicyAction, ...], tuple[PolicyActionOption, ...]]
        ] = []
        for size in range(2, max_size + 1):
            for option_batch in combinations(options, size):
                names = [option.policy_name for option in option_batch]
                if len(set(names)) != len(names):
                    continue
                if sum(option.cost for option in option_batch) > (
                    state.political_capital + simulator.EPSILON
                ):
                    continue
                combination_batches.append(
                    (
                        tuple(self._action(option) for option in option_batch),
                        tuple(option_batch),
                    )
                )
        if self.batch_candidate_limit is None:
            batches.extend(combination_batches)
        else:
            remaining = max(0, self.batch_candidate_limit - len(single_batches))
            if remaining:
                if self._random is not None and len(combination_batches) > remaining:
                    combination_batches = self._random.sample(
                        combination_batches,
                        remaining,
                    )
                batches.extend(combination_batches[:remaining])
        return batches

    def _resolved_search_horizon(self) -> int:
        """Resolve ``None`` to the number of turns before the next election."""

        if self.search_horizon is not None:
            return self.search_horizon
        return max(1, self.state.election_turns_until or 0)

    @staticmethod
    def _plan_key(
        plan: tuple[tuple[PolicyAction, ...], ...],
    ) -> tuple:
        return tuple(
            tuple(
                (
                    action.policy_name,
                    action.action_type or "",
                    round(action.delta, 9),
                )
                for action in actions
            )
            for actions in plan
        )

    def _fresh_names(self) -> tuple[str, str, str]:
        self._counter += 1
        base = f"autocracy_oracle_{os.getpid()}_{time.time_ns()}_{self._counter}"
        return f"{base}_loaded", f"{base}_turn", f"{base}_edited"

    def _cleanup(self, names: set[str], *, keep: set[str] | None = None) -> None:
        retained = keep or set()
        for name in names - retained:
            path = self.save_root / f"{name}.xml"
            try:
                path.unlink()
            except FileNotFoundError:
                pass

    def _clear_old_artifacts(self) -> None:
        keep = {self.current_save_name}
        self._cleanup(self._owned_artifacts, keep=keep)
        self._owned_artifacts.intersection_update(keep)

    def _prepare_root(self) -> None:
        """Materialize a resolved checkpoint if the input is at election zero."""
        if self.state.election_turns_until != 0:
            return
        resolved = simulator.resolve_election_if_ready(self.state, data=self.data)
        if resolved.election_result == "loss":
            raise OracleElectionLoss(resolved)
        _, normalized_name, _ = self._fresh_names()
        normalized_path = self.save_root / f"{normalized_name}.xml"
        shutil.copyfile(self.current_save_path, normalized_path)
        try:
            _write_resolved_election_save(normalized_path, resolved)
            self.save = parse_savegame(normalized_path)
        except Exception:
            try:
                normalized_path.unlink()
            except FileNotFoundError:
                pass
            raise
        self.current_save_name = normalized_name
        self.current_save_path = normalized_path
        self.state = resolved
        self._owned_artifacts.add(normalized_name)

    def _evaluate_native_turn(
        self,
        parent: _NativeBeamEntry,
        actions: tuple[PolicyAction, ...],
        options: Sequence[PolicyActionOption],
        generated: set[str],
    ) -> tuple[_NativeBeamEntry, set[str]]:
        loaded_name, output_name, edited_name = self._fresh_names()
        generated.update((loaded_name, output_name))
        native_orders = [native_order_for_option(option) for option in options]
        order_spec = encode_orders(native_orders) or None

        try:
            # inject_drive prints the native probe transcript.  Search can
            # evaluate dozens of branches, so keep that diagnostic noise out
            # of ordinary agent output; a failed run is raised below.
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                exit_code = self.runner(
                    load_name=parent.save_name,
                    loaded_name=loaded_name,
                    after_turn_name=output_name,
                    edited_name=edited_name,
                    edit_node=None,
                    edit_value=None,
                    turn_mode=self.turn_mode,
                    gameplay_turn=False,
                    sync_gameplay_turn=False,
                    skip_turn=False,
                    timeout=self.timeout,
                    order_spec=order_spec,
                    capture_specs=None,
                    capture_prefix=None,
                    orders_save_name=None,
                    manager_audit_path=None,
                    manager_save_name=None,
                    allow_existing=False,
                    game=Path(self.game),
                    probe=Path(self.probe),
                    save_root=self.save_root,
                )
            if exit_code not in (None, 0):
                raise RuntimeError(
                    f"native oracle branch failed with exit code {exit_code}"
                )
            output_path = self.save_root / f"{output_name}.xml"
            save = parse_savegame(output_path)
            state, _ = load_state_from_savegame(output_path, data=self.data)
            resolved_state = simulator.resolve_election_if_ready(
                state, data=self.data
            )
            if resolved_state.election_result == "loss":
                raise _NativeElectionLoss(resolved_state)
            if resolved_state is not state:
                _write_resolved_election_save(output_path, resolved_state)
                save = parse_savegame(output_path)
                state = resolved_state
        except Exception:
            self._cleanup(generated)
            raise

        plan = (*parent.plan, actions)
        artifacts = (*parent.artifacts, output_name)
        return (
            _NativeBeamEntry(
                save_name=output_name,
                save=save,
                state=state,
                plan=plan,
                score=validate_score(self.objective, save),
                artifacts=artifacts,
                first_save=(save if not parent.plan else parent.first_save),
                first_state=(state if not parent.plan else parent.first_state),
            ),
            generated,
        )

    def search(
        self,
        *,
        options: Sequence[PolicyActionOption] | None = None,
    ) -> OracleSearchResult[SaveGame]:
        """Evaluate native game results for every branch retained by the beam."""

        self._clear_old_artifacts()
        self._prepare_root()
        if self.state.election_result == "loss":
            raise OracleElectionLoss(self.state)
        root = _NativeBeamEntry(
            save_name=self.current_save_name,
            save=self.save,
            state=self.state,
            plan=(),
            score=validate_score(self.objective, self.save),
            artifacts=(),
        )
        beam = [root]
        generated: set[str] = set()
        evaluated = 0
        started = time.monotonic()
        deadline = (
            started + self.time_budget_seconds
            if self.time_budget_seconds is not None
            else None
        )
        timed_out = False
        completed_depth = 0

        try:
            for depth in range(self._resolved_search_horizon()):
                expanded: list[_NativeBeamEntry] = []
                losses: list[SimulationState] = []
                for entry in beam:
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        break
                    branch_options = self._options(
                        entry.state,
                        supplied=options if depth == 0 and entry is beam[0] else None,
                    )
                    candidates = self._action_batches(entry.state, branch_options)
                    for actions, native_options in candidates:
                        if deadline is not None and time.monotonic() >= deadline:
                            timed_out = True
                            break
                        try:
                            child, child_generated = self._evaluate_native_turn(
                                entry,
                                actions,
                                native_options,
                                set(),
                            )
                        except _NativeElectionLoss as loss:
                            losses.append(loss.state)
                            evaluated += 1
                            continue
                        generated.update(child_generated)
                        evaluated += 1
                        if self.beam_width is None:
                            expanded.append(child)
                        else:
                            # Native branch states carry the complete parsed
                            # voter population too; bound retained children
                            # during expansion rather than after a whole
                            # depth has been materialized.
                            expanded.append(child)
                            expanded.sort(
                                key=lambda item: (
                                    -item.score,
                                    self._plan_key(item.plan),
                                )
                            )
                            del expanded[self.beam_width :]
                    if timed_out:
                        break
                if not expanded:
                    if timed_out:
                        break
                    if losses:
                        raise OracleElectionLoss(losses[0])
                    raise RuntimeError("oracle search produced no native branches")
                expanded.sort(
                    key=lambda entry: (-entry.score, self._plan_key(entry.plan))
                )
                beam = (
                    expanded
                    if self.beam_width is None
                    else expanded[: self.beam_width]
                )
                completed_depth = depth + 1
                if timed_out:
                    break

            if beam[0].first_state is None:
                fallback, fallback_generated = self._evaluate_native_turn(
                    root,
                    (),
                    (),
                    set(),
                )
                generated.update(fallback_generated)
                evaluated += 1
                beam = [fallback]
                completed_depth = max(completed_depth, 1)

            winner = beam[0]
            if winner.first_save is None or winner.first_state is None:
                raise RuntimeError("native oracle search did not produce a first turn")
            result = OracleSearchResult(
                plan=winner.plan,
                score=winner.score,
                state=winner.save,
                first_state=winner.first_save,
                evaluated=evaluated,
                first_artifact=winner.artifacts[0],
                artifacts=winner.artifacts,
                first_runtime_state=winner.first_state,
                elapsed_seconds=time.monotonic() - started,
                completed_depth=completed_depth,
                timed_out=timed_out,
            )
            self._cleanup(generated, keep=set(winner.artifacts))
            self._owned_artifacts = set(winner.artifacts)
            self.last_search = result
            return result
        except Exception:
            self._cleanup(generated)
            raise

    def choose_actions(self) -> tuple[PolicyAction, ...]:
        """Search and return the first native order batch as policy actions."""

        return self.search().first_actions

    def commit_result(self, result: OracleSearchResult[SaveGame]) -> SimulationState:
        """Commit an already-evaluated winning result without rerunning it."""

        if result.first_artifact is None:
            raise RuntimeError("native oracle result has no continuation save")
        self.current_save_name = result.first_artifact
        self.current_save_path = self.save_root / f"{self.current_save_name}.xml"
        self.save = result.first_state
        if result.first_runtime_state is not None:
            self.state = result.first_runtime_state
        else:
            self.state, self.graph = load_state_from_savegame(
                self.current_save_path,
                data=self.data,
            )
        return self.state

    def step(self) -> SimulationState:
        """Search, then commit the first native turn from the winning path."""

        return self.commit_result(self.search())


class ElectionGameDriveOracleAgent(GameDriveOracleAgent):
    """Native GameDrive oracle with the simulator election-margin objective."""

    def __init__(
        self,
        load_name: str,
        *,
        save_root: str | Path = SAVE_ROOT,
        gamedata_root: str | Path | None = None,
        game: str | Path = GAME,
        probe: str | Path = PROBE,
        beam_width: int | None = 6,
        search_horizon: int | None = None,
        candidate_limit: int | None = None,
        max_actions_per_turn: int = 2,
        batch_candidate_limit: int | None = None,
        time_budget_seconds: float | None = 600.0,
        random_seed: int | None = None,
        timeout: int = 120,
        turn_mode: str = "sync",
        runner: NativeRunner | None = None,
    ) -> None:
        super().__init__(
            load_name,
            save_root=save_root,
            gamedata_root=gamedata_root,
            game=game,
            probe=probe,
            beam_width=beam_width,
            search_horizon=search_horizon,
            candidate_limit=candidate_limit,
            max_actions_per_turn=max_actions_per_turn,
            batch_candidate_limit=batch_candidate_limit,
            time_budget_seconds=time_budget_seconds,
            random_seed=random_seed,
            timeout=timeout,
            turn_mode=turn_mode,
            objective=score_savegame_election,
            runner=runner,
        )
