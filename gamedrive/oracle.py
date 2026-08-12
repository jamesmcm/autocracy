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
import os
from pathlib import Path
import random
import time
from typing import Callable, Sequence

from autocracy import simulator
from autocracy.models import PolicyAction, PolicyActionOption, SimulationState
from autocracy.oracle import (
    OracleSearchResult,
    score_savegame,
    validate_score,
)
from autocracy.savegame import SaveGame, load_state_from_savegame, parse_savegame

from .inject_drive import GAME, PROBE, SAVE_ROOT, run as native_run
from .order_plan import NativeOrder, encode_orders


NativeRunner = Callable[..., int]


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
        beam_width: int = 2,
        search_horizon: int = 2,
        candidate_limit: int | None = 16,
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
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if search_horizon < 1:
            raise ValueError("search_horizon must be at least 1")
        if candidate_limit is not None and candidate_limit < 1:
            raise ValueError("candidate_limit must be at least 1 or None")
        if timeout < 1:
            raise ValueError("timeout must be at least 1 second")
        if turn_mode not in {"sync", "direct", "async"}:
            raise ValueError(f"unknown turn mode: {turn_mode}")
        self.beam_width = beam_width
        self.search_horizon = search_horizon
        self.candidate_limit = candidate_limit
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

        try:
            for depth in range(self.search_horizon):
                expanded: list[_NativeBeamEntry] = []
                for entry in beam:
                    branch_options = self._options(
                        entry.state,
                        supplied=options if depth == 0 and entry is beam[0] else None,
                    )
                    candidates: list[tuple[tuple[PolicyAction, ...], tuple[PolicyActionOption, ...]]] = [
                        ((), ())
                    ]
                    candidates.extend(
                        ((self._action(option),), (option,))
                        for option in branch_options
                    )
                    for actions, native_options in candidates:
                        child, child_generated = self._evaluate_native_turn(
                            entry,
                            actions,
                            native_options,
                            set(),
                        )
                        generated.update(child_generated)
                        evaluated += 1
                        expanded.append(child)
                if not expanded:
                    raise RuntimeError("oracle search produced no native branches")
                expanded.sort(
                    key=lambda entry: (-entry.score, self._plan_key(entry.plan))
                )
                beam = expanded[: self.beam_width]

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
        self.state, self.graph = load_state_from_savegame(
            self.current_save_path,
            data=self.data,
        )
        return self.state

    def step(self) -> SimulationState:
        """Search, then commit the first native turn from the winning path."""

        return self.commit_result(self.search())
