from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
import random
import time
from typing import Callable, Iterable, Optional, Sequence

from .models import (
    PolicyAction,
    PolicyActionOption,
    SimulationConfig,
    SimulationState,
)
from . import simulator
from .oracle import (
    OracleElectionLoss,
    OracleSearchResult,
    score_election_state,
    score_simulation_state,
    validate_score,
)


class BaseAgent:
    """Minimal agent loop scaffold that can be subclassed for custom strategies."""

    def __init__(
        self,
        country: str = "uk",
        gamedata_root: Optional[str | Path] = None,
        state: Optional[SimulationState] = None,
        config: Optional[SimulationConfig] = None,
    ) -> None:
        self.gamedata_root = str(gamedata_root) if gamedata_root else None
        self.data = simulator.load_simulation_data(self.gamedata_root)
        self.config = config
        if state is None:
            self.state, self.graph = simulator.get_initial_state(
                country, self.gamedata_root
            )
        else:
            self.state = state
            self.graph = simulator.build_country_graph(
                state.country, self.gamedata_root
            )

    def available_actions(self):
        return simulator.list_available_actions(self.state, data=self.data)

    def choose_actions(self, state: SimulationState, options) -> Iterable[PolicyAction]:
        """Override to select actions before ending the turn. Default: no actions."""

        return []

    def apply_actions(self, actions: Iterable[PolicyAction]):
        if not actions:
            return
        self.state = simulator.apply_actions(self.state, actions, data=self.data)

    def end_turn(self):
        self.state = simulator.process_end_of_turn(
            self.state, self.graph, data=self.data, config=self.config
        )

    def step(self) -> SimulationState:
        actions = list(self.choose_actions(self.state, self.available_actions()))
        self.apply_actions(actions)
        self.end_turn()
        return self.state

    def load_state(self, path: str | Path) -> None:
        self.state = simulator.load_state(path)
        self.graph = simulator.build_country_graph(
            self.state.country, self.gamedata_root
        )

    def save_state(self, path: str | Path) -> None:
        simulator.save_state(self.state, path)


class PassiveAgent(BaseAgent):
    """Baseline agent that never spends political capital."""

    def choose_actions(self, state: SimulationState, options):
        return []


@dataclass(frozen=True, slots=True)
class _SimulatorBeamEntry:
    state: SimulationState
    plan: tuple[tuple[PolicyAction, ...], ...]
    score: float
    first_state: SimulationState | None = None


class SimulatorOracleAgent(BaseAgent):
    """Beam-search agent that evaluates every branch with the simulator.

    Each search depth is one decision turn.  A branch applies a legal batch of
    one or more ``PolicyAction`` objects (or no action), then calls the real
    simulator turn transition.  The agent executes only the first move of the
    winning plan and searches again next turn, so forecasts never replace
    observed state.

    ``candidate_limit`` is a deliberate safety valve: the UK data exposes a
    large action roster and a full search can be expensive.  ``None`` means
    exhaustive action enumeration.  The default objective is a weighted
    headline score; pass ``objective`` for a game-specific definition of best.
    """

    def __init__(
        self,
        country: str = "uk",
        gamedata_root: Optional[str | Path] = None,
        state: Optional[SimulationState] = None,
        config: Optional[SimulationConfig] = None,
        *,
        beam_width: int | None = 4,
        search_horizon: int | None = 2,
        candidate_limit: int | None = 32,
        max_actions_per_turn: int = 1,
        batch_candidate_limit: int | None = None,
        time_budget_seconds: float | None = None,
        random_seed: int | None = None,
        objective: Callable[[SimulationState], float] = score_simulation_state,
    ) -> None:
        super().__init__(
            country=country,
            gamedata_root=gamedata_root,
            state=state,
            config=config,
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
        self.beam_width = beam_width
        self.search_horizon = search_horizon
        self.candidate_limit = candidate_limit
        self.max_actions_per_turn = max_actions_per_turn
        self.batch_candidate_limit = batch_candidate_limit
        self.time_budget_seconds = time_budget_seconds
        self.random_seed = random_seed
        self._random = random.Random(random_seed) if random_seed is not None else None
        self.objective = objective
        self.last_search: OracleSearchResult[SimulationState] | None = None

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
        options = list(
            supplied
            if supplied is not None
            else simulator.list_available_actions(state, data=self.data)
        )
        if self.candidate_limit is None or len(options) <= self.candidate_limit:
            return options
        if self._random is not None:
            return self._random.sample(options, self.candidate_limit)
        # A bounded search must still be deterministic when no seed is given.
        # No-op is always retained separately, so this cap applies only to
        # policy moves.
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
    ) -> list[tuple[PolicyAction, ...]]:
        """Enumerate legal one-turn action batches.

        A policy may occur at most once in a batch because the native order
        protocol and the simulator both define each option relative to the
        start-of-turn slider.  The capital check is applied before launching a
        branch, while ``apply_actions`` remains the final legality check.
        ``batch_candidate_limit`` bounds combinations after the no-op branch;
        ``None`` is exhaustive for the supplied option set.
        """

        batches: list[tuple[PolicyAction, ...]] = [()]
        max_size = min(self.max_actions_per_turn, len(options))
        single_batches = [
            (self._action(option),)
            for option in options
            if option.cost <= state.political_capital + simulator.EPSILON
        ]
        batches.extend(single_batches)
        combination_batches: list[tuple[PolicyAction, ...]] = []
        for size in range(2, max_size + 1):
            for option_batch in combinations(options, size):
                policy_names = [option.policy_name for option in option_batch]
                if len(set(policy_names)) != len(policy_names):
                    continue
                if sum(option.cost for option in option_batch) > (
                    state.political_capital + simulator.EPSILON
                ):
                    continue
                combination_batches.append(
                    tuple(self._action(option) for option in option_batch)
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

    def _resolved_search_horizon(self, state: SimulationState) -> int:
        """Resolve ``None`` to the number of turns before the next election."""

        if self.search_horizon is not None:
            return self.search_horizon
        return max(1, state.election_turns_until)

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

    def _turn_transition(self, state: SimulationState) -> SimulationState:
        """Advance one simulated turn and resolve an election boundary."""
        advanced = simulator.process_end_of_turn(
            state,
            self.graph,
            data=self.data,
            config=self.config,
        )
        return simulator.resolve_election_if_ready(advanced, data=self.data)

    def _prepare_root(self, state: SimulationState) -> SimulationState:
        """Resolve a save captured exactly at a pending election boundary."""
        return simulator.resolve_election_if_ready(state, data=self.data)

    def search(
        self,
        state: SimulationState | None = None,
        *,
        options: Sequence[PolicyActionOption] | None = None,
    ) -> OracleSearchResult[SimulationState]:
        """Evaluate a beam of simulator-backed future action sequences."""

        root = self._prepare_root(state if state is not None else self.state)
        if root.election_result == "loss":
            raise OracleElectionLoss(root)
        beam = [
            _SimulatorBeamEntry(
                state=root,
                plan=(),
                score=validate_score(self.objective, root),
            )
        ]
        evaluated = 0
        started = time.monotonic()
        deadline = (
            started + self.time_budget_seconds
            if self.time_budget_seconds is not None
            else None
        )
        timed_out = False
        completed_depth = 0

        for depth in range(self._resolved_search_horizon(root)):
            expanded: list[_SimulatorBeamEntry] = []
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
                for actions in candidates:
                    if deadline is not None and time.monotonic() >= deadline:
                        timed_out = True
                        break
                    ordered = entry.state
                    if actions:
                        try:
                            ordered = simulator.apply_actions(
                                ordered,
                                actions,
                                data=self.data,
                            )
                        except ValueError:
                            # A branch can become invalid after an earlier
                            # forecast spent capital; it is simply not legal.
                            continue
                    next_state = self._turn_transition(ordered)
                    evaluated += 1
                    if next_state.election_result == "loss":
                        losses.append(next_state)
                        continue
                    plan = (*entry.plan, actions)
                    candidate = _SimulatorBeamEntry(
                        state=next_state,
                        plan=plan,
                        score=validate_score(self.objective, next_state),
                        first_state=(
                            next_state
                            if depth == 0
                            else entry.first_state
                        ),
                    )
                    if self.beam_width is None:
                        expanded.append(candidate)
                    else:
                        # Keep the beam bounded while expanding.  A UK
                        # branch owns 2,000 voter records, so retaining every
                        # child until the end of a depth can exhaust memory
                        # long before the configured wall-clock budget.
                        expanded.append(candidate)
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
                raise RuntimeError("oracle search produced no legal simulator branches")
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

        if not beam[0].first_state:
            # A very small budget can expire while assembling the first
            # branch.  Evaluate a no-op so the caller still receives a real,
            # safe first-turn transition rather than an unusable root result.
            fallback_state = self._turn_transition(root)
            evaluated += 1
            if fallback_state.election_result == "loss":
                raise OracleElectionLoss(fallback_state)
            beam = [
                _SimulatorBeamEntry(
                    state=fallback_state,
                    plan=((),),
                    score=validate_score(self.objective, fallback_state),
                    first_state=fallback_state,
                )
            ]
            completed_depth = max(completed_depth, 1)
        winner = beam[0]
        if winner.first_state is None:
            raise RuntimeError("oracle search did not produce a first-turn state")
        result = OracleSearchResult(
            plan=winner.plan,
            score=winner.score,
            state=winner.state,
            first_state=winner.first_state,
            evaluated=evaluated,
            elapsed_seconds=time.monotonic() - started,
            completed_depth=completed_depth,
            timed_out=timed_out,
        )
        self.last_search = result
        return result

    def choose_actions(
        self,
        state: SimulationState,
        options: Sequence[PolicyActionOption],
    ) -> Iterable[PolicyAction]:
        """Search ahead and return only the winning plan's first turn."""

        return self.search(state, options=options).first_actions

    def end_turn(self) -> None:
        """Apply the real transition and stop immediately on an election loss."""
        self.state = self._turn_transition(self.state)
        if self.state.election_result == "loss":
            raise OracleElectionLoss(self.state)

    def step(self) -> SimulationState:
        """Resolve a boundary save before acting, then execute one safe turn."""
        if self.state.election_turns_until == 0:
            self.state = self._prepare_root(self.state)
            if self.state.election_result == "loss":
                raise OracleElectionLoss(self.state)
        return super().step()


class ElectionOracleAgent(SimulatorOracleAgent):
    """Full-term simulator oracle optimized for the first election margin.

    This is the practical best-case baseline: it searches to the next
    election by default, considers every available option, permits two policy
    changes in one turn, and ranks branches by expected native-style vote
    margin.  The combination space is intentionally configurable because a
    completely exhaustive UK search is much larger than one game process can
    explore interactively.
    """

    def __init__(
        self,
        country: str = "uk",
        gamedata_root: Optional[str | Path] = None,
        state: Optional[SimulationState] = None,
        config: Optional[SimulationConfig] = None,
        *,
        beam_width: int | None = 6,
        search_horizon: int | None = None,
        candidate_limit: int | None = None,
        max_actions_per_turn: int = 2,
        batch_candidate_limit: int | None = None,
        time_budget_seconds: float | None = 600.0,
        random_seed: int | None = None,
    ) -> None:
        super().__init__(
            country=country,
            gamedata_root=gamedata_root,
            state=state,
            config=config,
            beam_width=beam_width,
            search_horizon=search_horizon,
            candidate_limit=candidate_limit,
            max_actions_per_turn=max_actions_per_turn,
            batch_candidate_limit=batch_candidate_limit,
            time_budget_seconds=time_budget_seconds,
            random_seed=random_seed,
            objective=score_election_state,
        )


# Short name for callers that only need the simulator implementation.
OracleAgent = SimulatorOracleAgent
