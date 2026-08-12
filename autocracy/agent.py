from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
from typing import Callable, Iterable, Optional, Sequence

from .models import (
    PolicyAction,
    PolicyActionOption,
    SimulationConfig,
    SimulationState,
)
from . import simulator
from .oracle import OracleSearchResult, score_simulation_state, validate_score


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

    Each search depth is one decision turn.  A branch applies one legal
    ``PolicyAction`` (or no action), then calls the real simulator turn
    transition.  The agent executes only the first move of the winning plan
    and searches again next turn, so forecasts never replace observed state.

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
        beam_width: int = 4,
        search_horizon: int = 2,
        candidate_limit: int | None = 32,
        random_seed: int | None = None,
        objective: Callable[[SimulationState], float] = score_simulation_state,
    ) -> None:
        super().__init__(
            country=country,
            gamedata_root=gamedata_root,
            state=state,
            config=config,
        )
        if beam_width < 1:
            raise ValueError("beam_width must be at least 1")
        if search_horizon < 1:
            raise ValueError("search_horizon must be at least 1")
        if candidate_limit is not None and candidate_limit < 1:
            raise ValueError("candidate_limit must be at least 1 or None")
        self.beam_width = beam_width
        self.search_horizon = search_horizon
        self.candidate_limit = candidate_limit
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

    def search(
        self,
        state: SimulationState | None = None,
        *,
        options: Sequence[PolicyActionOption] | None = None,
    ) -> OracleSearchResult[SimulationState]:
        """Evaluate a beam of simulator-backed future action sequences."""

        root = state if state is not None else self.state
        beam = [
            _SimulatorBeamEntry(
                state=root,
                plan=(),
                score=validate_score(self.objective, root),
            )
        ]
        evaluated = 0

        for depth in range(self.search_horizon):
            expanded: list[_SimulatorBeamEntry] = []
            for entry in beam:
                branch_options = self._options(
                    entry.state,
                    supplied=options if depth == 0 and entry is beam[0] else None,
                )
                candidates: list[tuple[PolicyAction, ...]] = [()]
                candidates.extend((self._action(option),) for option in branch_options)
                for actions in candidates:
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
                    next_state = simulator.process_end_of_turn(
                        ordered,
                        self.graph,
                        data=self.data,
                        config=self.config,
                    )
                    evaluated += 1
                    plan = (*entry.plan, actions)
                    expanded.append(
                        _SimulatorBeamEntry(
                            state=next_state,
                            plan=plan,
                            score=validate_score(self.objective, next_state),
                            first_state=(
                                next_state
                                if depth == 0
                                else entry.first_state
                            ),
                        )
                    )
            if not expanded:
                raise RuntimeError("oracle search produced no legal simulator branches")
            expanded.sort(
                key=lambda entry: (-entry.score, self._plan_key(entry.plan))
            )
            beam = expanded[: self.beam_width]

        winner = beam[0]
        if winner.first_state is None:
            raise RuntimeError("oracle search did not produce a first-turn state")
        result = OracleSearchResult(
            plan=winner.plan,
            score=winner.score,
            state=winner.state,
            first_state=winner.first_state,
            evaluated=evaluated,
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


# Short name for callers that only need the simulator implementation.
OracleAgent = SimulatorOracleAgent
