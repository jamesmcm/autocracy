"""Shared result types and objectives for best-case policy search.

The oracle agents deliberately score observed states rather than estimating a
policy's effect from its metadata.  The simulator-backed agent observes a
state produced by :func:`autocracy.simulator.process_end_of_turn`; the native
GameDrive agent observes a parsed XML save written by Democracy 3 itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Generic, Mapping, TypeVar

from .models import PolicyAction, SimulationState
from .savegame import SaveGame


StateT = TypeVar("StateT")

# These are intentionally normalized, headline metrics.  Callers should pass
# an objective when the task has a different definition of "best" (for
# example, maximize GDP while keeping expenditure below a limit).
DEFAULT_ORACLE_WEIGHTS: Mapping[str, float] = {
    "GDP": 1.0,
    "Health": 1.0,
    "Education": 1.0,
    "CrimeRate": -1.0,
    "Unemployment": -1.0,
}
DEFAULT_POLL_WEIGHT = 0.5


@dataclass(frozen=True, slots=True)
class OracleSearchResult(Generic[StateT]):
    """The winning beam path and the first move to execute.

    ``state`` is the state at the end of the search horizon, while
    ``first_state`` is the result after only the first turn.  Native searches
    also set ``first_artifact``/``artifacts`` to save names that can be used to
    continue the real-game branch without running the first turn twice.
    """

    plan: tuple[tuple[PolicyAction, ...], ...]
    score: float
    state: StateT
    first_state: StateT
    evaluated: int
    first_artifact: str | None = None
    artifacts: tuple[str, ...] = ()

    @property
    def first_actions(self) -> tuple[PolicyAction, ...]:
        """Actions to execute now; later plan entries are forecasts."""

        return self.plan[0] if self.plan else ()


def _weighted_score(
    values: Mapping[str, float],
    *,
    poll_rate: float = 0.0,
    weights: Mapping[str, float] = DEFAULT_ORACLE_WEIGHTS,
    poll_weight: float = DEFAULT_POLL_WEIGHT,
) -> float:
    score = sum(weight * float(values.get(name, 0.0)) for name, weight in weights.items())
    return score + poll_weight * float(poll_rate)


def score_simulation_state(
    state: SimulationState,
    *,
    weights: Mapping[str, float] = DEFAULT_ORACLE_WEIGHTS,
    poll_weight: float = DEFAULT_POLL_WEIGHT,
) -> float:
    """Score a simulator state with the default or caller-provided objective."""

    return _weighted_score(
        state.values,
        poll_rate=state.poll_rate,
        weights=weights,
        poll_weight=poll_weight,
    )


def score_savegame(
    save: SaveGame,
    *,
    weights: Mapping[str, float] = DEFAULT_ORACLE_WEIGHTS,
    poll_weight: float = DEFAULT_POLL_WEIGHT,
) -> float:
    """Score a native XML snapshot using the same objective as the simulator."""

    return _weighted_score(
        save.simvalues,
        poll_rate=save.poll_rate or 0.0,
        weights=weights,
        poll_weight=poll_weight,
    )


def validate_score(objective: Callable[[StateT], float], state: StateT) -> float:
    """Evaluate an objective and reject NaN/inf values early."""

    score = float(objective(state))
    if not math.isfinite(score):
        raise ValueError(f"oracle objective returned a non-finite score: {score!r}")
    return score
