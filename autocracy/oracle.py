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

# A full election-term search is intentionally allowed to spend more time on
# each real turn than the short VPS smoke runs.  Callers can still pass a
# smaller budget (or ``None`` for an unbounded search) for interactive work.
DEFAULT_ELECTION_TIME_BUDGET_SECONDS = 900.0

# The corrected UK experiment documented in SIMULATION.md won its first
# election from turn 0 with exactly this search configuration (seed
# 20260813): beam 6, five-turn lookahead, two policy moves per turn over 16
# sampled candidates with up to 64 legal pairs, under a 15-second decision
# budget.  :class:`~autocracy.agent.ElectionOracleAgent` uses these values as
# its defaults so experiments cannot silently degrade to a weaker search;
# pass overrides deliberately when ablating.
PROVEN_ELECTION_SEARCH: Mapping[str, object] = {
    "beam_width": 6,
    "search_horizon": 5,
    "candidate_limit": 16,
    "batch_candidate_limit": 64,
    "max_actions_per_turn": 2,
    "time_budget_seconds": 15.0,
}


class OracleElectionLoss(RuntimeError):
    """Raised when no searched branch survives an election.

    An election loss is terminal for a playable campaign.  Keeping the
    resolved state on the exception lets experiment runners report the vote
    totals without treating a continuation after the loss as a valid game.
    """

    def __init__(self, state: SimulationState) -> None:
        self.state = state
        super().__init__(
            "oracle branch lost the election: "
            f"{state.election_player_votes} player votes, "
            f"{state.election_opposition_votes} opposition votes, "
            f"{state.election_absent_votes} abstentions"
        )

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
    # Native XML does not serialize the result-screen-only vote totals.  The
    # native oracle fills this with the resolved Python runtime state so a
    # committed boundary remains observable to its caller.
    first_runtime_state: SimulationState | None = None
    # Search-budget telemetry.  A time-limited oracle may return the best
    # branch from a partially completed depth; the first move is still a real
    # simulator/native transition and is safe to commit.
    elapsed_seconds: float = 0.0
    completed_depth: int = 0
    timed_out: bool = False

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


def score_election_state(state: SimulationState) -> float:
    """Score a simulator state by expected first-election vote margin.

    A completed win receives a large bonus so a search that reaches the
    election boundary prefers survival over a slightly better economic
    intermediate state.  Before the boundary, the score is the native-style
    expected player-minus-opposition margin and therefore gives the beam a
    useful gradient instead of rewarding a generic GDP/poll composite.
    """

    from .simulator import forecast_election

    if state.election_result in {"win", "loss"}:
        margin = float(
            state.election_player_votes - state.election_opposition_votes
        )
        return margin + (1_000_000.0 if state.election_result == "win" else -1_000_000.0)
    return forecast_election(state).margin


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


def score_savegame_election(save: SaveGame) -> float:
    """Score a native XML state by its expected first-election margin."""

    from .simulator import forecast_election_from_voters

    return forecast_election_from_voters(save.voters, save.parties).margin


def validate_score(objective: Callable[[StateT], float], state: StateT) -> float:
    """Evaluate an objective and reject NaN/inf values early."""

    score = float(objective(state))
    if not math.isfinite(score):
        raise ValueError(f"oracle objective returned a non-finite score: {score!r}")
    return score
