"""Single-life active-learning campaign for the Chronos policy agent.

One run is one life: a fresh UK mission from ``uk0.xml`` (turn 0, first
election 16 turns out), a **fresh empty** treatment-effect memory, and no
information from any previous attempt.  The agent learns only from its own
observed transitions while playing:

* every executed turn appends a de-trended poll delta to the memory, so
  repeated actions converge on measured treatment effects within the life;
* an optimism bonus probes untried sliders early in the term and is scaled
  down by the election countdown (``exploration_countdown``), so the agent
  exploits what it measured before the vote;
* candidate pools are prioritised by learned effect + curiosity instead of
  uniform sampling, concentrating expensive forecasts on informative moves;
* winning the first election continues the same campaign into later terms,
  where online learning keeps running (survival).

Cross-episode memory reuse was deliberately rejected: replaying the same
save and carrying learned effects back into turn 0 would smuggle oracle
look-ahead into the start state.  ``--runs`` executes independent lives
(each with its own empty memory) purely to aggregate win statistics.

Usage::

    uv run --extra chronos python experiments/chronos_learning.py \
        --runs 5 --terms 3 --out reports/chronos_learning.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocracy import simulator
from autocracy.chronos import Chronos2SmallForecaster
from autocracy.learning import TreatmentEffectMemory
from autocracy.models import SimulationConfig
from autocracy.timeseries import (
    ELECTORAL_SUPPORT_FEATURE,
    TimeSeriesPolicyAgent,
)

TRACE_DIR = Path("reports/traces")

_SHARED_FORECASTER: Chronos2SmallForecaster | None = None


def _shared_forecaster() -> Chronos2SmallForecaster:
    """Load the pipeline once and reuse it for every independent life."""

    global _SHARED_FORECASTER
    if _SHARED_FORECASTER is None:
        _SHARED_FORECASTER = Chronos2SmallForecaster()
    return _SHARED_FORECASTER


@dataclass(slots=True)
class RunResult:
    run: int
    seed: int
    turns_played: int = 0
    elections_fought: int = 0
    margins: list[float] = field(default_factory=list)
    outcome: str = "incomplete"
    mean_poll: float = 0.0
    final_poll: float = 0.0
    actions_taken: int = 0
    interventions_tried: int = 0
    debt_crisis: bool = False
    wall_clock_seconds: float = 0.0


def build_agent(memory: TreatmentEffectMemory, args, *, seed_offset: int = 0) -> TimeSeriesPolicyAgent:
    """Fresh campaign from game start with an empty learned memory."""

    config = SimulationConfig(random_seed=args.seed + seed_offset)
    return TimeSeriesPolicyAgent(
        _shared_forecaster(),
        config=config,
        forecast_horizon=args.horizon,
        candidate_limit=args.candidate_limit,
        random_seed=args.seed + seed_offset,
        objective=lambda features: features[ELECTORAL_SUPPORT_FEATURE],
        visible_features_only=True,
        seed_pre_game_history=True,
        max_actions_per_turn=args.max_actions_per_turn,
        batch_candidate_limit=args.batch_candidate_limit,
        reverse_window=args.reverse_window,
        reverse_penalty=args.reverse_penalty,
        debt_growth_penalty=args.debt_growth_penalty,
        treatment_memory=memory,
        memory_effect_weight=args.memory_effect_weight,
        memory_drift_window=args.memory_drift_window,
        exploration_countdown=True,
        memory_credit_lag=args.memory_credit_lag,
        score_horizon_mean=args.score_horizon_mean,
        balance_guard_penalty=args.balance_guard_penalty,
        fiscal_prudence_weight=args.fiscal_prudence_weight,
        fiscal_prior_weight=args.fiscal_prior_weight,
    )


def run_single_life(run: int, args) -> tuple[RunResult, TimeSeriesPolicyAgent, TreatmentEffectMemory]:
    memory = TreatmentEffectMemory(
        decay=args.decay,
        exploration_bonus=args.exploration_bonus,
        reference_cost=args.reference_cost,
        family_shrinkage=args.family_shrinkage,
    )
    agent = build_agent(memory, args, seed_offset=run)
    result = RunResult(run=run, seed=args.seed + run)
    polls: list[float] = []
    previous_term = agent.state.election_current_term
    total_turns = agent.state.election_turns_until * args.terms
    started = time.monotonic()
    for _ in range(total_turns):
        agent.step()
        state = simulator.resolve_election_if_ready(agent.state, data=agent.data)
        agent.state = state
        polls.append(float(state.poll_rate))
        result.turns_played += 1
        if state.election_current_term > previous_term:
            previous_term = state.election_current_term
            result.elections_fought += 1
            margin = float(
                state.election_player_votes - state.election_opposition_votes
            )
            result.margins.append(margin)
            if state.election_result == "loss":
                result.outcome = f"lost-election-{result.elections_fought}"
                break
            if result.elections_fought >= args.terms:
                result.outcome = f"won-{args.terms}-terms"
                break
    result.mean_poll = sum(polls) / len(polls) if polls else 0.0
    result.final_poll = polls[-1] if polls else 0.0
    result.actions_taken = sum(len(d.actions) for d in agent.decisions)
    result.interventions_tried = memory.known_actions
    result.debt_crisis = "DebtCrisis" in agent.state.active_situations
    result.wall_clock_seconds = time.monotonic() - started
    return result, agent, memory


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--terms", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--reverse-window", type=int, default=8)
    parser.add_argument("--reverse-penalty", type=float, default=0.05)
    parser.add_argument("--max-actions-per-turn", type=int, default=2)
    parser.add_argument("--batch-candidate-limit", type=int, default=96)
    parser.add_argument("--debt-growth-penalty", type=float, default=0.25)
    parser.add_argument("--memory-effect-weight", type=float, default=4.0)
    parser.add_argument(
        "--score-horizon-mean",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Rank candidates by the objective averaged over the predicted "
        "path instead of the final step.",
    )
    parser.add_argument(
        "--balance-guard-penalty",
        type=float,
        default=0.0,
        help="While the visible budget runs a deficit, penalize candidates "
        "whose own forecast deepens it (relative share of the deficit).",
    )
    parser.add_argument(
        "--fiscal-prudence-weight",
        type=float,
        default=1.0,
        help="While in deficit, score candidates with their measured "
        "balance effect (expenditure-normalised share per turn).",
    )
    parser.add_argument(
        "--fiscal-prior-weight",
        type=float,
        default=0.0,
        help="While in deficit, score candidates with the game-declared £ "
        "effect of each move (option financial_delta, expenditure-"
        "normalised) — known before any measurement.",
    )
    parser.add_argument("--memory-drift-window", type=int, default=8)
    parser.add_argument(
        "--memory-credit-lag",
        type=int,
        default=2,
        help="Transitions over which an action's second, windowed poll "
        "effect is credited; lets slowly ramping introductions collect the "
        "credit their first transition hides.",
    )
    parser.add_argument("--exploration-bonus", type=float, default=0.05)
    parser.add_argument("--reference-cost", type=float, default=10.0)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument(
        "--family-shrinkage",
        type=float,
        default=0.8,
        help="How much untried slider steps inherit from measured siblings "
        "of the same policy and direction.",
    )
    parser.add_argument("--memory-out", type=Path, default=None)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=Path("reports/chronos_learning.json"))
    args = parser.parse_args()

    results: dict[str, object] = {
        "format": "autocracy-chronos-single-life-v1",
        "config": {
            "country": "uk",
            "runs": args.runs,
            "terms": args.terms,
            "forecast_horizon": args.horizon,
            "candidate_limit": args.candidate_limit,
            "random_seed": args.seed,
            "exploration_bonus": args.exploration_bonus,
            "reference_cost": args.reference_cost,
            "decay": args.decay,
            "family_shrinkage": args.family_shrinkage,
            "memory_effect_weight": args.memory_effect_weight,
            "memory_drift_window": args.memory_drift_window,
            "memory_credit_lag": args.memory_credit_lag,
            "exploration_countdown": True,
        },
        "runs": [],
    }

    wins_first_election = 0
    survived_all_terms = 0
    best_run: RunResult | None = None
    best_agent: TimeSeriesPolicyAgent | None = None
    best_memory: TreatmentEffectMemory | None = None
    for run in range(args.runs):
        print(f"=== life {run} (seed {args.seed + run}) ===", flush=True)
        result, agent, memory = run_single_life(run, args)
        print(
            f"    {result.outcome} margins={result.margins} "
            f"mean_poll={result.mean_poll:.4f} final_poll={result.final_poll:.4f} "
            f"actions={result.actions_taken} tried={result.interventions_tried} "
            f"({result.wall_clock_seconds:.1f}s)",
            flush=True,
        )
        results["runs"].append(asdict(result))  # type: ignore[arg-type]
        first_margin = result.margins[0] if result.margins else None
        if first_margin is not None and first_margin > 0:
            wins_first_election += 1
            if len(result.margins) >= args.terms:
                survived_all_terms += 1
            if (
                best_run is None
                or len(result.margins) > len(best_run.margins)
            ):
                best_run, best_agent, best_memory = result, agent, memory

    results["summary"] = {
        "lives": args.runs,
        "won_first_election": wins_first_election,
        "survived_all_terms": survived_all_terms,
        "win_rate": wins_first_election / max(args.runs, 1),
    }
    if best_agent is not None and best_memory is not None and best_run is not None:
        trace_path = args.trace_out or TRACE_DIR / "chronos-single-life-winner.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        best_agent.save_trace(trace_path)
        results["best_life"] = {  # type: ignore[assignment]
            "run": best_run.run,
            "margins": best_run.margins,
            "top_effects": [
                {"action": list(key), "effect": round(effect, 5), "visits": visits}
                for key, effect, visits in best_memory.ranked_actions(15)
            ],
        }
        if args.memory_out is not None:
            best_memory.save(args.memory_out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
