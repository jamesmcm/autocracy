"""Closed-loop control comparison over one full UK term.

Runs the first election term (16 turns) from the mission start with several
agents that only see what a player would see:

* ``no-op``        - never spends political capital;
* ``persistence``  - forecast = last observed row;
* ``empirical``    - recent action-conditioned delta baseline;
* ``chronos-2-small`` - multivariate Chronos-2 forecasting with policy
  sliders as known-future treatment covariates.  A scripted diverse
  warm-up program runs first so the context accumulates real
  treatment/response pairs; afterwards reverse-action damping suppresses
  flip-flops;
* ``oracle-beam``  - ElectionOracleAgent with its documented winning defaults
  (:data:`autocracy.oracle.PROVEN_ELECTION_SEARCH`); branches the real
  simulator and wins from turn zero (best case, not player-visible).

Every agent is evaluated on the true simulator trajectory: expected
election margin at the boundary, mean poll rate, and the weighted headline
composite.

Usage::

    uv run --extra chronos python experiments/control_comparison.py \
        --turns 16 --out reports/control_comparison_h5.json
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import sys
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocracy import simulator
from autocracy.agent import ElectionOracleAgent, PassiveAgent
from autocracy.chronos import Chronos2SmallForecaster
from autocracy.models import PolicyAction, SimulationConfig, SimulationState
from autocracy.oracle import (
    PROVEN_ELECTION_SEARCH,
    OracleElectionLoss,
    score_simulation_state,
)
from autocracy.timeseries import (
    ELECTORAL_SUPPORT_FEATURE,
    EmpiricalActionForecaster,
    PersistenceForecaster,
    TimeSeriesPolicyAgent,
    diverse_warmup_plan,
)

TERM_LENGTH = 16
TRACE_DIR = Path("reports/traces")


def asdict_action(action: PolicyAction) -> dict[str, object]:
    return {
        "policy_name": action.policy_name,
        "delta": action.delta,
        "action_type": action.action_type,
    }


@dataclass(slots=True)
class RunResult:
    agent: str
    turns: int
    wall_clock_seconds: float
    election_result: str | None
    player_votes: int = 0
    opposition_votes: int = 0
    absent_votes: int = 0
    margin: float = 0.0
    player_share: float = 0.0
    mean_poll_rate: float = 0.0
    final_poll_rate: float = 0.0
    mean_composite: float = 0.0
    final_composite: float = 0.0
    actions_taken: int = 0
    notes: list[str] = field(default_factory=list)


def _poll_objective(features) -> float:
    return float(features[ELECTORAL_SUPPORT_FEATURE])


def _advance(agent, polls: list[float], composites: list[float]) -> bool:
    """Advance one turn, resolve a pending election, record true metrics.

    Returns False when an election loss ends the campaign.
    """

    try:
        agent.step()
    except OracleElectionLoss as loss:
        # Keep the resolved losing state so the summary reports real votes.
        agent.state = loss.state
        _record(loss.state, polls, composites)
        return False
    state = simulator.resolve_election_if_ready(agent.state, data=agent.data)
    agent.state = state
    _record(state, polls, composites)
    return state.election_result != "loss"


def _record(state: SimulationState, polls: list[float], composites: list[float]) -> None:
    polls.append(float(state.poll_rate))
    composites.append(score_simulation_state(state))


def _finalise(
    name: str,
    state: SimulationState,
    elapsed: float,
    polls: list[float],
    composites: list[float],
    actions: int,
    notes: list[str],
) -> RunResult:
    votes = state.election_player_votes
    opposition = state.election_opposition_votes
    if state.election_result in {"win", "loss"}:
        margin = float(votes - opposition)
    else:
        # A truncated run never reached the boundary; keep the expected
        # margin so partial runs remain comparable.
        margin = simulator.forecast_election(state).margin
    share = (
        votes / (votes + opposition) if (votes + opposition) else 0.0
    )
    return RunResult(
        agent=name,
        turns=len(polls),
        wall_clock_seconds=elapsed,
        election_result=state.election_result,
        player_votes=int(votes),
        opposition_votes=int(opposition),
        absent_votes=int(state.election_absent_votes),
        margin=float(margin),
        player_share=float(share),
        mean_poll_rate=sum(polls) / len(polls),
        final_poll_rate=polls[-1] if polls else 0.0,
        mean_composite=sum(composites) / len(composites),
        final_composite=composites[-1] if composites else 0.0,
        actions_taken=actions,
        notes=notes,
    )


def _run_agent(name: str, factory: Callable[[], object], max_turns: int) -> RunResult:
    agent = factory()
    polls: list[float] = []
    composites: list[float] = []
    actions = 0
    started = time.monotonic()
    notes: list[str] = []
    for turn_index in range(max_turns):
        before_turn = agent.state.turn
        alive = _advance(agent, polls, composites)
        decisions = getattr(agent, "decisions", None)
        if agent.state.turn > before_turn and decisions:
            actions += len(decisions[-1].actions)
        if not alive:
            notes.append(f"campaign ended with election loss at turn {agent.state.turn}")
            break
        if agent.state.election_result == "win":
            notes.append("first election resolved as win")
            break
    result = _finalise(
        name, agent.state, time.monotonic() - started, polls, composites, actions, notes
    )
    trace_saver = getattr(agent, "save_trace", None)
    if trace_saver is not None:
        trace_path = TRACE_DIR / f"{name}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_saver(trace_path)
    return result


def diverse_warmup_with_contrast(initial_state, size: int) -> tuple[PolicyAction, ...]:
    plan = list(
        diverse_warmup_plan(initial_state, size=size, capital_share=0.5)
    )
    introductions = [
        option
        for option in simulator.list_available_actions(initial_state)
        if option.action_type == "introduce"
    ]
    if introductions:
        loudest = max(introductions, key=lambda option: option.cost)
        plan.append(
            PolicyAction(
                policy_name=loudest.policy_name,
                delta=loudest.delta,
                action_type=loudest.action_type,
            )
        )
    return tuple(plan)


def fiscal_warmup_plan(initial_state, size: int = 8) -> tuple[PolicyAction, ...]:
    """Warm-up anchored on income-tax raises before a diverse tail.

    The tax anchors teach the forecaster the raise-taxes -> income-up ->
    polls-dip-slightly trade-off and immediately slow the structural deficit,
    which is what drives the mid-term DebtCrisis under passive play.  The
    diverse tail maximises treatment coverage; with ``warmup_batch_size=2``
    the agent executes two planned moves per turn, so eight planned moves
    cost only four turns of control.
    """

    tax_raises = sorted(
        (
            option
            for option in simulator.list_available_actions(initial_state)
            if option.policy_name == "IncomeTax" and option.action_type == "raise"
        ),
        key=lambda option: option.resulting_level,
    )
    plan = [
        PolicyAction(
            policy_name=option.policy_name,
            delta=option.delta,
            action_type=option.action_type,
        )
        for option in tax_raises[:2]
    ]
    plan.extend(
        diverse_warmup_plan(initial_state, size=max(size - len(plan), 0), capital_share=0.8)
    )
    return tuple(plan)


def build_agents(config: SimulationConfig, args, warmup: tuple[PolicyAction, ...]) -> list[tuple[str, Callable[[], object]]]:
    agents: list[tuple[str, Callable[[], object]]] = [
        ("no-op", lambda: PassiveAgent(config=config)),
        (
            "persistence",
            lambda: TimeSeriesPolicyAgent(
                PersistenceForecaster(),
                config=config,
                forecast_horizon=args.horizon,
                candidate_limit=args.candidate_limit,
                random_seed=args.seed,
                objective=_poll_objective,
                visible_features_only=True,
            ),
        ),
        (
            "empirical",
            lambda: TimeSeriesPolicyAgent(
                EmpiricalActionForecaster(),
                config=config,
                forecast_horizon=args.horizon,
                candidate_limit=args.candidate_limit,
                random_seed=args.seed,
                objective=_poll_objective,
                visible_features_only=True,
            ),
        ),
        (
            "chronos-2-small",
            # Warm-up first (scripted diverse moves feed the context real
            # treatment/response pairs), then model-driven choice with
            # reverse-action damping against flip-flops, the same two-actions
            # per turn as the oracle, and a predicted debt-to-GDP growth
            # penalty acting on the forecast fiscal path.
            lambda: TimeSeriesPolicyAgent(
                Chronos2SmallForecaster(),
                config=config,
                forecast_horizon=args.horizon,
                candidate_limit=args.candidate_limit,
                random_seed=args.seed,
                objective=_poll_objective,
                visible_features_only=True,
                seed_pre_game_history=True,
                warmup_plan=warmup,
                reverse_window=args.reverse_window,
                reverse_penalty=args.reverse_penalty,
                warmup_batch_size=args.warmup_batch_size,
                max_actions_per_turn=args.max_actions_per_turn,
                batch_candidate_limit=args.batch_candidate_limit,
                debt_growth_penalty=args.debt_growth_penalty,
                max_action_delta=args.max_action_delta or None,
                score_horizon_mean=args.score_horizon_mean,
            ),
        ),
        (
            "oracle-beam",
            # ElectionOracleAgent's defaults ARE the documented winning
            # search (autocracy.oracle.PROVEN_ELECTION_SEARCH); do not
            # hand-tune a weaker variant here.
            lambda: ElectionOracleAgent(config=config, random_seed=args.seed),
        ),
    ]
    return agents


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turns", type=int, default=TERM_LENGTH)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--candidate-limit", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--warmup-size", type=int, default=8)
    parser.add_argument("--warmup-batch-size", type=int, default=2)
    parser.add_argument(
        "--warmup-preset",
        choices=("diverse-contrast", "fiscal"),
        default="fiscal",
        help="Warm-up program: cheapest diverse moves + loudest introduce, "
        "or income-tax anchors before a small diverse tail.",
    )
    parser.add_argument("--reverse-window", type=int, default=4)
    parser.add_argument("--reverse-penalty", type=float, default=0.01)
    parser.add_argument("--max-actions-per-turn", type=int, default=2)
    parser.add_argument("--batch-candidate-limit", type=int, default=48)
    parser.add_argument("--debt-growth-penalty", type=float, default=0.25)
    parser.add_argument(
        "--max-action-delta",
        type=float,
        default=0.0,
        help="Restrict candidates to slider steps of at most this size "
        "(0 disables the cap).",
    )
    parser.add_argument(
        "--score-horizon-mean",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Rank candidates by the objective averaged over the predicted "
        "path instead of the final step.",
    )
    parser.add_argument(
        "--only", nargs="*", default=None, help="Restrict to named agents."
    )
    parser.add_argument("--out", type=Path, default=Path("reports/control_comparison.json"))
    args = parser.parse_args()

    config = SimulationConfig(random_seed=args.seed)
    initial_state, _ = simulator.get_initial_state("uk")
    if args.warmup_preset == "fiscal":
        warmup = fiscal_warmup_plan(initial_state, size=args.warmup_size)
    else:
        warmup = diverse_warmup_with_contrast(initial_state, args.warmup_size)
    agents = build_agents(config, args, tuple(warmup))
    if args.only:
        wanted = set(args.only)
        agents = [entry for entry in agents if entry[0] in wanted]

    results: dict[str, object] = {
        "format": "autocracy-control-comparison-v2",
        "config": {
            "country": "uk",
            "max_turns": args.turns,
            "forecast_horizon": args.horizon,
            "candidate_limit": args.candidate_limit,
            "random_seed": args.seed,
            "events_enabled": False,
            "warmup_plan": [asdict_action(action) for action in warmup],
            "warmup_preset": args.warmup_preset,
            "reverse_window": args.reverse_window,
            "reverse_penalty": args.reverse_penalty,
            "max_actions_per_turn": args.max_actions_per_turn,
            "batch_candidate_limit": args.batch_candidate_limit,
            "debt_growth_penalty": args.debt_growth_penalty,
            "oracle_search": PROVEN_ELECTION_SEARCH,
        },
        "runs": [],
    }
    for name, factory in agents:
        print(f"=== running {name} ===", flush=True)
        run = _run_agent(name, factory, args.turns)
        print(
            f"    result={run.election_result} margin={run.margin:.0f} "
            f"mean_poll={run.mean_poll_rate:.4f} "
            f"mean_composite={run.mean_composite:.4f} "
            f"({run.wall_clock_seconds:.1f}s)",
            flush=True,
        )
        results["runs"].append(asdict(run))

    results["analysis"] = analyse(results["runs"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


def analyse(runs: list[dict]) -> dict[str, object]:
    by_name = {run["agent"]: run for run in runs}
    analysis: dict[str, object] = {}
    oracle = by_name.get("oracle-beam")
    if oracle:
        for name, run in by_name.items():
            if name == "oracle-beam":
                continue
            analysis[name] = {
                "margin_regret_vs_oracle": oracle["margin"] - run["margin"],
                "composite_regret_vs_oracle": (
                    oracle["mean_composite"] - run["mean_composite"]
                ),
            }
    return analysis


if __name__ == "__main__":
    main()
