"""Full-fidelity campaign tracing for long-horizon model comparisons.

Drives either the Chronos learning agent or the simulator oracle through a
fixed number of elections and, every turn, writes:

* the **full start-of-turn ``SimulationState``** (``state_to_dict``),
  so any recorded trajectory can be replayed deterministically later;
* the **actions** the agent executed that turn;
* **starting and ending political capital, total income, total
  expenditure, and poll rate** for the turn;
* election boundary totals when a term resolves.

Each life lands in ``reports/campaigns/<mode>/<seed>/`` as
``turns.jsonl.gz`` plus a ``summary.json``.  A replay pass re-runs every
life from its turn-0 state through the recorded action sequence and asserts
the end-of-turn metrics match, proving the stored data is sufficient to
reconstruct the campaign.

Usage::

    # Chronos-2 (full/base) — 4 seeds, 20 elections each, for a country
    uv run --extra chronos python experiments/campaign_trace.py \
        --mode chronos --model autogluon/chronos-2 --country germany \
        --seeds 4 --elections 20

    # Simulator oracle — the perfect-case reference, one life per country
    uv run --extra chronos python experiments/campaign_trace.py \
        --mode oracle --country germany --elections 20
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocracy import simulator
from autocracy.models import PolicyAction, SimulationConfig
from autocracy.oracle import OracleElectionLoss
from autocracy.timeseries import TimeSeriesPolicyAgent

OUT_ROOT = Path("reports/campaigns")


def _args_for_chronos(model: str, seed: int) -> argparse.Namespace:
    args = argparse.Namespace()
    args.model = model
    args.seed = seed
    args.horizon = 5
    args.candidate_limit = 32
    args.reverse_window = 8
    args.reverse_penalty = 0.05
    args.max_actions_per_turn = 2
    args.batch_candidate_limit = 96
    args.debt_growth_penalty = 0.25
    args.memory_effect_weight = 4.0
    args.memory_drift_window = 8
    args.memory_credit_lag = 2
    args.exploration_bonus = 0.05
    args.reference_cost = 10.0
    args.decay = 0.9
    args.family_shrinkage = 0.8
    args.level_keys = False
    args.score_horizon_mean = True
    args.fiscal_prudence_weight = 1.0
    args.fiscal_prior_weight = 0.0
    args.balance_guard_penalty = 0.0
    args.warmup_size = 8
    args.warmup_batch_size = 2
    return args


def _agent_actions(agent) -> tuple[PolicyAction, ...]:
    """Recover the batch the agent committed this turn, both agent kinds."""

    decision = getattr(agent, "last_decision", None)
    if decision is not None and getattr(decision, "actions", ()):
        return tuple(decision.actions)
    search = getattr(agent, "last_search", None)
    if search is not None and getattr(search, "first_actions", ()):
        return tuple(search.first_actions)
    return ()


def _record_turn(
    record: dict,
    start_dict: dict,
    agent,
    actions: tuple[PolicyAction, ...],
) -> None:
    record["actions"] = [
        {
            "policy_name": action.policy_name,
            "delta": action.delta,
            "action_type": action.action_type,
        }
        for action in actions
    ]
    record["capital_start"] = float(start_dict["political_capital"])
    record["capital_end"] = float(agent.state.political_capital)
    record["income_start"] = float(start_dict["total_income"])
    record["income_end"] = float(agent.state.total_income)
    record["expenditure_start"] = float(start_dict["total_expenditure"])
    record["expenditure_end"] = float(agent.state.total_expenditure)
    record["poll_start"] = float(start_dict["poll_rate"])
    record["poll_end"] = float(agent.state.poll_rate)


_FORECASTER_CACHE: dict[str, object] = {}


def _forecaster(model: str):
    if model not in _FORECASTER_CACHE:
        from autocracy.chronos import Chronos2SmallForecaster

        _FORECASTER_CACHE[model] = Chronos2SmallForecaster(model_name=model)
    return _FORECASTER_CACHE[model]


def run_chronos_life(
    country: str,
    model: str,
    seed: int,
    elections: int,
    out_dir: Path,
    args: argparse.Namespace | None = None,
):
    from autocracy.learning import TreatmentEffectMemory

    if args is None:
        args = _args_for_chronos(model, seed)
    else:
        merged = _args_for_chronos(model, seed)
        merged.decay = args.decay
        merged.level_keys = args.level_keys
        args = merged

    memory = TreatmentEffectMemory(
        decay=args.decay,
        exploration_bonus=args.exploration_bonus,
        reference_cost=args.reference_cost,
        family_shrinkage=args.family_shrinkage,
        level_keys=args.level_keys,
    )

    def build(state=None):
        from autocracy.timeseries import ActionRecord, diverse_warmup_plan

        agent = TimeSeriesPolicyAgent(
            _forecaster(model),
            country=country,
            config=SimulationConfig(random_seed=seed),
            forecast_horizon=args.horizon,
            candidate_limit=args.candidate_limit,
            random_seed=seed,
            objective=lambda features: features["politics/poll_rate"],
            visible_features_only=True,
            seed_pre_game_history=True,
            warmup_batch_size=args.warmup_batch_size,
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
        if args.warmup_size:
            agent.warmup_plan = [
                ActionRecord.from_action(action)
                for action in diverse_warmup_plan(
                    agent.state, size=args.warmup_size, capital_share=0.5
                )
            ]
        return agent

    return _drive(seed, elections, build, out_dir, kind="chronos", data_for=agent_data)


def agent_data(agent):
    return getattr(agent, "data", None)


def _drive(seed, elections, build, out_dir, kind, data_for):
    agent = build()
    data = data_for(agent) or simulator.load_simulation_data()
    turns_path = out_dir / "turns.jsonl.gz"
    with gzip.GzipFile(str(turns_path), "wb", compresslevel=1) as handle:
        summary = {"seed": seed, "kind": kind, "margins": [], "term_mean_polls": [], "term_crisis": []}
        polls = []
        term_polls = []
        previous_term = agent.state.election_current_term
        total_turns = agent.state.election_turns_until * elections
        started = time.monotonic()
        for _ in range(total_turns):
            # Deep copy: state_to_dict aliases mutable containers (notably
            # ``policies``), and list_available_actions mutates them in
            # place during the decision, so a shallow capture would record
            # corrupted states.
            start_dict = simulator.state_to_dict(
                simulator.state_from_dict(simulator.state_to_dict(agent.state))
            )
            record = {"turn": agent.state.turn}
            try:
                agent.step()
            except OracleElectionLoss as loss:  # oracle terminal loss
                agent.state = loss.state
            actions = _agent_actions(agent)
            state = simulator.resolve_election_if_ready(agent.state, data=data)
            agent.state = state
            _record_turn(record, start_dict, agent, actions)
            polls.append(record["poll_end"])
            term_polls.append(record["poll_end"])
            if state.election_current_term > previous_term:
                previous_term = state.election_current_term
                margin = float(
                    state.election_player_votes
                    - state.election_opposition_votes
                )
                summary["margins"].append(margin)
                summary["term_mean_polls"].append(
                    sum(term_polls) / len(term_polls) if term_polls else 0.0
                )
                term_polls = []
                summary["term_crisis"].append(
                    "DebtCrisis" in state.active_situations
                )
                record["election_result"] = state.election_result
                record["player_votes"] = int(state.election_player_votes)
                record["opposition_votes"] = int(state.election_opposition_votes)
                if state.election_result == "loss" and kind == "oracle":
                    break
            record["state"] = start_dict
            handle.write(
                (json.dumps(record) + "\n").encode("utf-8")
            )
        summary["turns_played"] = len(polls)
        summary["mean_poll"] = sum(polls) / len(polls) if polls else 0.0
        summary["wall_clock_seconds"] = time.monotonic() - started
        (out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
    return summary


def run_oracle_life(country: str, seed: int, elections: int, out_dir: Path):
    from autocracy.agent import ElectionOracleAgent

    # PROVEN_ELECTION_SEARCH's candidate sampling needs the documented
    # random_seed; without it the search falls back to deterministic
    # alphabetical truncation and loses election 1.
    agent = ElectionOracleAgent(
        country=country,
        config=SimulationConfig(random_seed=seed),
        random_seed=seed,
    )

    def build():
        return agent

    return _drive(seed, elections, build, out_dir, kind="oracle", data_for=lambda a: a.data)


def run_noop_life(country: str, seed: int, elections: int, out_dir: Path):
    """Drive a country with the passive no-action baseline."""

    from autocracy.agent import PassiveAgent

    agent = PassiveAgent(
        country=country,
        config=SimulationConfig(random_seed=seed),
    )

    def build():
        return agent

    return _drive(seed, elections, build, out_dir, kind="noop", data_for=lambda a: a.data)


def replay_verify(seed, out_dir, elections, kind) -> tuple[bool, int]:
    """Re-run the recorded actions from the turn-0 state; compare metrics.

    Proves the stored full states + actions fully determine the campaign.
    The decision-time ``list_available_actions`` side effect (default-level
    injection for uncancellable policies) is replicated because it is part
    of the real trajectory.
    """

    with gzip.open(str(out_dir / "turns.jsonl.gz"), "rt") as handle:
        records = [json.loads(line) for line in handle]
    if not records:
        return False, 0
    data = simulator.load_simulation_data()
    state = simulator.state_from_dict(records[0]["state"])
    graph = simulator.build_country_graph(state.country)
    checked = 0
    for record in records:
        simulator.list_available_actions(state, data=data)
        actions = [
            PolicyAction(
                policy_name=item["policy_name"],
                delta=item["delta"],
                action_type=item["action_type"],
            )
            for item in record["actions"]
        ]
        if actions:
            state = simulator.apply_actions(state, actions, data=data)
        state = simulator.process_end_of_turn(
            state, graph, data=data, config=None
        )
        state = simulator.resolve_election_if_ready(state, data=data)
        for field in ("capital_end", "income_end", "expenditure_end", "poll_end"):
            if abs(float(getattr(state, _STATE_FIELD[field])) - record[field]) > 1e-3:
                return False, checked
        checked += 1
    return True, checked


def repair_life(out_dir: Path) -> tuple[bool, int]:
    """Rewrite recorded states with true pre-turn snapshots.

    The original capture aliased ``state.policies`` and the decision-time
    ``list_available_actions`` mutation corrupted every recorded snapshot.
    Actions and end-of-turn metrics are unaffected, so a faithful replay of
    the recorded actions (replicating that mutation) reconstructs the true
    pre-turn states without re-running the forecaster.
    """

    path = out_dir / "turns.jsonl.gz"
    with gzip.open(str(path), "rt") as handle:
        records = [json.loads(line) for line in handle]
    if not records:
        return False, 0
    data = simulator.load_simulation_data()
    state = simulator.state_from_dict(records[0]["state"])
    graph = simulator.build_country_graph(state.country)
    for index, record in enumerate(records):
        start_dict = simulator.state_to_dict(
            simulator.state_from_dict(simulator.state_to_dict(state))
        )
        simulator.list_available_actions(state, data=data)
        actions = [
            PolicyAction(
                policy_name=item["policy_name"],
                delta=item["delta"],
                action_type=item["action_type"],
            )
            for item in record["actions"]
        ]
        if actions:
            state = simulator.apply_actions(state, actions, data=data)
        state = simulator.process_end_of_turn(
            state, graph, data=data, config=None
        )
        state = simulator.resolve_election_if_ready(state, data=data)
        for field in ("capital_end", "income_end", "expenditure_end", "poll_end"):
            if (
                abs(float(getattr(state, _STATE_FIELD[field])) - record[field])
                > 1e-3
            ):
                return False, index
        record["state"] = start_dict
    with gzip.open(str(path), "wb", compresslevel=1) as handle:
        for record in records:
            handle.write((json.dumps(record) + "\n").encode("utf-8"))
    return True, len(records)


_STATE_FIELD = {
    "capital_end": "political_capital",
    "income_end": "total_income",
    "expenditure_end": "total_expenditure",
    "poll_end": "poll_rate",
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("chronos", "oracle", "noop"),
        required=True,
        help="chronos=forecaster, oracle=simulator search, noop=no actions.",
    )
    parser.add_argument(
        "--model",
        default="autogluon/chronos-2",
        help="Chronos checkpoint (default the 120M base).",
    )
    parser.add_argument("--country", default="uk")
    parser.add_argument("--seeds", type=int, default=None)
    parser.add_argument("--elections", type=int, default=20)
    parser.add_argument("--seed-base", type=int, default=20260813)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument(
        "--level-keys",
        action="store_true",
        help="Credit memory samples to the resulting slider level instead "
        "of the action gesture, so cancels reverse the sampled build-up.",
    )
    parser.add_argument("--skip-replay", action="store_true")
    parser.add_argument(
        "--repair",
        action="store_true",
        help="Do not run campaigns; rewrite every recorded turns.jsonl.gz "
        "under reports/campaigns/<mode>/<country>/ with true pre-turn states "
        "via faithful replay of the recorded actions.",
    )
    args = parser.parse_args()

    if args.repair:
        root = OUT_ROOT / args.mode / args.country
        if not root.exists():
            raise SystemExit(f"no {args.mode} campaign data to repair")
        for life_dir in sorted(root.iterdir(), key=lambda p: int(p.name)):
            ok, n = repair_life(life_dir)
            print(f"repaired {life_dir}: {'OK' if ok else 'MISMATCH'} ({n} turns)")
        return
    if args.seeds is None:
        args.seeds = 1 if args.mode in ("oracle", "noop") else 10

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []
    for index in range(args.seeds):
        seed = args.seed_base + index
        out_dir = OUT_ROOT / args.mode / args.country / str(seed)
        out_dir.mkdir(parents=True, exist_ok=True)
        if args.mode == "chronos":
            summary = run_chronos_life(
                args.country, args.model, seed, args.elections, out_dir, args=args
            )
        elif args.mode == "oracle":
            summary = run_oracle_life(args.country, seed, args.elections, out_dir)
        else:
            summary = run_noop_life(args.country, seed, args.elections, out_dir)
        ok = None
        if not args.skip_replay:
            ok, checked = replay_verify(seed, out_dir, args.elections, args.mode)
        line = (
            f"{args.mode} {args.country} seed {seed}: {len(summary['margins'])} elections, "
            f"margin {summary['margins'] if summary['margins'] else '[]'}, "
            f"mean_poll {summary['mean_poll']:.3f} "
            f"({summary['wall_clock_seconds']:.0f}s)"
        )
        if ok is not None:
            line += f" replay={'OK' if ok else 'MISMATCH'} ({checked} turns)"
        print(line, flush=True)
        summaries.append(summary)
    print(f"wrote {OUT_ROOT}")


if __name__ == "__main__":
    main()