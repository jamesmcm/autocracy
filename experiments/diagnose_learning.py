"""Diagnostics: is the learning agent model-limited or sampling-limited?

Three instruments, all sharing the shipped single-life configuration:

* ``--mode probe`` — normal chronos lives, but every turn each candidate
  batch is additionally branched through the real simulator for one true
  transition.  Records Spearman correlation between predicted and true
  per-candidate poll deltas, the predicted spread (flatness), where the
  chosen action ranked truthfully, and how many true winners were even
  available in the sampled pool.
* ``--mode truth`` — the same pools, cadence, and capital rules, but
  candidates are scored by their TRUE one-turn poll delta.  This is the
  perfect-world-model ceiling: its win rate bounds what any better
  forecaster (e.g. chronos-2 base) could reach under this strategy.

Reading the pair together answers the question directly:

* truth ceiling wins a lot AND probe rho ~ 0  ->  model-limited; a larger
  forecaster is worth testing;
* truth ceiling loses too                      ->  sampling/time-limited;
  more candidates or turns matter, not parameters.

The simulator branching here is measurement-only instrumentation; no
oracle information flows into the chronos agent's own decisions.

Usage::

    uv run --extra chronos python experiments/diagnose_learning.py \
        --mode probe --lives 4
    uv run --extra chronos python experiments/diagnose_learning.py \
        --mode truth --lives 8
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from autocracy import simulator
from autocracy.learning import TreatmentEffectMemory
from autocracy.timeseries import (
    ELECTORAL_SUPPORT_FEATURE,
    ForecastDecision,
    StateForecast,
    TimeSeriesPolicyAgent,
    _records,
)

OUT_DEFAULT = Path("reports/learning_diagnostics.json")


def rankdata(values: list[float]) -> list[float]:
    """Average ranks, ties shared (1-based)."""

    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start
        while (
            end + 1 < len(order)
            and values[order[end + 1]] == values[order[start]]
        ):
            end += 1
        average = (start + end) / 2 + 1
        for position in range(start, end + 1):
            ranks[order[position]] = average
        start = end + 1
    return ranks


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra, rb = rankdata(a), rankdata(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = math.sqrt(
        sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)
    )
    return num / den if den else float("nan")


def _true_one_turn_poll(agent: TimeSeriesPolicyAgent, state, actions):
    """Branch the real simulator once; return the resulting poll rate."""

    try:
        ordered = (
            simulator.apply_actions(state, actions, data=agent.data)
            if actions
            else state
        )
        advanced = simulator.process_end_of_turn(
            ordered, agent.graph, data=agent.data, config=agent.config
        )
    except ValueError:
        return None
    return float(advanced.poll_rate)


class ProbeAgent(TimeSeriesPolicyAgent):
    """Chronos life with per-turn predicted-vs-true candidate telemetry."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.turn_stats: list[dict[str, float]] = []

    def choose_actions(self, state, options):
        chosen = super().choose_actions(state, options)
        base = float(state.poll_rate)
        candidates = list(getattr(self, "_last_candidates", []))
        forecasts = list(getattr(self, "_last_forecasts", []))
        predicted: list[float] = []
        actual: list[float] = []
        for actions, forecast in zip(candidates, forecasts):
            truth = _true_one_turn_poll(self, state, actions)
            if truth is None:
                continue
            predicted.append(
                float(forecast.first[ELECTORAL_SUPPORT_FEATURE]) - base
            )
            actual.append(truth - base)
        chosen_true = _true_one_turn_poll(self, state, chosen)
        chosen_delta = None if chosen_true is None else chosen_true - base
        winners_available = sum(1 for value in actual if value >= 0.03)
        # Counterfactual quality: the no-op candidate is always present, so
        # |predicted - true| for that single item measures how well this
        # forecaster tracks the un-treated world — the baseline every
        # memory sample is de-trended against.
        noop_index = next(
            (i for i, actions in enumerate(candidates) if not actions),
            None,
        )
        noop_error = float("nan")
        if noop_index is not None and noop_index < len(actual):
            noop_error = abs(predicted[noop_index] - actual[noop_index])
        rho = spearman(predicted, actual) if len(actual) >= 3 else float("nan")
        spread = (
            max(predicted) - min(predicted) if predicted else float("nan")
        )
        # Truthful percentile of the executed batch among evaluated ones.
        rank_pct = float("nan")
        if chosen_delta is not None and actual:
            better = sum(1 for value in actual if value > chosen_delta)
            rank_pct = better / len(actual)
        self.turn_stats.append(
            {
                "turn": float(state.turn),
                "spearman": rho,
                "predicted_spread": spread,
                "true_spread": max(actual) - min(actual) if actual else float("nan"),
                "chosen_true_delta": chosen_delta if chosen_delta is not None else float("nan"),
                "chosen_rank_pct": rank_pct,
                "winners_in_pool": float(winners_available),
                "evaluated": float(len(actual)),
                "noop_error": noop_error,
            }
        )
        return chosen


class TruthRankedAgent(TimeSeriesPolicyAgent):
    """Perfect-world-model ceiling: score candidates by true one-turn poll.

    Pools, exploration randomness, action cadence, and capital rules match
    the chronos lives exactly; only the ranking signal is replaced.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("treatment_memory", None)
        super().__init__(*args, **kwargs)

    def choose_actions(self, state, options):
        self._noop_forecast_row = None
        candidates = self._candidate_batches(options)
        base = float(state.poll_rate)
        scored: list[tuple[float, tuple, tuple]] = []
        for actions in candidates:
            truth = _true_one_turn_poll(self, state, actions)
            if truth is None:
                continue
            scored.append((truth - base, self._action_key(actions), actions))
        if not scored:
            return ()
        scored.sort(key=lambda item: (-item[0], item[1]))
        best_delta, _, best_actions = scored[0]
        names = self.feature_encoder.feature_names
        row = dict(zip(names, self.context.states[-1].row(names)))
        forecast = StateForecast.from_rows(
            self.context.model_input(best_actions, horizon=self.forecast_horizon),
            [dict(row) for _ in range(self.forecast_horizon)],
            model_name="true-one-turn",
        )
        self.last_decision = ForecastDecision(
            turn=state.turn,
            actions=_records(best_actions),
            forecast=forecast,
            score=float(base + best_delta),
            candidate_count=len(candidates),
        )
        return best_actions


def run_life(mode: str, index: int, args):
    if mode == "probe":
        memory = TreatmentEffectMemory(
            decay=args.decay,
            exploration_bonus=args.exploration_bonus,
            reference_cost=args.reference_cost,
            family_shrinkage=args.family_shrinkage,
        )
        agent = build_shared(memory, args, seed_offset=index, cls=ProbeAgent)
    else:
        memory = None
        agent = build_shared(None, args, seed_offset=index, cls=TruthRankedAgent)

    polls: list[float] = []
    margins: list[float] = []
    outcome = "incomplete"
    previous_term = agent.state.election_current_term
    total_turns = agent.state.election_turns_until * args.terms
    for _ in range(total_turns):
        agent.step()
        state = simulator.resolve_election_if_ready(agent.state, data=agent.data)
        agent.state = state
        polls.append(float(state.poll_rate))
        if state.election_current_term > previous_term:
            previous_term = state.election_current_term
            margins.append(
                float(
                    state.election_player_votes
                    - state.election_opposition_votes
                )
            )
            if state.election_result == "loss":
                outcome = f"lost-election-{len(margins)}"
                break
            if len(margins) >= args.terms:
                outcome = f"won-{args.terms}-terms"
                break
    summary = {
        "life": index,
        "seed": args.seed + index,
        "outcome": outcome,
        "margins": margins,
        "mean_poll": sum(polls) / len(polls) if polls else 0.0,
        "final_poll": polls[-1] if polls else 0.0,
        "debt_crisis": "DebtCrisis" in agent.state.active_situations,
    }
    if mode == "probe":
        summary["turn_stats"] = agent.turn_stats
        valid_rho = [
            t["spearman"] for t in agent.turn_stats if math.isfinite(t["spearman"])
        ]
        summary["mean_spearman"] = (
            sum(valid_rho) / len(valid_rho) if valid_rho else float("nan")
        )
        spreads = [t["predicted_spread"] for t in agent.turn_stats]
        summary["mean_predicted_spread"] = (
            sum(spreads) / len(spreads) if spreads else float("nan")
        )
        ranks = [
            t["chosen_rank_pct"]
            for t in agent.turn_stats
            if math.isfinite(t["chosen_rank_pct"])
        ]
        summary["mean_chosen_rank_pct"] = (
            sum(ranks) / len(ranks) if ranks else float("nan")
        )
        winners = [t["winners_in_pool"] for t in agent.turn_stats]
        summary["mean_winners_in_pool"] = (
            sum(winners) / len(winners) if winners else float("nan")
        )
    return summary


def build_shared(memory, args, *, seed_offset: int, cls):
    config_seed = args.seed + seed_offset
    from autocracy.models import SimulationConfig
    from autocracy.chronos import Chronos2SmallForecaster

    global _FORECASTER
    try:
        forecaster = _FORECASTER
    except NameError:
        forecaster = None
    if forecaster is None:
        forecaster = Chronos2SmallForecaster(model_name=args.model)
        _FORECASTER = forecaster
    return cls(
        forecaster,
        config=SimulationConfig(random_seed=config_seed),
        forecast_horizon=args.horizon,
        candidate_limit=args.candidate_limit,
        random_seed=config_seed,
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
        score_horizon_mean=True,
        fiscal_prior_weight=args.fiscal_prior_weight,
    )


_FORECASTER = None


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("probe", "truth"), default="probe")
    parser.add_argument("--lives", type=int, default=4)
    parser.add_argument("--terms", type=int, default=3)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--model", default="autogluon/chronos-2-small")
    parser.add_argument("--candidate-limit", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260813)
    parser.add_argument("--reverse-window", type=int, default=8)
    parser.add_argument("--reverse-penalty", type=float, default=0.05)
    parser.add_argument("--max-actions-per-turn", type=int, default=2)
    parser.add_argument("--batch-candidate-limit", type=int, default=96)
    parser.add_argument("--debt-growth-penalty", type=float, default=0.25)
    parser.add_argument("--memory-effect-weight", type=float, default=2.5)
    parser.add_argument("--memory-drift-window", type=int, default=8)
    parser.add_argument("--memory-credit-lag", type=int, default=2)
    parser.add_argument("--exploration-bonus", type=float, default=0.05)
    parser.add_argument("--reference-cost", type=float, default=10.0)
    parser.add_argument("--decay", type=float, default=0.9)
    parser.add_argument("--family-shrinkage", type=float, default=0.8)
    parser.add_argument("--fiscal-prior-weight", type=float, default=0.0)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    lives = [run_life(args.mode, index, args) for index in range(args.lives)]
    won_first = sum(
        1 for life in lives if life["margins"][:1] and life["margins"][0] > 0
    )
    payload = {
        "format": "autocracy-learning-diagnostics-v1",
        "mode": args.mode,
        "model": args.model,
        "config": {k: v for k, v in vars(args).items() if k != "out"},
        "won_first_election": won_first,
        "wins_all_terms": sum(
            1 for life in lives if life["outcome"].startswith("won-")
        ),
        "lives": lives,
    }
    out = args.out or Path(f"reports/learning_diagnostics_{args.mode}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    for life in lives:
        extra = ""
        if args.mode == "probe":
            extra = (
                f" rho={life['mean_spearman']:+.3f}"
                f" pred_spread={life['mean_predicted_spread']:.5f}"
                f" chosen_top={1 - life['mean_chosen_rank_pct']:.2f}"
                f" pool_winners={life['mean_winners_in_pool']:.1f}"
            )
        print(
            f"{args.mode} life {life['life']} seed {life['seed']}: "
            f"{life['outcome']} margins={life['margins']} "
            f"final_poll={life['final_poll']:.3f}"
            f" crisis={life['debt_crisis']}{extra}"
        )
    print(
        f"[{args.mode}] first-election wins: {won_first}/{len(lives)}, "
        f"all-term survivals: {payload['wins_all_terms']}/{len(lives)}"
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
