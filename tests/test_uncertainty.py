from __future__ import annotations

from argparse import Namespace
from dataclasses import replace

import pytest

from autocracy.models import PolicyAction
from autocracy.timeseries import (
    ELECTORAL_SUPPORT_FEATURE as POLL,
    ForecastModelInput,
    StateForecast,
    StateSnapshot,
    TimeSeriesPolicyAgent,
)
from autocracy.uncertainty import (
    UncertaintyConfig,
    context_windows,
    delta_moments,
    marginal_poll_shortfall,
)


class WindowForecaster:
    name = "window-test"

    def __init__(self) -> None:
        self.batches: list[list[ForecastModelInput]] = []

    def predict(self, item: ForecastModelInput) -> StateForecast:
        size = len(item.history)
        common_drift = {8: 0.0, 6: 0.1, 4: -0.1}[size]
        policy = item.pending_actions[0].policy_name if item.pending_actions else "noop"
        effect = {"noop": 0.0, "IncomeTax": 0.08,
                  "AlcoholTax": {8: 0.05, 6: -0.05, 4: 0.15}[size]}[policy]
        row = dict(zip(item.feature_names, item.history[-1]))
        row[POLL] = 0.5 + common_drift + effect
        low = {"noop": 0.45, "IncomeTax": 0.48, "AlcoholTax": 0.2}[policy]
        return StateForecast.from_rows(
            item, [row] * item.horizon,
            quantiles={
                0.05: [{POLL: low}] * item.horizon,
                0.1: [{POLL: 0.46}] * item.horizon,
            },
        )

    def predict_batch(self, inputs: list[ForecastModelInput]) -> list[StateForecast]:
        self.batches.append(list(inputs))
        return [self.predict(item) for item in inputs]


def make_agent(monkeypatch: pytest.MonkeyPatch, config: UncertaintyConfig) -> TimeSeriesPolicyAgent:
    agent = TimeSeriesPolicyAgent(
        WindowForecaster(), forecast_horizon=2, uncertainty=config,
        objective=lambda row: row[POLL], random_seed=7,
    )
    row = agent.context.states[-1].features
    agent.context = replace(
        agent.context,
        states=tuple(StateSnapshot(turn=turn, features=row) for turn in range(-7, 1)),
        actions=((),) * 7,
    )
    candidates = [(), (PolicyAction("IncomeTax", 0.05, "raise"),),
                  (PolicyAction("AlcoholTax", 0.05, "raise"),)]
    monkeypatch.setattr(agent, "_candidate_batches", lambda options: candidates)
    return agent


def test_beta_zero_preserves_greedy_choice_and_single_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(monkeypatch, UncertaintyConfig(beta=0, members=5))
    rng_state = agent._random.getstate()
    chosen = agent.choose_actions(agent.state, ())
    assert chosen[0].policy_name == "IncomeTax"
    assert agent.last_decision.score == pytest.approx(0.58)
    assert agent.last_decision.acquisition is None
    assert len(agent.forecaster.batches) == 1
    assert agent._random.getstate() == rng_state


def test_ucb_explores_paired_effect_disagreement(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(monkeypatch, UncertaintyConfig(beta=1))
    chosen = agent.choose_actions(agent.state, ())
    assert chosen[0].policy_name == "AlcoholTax"
    noop, safe, uncertain = agent._last_acquisitions
    assert noop["mean_delta"] == noop["epistemic_std"] == noop["score"] == 0
    assert safe["epistemic_std"] == pytest.approx(0, abs=1e-15)
    assert uncertain["mean_delta"] == pytest.approx(0.05)
    assert uncertain["epistemic_std"] == pytest.approx((0.02 / 3) ** 0.5)
    assert uncertain["context_rows"] == [8, 6, 4]
    assert len(agent.forecaster.batches) == 3
    assert all(len(batch) == 3 for batch in agent.forecaster.batches)
    assert agent.last_decision.to_dict()["acquisition"] == uncertain


@pytest.mark.parametrize("quantile,expected", [(0.05, "IncomeTax"), (0.1, "AlcoholTax")])
def test_downside_risk_can_outweigh_exploration(
    monkeypatch: pytest.MonkeyPatch, quantile: float, expected: str,
) -> None:
    agent = make_agent(monkeypatch, UncertaintyConfig(beta=1, risk_weight=1, risk_quantile=quantile))
    chosen = agent.choose_actions(agent.state, ())
    assert chosen[0].policy_name == expected
    assert agent._last_acquisitions[0]["risk_penalty"] == 0
    assert agent._last_acquisitions[2]["risk_penalty"] == pytest.approx(0.25 if quantile == 0.05 else 0)


def test_risk_only_uses_one_member(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(monkeypatch, UncertaintyConfig(risk_weight=1))
    agent.choose_actions(agent.state, ())
    assert len(agent.forecaster.batches) == 1
    assert all(stats["epistemic_std"] == 0 for stats in agent._last_acquisitions)


def test_context_windows_preserve_pairing_transitions_and_pending_actions(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(monkeypatch, UncertaintyConfig(beta=1))
    inputs = [agent.context.model_input(actions, horizon=2) for actions in agent._candidate_batches(())]
    windows = context_windows(inputs, agent.uncertainty)
    for window in windows:
        for original, perturbed in zip(inputs, window):
            assert perturbed.turns[-1] == original.turns[-1]
            assert perturbed.history[-1] == original.history[-1]
            assert perturbed.pending_actions == original.pending_actions
            assert len(perturbed.action_history) == len(perturbed.history) - 1
            assert perturbed.history == window[0].history
    short = [replace(item, history=item.history[-2:], turns=item.turns[-2:], action_history=((),)) for item in inputs]
    assert len(context_windows(short, agent.uncertainty)) == 1


def test_pairing_cancels_common_drift() -> None:
    assert delta_moments([100.1, -99.9, 0.1], [100, -100, 0]) == pytest.approx((0.1, 0))
    with pytest.raises(ValueError, match="matching"):
        delta_moments([1], [1, 2])
    with pytest.raises(ValueError, match="non-finite"):
        delta_moments([float("nan")], [0])


def test_tail_shortfall_is_per_step_and_requires_quantiles(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = make_agent(monkeypatch, UncertaintyConfig(risk_weight=1))
    item = agent.context.model_input((), horizon=2)
    forecast = StateForecast.from_rows(item, [{POLL: 0.5}] * 2,
                                       quantiles={0.05: [{POLL: 0.2}, {POLL: 0.6}]})
    assert marginal_poll_shortfall(forecast, agent.uncertainty) == pytest.approx(0.3)
    with pytest.raises(ValueError, match="requires q"):
        marginal_poll_shortfall(replace(forecast, quantiles={}), agent.uncertainty)
    with pytest.raises(ValueError, match="finite poll"):
        marginal_poll_shortfall(replace(forecast, quantiles={0.05: ({}, {})}), agent.uncertainty)


@pytest.mark.parametrize("kwargs", [
    {"beta": -1}, {"beta": float("nan")}, {"beta": 1, "members": 1},
    {"members": 0}, {"risk_weight": -1}, {"risk_quantile": 0.2},
    {"risk_floor": 2}, {"min_context_fraction": 1},
])
def test_invalid_uncertainty_config(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        UncertaintyConfig(**kwargs)


def test_conservative_gate_cannot_suppress_ucb() -> None:
    with pytest.raises(ValueError, match="non-conservative"):
        TimeSeriesPolicyAgent(WindowForecaster(), uncertainty=UncertaintyConfig(beta=1),
                              intervention_threshold=0.02)


def test_campaign_plumbing_records_effective_config(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    from experiments import campaign_trace

    captured = {}

    def fake_forecaster(model, **kwargs):
        captured["forecaster"] = (model, kwargs)
        return WindowForecaster()

    def fake_drive(seed, elections, build, out_dir, **kwargs):
        agent = build()
        captured["uncertainty"] = agent.uncertainty
        return {"config": kwargs["run_config"]}

    monkeypatch.setattr(campaign_trace, "_forecaster", fake_forecaster)
    monkeypatch.setattr(campaign_trace, "_drive", fake_drive)
    summary = campaign_trace.run_chronos_life(
        "germany", "autogluon/chronos-2", 17, 2, tmp_path,
        args=Namespace(uncertainty_beta=0.5, risk_weight=0.25, warmup_size=0,
                       intervention_threshold=None, intervention_lambda=None),
    )
    assert captured["uncertainty"] == UncertaintyConfig(beta=0.5, risk_weight=0.25)
    assert captured["forecaster"][1]["full_quantiles"] is True
    assert summary["config"]["country"] == "germany"
    assert summary["config"]["seed"] == 17
    assert summary["config"]["uncertainty"]["beta"] == 0.5
    assert summary["config"]["intervention_threshold"] == 0
