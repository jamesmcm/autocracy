"""Tests for the status-quo (no-op) evidence gate on TimeSeriesPolicyAgent."""

from __future__ import annotations

import dataclasses

import pytest

from autocracy import simulator
from autocracy.models import PolicyAction
from autocracy.timeseries import (
    ELECTORAL_SUPPORT_FEATURE,
    ForecastModelInput,
    StateForecast,
    TimeSeriesPolicyAgent,
)


class _ScriptedForecaster:
    """Deterministic forecaster with per-policy poll deltas (final step).

    The forecast is otherwise persistence; the band is a flat half-width
    ``band`` on the headline poll feature so uncertainty-aware margins can
    be exercised without Chronos.
    """

    name = "scripted"

    def __init__(
        self,
        deltas: dict[str, float],
        *,
        band: float = 0.0,
        with_bands: bool = False,
    ) -> None:
        self.deltas = deltas
        self.band = band
        self.with_bands = with_bands

    def predict(self, model_input: ForecastModelInput) -> StateForecast:
        row = dict(zip(model_input.feature_names, model_input.history[-1]))
        poll = float(row[ELECTORAL_SUPPORT_FEATURE])
        delta = sum(
            self.deltas.get(action.policy_name, 0.0)
            for action in model_input.pending_actions
        )
        rows = []
        for step in range(model_input.horizon):
            future = dict(row)
            future[ELECTORAL_SUPPORT_FEATURE] = poll + delta * (step + 1) / model_input.horizon
            rows.append(future)
        lower = upper = None
        if self.with_bands:
            lower = tuple(
                {**r, ELECTORAL_SUPPORT_FEATURE: r[ELECTORAL_SUPPORT_FEATURE] - self.band}
                for r in rows
            )
            upper = tuple(
                {**r, ELECTORAL_SUPPORT_FEATURE: r[ELECTORAL_SUPPORT_FEATURE] + self.band}
                for r in rows
            )
        return StateForecast.from_rows(
            model_input, rows, model_name=self.name, lower=lower, upper=upper
        )


def _poll_objective(features):
    return features[ELECTORAL_SUPPORT_FEATURE]


def _agent(deltas, **overrides):
    kwargs = dict(
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        objective=_poll_objective,
        forecaster=_ScriptedForecaster(deltas, **{
            k: overrides.pop(k) for k in list(overrides) if k in {"band", "with_bands"}
        }),
    )
    return TimeSeriesPolicyAgent(**overrides, **kwargs)


def test_without_gate_best_action_wins_by_score():
    state, _ = simulator.get_initial_state("uk")
    agent = _agent({"AlcoholTax": 0.05}, state=state)
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    assert any(action.policy_name == "AlcoholTax" for action in chosen)


def test_gate_blocks_action_without_credible_evidence():
    state, _ = simulator.get_initial_state("uk")
    # +0.001 forecast poll gain must not clear delta=0.01.
    agent = _agent({"AlcoholTax": 0.001}, state=state, intervention_threshold=0.01)
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    assert chosen == ()


def test_gate_passes_action_with_strong_evidence():
    state, _ = simulator.get_initial_state("uk")
    agent = _agent({"AlcoholTax": 0.10}, state=state, intervention_threshold=0.01)
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    assert any(action.policy_name == "AlcoholTax" for action in chosen)


def test_uncertainty_lambda_raises_bar_for_noisy_forecasts():
    state, _ = simulator.get_initial_state("uk")
    # Same forecast delta; wide band + lambda must block the trade the
    # narrow-band agent still takes.
    deltas = {"AlcoholTax": 0.05}
    plain = _agent(deltas, state=state, intervention_threshold=0.01,
                   intervention_lambda=1.0, band=0.01, with_bands=True)
    assert any(
        action.policy_name == "AlcoholTax"
        for action in plain.choose_actions(plain.state, plain.available_actions())
    )
    noisy = _agent(deltas, state=state, intervention_threshold=0.01,
                   intervention_lambda=10.0, band=0.01, with_bands=True)
    assert noisy.choose_actions(noisy.state, noisy.available_actions()) == ()


def test_gate_falls_back_to_noop_forecast_for_decision():
    state, _ = simulator.get_initial_state("uk")
    agent = _agent({"AlcoholTax": 0.0}, state=state, intervention_threshold=0.01)
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    assert chosen == ()
    decision = agent.last_decision
    assert decision is not None
    assert decision.actions == ()
    assert decision.evidence_margin is not None
    # No-op evidence equals its own score: margin is exactly zero.
    assert decision.evidence_margin == pytest.approx(0.0)
    # Counterfactual row recorded for the treatment-memory de-trending.
    assert agent._noop_forecast_row is not None


def test_cooldown_blocks_recent_policy_even_with_evidence():
    state, _ = simulator.get_initial_state("uk")
    agent = _agent(
        {"AlcoholTax": 0.20},
        state=state,
        intervention_threshold=0.01,
        reversal_cooldown=5,
    )
    before = agent.state
    after = dataclasses.replace(before, turn=before.turn + 1)
    agent.context = agent.context.append_transition(
        before, [PolicyAction("IncomeTax", 0.05, "raise")], after
    )
    agent.state = after
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    # IncomeTax is in its chill window; BonusPolicy has real evidence so it
    # may still go through, but no IncomeTax action may appear.
    assert all(action.policy_name != "IncomeTax" for action in chosen)


def test_action_cost_weight_penalises_expensive_candidates():
    state, _ = simulator.get_initial_state("uk")
    agent = _agent(
        {"AlcoholTax": 0.05},
        state=state,
        intervention_threshold=0.01,
        action_cost_weight=1.0,
    )
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    # With a capital cost charged against evidence, the +0.05 poll gain no
    # longer justifies the spend.
    assert chosen == ()


def test_gate_warmup_turns_bypasses_gate():
    state, _ = simulator.get_initial_state("uk")
    agent = _agent(
        {"AlcoholTax": 0.001},
        state=state,
        intervention_threshold=0.01,
        gate_warmup_turns=10**6,
    )
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    # Gate disabled: plain argmax wins even though evidence is tiny.
    assert any(action.policy_name == "AlcoholTax" for action in chosen)


def test_exploration_bonus_cannot_cross_gate():
    state, _ = simulator.get_initial_state("uk")
    agent = _agent({"AlcoholTax": 0.001}, state=state, intervention_threshold=0.01)
    memory = agent.treatment_memory
    assert memory is None  # sanity: no memory by default
    # Give the agent a memory with a large exploration bonus; the gate must
    # still refuse the near-zero-evidence move because bonuses are excluded
    # from evidence.
    from autocracy.learning import TreatmentEffectMemory

    memory = TreatmentEffectMemory(exploration_bonus=10.0)
    agent.treatment_memory = memory
    chosen = agent.choose_actions(agent.state, agent.available_actions())
    assert chosen == ()
