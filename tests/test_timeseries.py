from __future__ import annotations

import json
from dataclasses import replace

import pytest

from autocracy import simulator
from autocracy.models import PolicyAction
from autocracy.timeseries import (
    AutoregressiveContext,
    Chronos2Forecaster,
    EmpiricalActionForecaster,
    StateFeatureEncoder,
    StateForecast,
    TimeSeriesPolicyAgent,
)


def _small_context():
    state, _ = simulator.get_initial_state("uk")
    encoder = StateFeatureEncoder.from_state(
        state,
        value_names=["GDP"],
        policy_names=[],
    )
    return state, encoder, AutoregressiveContext.from_state(state, encoder=encoder)


def test_context_appends_real_state_and_round_trips(tmp_path):
    state, encoder, context = _small_context()
    after = replace(
        state,
        turn=state.turn + 1,
        values={**state.values, "GDP": state.values["GDP"] + 2.0},
    )
    action = PolicyAction("IncomeTax", 0.05, "raise")

    updated = context.append_transition(state, [action], after)
    model_input = updated.model_input([action], horizon=3)

    assert updated.current_turn == state.turn + 1
    assert updated.actions[0][0].policy_name == "IncomeTax"
    assert model_input.history[-1][0] == pytest.approx(state.values["GDP"] + 2.0)
    assert model_input.pending_actions[0].action_type == "raise"

    path = tmp_path / "context.json"
    updated.save(path)
    loaded = AutoregressiveContext.load(path)
    assert loaded.to_dict() == updated.to_dict()


def test_empirical_forecaster_conditions_on_observed_action():
    state, encoder, context = _small_context()
    action = PolicyAction("IncomeTax", 0.05, "raise")
    first_after = replace(
        state,
        turn=state.turn + 1,
        values={**state.values, "GDP": state.values["GDP"] + 2.0},
    )
    context = context.append_transition(state, [action], first_after)
    model_input = context.model_input([action], horizon=2)

    forecast = EmpiricalActionForecaster().predict(model_input)

    assert forecast.model_name == "empirical-action-delta"
    assert forecast.first["value/GDP"] == pytest.approx(first_after.values["GDP"] + 2.0)
    assert forecast.final["value/GDP"] == pytest.approx(first_after.values["GDP"] + 4.0)


def test_chronos2_adapter_keeps_backend_optional():
    state, _, context = _small_context()
    model_input = context.model_input(horizon=1)

    with pytest.raises(RuntimeError, match="has no backend"):
        Chronos2Forecaster().predict(model_input)

    seen = {}

    def backend(received):
        seen["input"] = received
        row = dict(zip(received.feature_names, received.history[-1]))
        row["value/GDP"] += 3.0
        return [row]

    forecast = Chronos2Forecaster.from_callable(backend).predict(model_input)

    assert seen["input"].horizon == 1
    assert forecast.model_name == "chronos2"
    assert forecast.first["value/GDP"] == pytest.approx(state.values["GDP"] + 3.0)


def test_timeseries_agent_appends_observed_state_to_context(tmp_path):
    class ActionRewardForecaster:
        name = "test-action-reward"

        def predict(self, model_input):
            row = dict(zip(model_input.feature_names, model_input.history[-1]))
            if model_input.pending_actions:
                row["value/GDP"] += 1.0
            return StateForecast.from_rows(
                model_input,
                [row for _ in range(model_input.horizon)],
                model_name=self.name,
            )

    agent = TimeSeriesPolicyAgent(
        ActionRewardForecaster(),
        forecast_horizon=2,
        candidate_limit=3,
        random_seed=7,
        objective=lambda features: features["value/GDP"],
    )

    before_turn = agent.state.turn
    agent.step()

    assert agent.state.turn == before_turn + 1
    assert len(agent.context.states) == 2
    assert len(agent.context.actions) == 1
    assert agent.decisions[0].observed == agent.context.states[-1]
    assert agent.decisions[0].candidate_count == 3

    path = tmp_path / "trace.json"
    agent.save_trace(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["format"] == "autocracy-timeseries-trace-v1"
    assert len(payload["context"]["states"]) == 2
    assert payload["decisions"][0]["observed"]["turn"] == before_turn + 1
