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
    diverse_warmup_plan,
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


class _RecordingForecaster:
    name = "recording"

    def __init__(self, bonus_action=None):
        self.inputs = []
        self.bonus_action = bonus_action

    def predict(self, model_input):
        pending = tuple(
            (action.policy_name, action.action_type or "", round(action.delta, 6))
            for action in model_input.pending_actions
        )
        self.inputs.append(pending)
        row = dict(zip(model_input.feature_names, model_input.history[-1]))
        if self.bonus_action is not None and pending == (self.bonus_action,):
            row["politics/poll_rate"] += 1.0
        return StateForecast.from_rows(
            model_input,
            [row for _ in range(model_input.horizon)],
            model_name=self.name,
        )


def test_warmup_plan_executes_scheduled_moves_before_model_choice():
    state, _ = simulator.get_initial_state("uk")
    plan = diverse_warmup_plan(state, size=3)
    assert len(plan) == 3

    forecaster = _RecordingForecaster()
    agent = TimeSeriesPolicyAgent(
        forecaster,
        forecast_horizon=2,
        candidate_limit=3,
        random_seed=7,
        visible_features_only=True,
        warmup_plan=plan,
    )

    for expected in plan:
        agent.step()
        chosen = agent.decisions[-1].actions
        assert len(chosen) == 1
        assert chosen[0].policy_name == expected.policy_name
        assert chosen[0].delta == pytest.approx(expected.delta)
    assert not agent.in_warmup
    assert agent.warmup_turns_taken == 3
    # Every warm-up transition entered the context as a real observation.
    assert [tuple(a.policy_name for a in batch) for batch in agent.context.actions[:3]] == [
        (expected.policy_name,) for expected in plan
    ]
    # After the plan is exhausted the agent scores full candidate sets again.
    agent.step()
    assert forecaster.inputs[-1] == () or len(forecaster.inputs[-1]) == 1


def test_warmup_skips_entries_that_are_not_legal_options():
    state, _ = simulator.get_initial_state("uk")
    legal = diverse_warmup_plan(state, size=1)
    stale = PolicyAction("NoSuchPolicy", 0.05, "raise")
    forecaster = _RecordingForecaster()
    agent = TimeSeriesPolicyAgent(
        forecaster,
        forecast_horizon=2,
        candidate_limit=3,
        random_seed=7,
        visible_features_only=True,
        warmup_plan=[stale, *legal],
    )

    agent.step()

    chosen = agent.decisions[-1].actions
    assert chosen[0].policy_name == legal[0].policy_name
    assert agent.warmup_turns_taken == 1
    assert all(action.policy_name != "NoSuchPolicy" for action in agent.context.actions[-1])


def test_reverse_penalty_dampens_flip_flops():
    bonus = ("IncomeTax", "lower", -0.05)

    def poll_objective(features):
        return features["politics/poll_rate"]

    common = dict(
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        objective=poll_objective,
    )

    undamped = TimeSeriesPolicyAgent(_RecordingForecaster(bonus_action=bonus), **common)
    undamped.choose_actions(undamped.state, undamped.available_actions())
    assert any(
        action.policy_name == "IncomeTax" and action.action_type == "lower"
        for action in undamped.last_decision.actions
    )

    damped = TimeSeriesPolicyAgent(
        _RecordingForecaster(bonus_action=bonus),
        reverse_window=4,
        reverse_penalty=5.0,
        **common,
    )
    # Simulate a recent raise of the same policy in the context.
    before = damped.state
    after = replace(before, turn=before.turn + 1)
    damped.context = damped.context.append_transition(
        before, [PolicyAction("IncomeTax", 0.05, "raise")], after
    )
    damped.state = after
    damped.choose_actions(damped.state, damped.available_actions())

    assert not any(
        action.policy_name == "IncomeTax" and action.action_type == "lower"
        for action in damped.last_decision.actions
    )


def test_seeded_context_grows_with_live_turns_and_skips_stale_rings():
    state, _ = simulator.get_initial_state("uk")
    encoder = StateFeatureEncoder.from_visible_state(state)
    seeded = AutoregressiveContext.from_state(
        state, encoder=encoder, include_value_histories=True
    )
    ring_rows = len(seeded.states)

    advanced = simulator.process_end_of_turn(state, simulator.build_country_graph("uk"))
    live_context = AutoregressiveContext.from_state(
        advanced, encoder=encoder, include_value_histories=True
    )

    # The pre-game rings end at their capture turn; once real turns advance
    # they are stale and must not duplicate live observations.
    assert advanced.turn > advanced.value_histories_turn
    assert len(live_context.states) == 1 < ring_rows


def _options_with_costs(costs):
    from autocracy.models import PolicyActionOption

    return [
        PolicyActionOption(f"Policy{i}", "raise", 0.1, 0.1, cost, 0.0)
        for i, cost in enumerate(costs)
    ]


def test_multi_action_batches_share_one_capital_budget():
    state, _ = simulator.get_initial_state("uk")
    agent = TimeSeriesPolicyAgent(
        _RecordingForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        visible_features_only=True,
        max_actions_per_turn=2,
    )
    # Two individually affordable moves whose sum exceeds capital.
    agent.state = replace(agent.state, political_capital=3.0)
    batches = agent._candidate_batches(_options_with_costs([2.0, 2.0, 0.5]))

    pairs = [batch for batch in batches if len(batch) == 2]
    assert ("Policy0", "Policy1") not in {
        tuple(action.policy_name for action in batch) for batch in pairs
    }
    assert any(
        {action.policy_name for action in batch} == {"Policy0", "Policy2"}
        for batch in pairs
    )
    for batch in batches:
        total = sum(
            option.cost
            for option in _options_with_costs([2.0, 2.0, 0.5])
            if option.policy_name in {action.policy_name for action in batch}
        )
        assert total <= 3.0 + 1e-9


def test_multi_action_pairs_are_enumerated_when_affordable():
    state, _ = simulator.get_initial_state("uk")
    agent = TimeSeriesPolicyAgent(
        _RecordingForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        visible_features_only=True,
        max_actions_per_turn=2,
    )
    batches = agent._candidate_batches(_options_with_costs([1.0, 1.0]))
    keys = [tuple(action.policy_name for action in batch) for batch in batches]

    assert () in keys
    assert ("Policy0",) in keys and ("Policy1",) in keys
    assert ("Policy0", "Policy1") in keys


def test_debt_growth_penalty_reorders_candidates():
    class DebtForecaster:
        name = "debt-fixture"
        POLL_BONUS = {"Spendy": 0.10, "Safe": 0.05}

        def __init__(self):
            self.scenarios = {}

        def predict(self, model_input):
            key = (
                model_input.pending_actions[0].policy_name
                if model_input.pending_actions
                else ""
            )
            base = dict(zip(model_input.feature_names, model_input.history[-1]))
            debt_start = base.get("finance/debt", 100.0)
            gdp_start = base.get("value/GDP", 1.0)
            rows = []
            for step in range(model_input.horizon):
                row = dict(base)
                debt_factor, gdp_factor = self.scenarios.get(key, ((1.0, 1.0),))[0]
                row["finance/debt"] = debt_start * debt_factor
                row["value/GDP"] = gdp_start * gdp_factor
                row["politics/poll_rate"] += self.POLL_BONUS.get(key, 0.0)
                rows.append(row)
            return StateForecast.from_rows(
                model_input, rows, model_name=self.name
            )

    def poll_objective(features):
        return features["politics/poll_rate"]

    state, _ = simulator.get_initial_state("uk")
    fixture = DebtForecaster()
    # "Spendy" grows debt 25% by horizon with flat GDP; "Safe" stays flat.
    fixture.scenarios["Spendy"] = [(1.25, 1.0)]
    fixture.scenarios["Safe"] = [(1.0, 1.0)]

    def agent_with(penalty):
        merged = DebtForecaster()
        merged.scenarios = fixture.scenarios
        return TimeSeriesPolicyAgent(
            merged,
            forecast_horizon=2,
            candidate_limit=None,
            random_seed=7,
            visible_features_only=True,
            objective=poll_objective,
            debt_growth_penalty=penalty,
        )

    options = _options_with_costs([1.0, 1.0])
    # Rename policies to match scenario keys.
    options = [
        replace(option, policy_name=name)
        for name, option in zip(("Spendy", "Safe"), options)
    ]

    neutral = agent_with(0.0)
    chosen_neutral = neutral.choose_actions(neutral.state, list(options))
    damped = agent_with(10.0)
    chosen_damped = damped.choose_actions(damped.state, list(options))

    assert chosen_neutral[0].policy_name == "Spendy"
    assert chosen_damped[0].policy_name == "Safe"


def test_max_action_delta_restricts_candidate_pool():
    state, _ = simulator.get_initial_state("uk")
    agent = TimeSeriesPolicyAgent(
        _RecordingForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        visible_features_only=True,
        max_actions_per_turn=2,
        max_action_delta=0.1,
    )
    options = _options_with_costs([1.0, 1.0])
    big = replace(options[0], policy_name="BigStep", delta=0.5)
    small = replace(options[1], policy_name="SmallStep", delta=0.05)

    batches = agent._candidate_batches([big, small])

    names = {action.policy_name for batch in batches for action in batch}
    assert "SmallStep" in names
    assert "BigStep" not in names
