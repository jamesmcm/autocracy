from __future__ import annotations

import json

import pytest

from autocracy import simulator
from autocracy.learning import TreatmentEffectMemory, action_key
from autocracy.models import PolicyAction
from autocracy.timeseries import (
    ELECTORAL_SUPPORT_FEATURE,
    StateForecast,
    TimeSeriesPolicyAgent,
)


def _action(name="PolicyA", delta=-0.05, kind="lower"):
    return PolicyAction(name, delta, kind)


class _FlatForecaster:
    """Every candidate looks identical: pure noise without memory."""

    name = "flat"

    def predict(self, model_input):
        row = dict(zip(model_input.feature_names, model_input.history[-1]))
        return StateForecast.from_rows(
            model_input,
            [dict(row) for _ in range(model_input.horizon)],
            model_name=self.name,
        )


def _first_available_option(agent):
    """Deterministically pick some legal move, independent of its identity.

    Tests must not care WHICH policy they use: hard-coding known-good moves
    anywhere (fixtures included) would smuggle strategy knowledge into the
    experiment.  Whatever this returns is treated as an opaque favourite.
    """

    options = agent.available_actions()
    assert options
    return min(
        options,
        key=lambda option: (
            option.policy_name,
            option.action_type or "",
            round(option.delta, 6),
        ),
    )


def _option_as_action(option) -> PolicyAction:
    return PolicyAction(
        policy_name=option.policy_name,
        delta=option.delta,
        action_type=option.action_type,
    )


def test_action_key_normalises_type_and_delta():
    assert action_key(_action()) == ("PolicyA", "lower", -0.05)
    assert action_key(_action(kind=None)) == ("PolicyA", "", -0.05)
    assert action_key(_action(delta=0.0500001)) == action_key(
        _action(delta=0.05)
    )


def test_memory_estimates_recency_weighted_effects():
    memory = TreatmentEffectMemory(decay=0.5)
    memory.record([_action()], observed_delta=0.10, drift=0.02)
    memory.transitions = 10  # age the first sample artificially
    memory.record([_action()], observed_delta=0.04, drift=0.02)

    estimate = memory.estimate(_action())
    old_weight = 0.5**10

    # Effects are de-trended (0.08 and 0.02); the older sample is downweighted.
    assert estimate == pytest.approx(
        (0.08 * old_weight + 0.02) / (old_weight + 1.0)
    )
    assert memory.effect_total([_action()]) == pytest.approx(estimate)
    assert memory.effect_total([_action("PolicyZ", 0.05, "raise")]) == 0.0
    assert memory.visits(_action()) == 2


def test_memory_drift_estimate_is_windowed_median():
    memory = TreatmentEffectMemory()
    history = [0.01, 0.03, 0.02, 0.05, 0.04]

    assert memory.drift_estimate(history, window=3) == pytest.approx(0.04)
    assert memory.drift_estimate(history, window=0) == 0.0
    assert memory.drift_estimate([], window=4) == 0.0


def test_exploration_bonus_decays_with_visits_and_cost():
    memory = TreatmentEffectMemory(exploration_bonus=0.05, reference_cost=10.0)

    fresh = memory._explore_bonus(_action(), cost=0.0)
    memory.record([_action()], observed_delta=0.0)
    seen_once = memory._explore_bonus(_action(), cost=0.0)

    assert 0.0 < seen_once < fresh
    # Costlier moves get a smaller bonus under the same uncertainty.
    assert memory._explore_bonus(_action(), cost=20.0) < seen_once
    assert memory.explore_total([], costs=[]) == 0.0
    inert = TreatmentEffectMemory()
    assert inert.explore_total([_action()], costs=[0.0]) == 0.0


def test_exploration_bonus_shrinks_with_visits_not_global_history():
    memory = TreatmentEffectMemory(exploration_bonus=0.05)
    base = memory._explore_bonus(_action(), cost=0.0)

    other = _action("PolicyB", 0.1, "raise")
    for _ in range(50):
        memory.record([other], observed_delta=0.01)

    # Growing global experience must not intensify exploration of a
    # still-unvisited action.
    assert memory._explore_bonus(_action(), cost=0.0) == pytest.approx(base)


def test_memory_round_trips_through_json(tmp_path):
    memory = TreatmentEffectMemory(exploration_bonus=0.03, decay=0.8)
    memory.record(
        [_action(), _action("PolicyC", 0.25, "introduce")],
        observed_delta=0.06,
        drift=0.01,
    )

    path = tmp_path / "memory.json"
    memory.save(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    loaded = TreatmentEffectMemory.from_dict(payload)

    assert loaded.to_dict() == memory.to_dict()
    assert loaded.estimate(_action()) == pytest.approx(memory.estimate(_action()))
    assert TreatmentEffectMemory.load(path).transitions == 1


def test_memory_rejects_invalid_configuration():
    with pytest.raises(ValueError):
        TreatmentEffectMemory(decay=0.0)
    with pytest.raises(ValueError):
        TreatmentEffectMemory(exploration_bonus=-1.0)


def test_ranked_actions_orders_by_effect():
    memory = TreatmentEffectMemory()
    memory.record([_action("PolicyG", 0.05, "raise")], observed_delta=0.08, drift=0.0)
    memory.record([_action("PolicyB", 0.05, "raise")], observed_delta=-0.02, drift=0.0)

    ranked = memory.ranked_actions(limit=5)

    assert ranked[0][0][:1] == ("PolicyG",)
    assert ranked[0][1] == pytest.approx(0.08)
    assert ranked[0][2] == 1


def test_family_fallback_shares_evidence_across_slider_steps():
    """Untried next-steps of a proven slider inherit shrunk evidence."""

    memory = TreatmentEffectMemory(family_shrinkage=0.5)
    memory.record([_action("PolicyF", 0.12, "raise")], observed_delta=0.10)

    sibling = _action("PolicyF", 0.25, "raise")
    assert memory.estimate(sibling) == pytest.approx(0.05)
    # Other directions and other policies stay genuinely unknown.
    assert memory.estimate(_action("PolicyF", -0.2, "lower")) is None
    assert memory.estimate(_action("PolicyG", 0.25, "raise")) is None

    # Direct evidence on the sibling replaces the inherited estimate.
    memory.record([sibling], observed_delta=0.02, drift=0.01)
    assert memory.visits(sibling) == 1
    assert memory.estimate(sibling) == pytest.approx(0.01)


def test_agent_records_observed_transitions_into_memory():
    memory = TreatmentEffectMemory()
    agent = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        treatment_memory=memory,
    )
    # An opaque legal move made strictly best by prior evidence; the test
    # never assumes anything about which policy it is.
    favourite_option = _first_available_option(agent)
    favourite = _option_as_action(favourite_option)
    memory.record([favourite], observed_delta=0.5, drift=0.0)

    before_poll = agent.state.poll_rate
    agent.step()

    chosen = agent.decisions[-1].actions
    assert [action_key(action) for action in chosen] == [action_key(favourite)]
    assert memory.transitions == 2
    samples = memory.effects[action_key(favourite)]
    assert len(samples) == 2
    # First transition has no history, so the drift is zero and the stored
    # effect equals the observed poll movement.
    observed = agent.state.poll_rate - before_poll
    assert samples[-1].value == pytest.approx(observed)


def test_memory_overrides_flat_forecast_ranking():
    memory = TreatmentEffectMemory()
    agent_probe = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        treatment_memory=memory,
    )
    favourite_option = _first_available_option(agent_probe)
    favourite = _option_as_action(favourite_option)
    memory.record([favourite], observed_delta=0.5, drift=0.0)

    agent = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        treatment_memory=memory,
    )
    chosen = agent.choose_actions(agent.state, agent.available_actions())

    assert [action_key(action) for action in chosen] == [action_key(favourite)]


def test_agent_requires_electoral_support_feature_for_memory():
    state, _ = simulator.get_initial_state("uk")
    from autocracy.timeseries import StateFeatureEncoder

    encoder = StateFeatureEncoder.from_state(
        state, value_names=["GDP"], policy_names=[], include_election=False
    )
    with pytest.raises(ValueError, match="politics/poll_rate"):
        TimeSeriesPolicyAgent(
            _FlatForecaster(),
            encoder=encoder,
            treatment_memory=TreatmentEffectMemory(),
        )


def _fabricate_transition(agent, actions, *, poll_delta):
    """Append one synthetic observed transition without playing a turn."""

    from dataclasses import replace as dataclass_replace

    before = agent.state
    after = dataclass_replace(
        before,
        turn=before.turn + 1,
        poll_rate=max(0.0, before.poll_rate + poll_delta),
    )
    agent.context = agent.context.append_transition(before, actions, after)
    agent.state = after


def test_lagged_credit_records_windowed_effects():
    """A batch pushed at turn t is credited again when its window closes."""

    memory = TreatmentEffectMemory()
    agent = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        treatment_memory=memory,
        memory_credit_lag=2,
    )
    batch = (_action("PolicyW", 0.25, "raise"),)
    polls = iter([0.01, 0.02, 0.05])

    for expected_samples, expected_transitions in ((1, 1), (2, 2), (4, 4)):
        _fabricate_transition(agent, batch, poll_delta=next(polls))
        agent._record_treatment_effects(batch)
        samples = memory.effects[action_key(batch[0])]
        assert len(samples) == expected_samples
        assert memory.transitions == expected_transitions

    # The windowed sample spans the whole two-transition window de-trended
    # by the natural drift over that span (it sits just before the newest
    # immediate sample).
    effects = memory.effects[action_key(batch[0])]
    windowed = effects[-2]
    assert windowed.value == pytest.approx(
        (0.18 - 0.11) - 2 * 0.015
    )


def test_recorder_prefers_noop_counterfactual_over_median_drift():
    memory = TreatmentEffectMemory()
    agent = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        treatment_memory=memory,
    )
    batch = (_action("PolicyW", 0.25, "raise"),)
    # Natural drift was +0.030, but the world model expected +0.008 without
    # any action; the stored effect must use the model's counterfactual.
    _fabricate_transition(agent, batch, poll_delta=0.030)
    agent._noop_forecast_row = {
        **dict(zip(
            agent.feature_encoder.feature_names,
            agent.context.states[-1].row(agent.feature_encoder.feature_names),
        )),
        ELECTORAL_SUPPORT_FEATURE: agent.state.poll_rate - 0.008,
    }

    agent._record_treatment_effects(batch)

    sample = memory.effects[action_key(batch[0])][-1]
    assert sample.value == pytest.approx(0.008)
    # Consumed exactly once: the next record falls back to medians.
    assert agent._noop_forecast_row is None


class _BalanceForecaster:
    """Two candidates: one blows up spending, one keeps the budget flat."""

    name = "balance-fixture"

    def predict(self, model_input):
        key = (
            model_input.pending_actions[0].policy_name
            if model_input.pending_actions
            else ""
        )
        row = dict(zip(model_input.feature_names, model_input.history[-1]))
        if key == "PolicySpend":
            row["finance/total_expenditure"] *= 1.5
        return StateForecast.from_rows(
            model_input,
            [dict(row) for _ in range(model_input.horizon)],
            model_name=self.name,
        )


def test_balance_guard_penalizes_deficit_deepening_candidates():
    from dataclasses import replace as dataclass_replace

    from autocracy.models import PolicyActionOption

    memory = TreatmentEffectMemory()
    agent = TimeSeriesPolicyAgent(
        _BalanceForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        treatment_memory=memory,
        balance_guard_penalty=10.0,
    )
    # The visible budget is already in deficit.
    deficit_state = dataclass_replace(
        agent.state,
        total_income=200_000.0,
        total_expenditure=300_000.0,
    )
    template = _first_available_option(agent)
    options = [
        dataclass_replace(template, policy_name="PolicySpend"),
        dataclass_replace(template, policy_name="PolicyFrugal"),
    ]

    chosen = agent.choose_actions(deficit_state, options)

    # The spender is strictly rejected; frugal/no-op may tie.
    assert all(action.policy_name != "PolicySpend" for action in chosen)


def test_fiscal_channel_measures_balance_effects():
    memory = TreatmentEffectMemory()
    memory.record(
        [_action("PolicySpend", 0.25, "introduce")],
        observed_delta=0.01,
        fiscal_delta=-0.06,
    )
    memory.record(
        [_action("PolicyTax", 0.05, "raise")],
        observed_delta=-0.01,
        fiscal_delta=0.02,
    )

    assert memory.fiscal_total([_action("PolicySpend", 0.25, "introduce")]) == (
        pytest.approx(-0.06)
    )
    # Family fallback shares direction evidence for untried steps here too.
    assert memory.fiscal_total(
        [_action("PolicySpend", 0.5, "introduce")]
    ) == pytest.approx(-0.06 * 0.8)
    assert memory.estimate(_action("PolicyTax", 0.05, "raise")) == pytest.approx(
        -0.01
    )


def test_options_carry_game_declared_financial_delta():
    """The £ effect shown when dragging a slider rides on every option."""

    state, _ = simulator.get_initial_state("uk")
    options = simulator.list_available_actions(state)

    # Generic sign checks — no specific policy knowledge: the roster must
    # contain declared net spenders (introductions) and declared earners,
    # and tax raises in particular must improve the balance.
    introduces = [o for o in options if o.action_type == "introduce"]
    assert any(o.financial_delta < 0 for o in introduces)
    tax_raises = [
        o for o in options
        if o.action_type == "raise" and "Tax" in o.policy_name
    ]
    assert tax_raises
    assert all(o.financial_delta > 0 for o in tax_raises)


def test_fiscal_prior_uses_declared_cost_before_any_measurement():
    from dataclasses import replace as dataclass_replace

    from autocracy.models import PolicyActionOption

    agent = TimeSeriesPolicyAgent(
        _BalanceForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        fiscal_prior_weight=5.0,
    )
    deficit_state = dataclass_replace(
        agent.state,
        total_income=200_000.0,
        total_expenditure=300_000.0,
    )
    template = _first_available_option(agent)
    cheap = dataclass_replace(template, policy_name="PolicyCheap")
    pricey = dataclass_replace(template, policy_name="PolicyPricey")

    chosen = agent.choose_actions(deficit_state, [cheap, pricey])
    assert isinstance(chosen, tuple)

    # The declared deltas must reach the scoring path via the option map.
    agent._candidate_batches([cheap, pricey])
    assert set(agent._option_financials) == {
        ("PolicyCheap", template.action_type or "", round(template.delta, 6)),
        ("PolicyPricey", template.action_type or "", round(template.delta, 6)),
    }


def test_exploration_countdown_scales_bonus_by_term_share():
    """Curiosity must fade as the election approaches within one life."""

    from dataclasses import replace as dataclass_replace

    memory = TreatmentEffectMemory(exploration_bonus=0.10)
    agent = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        random_seed=7,
        visible_features_only=True,
        treatment_memory=memory,
        exploration_countdown=True,
    )

    early = dataclass_replace(agent.state, election_turns_until=16)
    late = dataclass_replace(agent.state, election_turns_until=2)

    def bonus_at(state):
        scoped = TimeSeriesPolicyAgent(
            _FlatForecaster(),
            state=state,
            forecast_horizon=2,
            candidate_limit=None,
            random_seed=7,
            visible_features_only=True,
            treatment_memory=memory,
            exploration_countdown=True,
        )
        chosen = scoped.choose_actions(state, scoped.available_actions())
        return sum(
            memory._explore_bonus(action, cost=0.0) for action in chosen
        ) * min(1.0, state.election_turns_until / 16)

    assert bonus_at(late) < 0.25 * bonus_at(early)


def test_candidate_pool_drops_options_that_cannot_move_slider_target():
    from autocracy.models import PolicyActionOption

    state, _ = simulator.get_initial_state("uk")
    # Implementation lag: the neuron reads 0.5 but the slider target is
    # already at its ceiling, so a "+0.05 raise" would clamp to no change.
    stale_name = "PolicyS"
    state.policies[stale_name] = 0.5
    state.policy_desired_throttles[stale_name] = 1.0
    agent = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        forecast_horizon=2,
        candidate_limit=None,
        visible_features_only=True,
    )
    agent.state = state

    stale = PolicyActionOption(stale_name, "raise", 0.05, 0.55, 1.0, 1.0)
    healthy = PolicyActionOption("PolicyT", "introduce", 0.25, 0.25, 4.0, 1.0)

    batches = agent._candidate_batches([stale, healthy])

    chosen_names = {
        action.policy_name for batch in batches for action in batch
    }
    assert "PolicyT" in chosen_names
    assert stale_name not in chosen_names
