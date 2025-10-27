from __future__ import annotations

from dataclasses import replace

import pytest

from autocracy import simulator
from autocracy.agent import PassiveAgent
from autocracy.models import PolicyAction
from autocracy.savegame import parse_savegame


def test_state_serialization_round_trip(tmp_path):
    state, _ = simulator.get_initial_state("uk")
    payload = simulator.state_to_dict(state)
    restored = simulator.state_from_dict(payload)
    assert restored.country == state.country
    assert restored.policies == state.policies
    out_path = tmp_path / "state.json"
    simulator.save_state(state, out_path)
    reloaded = simulator.load_state(out_path)
    assert reloaded.values == state.values


def test_apply_actions_allows_small_adjustments_within_bounds():
    state, _ = simulator.get_initial_state("uk")
    updated = simulator.apply_actions(
        state,
        [PolicyAction(policy_name="BusLanes", delta=0.1)],
    )
    assert updated.policies["BusLanes"] > state.policies["BusLanes"]


def test_process_end_of_turn_advances_turn_and_updates_values():
    state, graph = simulator.get_initial_state("uk")
    next_state = simulator.process_end_of_turn(state, graph)
    assert next_state.turn == state.turn + 1
    assert next_state.values["GDP"] != pytest.approx(state.values["GDP"])


def test_passive_agent_step_runs_single_loop():
    agent = PassiveAgent(country="uk")
    initial_turn = agent.state.turn
    agent.step()
    assert agent.state.turn == initial_turn + 1
    # passive agent never spends capital
    assert agent.state.political_capital >= 0


def test_introduction_cost_used_for_new_policy():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    options = simulator.list_available_actions(state, data=data)
    micro = next(
        opt
        for opt in options
        if opt.policy_name == "MicrogenerationGrants" and opt.action_type == "introduce"
    )
    # introduction should cost the introduce_cost (10)
    assert micro.cost == data.policies["MicrogenerationGrants"].introduce_cost
    new_state = simulator.apply_actions(
        state, [PolicyAction(policy_name="MicrogenerationGrants", delta=micro.delta)], data=data
    )
    assert new_state.political_capital == pytest.approx(
        state.political_capital - micro.cost
    )


def test_cancel_action_present_for_active_policy():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    options = simulator.list_available_actions(state, data=data)
    cancels = [
        opt for opt in options if opt.policy_name == "FoodStandards" and opt.action_type == "cancel"
    ]
    assert cancels, "Expected cancel option for FoodStandards"
    cancel = cancels[0]
    assert cancel.resulting_level == pytest.approx(0.0)
    assert cancel.cost == data.policies["FoodStandards"].cancel_cost


def test_lower_and_cancel_are_distinct():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    options = simulator.list_available_actions(state, data=data)
    lowers = [
        opt for opt in options if opt.policy_name == "UnemployedBenefit" and opt.action_type == "lower"
    ]
    cancels = [
        opt for opt in options if opt.policy_name == "UnemployedBenefit" and opt.action_type == "cancel"
    ]
    assert lowers, "Expected lower option"
    assert cancels, "Expected cancel option"
    assert lowers[0].cost == data.policies["UnemployedBenefit"].lower_cost
    assert cancels[0].cost == data.policies["UnemployedBenefit"].cancel_cost
    assert lowers[0].resulting_level > 0.0


def test_uncancellable_policies_start_active():
    state, _ = simulator.get_initial_state("uk")
    assert state.policies["BorderControls"] > 0
    options = simulator.list_available_actions(state)
    introduces = [o for o in options if o.policy_name == "BorderControls" and o.action_type == "introduce"]
    cancels = [o for o in options if o.policy_name == "BorderControls" and o.action_type == "cancel"]
    assert not introduces
    assert not cancels


def test_uncancellable_cannot_be_cancelled():
    state, _ = simulator.get_initial_state("uk")
    with pytest.raises(ValueError):
        simulator.apply_actions(
            state,
            [PolicyAction(policy_name="BorderControls", delta=-state.policies["BorderControls"])]
        )


def test_initial_state_matches_initial_save():
    save = parse_savegame("gamedata/saves/uk0.xml")
    state, _ = simulator.get_initial_state("uk")
    for name, value in save.simvalues.items():
        assert state.values[name] == pytest.approx(value, rel=1e-3, abs=1e-3)
    for name, value in save.policies.items():
        assert state.policies[name] == pytest.approx(value, rel=1e-6, abs=1e-6)


def test_response_factors_seeded_from_turn_one_save():
    state, _ = simulator.get_initial_state("uk")
    assert "GDP" in state.response_factors


def test_street_gangs_activation_thresholds():
    state, graph = simulator.get_initial_state("uk")
    baseline = state.situations.get("StreetGangs")
    assert baseline is not None
    # Push all positive drivers to extremes so the latent value crosses the start trigger.
    state.values["PovertyRate"] = 1.0
    state.values["Unemployment"] = 1.0
    state.values["Homelessness"] = 1.0
    boosted = simulator.process_end_of_turn(state, graph)
    assert boosted.situations["StreetGangs"] > baseline
    assert "StreetGangs" in boosted.active_situations


def test_policy_effect_inertia_respected():
    data = simulator.load_simulation_data()
    state, graph = simulator.get_initial_state("uk")
    alcohol_effect = next(
        effect for effect in data.policies["AlcoholLaw"].effects if effect.target == "AlcoholConsumption"
    )
    baseline = state.effects.get(alcohol_effect.effect_id)
    assert baseline is not None
    state.policies["AlcoholLaw"] = 1.0
    next_state = simulator.process_end_of_turn(state, graph, data=data)
    context = {**state.values, **state.policies, **state.situations}
    target_value = simulator.evaluate_expression(alcohol_effect.expression, 1.0, context=context)
    inertia = alcohol_effect.inertia or 1.0
    expected = baseline + (target_value - baseline) / inertia
    assert next_state.effects[alcohol_effect.effect_id] == pytest.approx(expected)


def test_alcoholism_active_reduces_health():
    data = simulator.load_simulation_data()
    state, graph = simulator.get_initial_state("uk")
    assert "Alcoholism" in state.active_situations
    suppressed = replace(
        state,
        situations={**state.situations, "Alcoholism": 0.0},
        active_situations=[name for name in state.active_situations if name != "Alcoholism"],
        effects={
            key: (0.0 if key.startswith("situation::Alcoholism") else value)
            for key, value in state.effects.items()
        },
    )
    with_effect = simulator.process_end_of_turn(state, graph, data=data)
    without_effect = simulator.process_end_of_turn(suppressed, graph, data=data)
    assert with_effect.values["Health"] < without_effect.values["Health"]


def test_policy_costs_reflect_health_multiplier():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    baseline = state.policy_costs["StatePensions"]
    healthier = replace(state, values={**state.values, "Health": 1.0})
    simulator._recalculate_budget(healthier, data)
    assert healthier.policy_costs["StatePensions"] > baseline


def test_alcoholism_increases_state_health_service_cost():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    baseline = state.policy_costs["StateHealthService"]
    sober = replace(state, situations={**state.situations, "Alcoholism": 0.0})
    simulator._recalculate_budget(sober, data)
    assert baseline > sober.policy_costs["StateHealthService"]
