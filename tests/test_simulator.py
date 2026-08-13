from __future__ import annotations

from dataclasses import replace

import pytest

from autocracy import simulator
from autocracy.agent import PassiveAgent
from autocracy.models import Grudge, PartyState, PolicyAction, Voter
from autocracy.savegame import load_state_from_savegame, parse_savegame


def test_state_serialization_round_trip(tmp_path):
    state, _ = simulator.get_initial_state("uk")
    state.policy_effect_history_started["StateHealthService"] = True
    state.grudges = [
        Grudge(
            target="GDP",
            value=-0.2,
            decay=0.75,
            source="CreditRatingDowngraded",
        )
    ]
    payload = simulator.state_to_dict(state)
    restored = simulator.state_from_dict(payload)
    assert restored.country == state.country
    assert restored.policies == state.policies
    assert restored.policy_desired_throttles == state.policy_desired_throttles
    assert restored.policy_income_histories == state.policy_income_histories
    assert restored.policy_finance_levels == state.policy_finance_levels
    assert restored.policy_cost_histories == state.policy_cost_histories
    assert restored.policy_effect_history_started == state.policy_effect_history_started
    assert restored.political_capital_income == state.political_capital_income
    assert restored.voter_frequency_grudges == state.voter_frequency_grudges
    assert restored.grudges == state.grudges
    assert restored.voter_incomes == state.voter_incomes
    assert restored.election_turns_until == state.election_turns_until
    assert restored.election_current_term == state.election_current_term
    assert restored.poll_history == state.poll_history
    out_path = tmp_path / "state.json"
    simulator.save_state(state, out_path)
    reloaded = simulator.load_state(out_path)
    assert reloaded.values == state.values


def test_interest_rate_includes_global_interest_neuron():
    data = simulator.load_simulation_data()
    baseline = simulator._interest_rate(2, data, 0.5)
    elevated = simulator._interest_rate(2, data, 0.75)
    assert elevated > baseline


def test_apply_actions_allows_small_adjustments_within_bounds():
    state, _ = simulator.get_initial_state("uk")
    updated = simulator.apply_actions(
        state,
        [PolicyAction(policy_name="BusLanes", delta=0.1)],
    )
    assert updated.policies["BusLanes"] == state.policies["BusLanes"]
    assert updated.policy_desired_throttles["BusLanes"] > state.policies["BusLanes"]


def test_policy_slider_change_uses_fixed_step_output_throttle():
    data = simulator.load_simulation_data()
    state, graph = simulator.get_initial_state("uk")
    policy = data.policies["BusLanes"]
    current = state.policies[policy.name]
    current_throttle = state.effect_throttles[policy.name]
    updated = simulator.apply_actions(
        state,
        [PolicyAction(policy_name=policy.name, delta=0.05)],
        data=data,
    )

    assert updated.policy_desired_throttles[policy.name] == pytest.approx(current + 0.05)
    assert updated.effect_throttles[policy.name] == pytest.approx(current_throttle)

    advanced = simulator.process_end_of_turn(updated, graph, data=data)
    expected = min(
        current + 0.05,
        current_throttle + 1.0 / policy.implementation_time,
    )
    assert advanced.effect_throttles[policy.name] == pytest.approx(expected)
    assert advanced.policies[policy.name] == pytest.approx(expected)


def test_native_order_runtime_sets_slider_value_before_next_turn():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    updated = simulator.apply_actions(
        state,
        [
            PolicyAction(
                policy_name="StateHealthService",
                delta=-state.policies["StateHealthService"],
                action_type="lower",
            )
        ],
        data=data,
        native_order_runtime=True,
    )

    assert updated.policies["StateHealthService"] == pytest.approx(0.0)
    assert updated.effect_throttles["StateHealthService"] == pytest.approx(0.0)
    assert updated.policy_desired_throttles["StateHealthService"] == pytest.approx(
        0.0
    )


def test_native_order_batch_keeps_previous_finance_levels_for_debt_preview():
    data = simulator.load_simulation_data()
    state, _ = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )
    actions = [
        PolicyAction("CorporationTax", -0.22, action_type="lower"),
        PolicyAction("IncomeTax", -0.45, action_type="lower"),
        PolicyAction("Prisons", -0.5, action_type="lower"),
    ]

    updated = simulator.apply_actions(
        state, actions, data=data, native_order_runtime=True
    )

    assert updated.total_income == pytest.approx(state.total_income)
    assert updated.total_expenditure == pytest.approx(state.total_expenditure)
    assert updated.policy_finance_levels["CorporationTax"] == pytest.approx(0.22)
    assert updated.policy_finance_levels["IncomeTax"] == pytest.approx(0.45)


def test_native_introduction_seeds_previous_finance_level():
    data = simulator.load_simulation_data()
    state, _ = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )

    updated = simulator.apply_actions(
        state,
        [PolicyAction("PrivatePrisons", 1.0, action_type="introduce")],
        data=data,
        native_order_runtime=True,
    )

    assert updated.policies["PrivatePrisons"] == pytest.approx(1.0)
    assert updated.policy_finance_levels["PrivatePrisons"] == pytest.approx(0.5)


def test_policy_introduction_exposes_effects_during_implementation():
    data = simulator.load_simulation_data()
    state, graph = simulator.get_initial_state("uk")
    policy = data.policies["MicrogenerationGrants"]
    updated = simulator.apply_actions(
        state,
        [PolicyAction(policy_name=policy.name, delta=0.5)],
        data=data,
    )

    assert updated.policy_active[policy.name]
    assert updated.policy_implementations[policy.name] == pytest.approx(0.0)
    advanced = simulator.process_end_of_turn(updated, graph, data=data)
    _, _, next_effectiveness = simulator._advance_ministers(state, data)
    expected_implementation = (
        next_effectiveness[policy.department] / policy.implementation_time
    )
    assert advanced.policy_implementations[policy.name] == pytest.approx(
        expected_implementation
    )
    assert advanced.effect_throttles[policy.name] == pytest.approx(0.5)
    assert advanced.policies[policy.name] == pytest.approx(0.5)


def test_process_end_of_turn_advances_turn_and_updates_values():
    state, graph = simulator.get_initial_state("uk")
    next_state = simulator.process_end_of_turn(state, graph)
    assert next_state.turn == state.turn + 1
    assert next_state.values["GDP"] != pytest.approx(state.values["GDP"])


def test_create_grudge_is_a_decaying_neural_input():
    data = simulator.load_simulation_data()
    clean, graph = load_state_from_savegame("gamedata/saves/uk0.xml", data)
    with_grudge = replace(
        clean,
        grudges=[
            Grudge(
                target="GDP",
                value=-0.2,
                decay=0.75,
                source="CreditRatingDowngraded",
            )
        ],
    )

    clean_next = simulator.process_end_of_turn(clean, graph, data)
    grudge_next = simulator.process_end_of_turn(with_grudge, graph, data)

    assert grudge_next.values["GDP"] < clean_next.values["GDP"]
    assert grudge_next.grudges[0].value == pytest.approx(-0.15)


def test_native_runtime_bridges_restore_serialized_manager_fields():
    state, _ = simulator.get_initial_state("uk")
    policy = simulator.apply_native_policy_runtime(
        state,
        policy_implementations={"BusLanes": 0.25},
        policy_active={"BusLanes": True},
        policy_cost_multipliers={"BusLanes": 1.1},
        policy_income_multipliers={"BusLanes": 0.9},
        policy_cost_scalars={"BusLanes": 0.8},
        policy_income_scalars={"BusLanes": 1.2},
        effect_throttles={"BusLanes": 0.125},
    )
    restored = simulator.apply_native_finance_runtime(
        policy,
        total_income=123.0,
        total_expenditure=456.0,
        debt=789.0,
        interest_rate=0.12,
        credit_rating=3,
        turns_since_credit=1,
        policy_costs={"BusLanes": 10.0},
        policy_incomes={"BusLanes": 20.0},
        policy_cost_histories={"BusLanes": [10.0]},
        policy_income_histories={"BusLanes": [20.0]},
    )

    assert restored.policy_implementations["BusLanes"] == pytest.approx(0.25)
    assert restored.effect_throttles["BusLanes"] == pytest.approx(0.125)
    assert restored.total_income == pytest.approx(123.0)
    assert restored.total_expenditure == pytest.approx(456.0)
    assert restored.debt == pytest.approx(789.0)
    assert restored.credit_rating == 3
    assert restored.policy_cost_histories["BusLanes"] == [10.0]


def test_year_history_uses_native_normalized_samples():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame("gamedata/saves/uk0.xml", data)

    for _ in range(8):
        state = simulator.process_end_of_turn(state, graph, data=data)

    assert state.values["_year"] == pytest.approx(1.75)
    assert state.hidden_histories["_year"][0] == pytest.approx(1.0)


def test_process_end_of_turn_preserves_previous_manager_snapshot():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )
    before_voters = [
        (v.value, v.income, v.party, v.last_vote, dict(v.groups))
        for v in state.voters
    ]
    before_parties = {
        name: (
            party.status,
            party.members_last_turn,
            tuple(party.member_history),
            tuple(party.activist_history),
        )
        for name, party in state.parties.items()
    }
    before_values = state.voter_values.copy()

    next_state = simulator.process_end_of_turn(state, graph, data=data)

    assert next_state.voters is not state.voters
    assert [
        (v.value, v.income, v.party, v.last_vote, dict(v.groups))
        for v in state.voters
    ] == before_voters
    assert {
        name: (
            party.status,
            party.members_last_turn,
            tuple(party.member_history),
            tuple(party.activist_history),
        )
        for name, party in state.parties.items()
    } == before_parties
    assert state.voter_values == before_values


def test_election_countdown_matches_native_headless_turns():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )

    for expected in (*range(15, -1, -1), 15):
        state = simulator.process_end_of_turn(state, graph, data=data)
        assert state.election_turns_until == expected
        assert state.election_current_term == 0


def test_resolve_election_counts_votes_and_starts_new_term():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    state = replace(
        state,
        election_turns_until=0,
        parties={
            "Player": PartyState("Player", party_type=0),
            "Opposition": PartyState("Opposition", party_type=1),
        },
        voters=[
            Voter(party="Player"),
            Voter(party="Opposition"),
            Voter(player_sympathy=0.8),
        ],
        ministerial_loyalty={"TAX": 0.5},
    )

    result = simulator.resolve_election(state, data)

    assert result.election_result == "win"
    assert result.last_election_winner == "player"
    assert result.election_player_votes == 3
    assert result.election_opposition_votes == 0
    assert result.election_absent_votes == 0
    assert result.election_current_term == 1
    assert result.election_turns_until == 16
    assert [v.last_vote for v in result.voters] == [0, 0, 0]
    assert result.ministerial_loyalty["TAX"] == pytest.approx(0.62)


def test_election_forecast_models_turnout_for_unaffiliated_voters():
    state, _ = simulator.get_initial_state("uk")
    state = replace(
        state,
        parties={
            "Player": PartyState("Player", party_type=0),
            "Opposition": PartyState("Opposition", party_type=1),
        },
        voters=[
            Voter(party="Player"),
            Voter(party="Opposition"),
            Voter(value=0.0, voting_tech=0.8),
            Voter(value=-1.0, voting_tech=0.8),
        ],
    )

    forecast = simulator.forecast_election(state)

    assert forecast.expected_player_votes == pytest.approx(2.21)
    assert forecast.expected_opposition_votes == pytest.approx(0.41)
    assert forecast.expected_absent_votes == pytest.approx(1.38)
    assert forecast.margin == pytest.approx(1.8)
    assert forecast.winner == "player"

    resolved = simulator.resolve_election(
        replace(state, election_turns_until=0),
    )
    assert [v.last_vote for v in resolved.voters] == [0, 0, 2, 1]


def test_native_manager_roster_can_remove_all_missing_departments():
    state, _ = simulator.get_initial_state("uk")
    state = simulator.apply_native_manager_roster(state, {"ECONOMY"})

    assert set(state.ministerial_loyalty) == {"ECONOMY"}
    assert set(state.ministerial_experience) == {"ECONOMY"}
    assert set(state.ministerial_sympathies) == {"ECONOMY"}

    empty = simulator.apply_native_manager_roster(state, set())
    assert not empty.ministerial_loyalty


def test_party_membership_uses_native_approval_and_sympathy_guards():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    state.voters = [
        # Approval is 0, so opposition sympathy rises and this voter joins.
        Voter(value=-1.0, opposition_sympathy=0.7, player_sympathy=0.1),
        # Raw value -0.7 maps to approval 0.15; it is not below the 0.1
        # opposition-gain threshold.
        Voter(value=-0.7, party="0"),
        # A strongly approving unaffiliated voter joins the player party.
        Voter(value=1.0, player_sympathy=0.7, party="0"),
        # An existing opposition member below the leave threshold departs.
        Voter(
            value=1.0,
            opposition_sympathy=0.1,
            party="Opposition",
        ),
    ]
    state.parties = {
        "Opposition": PartyState("Opposition", party_type=1, member_history=[4]),
        "Player": PartyState("Player", party_type=0, member_history=[2]),
    }

    simulator._advance_party_memberships(state, data)

    assert state.voters[0].opposition_sympathy == pytest.approx(0.8)
    assert state.voters[0].party == "Opposition"
    assert state.voters[1].opposition_sympathy == pytest.approx(0.0)
    assert state.voters[1].party == "0"
    assert state.voters[2].player_sympathy == pytest.approx(0.8)
    assert state.voters[2].party == "Player"
    assert state.voters[3].party == "0"
    assert state.parties["Opposition"].status == 0
    assert state.parties["Opposition"].members_last_turn == 1
    assert state.parties["Opposition"].member_history == [1, 4]
    assert state.parties["Player"].status == 2
    assert state.parties["Player"].members_last_turn == 0
    assert state.parties["Player"].member_history == [0, 2]


def test_income_groups_use_native_sine_windows_and_threshold():
    data = simulator.load_simulation_data()

    wealthy = Voter(inincome=0.88792241)
    middle = Voter(inincome=0.5)
    poor = Voter(inincome=0.1)

    wealthy_groups = simulator._native_income_group_memberships(wealthy, 0.5)
    middle_groups = simulator._native_income_group_memberships(middle, 0.5)
    poor_groups = simulator._native_income_group_memberships(poor, 0.5)

    assert wealthy_groups[11] == pytest.approx(0.91634818, abs=1e-6)
    assert wealthy_groups[12] == pytest.approx(0.0)
    assert middle_groups[13] == pytest.approx(1.0)
    assert poor_groups[12] == pytest.approx(0.9330127, abs=1e-6)
    assert data.sim_config["VOTER_GROUP_MEMBERSHIP_THRESHHOLD"] == pytest.approx(0.5)


def test_noop_replay_reconstructs_income_group_memberships():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )
    reference = parse_savegame("parity_cases/dem3saves/turn1_initial.xml")

    advanced = simulator.process_end_of_turn(state, graph, data=data)

    assert advanced.voters[0].groups[11] == pytest.approx(
        reference.voters[0].groups[11], abs=1e-6
    )
    for name in ("Wealthy_perc", "Poor_perc", "MiddleIncome_perc"):
        assert advanced.voter_percentages[name] == pytest.approx(
            reference.voter_percentages[name]
        )
    for name in state.parties:
        assert advanced.parties[name].activist_history[:3] == reference.parties[
            name
        ].activist_history[:3]


def test_active_situation_outputs_reach_voter_values():
    """GeneralStrike's active output is part of the native voter effect vector."""
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn1_initial.xml", data
    )
    reference = parse_savegame("parity_cases/dem3saves/turn2_initial.xml")

    advanced = simulator.process_end_of_turn(state, graph, data=data)

    assert "GeneralStrike" in advanced.active_situations
    assert advanced.voter_values["Conservatives"] == pytest.approx(
        reference.voter_values["Conservatives"], abs=0.05
    )


def test_voter_percentages_use_previous_frequency_snapshot():
    """Ordinary group links use the saved VoterType base for this pass."""
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn1_initial.xml", data
    )
    previous_frequency = state.voter_frequencies["Commuter_freq"]
    threshold = data.sim_config["VOTER_GROUP_MEMBERSHIP_THRESHHOLD"]
    expected = sum(
        voter.groups.get(3, 0.0) + previous_frequency >= threshold
        for voter in state.voters
    ) / len(state.voters)

    advanced = simulator.process_end_of_turn(state, graph, data=data)

    assert advanced.voter_percentages["Commuter_perc"] == pytest.approx(expected)


def test_delayed_policy_ring_samples_after_one_ramp_turn():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn7_ordes.xml", data
    )
    state.political_capital = 100.0
    original = next(
        history.values[0]
        for history in state.effect_histories
        if (history.source, history.target) == ("StateHealthService", "Health")
    )
    state = simulator.apply_actions(
        state,
        [
            PolicyAction(
                policy_name="StateHealthService",
                delta=-0.68,
                action_type="lower",
            )
        ],
        data=data,
    )
    first = simulator.process_end_of_turn(state, graph, data=data)
    first_head = next(
        history.values[0]
        for history in first.effect_histories
        if (history.source, history.target) == ("StateHealthService", "Health")
    )
    second = simulator.process_end_of_turn(first, graph, data=data)
    second_head = next(
        history.values[0]
        for history in second.effect_histories
        if (history.source, history.target) == ("StateHealthService", "Health")
    )

    assert first_head == pytest.approx(original)
    assert second_head != pytest.approx(original)


def test_state_health_ring_waits_for_an_explicit_order():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )
    original = next(
        history.values[0]
        for history in state.effect_histories
        if (history.source, history.target) == ("StateHealthService", "Health")
    )

    advanced = simulator.process_end_of_turn(state, graph, data=data)
    current = next(
        history.values[0]
        for history in advanced.effect_histories
        if (history.source, history.target) == ("StateHealthService", "Health")
    )

    assert current == pytest.approx(original)
    assert not advanced.policy_effect_history_started.get(
        "StateHealthService", False
    )


def test_private_schools_ring_waits_on_the_loaded_native_head():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )
    original = next(
        history.values[0]
        for history in state.effect_histories
        if (history.source, history.target) == ("PrivateSchools", "Education")
    )

    advanced = simulator.process_end_of_turn(state, graph, data=data)
    current = next(
        history.values[0]
        for history in advanced.effect_histories
        if (history.source, history.target) == ("PrivateSchools", "Education")
    )

    assert current == pytest.approx(original)


def test_poll_rate_counts_live_voters_with_native_party_cutoffs():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )
    opposition = next(
        party.name for party in state.parties.values() if party.party_type == 1
    )
    state.voters = [
        Voter(value=0.0),
        Voter(value=0.2),
        Voter(value=0.199, party=opposition),
        Voter(value=-0.2),
    ]

    assert simulator._calculate_poll_rate(state) == pytest.approx(0.25)

    advanced = simulator.process_end_of_turn(state, graph, data=data)
    assert advanced.poll_rate == pytest.approx(
        simulator._calculate_poll_rate(
            replace(state, voters=advanced.voters)
        )
    )
    assert advanced.poll_history[0] == pytest.approx(advanced.poll_rate)


def test_year_neuron_is_monotonic_quarter_counter():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(
        "parity_cases/dem3saves/turn0_initial.xml", data
    )
    first = simulator.process_end_of_turn(state, graph, data=data)
    second = simulator.process_end_of_turn(first, graph, data=data)

    assert first.values["_year"] == pytest.approx(0.0)
    assert second.values["_year"] == pytest.approx(0.25)
    assert second.hidden_histories["_year"][:2] == pytest.approx([0.25, 0.0])


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
        opt for opt in options if opt.policy_name == "BusLanes" and opt.action_type == "lower"
    ]
    cancels = [
        opt for opt in options if opt.policy_name == "BusLanes" and opt.action_type == "cancel"
    ]
    assert lowers, "Expected lower option"
    assert cancels, "Expected cancel option"
    assert lowers[0].cost == data.policies["BusLanes"].lower_cost
    assert cancels[0].cost == data.policies["BusLanes"].cancel_cost
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
            [
                PolicyAction(
                    policy_name="BorderControls",
                    delta=-state.policies["BorderControls"],
                    action_type="cancel",
                )
            ],
        )


def test_uncancellable_can_be_lowered_to_floor():
    state, _ = simulator.get_initial_state("uk")
    policy = state.policies["BorderControls"]
    updated = simulator.apply_actions(
        state,
        [PolicyAction(policy_name="BorderControls", delta=-policy)],
    )
    assert updated.policy_desired_throttles["BorderControls"] == pytest.approx(0.0)
    assert updated.policy_active["BorderControls"]
    assert updated.political_capital == pytest.approx(
        state.political_capital - 13.0  # BorderControls lower_cost
    )


def test_initial_state_matches_initial_save():
    save = parse_savegame("gamedata/saves/uk0.xml")
    state, _ = simulator.get_initial_state("uk")
    for name, value in save.simvalues.items():
        assert state.values[name] == pytest.approx(value, rel=1e-3, abs=1e-3)
    for name, value in save.policies.items():
        assert state.policies[name] == pytest.approx(value, rel=1e-6, abs=1e-6)


def test_native_immigration_equation_calibration_matches_both_inputs():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame("gamedata/saves/uk0.xml", data)
    actual = simulator.process_end_of_turn(state, graph, data)
    expected = parse_savegame("gamedata/saves/uk1.xml")

    assert actual.values["Immigration"] == pytest.approx(
        expected.simvalues["Immigration"], abs=2e-4
    )
    assert actual.values["Wages"] == pytest.approx(
        expected.simvalues["Wages"], abs=2e-4
    )


def test_native_simvalue_checkpoint_overlay_does_not_mutate_source():
    state, _ = simulator.get_initial_state("uk")
    original = state.values["GDP"]
    restored = simulator.apply_native_sim_values(
        state,
        {"GDP": 0.123, "_not_a_loaded_node": 0.456},
    )

    assert state.values["GDP"] == original
    assert restored.values["GDP"] == pytest.approx(0.123)
    assert restored.values["_not_a_loaded_node"] == pytest.approx(0.456)


def test_initial_state_does_not_use_hidden_response_calibration():
    state, _ = simulator.get_initial_state("uk")
    assert not state.response_factors


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
    history = next(
        item
        for item in next_state.effect_histories
        if item.effect_id == alcohol_effect.effect_id
    )
    inertia = max(1, int(alcohol_effect.inertia or 0.0))
    expected = sum(history.values[:inertia]) / inertia
    expected *= simulator._policy_effect_scale(next_state, alcohol_effect, data)
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
    simulator._recalculate_budget(
        healthier, data, use_serialized_runtime_fields=False
    )
    assert healthier.policy_costs["StatePensions"] > baseline


def test_alcoholism_increases_state_health_service_cost():
    data = simulator.load_simulation_data()
    state, _ = simulator.get_initial_state("uk")
    baseline = state.policy_costs["StateHealthService"]
    sober = replace(state, situations={**state.situations, "Alcoholism": 0.0})
    simulator._recalculate_budget(sober, data, use_serialized_runtime_fields=False)
    assert baseline > sober.policy_costs["StateHealthService"]
