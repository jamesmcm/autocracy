from __future__ import annotations

from pathlib import Path

import pytest

from autocracy.savegame import (
    compare_state_to_savegame,
    load_state_from_savegame,
    parse_savegame,
)


SAVE_PATH = Path("gamedata/saves/uk0.xml")


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_parse_savegame_extracts_core_sections():
    save = parse_savegame(SAVE_PATH)
    assert save.country == "uk"
    assert save.turn == 0
    assert "Health" in save.simvalues
    assert pytest.approx(0.60939771, rel=0, abs=1e-6) == save.simvalues["Health"]
    assert "FoodStandards" in save.policies
    assert save.policy_costs["StateHealthService"] > 0
    assert save.policy_incomes["IncomeTax"] >= 0
    assert len(save.policy_income_histories["IncomeTax"]) == 20
    assert save.policy_income_histories["IncomeTax"][0] == pytest.approx(
        save.policy_incomes["IncomeTax"]
    )
    assert save.policy_desired_throttles["BusLanes"] == pytest.approx(
        save.policies["BusLanes"]
    )
    assert save.political_capital == pytest.approx(26.0)
    assert save.election_turns_until == 16
    assert save.election_current_term == 0
    assert save.poll_rate == pytest.approx(0.113)
    assert save.peak_poll_rate == pytest.approx(0.13277972)
    assert len(save.poll_history) == 20
    assert save.total_income > 0
    assert save.total_expenditure > 0
    assert len(save.hidden_histories["_globaleconomy_"]) == 33
    assert save.hidden_histories["_globaleconomy_"][0] == pytest.approx(0.270)
    assert len(save.hidden_histories["_year"]) == 33
    assert save.voter_incomes["Retired_income"] == pytest.approx(0.31690958)


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_parse_savegame_preserves_voter_runtime_fields():
    save = parse_savegame("parity_cases/dem3saves/turn0_initial.xml")
    voter = save.voters[0]
    assert voter.militancy == pytest.approx(0.52753878)
    assert voter.voting_tech == pytest.approx(0.99177366)
    assert voter.initial_socialism == pytest.approx(0.87373853)
    assert voter.initial_liberalism == pytest.approx(0.54105872)
    assert voter.gender == 1
    assert voter.opposition_sympathy == pytest.approx(0.0)
    assert voter.last_vote == 2
    assert voter.survival == 27
    assert voter.forecast == 90
    assert voter.party == "0"

    party_voter = next(v for v in save.voters if v.party == "The National Front")
    assert party_voter.opposition_sympathy == pytest.approx(1.0)
    national_front = save.parties["The National Front"]
    assert national_front.status == 2
    assert national_front.party_type == 1
    assert national_front.member_history[:3] == [422, 434, 407]
    assert national_front.activist_history[:3] == [55, 52, 59]


@pytest.mark.skipif(
    not Path("parity_cases/dem3saves/turn0_initial.xml").exists(),
    reason="parity saves missing",
)
def test_state_snapshot_round_trips_voter_runtime_fields():
    from autocracy.simulator import state_from_dict, state_to_dict

    state, _ = load_state_from_savegame("parity_cases/dem3saves/turn0_initial.xml")
    restored = state_from_dict(state_to_dict(state))
    assert restored.voters[0].voting_tech == pytest.approx(
        state.voters[0].voting_tech
    )
    assert restored.voters[0].organizations == state.voters[0].organizations
    assert restored.parties["The National Front"].member_history == state.parties[
        "The National Front"
    ].member_history
    assert restored.hidden_histories == state.hidden_histories
    assert restored.voter_incomes == state.voter_incomes


def test_parse_savegame_keeps_policy_target_separate_from_current_value(tmp_path):
    save_path = tmp_path / "target.xml"
    save_path.write_text(
        """
        <policies>
          <policy>
            <name>BusLanes</name>
            <val>0.5</val>
            <targ>0.75</targ>
          </policy>
        </policies>
        """,
        encoding="latin-1",
    )

    save = parse_savegame(save_path)

    assert save.policies["BusLanes"] == pytest.approx(0.5)
    assert save.policy_desired_throttles["BusLanes"] == pytest.approx(0.75)


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_state_loaded_from_save_matches_values():
    save = parse_savegame(SAVE_PATH)
    state, _ = load_state_from_savegame(SAVE_PATH)
    comparison = compare_state_to_savegame(state, save, tolerance=1e-2)
    assert not comparison.value_diffs
    assert not comparison.policy_diffs
    # The save carries the game's current internal finance-neuron values, so
    # loading the same snapshot should reproduce its monetary lines too. XML
    # serialization leaves only small decimal-rounding differences.
    assert not comparison.cost_diffs
    assert not comparison.income_diffs
    assert not comparison.budget_diffs


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_compare_state_to_savegame_reports_differences():
    save = parse_savegame(SAVE_PATH)
    state, _ = load_state_from_savegame(SAVE_PATH)
    state.values["Health"] += 0.1
    comparison = compare_state_to_savegame(state, save, tolerance=1e-2)
    assert comparison.value_diffs


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_budget_differences_surface_in_comparison():
    save = parse_savegame(SAVE_PATH)
    state, _ = load_state_from_savegame(SAVE_PATH)
    state.policy_cost_histories["StatePensions"][0] += 1000
    state.total_expenditure += 1000
    comparison = compare_state_to_savegame(state, save, tolerance=1e-6)
    assert comparison.cost_diffs
    assert any(diff.name == "StatePensions" for diff in comparison.cost_diffs)
    assert any(diff.name == "Total Expenditure" for diff in comparison.budget_diffs)
