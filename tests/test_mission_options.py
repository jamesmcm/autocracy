from __future__ import annotations

from pathlib import Path

import pytest

from autocracy import simulator
from autocracy import events
from autocracy.events import (
    _load_country_scripts,
    _load_dilemmas,
    _load_events,
    apply_country_scripts,
    run_events,
)
from autocracy.models import SimulationConfig, SimulationState
from autocracy.savegame import parse_savegame


def _fresh_state(data, country, winning=1.0):
    state = SimulationState(
        country=country,
        turn=0,
        values={name: node.default for name, node in data.nodes.items()},
        policies={name: 0.0 for name in data.policies},
        political_capital=26.0,
        effects={},
        policy_finance_levels={},
        political_capital_income=26.0,
        global_economy_position=0.0,
        election_turns_until=16,
        election_current_term=0,
    )
    state.values["_winning_"] = winning
    return state


def test_option_prereqs_are_parsed_from_event_and_dilemma_files():
    data = simulator.load_simulation_data()
    loaded = _load_events(data.gamedata_root)
    dilemmas = _load_dilemmas(data.gamedata_root)
    assert loaded["NaturalDisaster"].prereqs == ["EARTHQUAKES"]
    assert loaded["NaturalDisaster2"].prereqs == ["HURRICANES"]
    assert dilemmas["BanFoxHunting"].prereqs == ["FOXES"]
    assert dilemmas["RoyalScandal"].prereqs == ["MONARCHY"]
    assert loaded["PrisonRiot"].prereqs == []


def test_gated_events_only_fire_for_countries_with_the_option():
    data = simulator.load_simulation_data()
    config = SimulationConfig(random_events=True, random_seed=7)
    usa = run_events(_fresh_state(data, "usa"), data, config).event_log
    france = run_events(_fresh_state(data, "france"), data, config).event_log
    assert any("NaturalDisaster" in entry for entry in usa)
    assert not any("NaturalDisaster" in entry for entry in france)


def test_gated_dilemmas_only_fire_for_countries_with_the_option():
    data = simulator.load_simulation_data()
    options = {
        country: events._country_options(data, country)
        for country in ("uk", "australia", "canada", "france", "germany", "usa")
    }
    loaded = events._load_dilemmas(data.gamedata_root)
    enabled = {
        country: sorted(
            dilemma.name
            for dilemma in loaded.values()
            if dilemma.prereqs
            and events._prereqs_satisfied(dilemma.prereqs, options[country])
        )
        for country in options
    }
    assert enabled["uk"] == ["BanFoxHunting", "RoyalScandal"]
    assert enabled["australia"] == ["RoyalScandal"]
    assert enabled["canada"] == ["RoyalScandal"]
    assert enabled["france"] == []
    assert enabled["germany"] == []
    assert enabled["usa"] == []


def test_country_scripts_parse_to_create_grudge_actions():
    data = simulator.load_simulation_data()
    actions = _load_country_scripts(data.gamedata_root, "australia")
    assert len(actions) == 5
    assert all(action.name == "CreateGrudge" for action in actions)
    by_target = {action.args[2]: action.args[3] for action in actions}
    assert by_target["Farmers_freq"] == "0.01"
    assert by_target["EthnicMinorities_freq"] == "-0.32"


@pytest.mark.skipif(
    not Path("gamedata/saves/australia0.xml").exists(),
    reason="reference country saves missing",
)
def test_apply_country_scripts_matches_serialized_save_grudges():
    data = simulator.load_simulation_data()
    for country in ("australia", "france", "usa", "germany", "canada", "uk"):
        save = parse_savegame(Path(f"gamedata/saves/{country}0.xml"))
        state = _fresh_state(data, country)
        apply_country_scripts(state, data)
        assert state.voter_frequency_grudges == pytest.approx(
            save.voter_frequency_grudges
        )
        by_target = {g.target: g.value for g in state.grudges}
        save_by_target = {g.target: g.value for g in save.grudges}
        assert set(by_target) == set(save_by_target)
        for target, value in by_target.items():
            assert value == pytest.approx(save_by_target[target])


@pytest.mark.skipif(
    not Path("gamedata/saves/australia0.xml").exists(),
    reason="reference country saves missing",
)
def test_get_initial_state_does_not_double_apply_scripts_from_save():
    save = parse_savegame(Path("gamedata/saves/australia0.xml"))
    state, _ = simulator.get_initial_state("australia")
    assert state.voter_frequency_grudges == pytest.approx(
        save.voter_frequency_grudges
    )


def test_attack_prereqs_reference_fired_plot_names():
    data = simulator.load_simulation_data()
    attacks = events._load_attacks(data.gamedata_root)
    assassination = attacks["CapitalistAssassination"]
    plot = attacks["CapitalistPlot"]
    assert assassination.prereqs == ["CapitalistPlot"]
    assert plot.prereqs == []


def test_compulsory_voting_raises_turnout_for_australia_only():
    for country, expected in (("australia", True), ("france", False), ("usa", False)):
        state, _ = simulator.get_initial_state(country)
        forecast = simulator.forecast_election(state)
        without = simulator._forecast_election_voters(
            state.voters,
            state.parties,
            compulsory_voting=False,
        )
        if expected:
            assert forecast.expected_absent_votes == pytest.approx(0.0)
            assert forecast.expected_player_votes + forecast.expected_opposition_votes == pytest.approx(
                without.expected_player_votes + without.expected_opposition_votes + without.expected_absent_votes
            )
        else:
            assert forecast.expected_absent_votes == pytest.approx(
                without.expected_absent_votes
            )