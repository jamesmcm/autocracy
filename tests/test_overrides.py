from __future__ import annotations

from pathlib import Path

import pytest

from autocracy import simulator
from autocracy.savegame import load_state_from_savegame
from autocracy.simulator import collect_node_effects


_REFERENCE_SAVES = [
    Path(f"gamedata/saves/{country}0.xml")
    for country in ("australia", "canada", "france", "germany", "uk", "usa")
]

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in _REFERENCE_SAVES),
    reason="reference country saves missing",
)


def test_situation_override_lowers_france_alcoholism_latent():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame("gamedata/saves/france0.xml", data)
    state = simulator.process_end_of_turn(state, graph, data)
    # France's alcoholabuse.ini rewrites AlcoholConsumption -> Alcoholism
    # from 0.9*x to 0.1*x.  Without the override the latent clamps at 1.0
    # and the situation wrongly activates; with it the latent stays below the
    # 0.6 start trigger, matching the native capture.
    assert state.situations["Alcoholism"] < 0.6
    assert "Alcoholism" not in state.active_situations


def test_situation_override_keeps_germany_general_strike_inactive():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame("gamedata/saves/germany0.xml", data)
    state = simulator.process_end_of_turn(state, graph, data)
    # Germany's poorearning_generalstrike.ini rewrites _LowIncome ->
    # GeneralStrike from 0.3-(0.6*x) to 0-(0.6*x), dropping the latent below
    # the activation threshold instead of letting a spurious strike drag GDP.
    assert state.situations["GeneralStrike"] < 0.6
    assert "GeneralStrike" not in state.active_situations


def test_graph_override_replaces_base_effect():
    data = simulator.load_simulation_data()
    graph = simulator.build_country_graph("germany", None)
    inputs, _ = collect_node_effects("WorkerProductivity", graph, data=data)
    alcohol_effects = [
        (effect.expression, effect.effect_id)
        for effect in inputs
        if effect.source == "AlcoholConsumption"
    ]
    # Germany's alcoholproductivity.ini replaces the base 0-(0.17*x) term
    # with 0-(0.07*x) rather than adding a second AlcoholConsumption line.
    assert alcohol_effects == [
        ("0-(0.07*x)", "override::AlcoholConsumption::WorkerProductivity::1")
    ]


def test_graph_override_does_not_break_unoverridden_edges():
    data = simulator.load_simulation_data()
    graph = simulator.build_country_graph("australia", None)
    inputs, _ = collect_node_effects("WorkerProductivity", graph, data=data)
    alcohol_effects = [
        effect for effect in inputs if effect.source == "AlcoholConsumption"
    ]
    # Australia has no alcoholproductivity override, so the base term stays.
    assert any(effect.expression == "0-(0.17*x)" for effect in alcohol_effects)


def test_income_percentages_recompute_on_first_noorder_pass():
    data = simulator.load_simulation_data()
    for country in ("uk", "usa", "france", "germany", "australia", "canada"):
        state, graph = load_state_from_savegame(
            f"gamedata/saves/{country}0.xml", data
        )
        # The save is before the first percentage pass: income groups are 0.
        assert state.voter_percentages.get("Poor_perc", 0.0) == 0.0
        state = simulator.process_end_of_turn(state, graph, data)
        expected = (
            sum(1 for voter in state.voters if voter.groups.get(12, 0.0) > 0.0)
            / len(state.voters)
        )
        # Every country recomputes the income-group percentages on the first
        # no-order pass (matching fresh native captures), not a UK-only freeze.
        assert state.voter_percentages.get("Poor_perc", 0.0) == pytest.approx(
            expected, abs=1e-9
        )


def test_nonincome_percentages_use_frequency_base():
    data = simulator.load_simulation_data()
    for country in ("uk", "usa", "france", "germany"):
        state, graph = load_state_from_savegame(
            f"gamedata/saves/{country}0.xml", data
        )
        # Percentages use the previous turn's frequency snapshot as the
        # membership base, matching the native manager ordering.
        previous_frequencies = state.voter_frequencies.copy()
        state = simulator.process_end_of_turn(state, graph, data)
        for symbol, name in simulator.VOTER_SYMBOL_NAMES.items():
            if (
                symbol in simulator.INCOME_GROUP_NODES
                or symbol in simulator.POLITICAL_GROUP_SYMBOLS
            ):
                continue
            frequency = previous_frequencies.get(f"{name}_freq", 0.0)
            expected = (
                sum(
                    1
                    for voter in state.voters
                    if voter.groups.get(symbol, 0.0) + frequency >= 0.5
                )
                / len(state.voters)
            )
            assert state.voter_percentages.get(f"{name}_perc", 0.0) == pytest.approx(
                expected, abs=1e-9
            )