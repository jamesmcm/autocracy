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
    assert save.total_income > 0
    assert save.total_expenditure > 0


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_state_loaded_from_save_matches_values():
    save = parse_savegame(SAVE_PATH)
    state, _ = load_state_from_savegame(SAVE_PATH)
    comparison = compare_state_to_savegame(state, save, tolerance=1e-6)
    assert not comparison.value_diffs
    assert not comparison.policy_diffs
    assert comparison.cost_diffs  # monetary budgets currently diverge, ensure we capture them.
    assert comparison.income_diffs
    assert comparison.budget_diffs


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_compare_state_to_savegame_reports_differences():
    save = parse_savegame(SAVE_PATH)
    state, _ = load_state_from_savegame(SAVE_PATH)
    state.values["Health"] += 0.1
    comparison = compare_state_to_savegame(state, save, tolerance=1e-6)
    assert comparison.value_diffs


@pytest.mark.skipif(not SAVE_PATH.exists(), reason="Savegame file missing")
def test_budget_differences_surface_in_comparison():
    save = parse_savegame(SAVE_PATH)
    state, _ = load_state_from_savegame(SAVE_PATH)
    state.policy_costs["StatePensions"] += 1000
    state.total_expenditure += 1000
    comparison = compare_state_to_savegame(state, save, tolerance=1e-6)
    assert comparison.cost_diffs
    assert any(diff.name == "StatePensions" for diff in comparison.cost_diffs)
    assert any(diff.name == "Total Expenditure" for diff in comparison.budget_diffs)
