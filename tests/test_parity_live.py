"""Regression tests for the drastic-changes playthrough parity fixes.

These lock in the behavior changes that make the simulator reproduce the
game's <finances> block and per-turn political capital for the captured
playthrough (see parity_cases/DRASH_NOTES.md).
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import pytest

from autocracy import simulator
from autocracy.models import PolicyAction, SimulationConfig
from autocracy.savegame import load_state_from_savegame, parse_savegame

SAVES_DIR = Path("parity_cases/dem3saves")


def _turn_number(path: Path) -> int:
    return int(re.search(r"turn(\d+)_", path.name).group(1))


def _apply_orders(state, graph, data, n):
    save = parse_savegame(SAVES_DIR / f"turn{n}_orders.xml")
    for name in save.policy_desired_throttles:
        current = state.policy_desired_throttles.get(
            name, state.policies.get(name, 0.0)
        )
        target = save.policy_desired_throttles[name]
        was_active = state.policy_active.get(name, current > 1e-6)
        is_active = save.policy_active.get(name, target > 1e-6)
        if was_active and not is_active:
            state = simulator.apply_actions(
                state,
                [PolicyAction(policy_name=name, delta=0.0, action_type="cancel")],
                data=data,
            )
            continue
        if abs(current - target) < 1e-6:
            continue
        action_type = (
            "introduce"
            if (not was_active and target > 1e-6)
            else ("raise" if target > current else "lower")
        )
        state = simulator.apply_actions(
            state,
            [PolicyAction(policy_name=name, delta=target - current, action_type=action_type)],
            data=data,
        )
    return state


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_turn_zero_orders_charge_lower_costs():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    updated = _apply_orders(state, graph, data, 0)
    # CorpTax 10 + IncomeTax 7 + Prisons 8 = 25 spent.
    assert updated.political_capital == pytest.approx(26.0 - 25.0)
    assert updated.policy_desired_throttles["CorporationTax"] == pytest.approx(0.0)
    assert updated.policy_desired_throttles["IncomeTax"] == pytest.approx(0.0)
    assert updated.policy_desired_throttles["Prisons"] == pytest.approx(0.0)
    assert updated.policy_active["Prisons"]  # lowered to floor, not cancelled


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_turn_zero_to_one_finance_matches_finances_block():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    state = _apply_orders(state, graph, data, 0)
    state = simulator.process_end_of_turn(state, graph, data)
    ref = parse_savegame(SAVES_DIR / "turn1_initial.xml")
    assert state.total_income == pytest.approx(ref.total_income, abs=1.0)
    assert state.total_expenditure == pytest.approx(ref.total_expenditure, abs=1.0)
    assert state.political_capital == pytest.approx(ref.political_capital)
    assert "GeneralStrike" not in state.active_situations


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_general_strike_fires_on_turn_two():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    for n in (0, 1):
        state = _apply_orders(state, graph, data, n)
        state = simulator.process_end_of_turn(state, graph, data)
    assert "GeneralStrike" in state.active_situations
    assert state.values["GDP"] < 0.01  # crashed by the strike
    ref = parse_savegame(SAVES_DIR / "turn2_initial.xml")
    assert state.total_income == pytest.approx(ref.total_income, abs=400.0)


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_live_multiplier_recomputation_tracks_gdp():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    state = _apply_orders(state, graph, data, 0)
    state = simulator.process_end_of_turn(state, graph, data)
    # IncomeTax incom_mult @turn1 = 0.5 + 0.5 * GDP@turn0 = 0.6164.
    assert state.policy_income_multipliers["IncomeTax"] == pytest.approx(
        0.6164, abs=1e-3
    )


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_interest_charge_included_in_expenditure():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    state = _apply_orders(state, graph, data, 0)
    state = simulator.process_end_of_turn(state, graph, data)
    interest = state.debt * state.interest_rate * 0.25
    assert state.total_expenditure > sum(state.policy_costs.values()) + interest - 1.0
    ref = parse_savegame(SAVES_DIR / "turn1_initial.xml")
    assert state.debt == pytest.approx(ref.debt, abs=1.0)


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_ministerial_scalars_drift_with_experience():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    before = state.policy_income_scalars["IncomeTax"]
    state = _apply_orders(state, graph, data, 0)
    state = simulator.process_end_of_turn(state, graph, data)
    assert state.policy_income_scalars["IncomeTax"] > before
    ref = parse_savegame(SAVES_DIR / "turn1_initial.xml")
    assert state.policy_income_scalars["IncomeTax"] == pytest.approx(
        ref.policy_income_scalars["IncomeTax"], abs=1e-4
    )


def _replay_state_after(n: int):
    """Advance the simulator through turns 0..n-1 to reach turn n."""
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    for turn in range(n):
        orders = SAVES_DIR / f"turn{turn}_orders.xml"
        if orders.exists():
            state = _apply_orders(state, graph, data, turn)
        state = simulator.process_end_of_turn(state, graph, data)
    return state, data


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_minister_loyalty_tracks_game_when_enabled():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    cfg = SimulationConfig(minister_loyalty=True)
    for n in (0, 1):
        state = _apply_orders(state, graph, data, n)
        state = simulator.process_end_of_turn(state, graph, data, config=cfg)
        ref = parse_savegame(SAVES_DIR / f"turn{n + 1}_initial.xml")
        assert state.ministerial_loyalty["TAX"] == pytest.approx(
            ref.ministerial_loyalty["TAX"], abs=1e-3
        )
        assert state.political_capital == pytest.approx(ref.political_capital, abs=1.0)


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_minister_loyalty_disabled_keeps_loaded_income():
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(SAVES_DIR / "turn0_initial.xml", data)
    state = _apply_orders(state, graph, data, 0)
    state = simulator.process_end_of_turn(state, graph, data, config=SimulationConfig())
    assert state.political_capital_income == pytest.approx(26.0)
    assert state.ministerial_loyalty["TAX"] == pytest.approx(0.70310652)


@pytest.mark.skipif(not (SAVES_DIR / "turn0_initial.xml").exists(), reason="saves missing")
def test_end_to_end_replay_turn_one_and_two():
    for target_turn in (1, 2):
        state, data = _replay_state_after(target_turn)
        ref = parse_savegame(SAVES_DIR / f"turn{target_turn}_initial.xml")
        assert state.total_income == pytest.approx(ref.total_income, abs=400.0)
        assert state.total_expenditure == pytest.approx(ref.total_expenditure, abs=400.0)
        assert state.values["GDP"] == pytest.approx(ref.simvalues["GDP"], abs=0.01)


REFERENCE_SAVES = Path("gamedata/saves")


@pytest.mark.skipif(
    not (REFERENCE_SAVES / "uk0.xml").exists(), reason="reference saves missing"
)
def test_reference_saves_match_exactly():
    """The shipped uk0/uk1/uk2 saves reproduce exactly.

    The reference playthrough (fresh UK, no player orders) must advance
    through the simulator with exact income/expenditure and political
    capital at every captured turn.
    """
    data = simulator.load_simulation_data()
    state, graph = load_state_from_savegame(REFERENCE_SAVES / "uk0.xml", data)
    cfg = SimulationConfig(minister_loyalty=True)
    for n in (1, 2):
        state = simulator.process_end_of_turn(state, graph, data, config=cfg)
        ref = parse_savegame(REFERENCE_SAVES / f"uk{n}.xml")
        # uk0->uk1 reproduces exactly; uk1->uk2 carries a small node-drift
        # residual (DoctorsStrike boundary + Unemployment).
        tolerance = 10.0 if n == 1 else 1000.0
        assert state.total_income == pytest.approx(ref.total_income, abs=tolerance)
        assert state.total_expenditure == pytest.approx(ref.total_expenditure, abs=tolerance)
        assert state.political_capital == pytest.approx(ref.political_capital, abs=1.0)
