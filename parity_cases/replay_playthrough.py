"""Replay the captured Democracy 3 playthrough and compare to ground truth.

The ground-truth saves under parity_cases/dem3saves/ were taken from the real
game (v1.30.2) during a single 12-turn playthrough that made drastic policy
changes.  Each turn has an ``_orders`` save (state after placing orders,
before ending the turn) and an ``_initial`` save (state at the start of the
next turn).

This harness drives the simulator through the same orders and reports how
closely its per-turn state matches the game's saves.

Usage:
    uv run python parity_cases/replay_playthrough.py [--verbose]
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

from autocracy import simulator
from autocracy.models import PolicyAction, SimulationConfig, SimulationState
from autocracy.savegame import load_state_from_savegame, parse_savegame

SAVES_DIR = Path("parity_cases/dem3saves")

REPLAY_CONFIG = SimulationConfig(minister_loyalty=True)

DEFAULT_METRICS = ["GDP", "Health", "Education", "CrimeRate", "Unemployment", "IncomeTax", "SalesTax"]


def load_orders(
    save_path: Path, state: "SimulationState"
) -> list[PolicyAction]:
    save = parse_savegame(save_path)
    actions = []
    for name in save.policy_desired_throttles:
        # The game's orders save the slider *target*; a target that already
        # matches the current desired throttle is a no-op (the player did not
        # move that slider this turn).
        current = state.policy_desired_throttles.get(
            name, state.policies.get(name, 0.0)
        )
        target = save.policy_desired_throttles[name]
        # The game distinguishes dragging a slider to its floor (a ``lower``,
        # policy stays active) from a true switch-off (a ``cancel``, the
        # policy's active flag flips).  The orders save records the active
        # flag after the action, so a flip tells us which it was.
        was_active = state.policy_active.get(name, current > 1e-6)
        is_active = save.policy_active.get(name, target > 1e-6)
        if was_active and not is_active:
            # Cancellation keeps the value/target unchanged, so delta is 0.
            actions.append(
                PolicyAction(policy_name=name, delta=0.0, action_type="cancel")
            )
            continue
        if abs(current - target) < 1e-6:
            continue
        if not was_active and target > 1e-6:
            action_type = "introduce"
        elif target > current:
            action_type = "raise"
        else:
            action_type = "lower"
        actions.append(
            PolicyAction(
                policy_name=name, delta=target - current, action_type=action_type
            )
        )
    return actions


def turn_number(path: Path) -> int:
    return int(re.search(r"turn(\d+)_", path.name).group(1))


def main(verbose: bool) -> None:
    data = simulator.load_simulation_data()
    initial = SAVES_DIR / "turn0_initial.xml"
    state, graph = load_state_from_savegame(initial, data)
    ref = parse_savegame(initial)

    print(f"{'turn':>4} | {'income':>12} {'exp':>12} | {'GDP':>7} {'GS':>5} {'GSact':>5} | income err | exp err")
    print("-" * 96)

    turns = sorted(
        (Path(p) for p in glob.glob(str(SAVES_DIR / "turn*_o*.xml"))),
        key=turn_number,
    )
    for orders_path in turns:
        n = turn_number(Path(orders_path))
        # The capture skips some turns (e.g. no turn3_orders).  The orders
        # file for turn n is placed on top of the state at the *start* of
        # turn n, so advance past any missing turns first.
        while state.turn < n:
            state = simulator.process_end_of_turn(state, graph, data=data, config=REPLAY_CONFIG)
        orders = load_orders(Path(orders_path), state)
        for action in orders:
            try:
                state = simulator.apply_actions(state, [action], data=data)
            except Exception as exc:
                print(f"  turn {n} action {action.policy_name}: ERROR {exc}")
        state = simulator.process_end_of_turn(state, graph, data=data, config=REPLAY_CONFIG)

        # Ground truth: the _initial save for the NEXT turn
        nxt = SAVES_DIR / f"turn{n + 1}_initial.xml"
        if not nxt.exists():
            continue
        ref = parse_savegame(nxt)
        income_err = state.total_income - ref.total_income
        exp_err = state.total_expenditure - ref.total_expenditure
        gs = state.situations.get("GeneralStrike", 0.0)
        gs_active = "GeneralStrike" in state.active_situations
        print(
            f"{n:>4} | {state.total_income:>12,.1f} {state.total_expenditure:>12,.1f} | "
            f"{state.values.get('GDP',0):>7.3f} {gs:>5.3f} {str(gs_active):>5} | "
            f"{income_err:>+10,.1f} | {exp_err:>+10,.1f}"
        )

        if verbose:
            for name in DEFAULT_METRICS:
                print(f"    {name}: sim={state.values.get(name, 0):.3f} game={ref.simvalues.get(name, 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.verbose)
