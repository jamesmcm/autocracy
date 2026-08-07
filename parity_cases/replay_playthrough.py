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
from autocracy.models import PolicyAction
from autocracy.savegame import load_state_from_savegame, parse_savegame

SAVES_DIR = Path("parity_cases/dem3saves")

DEFAULT_METRICS = ["GDP", "Health", "Education", "CrimeRate", "Unemployment", "IncomeTax", "SalesTax"]


def load_orders(save_path: Path) -> list[PolicyAction]:
    save = parse_savegame(save_path)
    actions = []
    for name in save.policy_desired_throttles:
        current = save.policies.get(name, 0.0)
        target = save.policy_desired_throttles[name]
        if abs(current - target) > 1e-6:
            actions.append(PolicyAction(policy_name=name, delta=target - current))
    return actions


def turn_number(path: Path) -> int:
    return int(re.search(r"turn(\d+)_", path.name).group(1))


def main(verbose: bool) -> None:
    data = simulator.load_simulation_data()
    initial = SAVES_DIR / "turn0_initial.xml"
    state, graph = load_state_from_savegame(initial, data)
    ref = parse_savegame(initial)

    print(f"{'turn':>4} | {'income':>12} {'exp':>12} | {'GDP':>7} {'GS':>5} {'GSact':>5} | income err")
    print("-" * 80)

    turns = sorted(
        (Path(p) for p in glob.glob(str(SAVES_DIR / "turn*_orders.xml"))),
        key=turn_number,
    )
    for orders_path in turns:
        n = turn_number(Path(orders_path))
        orders = load_orders(Path(orders_path))
        for action in orders:
            try:
                state = simulator.apply_actions(state, [action], data=data)
            except Exception as exc:
                print(f"  turn {n} action {action.policy_name}: ERROR {exc}")
        state = simulator.process_end_of_turn(state, graph, data=data)

        # Ground truth: the _initial save for the NEXT turn
        nxt = SAVES_DIR / f"turn{n}_initial.xml"
        if not nxt.exists():
            # orders and initial share a turn number offset; find the matching one
            candidates = sorted(
                (Path(p) for p in glob.glob(str(SAVES_DIR / "turn*_initial.xml"))),
                key=turn_number,
            )
            for cand in candidates:
                if turn_number(Path(cand)) > n:
                    nxt = Path(cand)
                    break
        if not nxt.exists():
            continue
        ref = parse_savegame(nxt)
        income_err = state.total_income - ref.total_income
        gs = state.situations.get("GeneralStrike", 0.0)
        gs_active = "GeneralStrike" in state.active_situations
        print(
            f"{n:>4} | {state.total_income:>12,.1f} {state.total_expenditure:>12,.1f} | "
            f"{state.values.get('GDP',0):>7.3f} {gs:>5.3f} {str(gs_active):>5} | "
            f"{income_err:>+10,.1f}"
        )

        if verbose:
            for name in DEFAULT_METRICS:
                print(f"    {name}: sim={state.values.get(name, 0):.3f} game={ref.simvalues.get(name, 0):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    main(args.verbose)
