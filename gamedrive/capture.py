"""Plan and compare bounded native captures offline.

The native process is intentionally kept separate from this module.  The
driver can produce ``<prefix>_turnN.xml`` files; this module replays the same
orders through the Python simulator and compares each completed native save.
It is therefore useful on hosts where the installed game cannot be launched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from autocracy import simulator
from autocracy.models import PolicyAction, SimulationConfig, SimulationData, SimulationState
from autocracy.savegame import SaveGame, load_state_from_savegame, parse_savegame

from gamedrive.order_plan import infer_initial_path, plan_orders, turn_number


@dataclass(frozen=True, slots=True)
class CaptureComparison:
    """Per-turn fields that are stable across XML and simulator snapshots."""

    native_path: Path
    native_turn: int
    simulator_turn: int
    income_delta: float
    expenditure_delta: float
    max_node_delta: float
    max_node_name: str
    policy_differences: int


def _order_paths(paths: Iterable[str | Path]) -> list[Path]:
    return sorted((Path(path) for path in paths), key=turn_number)


def _source_for_orders(path: Path, previous: Path) -> Path:
    inferred = infer_initial_path(path)
    if inferred.is_file() and parse_savegame(inferred).turn == turn_number(path):
        return inferred
    return previous


def _simulator_actions(
    before: SaveGame,
    after: SaveGame,
    state: SimulationState,
) -> list[PolicyAction]:
    actions: list[PolicyAction] = []
    for native_order in plan_orders(before, after):
        current = state.policy_desired_throttles.get(
            native_order.policy_name,
            state.policies.get(native_order.policy_name, 0.0),
        )
        if native_order.action == "cancel":
            actions.append(
                PolicyAction(
                    policy_name=native_order.policy_name,
                    delta=0.0,
                    action_type="cancel",
                )
            )
            continue
        action_type = (
            "introduce"
            if native_order.action == "implement"
            else ("raise" if native_order.target > current else "lower")
        )
        actions.append(
            PolicyAction(
                policy_name=native_order.policy_name,
                delta=native_order.target - current,
                action_type=action_type,
            )
        )
    return actions


def replay_simulator(
    initial_path: str | Path,
    orders_paths: Sequence[str | Path],
    *,
    data: SimulationData | None = None,
    config: SimulationConfig | None = None,
) -> dict[int, SimulationState]:
    """Replay every captured turn, including omitted no-order turns."""
    order_paths = _order_paths(orders_paths)
    if not order_paths:
        raise ValueError("at least one orders save is required")
    simulation_data = data or simulator.load_simulation_data()
    state, graph = load_state_from_savegame(initial_path, simulation_data)
    replay_config = config or SimulationConfig(minister_loyalty=True)
    by_turn = {turn_number(path): path for path in order_paths}
    previous_orders = Path(initial_path)
    snapshots: dict[int, SimulationState] = {}

    for number in range(max(by_turn) + 1):
        orders_path = by_turn.get(number)
        if orders_path is not None:
            source_path = _source_for_orders(orders_path, previous_orders)
            before = parse_savegame(source_path)
            after = parse_savegame(orders_path)
            for action in _simulator_actions(before, after, state):
                state = simulator.apply_actions(
                    state, [action], data=simulation_data
                )
            previous_orders = orders_path
        state = simulator.process_end_of_turn(
            state, graph, simulation_data, config=replay_config
        )
        snapshots[state.turn] = state
    return snapshots


def compare_native_saves(
    native_paths: Sequence[str | Path],
    simulator_snapshots: dict[int, SimulationState],
    *,
    policy_tolerance: float = 1e-5,
) -> list[CaptureComparison]:
    """Compare completed native XML saves against simulator turn snapshots."""
    comparisons: list[CaptureComparison] = []
    for raw_path in native_paths:
        path = Path(raw_path)
        native = parse_savegame(path)
        state = simulator_snapshots.get(native.turn)
        if state is None:
            raise ValueError(
                f"native turn {native.turn} has no simulator snapshot: {path}"
            )
        node_name = ""
        max_node_delta = 0.0
        for name, expected in native.simvalues.items():
            actual = state.values.get(name)
            if actual is None:
                continue
            delta = abs(actual - expected)
            if delta > max_node_delta:
                max_node_delta = delta
                node_name = name
        policy_differences = sum(
            abs(state.policies.get(name, 0.0) - value) > policy_tolerance
            for name, value in native.policies.items()
        )
        comparisons.append(
            CaptureComparison(
                native_path=path,
                native_turn=native.turn,
                simulator_turn=state.turn,
                income_delta=state.total_income - native.total_income,
                expenditure_delta=state.total_expenditure - native.total_expenditure,
                max_node_delta=max_node_delta,
                max_node_name=node_name,
                policy_differences=policy_differences,
            )
        )
    return comparisons


def capture_paths(root: str | Path, prefix: str, turns: int) -> list[Path]:
    """Return the native output names produced by the bounded probe."""
    if turns < 1:
        raise ValueError("turn count must be positive")
    return [
        Path(root) / f"{prefix}_turn{turn}.xml"
        for turn in range(1, turns + 1)
    ]


def _print_comparisons(comparisons: Sequence[CaptureComparison]) -> None:
    print(
        f"{'turn':>4} | {'income err':>12} {'exp err':>12} | "
        f"{'max node err':>12} {'node':<24} | {'policy diffs':>12}"
    )
    print("-" * 88)
    for comparison in comparisons:
        print(
            f"{comparison.native_turn:>4} | "
            f"{comparison.income_delta:>+12,.1f} "
            f"{comparison.expenditure_delta:>+12,.1f} | "
            f"{comparison.max_node_delta:>12.6f} "
            f"{comparison.max_node_name:<24} | "
            f"{comparison.policy_differences:>12}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-file", type=Path, required=True)
    parser.add_argument("--orders-dir", type=Path, required=True)
    parser.add_argument("--native-file", action="append", type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--native-prefix")
    parser.add_argument("--turns", type=int)
    args = parser.parse_args()

    order_files = sorted(args.orders_dir.glob("turn*_o*.xml"), key=turn_number)
    snapshots = replay_simulator(args.initial_file, order_files)
    native_paths = list(args.native_file or ())
    if args.native_dir is not None and args.native_prefix is not None:
        if args.turns is None:
            parser.error("--turns is required with --native-dir/--native-prefix")
        native_paths.extend(
            capture_paths(args.native_dir, args.native_prefix, args.turns)
        )
    if not native_paths:
        parser.error("provide --native-file or --native-dir with --native-prefix")
    _print_comparisons(compare_native_saves(native_paths, snapshots))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
