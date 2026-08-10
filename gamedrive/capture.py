"""Plan and compare bounded native captures offline.

The native process is intentionally kept separate from this module.  The
driver can produce ``<prefix>_turnN.xml`` files; this module replays the same
orders through the Python simulator and compares each completed native save.
It is therefore useful on hosts where the installed game cannot be launched.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from autocracy import simulator
from autocracy.models import (
    EffectHistory,
    PolicyAction,
    SimulationConfig,
    SimulationData,
    SimulationState,
)
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
    turns: int | None = None,
    data: SimulationData | None = None,
    config: SimulationConfig | None = None,
    native_order_runtime: bool = True,
    native_manager_rosters: Mapping[int, Iterable[str]] | None = None,
    native_active_situations: Mapping[int, Iterable[str]] | None = None,
    native_voter_states: Mapping[int, SaveGame] | None = None,
    native_effect_histories: Mapping[int, Iterable[EffectHistory]] | None = None,
) -> dict[int, SimulationState]:
    """Replay captured orders, optionally continuing through no-order turns.

    Native captures use ``SIM_Policy::SetSlider`` before ``NextTurn``. The
    default therefore applies native current-value/throttle semantics while
    retaining the simulator's delayed runtime for direct interactive calls.
    Set ``native_order_runtime=False`` for the older abstract action model.

    The optional native roster, situation, voter, and effect-history schedules
    are explicit serialized-checkpoint inputs for parity audits. They model
    native manager state that is not reconstructible from the deterministic
    simulator alone and are never used by ordinary callers.
    """
    order_paths = _order_paths(orders_paths)
    if turns is not None and turns < 1:
        raise ValueError("turn count must be positive")
    if not order_paths and turns is None:
        raise ValueError("turns is required when no orders saves are supplied")
    simulation_data = data or simulator.load_simulation_data()
    state, graph = load_state_from_savegame(initial_path, simulation_data)
    replay_config = replace(
        config or SimulationConfig(minister_loyalty=True),
        native_order_runtime=native_order_runtime,
    )
    roster_by_turn = (
        {int(turn): {str(department) for department in departments}
         for turn, departments in native_manager_rosters.items()}
        if native_manager_rosters is not None
        else {}
    )
    situations_by_turn = (
        {int(turn): [str(name) for name in situations]
         for turn, situations in native_active_situations.items()}
        if native_active_situations is not None
        else {}
    )
    if roster_by_turn:
        # A supplied native roster is authoritative for this audit.  It
        # replaces the unsaved native resignation RNG, rather than allowing
        # the deterministic threshold mode to remove additional departments.
        replay_config = replace(replay_config, minister_resignations=False)
    by_turn = {turn_number(path): path for path in order_paths}
    capture_turns = turns if turns is not None else max(by_turn) + 1
    if by_turn and capture_turns <= max(by_turn):
        raise ValueError("turn count ends before the last orders save")
    previous_orders = Path(initial_path)
    snapshots: dict[int, SimulationState] = {}

    for number in range(capture_turns):
        orders_path = by_turn.get(number)
        if orders_path is not None:
            source_path = _source_for_orders(orders_path, previous_orders)
            before = parse_savegame(source_path)
            after = parse_savegame(orders_path)
            actions = _simulator_actions(before, after, state)
            if native_order_runtime:
                # The injector applies the complete order save before calling
                # NextTurn.  Keep one finance preview for the whole batch so
                # slider moves do not successively become inputs to the next
                # order's preview.
                if actions:
                    state = simulator.apply_actions(
                        state,
                        actions,
                        data=simulation_data,
                        native_order_runtime=True,
                    )
            else:
                for action in actions:
                    state = simulator.apply_actions(
                        state,
                        [action],
                        data=simulation_data,
                        native_order_runtime=False,
                    )
            previous_orders = orders_path
        native_departed: set[str] = set()
        target_roster = roster_by_turn.get(state.turn + 1)
        if target_roster is not None:
            native_departed = set(state.ministerial_loyalty) - target_roster
            state = simulator.apply_native_manager_roster(state, target_roster)
        target_situations = situations_by_turn.get(state.turn + 1)
        if target_situations is not None:
            state = replace(
                state,
                active_situations=list(target_situations),
            )
        state = simulator.process_end_of_turn(
            state,
            graph,
            simulation_data,
            config=replay_config,
            native_resigned_departments=native_departed,
            native_active_situations=target_situations,
        )
        target_histories = (
            native_effect_histories.get(state.turn)
            if native_effect_histories is not None
            else None
        )
        if target_histories is not None:
            state = simulator.apply_native_effect_histories(
                state,
                target_histories,
                graph=graph,
                data=simulation_data,
            )
        target_voters = (
            native_voter_states.get(state.turn)
            if native_voter_states is not None
            else None
        )
        if target_voters is not None:
            state = simulator.apply_native_voter_runtime(
                state,
                voter_values=target_voters.voter_values,
                voter_percentages=target_voters.voter_percentages,
                voter_frequencies=target_voters.voter_frequencies,
                voter_incomes=target_voters.voter_incomes,
                voter_frequency_grudges=target_voters.voter_frequency_grudges,
                voters=target_voters.voters,
                parties=target_voters.parties,
                poll_rate=target_voters.poll_rate or 0.0,
                peak_poll_rate=target_voters.peak_poll_rate or 0.0,
                poll_history=target_voters.poll_history,
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
    orders_group = parser.add_mutually_exclusive_group()
    orders_group.add_argument("--orders-dir", type=Path)
    orders_group.add_argument(
        "--no-orders",
        action="store_true",
        help="replay only no-order turns; pair with --turns",
    )
    parser.add_argument("--native-file", action="append", type=Path)
    parser.add_argument("--native-dir", type=Path)
    parser.add_argument("--native-prefix")
    parser.add_argument("--turns", type=int)
    args = parser.parse_args()

    order_files = (
        []
        if args.no_orders or args.orders_dir is None
        else sorted(args.orders_dir.glob("turn*_o*.xml"), key=turn_number)
    )
    if args.no_orders and args.turns is None:
        parser.error("--turns is required with --no-orders")
    snapshots = replay_simulator(
        args.initial_file,
        order_files,
        turns=args.turns,
    )
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
