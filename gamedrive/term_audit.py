"""Audit long native chains against simulator snapshots.

The native XML is the source of truth for serialized state.  This report keeps
that state separate from the live party/poll manager census: those pointers are
not saved and therefore cannot be reconstructed by the offline simulator.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import numbers
from pathlib import Path
import re
from typing import Iterable, Mapping, Sequence

from autocracy import simulator
from autocracy.models import SimulationConfig, SimulationData, SimulationState
from autocracy.savegame import SaveGame, parse_savegame

from gamedrive.capture import replay_simulator
from gamedrive.order_plan import turn_number
from gamedrive.savecheck import validate_native_save


@dataclass(frozen=True, slots=True)
class NativeManagerSummary:
    """Manager-owned fields visible in a serialized save."""

    poll_rate: float | None
    peak_poll_rate: float | None
    turns_until_election: int | None
    current_term: int | None
    active_minister_departments: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class TermAuditRow:
    """Comparable and intentionally non-comparable fields for one turn."""

    turn: int
    income_delta: float
    expenditure_delta: float
    ordinary_max_delta: float
    ordinary_max_name: str
    situation_max_delta: float
    situation_max_name: str
    hidden_max_delta: float
    hidden_max_name: str
    hidden_history_max_delta: float
    voter_value_max_delta: float
    voter_value_max_name: str
    voter_percentage_max_delta: float
    voter_percentage_max_name: str
    voter_frequency_max_delta: float
    voter_frequency_max_name: str
    voter_income_max_delta: float
    voter_income_max_name: str
    individual_voter_value_max_delta: float
    policy_current_differences: int
    policy_target_differences: int
    policy_runtime_differences: int
    active_situation_missing: int
    active_situation_extra: int
    party_differences: int
    effect_history_max_delta: float
    active_minister_missing: int
    active_minister_extra: int
    native_poll_rate: float | None
    native_turns_until_election: int | None
    native_current_term: int | None


def chain_capture_paths(
    root: str | Path, prefix: str, turns: int
) -> list[Path]:
    """Return per-turn paths produced by ``term_capture --one-process-per-turn``."""
    if turns < 1:
        raise ValueError("turn count must be positive")
    return [
        Path(root) / f"{prefix}_step{turn}_turn1.xml"
        for turn in range(1, turns + 1)
    ]


def _section(raw: str, name: str) -> str:
    match = re.search(fr"<{name}>(.*?)</{name}>", raw, re.DOTALL)
    return match.group(1) if match else ""


def _number(body: str, name: str) -> float | None:
    match = re.search(fr"<{name}>([^<]*)</{name}>", body)
    if match is None:
        return None
    try:
        return float(match.group(1).strip())
    except ValueError:
        return None


def _integer(body: str, name: str) -> int | None:
    value = _number(body, name)
    return int(value) if value is not None else None


def native_manager_summary(path: str | Path) -> NativeManagerSummary:
    """Extract serialized poll/election/minister-manager observations."""
    raw = Path(path).read_text(encoding="latin-1")
    polls = _section(raw, "polls")
    election = _section(raw, "election")
    ministers = _section(raw, "ministers")
    jobs = sorted(
        {
            match.group(1).strip()
            for match in re.finditer(r"<job>([^<]+)</job>", ministers)
            if match.group(1).strip().lower() != "none"
        }
    )
    return NativeManagerSummary(
        poll_rate=_number(polls, "potentialvoterate"),
        peak_poll_rate=_number(polls, "peakvoterate"),
        turns_until_election=_integer(election, "turnsuntilelection"),
        current_term=_integer(election, "currentterm"),
        active_minister_departments=tuple(jobs),
    )


def _max_mapping_delta(
    actual: Mapping[str, float],
    expected: Mapping[str, float],
) -> tuple[float, str]:
    maximum = 0.0
    name = ""
    for key, expected_value in expected.items():
        if key not in actual:
            continue
        delta = abs(float(actual[key]) - float(expected_value))
        if delta > maximum:
            maximum = delta
            name = key
    return maximum, name


def _count_mapping_differences(
    actual: Mapping[str, object],
    expected: Mapping[str, object],
    *,
    tolerance: float = 1e-5,
) -> int:
    differences = 0
    for name, expected_value in expected.items():
        actual_value = actual.get(name)
        if isinstance(actual_value, numbers.Real) and isinstance(
            expected_value, numbers.Real
        ):
            different = abs(float(actual_value) - float(expected_value)) > tolerance
        else:
            different = actual_value != expected_value
        differences += different
    return differences


def _history_max_delta(
    actual: Mapping[str, Sequence[float]],
    expected: Mapping[str, Sequence[float]],
) -> float:
    maximum = 0.0
    for name, expected_values in expected.items():
        actual_values = actual.get(name, ())
        for actual_value, expected_value in zip(actual_values, expected_values):
            maximum = max(maximum, abs(actual_value - expected_value))
        if len(actual_values) != len(expected_values):
            maximum = max(maximum, 1.0)
    return maximum


def _effect_history_max_delta(
    actual: Iterable[object], expected: Iterable[object]
) -> float:
    def key(history: object) -> tuple[str, str, str]:
        return (
            getattr(history, "effect_id", "") or "",
            getattr(history, "source", ""),
            getattr(history, "target", ""),
        )

    actual_by_key = {key(history): history for history in actual}
    maximum = 0.0
    for expected_history in expected:
        actual_history = actual_by_key.get(key(expected_history))
        if actual_history is None:
            maximum = max(maximum, 1.0)
            continue
        actual_values = getattr(actual_history, "values", ())
        expected_values = getattr(expected_history, "values", ())
        for actual_value, expected_value in zip(actual_values, expected_values):
            maximum = max(maximum, abs(actual_value - expected_value))
        if len(actual_values) != len(expected_values):
            maximum = max(maximum, 1.0)
    return maximum


def _party_differences(state: SimulationState, save: SaveGame) -> int:
    differences = 0
    for name, expected in save.parties.items():
        actual = state.parties.get(name)
        if actual is None:
            differences += 1
            continue
        for field in (
            "status",
            "party_type",
            "members_last_turn",
            "member_history",
            "activist_history",
        ):
            if getattr(actual, field) != getattr(expected, field):
                differences += 1
                break
    return differences


def audit_turn(
    native_path: str | Path,
    state: SimulationState,
    data: SimulationData,
) -> TermAuditRow:
    """Compare one native save to its simulator snapshot."""
    path = Path(native_path)
    save = parse_savegame(path)
    manager = native_manager_summary(path)
    ordinary_delta, ordinary_name = _max_mapping_delta(state.values, save.simvalues)
    situation_delta, situation_name = _max_mapping_delta(
        state.situations, save.situations
    )
    hidden_actual = {
        name: state.values.get(name, 0.0)
        for name in save.hidden_values
    }
    hidden_delta, hidden_name = _max_mapping_delta(hidden_actual, save.hidden_values)
    voter_value_delta, voter_value_name = _max_mapping_delta(
        state.voter_values, save.voter_values
    )
    voter_percentage_delta, voter_percentage_name = _max_mapping_delta(
        state.voter_percentages, save.voter_percentages
    )
    voter_frequency_delta, voter_frequency_name = _max_mapping_delta(
        state.voter_frequencies, save.voter_frequencies
    )
    voter_income_delta, voter_income_name = _max_mapping_delta(
        state.voter_incomes, save.voter_incomes
    )
    individual_voter_value = max(
        (
            abs(actual.value - expected.value)
            for actual, expected in zip(state.voters, save.voters)
        ),
        default=0.0,
    )
    active_expected = set(save.active_situations)
    active_actual = set(state.active_situations)
    active_ministers = set(state.ministerial_loyalty)
    native_ministers = set(manager.active_minister_departments)
    return TermAuditRow(
        turn=save.turn,
        income_delta=state.total_income - save.total_income,
        expenditure_delta=state.total_expenditure - save.total_expenditure,
        ordinary_max_delta=ordinary_delta,
        ordinary_max_name=ordinary_name,
        situation_max_delta=situation_delta,
        situation_max_name=situation_name,
        hidden_max_delta=hidden_delta,
        hidden_max_name=hidden_name,
        hidden_history_max_delta=_history_max_delta(
            state.hidden_histories, save.hidden_histories
        ),
        voter_value_max_delta=voter_value_delta,
        voter_value_max_name=voter_value_name,
        voter_percentage_max_delta=voter_percentage_delta,
        voter_percentage_max_name=voter_percentage_name,
        voter_frequency_max_delta=voter_frequency_delta,
        voter_frequency_max_name=voter_frequency_name,
        voter_income_max_delta=voter_income_delta,
        voter_income_max_name=voter_income_name,
        individual_voter_value_max_delta=individual_voter_value,
        policy_current_differences=_count_mapping_differences(
            state.policies, save.policies
        ),
        policy_target_differences=_count_mapping_differences(
            state.policy_desired_throttles, save.policy_desired_throttles
        ),
        policy_runtime_differences=(
            _count_mapping_differences(
                state.policy_active, save.policy_active
            )
            + _count_mapping_differences(
                state.policy_income_scalars, save.policy_income_scalars
            )
            + _count_mapping_differences(
                state.policy_cost_scalars, save.policy_cost_scalars
            )
        ),
        active_situation_missing=len(active_expected - active_actual),
        active_situation_extra=len(active_actual - active_expected),
        party_differences=_party_differences(state, save),
        effect_history_max_delta=_effect_history_max_delta(
            state.effect_histories, save.effect_histories
        ),
        active_minister_missing=len(native_ministers - active_ministers),
        active_minister_extra=len(active_ministers - native_ministers),
        native_poll_rate=manager.poll_rate,
        native_turns_until_election=manager.turns_until_election,
        native_current_term=manager.current_term,
    )


def audit_chain(
    native_paths: Sequence[str | Path],
    snapshots: Mapping[int, SimulationState],
    data: SimulationData,
) -> list[TermAuditRow]:
    """Audit every native checkpoint in serialized-turn order."""
    rows: list[TermAuditRow] = []
    for path in native_paths:
        native_turn = parse_savegame(path).turn
        state = snapshots.get(native_turn)
        if state is None:
            raise ValueError(f"no simulator snapshot for native turn {native_turn}")
        rows.append(audit_turn(path, state, data))
    return rows


def _print_rows(rows: Sequence[TermAuditRow]) -> None:
    print(
        "turn | finance (income, expenditure) | ordinary | situations | "
        "voters (value, %, freq, income) | policy targets | parties | poll/election"
    )
    print("-" * 124)
    for row in rows:
        print(
            f"{row.turn:>4} | {row.income_delta:+8.1f}, {row.expenditure_delta:+8.1f} | "
            f"{row.ordinary_max_delta:.4f} | {row.situation_max_delta:.4f} | "
            f"{row.voter_value_max_delta:.4f}, {row.voter_percentage_max_delta:.4f}, "
            f"{row.voter_frequency_max_delta:.4f}, {row.voter_income_max_delta:.4f} | "
            f"{row.policy_target_differences} | {row.party_differences} | "
            f"{row.native_poll_rate!s}, {row.native_turns_until_election!s}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--initial-file", type=Path, required=True)
    parser.add_argument("--native-dir", type=Path, required=True)
    parser.add_argument("--native-prefix", required=True)
    parser.add_argument("--turns", type=int, required=True)
    parser.add_argument("--orders-dir", type=Path)
    parser.add_argument(
        "--minister-resignations",
        action="store_true",
        help="opt into deterministic below-threshold minister removal",
    )
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    if args.orders_dir is None:
        order_paths: list[Path] = []
    else:
        order_paths = sorted(args.orders_dir.glob("turn*_o*.xml"), key=turn_number)
    native_paths = chain_capture_paths(
        args.native_dir, args.native_prefix, args.turns
    )
    for path in native_paths:
        validate_native_save(path)
    data = simulator.load_simulation_data()
    snapshots = replay_simulator(
        args.initial_file,
        order_paths,
        turns=args.turns,
        data=data,
        config=SimulationConfig(
            minister_loyalty=True,
            minister_resignations=args.minister_resignations,
        ),
    )
    rows = audit_chain(native_paths, snapshots, data)
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2, sort_keys=True))
    else:
        _print_rows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
