"""Translate Democracy 3 ``*_orders`` saves into native probe actions.

An orders save is a pre-turn state after the player has moved policy sliders;
it is not a completed turn.  This module keeps that distinction explicit and
produces the small, delimiter-safe protocol consumed by ``native_probe.cpp``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Literal, Sequence

from autocracy.savegame import SaveGame, parse_savegame


OrderAction = Literal["slider", "implement", "cancel"]
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_]+$")
_EPSILON = 1e-6


@dataclass(frozen=True, slots=True)
class NativeOrder:
    """One native policy operation.

    ``slider`` calls ``SIM_Policy::SetSlider``.  ``implement`` first calls
    ``SIM_Policy::Implement`` and then sets the requested target, because the
    native implementation entrypoint starts a new policy at its default
    implementation level.  ``cancel`` leaves the native cancellation logic in
    charge of its own state and political-capital cost.
    """

    action: OrderAction
    policy_name: str
    target: float = 0.0

    def encode(self) -> str:
        """Return ``action|policy|target`` for the native probe."""
        if not _SAFE_NAME.fullmatch(self.policy_name):
            raise ValueError(
                f"policy name contains protocol delimiters: {self.policy_name!r}"
            )
        if not math.isfinite(self.target) or not 0.0 <= self.target <= 1.0:
            raise ValueError(f"policy target must be finite in [0, 1]: {self.target!r}")
        return f"{self.action}|{self.policy_name}|{self.target:.9g}"


def _active(save: SaveGame, name: str, target: float) -> bool:
    """Read a serialized active flag with the game's useful fallback."""
    return save.policy_active.get(name, target > _EPSILON)


def plan_orders(before: SaveGame, after: SaveGame) -> list[NativeOrder]:
    """Build native actions that transform *before* into *after* orders.

    The target field is authoritative for slider moves.  A change in the
    serialized active flag is authoritative for cancellation, which is how a
    floor move (target zero but still active) remains distinct from a true
    policy cancellation.
    """
    actions: list[NativeOrder] = []
    names = list(after.policy_desired_throttles)
    missing = set(before.policy_desired_throttles) - set(after.policy_desired_throttles)
    if missing:
        raise ValueError(
            "orders save is missing policies from its source: "
            + ", ".join(sorted(missing))
        )

    for name in names:
        target = after.policy_desired_throttles[name]
        current = before.policy_desired_throttles.get(
            name, before.policies.get(name, 0.0)
        )
        was_active = _active(before, name, current)
        is_active = _active(after, name, target)

        if was_active and not is_active:
            actions.append(NativeOrder("cancel", name))
            continue
        if abs(current - target) <= _EPSILON:
            # An inactive policy with a non-zero target is malformed for the
            # game data, but an unchanged record is still a no-op here.
            continue
        if not was_active and is_active:
            actions.append(NativeOrder("implement", name, target))
            continue
        actions.append(NativeOrder("slider", name, target))
    return actions


def plan_order_files(before_path: str | Path, after_path: str | Path) -> list[NativeOrder]:
    """Parse two saves and build their native order transition."""
    return plan_orders(parse_savegame(before_path), parse_savegame(after_path))


def encode_orders(actions: Sequence[NativeOrder]) -> str:
    """Encode one turn's actions, returning an empty string for no orders."""
    return ";".join(action.encode() for action in actions)


def turn_number(path: Path) -> int:
    """Extract the turn number from an ``*_orders`` filename."""
    match = re.search(r"turn(\d+)_", path.name)
    if match is None:
        raise ValueError(f"cannot find turn number in {path}")
    return int(match.group(1))


def infer_initial_path(orders_path: Path) -> Path:
    """Infer a matching initial save, including the captured ``ordes`` typo."""
    for marker in ("_orders", "_ordes"):
        if marker in orders_path.stem:
            return orders_path.with_name(
                orders_path.name.replace(marker, "_initial")
            )
    raise ValueError(f"cannot infer initial save from {orders_path}")


def build_capture_specs(
    initial_path: str | Path,
    orders_paths: Sequence[str | Path],
) -> list[str]:
    """Build one encoded order spec per captured turn.

    Captures occasionally omit an ``*_orders`` file for a no-order turn.  An
    empty spec preserves that turn in the native bounded loop.  When an
    ``*_initial`` save is unavailable, the previous orders save is sufficient
    for policy target/active comparisons because no-order turns do not change
    either field.
    """
    initial = Path(initial_path)
    order_paths = sorted((Path(path) for path in orders_paths), key=turn_number)
    if not order_paths:
        raise ValueError("at least one orders save is required")
    by_turn = {turn_number(path): path for path in order_paths}
    max_turn = max(by_turn)
    specs: list[str] = []
    previous_policy_state = initial

    for number in range(max_turn + 1):
        orders_path = by_turn.get(number)
        if orders_path is None:
            specs.append("")
            continue
        source_path = infer_initial_path(orders_path)
        if not source_path.is_file() or parse_savegame(source_path).turn != number:
            source_path = previous_policy_state
        specs.append(encode_orders(plan_order_files(source_path, orders_path)))
        previous_policy_state = orders_path
    return specs
