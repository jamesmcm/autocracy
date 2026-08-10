"""Generate a long native Democracy 3 capture for term-length parity work.

The launcher keeps raw XML in the configured native save directory.  It starts
one fresh game process, applies either no orders or a captured order sequence,
and runs the synchronous native worker for every requested turn.  The default
UK target is its 16-turn electoral term plus eight additional turns.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import sys
import time

from gamedrive import inject_drive
from gamedrive.order_plan import build_capture_specs, turn_number
from gamedrive.savecheck import validate_native_save


DEFAULT_COUNTRY = "uk"
DEFAULT_EXTRA_TURNS = 8
_TERM_LENGTH = re.compile(r"^\s*term_length\s*=\s*(\d+)\s*$")


def mission_term_length(
    country: str,
    *,
    gamedata_root: str | Path = "gamedata",
) -> int:
    """Read the mission's configured electoral term length in turns."""
    mission = Path(gamedata_root) / "data" / "missions" / country / f"{country}.txt"
    if not mission.is_file():
        raise FileNotFoundError(mission)
    for line in mission.read_text(encoding="utf-8").splitlines():
        match = _TERM_LENGTH.match(line)
        if match is not None:
            return int(match.group(1))
    raise ValueError(f"mission has no term_length: {mission}")


def capture_turns(
    country: str,
    *,
    turns: int | None,
    extra_turns: int,
    gamedata_root: str | Path = "gamedata",
) -> int:
    """Resolve an explicit turn count or term length plus an extension."""
    if turns is not None:
        if turns < 1:
            raise ValueError("turn count must be positive")
        return turns
    if extra_turns < 0:
        raise ValueError("extra turns cannot be negative")
    return mission_term_length(country, gamedata_root=gamedata_root) + extra_turns


def build_term_specs(
    initial_file: str | Path,
    orders_dir: str | Path | None,
    *,
    turns: int,
) -> list[str]:
    """Build a term-length order vector, padding its tail with no-ops."""
    if turns < 1:
        raise ValueError("turn count must be positive")
    if orders_dir is None:
        return [""] * turns
    order_root = Path(orders_dir)
    order_files = sorted(order_root.glob("turn*_o*.xml"), key=turn_number)
    if not order_files:
        raise ValueError(f"orders directory contains no captured orders: {order_root}")
    specs = build_capture_specs(initial_file, order_files)
    if len(specs) > turns:
        raise ValueError(
            f"orders reach turn {len(specs) - 1}, beyond requested capture length {turns}"
        )
    return specs + [""] * (turns - len(specs))


def _fresh_name(prefix: str) -> str:
    return f"{prefix}_{os.getpid()}_{time.time_ns() % 1_000_000_000:09d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--country", default=DEFAULT_COUNTRY)
    parser.add_argument("--gamedata-root", type=Path, default=Path("gamedata"))
    parser.add_argument("--turns", type=int)
    parser.add_argument("--extra-turns", type=int, default=DEFAULT_EXTRA_TURNS)
    parser.add_argument("--load-name", default="turn0_initial")
    parser.add_argument("--initial-file", type=Path)
    parser.add_argument("--orders-dir", type=Path)
    parser.add_argument("--capture-prefix")
    parser.add_argument("--loaded-name")
    parser.add_argument(
        "--one-process-per-turn",
        action="store_true",
        help="chain reliable one-turn native workers instead of one long worker",
    )
    parser.add_argument("--save-root", type=Path, default=inject_drive.SAVE_ROOT)
    parser.add_argument("--game", type=Path, default=inject_drive.GAME)
    parser.add_argument("--probe", type=Path, default=inject_drive.PROBE)
    parser.add_argument("--timeout", type=int, default=180)
    args = parser.parse_args()

    try:
        total_turns = capture_turns(
            args.country,
            turns=args.turns,
            extra_turns=args.extra_turns,
            gamedata_root=args.gamedata_root,
        )
        initial_file = args.initial_file
        if initial_file is None:
            initial_file = args.save_root / f"{args.load_name}.xml"
        if not initial_file.is_file() and args.orders_dir is not None:
            initial_file = args.orders_dir / "turn0_initial.xml"
        validate_native_save(initial_file)
        specs = build_term_specs(
            initial_file,
            args.orders_dir,
            turns=total_turns,
        )
    except (FileNotFoundError, OSError, ValueError) as exc:
        parser.error(str(exc))

    prefix = args.capture_prefix or _fresh_name(
        f"autocracy_{args.country}_term_capture"
    )
    loaded_name = args.loaded_name or _fresh_name(f"{prefix}_loaded")
    after_turn_name = _fresh_name(f"{prefix}_after")
    edited_name = _fresh_name(f"{prefix}_edited")
    print(f"country={args.country}")
    print(f"term_capture_turns={total_turns}")
    print(f"orders={'captured-plus-noop-tail' if args.orders_dir else 'none'}")
    print(f"source={args.load_name}")
    print(f"initial_file={initial_file}")
    print(f"capture_prefix={prefix}")
    print(f"save_root={args.save_root}")

    try:
        if args.one_process_per_turn:
            current_load_name = args.load_name
            for current_turn, spec in enumerate(specs, start=1):
                segment_prefix = f"{prefix}_step{current_turn}"
                loaded_name = _fresh_name(f"{segment_prefix}_loaded")
                after_turn_name = _fresh_name(f"{segment_prefix}_after")
                edited_name = _fresh_name(f"{segment_prefix}_edited")
                result = inject_drive.run(
                    load_name=current_load_name,
                    loaded_name=loaded_name,
                    after_turn_name=after_turn_name,
                    edited_name=edited_name,
                    edit_node=None,
                    edit_value=None,
                    turn_mode="sync",
                    skip_turn=False,
                    timeout=args.timeout,
                    capture_specs=[spec],
                    capture_prefix=segment_prefix,
                    game=args.game,
                    probe=args.probe,
                    save_root=args.save_root,
                )
                if result != 0:
                    return result
                current_load_name = f"{segment_prefix}_turn1"
            return 0
        return inject_drive.run(
            load_name=args.load_name,
            loaded_name=loaded_name,
            after_turn_name=after_turn_name,
            edited_name=edited_name,
            edit_node=None,
            edit_value=None,
            turn_mode="sync",
            skip_turn=False,
            timeout=args.timeout,
            capture_specs=specs,
            capture_prefix=prefix,
            game=args.game,
            probe=args.probe,
            save_root=args.save_root,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"native term capture not started: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
