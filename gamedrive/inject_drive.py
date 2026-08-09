"""Run the version-pinned native Democracy 3 probe under gdb and Xvfb.

The probe loads a save through the game's own asynchronous LoadGame path,
serializes the loaded state, optionally edits one live neuron, advances one
native turn, and serializes again.  The game process is terminated by gdb after
the probe; it never writes back to the source capture unless the caller picks
an existing save name explicitly.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


GAME = Path(
    "/home/gopostal/.local/share/Steam/steamapps/common/Democracy 3/"
    "Democracy3.bin.x86_64"
)
PROBE = Path(__file__).with_name("libd3probe.so")
HARNESS = Path(__file__).with_name("harness_inject.gdb")
SAVE_ROOT = Path("/home/gopostal/.local/share/democracy3/savegames")


def _save_path(name: str) -> Path:
    return SAVE_ROOT / f"{name}.xml"


def run(
    *,
    load_name: str,
    loaded_name: str,
    after_turn_name: str,
    edited_name: str,
    edit_node: str | None,
    edit_value: float | None,
    gameplay_turn: bool,
    sync_gameplay_turn: bool,
    skip_turn: bool,
    timeout: int,
) -> int:
    if not GAME.is_file():
        raise FileNotFoundError(GAME)
    if not PROBE.is_file():
        raise FileNotFoundError(
            f"{PROBE}; build it first with `make -C {PROBE.parent}`"
        )

    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.pop("XAUTHORITY", None)
    encoded_edited_name = edited_name
    if edit_node is not None and edit_value is not None:
        encoded_edited_name = f"{edited_name}::{edit_node}::{edit_value:.9g}"
    gdb_args = [
        "gdb",
        "-batch",
        "-ex",
        f"set environment LD_PRELOAD {PROBE}",
        "-ex",
        f"set environment D3_LOAD_NAME {load_name}",
        "-ex",
        f"set environment D3_SAVE_LOADED {loaded_name}",
        "-ex",
        f"set environment D3_SAVE_AFTER_TURN {after_turn_name}",
        "-ex",
        f"set environment D3_SAVE_EDITED {encoded_edited_name}",
    ]
    turn_mode = ""
    if skip_turn:
        turn_mode = "1"
    elif gameplay_turn:
        turn_mode = "gameplay"
    elif sync_gameplay_turn:
        turn_mode = "sync"
    if turn_mode:
        gdb_args.extend(["-ex", f"set environment D3_SKIP_TURN {turn_mode}"])
    gdb_args.extend(["-x", str(HARNESS), str(GAME)])

    result = subprocess.run(
        [
            "xvfb-run",
            "-a",
            "-s",
            "-screen 0 1920x1080x24",
            "timeout",
            str(timeout),
            *gdb_args,
        ],
        capture_output=True,
        text=True,
        env=environment,
        timeout=timeout + 15,
        check=False,
    )
    output = result.stdout + result.stderr

    print(output)
    print("== native probe output files ==")
    for name in (loaded_name, edited_name, after_turn_name):
        path = _save_path(name)
        print(f"{path}: {'present' if path.is_file() else 'missing'}")
    required = [loaded_name]
    if edit_node is not None:
        required.append(edited_name)
    if not skip_turn:
        required.append(after_turn_name)
    missing = [name for name in required if not _save_path(name).is_file()]
    if result.returncode == 124:
        print(f"probe timed out after {timeout}s", file=sys.stderr)
        return 124
    if missing:
        print(
            f"probe did not produce required outputs: {', '.join(missing)}",
            file=sys.stderr,
        )
        return 2
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-name", default="d3_probe_turn0")
    parser.add_argument("--loaded-name", default="d3_probe_loaded")
    parser.add_argument("--after-turn-name", default="d3_probe_after_turn")
    parser.add_argument("--edited-name", default="d3_probe_edited")
    parser.add_argument("--edit-node")
    parser.add_argument("--edit-value", type=float)
    parser.add_argument(
        "--gameplay-turn",
        action="store_true",
        help="use SIM_Gameplay::NextTurn and wait for its native worker",
    )
    parser.add_argument(
        "--sync-gameplay-turn",
        action="store_true",
        help="run the game's NextTurnThread entrypoint synchronously",
    )
    parser.add_argument("--skip-turn", action="store_true")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    if (args.edit_node is None) != (args.edit_value is None):
        parser.error("--edit-node and --edit-value must be supplied together")
    if args.edit_node is not None and not args.skip_turn:
        parser.error(
            "memory edits require --skip-turn; load the edited output in a "
            "fresh probe for a subsequent native turn"
        )
    if sum((args.skip_turn, args.gameplay_turn, args.sync_gameplay_turn)) > 1:
        parser.error(
            "choose only one of --skip-turn, --gameplay-turn, or "
            "--sync-gameplay-turn"
        )

    return run(
        load_name=args.load_name,
        loaded_name=args.loaded_name,
        after_turn_name=args.after_turn_name,
        edited_name=args.edited_name,
        edit_node=args.edit_node,
        edit_value=args.edit_value,
        gameplay_turn=args.gameplay_turn,
        sync_gameplay_turn=args.sync_gameplay_turn,
        skip_turn=args.skip_turn,
        timeout=args.timeout,
    )


if __name__ == "__main__":
    raise SystemExit(main())
