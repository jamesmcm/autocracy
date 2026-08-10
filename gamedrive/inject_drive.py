"""Run the version-pinned native Democracy 3 probe under gdb and Xvfb.

The driver launches one fresh game process, loads a source save through the
game's asynchronous loader, and then invokes native policy/turn entrypoints
from a bounded main-loop harness.  Output names are fresh by default and are
validated as complete native XML before the command succeeds.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Sequence

from gamedrive.order_plan import build_capture_specs
from gamedrive.savecheck import NativeSaveError, validate_native_save


GAME = Path(
    "/home/gopostal/.local/share/Steam/steamapps/common/Democracy 3/"
    "Democracy3.bin.x86_64"
)
PROBE = Path(__file__).with_name("libd3probe.so")
HARNESS = Path(__file__).with_name("harness_inject.gdb")
SAVE_ROOT = Path("/home/gopostal/.local/share/democracy3/savegames")
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_.-]+$")


def _save_path(name: str, save_root: Path = SAVE_ROOT) -> Path:
    return save_root / f"{name}.xml"


def _validate_name(name: str, label: str) -> None:
    if not _SAFE_NAME.fullmatch(name):
        raise ValueError(
            f"{label} must contain only letters, digits, '.', '_' or '-': {name!r}"
        )


def _encoded_edit(edited_name: str, node: str | None, value: float | None) -> str:
    if node is None or value is None:
        return edited_name
    if "::" in node:
        raise ValueError("--edit-node cannot contain '::'")
    return f"{edited_name}::{node}::{value:.9g}"


def _set_environment(arguments: list[str], name: str, value: str) -> None:
    """Append a gdb environment command for a delimiter-safe value."""
    if any(character in value for character in "\n\r"):
        raise ValueError(f"environment value contains a newline: {name}")
    arguments.extend(["-ex", f"set environment {name} {value}"])


def _default_name(prefix: str) -> str:
    return f"{prefix}_{os.getpid()}_{time.time_ns() % 1_000_000_000:09d}"


def _output_paths(
    *,
    loaded_name: str,
    after_turn_name: str,
    edited_name: str,
    orders_save_name: str | None,
    manager_save_name: str | None,
    capture_prefix: str | None,
    capture_count: int,
    edit_node: str | None,
    skip_turn: bool,
    save_root: Path,
) -> list[Path]:
    paths = [
        _save_path(loaded_name, save_root),
    ]
    if edit_node is not None:
        paths.append(_save_path(edited_name, save_root))
    if orders_save_name is not None:
        paths.append(_save_path(orders_save_name, save_root))
    if capture_prefix is not None:
        paths.extend(
            _save_path(f"{capture_prefix}_turn{index}", save_root)
            for index in range(1, capture_count + 1)
        )
    elif not skip_turn:
        paths.append(_save_path(after_turn_name, save_root))
    if manager_save_name is not None:
        paths.append(_save_path(manager_save_name, save_root))
    return paths


def _validate_output_files(
    paths: Sequence[Path],
    *,
    before: dict[Path, tuple[int, int]],
    allow_existing: bool,
    started_ns: int,
) -> list[str]:
    errors: list[str] = []
    for path in paths:
        if not path.is_file():
            errors.append(f"missing {path}")
            continue
        try:
            stat = path.stat()
            prior = before.get(path)
            if prior is not None and not allow_existing:
                errors.append(f"output already existed before probe: {path}")
                continue
            if prior is not None and (stat.st_mtime_ns <= prior[0]):
                errors.append(f"output was not rewritten by probe: {path}")
                continue
            if prior is None and stat.st_mtime_ns < started_ns:
                errors.append(f"output predates probe: {path}")
                continue
            validate_native_save(path)
        except (OSError, NativeSaveError) as exc:
            errors.append(str(exc))
    return errors


def _validate_manager_audit(
    path: Path,
    *,
    before: tuple[int, int] | None,
    allow_existing: bool,
    started_ns: int,
) -> list[str]:
    """Validate the opt-in text census written beside native XML output."""
    if not path.is_file():
        return [f"missing manager audit: {path}"]
    try:
        stat = path.stat()
        if before is not None and not allow_existing:
            return [f"manager audit already existed before probe: {path}"]
        if before is not None and stat.st_mtime_ns <= before[0]:
            return [f"manager audit was not rewritten by probe: {path}"]
        if before is None and stat.st_mtime_ns < started_ns:
            return [f"manager audit predates probe: {path}"]
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        return [str(exc)]
    required = (
        "format=democracy3-v1.30.2-manager-audit",
        "voter_types=",
        "voters=",
        "parties=",
        "poll_rate=",
        "refresh=",
    )
    if any(marker not in text for marker in required):
        return [f"manager audit is incomplete: {path}"]
    return []


def run(
    *,
    load_name: str,
    loaded_name: str,
    after_turn_name: str,
    edited_name: str,
    edit_node: str | None,
    edit_value: float | None,
    turn_mode: str = "sync",
    gameplay_turn: bool = False,
    sync_gameplay_turn: bool = False,
    skip_turn: bool,
    timeout: int,
    order_spec: str | None = None,
    capture_specs: Sequence[str] | None = None,
    capture_prefix: str | None = None,
    orders_save_name: str | None = None,
    manager_audit_path: Path | None = None,
    manager_save_name: str | None = None,
    allow_existing: bool = False,
    game: Path = GAME,
    probe: Path = PROBE,
    save_root: Path = SAVE_ROOT,
) -> int:
    """Run one bounded probe and validate all outputs it requested."""
    if gameplay_turn:
        if turn_mode != "sync":
            raise ValueError("legacy --gameplay-turn conflicts with --turn-mode")
        turn_mode = "async"
    if sync_gameplay_turn:
        if turn_mode not in ("sync", "async"):
            raise ValueError("legacy --sync-gameplay-turn conflicts with --turn-mode")
        turn_mode = "sync"
    if turn_mode not in {"sync", "direct", "async"}:
        raise ValueError(f"unknown turn mode: {turn_mode}")
    if capture_specs and turn_mode != "sync":
        raise ValueError("bounded multi-turn capture requires --turn-mode sync")
    if capture_specs and order_spec is not None:
        raise ValueError("single order and capture order specs cannot be combined")
    if edit_node is not None and not skip_turn:
        raise ValueError("memory edits require --skip-turn")
    if not game.is_file():
        raise FileNotFoundError(game)
    if not probe.is_file():
        raise FileNotFoundError(
            f"{probe}; build it first with `make -C {probe.parent}`"
        )

    for name, label in (
        (load_name, "--load-name"),
        (loaded_name, "--loaded-name"),
        (after_turn_name, "--after-turn-name"),
        (edited_name, "--edited-name"),
        (orders_save_name, "--orders-save-name"),
        (manager_save_name, "--manager-save-name"),
        (capture_prefix, "--capture-prefix"),
    ):
        if name is not None:
            _validate_name(name, label)
    if any(name == load_name for name in (loaded_name, edited_name, orders_save_name, manager_save_name)):
        raise ValueError("probe output names must differ from --load-name")

    capture_count = len(capture_specs or ())
    output_paths = _output_paths(
        loaded_name=loaded_name,
        after_turn_name=after_turn_name,
        edited_name=edited_name,
        orders_save_name=orders_save_name,
        manager_save_name=manager_save_name,
        capture_prefix=capture_prefix,
        capture_count=capture_count,
        edit_node=edit_node,
        skip_turn=skip_turn,
        save_root=save_root,
    )
    if len(set(output_paths)) != len(output_paths):
        raise ValueError("probe output names must be unique")
    audit_path = manager_audit_path.resolve() if manager_audit_path is not None else None
    if audit_path is not None and audit_path in {
        path.resolve() for path in output_paths
    } | {_save_path(load_name, save_root).resolve()}:
        raise ValueError("manager audit path must differ from every native save")
    before = {
        path: (path.stat().st_mtime_ns, path.stat().st_size)
        for path in output_paths
        if path.exists()
    }
    audit_before = (
        (manager_audit_path.stat().st_mtime_ns, manager_audit_path.stat().st_size)
        if manager_audit_path is not None and manager_audit_path.exists()
        else None
    )
    if before and not allow_existing:
        names = ", ".join(str(path) for path in before)
        raise FileExistsError(
            f"refusing to reuse native output(s): {names}; choose fresh names "
            "or pass --allow-existing"
        )
    if audit_before is not None and not allow_existing:
        raise FileExistsError(
            f"refusing to reuse manager audit: {manager_audit_path}; choose a fresh "
            "path or pass --allow-existing"
        )

    environment = dict(os.environ)
    environment.pop("DISPLAY", None)
    environment.pop("XAUTHORITY", None)
    gdb_args = [
        "gdb",
        "-batch",
        "-ex",
        f"set environment LD_PRELOAD {probe}",
    ]
    _set_environment(gdb_args, "D3_LOAD_NAME", load_name)
    _set_environment(gdb_args, "D3_SAVE_LOADED", loaded_name)
    _set_environment(gdb_args, "D3_SAVE_AFTER_TURN", after_turn_name)
    _set_environment(
        gdb_args,
        "D3_SAVE_EDITED",
        _encoded_edit(edited_name, edit_node, edit_value),
    )
    if order_spec:
        _set_environment(gdb_args, "D3_ORDERS", order_spec)
    if capture_specs:
        _set_environment(gdb_args, "D3_TURN_COUNT", str(capture_count))
        _set_environment(gdb_args, "D3_CAPTURE_PREFIX", capture_prefix or "d3_probe_capture")
        for index, spec in enumerate(capture_specs):
            if spec:
                _set_environment(gdb_args, f"D3_ORDERS_{index}", spec)
    if orders_save_name is not None:
        _set_environment(gdb_args, "D3_SAVE_ORDERS", orders_save_name)
    if manager_save_name is not None:
        _set_environment(gdb_args, "D3_SAVE_MANAGERS", manager_save_name)
    if manager_audit_path is not None:
        _set_environment(gdb_args, "D3_MANAGER_AUDIT", "1")
        _set_environment(gdb_args, "D3_MANAGER_AUDIT_PATH", str(manager_audit_path))
    elif manager_save_name is not None:
        _set_environment(gdb_args, "D3_MANAGER_AUDIT", "1")

    if skip_turn:
        _set_environment(gdb_args, "D3_SKIP_TURN", "1")
    elif turn_mode == "sync":
        _set_environment(gdb_args, "D3_SKIP_TURN", "sync")
    elif turn_mode == "async":
        _set_environment(gdb_args, "D3_SKIP_TURN", "gameplay")

    gdb_args.extend(["-x", str(HARNESS), str(game)])
    started_ns = time.time_ns()
    try:
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
    except (OSError, subprocess.TimeoutExpired) as exc:
        print(f"unable to run native probe: {exc}", file=sys.stderr)
        return 2

    output = result.stdout + result.stderr
    print(output)
    print("== native probe output files ==")
    for path in output_paths:
        print(f"{path}: {'present' if path.is_file() else 'missing'}")
    if manager_audit_path is not None:
        print(
            f"{manager_audit_path}: "
            f"{'present' if manager_audit_path.is_file() else 'missing'}"
        )

    if result.returncode == 124:
        print(f"probe timed out after {timeout}s", file=sys.stderr)
        return 124
    errors = _validate_output_files(
        output_paths,
        before=before,
        allow_existing=allow_existing,
        started_ns=started_ns,
    )
    if manager_audit_path is not None:
        errors.extend(
            _validate_manager_audit(
                manager_audit_path,
                before=audit_before,
                allow_existing=allow_existing,
                started_ns=started_ns,
            )
        )
    if errors:
        print("probe output validation failed:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    return result.returncode


def _collect_order_files(args: argparse.Namespace) -> list[Path]:
    paths = list(args.orders_file or ())
    if args.orders_dir is not None:
        paths.extend(sorted(args.orders_dir.glob("turn*_o*.xml")))
    unique = sorted(set(paths), key=lambda path: path.name)
    if not unique:
        raise ValueError("--orders-file or --orders-dir supplied no orders saves")
    return unique


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--game", type=Path, default=GAME)
    parser.add_argument("--probe", type=Path, default=PROBE)
    parser.add_argument("--save-root", type=Path, default=SAVE_ROOT)
    parser.add_argument("--load-name", default="d3_probe_turn0")
    parser.add_argument("--loaded-name")
    parser.add_argument("--after-turn-name")
    parser.add_argument("--edited-name")
    parser.add_argument("--edit-node")
    parser.add_argument("--edit-value", type=float)
    parser.add_argument(
        "--turn-mode",
        choices=("sync", "direct", "async"),
        default="sync",
        help="sync uses NextTurnThread (reliable); async is ptrace-sensitive",
    )
    parser.add_argument(
        "--gameplay-turn",
        action="store_true",
        help="legacy alias for --turn-mode async",
    )
    parser.add_argument(
        "--sync-gameplay-turn",
        action="store_true",
        help="legacy alias for --turn-mode sync",
    )
    parser.add_argument("--skip-turn", action="store_true")
    parser.add_argument(
        "--orders-file",
        action="append",
        type=Path,
        help="captured *_orders.xml; repeat or use --orders-dir for a bounded replay",
    )
    parser.add_argument("--orders-dir", type=Path)
    parser.add_argument("--initial-file", type=Path)
    parser.add_argument("--capture-prefix")
    parser.add_argument("--orders-save-name")
    parser.add_argument("--manager-audit", action="store_true")
    parser.add_argument("--manager-audit-path", type=Path)
    parser.add_argument("--manager-save-name")
    parser.add_argument(
        "--allow-existing",
        action="store_true",
        help="allow rewriting named outputs after checking their mtimes",
    )
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    loaded_name = args.loaded_name or _default_name("d3_probe_loaded")
    after_turn_name = args.after_turn_name or _default_name("d3_probe_after_turn")
    edited_name = args.edited_name or _default_name("d3_probe_edited")
    capture_prefix = args.capture_prefix
    orders_save_name = args.orders_save_name
    manager_save_name = args.manager_save_name
    if (args.manager_audit or manager_save_name is not None) and args.manager_audit_path is None:
        manager_audit_path = args.save_root / f"{loaded_name}.manager.txt"
    else:
        manager_audit_path = args.manager_audit_path

    if (args.edit_node is None) != (args.edit_value is None):
        parser.error("--edit-node and --edit-value must be supplied together")
    if args.edit_node is not None and not args.skip_turn:
        parser.error("memory edits require --skip-turn")
    if args.skip_turn and args.orders_file and not orders_save_name:
        orders_save_name = _default_name("d3_probe_orders")
    if args.orders_file or args.orders_dir:
        if args.skip_turn and args.orders_dir:
            parser.error("--orders-dir is a multi-turn capture and cannot use --skip-turn")
        try:
            order_files = _collect_order_files(args)
            initial_file = args.initial_file
            if initial_file is None:
                initial_file = args.save_root / f"{args.load_name}.xml"
            if not initial_file.is_file():
                # The captured turn-zero filename is conventional and avoids
                # requiring a duplicate CLI option for the common fixture.
                initial_file = order_files[0].with_name("turn0_initial.xml")
            specs = build_capture_specs(initial_file, order_files)
        except (OSError, ValueError) as exc:
            parser.error(str(exc))
        if capture_prefix is None and not args.skip_turn:
            capture_prefix = _default_name("d3_probe_capture")
        order_spec = specs[0] if args.skip_turn and len(specs) == 1 else None
        if args.skip_turn and len(specs) != 1:
            parser.error("--skip-turn accepts exactly one --orders-file")
        capture_specs = None if args.skip_turn else specs
    else:
        specs = None
        order_spec = None
        capture_specs = None

    try:
        return run(
            load_name=args.load_name,
            loaded_name=loaded_name,
            after_turn_name=after_turn_name,
            edited_name=edited_name,
            edit_node=args.edit_node,
            edit_value=args.edit_value,
            turn_mode=args.turn_mode,
            gameplay_turn=args.gameplay_turn,
            sync_gameplay_turn=args.sync_gameplay_turn,
            skip_turn=args.skip_turn,
            timeout=args.timeout,
            order_spec=order_spec,
            capture_specs=capture_specs,
            capture_prefix=capture_prefix,
            orders_save_name=orders_save_name,
            manager_audit_path=manager_audit_path,
            manager_save_name=manager_save_name,
            allow_existing=args.allow_existing,
            game=args.game,
            probe=args.probe,
            save_root=args.save_root,
        )
    except (FileExistsError, FileNotFoundError, OSError, ValueError) as exc:
        print(f"native probe not started: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
