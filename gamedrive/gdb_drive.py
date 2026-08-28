"""Drive the real Democracy 3 simulation externally via gdb under Xvfb.

This launches the installed game under a virtual X server and gdb, stops it at
``mainLoop()`` (before the graphics resize crash), and reports what of the
simulation is reachable from outside the process. It remains a reachability
diagnostic; use ``inject_drive.py`` for the bounded load/order/turn/save path.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import time
from pathlib import Path

from gamedrive.paths import game_binary

GAME = game_binary()
HARNESS = Path(__file__).with_name("harness.gdb")
DISPLAY = ":99"


def _xvfb_running() -> bool:
    try:
        out = subprocess.run(
            ["pgrep", "-f", f"Xvfb {DISPLAY}"], capture_output=True, text=True
        )
        return out.returncode == 0
    except FileNotFoundError:
        return False


def run(verbose: bool = False, timeout: int = 90) -> str:
    """Run the game under gdb and return the harness output."""
    xvfb = None
    if not _xvfb_running():
        xvfb = subprocess.Popen(
            ["Xvfb", DISPLAY, "-screen", "0", "1920x1080x24"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        time.sleep(2)
    env = dict(os.environ, DISPLAY=DISPLAY)
    try:
        proc = subprocess.run(
            ["timeout", str(timeout), "gdb", "-batch", "-x", str(HARNESS), GAME],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout + 15,
        )
        return proc.stdout + proc.stderr
    finally:
        if xvfb is not None:
            xvfb.terminate()
            try:
                xvfb.wait(timeout=5)
            except subprocess.TimeoutExpired:
                xvfb.kill()


def summarize(output: str) -> str:
    """Extract the interesting markers from the raw gdb output."""
    lines = []
    for line in output.splitlines():
        text = line.strip()
        if any(
            marker in text
            for marker in (
                "STOPPED at mainLoop",
                "SIM_GetSimulation() =",
                "NextTurn",
                "entry points available",
                "fault without a country",
                "SIM_Simulation::",
                "SIM_Policy::",
                "SIM_LoadGame::",
                "received signal",
            )
        ):
            lines.append(text)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verbose", action="store_true", help="show raw output")
    parser.add_argument("--timeout", type=int, default=90)
    args = parser.parse_args()

    print(f"launching the game under gdb on {DISPLAY} ...")
    output = run(verbose=args.verbose, timeout=args.timeout)
    if args.verbose:
        print(output)
    print(summarize(output))


if __name__ == "__main__":
    main()
