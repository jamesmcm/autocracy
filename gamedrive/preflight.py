"""Check version-specific native entry points without launching the game.

This is the safe first step for the gdb/LD_PRELOAD work. It reads the ELF's
demangled symbol table and verifies the addresses used by the static audit;
it never creates a display or starts Democracy 3.
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


DEFAULT_GAME = Path(
    "/home/gopostal/.local/share/Steam/steamapps/common/Democracy 3/"
    "Democracy3.bin.x86_64"
)

REQUIRED_SYMBOLS: dict[str, int] = {
    "SIM_LoadGame::LoadMission()": 0x5D9DB0,
    "SIM_MissionManager::LoadMissions()": 0x5ED4B0,
    "SIM_MissionManager::GetByName(std::string)": 0x5ED750,
    "SIM_MissionManager::SetCurrent(SIM_Mission*)": 0x5EC9D0,
    "SIM_Simulation::ApplyMissionSpecificData(bool)": 0x60D940,
    "SIM_Gameplay::PreLoadGame()": 0x5CEC10,
    "SIM_Gameplay::PostLoadGame()": 0x5D0080,
    "SIM_Simulation::PreLoad()": 0x60EF50,
    "SIM_Simulation::PostLoad()": 0x60D8F0,
    "SIM_Simulation::NextTurn()": 0x60E120,
    "SIM_NeuralEffect::NextTurn()": 0x5F0210,
}

_SYMBOL_LINE = re.compile(r"^[0-9a-fA-F]+\s+\w\s+(.+)$")


def read_symbols(binary: Path) -> dict[str, int]:
    """Return demangled symbol addresses from *binary* using ``nm``."""
    result = subprocess.run(
        ["nm", "-C", "--defined-only", str(binary)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or "nm failed"
        raise RuntimeError(detail)

    symbols: dict[str, int] = {}
    for line in result.stdout.splitlines():
        fields = line.split(maxsplit=2)
        if len(fields) != 3 or not re.fullmatch(r"[0-9a-fA-F]+", fields[0]):
            continue
        match = _SYMBOL_LINE.match(line)
        if match is not None:
            symbols.setdefault(match.group(1), int(fields[0], 16))
    return symbols


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_GAME)
    args = parser.parse_args()

    binary = args.binary.resolve()
    if not binary.is_file():
        print(f"missing binary: {binary}")
        return 2

    try:
        symbols = read_symbols(binary)
    except (OSError, RuntimeError) as exc:
        print(f"unable to inspect {binary}: {exc}")
        return 2

    mismatches: list[str] = []
    for name, expected in REQUIRED_SYMBOLS.items():
        actual = symbols.get(name)
        if actual is None:
            mismatches.append(f"missing {name}")
            continue
        if actual != expected:
            mismatches.append(
                f"{name}: expected 0x{expected:x}, found 0x{actual:x}"
            )

    if mismatches:
        print(f"static preflight failed: {binary}")
        for mismatch in mismatches:
            print(f"- {mismatch}")
        return 1

    print(f"static preflight passed: {binary}")
    print(f"verified {len(REQUIRED_SYMBOLS)} version-specific entry points")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
