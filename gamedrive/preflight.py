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
    "SIM_GetSimulation()": 0x5AB1A0,
    "SIM_LoadGame::LoadMission()": 0x5D9DB0,
    "SIM_LoadGame::OpenSavedFile(std::string const&)": 0x5D4A90,
    "SIM_LoadGame::LoadGameData()": 0x5DCBA0,
    "SIM_LoadGame::ProcessGameLoad()": 0x5DCEE0,
    "SIM_MissionManager::LoadMissions()": 0x5ED4B0,
    "SIM_MissionManager::GetByName(std::string)": 0x5ED750,
    "SIM_MissionManager::SetCurrent(SIM_Mission*)": 0x5EC9D0,
    "SIM_Simulation::ApplyMissionSpecificData(bool)": 0x60D940,
    "SIM_Gameplay::PreLoadGame()": 0x5CEC10,
    "SIM_Gameplay::PostLoadGame()": 0x5D0080,
    "SIM_Simulation::PreLoad()": 0x60EF50,
    "SIM_Simulation::PostLoad()": 0x60D8F0,
    "SIM_Simulation::NextTurn()": 0x60E120,
    "SIM_Simulation::GetNeuronByName(std::string)": 0x60C140,
    "SIM_Simulation::Initialise()": 0x60F560,
    "SIM_Neuron::CalculateValue()": 0x5F12E0,
    "SIM_NeuralEffect::NextTurn()": 0x5F0210,
    "SIM_Situation::NextTurn()": 0x6127F0,
    "SIM_SituationManager::NextTurn()": 0x612B40,
    "SIM_FinanceManager::NextTurn()": 0x5CB7D0,
    "SIM_GlobalEconomy::BackProjectHistory()": 0x5D1370,
    "SIM_GlobalEconomy::Calculate()": 0x5D14E0,
    "SIM_GlobalEconomy::NextTurn()": 0x5D1670,
    "SIM_Voter::UpdateIncome()": 0x61E620,
    "SIM_Voter::CalculateApproval()": 0x61B880,
    "SIM_Voter::ConsiderPartyMembership(int)": 0x61D000,
    "SIM_Voter::NextTurn()": 0x61EF40,
    "SIM_VoterManager::NextTurn()": 0x620570,
    "SIM_VoterManager::PreJoinParties()": 0x6208E0,
    "SIM_VoterManager::PostLoad()": 0x6209C0,
    "SIM_VoterType::NextTurn()": 0x6223A0,
    "SIM_VoterType::PostLoad()": 0x6228E0,
    "SIM_VoterType::CalculatePercentage()": 0x6229C0,
    "SIM_VoterType::ForceVoter(SIM_Voter*, float)": 0x623350,
    "SIM_Party::NextTurn()": 0x5F4650,
    "SIM_PartyManager::NextTurn()": 0x5F4920,
    "SIM_SaveGame::SaveParties()": 0x604D40,
    "SIM_Policy::ForceSlider(float)": 0x5F5BA0,
    "GRandom::RandUnitFloat()": 0x635310,
    "GRandom::Init(int, int)": 0x6354A0,
    "GRandom::RandReal(float, float)": 0x6356F0,
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
