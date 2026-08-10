from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from gamedrive.savecheck import validate_native_save


DEFAULT_GAME = Path(
    "/home/gopostal/.local/share/Steam/steamapps/common/Democracy 3/"
    "Democracy3.bin.x86_64"
)
DEFAULT_SAVE_ROOT = Path("/home/gopostal/.local/share/democracy3/savegames")


@pytest.mark.integration
def test_native_probe_smoke_is_opt_in():
    """Exercise one load/save boundary only when explicitly requested."""
    if os.environ.get("AUTOCRACY_RUN_NATIVE_INTEGRATION") != "1":
        pytest.skip("set AUTOCRACY_RUN_NATIVE_INTEGRATION=1 to run native smoke")

    game = Path(os.environ.get("AUTOCRACY_DEMOCRACY3_BINARY", str(DEFAULT_GAME)))
    save_root = Path(
        os.environ.get("AUTOCRACY_DEMOCRACY3_SAVE_ROOT", str(DEFAULT_SAVE_ROOT))
    )
    source_name = os.environ.get("AUTOCRACY_NATIVE_SOURCE_NAME")
    if not game.is_file():
        pytest.skip(f"installed Democracy 3 binary unavailable: {game}")
    if source_name is None or not (save_root / f"{source_name}.xml").is_file():
        pytest.skip("AUTOCRACY_NATIVE_SOURCE_NAME must identify an installed save")
    if not Path("gamedrive/libd3probe.so").is_file():
        pytest.skip("native probe is not built; run make -C gamedrive")
    missing_tools = [
        tool
        for tool in ("xvfb-run", "gdb", "timeout")
        if shutil.which(tool) is None
    ]
    if missing_tools:
        pytest.skip("native smoke tools unavailable: " + ", ".join(missing_tools))

    output_name = f"autocracy_pytest_probe_{os.getpid()}"
    command = [
        sys.executable,
        "gamedrive/inject_drive.py",
        "--game",
        str(game),
        "--save-root",
        str(save_root),
        "--load-name",
        source_name,
        "--loaded-name",
        output_name,
        "--skip-turn",
        "--timeout",
        "60",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        assert result.returncode == 0, result.stdout + result.stderr
        validate_native_save(save_root / f"{output_name}.xml")
    finally:
        output_path = save_root / f"{output_name}.xml"
        if output_path.is_file():
            output_path.unlink()
