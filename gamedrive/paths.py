"""Resolve the installed Democracy 3 binary and save directory per host.

The probe is version-pinned to the v1.30.2 ELF, but the install location varies
by user.  These helpers honour the same environment overrides as the native
integration test, then fall back to the standard Steam prefix and Democracy 3
save path under the current user's home directory.
"""

from __future__ import annotations

import os
from pathlib import Path


def game_binary() -> Path:
    """Return the Democracy 3 x86_64 binary, honouring an explicit override."""
    override = os.environ.get("AUTOCRACY_DEMOCRACY3_BINARY")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local/share/Steam/steamapps/common/Democracy 3" / "Democracy3.bin.x86_64"


def save_root() -> Path:
    """Return the native savegames directory, honouring an explicit override."""
    override = os.environ.get("AUTOCRACY_DEMOCRACY3_SAVE_ROOT")
    if override:
        return Path(override).expanduser()
    return Path.home() / ".local/share/democracy3/savegames"


def _steam_runtime_lib_dirs() -> list[Path]:
    runtime = (
        Path.home()
        / ".local/share/Steam/steamapps/common/SteamLinuxRuntime/steam-runtime"
    )
    lib = runtime / "lib"
    if not lib.is_dir():
        return []
    return [lib / "x86_64-linux-gnu", lib / "i386-linux-gnu"]


def _runtime_libpng12() -> Path | None:
    for directory in _steam_runtime_lib_dirs():
        candidate = directory / "libpng12.so.0"
        if candidate.is_file():
            return candidate
    return None


def native_compat_lib_dir() -> Path | None:
    """Return a minimal library dir for the legacy shared libraries.

    Modern distributions no longer ship ``libpng12``, which the pinned
    v1.30.2 binary still links against.  Rather than putting the entire Steam
    runtime on ``LD_LIBRARY_PATH`` (that breaks gdb itself on hosts with newer
    system libraries), stage just the missing libraries beside the save root.
    """
    override = os.environ.get("AUTOCRACY_DEMOCRACY3_RUNTIME_LIB")
    if override:
        candidate = Path(override).expanduser()
        return candidate if candidate.is_dir() else None
    source = _runtime_libpng12()
    if source is None:
        return None
    directory = save_root().parent / "compat"
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "libpng12.so.0"
    if not target.is_file():
        target.symlink_to(source)
    return directory