"""Validate native Democracy 3 save output without launching the game."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import xml.etree.ElementTree as ET

from autocracy.savegame import ENCODING, parse_savegame


EXPECTED_VERSION = "1.30.2"
REQUIRED_SECTIONS = (
    "<header>",
    "<load_data>",
    "<simvalues>",
    "<policies>",
    "<finances>",
    "<parties>",
    "<voters>",
    "<polls>",
    "<stats>",
)


class NativeSaveError(ValueError):
    """Raised when an output is missing a native save boundary or section."""


@dataclass(frozen=True, slots=True)
class NativeSaveReport:
    """Small parsed summary used by the driver and offline comparisons."""

    path: Path
    byte_count: int
    country: str
    turn: int
    policy_count: int
    section_count: int


def validate_native_save(
    path: str | Path,
    *,
    expected_version: str = EXPECTED_VERSION,
) -> NativeSaveReport:
    """Check that *path* is a complete, parser-readable native save.

    Democracy 3 writes several top-level XML sections followed by one final
    ``</xml>`` marker.  The save parser wraps those sections for inspection;
    this function additionally checks the native terminator and all sections
    needed by the simulator bridge.  It deliberately does not rewrite or
    repair the file.
    """
    save_path = Path(path)
    if not save_path.is_file():
        raise NativeSaveError(f"missing native save: {save_path}")

    raw = save_path.read_text(encoding=ENCODING)
    stripped = raw.rstrip()
    if not stripped.endswith("</xml>"):
        raise NativeSaveError(
            f"native save is truncated (missing final </xml>): {save_path}"
        )
    if stripped.count("</xml>") != 1:
        raise NativeSaveError(f"native save has multiple XML terminators: {save_path}")

    version_marker = f"<version>{expected_version}</version>"
    if version_marker not in raw:
        raise NativeSaveError(
            f"native save version is not {expected_version}: {save_path}"
        )
    missing = [section for section in REQUIRED_SECTIONS if section not in raw]
    if missing:
        raise NativeSaveError(
            f"native save is missing sections {', '.join(missing)}: {save_path}"
        )

    try:
        save = parse_savegame(save_path)
    except (OSError, ET.ParseError, ValueError) as exc:
        raise NativeSaveError(f"native save cannot be parsed: {save_path}: {exc}") from exc
    if not save.policies:
        raise NativeSaveError(f"native save has no policies: {save_path}")

    return NativeSaveReport(
        path=save_path,
        byte_count=len(raw.encode(ENCODING)),
        country=save.country,
        turn=save.turn,
        policy_count=len(save.policies),
        section_count=len(REQUIRED_SECTIONS),
    )
