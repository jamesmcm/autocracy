from __future__ import annotations

from pathlib import Path

import pytest

from gamedrive.savecheck import NativeSaveError, validate_native_save


SAVE = Path("parity_cases/dem3saves/turn0_initial.xml")


@pytest.mark.skipif(not SAVE.exists(), reason="parity saves missing")
def test_captured_save_passes_native_output_validation():
    report = validate_native_save(SAVE)

    assert report.turn == 0
    assert report.country == "uk"
    assert report.policy_count == 123
    assert report.section_count == 9


def test_validation_rejects_truncated_output(tmp_path: Path):
    output = tmp_path / "truncated.xml"
    output.write_text("<header><version>1.30.2</version></header>", encoding="latin-1")

    with pytest.raises(NativeSaveError, match="truncated"):
        validate_native_save(output)
