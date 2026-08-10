from pathlib import Path

import pytest

from autocracy.models import EffectHistory
from gamedrive.term_audit import (
    _effect_history_max_delta,
    chain_capture_paths,
    native_manager_summary,
)


SAVES_DIR = Path("parity_cases/dem3saves")


def test_chain_capture_paths_use_step_turn_names(tmp_path: Path):
    assert chain_capture_paths(tmp_path, "uk_term", 3) == [
        tmp_path / "uk_term_step1_turn1.xml",
        tmp_path / "uk_term_step2_turn1.xml",
        tmp_path / "uk_term_step3_turn1.xml",
    ]


def test_effect_history_comparison_binds_native_idless_records_by_pair():
    actual = [EffectHistory("A", "B", [0.1, 0.2], effect_id="sim::1")]
    expected = [EffectHistory("A", "B", [0.1, 0.2])]

    assert _effect_history_max_delta(actual, expected) == pytest.approx(0.0)


@pytest.mark.skipif(
    not (SAVES_DIR / "turn1_initial.xml").exists(), reason="saves missing"
)
def test_native_manager_summary_reads_serialized_term_fields():
    summary = native_manager_summary(SAVES_DIR / "turn1_initial.xml")

    assert summary.poll_rate is not None
    assert summary.peak_poll_rate is not None
    assert summary.turns_until_election is not None
    assert summary.current_term is not None
    assert summary.active_minister_departments
