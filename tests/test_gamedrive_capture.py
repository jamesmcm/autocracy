from __future__ import annotations

from pathlib import Path

import pytest

from gamedrive.capture import capture_paths, compare_native_saves, replay_simulator
from autocracy.savegame import parse_savegame


SAVES = Path("parity_cases/dem3saves")


@pytest.mark.skipif(
    not (SAVES / "turn0_initial.xml").exists(), reason="parity saves missing"
)
def test_bounded_replay_includes_no_order_turns():
    order_files = sorted(SAVES.glob("turn*_o*.xml"))

    snapshots = replay_simulator(SAVES / "turn0_initial.xml", order_files)
    reference = parse_savegame(SAVES / "turn1_initial.xml")

    assert set(snapshots) == set(range(1, 13))
    assert snapshots[1].total_income == pytest.approx(
        reference.total_income, abs=1.0
    )


@pytest.mark.skipif(
    not (SAVES / "turn1_initial.xml").exists(), reason="parity saves missing"
)
def test_capture_comparison_matches_native_turn_number():
    order_files = sorted(SAVES.glob("turn*_o*.xml"))
    snapshots = replay_simulator(SAVES / "turn0_initial.xml", order_files)

    comparison = compare_native_saves(
        [SAVES / "turn1_initial.xml"], snapshots
    )[0]

    assert comparison.native_turn == 1
    assert comparison.simulator_turn == 1
    assert comparison.max_node_name


def test_capture_paths_use_native_xml_names(tmp_path: Path):
    assert capture_paths(tmp_path, "run", 3) == [
        tmp_path / "run_turn1.xml",
        tmp_path / "run_turn2.xml",
        tmp_path / "run_turn3.xml",
    ]
