from __future__ import annotations

from pathlib import Path

import pytest

from gamedrive.capture import replay_simulator
from gamedrive.inject_drive import _output_paths
from gamedrive.term_capture import build_term_specs, capture_turns, mission_term_length


SAVES = Path("parity_cases/dem3saves")


def test_uk_term_length_and_extension():
    assert mission_term_length("uk") == 16
    assert capture_turns("uk", turns=None, extra_turns=8) == 24
    assert capture_turns("uk", turns=3, extra_turns=99) == 3


@pytest.mark.skipif(
    not (SAVES / "turn0_initial.xml").exists(), reason="parity saves missing"
)
def test_existing_orders_are_padded_to_term_length():
    specs = build_term_specs(SAVES / "turn0_initial.xml", SAVES, turns=24)

    assert len(specs) == 24
    assert specs[0].startswith("slider|CorporationTax|0")
    assert specs[11].startswith("cancel|ChildBenefit|0")
    assert specs[12:] == [""] * 12


@pytest.mark.skipif(
    not (SAVES / "turn0_initial.xml").exists(), reason="parity saves missing"
)
def test_no_order_replay_can_cover_a_full_term_plus_extra_turns():
    snapshots = replay_simulator(
        SAVES / "turn0_initial.xml",
        [],
        turns=24,
    )

    assert sorted(snapshots) == list(range(1, 25))


def test_capture_can_keep_only_the_final_native_checkpoint():
    paths = _output_paths(
        loaded_name="loaded",
        after_turn_name="after",
        edited_name="edited",
        orders_save_name=None,
        manager_save_name=None,
        capture_prefix="capture",
        capture_count=3,
        write_each_step=False,
        edit_node=None,
        skip_turn=False,
        save_root=Path("/tmp/savegames"),
    )

    assert paths == [Path("/tmp/savegames/loaded.xml"), Path("/tmp/savegames/after.xml")]
