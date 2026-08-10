from __future__ import annotations

from pathlib import Path

import pytest

from gamedrive.capture import replay_simulator
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
