from __future__ import annotations

from pathlib import Path

import pytest

from gamedrive.order_plan import (
    NativeOrder,
    build_capture_specs,
    plan_order_files,
)


SAVES = Path("parity_cases/dem3saves")


@pytest.mark.skipif(
    not (SAVES / "turn0_initial.xml").exists(), reason="parity saves missing"
)
def test_floor_moves_are_slider_orders_not_cancellations():
    actions = plan_order_files(
        SAVES / "turn0_initial.xml", SAVES / "turn0_orders.xml"
    )

    assert [(action.action, action.policy_name, action.target) for action in actions] == [
        ("slider", "CorporationTax", 0.0),
        ("slider", "IncomeTax", 0.0),
        ("slider", "Prisons", 0.0),
    ]


@pytest.mark.skipif(
    not (SAVES / "turn2_initial.xml").exists(), reason="parity saves missing"
)
def test_orders_use_active_flag_to_detect_real_cancellation():
    actions = plan_order_files(
        SAVES / "turn2_initial.xml", SAVES / "turn2_orders.xml"
    )

    assert [(action.action, action.policy_name) for action in actions] == [
        ("slider", "CCTVCameras"),
        ("cancel", "StateHousing"),
    ]


@pytest.mark.skipif(
    not (SAVES / "turn0_initial.xml").exists(), reason="parity saves missing"
)
def test_capture_specs_preserve_missing_no_order_turns():
    order_files = sorted(SAVES.glob("turn*_o*.xml"))
    specs = build_capture_specs(SAVES / "turn0_initial.xml", order_files)

    assert len(specs) == 12
    assert specs[0].startswith("slider|CorporationTax|0")
    assert specs[3] == ""  # no turn3_orders capture was supplied
    assert "cancel|ChildBenefit|0" in specs[11]


def test_native_order_encoding_rejects_ambiguous_names():
    with pytest.raises(ValueError, match="protocol delimiters"):
        NativeOrder("slider", "bad|name", 0.5).encode()
