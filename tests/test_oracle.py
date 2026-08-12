from __future__ import annotations

from pathlib import Path
import shutil

import pytest

from autocracy import simulator
from autocracy.agent import SimulatorOracleAgent
from autocracy.models import PolicyAction, PolicyActionOption
from gamedrive.oracle import GameDriveOracleAgent, native_order_for_option


def test_simulator_oracle_scores_real_turn_branches():
    agent = SimulatorOracleAgent(
        beam_width=1,
        search_horizon=1,
        candidate_limit=2,
        objective=lambda state: state.values["GDP"],
    )
    options = agent.available_actions()
    bounded = sorted(
        options,
        key=lambda option: (
            option.policy_name,
            option.action_type,
            option.resulting_level,
        ),
    )[:2]
    expected = []
    for option in bounded:
        ordered = simulator.apply_actions(
            agent.state,
            [
                PolicyAction(
                    option.policy_name,
                    option.delta,
                    option.action_type,
                )
            ],
            data=agent.data,
        )
        expected.append(
            simulator.process_end_of_turn(ordered, agent.graph, data=agent.data)
            .values["GDP"]
        )
    expected.append(
        simulator.process_end_of_turn(agent.state, agent.graph, data=agent.data)
        .values["GDP"]
    )

    result = agent.search(options=options)

    assert result.evaluated == 3  # two policy moves plus the no-op branch
    assert result.first_state.turn == agent.state.turn + 1
    assert result.score == pytest.approx(max(expected))


def test_seeded_candidate_sampling_is_reproducible():
    first = SimulatorOracleAgent(candidate_limit=3, random_seed=17)
    second = SimulatorOracleAgent(candidate_limit=3, random_seed=17)

    first_options = first._options(first.state, first.available_actions())
    second_options = second._options(second.state, second.available_actions())

    assert [option.policy_name for option in first_options] == [
        option.policy_name for option in second_options
    ]


def test_native_order_translation_preserves_introduce_and_cancel_semantics():
    introduce = native_order_for_option(
        PolicyActionOption("CarbonTax", "introduce", 0.5, 0.5, 10.0, 4.0)
    )
    cancel = native_order_for_option(
        PolicyActionOption("CarbonTax", "cancel", -0.5, 0.0, 10.0, 4.0)
    )
    raise_order = native_order_for_option(
        PolicyActionOption("CarbonTax", "raise", 0.05, 0.55, 10.0, 4.0)
    )

    assert introduce.encode() == "implement|CarbonTax|0.5"
    assert cancel.encode() == "cancel|CarbonTax|0"
    assert raise_order.encode() == "slider|CarbonTax|0.55"


def test_gamedrive_oracle_uses_fresh_native_branch_artifact(tmp_path: Path):
    source = Path("parity_cases/dem3saves/turn0_initial.xml")
    shutil.copyfile(source, tmp_path / "source.xml")
    calls: list[dict] = []

    def fake_runner(**kwargs):
        calls.append(kwargs)
        root = Path(kwargs["save_root"])
        source_path = root / f"{kwargs['load_name']}.xml"
        shutil.copyfile(source_path, root / f"{kwargs['loaded_name']}.xml")
        shutil.copyfile(source_path, root / f"{kwargs['after_turn_name']}.xml")
        return 0

    agent = GameDriveOracleAgent(
        "source",
        save_root=tmp_path,
        beam_width=1,
        search_horizon=1,
        candidate_limit=1,
        runner=fake_runner,
    )

    result = agent.search(options=[])

    assert len(calls) == 1
    assert calls[0]["order_spec"] is None
    assert result.first_actions == ()
    assert result.first_artifact is not None
    assert (tmp_path / f"{result.first_artifact}.xml").is_file()
    assert not list(tmp_path.glob("*_loaded.xml"))
