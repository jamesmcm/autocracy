from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import shutil

import pytest

from autocracy import simulator
from autocracy.agent import ElectionOracleAgent, SimulatorOracleAgent
from autocracy.models import PartyState, PolicyAction, PolicyActionOption, Voter
from autocracy.oracle import OracleElectionLoss
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


def test_simulator_oracle_enumerates_multi_policy_batches():
    agent = SimulatorOracleAgent(
        beam_width=1,
        search_horizon=1,
        candidate_limit=None,
        max_actions_per_turn=2,
        batch_candidate_limit=4,
    )
    options = [
        PolicyActionOption("First", "raise", 0.1, 0.1, 1.0, 0.0),
        PolicyActionOption("Second", "raise", 0.1, 0.1, 1.0, 0.0),
    ]
    state = replace(agent.state, political_capital=3.0)

    batches = agent._action_batches(state, options)

    assert () in batches
    assert any(len(batch) == 2 for batch in batches)


def test_seeded_batch_sampling_keeps_all_single_moves_and_samples_pairs():
    agent = SimulatorOracleAgent(
        beam_width=1,
        search_horizon=1,
        candidate_limit=None,
        max_actions_per_turn=2,
        batch_candidate_limit=4,
        random_seed=17,
    )
    options = [
        PolicyActionOption("First", "raise", 0.1, 0.1, 1.0, 0.0),
        PolicyActionOption("Second", "raise", 0.1, 0.1, 1.0, 0.0),
        PolicyActionOption("Third", "raise", 0.1, 0.1, 1.0, 0.0),
    ]
    state = replace(agent.state, political_capital=4.0)

    batches = agent._action_batches(state, options)

    assert len(batches) == 5  # no-op, all singles, and one sampled pair
    assert {batch[0].policy_name for batch in batches[1:4] if len(batch) == 1} == {
        "First",
        "Second",
        "Third",
    }


def test_election_oracle_defaults_to_the_documented_winning_search():
    from autocracy.oracle import PROVEN_ELECTION_SEARCH

    agent = ElectionOracleAgent()

    assert agent.beam_width == PROVEN_ELECTION_SEARCH["beam_width"] == 6
    assert agent.search_horizon == PROVEN_ELECTION_SEARCH["search_horizon"] == 5
    assert agent.candidate_limit == PROVEN_ELECTION_SEARCH["candidate_limit"] == 16
    assert (
        agent.batch_candidate_limit
        == PROVEN_ELECTION_SEARCH["batch_candidate_limit"]
        == 64
    )
    assert (
        agent.max_actions_per_turn
        == PROVEN_ELECTION_SEARCH["max_actions_per_turn"]
        == 2
    )
    assert (
        agent.time_budget_seconds
        == PROVEN_ELECTION_SEARCH["time_budget_seconds"]
        == 15.0
    )


def test_election_oracle_full_term_horizon_still_resolves_to_the_boundary():
    agent = ElectionOracleAgent(search_horizon=None)

    assert agent._resolved_search_horizon(agent.state) == 16


def test_simulator_oracle_returns_safe_fallback_when_budget_expires():
    agent = SimulatorOracleAgent(
        beam_width=1,
        search_horizon=1,
        candidate_limit=1,
        time_budget_seconds=1e-9,
        objective=lambda state: float(state.turn),
    )

    result = agent.search(options=[])

    assert result.timed_out
    assert result.completed_depth == 1
    assert result.first_actions == ()
    assert result.first_state.turn == agent.state.turn + 1


def test_simulator_oracle_resolves_election_and_rejects_loss(monkeypatch):
    agent = SimulatorOracleAgent(
        beam_width=1,
        search_horizon=1,
        candidate_limit=1,
        objective=lambda state: float(state.turn),
    )
    agent.state = replace(
        agent.state,
        election_turns_until=1,
        parties={
            "Player": PartyState("Player", party_type=0),
            "Opposition": PartyState("Opposition", party_type=1),
        },
        voters=[
            Voter(party="Player"),
            Voter(party="Opposition", value=-1.0),
            Voter(party="Opposition", value=-1.0),
        ],
    )

    def fake_turn(state, graph, data=None, config=None, **kwargs):
        return replace(state, turn=state.turn + 1, election_turns_until=0)

    monkeypatch.setattr(simulator, "process_end_of_turn", fake_turn)

    with pytest.raises(OracleElectionLoss) as caught:
        agent.search(options=[])

    assert caught.value.state.election_result == "loss"
    assert caught.value.state.election_player_votes == 1
    assert caught.value.state.election_opposition_votes == 2


def test_simulator_oracle_carries_a_winning_election_into_the_next_term(monkeypatch):
    agent = SimulatorOracleAgent(
        beam_width=1,
        search_horizon=1,
        candidate_limit=1,
        objective=lambda state: float(state.turn),
    )
    agent.state = replace(
        agent.state,
        election_turns_until=1,
        parties={
            "Player": PartyState("Player", party_type=0),
            "Opposition": PartyState("Opposition", party_type=1),
        },
        voters=[Voter(party="Player"), Voter(party="Player")],
    )

    def fake_turn(state, graph, data=None, config=None, **kwargs):
        return replace(state, turn=state.turn + 1, election_turns_until=0)

    monkeypatch.setattr(simulator, "process_end_of_turn", fake_turn)

    result = agent.search(options=[])

    assert result.first_state.election_result == "win"
    assert result.first_state.election_current_term == 1
    assert result.first_state.election_turns_until == 16


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


def test_gamedrive_oracle_resolves_and_persists_native_election_boundary(
    tmp_path: Path, monkeypatch
):
    source = Path("parity_cases/dem3saves/turn0_initial.xml")
    shutil.copyfile(source, tmp_path / "source.xml")

    def fake_runner(**kwargs):
        root = Path(kwargs["save_root"])
        source_path = root / f"{kwargs['load_name']}.xml"
        shutil.copyfile(source_path, root / f"{kwargs['loaded_name']}.xml")
        output_path = root / f"{kwargs['after_turn_name']}.xml"
        text = source_path.read_text(encoding="latin-1")
        text = text.replace(
            "<turnsuntilelection>16</turnsuntilelection>",
            "<turnsuntilelection>0</turnsuntilelection>",
            1,
        )
        output_path.write_text(text, encoding="latin-1")
        return 0

    def fake_resolve(state, data=None):
        return replace(
            state,
            election_turns_until=16,
            election_current_term=state.election_current_term + 1,
            election_result="win",
            last_election_winner="player",
        )

    monkeypatch.setattr(simulator, "resolve_election", fake_resolve)
    agent = GameDriveOracleAgent(
        "source",
        save_root=tmp_path,
        beam_width=1,
        search_horizon=1,
        candidate_limit=1,
        runner=fake_runner,
    )

    result = agent.search(options=[])

    assert result.first_state.election_turns_until == 16
    assert result.first_state.election_current_term == 1
    assert result.first_runtime_state is not None
    assert result.first_runtime_state.election_result == "win"
    assert result.first_artifact is not None
    saved_text = (tmp_path / f"{result.first_artifact}.xml").read_text(
        encoding="latin-1"
    )
    assert "<turnsuntilelection>16</turnsuntilelection>" in saved_text
    assert "<currentterm>1</currentterm>" in saved_text

    committed = agent.commit_result(result)
    assert committed.election_result == "win"
