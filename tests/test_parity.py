from __future__ import annotations

import json
from pathlib import Path

import pytest

from autocracy.parity import load_parity_corpus, snapshot_from_savegame
from autocracy.models import PolicyAction
from autocracy.savegame import load_state_from_savegame, parse_savegame
from autocracy.simulator import (
    apply_actions,
    get_initial_state,
    load_simulation_data,
    process_end_of_turn,
)


CORPUS_PATH = Path("parity_cases/uk_noop.json")
LIVE_CORPUS_PATH = Path("parity_cases/uk_bus_lanes_live.json")


def test_save_snapshot_extracts_full_effect_memory():
    snapshot = snapshot_from_savegame(parse_savegame("gamedata/saves/uk0.xml"))
    effect_memory = snapshot["effect_memory"]
    assert snapshot["political_capital"] == pytest.approx(26.0)
    assert snapshot["parties"]["The Republicans"]["member_history"][:3] == [
        422,
        434,
        407,
    ]
    assert effect_memory["record_count"] == 303
    assert effect_memory["history_lengths"] == [33]
    assert len(effect_memory["records"]) == 303
    assert len(effect_memory["records"][0]["values"]) == 33


def test_uk_noop_corpus_covers_both_shipped_transitions():
    corpus = load_parity_corpus(CORPUS_PATH)
    assert [case["id"] for case in corpus["cases"]] == [
        "uk_noop_turn_0_to_1",
        "uk_noop_turn_1_to_2",
    ]
    required = {
        "simvalues",
        "political_capital",
        "policies",
        "finance",
        "situations",
        "hidden_values",
        "inherited",
        "voters",
        "policy_runtime",
        "effect_memory",
    }
    for case in corpus["cases"]:
        expected = case["expected"]
        assert required <= expected.keys()
        assert case["action"] == {"type": "noop"}
        assert case["initial_save"].endswith("gamedata/saves/uk0.xml") or case[
            "initial_save"
        ].endswith("gamedata/saves/uk1.xml")
        assert case["expected_save"].endswith("gamedata/saves/uk1.xml") or case[
            "expected_save"
        ].endswith("gamedata/saves/uk2.xml")

        reference = snapshot_from_savegame(parse_savegame(case["expected_save"]))
        for section in required - {"effect_memory"}:
            assert expected[section] == reference[section]
        reference_memory = reference["effect_memory"]
        expected_memory = expected["effect_memory"]
        assert expected_memory["record_count"] == reference_memory["record_count"]
        assert expected_memory["history_lengths"] == reference_memory["history_lengths"]
        assert expected_memory["sample_records"] == [
            reference_memory["records"][0],
            reference_memory["records"][150],
            reference_memory["records"][-1],
        ]

    # Keep the fixture valid JSON independently of the loader implementation.
    json.dumps(corpus)


def test_uk_noop_shipped_transitions_match_credited_one_turn_error_budget():
    """Both shipped no-op runs stay inside the one-turn error budget.

    The remaining outlier is Immigration: the game applies BorderControls'
    output at a strength the shipped saves imply is slightly above its raw
    ring value (the residual is documented in SIMULATION.md).  Everything
    else reproduces within 0.01.
    """

    data = load_simulation_data()
    for case in load_parity_corpus(CORPUS_PATH)["cases"]:
        state, graph = load_state_from_savegame(case["initial_save"], data)
        actual = process_end_of_turn(state, graph, data)
        expected = parse_savegame(case["expected_save"])

        errors = [
            abs(actual.values[name] - value)
            for name, value in expected.simvalues.items()
        ]
        assert sum(errors) / len(errors) < 0.004
        assert max(errors) < 0.05
        assert sum(error <= 0.01 for error in errors) >= 38
        assert set(actual.active_situations) == set(expected.active_situations)
        assert actual.total_income == pytest.approx(expected.total_income, abs=0.01)
        assert actual.total_expenditure == pytest.approx(
            expected.total_expenditure, abs=0.01
        )
        assert actual.political_capital == pytest.approx(expected.political_capital)


def test_live_bus_lanes_case_matches_policy_runtime_semantics():
    corpus = load_parity_corpus(LIVE_CORPUS_PATH)
    case = corpus["cases"][0]
    action = case["action"]
    expected = case["expected"]
    expected_runtime = expected["policy_runtime"]

    assert action["policy"] == "BusLanes"
    assert action["type"] == "set_policy"
    assert expected["turn"] == 1
    assert expected["political_capital"] == pytest.approx(45.0)
    assert set(case["expected_deltas"]) == set(expected["simvalues"])
    assert case["tolerances"]["stochastic_fields"] == [
        "simvalues",
        "hidden_values",
        "voters",
    ]

    state, graph = get_initial_state("uk")
    updated = apply_actions(
        state,
        [
            PolicyAction(
                policy_name=action["policy"],
                delta=action["requested_level"] - action["from_level"],
            )
        ],
    )
    actual = process_end_of_turn(updated, graph)

    assert actual.policies["BusLanes"] == pytest.approx(
        expected["policies"]["BusLanes"], abs=1e-7
    )
    assert actual.policy_desired_throttles["BusLanes"] == pytest.approx(
        expected_runtime["desired_throttles"]["BusLanes"], abs=1e-7
    )
    assert actual.effect_throttles["BusLanes"] == pytest.approx(
        expected_runtime["effect_throttles"]["BusLanes"], abs=1e-6
    )
    assert actual.policy_implementations["BusLanes"] == pytest.approx(
        expected_runtime["implementations"]["BusLanes"], abs=1e-7
    )
    assert actual.policy_active["BusLanes"] == expected_runtime["active"]["BusLanes"]
    assert actual.total_income == pytest.approx(
        expected["finance"]["total_income"], abs=0.01
    )
    assert actual.total_expenditure == pytest.approx(
        expected["finance"]["total_expenditure"], abs=0.01
    )
    assert actual.political_capital == pytest.approx(
        expected["political_capital"], abs=0.01
    )
