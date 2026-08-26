"""Voter-population generation for countries without a reference save."""

from __future__ import annotations

import statistics as st

import pytest

from autocracy import simulator
from autocracy.models import Voter
from autocracy.voters import (
    OPPOSITION_PARTY,
    PLAYER_PARTY,
    apply_electorate,
    generate_country_state,
    generate_electorate,
)


def _data():
    return simulator.load_simulation_data()


def test_generated_electorate_structure_and_distributions():
    data = _data()
    state = generate_country_state("germany", base_value=-0.35)
    voters = state.voters

    assert len(voters) == 2000
    # Every voter carries all 21 group slots, with _All_ set to 1.0.
    for voter in voters:
        assert set(voter.groups) == set(range(21))
        assert voter.groups[20] == pytest.approx(1.0)
        assert 0.0 <= voter.inincome <= 1.0
    # The _All_ base shift lands the approval distribution at the target.
    assert st.mean(voter.value for voter in voters) == pytest.approx(
        -0.35, abs=1e-6
    )
    # Income is near-uniform over the population.
    assert st.mean(voter.inincome for voter in voters) == pytest.approx(
        0.5, abs=0.02
    )
    # The two-party setup exposes player (0) and opposition (1) types.
    assert state.parties[PLAYER_PARTY].party_type == 0
    assert state.parties[OPPOSITION_PARTY].party_type == 1
    affiliated = sum(
        1 for voter in voters if voter.party in (PLAYER_PARTY, OPPOSITION_PARTY)
    )
    assert 0 < affiliated < len(voters)


def test_generation_is_deterministic_per_country():
    germany_a = generate_country_state("germany", seed=7)
    germany_b = generate_country_state("germany", seed=7)
    usa = generate_country_state("usa", seed=7)

    assert [v.value for v in germany_a.voters] == [
        v.value for v in germany_b.voters
    ]
    # Different countries (different RNG streams) differ.
    assert [v.value for v in germany_a.voters] != [v.value for v in usa.voters]


def test_applied_electorate_matches_native_value_machinery():
    data = _data()
    state, _ = simulator.get_initial_state("germany")
    voters, parties = generate_electorate(data, "germany", base_value=-0.35)
    apply_electorate(state, data, voters, parties, base_value=-0.35)

    assert len(state.voters) == 2000
    assert set(state.parties) == {PLAYER_PARTY, OPPOSITION_PARTY}
    # The per-type neurons are seeded from the CSV defaults and the
    # population value recovers the target mean.
    assert state.voter_values["Socialist"] == pytest.approx(-0.3)
    assert st.mean(voter.value for voter in state.voters) == pytest.approx(
        -0.35, abs=1e-6
    )
    # Frequencies/percentages reflect the sampled membership shares.
    assert state.voter_frequencies["_All__freq"] == pytest.approx(1.0)


def test_generated_state_runs_elections_end_to_end():
    state, graph = simulator.get_initial_state("germany")
    data = simulator.load_simulation_data()

    for _ in range(20):
        state = simulator.process_end_of_turn(
            state, graph, data=data, config=None
        )
        state = simulator.resolve_election_if_ready(state, data=data)

    assert len(state.voters) == 2000
    assert state.poll_rate >= 0.0
    assert state.election_result is None or state.election_result in {
        "win",
        "loss",
    }


def test_generated_state_round_trips_for_turn_zero_saves():
    """The generated state must serialize so a turn-0 save can be written."""

    state = generate_country_state("france", seed=11)
    payload = simulator.state_to_dict(state)
    restored = simulator.state_from_dict(payload)

    assert len(restored.voters) == 2000
    assert restored.voters[0].groups == state.voters[0].groups
    assert restored.voters[0].value == pytest.approx(state.voters[0].value)


def test_uk_still_uses_the_reference_save():
    """The real UK save must keep winning over the generator."""

    state, _ = simulator.get_initial_state("uk")
    assert len(state.voters) == 2000
    # UK approval comes from the save's calibrated electorate, not the
    # generator's -0.35 default.
    assert st.mean(voter.value for voter in state.voters) != pytest.approx(
        -0.35, abs=1e-4
    )


def test_all_playable_countries_generate():
    for country in ("usa", "germany", "france", "canada", "australia"):
        state, _ = simulator.get_initial_state(country)
        assert len(state.voters) == 2000
        assert all(isinstance(v, Voter) for v in state.voters)