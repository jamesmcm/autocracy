"""Deterministic voter-population generation for countries without a save.

Democracy 3 seeds a country's electorate at load time from its voter-type
definitions and mission config; only the UK ships a reference save in this
repository, so every other country starts with an empty voter list and can
not run elections.  This module synthesises a statistically plausible
population for any country:

* every voter is assigned a membership in each voter type with probability
  equal to that type's base percentage (the ``votertypes.csv`` column), and
  a uniform membership weight;
* income-group memberships use the same sinusoidal windows the simulator's
  ``_native_income_group_memberships`` uses, so generated and saved voters
  behave identically;
* political-group memberships come from the native ``ForceVoter`` formulas
  over the voter's initial socialism/liberalism;
* ``_All_`` is 1.0 for everyone; the per-type neuron values start at the CSV
  defaults and a country ``base_value`` applied to ``_All_`` places the mean
  approval where the mission's starting conditions would leave it;
* generation is deterministic (seeded per country), so turn-0 saves for all
  countries are reproducible.

The generated electorate drives the same voter/value/party machinery as a
captured save, so the simulator and the Chronos agent run unmodified.
"""

from __future__ import annotations

import random
from typing import Mapping, Sequence

from . import simulator
from .models import PartyState, SimulationData, SimulationState, Voter

_POLITICAL_GROUP_SYMBOLS = (0, 1, 6, 17)
_INCOME_GROUP_SYMBOLS = (11, 12, 13)
# Ordinary interest groups whose membership is sampled from the CSV base
# percentage (everything except the party-ideology, income, and _All_ groups).
_INTEREST_GROUP_SYMBOLS = (2, 3, 4, 5, 7, 8, 9, 10, 14, 15, 16, 18, 19)
_ALL_SYMBOL = 20

# Generic two-party setup.  The simulator only reads ``party_type`` (0 =
# player, 1 = opposition); names are display-only, so no country-specific
# party data is required.
PLAYER_PARTY = "Government"
OPPOSITION_PARTY = "Opposition"
# Default affiliation split, matching the UK save's proportions
# (1577/2000 unaffiliated, 422/2000 opposition, 1/2000 player).
_UNAFFILIATED_SHARE = 0.79
_PLAYER_SHARE = 0.01


def _voter_type_default(data: SimulationData, name: str) -> float:
    node = data.nodes.get(name)
    return float(node.default) if node is not None else 0.0


def _voter_type_percentage(data: SimulationData, name: str) -> float:
    node = data.nodes.get(f"{name}_freq")
    if node is None:
        return 0.0
    return float(node.initial_percentage or 0.0)


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def generate_electorate(
    data: SimulationData,
    country: str,
    *,
    n: int = 2000,
    seed: int = 0,
    base_value: float = -0.35,
) -> tuple[list[Voter], dict[str, PartyState]]:
    """Build a deterministic electorate and the two-party setup for a country.

    ``base_value`` is the mean approval the starting conditions should leave
    (the UK save averages about -0.4); it is applied to the ``_All_`` neuron
    so every voter shifts uniformly.
    """

    rng = random.Random(f"{country}:{seed}")
    names = list(simulator.VOTER_SYMBOL_NAMES.values())
    percentages = {name: _voter_type_percentage(data, name) for name in names}
    threshold = float(
        data.sim_config.get("VOTER_GROUP_MEMBERSHIP_THRESHHOLD", 0.5)
    )

    parties: dict[str, PartyState] = {
        PLAYER_PARTY: PartyState(
            name=PLAYER_PARTY,
            status=0,
            party_type=0,
            members_last_turn=0,
            member_history=[0] * 10,
            activist_history=[0] * 10,
        ),
        OPPOSITION_PARTY: PartyState(
            name=OPPOSITION_PARTY,
            status=0,
            party_type=1,
            members_last_turn=0,
            member_history=[0] * 10,
            activist_history=[0] * 10,
        ),
    }

    voters: list[Voter] = []
    for _ in range(n):
        voter = Voter()
        # Income is near-uniform over the population (the UK save's inincome
        # standard deviation matches a uniform draw).
        voter.inincome = rng.uniform(0.0, 1.0)
        voter.initial_socialism = _clamp01(rng.triangular(0.0, 1.0, 0.8))
        voter.initial_liberalism = _clamp01(rng.triangular(0.0, 1.0, 0.5))
        voter.militancy = rng.uniform(0.0, 0.6)
        voter.radicalism = rng.uniform(0.0, 1.0)
        voter.voting_tech = rng.uniform(0.0, 1.0)
        voter.survival = rng.randrange(0, 25)
        voter.gender = rng.randrange(2)
        voter.opposition_sympathy = rng.uniform(0.0, 0.3)
        voter.player_sympathy = rng.uniform(0.0, 0.3)

        # Party-ideology groups from the native ForceVoter formulas (global
        # ideologies sit at their 0.5 base before the first pass).
        socialism = 0.5 * (voter.initial_socialism + 0.5)
        liberalism = 0.5 * (voter.initial_liberalism + 0.5)
        voter.groups[0] = socialism
        voter.groups[1] = 1.0 - socialism
        voter.groups[6] = liberalism
        voter.groups[17] = 1.0 - liberalism

        # Income groups via the same sinusoidal windows the simulator uses.
        voter.groups.update(
            simulator._native_income_group_memberships(voter, threshold)
        )

        # Ordinary interest groups: join with the type's base probability.
        for symbol in _INTEREST_GROUP_SYMBOLS:
            name = simulator.VOTER_SYMBOL_NAMES.get(symbol)
            if name is None:
                continue
            if rng.random() < percentages.get(name, 0.0):
                voter.groups[symbol] = rng.uniform(0.0, 1.0)
            else:
                voter.groups[symbol] = 0.0

        # Everyone belongs to _All_.
        voter.groups[_ALL_SYMBOL] = 1.0

        draw = rng.random()
        if draw < _UNAFFILIATED_SHARE:
            voter.party = "0"
        elif draw < _UNAFFILIATED_SHARE + _PLAYER_SHARE:
            voter.party = PLAYER_PARTY
        else:
            voter.party = OPPOSITION_PARTY
        voters.append(voter)

    _init_party_counts(voters, parties)
    return voters, parties


def _init_party_counts(
    voters: Sequence[Voter], parties: Mapping[str, PartyState]
) -> None:
    counts = {name: 0 for name in parties}
    for voter in voters:
        if voter.party in counts:
            counts[voter.party] += 1
    for name, party in parties.items():
        party.members_last_turn = counts[name]
        party.member_history = [counts[name]] * 10


def apply_electorate(
    state: SimulationState,
    data: SimulationData,
    voters: Sequence[Voter],
    parties: Mapping[str, PartyState],
    *,
    base_value: float = -0.35,
) -> None:
    """Install a generated electorate into a fresh state.

    Per-type neuron values start at the CSV defaults; frequencies and
    percentages are the observed membership shares; each voter's value is
    computed with the simulator's own ``_native_voter_value``, then the
    ``_All_`` neuron is shifted so the population mean approval equals
    ``base_value``.
    """

    state.voters = [_copy_voter(voter) for voter in voters]
    state.parties = {name: _copy_party(party) for name, party in parties.items()}
    threshold = float(
        data.sim_config.get("VOTER_GROUP_MEMBERSHIP_THRESHHOLD", 0.5)
    )
    n = len(state.voters)
    for symbol, name in simulator.VOTER_SYMBOL_NAMES.items():
        if name not in data.nodes:
            continue
        state.voter_values[name] = _voter_type_default(data, name)
        count = sum(
            1
            for voter in state.voters
            if voter.groups.get(symbol, 0.0) > 0.0
        )
        share = count / n if n else 0.0
        state.voter_frequencies[f"{name}_freq"] = share
        state.voter_percentages[f"{name}_perc"] = share

    def refresh_values() -> None:
        for voter in state.voters:
            value = simulator._native_voter_value(
                voter, state.voter_values, state.voter_frequencies, threshold
            )
            if value is not None:
                voter.value = value

    refresh_values()
    mean = sum(voter.value for voter in state.voters) / n if n else 0.0
    # _All_ contributes exactly its neuron value to every voter, so shifting
    # it places the whole distribution at the desired mean.
    all_value = max(
        -1.0, min(1.0, state.voter_values.get("_All_", 0.0) + (float(base_value) - mean))
    )
    state.voter_values["_All_"] = all_value
    refresh_values()


def _copy_voter(voter: Voter) -> Voter:
    return Voter(
        groups=dict(voter.groups),
        value=voter.value,
        income=voter.income,
        inincome=voter.inincome,
        militancy=voter.militancy,
        voting_tech=voter.voting_tech,
        initial_socialism=voter.initial_socialism,
        initial_liberalism=voter.initial_liberalism,
        radicalism=voter.radicalism,
        gender=voter.gender,
        opposition_sympathy=voter.opposition_sympathy,
        player_sympathy=voter.player_sympathy,
        last_vote=voter.last_vote,
        survival=voter.survival,
        party=voter.party,
    )


def _copy_party(party: PartyState) -> PartyState:
    return PartyState(
        name=party.name,
        status=party.status,
        party_type=party.party_type,
        members_last_turn=party.members_last_turn,
        member_history=list(party.member_history),
        activist_history=list(party.activist_history),
    )


def generate_country_state(
    country: str,
    *,
    n: int = 2000,
    seed: int = 0,
    base_value: float = -0.35,
) -> SimulationState:
    """Build a fully populated initial state for any country.

    Wraps :func:`autocracy.simulator.get_initial_state` and installs a
    generated electorate, returning a state ready for simulation or for
    serialization into a turn-0 save.
    """

    state, _ = simulator.get_initial_state(country)
    data = simulator.load_simulation_data()
    voters, parties = generate_electorate(
        data, country, n=n, seed=seed, base_value=base_value
    )
    apply_electorate(state, data, voters, parties, base_value=base_value)
    return state


__all__ = [
    "PLAYER_PARTY",
    "OPPOSITION_PARTY",
    "apply_electorate",
    "generate_country_state",
    "generate_electorate",
]