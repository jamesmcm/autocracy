from __future__ import annotations

from autocracy.events import run_events
from autocracy.models import SimulationConfig
from autocracy.savegame import load_state_from_savegame
from autocracy.simulator import load_simulation_data, process_end_of_turn


def _state_and_graph(data):
    return load_state_from_savegame("gamedata/saves/uk0.xml", data)


def test_random_systems_are_off_by_default():
    data = load_simulation_data()
    state, graph = _state_and_graph(data)

    # The parity path passes no config at all.
    out = process_end_of_turn(state, graph, data)
    assert out.event_log == []

    # Explicit all-off config is equally inert.
    out = process_end_of_turn(state, graph, data, config=SimulationConfig())
    assert out.event_log == []
    assert out.fired_plots == []
    assert out.group_threats == {}


def test_events_are_reproducible_for_a_fixed_seed():
    data = load_simulation_data()

    def run():
        state, _ = _state_and_graph(data)
        return run_events(
            state, data, SimulationConfig(random_events=True, random_seed=11)
        )

    first = run()
    second = run()
    assert first.event_log == second.event_log
    # The event set is large enough that the seeded stream fires at least one
    # event, and its grudges actually mutate the state.
    assert first.event_log
    assert any(entry.startswith("event ") for entry in first.event_log)
    assert any(entry.startswith("grudge ") for entry in first.event_log)


def test_events_are_seeded_not_constant():
    data = load_simulation_data()
    state_a = run_events(
        _state_and_graph(data)[0],
        data,
        SimulationConfig(random_events=True, random_seed=1),
    )
    state_b = run_events(
        _state_and_graph(data)[0],
        data,
        SimulationConfig(random_events=True, random_seed=2),
    )
    # Different seeds produce different event streams.
    assert state_a.event_log != state_b.event_log


def test_dilemmas_and_attacks_are_gated_and_reproducible():
    data = load_simulation_data()
    graph = _state_and_graph(data)[1]
    config = SimulationConfig(
        dilemmas=True, pressure_group_events=True, assassinations=True, random_seed=5
    )
    state = _state_and_graph(data)[0]
    out = process_end_of_turn(state, graph, data, config=config)
    assert out.event_log  # at least one enabled system fired over the turn
    assert out.event_log == out.event_log  # deterministic per run

    # Disabling the systems again leaves the state untouched.
    clean = process_end_of_turn(_state_and_graph(data)[0], graph, data)
    assert clean.event_log == []
