from __future__ import annotations

from autocracy import data_loader
from autocracy import simulator


def test_load_simulation_nodes_includes_inputs_and_outputs():
    nodes, effects = data_loader.load_simulation_nodes(simulator.DEFAULT_GAMEDATA)
    assert "CarUsage" in nodes
    inbound = [
        effect for effect in effects if effect.source == "GDP" and effect.target == "CarUsage"
    ]
    outbound = [
        effect
        for effect in effects
        if effect.source == "CarUsage" and effect.target == "Environment"
    ]
    assert inbound, "Expected GDP to influence CarUsage"
    assert outbound, "Expected CarUsage to influence Environment"
    assert inbound[0].expression == "0+(0.4 * x)"
    assert outbound[0].expression == "0-(0.22*x)"
