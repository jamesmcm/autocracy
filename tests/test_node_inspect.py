from __future__ import annotations

from autocracy import simulator


def test_collect_node_effects_for_health():
    data = simulator.load_simulation_data()
    graph = simulator.build_country_graph("uk")
    inputs, outputs = simulator.collect_node_effects("Health", graph, data=data)
    assert any(
        effect.source == "AgricultureSubsidies"
        and effect.target == "Health"
        and effect.expression == "0-(0.04*x)"
        for effect in inputs
    )
    assert any(effect.source == "AlcoholConsumption" and effect.target == "Health" for effect in inputs)
    assert any(effect.source == "Alcoholism" and effect.target == "Health" for effect in inputs)
    assert any(effect.target == "WorkerProductivity" for effect in outputs)
    assert any(effect.target == "Immigration" for effect in outputs)
