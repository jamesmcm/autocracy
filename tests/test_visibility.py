"""The Chronos agent must never observe the simulation DAG structure.

The model is allowed exactly what a human player sees: ordinary (non-hidden)
simvalue levels, policy slider positions, the budget-screen finance lines,
the published poll rate, and the election countdown.  Everything the
simulator computes from the graph — effect edges, equations, inertia,
situation latents, hidden meta-neurons, voter internals — must stay out of
the feature schema, the forecast input, and the option set the agent ranks.

These tests fail loudly if any future change leaks graph-derived state into
the decision path.
"""

from __future__ import annotations

import pytest

from autocracy import simulator
from autocracy.learning import TreatmentEffectMemory
from autocracy.timeseries import (
    ELECTORAL_SUPPORT_FEATURE,
    AutoregressiveContext,
    FeatureRoleSchema,
    StateFeatureEncoder,
    TimeSeriesPolicyAgent,
    player_visible_value_names,
)

_ALLOWED_PREFIXES = ("value/", "policy/", "finance/", "politics/", "election/")


def _visible_feature_names(state) -> tuple[str, ...]:
    return StateFeatureEncoder.from_visible_state(state).feature_names


def test_visible_schema_excludes_every_dag_derived_signal():
    state, _ = simulator.get_initial_state("uk")
    data = simulator.load_simulation_data()
    names = _visible_feature_names(state)

    # 1. Every value column must be a genuinely player-visible node.
    visible_values = set(player_visible_value_names(state))
    encoded_values = {
        name.split("/", 1)[1] for name in names if name.startswith("value/")
    }
    assert encoded_values == visible_values

    # 2. No hidden / meta / runtime mirror survives in any column.
    for name in names:
        assert not name.startswith("_"), name
        assert not name.split("/", 1)[-1].startswith("_"), name
    hidden_categories = {"HIDDEN", "PLACEHOLDER"}
    hidden_nodes = {
        node.name
        for node in data.nodes.values()
        if node.category in hidden_categories
    }
    for name in names:
        if name.startswith(("value/", "policy/")):
            assert name.split("/", 1)[1] not in hidden_nodes, name

    # 3. Situation latents are computed from the DAG; they must not appear.
    assert not any(
        name.split("/", 1)[1] in data.situations
        for name in names
        if "/" in name
    )

    # 4. Only the curated visible prefixes are permitted.
    for name in names:
        assert name.startswith(_ALLOWED_PREFIXES), name


def test_forecast_input_schema_is_the_visible_schema():
    state, _ = simulator.get_initial_state("uk")
    encoder = StateFeatureEncoder.from_visible_state(state)
    context = AutoregressiveContext.from_state(state, encoder=encoder)

    model_input = context.model_input(horizon=2)

    assert set(model_input.feature_names) == set(encoder.feature_names)
    assert ELECTORAL_SUPPORT_FEATURE in model_input.feature_names
    for row in model_input.history:
        assert len(row) == len(encoder.feature_names)


def test_role_schema_targets_do_not_include_treatments_or_hidden():
    state, _ = simulator.get_initial_state("uk")
    schema = FeatureRoleSchema.from_encoder(
        StateFeatureEncoder.from_visible_state(state)
    )

    treatments = set(schema.treatment_names)
    assert all(name.startswith("policy/") for name in treatments)
    for name in schema.target_names:
        assert name.startswith(_ALLOWED_PREFIXES), name


def test_available_actions_carry_no_graph_metadata():
    state, _ = simulator.get_initial_state("uk")
    options = simulator.list_available_actions(state)

    allowed_fields = {
        "policy_name",
        "action_type",
        "delta",
        "resulting_level",
        "cost",
        "implementation_time",
        "financial_delta",
    }
    for option in options:
        # Every exposed field is player-visible UI arithmetic; no effect ids,
        # no inertia, no graph edge identifiers.
        assert set(option.__dataclass_fields__) == allowed_fields


def test_agent_context_stays_visible_across_turns():
    state, graph = simulator.get_initial_state("uk")
    agent = TimeSeriesPolicyAgent(
        _FlatForecaster(),
        visible_features_only=True,
        treatment_memory=TreatmentEffectMemory(),
    )
    agent.state = state
    agent.graph = graph

    agent.step()
    agent.step()

    for snapshot in agent.context.states:
        for name in snapshot.features:
            assert name.startswith(_ALLOWED_PREFIXES), name
    # The memory must only ever receive visible features.
    visible = set(_visible_feature_names(agent.state))
    for decision in agent.decisions:
        assert set(decision.forecast.feature_names) == visible


class _FlatForecaster:
    """Persistence-like forecaster returning the latest row unchanged."""

    name = "flat"

    def predict(self, model_input):
        from autocracy.timeseries import StateForecast

        row = dict(zip(model_input.feature_names, model_input.history[-1]))
        return StateForecast.from_rows(
            model_input,
            [dict(row) for _ in range(model_input.horizon)],
            model_name=self.name,
        )