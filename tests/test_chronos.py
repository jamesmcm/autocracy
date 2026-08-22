from __future__ import annotations

import pytest

from autocracy import simulator
from autocracy.chronos import (
    Chronos2SmallForecaster,
    chronos_frames,
    projected_policy_paths,
)
from autocracy.models import PolicyAction
from autocracy.timeseries import (
    AutoregressiveContext,
    ELECTORAL_SUPPORT_FEATURE,
    FeatureRoleSchema,
    StateFeatureEncoder,
    StateForecast,
    TimeSeriesPolicyAgent,
    player_visible_value_names,
)

pytest.importorskip("pandas", reason="chronos extra requires pandas")


def _visible_encoder(state):
    return StateFeatureEncoder.from_visible_state(state)


def test_savegame_histories_seed_the_initial_context():
    state, _ = simulator.get_initial_state("uk")

    assert state.value_histories, "uk0.xml must provide pre-game rings"
    assert state.value_histories_turn == state.turn == 0
    context = AutoregressiveContext.from_state(
        state,
        encoder=_visible_encoder(state),
        include_value_histories=True,
    )

    turns = [snapshot.turn for snapshot in context.states]
    assert len(turns) > 1
    assert turns == sorted(set(turns))
    assert context.current_turn == 0
    gdp_ring = list(reversed(state.value_histories["GDP"]))
    expected = [
        value for value in gdp_ring if value != 0.0
    ]
    observed = [
        snapshot.features["value/GDP"]
        for snapshot in context.states
    ]
    assert observed[-1] == pytest.approx(expected[-1])
    # Placeholder zeros are carried forward, never forecast as zeros.
    for latest, previous in zip(observed[1:], observed[:-1]):
        if latest == 0.0:
            assert previous != 0.0


def test_seeded_poll_history_matches_electoral_support_target():
    state, _ = simulator.get_initial_state("uk")
    context = AutoregressiveContext.from_state(
        state,
        encoder=_visible_encoder(state),
        include_value_histories=True,
    )

    poll_rows = [
        snapshot.features[ELECTORAL_SUPPORT_FEATURE]
        for snapshot in context.states
    ]
    ring = list(reversed(state.poll_history))
    assert poll_rows[-len(ring):] == pytest.approx(ring)
    # Rows older than the poll ring hold the current observation.
    backfilled = poll_rows[: -len(ring)]
    assert backfilled == pytest.approx([state.poll_rate] * len(backfilled))


def test_player_visible_schema_excludes_hidden_meta_nodes():
    state, _ = simulator.get_initial_state("uk")
    names = player_visible_value_names(state)

    assert "GDP" in names
    assert "_globaleconomy_" not in names
    assert "_year" not in names
    assert "_Terrorism" not in names
    assert "_LowIncome" not in names
    assert "Obesity" not in names
    assert "Motorist_income" not in names
    # Derived runtime mirrors are not game state a player reads.
    assert not any(name.endswith("_perc") for name in names)
    encoder = _visible_encoder(state)
    feature_names = encoder.feature_names
    assert "finance/debt" in feature_names
    # Only budget-screen figures and displayed politics fields survive.
    assert not any(name.startswith("finance/political_capital_income") for name in feature_names)
    assert not any(name.startswith("politics/peak_poll_rate") for name in feature_names)
    assert not any(name.startswith("election/current_term") for name in feature_names)
    assert all(not name.split("/", 1)[-1].startswith("_") for name in feature_names)


def test_role_schema_splits_treatments_from_predicted_targets():
    state, _ = simulator.get_initial_state("uk")
    schema = FeatureRoleSchema.from_encoder(_visible_encoder(state))

    assert ELECTORAL_SUPPORT_FEATURE in schema.target_names
    assert schema.target_names[0] == ELECTORAL_SUPPORT_FEATURE
    assert "policy/IncomeTax" in schema.treatment_names
    assert "value/GDP" in schema.target_names
    assert "finance/debt" in schema.target_names
    assert not (set(schema.target_names) & set(schema.treatment_names))
    assert set(schema.target_names) | set(schema.treatment_names) == set(
        schema.feature_names
    )


def test_projected_policy_paths_persist_pending_action():
    base = simulator.get_initial_state("uk")[0]
    enc = StateFeatureEncoder.from_visible_state(
        base,
        include_finance=False,
        include_election=False,
    )
    context = AutoregressiveContext.from_state(base, encoder=enc)
    action = PolicyAction("IncomeTax", 0.05, "raise")
    model_input = context.model_input([action], horizon=3)

    paths = projected_policy_paths(model_input)

    assert len(paths) == 3
    start = model_input.history[-1][model_input.feature_names.index("policy/IncomeTax")]
    assert paths[0]["policy/IncomeTax"] == pytest.approx(start + 0.05)
    assert paths[1]["policy/IncomeTax"] == pytest.approx(start + 0.05)
    assert paths[2]["policy/IncomeTax"] == pytest.approx(start + 0.05)


def test_chronos_frames_split_targets_and_known_future_covariates():
    pandas = pytest.importorskip("pandas")
    base = simulator.get_initial_state("uk")[0]
    enc = StateFeatureEncoder.from_visible_state(
        base,
        include_finance=False,
        include_election=False,
    )
    context = AutoregressiveContext.from_state(base, encoder=enc)
    action = PolicyAction("IncomeTax", 0.05, "raise")
    inputs = [
        context.model_input((), horizon=2),
        context.model_input([action], horizon=2),
    ]

    context_frame, future_frame, targets, treatments = chronos_frames(inputs)

    assert "value/GDP" in targets
    assert "policy/IncomeTax" in treatments
    assert "value/GDP" not in future_frame.columns
    assert set(treatments).issubset(future_frame.columns)
    item_ids = sorted(context_frame["item_id"].unique())
    assert item_ids == ["candidate-0", "candidate-1"]
    candidate_1 = future_frame[future_frame["item_id"] == "candidate-1"]
    start = inputs[1].history[-1][inputs[1].feature_names.index("policy/IncomeTax")]
    assert candidate_1["policy/IncomeTax"].tolist() == pytest.approx(
        [start + 0.05, start + 0.05]
    )
    candidate_0 = future_frame[future_frame["item_id"] == "candidate-0"]
    assert candidate_0["policy/IncomeTax"].nunique() == 1
    stamps = pandas.to_datetime(candidate_1["timestamp"])
    assert (stamps.diff().dropna() == pandas.Timedelta(days=1)).all()


def test_forecast_mapping_restores_treatments_and_orders_steps():
    base = simulator.get_initial_state("uk")[0]
    enc = StateFeatureEncoder.from_visible_state(
        base,
        include_finance=False,
        include_election=False,
    )
    context = AutoregressiveContext.from_state(base, encoder=enc)
    action = PolicyAction("IncomeTax", 0.05, "raise")
    model_input = context.model_input([action], horizon=3)

    forecaster = Chronos2SmallForecaster()
    forecaster._pipeline = _FakePipeline(model_input)

    forecast = forecaster.predict(model_input)

    assert forecast.horizon == 3
    assert forecast.model_name == "chronos-2-small"
    for step, row in enumerate(forecast.values):
        expected_target = base.values["GDP"] + step
        assert row["value/GDP"] == pytest.approx(expected_target)
        start = model_input.history[-1][
            model_input.feature_names.index("policy/IncomeTax")
        ]
        assert row["policy/IncomeTax"] == pytest.approx(start + 0.05)


class _FakePipeline:
    """Minimal predict_df stand-in returning deterministic rows."""

    def __init__(self, template):
        self._template = template

    def predict_df(self, context_frame, **kwargs):
        pandas = __import__("pandas")
        target_columns = kwargs["target"]
        prediction_length = kwargs["prediction_length"]
        records = []
        stamps = sorted(context_frame["timestamp"].unique())
        future_stamps = [max(stamps) + pandas.Timedelta(days=i + 1) for i in range(prediction_length)]
        for item_id in sorted(context_frame["item_id"].unique()):
            last = (
                context_frame[context_frame["item_id"] == item_id]
                .sort_values("timestamp")
                .iloc[-1]
            )
            for step, stamp in enumerate(future_stamps):
                for name in target_columns:
                    records.append(
                        {
                            "item_id": item_id,
                            "timestamp": stamp,
                            "target_name": name,
                            "predictions": float(last[name]) + step
                            if name == "value/GDP"
                            else float(last[name]),
                        }
                    )
        return pandas.DataFrame(records)


def test_agent_uses_batched_prediction_and_seeded_context():
    class BatchCountingForecaster:
        name = "batch-counter"

        def __init__(self) -> None:
            self.batch_calls = 0
            self.single_calls = 0

        def predict(self, model_input):
            self.single_calls += 1
            row = dict(zip(model_input.feature_names, model_input.history[-1]))
            return StateForecast.from_rows(
                model_input, [row] * model_input.horizon, model_name=self.name
            )

        def predict_batch(self, inputs):
            self.batch_calls += 1
            return [self.predict(item) for item in inputs]

    forecaster = BatchCountingForecaster()
    agent = TimeSeriesPolicyAgent(
        forecaster,
        forecast_horizon=2,
        candidate_limit=3,
        random_seed=5,
        visible_features_only=True,
        seed_pre_game_history=True,
        objective=lambda features: features[ELECTORAL_SUPPORT_FEATURE],
    )

    assert len(agent.context.states) > 1
    agent.step()

    assert forecaster.batch_calls == 1
    assert forecaster.single_calls == agent.decisions[0].candidate_count
    assert agent.context.actions[0] == ()
    assert agent.state.turn == 1
