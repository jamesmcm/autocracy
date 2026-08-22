"""GPU-backed Chronos-2 forecasting for the action-conditioned agent.

The forecaster treats every observed feature as a multivariate target and
every ``policy/`` column as a known-future covariate (a treatment variable
the player controls).  At decision time each candidate action produces a
different future policy path, so all candidates are forecast jointly in one
batched :meth:`Chronos2Pipeline.predict_df` call as independent items.

The heavy dependencies (``torch``, ``chronos-forecasting``, ``pandas``) are
imported lazily so the base package stays installable without them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any, Sequence

from .timeseries import (
    ForecastModelInput,
    StateForecast,
    TREATMENT_FEATURE_PREFIX,
)

DEFAULT_CHRONOS2_SMALL_MODEL = "autogluon/chronos-2-small"
DEFAULT_QUANTILE_LEVEL = 0.5
_TIMESTAMP_ORIGIN = "2000-01-01"


def _require_pandas() -> Any:
    try:
        import pandas
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Chronos2SmallForecaster requires pandas; install the 'chronos' "
            "extra (uv sync --extra chronos)"
        ) from error
    return pandas


def projected_policy_paths(
    model_input: ForecastModelInput,
) -> tuple[dict[str, float], ...]:
    """Return the known future treatment path for each forecast step.

    The path starts from the latest observed policy levels; a pending action
    moves its slider once at the first future step and the level then
    persists, matching the simulator's persistent-slider semantics.
    """

    last_row = dict(zip(model_input.feature_names, model_input.history[-1]))
    levels = {
        name: float(value)
        for name, value in last_row.items()
        if name.startswith(TREATMENT_FEATURE_PREFIX)
    }
    pending = {
        f"{TREATMENT_FEATURE_PREFIX}{action.policy_name}": float(action.delta)
        for action in model_input.pending_actions
    }
    paths: list[dict[str, float]] = []
    for step in range(model_input.horizon):
        if step == 0:
            for name, delta in pending.items():
                if name in levels:
                    levels[name] = min(1.0, max(0.0, levels[name] + delta))
        paths.append(dict(levels))
    return paths


def _turn_timestamps(turns: Sequence[int], pandas: Any) -> Any:
    return pandas.to_datetime(
        list(turns), unit="D", origin=_TIMESTAMP_ORIGIN
    )


def chronos_frames(
    inputs: Sequence[ForecastModelInput],
    *,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> tuple[Any, Any, tuple[str, ...], tuple[str, ...]]:
    """Build the context and known-future frames for a candidate batch.

    Every candidate becomes its own item so the pipeline processes the tasks
    together without cross-learning between candidates.
    """

    pandas = _require_pandas()
    if not inputs:
        raise ValueError("chronos frames need at least one model input")
    feature_names = inputs[0].feature_names
    horizon = inputs[0].horizon
    for item in inputs[1:]:
        if item.feature_names != feature_names or item.horizon != horizon:
            raise ValueError(
                "batched model inputs must share one schema and horizon"
            )
    treatment_names = tuple(
        name
        for name in feature_names
        if name.startswith(TREATMENT_FEATURE_PREFIX)
    )
    target_names = tuple(
        name
        for name in feature_names
        if not name.startswith(TREATMENT_FEATURE_PREFIX)
    )
    context_rows: list[dict[str, object]] = []
    future_rows: list[dict[str, object]] = []
    for index, item in enumerate(inputs):
        item_id = f"candidate-{index}"
        stamps = _turn_timestamps(item.turns, pandas)
        for stamp, row_values in zip(stamps, item.history):
            row: dict[str, object] = {id_column: item_id, timestamp_column: stamp}
            row.update(zip(feature_names, row_values))
            context_rows.append(row)
        future_stamps = _turn_timestamps(
            range(item.turns[-1] + 1, item.turns[-1] + 1 + horizon), pandas
        )
        for stamp, path in zip(future_stamps, projected_policy_paths(item)):
            future_row: dict[str, object] = {
                id_column: item_id,
                timestamp_column: stamp,
            }
            for name in treatment_names:
                future_row[name] = path.get(name, 0.0)
            future_rows.append(future_row)
    return pandas.DataFrame(context_rows), pandas.DataFrame(future_rows), target_names, treatment_names


@dataclass(slots=True)
class Chronos2SmallForecaster:
    """Action-conditioned world model backed by ``autogluon/chronos-2-small``.

    All observed features are forecast jointly while the policy sliders are
    supplied as known-future covariates, so a candidate's predictions respond
    to the treatment path it implies.  Point forecasts use the median by
    default.
    """

    model_name: str = DEFAULT_CHRONOS2_SMALL_MODEL
    device_map: str | None = None
    torch_dtype: str = "float32"
    quantile_level: float = DEFAULT_QUANTILE_LEVEL
    batch_size: int = 256
    max_context_rows: int | None = None
    pipeline_kwargs: dict[str, Any] = field(default_factory=dict)
    name: str = field(init=False, default="chronos-2-small")
    _pipeline: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if not math.isfinite(self.quantile_level):
            raise ValueError("quantile_level must be finite")

    def _ensure_pipeline(self) -> Any:
        if self._pipeline is None:
            try:
                import torch
                from chronos import Chronos2Pipeline
            except ImportError as error:  # pragma: no cover - environment dependent
                raise RuntimeError(
                    "Chronos2SmallForecaster needs the 'chronos' extra "
                    "(uv sync --extra chronos)"
                ) from error
            device = self.device_map or (
                "cuda" if torch.cuda.is_available() else "cpu"
            )
            self._pipeline = Chronos2Pipeline.from_pretrained(
                self.model_name,
                device_map=device,
                dtype=getattr(torch, self.torch_dtype),
                **self.pipeline_kwargs,
            )
        return self._pipeline

    def predict(self, model_input: ForecastModelInput) -> StateForecast:
        return self.predict_batch([model_input])[0]

    def predict_batch(
        self, inputs: Sequence[ForecastModelInput]
    ) -> list[StateForecast]:
        if not inputs:
            return []
        context_frame, future_frame, target_names, _ = chronos_frames(inputs)
        if self.max_context_rows is not None:
            context_frame = context_frame.groupby("item_id", sort=False).tail(
                self.max_context_rows
            )
        pipeline = self._ensure_pipeline()
        prediction_frame = pipeline.predict_df(
            context_frame,
            future_df=future_frame,
            prediction_length=inputs[0].horizon,
            quantile_levels=[self.quantile_level],
            id_column="item_id",
            timestamp_column="timestamp",
            target=list(target_names),
            freq="D",
            batch_size=self.batch_size,
        )
        return _forecasts_from_prediction_frame(
            inputs, prediction_frame, target_names
        )


def _forecasts_from_prediction_frame(
    inputs: Sequence[ForecastModelInput],
    prediction_frame: Any,
    target_names: Sequence[str],
) -> list[StateForecast]:
    """Map the long prediction frame back into per-candidate forecasts."""

    if "predictions" not in prediction_frame.columns:
        raise RuntimeError(
            "chronos prediction frame lacks a 'predictions' column"
        )
    steps = sorted(set(prediction_frame["timestamp"]))
    step_index = {stamp: step for step, stamp in enumerate(steps)}
    expected_points = len(tuple(target_names)) * (
        inputs[0].horizon if inputs else 0
    )
    grouped: dict[str, dict[tuple[str, int], float]] = {}
    for row in prediction_frame.to_dict("records"):
        key = (str(row["target_name"]), step_index[row["timestamp"]])
        grouped.setdefault(str(row["item_id"]), {})[key] = float(
            row["predictions"]
        )
    forecasts: list[StateForecast] = []
    for index, model_input in enumerate(inputs):
        item_grouped = grouped.get(f"candidate-{index}")
        if item_grouped is None or len(item_grouped) != expected_points:
            raise RuntimeError(
                f"chronos prediction frame is incomplete for candidate-{index}"
            )
        known_paths = projected_policy_paths(model_input)
        rows: list[dict[str, float]] = []
        for step in range(model_input.horizon):
            row = {
                name: item_grouped[(name, step)]
                for name in target_names
            }
            # Known-future treatment columns are never forecast by the
            # model; restore the deterministic path so scoring sees a full
            # feature row.
            for name, value in known_paths[step].items():
                row.setdefault(name, value)
            rows.append(row)
        forecasts.append(
            StateForecast.from_rows(
                model_input, rows, model_name="chronos-2-small"
            )
        )
    return forecasts


__all__ = [
    "Chronos2SmallForecaster",
    "DEFAULT_CHRONOS2_SMALL_MODEL",
    "chronos_frames",
    "projected_policy_paths",
]
