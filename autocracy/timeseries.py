"""Autoregressive time-series experiment plumbing.

The simulator state is much richer than a single scalar time series.  This
module gives forecasting experiments a stable, dependency-free boundary:

* :class:`StateFeatureEncoder` turns a live state into a fixed feature row;
* :class:`AutoregressiveContext` records observed state/action transitions;
* :class:`ActionConditionedForecaster` receives the complete history plus a
  pending action batch and returns future feature rows; and
* :class:`TimeSeriesPolicyAgent` uses those forecasts to choose actions, then
  appends the real post-action state to the context.

The CPU baselines are useful for smoke tests and calibration.  The
``Chronos2Forecaster`` is deliberately an injected-backend adapter: it does
not import torch or download a model on this CPU-only host, but its input and
output contract is the one a later GPU Chronos2 runner can implement.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import json
import math
from pathlib import Path
import random
from typing import Callable, Mapping, Protocol, Sequence, TypeAlias

from .agent import BaseAgent
from .models import PolicyAction, SimulationConfig, SimulationState


FeatureRow: TypeAlias = Mapping[str, float]


def _finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"feature {name!r} is not finite: {value!r}")
    return number


@dataclass(frozen=True, slots=True)
class StateFeatureEncoder:
    """Encode selected simulator fields into a stable multivariate row.

    ``value_names`` and ``policy_names`` are fixed when the context is
    created.  This prevents a later state from silently changing the model's
    column order.  ``from_state`` is the usual constructor and captures all
    ordinary value and policy names by default; callers can pass a smaller
    list for a compact foundation-model experiment.
    """

    value_names: tuple[str, ...] = ()
    policy_names: tuple[str, ...] = ()
    include_finance: bool = True
    include_election: bool = True

    @classmethod
    def from_state(
        cls,
        state: SimulationState,
        *,
        value_names: Sequence[str] | None = None,
        policy_names: Sequence[str] | None = None,
        include_finance: bool = True,
        include_election: bool = True,
    ) -> "StateFeatureEncoder":
        return cls(
            value_names=tuple(
                sorted(value_names if value_names is not None else state.values)
            ),
            policy_names=tuple(
                sorted(policy_names if policy_names is not None else state.policies)
            ),
            include_finance=include_finance,
            include_election=include_election,
        )

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [f"value/{name}" for name in self.value_names]
        names.extend(f"policy/{name}" for name in self.policy_names)
        if self.include_finance:
            names.extend(
                (
                    "finance/political_capital",
                    "finance/political_capital_income",
                    "finance/total_expenditure",
                    "finance/total_income",
                    "finance/debt",
                    "finance/interest_rate",
                )
            )
        if self.include_election:
            names.extend(
                (
                    "politics/poll_rate",
                    "politics/peak_poll_rate",
                    "politics/active_situations",
                    "election/turns_until",
                    "election/current_term",
                    "election/result",
                    "election/last_winner",
                )
            )
        return tuple(names)

    def encode(self, state: SimulationState) -> dict[str, float]:
        """Return one fixed-order-compatible feature row for ``state``."""

        row: dict[str, float] = {}
        for name in self.value_names:
            row[f"value/{name}"] = _finite(
                state.values.get(name, 0.0), name=f"value/{name}"
            )
        for name in self.policy_names:
            row[f"policy/{name}"] = _finite(
                state.policies.get(name, 0.0), name=f"policy/{name}"
            )
        if self.include_finance:
            finance = {
                "finance/political_capital": state.political_capital,
                "finance/political_capital_income": state.political_capital_income,
                "finance/total_expenditure": state.total_expenditure,
                "finance/total_income": state.total_income,
                "finance/debt": state.debt,
                "finance/interest_rate": state.interest_rate,
            }
            row.update(
                {
                    name: _finite(value, name=name)
                    for name, value in finance.items()
                }
            )
        if self.include_election:
            result_value = {None: 0.0, "loss": -1.0, "win": 1.0}.get(
                state.election_result, 0.0
            )
            winner_value = {
                None: 0.0,
                "opposition": -1.0,
                "player": 1.0,
            }.get(state.last_election_winner, 0.0)
            politics = {
                "politics/poll_rate": state.poll_rate,
                "politics/peak_poll_rate": state.peak_poll_rate,
                "politics/active_situations": float(len(state.active_situations)),
                "election/turns_until": float(state.election_turns_until),
                "election/current_term": float(state.election_current_term),
                "election/result": result_value,
                "election/last_winner": winner_value,
            }
            row.update(
                {
                    name: _finite(value, name=name)
                    for name, value in politics.items()
                }
            )
        return row

    def to_dict(self) -> dict[str, object]:
        return {
            "value_names": list(self.value_names),
            "policy_names": list(self.policy_names),
            "include_finance": self.include_finance,
            "include_election": self.include_election,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateFeatureEncoder":
        return cls(
            value_names=tuple(str(name) for name in payload.get("value_names", [])),
            policy_names=tuple(str(name) for name in payload.get("policy_names", [])),
            include_finance=bool(payload.get("include_finance", True)),
            include_election=bool(payload.get("include_election", True)),
        )


@dataclass(frozen=True, slots=True)
class ActionRecord:
    """Serializable representation of a policy action in the context."""

    policy_name: str
    delta: float
    action_type: str | None = None

    @classmethod
    def from_action(cls, action: PolicyAction) -> "ActionRecord":
        return cls(
            policy_name=action.policy_name,
            delta=float(action.delta),
            action_type=action.action_type,
        )

    def to_action(self) -> PolicyAction:
        return PolicyAction(
            policy_name=self.policy_name,
            delta=self.delta,
            action_type=self.action_type,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "policy_name": self.policy_name,
            "delta": self.delta,
            "action_type": self.action_type,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "ActionRecord":
        return cls(
            policy_name=str(payload["policy_name"]),
            delta=float(payload["delta"]),
            action_type=(
                str(payload["action_type"])
                if payload.get("action_type") is not None
                else None
            ),
        )


def _records(actions: Sequence[PolicyAction | ActionRecord]) -> tuple[ActionRecord, ...]:
    return tuple(
        action
        if isinstance(action, ActionRecord)
        else ActionRecord.from_action(action)
        for action in actions
    )


@dataclass(frozen=True, slots=True)
class StateSnapshot:
    """One observed feature row at a simulator turn."""

    turn: int
    features: Mapping[str, float]

    @classmethod
    def from_state(
        cls, state: SimulationState, encoder: StateFeatureEncoder
    ) -> "StateSnapshot":
        return cls(turn=state.turn, features=encoder.encode(state))

    def row(self, feature_names: Sequence[str]) -> tuple[float, ...]:
        return tuple(
            _finite(self.features.get(name, 0.0), name=name)
            for name in feature_names
        )

    def to_dict(self) -> dict[str, object]:
        return {"turn": self.turn, "features": dict(self.features)}

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateSnapshot":
        return cls(
            turn=int(payload["turn"]),
            features={
                str(name): float(value)
                for name, value in dict(payload["features"]).items()
            },
        )


@dataclass(frozen=True, slots=True)
class ForecastModelInput:
    """Plain-Python model input suitable for a local or remote backend."""

    feature_names: tuple[str, ...]
    history: tuple[tuple[float, ...], ...]
    turns: tuple[int, ...]
    action_history: tuple[tuple[ActionRecord, ...], ...]
    pending_actions: tuple[ActionRecord, ...]
    horizon: int

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "history": [list(row) for row in self.history],
            "turns": list(self.turns),
            "action_history": [
                [action.to_dict() for action in actions]
                for actions in self.action_history
            ],
            "pending_actions": [
                action.to_dict() for action in self.pending_actions
            ],
            "horizon": self.horizon,
        }


@dataclass(frozen=True, slots=True)
class AutoregressiveContext:
    """Observed state/action history used by an action-conditioned forecaster."""

    encoder: StateFeatureEncoder
    states: tuple[StateSnapshot, ...]
    actions: tuple[tuple[ActionRecord, ...], ...] = ()

    def __post_init__(self) -> None:
        if not self.states:
            raise ValueError("an autoregressive context needs an initial state")
        if len(self.actions) != len(self.states) - 1:
            raise ValueError("actions must contain one entry per observed transition")
        turns = [snapshot.turn for snapshot in self.states]
        if turns != sorted(turns) or len(set(turns)) != len(turns):
            raise ValueError("context state turns must increase strictly")

    @classmethod
    def from_state(
        cls,
        state: SimulationState,
        *,
        encoder: StateFeatureEncoder | None = None,
    ) -> "AutoregressiveContext":
        schema = encoder or StateFeatureEncoder.from_state(state)
        return cls(schema, (StateSnapshot.from_state(state, schema),))

    @property
    def current_turn(self) -> int:
        return self.states[-1].turn

    @property
    def feature_names(self) -> tuple[str, ...]:
        return self.encoder.feature_names

    def append_transition(
        self,
        before: SimulationState,
        actions: Sequence[PolicyAction | ActionRecord],
        after: SimulationState,
    ) -> "AutoregressiveContext":
        """Append the observed result of one real action/turn transition."""

        if before.turn != self.current_turn:
            raise ValueError(
                f"transition starts at turn {before.turn}, "
                f"but context ends at turn {self.current_turn}"
            )
        if after.turn <= before.turn:
            raise ValueError("a transition must advance the simulator turn")
        return replace(
            self,
            states=(*self.states, StateSnapshot.from_state(after, self.encoder)),
            actions=(*self.actions, _records(actions)),
        )

    def model_input(
        self,
        actions: Sequence[PolicyAction | ActionRecord] = (),
        *,
        horizon: int = 1,
    ) -> ForecastModelInput:
        """Build a model-neutral input with a pending action batch."""

        if horizon < 1:
            raise ValueError("forecast horizon must be at least one")
        return ForecastModelInput(
            feature_names=self.feature_names,
            history=tuple(
                snapshot.row(self.feature_names) for snapshot in self.states
            ),
            turns=tuple(snapshot.turn for snapshot in self.states),
            action_history=self.actions,
            pending_actions=_records(actions),
            horizon=horizon,
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "autocracy-timeseries-context-v1",
            "encoder": self.encoder.to_dict(),
            "states": [state.to_dict() for state in self.states],
            "actions": [
                [action.to_dict() for action in transition]
                for transition in self.actions
            ],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "AutoregressiveContext":
        states = tuple(
            StateSnapshot.from_dict(item)
            for item in payload.get("states", [])
        )
        actions = tuple(
            tuple(ActionRecord.from_dict(action) for action in transition)
            for transition in payload.get("actions", [])
        )
        return cls(
            encoder=StateFeatureEncoder.from_dict(payload["encoder"]),
            states=states,
            actions=actions,
        )

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "AutoregressiveContext":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


@dataclass(frozen=True, slots=True)
class StateForecast:
    """Future feature rows returned by a forecaster."""

    feature_names: tuple[str, ...]
    values: tuple[Mapping[str, float], ...]
    origin_turn: int
    pending_actions: tuple[ActionRecord, ...] = ()
    model_name: str = ""

    @classmethod
    def from_rows(
        cls,
        model_input: ForecastModelInput,
        rows: Sequence[Mapping[str, float]],
        *,
        model_name: str = "",
    ) -> "StateForecast":
        if not rows:
            raise ValueError("a forecaster must return at least one future row")
        values = tuple(
            {
                name: _finite(row.get(name, 0.0), name=name)
                for name in model_input.feature_names
            }
            for row in rows
        )
        return cls(
            feature_names=model_input.feature_names,
            values=values,
            origin_turn=model_input.turns[-1],
            pending_actions=model_input.pending_actions,
            model_name=model_name,
        )

    @property
    def horizon(self) -> int:
        return len(self.values)

    @property
    def first(self) -> Mapping[str, float]:
        return self.values[0]

    @property
    def final(self) -> Mapping[str, float]:
        return self.values[-1]

    def to_dict(self) -> dict[str, object]:
        return {
            "feature_names": list(self.feature_names),
            "values": [dict(row) for row in self.values],
            "origin_turn": self.origin_turn,
            "pending_actions": [
                action.to_dict() for action in self.pending_actions
            ],
            "model_name": self.model_name,
        }


class ActionConditionedForecaster(Protocol):
    """Protocol implemented by CPU baselines and future GPU model adapters."""

    name: str

    def predict(self, model_input: ForecastModelInput) -> StateForecast:
        """Predict ``model_input.horizon`` rows after its pending actions."""


@dataclass(slots=True)
class PersistenceForecaster:
    """Dependency-free baseline that repeats the latest observed row."""

    name: str = field(init=False, default="persistence")

    def predict(self, model_input: ForecastModelInput) -> StateForecast:
        last = dict(zip(model_input.feature_names, model_input.history[-1]))
        rows = [dict(last) for _ in range(model_input.horizon)]
        return StateForecast.from_rows(model_input, rows, model_name=self.name)


@dataclass(slots=True)
class EmpiricalActionForecaster:
    """Small CPU baseline that learns recent action-conditioned deltas.

    This is not intended to replace Chronos2.  It gives the experiment a
    reproducible reference on the VPS and exercises the same autoregressive
    interface.  The first predicted step uses the mean observed delta for the
    requested policy/action type; later steps use the recent unconditional
    drift, so each predicted row becomes the input to the next one.
    """

    recent_window: int = 8
    name: str = field(init=False, default="empirical-action-delta")

    def __post_init__(self) -> None:
        if self.recent_window < 1:
            raise ValueError("recent_window must be at least one")

    @staticmethod
    def _mean(rows: Sequence[Mapping[str, float]], names: Sequence[str]) -> dict[str, float]:
        if not rows:
            return {name: 0.0 for name in names}
        return {
            name: sum(float(row.get(name, 0.0)) for row in rows) / len(rows)
            for name in names
        }

    @staticmethod
    def _action_key(action: ActionRecord) -> tuple[str, str]:
        return action.policy_name, action.action_type or ""

    def predict(self, model_input: ForecastModelInput) -> StateForecast:
        names = model_input.feature_names
        rows = [
            dict(zip(names, current))
            for current in model_input.history
        ]
        deltas: list[dict[str, float]] = []
        action_deltas: dict[tuple[str, str], list[dict[str, float]]] = {}
        for index, transition_actions in enumerate(model_input.action_history):
            delta = {
                name: rows[index + 1][name] - rows[index][name]
                for name in names
            }
            deltas.append(delta)
            for action in transition_actions:
                action_deltas.setdefault(self._action_key(action), []).append(delta)
        recent = deltas[-self.recent_window :]
        baseline = self._mean(recent, names)
        pending_deltas: list[Mapping[str, float]] = []
        for action in model_input.pending_actions:
            samples = action_deltas.get(self._action_key(action), [])
            if samples:
                pending_deltas.append(self._mean(samples[-self.recent_window :], names))
        if model_input.pending_actions:
            first_delta = (
                {
                    name: sum(delta.get(name, 0.0) for delta in pending_deltas)
                    for name in names
                }
                if pending_deltas
                else baseline
            )
        else:
            first_delta = baseline

        current = dict(rows[-1])
        predictions: list[dict[str, float]] = []
        for step in range(model_input.horizon):
            delta = first_delta if step == 0 else baseline
            current = {
                name: current[name] + float(delta.get(name, 0.0))
                for name in names
            }
            predictions.append(dict(current))
        return StateForecast.from_rows(
            model_input,
            predictions,
            model_name=self.name,
        )


Chronos2Predictor: TypeAlias = Callable[
    [ForecastModelInput], Sequence[Mapping[str, float]] | StateForecast
]


@dataclass(slots=True)
class Chronos2Forecaster:
    """Lazy adapter boundary for a later GPU-backed Chronos2 implementation.

    The predictor is injected so this package remains installable without
    torch, CUDA, or model weights.  A GPU runner can construct a Chronos2
    pipeline and wrap its tensor-to-feature conversion in a callable accepting
    :class:`ForecastModelInput`; the agent and trace format then remain
    unchanged.  Calling this class without a predictor fails explicitly
    instead of silently falling back to a different model.
    """

    predictor: Chronos2Predictor | None = None
    name: str = field(init=False, default="chronos2")

    @classmethod
    def from_callable(cls, predictor: Chronos2Predictor) -> "Chronos2Forecaster":
        return cls(predictor=predictor)

    def predict(self, model_input: ForecastModelInput) -> StateForecast:
        if self.predictor is None:
            raise RuntimeError(
                "Chronos2Forecaster has no backend; provide a GPU-backed "
                "predictor with Chronos2Forecaster.from_callable(...)"
            )
        result = self.predictor(model_input)
        if isinstance(result, StateForecast):
            return result
        return StateForecast.from_rows(
            model_input,
            result,
            model_name=self.name,
        )


DEFAULT_FORECAST_WEIGHTS: Mapping[str, float] = {
    "value/GDP": 1.0,
    "value/Health": 1.0,
    "value/Education": 1.0,
    "value/CrimeRate": -1.0,
    "value/Unemployment": -1.0,
}


def score_forecast(
    features: FeatureRow,
    *,
    weights: Mapping[str, float] = DEFAULT_FORECAST_WEIGHTS,
    poll_weight: float = 0.5,
) -> float:
    """Score a predicted feature row for action selection."""

    score = sum(
        float(weight) * float(features.get(name, 0.0))
        for name, weight in weights.items()
    )
    return score + poll_weight * float(features.get("politics/poll_rate", 0.0))


@dataclass(frozen=True, slots=True)
class ForecastDecision:
    """One model choice plus its later observed outcome."""

    turn: int
    actions: tuple[ActionRecord, ...]
    forecast: StateForecast
    score: float
    candidate_count: int
    observed: StateSnapshot | None = None

    @property
    def one_step_mae(self) -> float | None:
        if self.observed is None:
            return None
        errors = [
            abs(self.forecast.first.get(name, 0.0) - self.observed.features.get(name, 0.0))
            for name in self.forecast.feature_names
        ]
        return sum(errors) / len(errors) if errors else 0.0

    def to_dict(self) -> dict[str, object]:
        return {
            "turn": self.turn,
            "actions": [action.to_dict() for action in self.actions],
            "forecast": self.forecast.to_dict(),
            "score": self.score,
            "candidate_count": self.candidate_count,
            "observed": self.observed.to_dict() if self.observed else None,
            "one_step_mae": self.one_step_mae,
        }


@dataclass(frozen=True, slots=True)
class ForecastTrace:
    """Persistable episode data for later model comparison."""

    context: AutoregressiveContext
    decisions: tuple[ForecastDecision, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "autocracy-timeseries-trace-v1",
            "context": self.context.to_dict(),
            "decisions": [decision.to_dict() for decision in self.decisions],
        }

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


class TimeSeriesPolicyAgent(BaseAgent):
    """Choose policy actions from autoregressive state forecasts.

    This agent intentionally does not predict a policy from policy metadata.
    It enumerates legal action candidates, asks the supplied forecaster for a
    future trajectory conditioned on each candidate, selects the best score,
    executes only that batch in the simulator, and appends the observed state
    to the context before the next decision.
    """

    def __init__(
        self,
        forecaster: ActionConditionedForecaster,
        *,
        country: str = "uk",
        gamedata_root: str | Path | None = None,
        state: SimulationState | None = None,
        config: SimulationConfig | None = None,
        encoder: StateFeatureEncoder | None = None,
        forecast_horizon: int = 4,
        candidate_limit: int | None = 32,
        random_seed: int | None = None,
        objective: Callable[[FeatureRow], float] = score_forecast,
    ) -> None:
        super().__init__(
            country=country,
            gamedata_root=gamedata_root,
            state=state,
            config=config,
        )
        self.forecaster = forecaster
        self.feature_encoder = encoder or StateFeatureEncoder.from_state(
            self.state
        )
        self.forecast_horizon = forecast_horizon
        self.candidate_limit = candidate_limit
        self.random_seed = random_seed
        self._random = random.Random(random_seed) if random_seed is not None else None
        self.objective = objective
        self.context = AutoregressiveContext.from_state(
            self.state,
            encoder=self.feature_encoder,
        )
        self.decisions: list[ForecastDecision] = []
        self.last_decision: ForecastDecision | None = None

    def _candidate_batches(self, options) -> list[tuple[PolicyAction, ...]]:
        ordered = list(options)
        if self.candidate_limit is not None:
            if self.candidate_limit < 1:
                raise ValueError("candidate_limit must be at least one or None")
            limit = max(0, self.candidate_limit - 1)
            if len(ordered) > limit:
                if self._random is not None:
                    ordered = self._random.sample(ordered, limit)
                else:
                    ordered = sorted(
                        ordered,
                        key=lambda option: (
                            option.policy_name,
                            option.action_type,
                            option.resulting_level,
                        ),
                    )[:limit]
        batches: list[tuple[PolicyAction, ...]] = [()]
        batches.extend(
            (
                PolicyAction(
                    policy_name=option.policy_name,
                    delta=option.delta,
                    action_type=option.action_type,
                ),
            )
            for option in ordered
        )
        return batches

    @staticmethod
    def _action_key(actions: Sequence[PolicyAction]) -> tuple:
        return tuple(
            (action.policy_name, action.action_type or "", round(action.delta, 9))
            for action in actions
        )

    def choose_actions(self, state: SimulationState, options) -> tuple[PolicyAction, ...]:
        if state.turn != self.context.current_turn:
            self.context = AutoregressiveContext.from_state(
                state, encoder=self.feature_encoder
            )
            self.decisions.clear()
        candidates = self._candidate_batches(options)
        best_actions: tuple[PolicyAction, ...] | None = None
        best_forecast: StateForecast | None = None
        best_score = -math.inf
        for actions in candidates:
            model_input = self.context.model_input(
                actions, horizon=self.forecast_horizon
            )
            forecast = self.forecaster.predict(model_input)
            if forecast.horizon < self.forecast_horizon:
                raise ValueError(
                    f"forecaster returned {forecast.horizon} steps, "
                    f"expected {self.forecast_horizon}"
                )
            score = float(self.objective(forecast.final))
            if not math.isfinite(score):
                raise ValueError(f"forecast objective returned a non-finite score: {score!r}")
            if (
                best_actions is None
                or score > best_score
                or (
                    score == best_score
                    and self._action_key(actions) < self._action_key(best_actions)
                )
            ):
                best_actions = actions
                best_forecast = forecast
                best_score = score
        if best_actions is None or best_forecast is None:
            raise RuntimeError("forecast agent produced no candidate action")
        self.last_decision = ForecastDecision(
            turn=state.turn,
            actions=_records(best_actions),
            forecast=best_forecast,
            score=best_score,
            candidate_count=len(candidates),
        )
        return best_actions

    def step(self) -> SimulationState:
        before = self.state
        actions = self.choose_actions(before, self.available_actions())
        self.apply_actions(actions)
        self.end_turn()
        if self.last_decision is None:
            raise RuntimeError("forecast agent has no decision after choosing actions")
        self.context = self.context.append_transition(before, actions, self.state)
        observed = self.context.states[-1]
        decision = replace(self.last_decision, observed=observed)
        self.decisions.append(decision)
        self.last_decision = decision
        return self.state

    @property
    def trace(self) -> ForecastTrace:
        return ForecastTrace(self.context, tuple(self.decisions))

    def save_trace(self, path: str | Path) -> None:
        self.trace.save(path)

    def load_state(self, path: str | Path) -> None:
        super().load_state(path)
        self.feature_encoder = StateFeatureEncoder.from_state(self.state)
        self.context = AutoregressiveContext.from_state(
            self.state, encoder=self.feature_encoder
        )
        self.decisions.clear()
        self.last_decision = None


__all__ = [
    "ActionConditionedForecaster",
    "ActionRecord",
    "AutoregressiveContext",
    "Chronos2Forecaster",
    "DEFAULT_FORECAST_WEIGHTS",
    "EmpiricalActionForecaster",
    "ForecastDecision",
    "ForecastModelInput",
    "ForecastTrace",
    "PersistenceForecaster",
    "StateFeatureEncoder",
    "StateForecast",
    "StateSnapshot",
    "TimeSeriesPolicyAgent",
    "score_forecast",
]
