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
import itertools
import json
import math
from pathlib import Path
import random
from typing import TYPE_CHECKING, Callable, Mapping, Protocol, Sequence, TypeAlias

from .agent import BaseAgent
from .models import (
    PolicyAction,
    PolicyActionOption,
    SimulationConfig,
    SimulationData,
    SimulationState,
)
from . import simulator

if TYPE_CHECKING:  # pragma: no cover - import cycle guard for type checkers
    from .learning import TreatmentEffectMemory


FeatureRow: TypeAlias = Mapping[str, float]

# Feature prefix marking a player-controlled treatment variable.  Treatments
# are known in every future step of a forecast because the agent itself
# chooses them; every other observed feature must be predicted.
TREATMENT_FEATURE_PREFIX = "policy/"
# Headline prediction target: total voter electoral support as published by
# the in-game opinion polls.
ELECTORAL_SUPPORT_FEATURE = "politics/poll_rate"

_PLAYER_INVISIBLE_CATEGORIES = frozenset({"HIDDEN", "PLACEHOLDER"})


def player_visible_value_names(
    state: SimulationState, data: SimulationData | None = None
) -> tuple[str, ...]:
    """Return the simulator value names the player can observe in-game.

    Special meta nodes that drive the simulation but are never shown to the
    player are excluded: every ``_``-prefixed neuron (``_globaleconomy_``,
    ``_year``, ``_Terrorism``, ...), every ``HIDDEN``/``PLACEHOLDER``
    category, and derived runtime neurons such as the nested voter income
    and percentage mirrors.  Only ordinary simulation nodes remain.
    """

    data = data or simulator.load_simulation_data()
    names: list[str] = []
    for name in state.values:
        if name.startswith("_"):
            continue
        node = data.nodes.get(name)
        if node is None:
            # Runtime-only mirrors (voter percentages, incomes) are derived
            # bookkeeping, not game state a player reads off a screen.
            continue
        if node.category in _PLAYER_INVISIBLE_CATEGORIES:
            continue
        names.append(name)
    return tuple(sorted(names))


def _finite(value: float, *, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"feature {name!r} is not finite: {value!r}")
    return number


def _result_value(state: SimulationState) -> float:
    return {None: 0.0, "loss": -1.0, "win": 1.0}.get(state.election_result, 0.0)


def _winner_value(state: SimulationState) -> float:
    return {
        None: 0.0,
        "opposition": -1.0,
        "player": 1.0,
    }.get(state.last_election_winner, 0.0)


_FINANCE_SOURCES: dict[str, Callable[[SimulationState], float]] = {
    "finance/political_capital": lambda state: state.political_capital,
    "finance/political_capital_income": lambda state: state.political_capital_income,
    "finance/total_expenditure": lambda state: state.total_expenditure,
    "finance/total_income": lambda state: state.total_income,
    "finance/debt": lambda state: state.debt,
    "finance/interest_rate": lambda state: state.interest_rate,
}

_ELECTION_SOURCES: dict[str, Callable[[SimulationState], float]] = {
    "politics/poll_rate": lambda state: state.poll_rate,
    "politics/peak_poll_rate": lambda state: state.peak_poll_rate,
    "politics/active_situations": lambda state: float(len(state.active_situations)),
    "election/turns_until": lambda state: float(state.election_turns_until),
    "election/current_term": lambda state: float(state.election_current_term),
    "election/result": _result_value,
    "election/last_winner": _winner_value,
}

# Auxiliary fields the game actually displays.  The player-visible schema
# encodes only these; accrual rates, derived peaks, situation counts, and
# past-result markers stay out of the model's covariates.
VISIBLE_FINANCE_FEATURES: tuple[str, ...] = (
    "finance/political_capital",
    "finance/total_expenditure",
    "finance/total_income",
    "finance/debt",
    "finance/interest_rate",
)
VISIBLE_ELECTION_FEATURES: tuple[str, ...] = (
    "politics/poll_rate",
    "election/turns_until",
)


@dataclass(frozen=True, slots=True)
class StateFeatureEncoder:
    """Encode selected simulator fields into a stable multivariate row.

    ``value_names`` and ``policy_names`` are fixed when the context is
    created.  This prevents a later state from silently changing the model's
    column order.  ``from_state`` is the usual constructor and captures all
    ordinary value and policy names by default; callers can pass a smaller
    list for a compact foundation-model experiment.

    ``finance_names`` and ``election_names`` select which auxiliary columns
    accompany the values.  ``None`` means the legacy full group; the
    player-visible constructor curates only fields the game displays.
    """

    value_names: tuple[str, ...] = ()
    policy_names: tuple[str, ...] = ()
    include_finance: bool = True
    include_election: bool = True
    finance_names: tuple[str, ...] | None = None
    election_names: tuple[str, ...] | None = None

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

    @classmethod
    def from_visible_state(
        cls,
        state: SimulationState,
        *,
        data: SimulationData | None = None,
        include_finance: bool = True,
        include_election: bool = True,
    ) -> "StateFeatureEncoder":
        """Build the schema over what the player actually sees in-game.

        Values are restricted to ordinary (non-meta) simulation nodes;
        finance and election columns keep only fields the UI displays.
        Nothing hidden or derived-by-us is encoded.
        """

        return cls.from_state(
            state,
            value_names=player_visible_value_names(state, data),
            include_finance=include_finance,
            include_election=include_election,
        )._curated_auxiliary_columns()

    def _curated_auxiliary_columns(self) -> "StateFeatureEncoder":
        return replace(
            self,
            finance_names=(
                VISIBLE_FINANCE_FEATURES if self.include_finance else ()
            ),
            election_names=(
                VISIBLE_ELECTION_FEATURES if self.include_election else ()
            ),
        )

    @property
    def _active_finance_names(self) -> tuple[str, ...]:
        if not self.include_finance:
            return ()
        if self.finance_names is None:
            return tuple(_FINANCE_SOURCES)
        return self.finance_names

    @property
    def _active_election_names(self) -> tuple[str, ...]:
        if not self.include_election:
            return ()
        if self.election_names is None:
            return tuple(_ELECTION_SOURCES)
        return self.election_names

    @property
    def feature_names(self) -> tuple[str, ...]:
        names = [f"value/{name}" for name in self.value_names]
        names.extend(f"policy/{name}" for name in self.policy_names)
        names.extend(self._active_finance_names)
        names.extend(self._active_election_names)
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
        for name in self._active_finance_names:
            source = _FINANCE_SOURCES[name]
            row[name] = _finite(source(state), name=name)
        for name in self._active_election_names:
            source = _ELECTION_SOURCES[name]
            row[name] = _finite(source(state), name=name)
        return row

    def to_dict(self) -> dict[str, object]:
        return {
            "value_names": list(self.value_names),
            "policy_names": list(self.policy_names),
            "include_finance": self.include_finance,
            "include_election": self.include_election,
            "finance_names": (
                None if self.finance_names is None else list(self.finance_names)
            ),
            "election_names": (
                None if self.election_names is None else list(self.election_names)
            ),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "StateFeatureEncoder":
        finance_names = payload.get("finance_names")
        election_names = payload.get("election_names")
        return cls(
            value_names=tuple(str(name) for name in payload.get("value_names", [])),
            policy_names=tuple(str(name) for name in payload.get("policy_names", [])),
            include_finance=bool(payload.get("include_finance", True)),
            include_election=bool(payload.get("include_election", True)),
            finance_names=(
                None
                if finance_names is None
                else tuple(str(name) for name in finance_names)
            ),
            election_names=(
                None
                if election_names is None
                else tuple(str(name) for name in election_names)
            ),
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
class FeatureRoleSchema:
    """Covariate roles for a fixed feature schema.

    ``target_names`` are observed features the model must predict (the first
    entry, :data:`ELECTORAL_SUPPORT_FEATURE`, is the headline objective).
    ``treatment_names`` are player-controlled policy sliders: covariates that
    stay known at every future step because the agent chooses them.  All
    remaining columns are past-only covariates whose future values are
    unknown and therefore predicted.
    """

    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    treatment_names: tuple[str, ...]

    @classmethod
    def from_encoder(cls, encoder: StateFeatureEncoder) -> "FeatureRoleSchema":
        treatments = tuple(
            name
            for name in encoder.feature_names
            if name.startswith(TREATMENT_FEATURE_PREFIX)
        )
        treatment_set = set(treatments)
        targets = [
            name for name in encoder.feature_names if name not in treatment_set
        ]
        # The headline objective leads the predicted columns so consumers
        # can index the primary target deterministically.
        if ELECTORAL_SUPPORT_FEATURE in targets:
            targets.remove(ELECTORAL_SUPPORT_FEATURE)
            targets.insert(0, ELECTORAL_SUPPORT_FEATURE)
        return cls(
            feature_names=encoder.feature_names,
            target_names=tuple(targets),
            treatment_names=treatments,
        )

    def is_treatment(self, feature_name: str) -> bool:
        return feature_name in self._treatment_set()

    def _treatment_set(self) -> frozenset[str]:
        return frozenset(self.treatment_names)


def _fill_placeholder_zeros(ring: list[float]) -> list[float]:
    """Replace serialized placeholder zeros with carried-forward values.

    The game writes exact ``0`` into a node's history ring whenever its
    simulation pass has not run yet, and some statistics (GDP among them)
    are only recomputed periodically, so raw rings contain long zero runs
    both before a node goes live and between its periodic samples.  The
    player never sees those zeros: the UI keeps showing the last computed
    value.  Carrying the most recent non-zero reading forward therefore
    reconstructs the observable series instead of teaching the forecaster
    that live series repeatedly collapse to zero.
    """

    filled: list[float] = []
    carry = 0.0
    for value in ring:
        if value != 0.0:
            carry = value
        elif not filled:
            continue
        filled.append(carry)
    return filled


def _pre_game_snapshots(
    state: SimulationState, schema: StateFeatureEncoder
) -> list[StateSnapshot] | None:
    """Replay the save's value rings as snapshots ending at ``state.turn``.

    The rings are only aligned with the current turn when they were loaded
    from the same snapshot; once real turns advance, the live observations
    take over and the pre-game rows would overlap.  Features without a ring
    (policies, finance fields, most politics fields) are held constant at
    their current level, which is exactly the information a fresh game gives
    the player about the pre-game period.
    """

    if state.value_histories_turn is None or state.value_histories_turn != state.turn:
        return None
    current = schema.encode(state)
    oldest_first_rings: dict[str, list[float]] = {}
    for name in schema.value_names:
        ring = state.value_histories.get(name)
        if ring:
            oldest_first = [float(value) for value in reversed(ring)]
            oldest_first_rings[f"value/{name}"] = _fill_placeholder_zeros(
                oldest_first
            )
    if schema.include_election and state.poll_history:
        oldest_first_rings[ELECTORAL_SUPPORT_FEATURE] = [
            float(value) for value in reversed(state.poll_history)
        ]
    length = max((len(ring) for ring in oldest_first_rings.values()), default=0)
    if length < 2:
        return None
    snapshots: list[StateSnapshot] = []
    for offset in range(length - 1, -1, -1):
        row = dict(current)
        turn = state.turn - offset
        for feature_name, ring in oldest_first_rings.items():
            index = len(ring) - 1 - offset
            if index >= 0:
                row[feature_name] = float(ring[index])
        snapshots.append(StateSnapshot(turn=turn, features=row))
    return snapshots


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
        include_value_histories: bool = False,
    ) -> "AutoregressiveContext":
        schema = encoder or StateFeatureEncoder.from_state(state)
        if include_value_histories:
            snapshots = _pre_game_snapshots(state, schema)
            if snapshots is not None:
                return cls(
                    schema,
                    tuple(snapshots),
                    ((),) * (len(snapshots) - 1),
                )
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


def forecast_debt_gdp_growth(
    forecast: StateForecast, baseline: FeatureRow | None = None
) -> float:
    """Scale-free rise in predicted debt burden across the forecast path.

    Debt, GDP, income, and expenditure are ordinary player-visible targets
    the forecaster predicts for every step of the horizon; this reads that
    predicted fiscal trajectory so scoring can act on it exactly like the
    simulator oracle does.  With ``baseline`` (the latest observed feature
    row) the measure runs from today's observation to the final predicted
    step, so an immediate debt jump counts; otherwise it measures slope
    inside the predicted window.  The measure is relative — debt growth
    divided by GDP growth minus one — so mixing normalized gauges with
    absolute currency is harmless.  Returns 0 when either column is absent.
    """

    start_row: FeatureRow = baseline if baseline is not None else forecast.first
    final = forecast.final
    if "value/GDP" not in start_row or "finance/debt" not in start_row:
        return 0.0
    start_gdp = float(start_row.get("value/GDP", 0.0))
    start_debt = float(start_row.get("finance/debt", 0.0))
    final_gdp = float(final.get("value/GDP", 0.0))
    final_debt = float(final.get("finance/debt", 0.0))
    eps = 1e-9
    if abs(start_debt) < eps or abs(start_gdp) < eps:
        return 0.0
    try:
        growth = (final_debt / start_debt) / (final_gdp / start_gdp) - 1.0
    except ZeroDivisionError:
        return 0.0
    return growth if math.isfinite(growth) else 0.0


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


def diverse_warmup_plan(
    state: SimulationState,
    *,
    data: SimulationData | None = None,
    size: int = 6,
    capital_share: float = 0.5,
) -> tuple[PolicyAction, ...]:
    """Build a deterministic starter program of legal single moves.

    Warm-up exists to give an in-context forecaster interventional variety:
    alternating small raises and lowers across distinct policies produces
    treatment/response pairs without spending most of the political capital.
    Options come from :func:`autocracy.simulator.list_available_actions`, so
    every planned move is legal at planning time; the agent still re-checks
    legality each warm-up turn and skips stale entries.
    """

    options = simulator.list_available_actions(state, data=data)
    # Cheapest moves first so a fixed capital share buys as many distinct
    # raise/lower contrasts as possible.
    ordered = sorted(
        options,
        key=lambda option: (
            round(option.cost, 4),
            option.policy_name,
            option.action_type,
            round(option.delta, 6),
        ),
    )
    pools = (
        [o for o in ordered if o.action_type in {"raise", "introduce"}],
        [o for o in ordered if o.action_type == "lower"],
    )
    budget = state.political_capital * capital_share
    spent = 0.0
    plan: list[PolicyAction] = []
    used_policies: set[str] = set()
    cursors = [0, 0]
    preferred = 0
    while len(plan) < size:
        progressed = False
        for offset in range(2):
            pool_index = (preferred + offset) % 2
            pool = pools[pool_index]
            cursor = cursors[pool_index]
            while cursor < len(pool):
                option = pool[cursor]
                cursor += 1
                if option.policy_name in used_policies:
                    continue
                if spent + option.cost > budget + simulator.EPSILON:
                    continue
                plan.append(
                    PolicyAction(
                        policy_name=option.policy_name,
                        delta=option.delta,
                        action_type=option.action_type,
                    )
                )
                used_policies.add(option.policy_name)
                spent += option.cost
                progressed = True
                break
            cursors[pool_index] = cursor
            if progressed:
                break
        if not progressed:
            # Every pool is exhausted or unaffordable within the budget.
            break
        preferred = (preferred + 1) % 2
    return tuple(plan)


class TimeSeriesPolicyAgent(BaseAgent):
    """Choose policy actions from autoregressive state forecasts.

    This agent intentionally does not predict a policy from policy metadata.
    It enumerates legal action candidates, asks the supplied forecaster for a
    future trajectory conditioned on each candidate, selects the best score,
    executes only that batch in the simulator, and appends the observed state
    to the context before the next decision.

    ``warmup_plan`` runs a scripted sequence of moves for the first decision
    turns instead of trusting the forecaster's zero-shot preferences.  Every
    executed warm-up transition enters the context, so the model observes real
    treatment/response pairs before it starts choosing.  After warm-up,
    ``reverse_window``/``reverse_penalty`` damp flip-flops: a candidate that
    reverses an action taken within the last ``reverse_window`` transitions
    on the same policy loses ``reverse_penalty`` score points.
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
        visible_features_only: bool = False,
        seed_pre_game_history: bool = False,
        warmup_plan: Sequence[PolicyAction | ActionRecord] | None = None,
        reverse_window: int = 0,
        reverse_penalty: float = 0.0,
        warmup_batch_size: int = 1,
        max_actions_per_turn: int = 1,
        batch_candidate_limit: int | None = None,
        debt_growth_penalty: float = 0.0,
        max_action_delta: float | None = None,
        score_horizon_mean: bool = False,
        treatment_memory: "TreatmentEffectMemory | None" = None,
        memory_effect_weight: float = 1.0,
        memory_drift_window: int = 8,
        exploration_countdown: bool = False,
        memory_credit_lag: int = 0,
        balance_guard_penalty: float = 0.0,
        fiscal_prudence_weight: float = 0.0,
    ) -> None:
        super().__init__(
            country=country,
            gamedata_root=gamedata_root,
            state=state,
            config=config,
        )
        if reverse_window < 0:
            raise ValueError("reverse_window must be non-negative")
        if max_actions_per_turn < 1:
            raise ValueError("max_actions_per_turn must be at least one")
        if batch_candidate_limit is not None and batch_candidate_limit < 1:
            raise ValueError("batch_candidate_limit must be at least one or None")
        self.forecaster = forecaster
        self.visible_features_only = visible_features_only
        self.seed_pre_game_history = seed_pre_game_history
        self.feature_encoder = encoder or self._default_encoder(self.state)
        self.role_schema = FeatureRoleSchema.from_encoder(self.feature_encoder)
        self.forecast_horizon = forecast_horizon
        self.candidate_limit = candidate_limit
        self.random_seed = random_seed
        self._random = random.Random(random_seed) if random_seed is not None else None
        self.objective = objective
        self.warmup_plan: list[ActionRecord] = (
            [ActionRecord.from_action(action) for action in warmup_plan]
            if warmup_plan is not None
            else []
        )
        self.warmup_turns_taken = 0
        self.reverse_window = reverse_window
        if warmup_batch_size < 1:
            raise ValueError("warmup_batch_size must be at least one")
        self.warmup_batch_size = warmup_batch_size
        self.reverse_penalty = float(reverse_penalty)
        self.max_actions_per_turn = max_actions_per_turn
        self.batch_candidate_limit = batch_candidate_limit
        self.debt_growth_penalty = float(debt_growth_penalty)
        self.max_action_delta = (
            None if max_action_delta is None else float(max_action_delta)
        )
        # When set, candidates are ranked by the objective averaged over the
        # whole predicted path instead of the final step alone: a move whose
        # gain is a single-step blip scores less than one that compounds.
        self.score_horizon_mean = bool(score_horizon_mean)
        self.treatment_memory = treatment_memory
        self.memory_effect_weight = float(memory_effect_weight)
        if memory_drift_window < 0:
            raise ValueError("memory_drift_window must be non-negative")
        self.memory_drift_window = int(memory_drift_window)
        if treatment_memory is not None and (
            ELECTORAL_SUPPORT_FEATURE
            not in self.feature_encoder.feature_names
        ):
            raise ValueError(
                "treatment memory records electoral support; the feature "
                "encoder must include politics/poll_rate"
            )
        # Per-single-action political-capital costs captured while enumerating
        # candidate batches, used to discount exploration bonuses.
        self._option_costs: dict[tuple[str, str, float], float] = {}
        # Single-life active learning: exploration is worth most early in a
        # term and must vanish as the election approaches.  The scale is the
        # share of the term still ahead at each decision.
        self.exploration_countdown = bool(exploration_countdown)
        if memory_credit_lag < 0:
            raise ValueError("memory_credit_lag must be non-negative")
        self.memory_credit_lag = int(memory_credit_lag)
        # Budget-screen prudence: while the treasury runs a deficit, a
        # candidate whose own forecast deepens it pays a score penalty.  The
        # rule uses only displayed income/expenditure lines and their
        # predicted next-step values — no policy metadata, no thresholds.
        self.balance_guard_penalty = float(balance_guard_penalty)
        # While the visible budget is in deficit, candidates are scored with
        # their *measured* balance effect (learned channel, normalised by
        # expenditure).  Symmetric: balance-repairing moves earn credit.
        self.fiscal_prudence_weight = float(fiscal_prudence_weight)
        self._fiscal_history: list[float] = []
        # Actions awaiting their multi-turn effect window: [remaining
        # transitions, poll at execution time, actions].
        self._pending_credits: list[list[object]] = []
        # First-step feature row the forecaster predicted for this turn's
        # no-op candidate; the recorder uses it as the counterfactual.
        self._noop_forecast_row: Mapping[str, float] | None = None
        self._initial_term_length = max(1, int(self.state.election_turns_until))
        self.context = self._initial_context()
        self.decisions: list[ForecastDecision] = []
        self.last_decision: ForecastDecision | None = None

    @property
    def in_warmup(self) -> bool:
        return bool(self.warmup_plan)

    def _default_encoder(self, state: SimulationState) -> StateFeatureEncoder:
        if self.visible_features_only:
            return StateFeatureEncoder.from_visible_state(state, data=self.data)
        return StateFeatureEncoder.from_state(state)

    def _initial_context(self) -> AutoregressiveContext:
        return AutoregressiveContext.from_state(
            self.state,
            encoder=self.feature_encoder,
            include_value_histories=self.seed_pre_game_history,
        )

    def _candidate_batches(self, options) -> list[tuple[PolicyAction, ...]]:
        """Enumerate legal action batches for this turn.

        Every batch — including multi-action ones — must satisfy the hard
        per-turn budget: the summed political-capital cost of its actions may
        not exceed the capital available at the start of the turn.  Singles
        are bounded by ``candidate_limit`` (which includes the no-op); pairs
        and larger combinations are formed from that same sampled option
        pool and additionally capped by ``batch_candidate_limit``.
        ``max_action_delta``, when set, restricts the pool to smaller slider
        steps so the agent compounds many modest moves instead of a few
        large ones.
        """

        ordered = [
            option
            for option in options
            if self.max_action_delta is None
            or abs(option.delta) <= self.max_action_delta + simulator.EPSILON
        ]
        # Options are enumerated against the policy-neuron value, but actions
        # apply against the slider target; when implementation lag separates
        # the two, an option can clamp to "no change" and crash the real
        # apply phase.  Drop those candidates up front.
        ordered = [
            option for option in ordered if self._moves_desired_level(option)
        ]
        self._option_costs = {
            (
                option.policy_name,
                option.action_type or "",
                round(option.delta, 6),
            ): float(option.cost)
            for option in ordered
        }
        if self.candidate_limit is not None:
            if self.candidate_limit < 1:
                raise ValueError("candidate_limit must be at least one or None")
            limit = max(0, self.candidate_limit - 1)
            if len(ordered) > limit:
                ordered = self._prioritize_options(ordered, limit)
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
        if self.max_actions_per_turn >= 2:
            capital = self.state.political_capital
            combos = [
                tuple(
                    PolicyAction(
                        policy_name=option.policy_name,
                        delta=option.delta,
                        action_type=option.action_type,
                    )
                    for option in combo
                )
                for combo in itertools.combinations(ordered, 2)
                if combo[0].policy_name != combo[1].policy_name
                and combo[0].cost + combo[1].cost <= capital + simulator.EPSILON
            ]
            combos.sort(key=self._action_key)
            if self.batch_candidate_limit is not None and len(combos) > (
                self.batch_candidate_limit
            ):
                combos = self._prioritize_combos(ordered, combos)
            batches.extend(combos)
        return batches

    def _option_priority(self, option) -> float:
        """Learned usefulness plus curiosity, with randomized novelty.

        Never-tried options are ordered uniformly at random so each life
        covers a different slice of the roster; once an action has samples,
        its measured effect (plus a shrinking visit bonus) dominates.
        """

        assert self.treatment_memory is not None
        estimate = self.treatment_memory.estimate(option)
        if estimate is None:
            scale = self.treatment_memory.exploration_bonus
            if self._random is not None:
                return self._random.uniform(0.0, scale)
            return 0.5 * scale / (
                1.0 + float(option.cost) / self.treatment_memory.reference_cost
            )
        jitter = (
            self._random.uniform(0.0, 0.05 * self.treatment_memory.exploration_bonus)
            if self._random
            else 0.0
        )
        return estimate + self.treatment_memory._explore_bonus(
            option, cost=float(option.cost)
        ) + jitter

    def _prioritize_options(self, ordered, limit: int):
        """Pick which legal moves enter the forecast batch.

        With a treatment memory the pool concentrates on the highest learned-
        effect / highest-uncertainty options instead of a uniform sample —
        candidate forecasts are expensive and flat rankings waste them.
        """

        if self.treatment_memory is not None:
            ranked = sorted(
                ordered,
                key=lambda option: (
                    -self._option_priority(option),
                    option.policy_name,
                    option.action_type,
                    round(option.delta, 6),
                ),
            )
            return ranked[:limit]
        if self._random is not None:
            return self._random.sample(ordered, limit)
        # A bounded search must still be deterministic when no seed is given.
        return sorted(
            ordered,
            key=lambda option: (
                option.policy_name,
                option.action_type,
                option.resulting_level,
            ),
        )[:limit]

    def _prioritize_combos(self, ordered, combos):
        """Keep the pairs that combine the most promising singles."""

        assert self.batch_candidate_limit is not None
        if self.treatment_memory is None:
            if self._random is not None:
                return self._random.sample(combos, self.batch_candidate_limit)
            return combos[: self.batch_candidate_limit]
        priority = {
            (
                option.policy_name,
                option.action_type or "",
                round(option.delta, 6),
            ): self._option_priority(option)
            for option in ordered
        }

        def combo_priority(combo) -> float:
            return sum(
                priority.get(
                    (
                        action.policy_name,
                        action.action_type or "",
                        round(action.delta, 6),
                    ),
                    0.0,
                )
                for action in combo
            )

        combos = sorted(
            combos,
            key=lambda combo: (-combo_priority(combo), self._action_key(combo)),
        )
        return combos[: self.batch_candidate_limit]

    def _moves_desired_level(self, option) -> bool:
        """Check the option changes the slider target the apply phase uses."""

        desired = self.state.policy_desired_throttles.get(option.policy_name)
        if desired is None or option.action_type == "cancel":
            return True
        target = min(1.0, max(0.0, float(desired) + option.delta))
        return abs(target - float(desired)) > simulator.EPSILON

    @staticmethod
    def _action_key(actions: Sequence[PolicyAction]) -> tuple:
        return tuple(
            (action.policy_name, action.action_type or "", round(action.delta, 9))
            for action in actions
        )

    def _pop_legal_warmup_actions(
        self, options: Sequence[PolicyActionOption], limit: int
    ) -> tuple[PolicyAction, ...]:
        """Return the next legal warm-up moves, up to ``limit`` this turn.

        Planned moves whose option no longer exists (or is unaffordable,
        alone or together with the moves already picked this turn) are
        skipped so stale entries cannot stall the schedule.
        """

        chosen: list[PolicyAction] = []
        committed_cost = 0.0
        while len(chosen) < limit and self.warmup_plan:
            planned = self.warmup_plan.pop(0)
            matched: PolicyActionOption | None = None
            for option in options:
                same_move = (
                    option.policy_name == planned.policy_name
                    and (option.action_type or "") == (planned.action_type or "")
                    and abs(option.delta - planned.delta) <= simulator.EPSILON
                )
                if same_move:
                    matched = option
                    break
            if matched is None:
                continue
            new_total = committed_cost + matched.cost
            if new_total > self.state.political_capital + simulator.EPSILON:
                continue
            committed_cost = new_total
            chosen.append(
                PolicyAction(
                    policy_name=planned.policy_name,
                    delta=planned.delta,
                    action_type=planned.action_type,
                )
            )
        return tuple(chosen)

    def _recent_actions(self, window: int) -> list[ActionRecord]:
        recent: list[ActionRecord] = []
        for transition in reversed(self.context.actions):
            for action in transition:
                recent.append(action)
                if len(recent) >= window:
                    return recent
        return recent

    def _reverse_adjusted_score(
        self,
        actions: Sequence[PolicyAction],
        score: float,
        recent: Sequence[ActionRecord],
    ) -> float:
        for action in actions:
            for previous in recent:
                if previous.policy_name != action.policy_name:
                    continue
                # Opposite-direction move on a recently touched slider.
                if previous.delta * action.delta < 0.0:
                    return score - self.reverse_penalty
        return score

    def choose_actions(self, state: SimulationState, options) -> tuple[PolicyAction, ...]:
        if state.turn != self.context.current_turn:
            self.context = self._initial_context()
            self.decisions.clear()
        self._noop_forecast_row = None
        candidates = self._candidate_batches(options)
        if self.warmup_plan:
            scheduled = self._pop_legal_warmup_actions(
                options, self.warmup_batch_size
            )
            if scheduled:
                self.warmup_turns_taken += 1
                model_input = self.context.model_input(
                    scheduled, horizon=self.forecast_horizon
                )
                forecast = self.forecaster.predict(model_input)
                score = float(self.objective(forecast.final))
                self.last_decision = ForecastDecision(
                    turn=state.turn,
                    actions=_records(scheduled),
                    forecast=forecast,
                    score=score,
                    candidate_count=1,
                )
                return scheduled
            # Nothing legal remains; fall through to model-driven choice.

        model_inputs = [
            self.context.model_input(actions, horizon=self.forecast_horizon)
            for actions in candidates
        ]
        batch_predict = getattr(self.forecaster, "predict_batch", None)
        if callable(batch_predict):
            forecasts = list(batch_predict(model_inputs))
        else:
            forecasts = [self.forecaster.predict(item) for item in model_inputs]
        # Remember the no-op counterfactual: the recorder de-trends observed
        # movement against what the world model expected without any action,
        # a sharper baseline than recent-transition medians.
        noop_index = candidates.index(()) if () in candidates else None
        self._noop_forecast_row = (
            dict(forecasts[noop_index].first) if noop_index is not None else None
        )
        best_actions: tuple[PolicyAction, ...] | None = None
        best_forecast: StateForecast | None = None
        best_score = -math.inf
        recent = (
            self._recent_actions(self.reverse_window)
            if self.reverse_window and self.reverse_penalty
            else ()
        )
        feature_names = self.feature_encoder.feature_names
        fiscal_baseline = dict(
            zip(
                feature_names,
                self.context.states[-1].row(feature_names),
            )
        )
        for actions, forecast in zip(candidates, forecasts):
            if forecast.horizon < self.forecast_horizon:
                raise ValueError(
                    f"forecaster returned {forecast.horizon} steps, "
                    f"expected {self.forecast_horizon}"
                )
            if self.score_horizon_mean:
                score = sum(
                    float(self.objective(row)) for row in forecast.values
                ) / max(len(forecast.values), 1)
            else:
                score = float(self.objective(forecast.final))
            if not math.isfinite(score):
                raise ValueError(f"forecast objective returned a non-finite score: {score!r}")
            if self.debt_growth_penalty:
                # The forecaster predicts the visible fiscal lines (debt,
                # GDP, income, expenditure) for every horizon step; make a
                # rising predicted debt-to-GDP ratio cost score.
                score -= self.debt_growth_penalty * max(
                    0.0,
                    forecast_debt_gdp_growth(forecast, fiscal_baseline),
                )
            score = self._reverse_adjusted_score(actions, score, recent)
            if self.treatment_memory is not None:
                # Learned treatment attribution from observed transitions:
                # de-trended poll effects plus an optimism bonus for rarely
                # tried, affordable moves.
                score += self.memory_effect_weight * (
                    self.treatment_memory.effect_total(actions)
                )
                if self.fiscal_prudence_weight and self._in_deficit():
                    score += self.fiscal_prudence_weight * (
                        self.treatment_memory.fiscal_total(actions)
                    )
                exploration = self.treatment_memory.explore_total(
                    actions,
                    costs=[
                        self._option_costs.get(
                            (
                                action.policy_name,
                                action.action_type or "",
                                round(action.delta, 6),
                            ),
                            0.0,
                        )
                        for action in actions
                    ],
                )
                if self.exploration_countdown:
                    # Active learning within one life: curiosity is scaled by
                    # the share of the current term still ahead, so late-term
                    # decisions exploit what earlier turns measured.
                    remaining = max(0.0, float(state.election_turns_until))
                    exploration *= min(1.0, remaining / self._initial_term_length)
                score += exploration
            score = self._balance_guard_penalty(actions, forecast, score)
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
        if self.treatment_memory is not None:
            self._record_treatment_effects(actions)
        observed = self.context.states[-1]
        decision = replace(self.last_decision, observed=observed)
        self.decisions.append(decision)
        self.last_decision = decision
        return self.state

    def _in_deficit(self) -> bool:
        """Whether the visible budget currently runs a deficit."""

        names = self.feature_encoder.feature_names
        current = dict(zip(names, self.context.states[-1].row(names)))
        balance = (
            float(current.get("finance/total_income", 0.0))
            - float(current.get("finance/total_expenditure", 0.0))
        )
        return balance < -simulator.EPSILON

    def _balance_guard_penalty(
        self,
        actions: Sequence[PolicyAction],
        forecast: StateForecast,
        score: float,
    ) -> float:
        """Penalize candidates that deepen an already-negative forecast balance."""

        if not self.balance_guard_penalty:
            return score
        names = self.feature_encoder.feature_names
        current = dict(
            zip(names, self.context.states[-1].row(names))
        )
        cur_income = float(current.get("finance/total_income", 0.0))
        cur_exp = float(current.get("finance/total_expenditure", 0.0))
        cur_balance = cur_income - cur_exp
        if cur_balance >= 0.0 or abs(cur_balance) < simulator.EPSILON:
            return score
        pred_income = float(forecast.first.get("finance/total_income", cur_income))
        pred_exp = float(forecast.first.get("finance/total_expenditure", cur_exp))
        worsening = cur_balance - (pred_income - pred_exp)
        if worsening <= 0.0:
            return score
        return score - self.balance_guard_penalty * worsening / abs(cur_balance)

    def _poll_deltas(self) -> list[float]:
        """Observed per-transition poll deltas, oldest first."""

        states = self.context.states
        return [
            current.features[ELECTORAL_SUPPORT_FEATURE]
            - previous.features[ELECTORAL_SUPPORT_FEATURE]
            for previous, current in zip(states[:-1], states[1:])
        ]

    def _record_treatment_effects(self, actions: Sequence[PolicyAction]) -> None:
        """Feed the just-observed transition into the treatment memory.

        The observed poll delta is de-trended with the median of previous
        transitions' deltas so repeated actions converge on their treatment
        effect instead of the game's natural drift.  With
        ``memory_credit_lag > 0`` each batch also receives a second sample
        once its multi-turn window closes, which is what lets slowly
        implemented policies (introductions ramping up over several turns)
        collect credit their first transition cannot show yet.
        """

        assert self.treatment_memory is not None
        states = self.context.states
        if len(states) < 2:
            return
        names = self.feature_encoder.feature_names
        current_row = dict(zip(names, states[-1].row(names)))
        previous_row = dict(zip(names, states[-2].row(names)))
        latest_poll = current_row[ELECTORAL_SUPPORT_FEATURE]
        previous_poll = previous_row[ELECTORAL_SUPPORT_FEATURE]
        noop_row = self._noop_forecast_row
        self._noop_forecast_row = None
        if noop_row is not None:
            # The world model's counterfactual: what each metric was expected
            # to do this transition without any action.
            drift = float(noop_row.get(ELECTORAL_SUPPORT_FEATURE, previous_poll)) - previous_poll
            fiscal_drift = (
                self._normalised_balance_delta(previous_row, noop_row)
            )
        else:
            history = self._poll_deltas()[:-1]
            drift = self.treatment_memory.drift_estimate(
                history, window=self.memory_drift_window
            )
            fiscal_drift = self.treatment_memory.drift_estimate(
                self._fiscal_history, window=self.memory_drift_window
            )
        observed_delta = latest_poll - previous_poll

        expenditure_ref = max(
            abs(float(previous_row.get("finance/total_expenditure", 0.0))),
            simulator.EPSILON,
        )
        fiscal_delta = (
            self._balance(current_row) - self._balance(previous_row)
        ) / expenditure_ref
        self._fiscal_history.append(fiscal_delta)
        del self._fiscal_history[: -self.memory_drift_window]
        # Age outstanding windows first; a batch pushed below never pays for
        # the transition it was executed in.
        for entry in self._pending_credits:
            entry[0] = float(entry[0]) - 1
        due = [entry for entry in self._pending_credits if entry[0] <= 0]
        self._pending_credits = [
            entry for entry in self._pending_credits if entry[0] > 0
        ]
        for entry in due:
            window_delta = latest_poll - entry[1]
            self.treatment_memory.record(
                entry[2],
                observed_delta=window_delta,
                drift=drift * self.memory_credit_lag,
            )
        if actions and self.memory_credit_lag > 0:
            self._pending_credits.append(
                [float(self.memory_credit_lag), latest_poll, tuple(actions)]
            )
        self.treatment_memory.record(
            actions,
            observed_delta=observed_delta,
            drift=drift,
            fiscal_delta=fiscal_delta,
            fiscal_drift=fiscal_drift,
        )

    @staticmethod
    def _balance(row: Mapping[str, float]) -> float:
        return (
            float(row.get("finance/total_income", 0.0))
            - float(row.get("finance/total_expenditure", 0.0))
        )

    @staticmethod
    def _normalised_balance_delta(
        baseline: Mapping[str, float], predicted: Mapping[str, float]
    ) -> float:
        """Counterfactual balance change, normalised by expenditure."""

        reference = max(
            abs(float(baseline.get("finance/total_expenditure", 0.0))),
            simulator.EPSILON,
        )
        return (
            (
                float(predicted.get("finance/total_income", 0.0))
                - float(predicted.get("finance/total_expenditure", 0.0))
            )
            - (
                float(baseline.get("finance/total_income", 0.0))
                - float(baseline.get("finance/total_expenditure", 0.0))
            )
        ) / reference

    @property
    def trace(self) -> ForecastTrace:
        return ForecastTrace(self.context, tuple(self.decisions))

    def save_trace(self, path: str | Path) -> None:
        self.trace.save(path)

    def load_state(self, path: str | Path) -> None:
        super().load_state(path)
        self.feature_encoder = self._default_encoder(self.state)
        self.role_schema = FeatureRoleSchema.from_encoder(self.feature_encoder)
        self.context = self._initial_context()
        self.decisions.clear()
        self.last_decision = None
        self._pending_credits = []


__all__ = [
    "ActionConditionedForecaster",
    "ActionRecord",
    "AutoregressiveContext",
    "Chronos2Forecaster",
    "DEFAULT_FORECAST_WEIGHTS",
    "ELECTORAL_SUPPORT_FEATURE",
    "EmpiricalActionForecaster",
    "FeatureRoleSchema",
    "ForecastDecision",
    "ForecastModelInput",
    "ForecastTrace",
    "PersistenceForecaster",
    "StateFeatureEncoder",
    "StateForecast",
    "StateSnapshot",
    "TimeSeriesPolicyAgent",
    "TREATMENT_FEATURE_PREFIX",
    "diverse_warmup_plan",
    "forecast_debt_gdp_growth",
    "player_visible_value_names",
    "score_forecast",
]
