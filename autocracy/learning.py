"""Cross-episode treatment-effect memory for forecast-driven agents.

Zero-shot candidate rankings from the Chronos forecaster are nearly flat
(``reports/chronos-2-small-vs-oracle.md`` measures a predicted poll spread of
7.5e-5 against true swings of ±6 points), so a purely model-ranked campaign is
noise.  This module supplies the missing signal from the one source a player
legitimately has — observed post-turn states:

* every executed transition appends a *de-trended* change in electoral support
  (observed delta minus the recent natural drift) to a persistent per-action
  table, so repeats of one action average toward its treatment effect;
* the table survives across episodes.  Each episode restarts the mission from
  the same save, which makes cross-episode pooling stationary, and lets the
  agent return a genuinely better policy every restart;
* an optimism bonus for rarely-tried actions turns the roster into a
  cost-aware bandit: early episodes deliberately spread interventions across
  untried sliders, later episodes exploit the highest measured effects;
* recency weighting (:attr:`decay`) keeps estimates current when a campaign
  continues past the first election and the world drifts.

Everything here is plain Python and serializable, so experiments can resume a
trained memory from disk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

from .models import PolicyAction
from .timeseries import ActionRecord


def action_key(action: PolicyAction | ActionRecord) -> tuple[str, str, float]:
    """Return the canonical memory key for one policy move."""

    return (
        action.policy_name,
        action.action_type or "",
        round(float(action.delta), 6),
    )


def _median(values: Sequence[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return 0.5 * (ordered[middle - 1] + ordered[middle])


@dataclass(slots=True)
class _EffectSample:
    """One de-trended observation of an action's effect."""

    value: float
    seq: int


@dataclass(slots=True)
class TreatmentEffectMemory:
    """Learned per-action effect on electoral support, pooled across episodes.

    ``record`` stores ``observed_delta - drift`` for every action of a
    transition; ``effect_total`` scores a candidate batch with recency-weighted
    means; ``explore_total`` pays a bonus that shrinks like ``1/sqrt(1+n)``
    as an action accumulates observations and is discounted by political-
    capital cost so expensive moves are probed deliberately rather than
    compulsively.  A second channel (:meth:`estimate_fiscal`,
    :meth:`fiscal_total`) stores de-trended changes in the visible budget
    balance per action, so prudence can be scored from what the treasury
    actually did instead of from model guesses.  Annealing the global
    :attr:`exploration_bonus` across episodes (the experiment divides it by
    ``sqrt(episode + 1)``) turns the campaign sequence into explore-then-
    exploit.
    """

    decay: float = 0.9
    max_samples: int = 64
    exploration_bonus: float = 0.0
    reference_cost: float = 10.0
    family_shrinkage: float = 0.8
    # Level-keyed channels: instead of crediting the *gesture* (raise/cancel
    # of some delta), credit the *resulting slider level*.  A cancel is then
    # literally "set the level to 0", so its predicted poll effect reverses
    # the sampled contributions accumulated on the way up.  Buckets follow
    # ``level_step``.
    level_keys: bool = False
    level_step: float = 0.05
    level_effects: dict[tuple[str, float], list[_EffectSample]] = field(
        default_factory=dict
    )
    fiscal_level_effects: dict[tuple[str, float], list[_EffectSample]] = field(
        default_factory=dict
    )
    effects: dict[tuple[str, str, float], list[_EffectSample]] = field(
        default_factory=dict
    )
    # Same shape as ``effects`` but storing de-trended changes in the
    # visible budget balance (normalised by expenditure) per action.
    fiscal_effects: dict[tuple[str, str, float], list[_EffectSample]] = field(
        default_factory=dict
    )
    transitions: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.decay <= 1.0:
            raise ValueError("decay must be in (0, 1]")
        if self.max_samples < 1:
            raise ValueError("max_samples must be at least one")
        if self.exploration_bonus < 0.0:
            raise ValueError("exploration_bonus must be non-negative")
        if self.reference_cost <= 0.0:
            raise ValueError("reference_cost must be positive")
        if not 0.0 <= self.family_shrinkage <= 1.0:
            raise ValueError("family_shrinkage must be in [0, 1]")
        if self.level_step <= 0.0:
            raise ValueError("level_step must be positive")

    @property
    def known_actions(self) -> int:
        return len(self.effects)

    def record(
        self,
        actions: Sequence[PolicyAction | ActionRecord],
        *,
        observed_delta: float,
        drift: float = 0.0,
        fiscal_delta: float | None = None,
        fiscal_drift: float = 0.0,
    ) -> None:
        """Store the de-trended effect of one executed transition.

        ``observed_delta``/``drift`` feed the electoral-support channel;
        when ``fiscal_delta`` is supplied it records the balance channel the
        same way (already normalised by the caller).
        """

        effect = float(observed_delta) - float(drift)
        if not math.isfinite(effect):
            return
        self.transitions += 1
        for action in actions:
            key = action_key(action)
            samples = self.effects.setdefault(key, [])
            samples.append(_EffectSample(value=effect, seq=self.transitions))
            del samples[: len(samples) - self.max_samples]
        if fiscal_delta is None:
            return
        fiscal_effect = float(fiscal_delta) - float(fiscal_drift)
        if not math.isfinite(fiscal_effect):
            return
        for action in actions:
            key = action_key(action)
            samples = self.fiscal_effects.setdefault(key, [])
            samples.append(
                _EffectSample(value=fiscal_effect, seq=self.transitions)
            )
            del samples[: len(samples) - self.max_samples]

    def visits(self, action: PolicyAction | ActionRecord) -> int:
        samples = self.effects.get(action_key(action))
        return len(samples) if samples else 0

    def level_bucket(self, level: float) -> float:
        """Discretise a slider level onto the fixed bucket grid."""

        return round(float(level) / self.level_step) * self.level_step

    def resulting_level(
        self, action: PolicyAction | ActionRecord, current: float
    ) -> float:
        """The effective slider level an action leaves the policy at.

        A cancellation sets the *contribution* level to zero even though the
        neuron value freezes: the policy stops contributing immediately.
        """

        if getattr(action, "action_type", None) == "cancel":
            return 0.0
        return max(0.0, min(1.0, float(current) + float(action.delta)))

    def record_level(
        self,
        actions: Sequence[PolicyAction | ActionRecord],
        *,
        observed_delta: float,
        drift: float = 0.0,
        fiscal_delta: float | None = None,
        fiscal_drift: float = 0.0,
        current_levels: Mapping[str, float] | None = None,
    ) -> None:
        """Credit each action to the slider level it *arrives at*.

        Samples keyed by ``(policy, level bucket)`` are the marginal poll
        change of moving the slider to that level.  Summing the buckets on
        any path from ``current`` to ``target`` reconstructs the total
        contribution of that level range, so a cancel (target 0) reverses
        everything observed while the slider was built up.
        """

        if not self.level_keys:
            return
        effect = float(observed_delta) - float(drift)
        if not math.isfinite(effect):
            return
        current_levels = current_levels or {}
        fiscal_effect = None
        if fiscal_delta is not None:
            fiscal_effect = float(fiscal_delta) - float(fiscal_drift)
            if not math.isfinite(fiscal_effect):
                fiscal_effect = None
        for action in actions:
            current = current_levels.get(action.policy_name, 0.0)
            bucket = self.level_bucket(self.resulting_level(action, current))
            key = (action.policy_name, bucket)
            samples = self.level_effects.setdefault(key, [])
            samples.append(_EffectSample(value=effect, seq=self.transitions))
            del samples[: len(samples) - self.max_samples]
            if fiscal_effect is not None:
                fiscal_samples = self.fiscal_level_effects.setdefault(key, [])
                fiscal_samples.append(
                    _EffectSample(value=fiscal_effect, seq=self.transitions)
                )
                del fiscal_samples[: len(fiscal_samples) - self.max_samples]

    def level_effect(
        self, policy_name: str, current: float, target: float
    ) -> float:
        """Predicted poll change of moving one slider from ``current`` to
        ``target``, by summing the sampled arrivals on the crossed path.

        Raises sum the (positive) arrival samples; lowers and cancels reverse
        them.  Returns 0 when no bucket on the path has been sampled.
        """

        if not self.level_keys:
            return 0.0
        lo, hi = min(float(current), float(target)), max(
            float(current), float(target)
        )
        sign = 1.0 if target >= current else -1.0
        total = 0.0
        for (name, bucket), samples in self.level_effects.items():
            if name != policy_name:
                continue
            if lo < bucket <= hi:
                mean = self._weighted_mean(samples)
                if mean is not None:
                    total += mean
        return sign * total

    def fiscal_level_effect(
        self, policy_name: str, current: float, target: float
    ) -> float:
        """Predicted balance effect of a slider move, level-path summed."""

        if not self.level_keys:
            return 0.0
        lo, hi = min(float(current), float(target)), max(
            float(current), float(target)
        )
        sign = 1.0 if target >= current else -1.0
        total = 0.0
        for (name, bucket), samples in self.fiscal_level_effects.items():
            if name != policy_name:
                continue
            if lo < bucket <= hi:
                mean = self._weighted_mean(samples)
                if mean is not None:
                    total += mean
        return sign * total

    def level_visits(self, policy_name: str, target: float) -> int:
        return len(self.level_effects.get((policy_name, self.level_bucket(target)), []))

    def estimate(self, action: PolicyAction | ActionRecord) -> float | None:
        """Return the recency-weighted mean effect, or ``None`` if unseen.

        A signature never tried itself inherits a shrunk estimate from its
        ``(policy, direction)`` family: once raising one slider step proved
        popular, the *next* step of the same slider should be treated as
        promising rather than as unknown.  Direct evidence always wins.
        """

        return self._channel_estimate(self.effects, action)

    def estimate_fiscal(
        self, action: PolicyAction | ActionRecord
    ) -> float | None:
        """Recency-weighted balance effect (expenditure-normalised share)."""

        return self._channel_estimate(self.fiscal_effects, action)

    def fiscal_total(
        self, actions: Sequence[PolicyAction | ActionRecord]
    ) -> float:
        """Sum of learned balance effects for a candidate batch."""

        total = 0.0
        for action in actions:
            estimate = self.estimate_fiscal(action)
            if estimate is not None:
                total += estimate
        return total

    def _weighted_mean(
        self, samples: Sequence[_EffectSample]
    ) -> float | None:
        now = self.transitions
        weights = [self.decay ** (now - sample.seq) for sample in samples]
        total = sum(weights)
        if total <= 0.0:
            return None
        return sum(w * s.value for w, s in zip(weights, samples)) / total

    def _channel_estimate(
        self,
        table: dict[tuple[str, str, float], list[_EffectSample]],
        action: PolicyAction | ActionRecord,
    ) -> float | None:
        key = action_key(action)
        samples = table.get(key)
        if samples:
            return self._weighted_mean(samples)
        pool: list[_EffectSample] = []
        for other_key, family_samples in table.items():
            if other_key[0] == key[0] and other_key[1] == key[1]:
                pool.extend(family_samples)
        if not pool:
            return None
        family = self._weighted_mean(pool)
        if family is None:
            return None
        return self.family_shrinkage * family

    def drift_estimate(self, history: Sequence[float], *, window: int) -> float:
        """Return the natural metric drift used to de-trend new observations.

        ``history`` holds per-transition observed deltas oldest-first and must
        exclude the transition being scored; the median of the most recent
        ``window`` entries represents what the metric would have done anyway.
        """

        recent = list(history[-window:]) if window > 0 else []
        if not recent:
            return 0.0
        return _median([float(value) for value in recent])

    def effect_total(
        self, actions: Sequence[PolicyAction | ActionRecord]
    ) -> float:
        """Sum of learned effects for a candidate batch (0 for unseen moves)."""

        total = 0.0
        for action in actions:
            estimate = self.estimate(action)
            if estimate is not None:
                total += estimate
        return total

    def _explore_bonus(self, action: PolicyAction | ActionRecord, cost: float) -> float:
        if self.exploration_bonus <= 0.0:
            return 0.0
        # Uncertainty shrinks purely with visits; callers anneal the global
        # bonus magnitude across episodes (a log(transitions) factor here
        # would make exploration *intensify* as memory grows).
        uncertainty = 1.0 / math.sqrt(1.0 + self.visits(action))
        affordability = 1.0 / (1.0 + max(cost, 0.0) / self.reference_cost)
        return self.exploration_bonus * uncertainty * affordability

    def explore_total(
        self,
        actions: Sequence[PolicyAction | ActionRecord],
        *,
        costs: Sequence[float] | None = None,
    ) -> float:
        """Optimism bonus for a candidate batch given per-action costs."""

        if self.exploration_bonus <= 0.0:
            return 0.0
        if costs is None:
            costs = [0.0] * len(actions)
        return sum(
            self._explore_bonus(action, cost)
            for action, cost in zip(actions, costs)
        )

    def ranked_actions(self, limit: int = 10) -> list[tuple[tuple[str, str, float], float, int]]:
        """Best-known actions by estimated effect, for reporting."""

        ranked = []
        for key, samples in self.effects.items():
            holder = _ActionProxy(key)
            estimate = self.estimate(holder)
            if estimate is not None:
                ranked.append((key, estimate, len(samples)))
        ranked.sort(key=lambda item: (-item[1], item[0]))
        return ranked[:limit]

    def to_dict(self) -> dict[str, object]:
        def serialise(table):
            return {
                "|".join(str(part) for part in key): [
                    {"value": sample.value, "seq": sample.seq}
                    for sample in samples
                ]
                for key, samples in table.items()
            }

        return {
            "format": "autocracy-treatment-memory-v1",
            "decay": self.decay,
            "max_samples": self.max_samples,
            "exploration_bonus": self.exploration_bonus,
            "reference_cost": self.reference_cost,
            "family_shrinkage": self.family_shrinkage,
            "level_keys": self.level_keys,
            "level_step": self.level_step,
            "transitions": self.transitions,
            "effects": serialise(self.effects),
            "fiscal_effects": serialise(self.fiscal_effects),
            "level_effects": serialise(self.level_effects),
            "fiscal_level_effects": serialise(self.fiscal_level_effects),
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TreatmentEffectMemory":
        def deserialise(raw) -> dict[tuple[str, str, float], list[_EffectSample]]:
            table: dict[tuple[str, str, float], list[_EffectSample]] = {}
            for joined, samples in dict(raw or {}).items():
                parts = str(joined).split("|")
                key = (parts[0], parts[1], float(parts[2]))
                table[key] = [
                    _EffectSample(value=float(sample["value"]), seq=int(sample["seq"]))
                    for sample in list(samples)
                ]
            return table

        def deserialise_levels(
            raw,
        ) -> dict[tuple[str, float], list[_EffectSample]]:
            table: dict[tuple[str, float], list[_EffectSample]] = {}
            for joined, samples in dict(raw or {}).items():
                parts = str(joined).split("|")
                key = (parts[0], float(parts[1]))
                table[key] = [
                    _EffectSample(value=float(sample["value"]), seq=int(sample["seq"]))
                    for sample in list(samples)
                ]
            return table

        memory = cls(
            decay=float(payload.get("decay", 0.9)),
            max_samples=int(payload.get("max_samples", 64)),
            exploration_bonus=float(payload.get("exploration_bonus", 0.0)),
            reference_cost=float(payload.get("reference_cost", 10.0)),
            family_shrinkage=float(payload.get("family_shrinkage", 0.8)),
            level_keys=bool(payload.get("level_keys", False)),
            level_step=float(payload.get("level_step", 0.05)),
            effects=deserialise(payload.get("effects")),
            fiscal_effects=deserialise(payload.get("fiscal_effects")),
            level_effects=deserialise_levels(payload.get("level_effects")),
            fiscal_level_effects=deserialise_levels(
                payload.get("fiscal_level_effects")
            ),
            transitions=int(payload.get("transitions", 0)),
        )
        return memory

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: str | Path) -> "TreatmentEffectMemory":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(payload)


class _ActionProxy:
    """Duck-typed stand-in so ``estimate`` can accept a stored key."""

    __slots__ = ("policy_name", "action_type", "delta")

    def __init__(self, key: tuple[str, str, float]) -> None:
        self.policy_name, self.action_type, self.delta = key


__all__ = [
    "TreatmentEffectMemory",
    "action_key",
]
