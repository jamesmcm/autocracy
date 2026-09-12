"""Context disagreement and marginal downside risk for the first UCB experiment.

Window sensitivity is an epistemic proxy, not a calibrated Bayesian posterior.
Quantile paths are used only for per-step shortfalls, never reward quantiles.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import TYPE_CHECKING, Sequence

if TYPE_CHECKING:
    from .timeseries import ForecastModelInput, StateForecast


@dataclass(frozen=True, slots=True)
class UncertaintyConfig:
    beta: float = 0.0
    members: int = 3
    min_context_fraction: float = 0.5
    risk_weight: float = 0.0
    risk_quantile: float = 0.05
    risk_floor: float = 0.5

    def __post_init__(self) -> None:
        for name in ("beta", "risk_weight"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not isinstance(self.members, int) or self.members < 1:
            raise ValueError("members must be a positive integer")
        if self.beta and self.members < 2:
            raise ValueError("positive beta requires at least two context members")
        if not 0 < self.min_context_fraction < 1:
            raise ValueError("min_context_fraction must lie in (0, 1)")
        if self.risk_quantile not in (0.05, 0.1):
            raise ValueError("risk_quantile must be 0.05 or 0.1")
        if not math.isfinite(self.risk_floor) or not 0 <= self.risk_floor <= 1:
            raise ValueError("risk_floor must lie in [0, 1]")

    @property
    def active(self) -> bool:
        return bool(self.beta or self.risk_weight)

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def context_windows(
    inputs: Sequence[ForecastModelInput], config: UncertaintyConfig
) -> list[list[ForecastModelInput]]:
    """Paired chronological suffixes, including full context as member zero.

    All candidates receive the same history perturbation and retain today's
    observed row and their original pending actions. Rounded duplicate windows
    in short histories are dropped. No candidate-sampling RNG is consumed.
    """

    if not inputs:
        return []
    length = len(inputs[0].history)
    if any(item.history != inputs[0].history or item.turns != inputs[0].turns
           or item.action_history != inputs[0].action_history for item in inputs):
        raise ValueError("context ensemble candidates must share observed history")
    sizes = [length]
    if config.beta:
        for index in range(1, config.members):
            fraction = 1 - (1 - config.min_context_fraction) * index / (config.members - 1)
            size = min(length, max(2, math.ceil(length * fraction)))
            if size not in sizes:
                sizes.append(size)
    return [
        [replace(
            item,
            history=item.history[-size:],
            turns=item.turns[-size:],
            action_history=item.action_history[-(size - 1):] if size > 1 else (),
        ) for item in inputs]
        for size in sizes
    ]


def delta_moments(candidate: Sequence[float], noop: Sequence[float]) -> tuple[float, float]:
    """Population mean/std of paired treatment effects, not raw forecasts."""

    if not candidate or len(candidate) != len(noop):
        raise ValueError("candidate and noop scores need matching nonempty members")
    deltas = [a - b for a, b in zip(candidate, noop)]
    if not all(math.isfinite(value) for value in deltas):
        raise ValueError("ensemble objective returned non-finite treatment effects")
    mean = sum(deltas) / len(deltas)
    return mean, math.sqrt(sum((value - mean) ** 2 for value in deltas) / len(deltas))


def marginal_poll_shortfall(forecast: StateForecast, config: UncertaintyConfig) -> float:
    """Worst per-step poll q05/q10 shortfall below the configured floor.

    This is a marginal stress indicator, not a quantile of cumulative reward
    or an estimated probability of losing an election.
    """

    if not config.risk_weight:
        return 0.0
    path = forecast.quantiles.get(config.risk_quantile)
    if path is None or len(path) != forecast.horizon:
        raise ValueError(f"risk penalty requires q{config.risk_quantile:g} forecasts")
    values = [row.get("politics/poll_rate", math.nan) for row in path]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("risk penalty requires finite poll quantiles at every step")
    return max(max(0.0, config.risk_floor - value) for value in values)
