from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx


@dataclass(slots=True)
class BudgetModifier:
    """Describes how another node/policy scales a budget line."""

    source: str
    expression: str


@dataclass(slots=True)
class Effect:
    """Represents a directed influence defined in the Democracy 3 data files."""

    source: str
    target: str
    expression: str
    inertia: Optional[float] = None
    effect_id: Optional[str] = None


@dataclass(slots=True)
class NodeDefinition:
    """Metadata describing a simulation node (statistic, gauge, voter metric)."""

    name: str
    display_name: str
    description: str
    category: str
    default: float
    minimum: float
    maximum: float
    emotion: str = ""
    icon: str = ""


@dataclass(slots=True)
class PolicyDefinition:
    """Configuration for a policy slider."""

    name: str
    display_name: str
    description: str
    slider: str
    introduce_cost: float
    cancel_cost: float
    raise_cost: float
    lower_cost: float
    department: str
    flags: List[str]
    min_cost: float = 0.0
    max_cost: float = 0.0
    cost_multiplier: str = ""
    implementation_time: float = 0.0
    min_income: float = 0.0
    max_income: float = 0.0
    income_multiplier: str = ""
    cost_multipliers: List[BudgetModifier] = field(default_factory=list)
    income_multipliers: List[BudgetModifier] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)


@dataclass(slots=True)
class SliderDefinition:
    """Metadata describing how a policy slider behaves."""

    name: str
    kind: str
    labels: List[str] = field(default_factory=list)
    min_value: float = 0.0
    max_value: float = 1.0

    def allowed_levels(self) -> List[float]:
        if self.kind.upper() != "DISCRETE":
            return []
        if not self.labels:
            return [0.0, 1.0]
        steps = max(len(self.labels) - 1, 1)
        return [index / steps for index in range(len(self.labels))]


@dataclass(slots=True)
class CountrySetup:
    """Settings read from the mission folder for a country."""

    name: str
    currency: str
    description: str
    policy_levels: Dict[str, float]
    options: List[str] = field(default_factory=list)
    stats: Dict[str, str] = field(default_factory=dict)
    overrides: List[dict] = field(default_factory=list)


@dataclass(slots=True)
class SimulationData:
    """All static metadata required to simulate a turn."""

    nodes: Dict[str, NodeDefinition]
    policies: Dict[str, PolicyDefinition]
    sliders: Dict[str, SliderDefinition]
    situations: Dict[str, "SituationDefinition"]
    graph: nx.DiGraph
    sim_config: Dict[str, float]
    gamedata_root: Path


@dataclass(slots=True)
class SimulationState:
    """Mutable per-turn state for a single country."""

    country: str
    turn: int
    values: Dict[str, float]
    policies: Dict[str, float]
    political_capital: float
    effects: Dict[str, float]
    situations: Dict[str, float] = field(default_factory=dict)
    active_situations: List[str] = field(default_factory=list)
    response_factors: Dict[str, float] = field(default_factory=dict)
    policy_costs: Dict[str, float] = field(default_factory=dict)
    policy_incomes: Dict[str, float] = field(default_factory=dict)
    total_expenditure: float = 0.0
    total_income: float = 0.0


@dataclass(slots=True)
class PolicyAction:
    """A requested change to a policy slider."""

    policy_name: str
    delta: float


@dataclass(slots=True)
class PolicyActionOption:
    """Describes an available policy adjustment from the current state."""

    policy_name: str
    action_type: str  # introduce | cancel | raise | lower
    delta: float
    resulting_level: float
    cost: float
    implementation_time: float


@dataclass(slots=True)
class SituationDefinition:
    """Metadata describing a situation that can trigger during simulation."""

    name: str
    display_name: str
    description: str
    category: str
    icon: str
    positive: bool
    start_trigger: float
    stop_trigger: float
    cost: float
    inputs: List[Effect] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
