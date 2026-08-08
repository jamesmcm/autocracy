from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import networkx as nx


@dataclass(slots=True)
class SimulationConfig:
    """Toggle the game's stochastic systems.

    All systems default to **off**: the shipped save pair is reproduced
    deterministically, and only the systems explicitly enabled by a caller
    mutate the state between turns.  ``random_seed`` makes enabled systems
    reproducible across runs.
    """

    random_events: bool = False
    dilemmas: bool = False
    pressure_group_events: bool = False
    assassinations: bool = False
    random_seed: int = 0
    # Deterministic minister-loyalty subsystem.  When enabled, ministers
    # gain/lose loyalty every turn from their satisfaction with the enacted
    # policies, which feeds the per-turn political-capital income.  Disabling
    # it keeps the capital income at the value loaded from the save.
    minister_loyalty: bool = False


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
class EffectHistory:
    """Serialized effect memory recovered from a Democracy 3 save."""

    source: str
    target: str
    values: List[float] = field(default_factory=list)
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
    economic_cycle_start: float = 0.0
    wealth_mod: float = 1.0
    difficulty: float = 0.5
    min_income: float = 0.0
    max_income: float = 0.0
    min_gdp: float = 0.0
    max_gdp: float = 0.0
    starting_debt: float = 0.0


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
    effect_histories: List[EffectHistory] = field(default_factory=list)
    situations: Dict[str, float] = field(default_factory=dict)
    active_situations: List[str] = field(default_factory=list)
    response_factors: Dict[str, float] = field(default_factory=dict)
    policy_costs: Dict[str, float] = field(default_factory=dict)
    policy_incomes: Dict[str, float] = field(default_factory=dict)
    total_expenditure: float = 0.0
    total_income: float = 0.0
    global_economy_position: float = 0.0
    voter_values: Dict[str, float] = field(default_factory=dict)
    voter_percentages: Dict[str, float] = field(default_factory=dict)
    voter_frequencies: Dict[str, float] = field(default_factory=dict)
    policy_implementations: Dict[str, float] = field(default_factory=dict)
    policy_active: Dict[str, bool] = field(default_factory=dict)
    policy_cost_multipliers: Dict[str, float] = field(default_factory=dict)
    policy_income_multipliers: Dict[str, float] = field(default_factory=dict)
    policy_cost_scalars: Dict[str, float] = field(default_factory=dict)
    policy_income_scalars: Dict[str, float] = field(default_factory=dict)
    effect_throttles: Dict[str, float] = field(default_factory=dict)
    policy_desired_throttles: Dict[str, float] = field(default_factory=dict)
    ministerial_effectiveness: Dict[str, float] = field(default_factory=dict)
    ministerial_competence: Dict[str, float] = field(default_factory=dict)
    political_capital_income: float = 0.0
    # Finance-manager runtime fields.  The game serializes these in the
    # <finances>/<creditrating> blocks and recomputes them every turn:
    #   debt = previous debt + (expenditure - income) charged last turn
    #   credit_rating = credit-rating derived from the debt-to-GDP ratio
    #   turns_since_credit = countdown to the next rating recomputation
    #   ministerial_experience/suitability = per-department inputs to the
    #       competence/efficiency scalars (experience grows each turn)
    debt: float = 0.0
    credit_rating: int = 0
    turns_since_credit: int = 0
    # The interest rate currently charged on the debt.  The game derives it
    # from the credit rating each turn (the very first turn keeps the rate
    # it started with, which the serialized <finances> n_3 records).
    interest_rate: float = 0.0
    ministerial_experience: Dict[str, float] = field(default_factory=dict)
    ministerial_suitability: Dict[str, float] = field(default_factory=dict)
    # Per-department minister loyalty, used to recompute the per-turn
    # political-capital income (it drifts with loyalty).
    ministerial_loyalty: Dict[str, float] = field(default_factory=dict)
    # Minister volatility and current satisfaction (the game's minister
    # ``volatility`` and ``value`` fields).  Satisfaction drives the loyalty
    # dynamics in ``SIM_Minister::ProcessLoyalty``.
    ministerial_volatility: Dict[str, float] = field(default_factory=dict)
    ministerial_value: Dict[str, float] = field(default_factory=dict)
    # Human-readable record of stochastic-system firings this run (events,
    # dilemmas, attacks).  Only populated when the corresponding
    # ``SimulationConfig`` toggle is enabled.
    event_log: List[str] = field(default_factory=list)
    # Cross-turn bookkeeping for the attack and pressure-group systems.
    fired_plots: List[str] = field(default_factory=list)
    group_threats: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyAction:
    """A requested change to a policy slider.

    ``action_type`` records whether the change is a slider move (``raise`` /
    ``lower``), a fresh introduction or a true switch-off (``cancel``).  The
    game treats dragging a slider down to its floor as a ``lower`` (charged
    at the lower cost, policy stays active), which is distinct from a
    ``cancel`` (charged at the cancel cost, policy deactivates).  When the
    type is not supplied the simulator infers it from the slider metadata.
    """

    policy_name: str
    delta: float
    action_type: Optional[str] = None


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
    default: float = 0.0
    prerequisites: List[str] = field(default_factory=list)
    inputs: List[Effect] = field(default_factory=list)
    effects: List[Effect] = field(default_factory=list)
