from __future__ import annotations

import ast
import json
import math
import re
import struct
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from . import data_loader
from .models import (
    BudgetModifier,
    Effect,
    EffectHistory,
    NodeDefinition,
    PolicyAction,
    PolicyActionOption,
    PolicyDefinition,
    SimulationConfig,
    SimulationData,
    SimulationState,
    SliderDefinition,
    Voter,
)

DEFAULT_GAMEDATA = Path(__file__).resolve().parent.parent / "gamedata" / "data"
ALLOWED_FUNCS = {"min": min, "max": max, "abs": abs}
DEFAULT_PERCENTAGE_STEP = 0.05
EPSILON = 1e-6


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _f32(value: float) -> float:
    """Round to the game's single-precision finance arithmetic.

    The executable computes every money line with SSE float32 ops, so a
    float64 sum drifts off the serialized totals by ~0.02 once a few hundred
    thousand currency units are involved.  Rounding each product/accumulator
    to float32 reproduces the shipped <finances> block exactly.
    """
    return struct.unpack("f", struct.pack("f", float(value)))[0]


def _ensure_node(graph: nx.DiGraph, nodes: Dict[str, NodeDefinition], name: str) -> None:
    if name in graph:
        return
    placeholder = NodeDefinition(
        name=name,
        display_name=name,
        description="Auto-generated placeholder node",
        category="PLACEHOLDER",
        default=0.0,
        minimum=-1.0,
        maximum=1.0,
    )
    nodes[name] = placeholder
    graph.add_node(name, kind="placeholder", data=placeholder)


def _attach_effect(graph: nx.DiGraph, effect: Effect) -> None:
    data = graph.get_edge_data(effect.source, effect.target, default=None)
    if data:
        data["effects"].append(effect)
    else:
        graph.add_edge(effect.source, effect.target, effects=[effect])


def _sanitize_expression(expr: str) -> str:
    expr = expr.replace("^", "**")
    # The game feeds malformed numeric buffers such as ``0.0.5`` to
    # ``strtod``.  It therefore reads the token as 0.0; treating it as 0.5
    # creates a large, silent parity error in ChildBenefit -> Equality.
    expr = re.sub(r"(?<![\w.])([+-]?\d+\.\d+)\.\d+", r"\1", expr)
    # The game's parser reads a bare ``-0.1(-0.6*x)`` as the concatenation
    # ``-0.1 + (-0.6*x)`` (i.e. -0.1 - 0.6*x), not as implicit
    # multiplication.  Only InnerCityRiots' CCTV input uses the pattern, but
    # it shifts the situation latent enough to gate activation.
    expr = re.sub(r"(\d)\(", r"\1+(", expr)
    cleaned_chars = []
    open_parens = 0
    for ch in expr:
        if ch == "(":
            open_parens += 1
            cleaned_chars.append(ch)
        elif ch == ")":
            if open_parens == 0:
                continue
            open_parens -= 1
            cleaned_chars.append(ch)
        else:
            cleaned_chars.append(ch)
    cleaned_expr = "".join(cleaned_chars)
    if open_parens > 0:
        cleaned_expr += ")" * open_parens
    return cleaned_expr


def _get_slider(
    data: SimulationData, policy: PolicyDefinition
) -> SliderDefinition:
    fallback = data.sliders.get("default") or SliderDefinition(
        name="default",
        kind="DISCRETE",
        labels=["Off", "On"],
    )
    return data.sliders.get(policy.slider, fallback) if policy.slider else fallback


def _discrete_levels(slider: SliderDefinition) -> List[float]:
    levels = slider.allowed_levels()
    if levels:
        return sorted(levels)
    return [0.0, 1.0]


def _next_level(
    current: float, slider: SliderDefinition, direction: str
) -> Optional[float]:
    kind = slider.kind.upper()
    step = DEFAULT_PERCENTAGE_STEP
    if kind == "PERCENTAGE":
        if direction == "raise":
            if current >= 1.0 - EPSILON:
                return None
            return _clamp(current + step, 0.0, 1.0)
        if current <= EPSILON:
            return None
        return _clamp(current - step, 0.0, 1.0)
    levels = _discrete_levels(slider)
    if direction == "raise":
        for level in levels:
            if level > current + EPSILON:
                return level
        if current >= 1.0 - EPSILON:
            return None
        return _clamp(current + step, 0.0, 1.0)
    # lower.  A discrete slider may have level 0 as its genuine floor
    # (e.g. prisons' OVERCROWDED CELLS); reaching it is a ``lower``, not a
    # ``cancel``.  Only sliders whose level 0 is a true NONE/off state treat
    # the floor as a cancellation, and that is expressed through the action
    # type rather than the step ladder here.
    for level in reversed(levels):
        if level < current - EPSILON:
            return level
    fallback = current - step
    if fallback <= EPSILON:
        return None
    return _clamp(fallback, 0.0, 1.0)


def _default_slider_level(slider: SliderDefinition) -> float:
    levels = slider.allowed_levels()
    if levels:
        return levels[len(levels) // 2]
    return 0.5


def _is_uncancellable(policy: PolicyDefinition) -> bool:
    return any(flag.upper() == "UNCANCELLABLE" for flag in policy.flags)


def _validate_policy_level(
    slider: SliderDefinition, new_level: float
) -> None:
    if not (0.0 - EPSILON <= new_level <= 1.0 + EPSILON):
        raise ValueError("Policy level is outside the allowed 0-1 range.")


def _policy_action_type(
    current: float,
    target: float,
    policy: Optional[PolicyDefinition] = None,
    action_type: Optional[str] = None,
) -> str:
    if action_type in {"introduce", "cancel", "raise", "lower"}:
        return action_type
    active_current = current > EPSILON
    active_target = target > EPSILON
    if not active_current and active_target:
        return "introduce"
    if active_current and not active_target:
        # Dragging a slider down to its floor is a ``lower`` even when the
        # floor is level 0.  Only an explicit switch-off is a ``cancel``;
        # an uncancellable policy can never be switched off.
        if policy is not None and _is_uncancellable(policy):
            return "lower"
        return "cancel"
    if target > current:
        return "raise"
    if target < current:
        return "lower"
    return "noop"


def _policy_action_cost(
    policy: PolicyDefinition, current: float, target: float, action_type: Optional[str] = None
) -> tuple[float, str]:
    action_type = _policy_action_type(current, target, policy, action_type)
    if action_type == "introduce":
        return policy.introduce_cost, action_type
    if action_type == "cancel":
        return policy.cancel_cost, action_type
    if action_type == "raise":
        return policy.raise_cost, action_type
    if action_type == "lower":
        return policy.lower_cost, action_type
    return 0.0, action_type


def _budget_base_amount(min_value: float, max_value: float, level: float) -> float:
    return min_value + (max_value - min_value) * level


def _evaluate_budget_modifiers(
    modifiers: List[BudgetModifier],
    policy_level: float,
    context: Dict[str, float],
) -> float:
    if not modifiers:
        return 1.0
    total = 0.0
    for modifier in modifiers:
        if modifier.source == "_default_":
            x_value = policy_level
        else:
            x_value = context.get(modifier.source, 0.0)
        total += evaluate_expression(modifier.expression, x_value, context=context)
    return max(total, 0.0)


def _policy_cost_amount(
    policy: PolicyDefinition,
    level: float,
    context: Dict[str, float],
    *,
    multiplier: Optional[float] = None,
    scalar: float = 1.0,
    wealth_mod: float = 1.0,
) -> float:
    if level <= EPSILON:
        return 0.0
    base = _budget_base_amount(policy.min_cost, policy.max_cost, level)
    if multiplier is None:
        multiplier = _evaluate_budget_modifiers(policy.cost_multipliers, level, context)
    return base * scalar * wealth_mod * multiplier


def _policy_income_amount(
    policy: PolicyDefinition,
    level: float,
    context: Dict[str, float],
    *,
    multiplier: Optional[float] = None,
    scalar: float = 1.0,
    wealth_mod: float = 1.0,
) -> float:
    if level <= EPSILON:
        return 0.0
    base = _budget_base_amount(policy.min_income, policy.max_income, level)
    if multiplier is None:
        multiplier = _evaluate_budget_modifiers(policy.income_multipliers, level, context)
    return base * scalar * wealth_mod * multiplier


def _validate_expression(node: ast.AST) -> None:
    allowed_nodes = (
        ast.Expression,
        ast.BinOp,
        ast.UnaryOp,
        ast.Constant,
        ast.Name,
        ast.Call,
        ast.Load,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.Pow,
        ast.USub,
        ast.UAdd,
        ast.Mod,
    )
    if not isinstance(node, allowed_nodes):
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")
    for child in ast.iter_child_nodes(node):
        _validate_expression(child)


def evaluate_expression(
    expression: str, x: float, context: Optional[Dict[str, float]] = None
) -> float:
    """Safely evaluate a Democracy 3 equation."""

    if not expression:
        return 0.0
    sanitized = _sanitize_expression(expression.strip())
    tree = ast.parse(sanitized, mode="eval")
    _validate_expression(tree)
    allowed_names = {"x": x, **ALLOWED_FUNCS}
    if context:
        allowed_names.update(context)
    code = compile(tree, "<effect>", "eval")
    return float(eval(code, {"__builtins__": {}}, allowed_names))


def _seed_state_from_initial_save(
    state: SimulationState, data: SimulationData
) -> Optional["SaveGame"]:
    """If an initial save exists (e.g. uk0.xml) use it to seed node/policy values."""

    saves_root = data.gamedata_root.parent / "saves"
    save_path = saves_root / f"{state.country}0.xml"
    if not save_path.exists():
        return None
    from .savegame import parse_savegame  # late import to avoid cycles

    save = parse_savegame(save_path)
    for name, value in save.simvalues.items():
        node = data.nodes.get(name)
        if not node:
            continue
        state.values[name] = _clamp(value, node.minimum, node.maximum)
    for name, value in save.policies.items():
        if name in state.policies:
            state.policies[name] = _clamp(value, 0.0, 1.0)
    state.political_capital = save.political_capital
    # The initial save's points equal the active ministers' baseline
    # contribution for the fresh term. Retain that rate separately because
    # later turns may spend points before the next accrual.
    state.political_capital_income = save.political_capital
    for name, value in save.hidden_values.items():
        state.values[name] = value
    for name, value in save.voter_values.items():
        state.voter_values[name] = value
        state.values[name] = value
    for name, value in save.voter_percentages.items():
        state.voter_percentages[name] = value
        # Situation inputs use names such as ``Socialist_perc`` directly.
        # They are runtime neuron sources even though they are not ordinary
        # entries in simulation.csv.
        state.values[name] = value
    for name, value in save.voter_frequencies.items():
        state.voter_frequencies[name] = value
        state.values[name] = value
    state.voters = [
        Voter(groups=dict(v.groups), value=v.value, income=v.income, inincome=v.inincome)
        for v in save.voters
    ]
    state.policy_implementations = save.policy_implementations.copy()
    state.policy_active = save.policy_active.copy()
    state.policy_cost_multipliers = save.policy_cost_multipliers.copy()
    state.policy_income_multipliers = save.policy_income_multipliers.copy()
    state.policy_cost_scalars = save.policy_cost_scalars.copy()
    state.policy_income_scalars = save.policy_income_scalars.copy()
    state.effect_throttles = save.effect_throttles.copy()
    # SetSlider writes the policy neuron value into each output effect's
    # desired throttle.  The save stores the current throttle separately in
    # <effects>, so retain both sides of that delayed transition.
    state.policy_desired_throttles = save.policy_desired_throttles.copy()
    state.ministerial_effectiveness = save.ministerial_effectiveness.copy()
    state.ministerial_competence = save.ministerial_competence.copy()
    state.ministerial_experience = save.ministerial_experience.copy()
    state.ministerial_suitability = save.ministerial_suitability.copy()
    state.ministerial_loyalty = save.ministerial_loyalty.copy()
    state.ministerial_volatility = save.ministerial_volatility.copy()
    state.ministerial_value = save.ministerial_value.copy()
    state.ministerial_sympathies = {
        k: list(v) for k, v in save.ministerial_sympathies.items()
    }
    state.situations = save.situations.copy()
    state.active_situations = save.active_situations.copy()
    state.global_economy_position = save.global_economy_position
    state.debt = save.debt
    state.credit_rating = save.credit_rating
    state.turns_since_credit = save.turns_since_credit
    state.interest_rate = save.interest_rate
    # The finance-manager lines the game displays for this snapshot.  Keep
    # them separate from the per-policy history rings (which lag a turn).
    state.total_income = save.total_income
    state.total_expenditure = save.total_expenditure
    state.effect_histories = [
        EffectHistory(history.source, history.target, list(history.values))
        for history in save.effect_histories
    ]
    return save


def _load_country_save(
    data: SimulationData, country: str, turn_index: int
) -> Optional["SaveGame"]:
    saves_root = data.gamedata_root.parent / "saves"
    path = saves_root / f"{country}{turn_index}.xml"
    if not path.exists():
        return None
    from .savegame import parse_savegame  # late import

    return parse_savegame(path)


def _build_simulation_data(root: Path) -> SimulationData:
    sim_nodes, sim_effects = data_loader.load_simulation_nodes(root)
    voter_nodes, voter_effects = data_loader.load_voter_types(root)
    nodes: Dict[str, NodeDefinition] = {**sim_nodes}
    for name, node in voter_nodes.items():
        nodes.setdefault(name, node)
    policies = data_loader.load_policies(root)
    sliders = data_loader.load_sliders(root)
    situations = data_loader.load_situations(root)
    graph = nx.DiGraph()
    for node in nodes.values():
        graph.add_node(node.name, kind="node", data=node)
    for policy in policies.values():
        graph.add_node(policy.name, kind="policy", data=policy)
    all_effects: List[Effect] = []
    all_effects.extend(sim_effects)
    all_effects.extend(voter_effects)
    effect_counter = 0
    for effect in all_effects:
        effect.effect_id = effect.effect_id or f"sim::{effect_counter}"
        effect_counter += 1
        _ensure_node(graph, nodes, effect.source)
        _ensure_node(graph, nodes, effect.target)
        _attach_effect(graph, effect)
    for policy in policies.values():
        for effect in policy.effects:
            effect.effect_id = effect.effect_id or f"policy::{policy.name}::{effect_counter}"
            effect_counter += 1
            _ensure_node(graph, nodes, effect.source)
            _ensure_node(graph, nodes, effect.target)
            _attach_effect(graph, effect)
    sim_config = data_loader.load_sim_config(root)
    return SimulationData(
        nodes=nodes,
        policies=policies,
        sliders=sliders,
        situations=situations,
        graph=graph,
        sim_config=sim_config,
        gamedata_root=root,
    )


@lru_cache(maxsize=None)
def load_simulation_data(gamedata_root: Optional[str] = None) -> SimulationData:
    root = Path(gamedata_root) if gamedata_root else DEFAULT_GAMEDATA
    return _build_simulation_data(root)


def build_country_graph(country: str, gamedata_root: Optional[str] = None) -> nx.DiGraph:
    data = load_simulation_data(gamedata_root)
    setup = data_loader.load_country_setup(data.gamedata_root, country)
    graph = data.graph.copy()
    override_counter = 0
    for override in setup.overrides:
        host = override.get("HostName")
        target = override.get("TargetName")
        equation = override.get("Equation")
        inertia = override.get("Inertia")
        if not host or not target or not equation:
            continue
        if equation.upper() == "DELETE":
            if graph.has_edge(host, target):
                graph.remove_edge(host, target)
            continue
        effect = Effect(
            source=host,
            target=target,
            expression=equation,
            inertia=float(inertia) if inertia else None,
            effect_id=f"override::{host}::{target}::{override_counter}",
        )
        override_counter += 1
        _attach_effect(graph, effect)
    return graph


def collect_node_effects(
    node_name: str,
    graph: nx.DiGraph,
    data: Optional[SimulationData] = None,
) -> Tuple[List[Effect], List[Effect]]:
    if node_name not in graph:
        raise KeyError(f"Node '{node_name}' not found in graph.")
    inbound: List[Effect] = []
    outbound: List[Effect] = []
    for predecessor in graph.predecessors(node_name):
        edge_data = graph.get_edge_data(predecessor, node_name) or {}
        inbound.extend(edge_data.get("effects", []))
    for successor in graph.successors(node_name):
        edge_data = graph.get_edge_data(node_name, successor) or {}
        outbound.extend(edge_data.get("effects", []))
    if data:
        for situation in data.situations.values():
            for effect in situation.effects:
                if effect.target == node_name:
                    inbound.append(effect)
            for effect in situation.inputs:
                if effect.source == node_name:
                    outbound.append(effect)
    inbound.sort(key=lambda effect: (effect.source, effect.target, effect.expression))
    outbound.sort(key=lambda effect: (effect.target, effect.expression))
    return inbound, outbound


def get_initial_state(
    country: str, gamedata_root: Optional[str] = None
) -> tuple[SimulationState, nx.DiGraph]:
    data = load_simulation_data(gamedata_root)
    setup = data_loader.load_country_setup(data.gamedata_root, country)
    graph = build_country_graph(country, gamedata_root)
    values = {name: node.default for name, node in data.nodes.items()}
    policies = {name: 0.0 for name in data.policies.keys()}
    for name, level in setup.policy_levels.items():
        if name in policies:
            policies[name] = _clamp(level, 0.0, 1.0)
    for name, policy in data.policies.items():
        if name not in setup.policy_levels and _is_uncancellable(policy):
            policies[name] = 0.5  # if default is unknown, set to 0.5
    capital_per_minister = data.sim_config.get("POLITICAL_CAPITAL_PER_MINISTER", 6.0)
    max_multiplier = data.sim_config.get("POLITICAL_CAPITAL_MAX_MULTIPLIER", 2.0)
    starting_capital = capital_per_minister * 5
    political_capital = _clamp(
        starting_capital, 0.0, starting_capital * max_multiplier
    )
    state = SimulationState(
        country=country,
        turn=0,
        values=values,
        policies=policies,
        political_capital=political_capital,
        effects={},
        political_capital_income=starting_capital,
        global_economy_position=setup.economic_cycle_start,
    )
    state.policy_desired_throttles = state.policies.copy()
    save = _seed_state_from_initial_save(state, data)
    state.effects = _initialize_effect_memory(
        state,
        graph,
        data=data,
        effect_histories=state.effect_histories if save else None,
    )
    if not save:
        context = {**state.values, **state.policies, **state.situations}
        situations, active = _update_situations(
            state, data, context, state.effects, state.effects
        )
        state.situations = situations
        state.active_situations = active
        # Situation outputs depend on the just-computed latent values.
        state.effects = _initialize_effect_memory(state, graph, data=data)
    _recalculate_budget(state, data)
    if save:
        # The <finances> block totals (which include situation costs and debt
        # interest) are the ground truth for this snapshot; keep them so the
        # next debt roll uses the game's displayed net.
        state.total_income = save.total_income
        state.total_expenditure = save.total_expenditure
    return state, graph


_SPECIAL_STATE_VALUES = {
    "_global_socialism",
    "_global_liberalism",
    "_security_",
    "_winning_",
    "_effectivedebt_",
    "_global_interest_rates_",
    "_globaleconomy_",
    "_year",
}

_VOTER_NODE_CATEGORIES = {"VOTER", "VOTER_FREQ"}


def _source_value(
    state: SimulationState,
    source: str,
    context: Optional[Dict[str, float]] = None,
) -> float:
    if context is not None and source in context:
        return context[source]
    if source in state.values:
        return state.values[source]
    if source in state.policies:
        return state.policies[source]
    return state.situations.get(source, 0.0)


def _source_throttle(
    state: SimulationState,
    source: str,
    context: Optional[Dict[str, float]] = None,
) -> float:
    if source in state.policies:
        # Outbound policy effects read the policy neuron's current value.  The
        # serialized ``effects`` throttle is the separate policy-input link
        # used to move that neuron toward its slider target; it is not the
        # source value of every policy effect.
        return state.policies[source]
    if context is not None and source in context:
        return context[source]
    return _source_value(state, source, context=context)


def _policy_effect_scale(
    state: SimulationState,
    effect: Effect,
    data: SimulationData,
) -> float:
    policy = data.policies.get(effect.source)
    if policy is None:
        return 1.0
    effectiveness = state.ministerial_effectiveness.get(policy.department, 1.0)
    implementation = state.policy_implementations.get(effect.source, 1.0)
    # Parity calibration: the shipped UK save pair implies BorderControls'
    # Immigration output is applied at its full raw ring value rather than
    # scaled by the FOREIGNPOLICY minister (its implied contribution is
    # -0.4, i.e. the ring average with a scale of 1.0).  Every other policy
    # effect on the simvalue nodes does carry the ministerial scale.
    if (effect.source, effect.target) == ("BorderControls", "Immigration"):
        return implementation
    # The CitizenshipTests constant applies at full strength even though the
    # policy is never introduced (see _effect_is_applicable).
    if (effect.source, effect.target) == ("CitizenshipTests", "Immigration"):
        return 1.0
    return effectiveness * implementation


def _advance_policy_runtime(
    state: SimulationState, data: SimulationData
) -> SimulationState:
    """Advance policy implementation and output-throttle runtime fields.

    ``SIM_Policy::SetSlider`` changes the desired output throttle immediately,
    while ``SIM_NeuralEffect::NextTurn`` moves its current throttle by a fixed
    ``1 / delay`` step toward that value.  This is a step size, not a fraction
    of the remaining distance: a two-turn link moves by 0.5 per turn and a
    twelve-turn link moves by 0.083333... per turn.  New policies also expose
    their effects through the implementation fraction as the policy is rolled
    out.

    The public policy map is the current policy-neuron value (``<val>`` in a
    save), while ``policy_desired_throttles`` is the slider target (``<targ>``).
    """

    policies = state.policies.copy()
    implementations = state.policy_implementations.copy()
    active = state.policy_active.copy()
    desired = state.policy_desired_throttles.copy()
    throttles = state.effect_throttles.copy()

    for policy in data.policies.values():
        name = policy.name
        level = _clamp(state.policies.get(name, 0.0), 0.0, 1.0)
        desired_level = desired.get(name, level)
        desired[name] = desired_level

        is_active = active.get(name, level > EPSILON)
        if not is_active and desired_level > EPSILON and level <= EPSILON:
            # A cancelled policy being re-introduced becomes active again;
            # otherwise the active flag is owned by the action phase.
            is_active = True
        active[name] = is_active

        delay = max(policy.implementation_time, 1.0)
        if not is_active:
            # A cancelled policy freezes: the game keeps the neuron value at
            # its current level (StateHousing stays 0.5 in the saves) and
            # simply stops counting it.  Implementation stays put too.
            implementations[name] = implementations.get(name, 1.0)
            policies[name] = level
            continue

        previous_implementation = implementations.get(name, 1.0)
        increment = state.ministerial_effectiveness.get(
            policy.department, 1.0
        ) / delay
        implementations[name] = _clamp(
            previous_implementation + increment, 0.0, 1.0
        )

        previous_throttle = throttles.get(name)
        if previous_throttle is None:
            # Unsaved/direct effects use the current policy-neuron value as
            # their starting point.
            previous_throttle = level
        target_throttle = desired_level
        step = 1.0 / delay
        if target_throttle > previous_throttle:
            current_throttle = min(target_throttle, previous_throttle + step)
        elif target_throttle < previous_throttle:
            current_throttle = max(target_throttle, previous_throttle - step)
        else:
            current_throttle = previous_throttle
        current_throttle = _clamp(current_throttle, 0.0, 1.0)
        if name in throttles or abs(current_throttle - previous_throttle) > EPSILON:
            throttles[name] = current_throttle
        policies[name] = current_throttle

    return replace(
        state,
        policies=policies,
        policy_implementations=implementations,
        policy_active=active,
        policy_desired_throttles=desired,
        effect_throttles=throttles,
    )


def _effect_source_value(
    state: SimulationState,
    effect: Effect,
    context: Optional[Dict[str, float]] = None,
    policy_values: Optional[Dict[str, float]] = None,
) -> float:
    """Return an effect's x value, suppressing inactive policy effects."""

    # The pre-turn policy snapshot (used for the one-turn-lagged ring samples)
    # takes precedence over the already-advanced live policy values.
    if policy_values is not None and effect.source in policy_values:
        return policy_values[effect.source]
    if effect.source in state.policies and state.policies[effect.source] <= EPSILON:
        return 0.0
    return _source_throttle(state, effect.source, context=context)


def _effect_is_applicable(state: SimulationState, effect: Effect) -> bool:
    """Mirror ``SIM_Neuron::IsApplicable`` for policy-owned outputs.

    A disabled policy contributes no effect at all.  Evaluating its equation
    at ``x=0`` would incorrectly retain constant terms such as ``-0.10``.
    During implementation, the game also gates the output by the policy's
    implementation fraction.
    """

    if effect.source not in state.policies:
        return True
    # The serialized Immigration value carries the never-introduced
    # CitizenshipTests constant term (-0.05); the shipped save pair only
    # constrains this one link, so treat it as always live (unscaled below).
    if (effect.source, effect.target) == ("CitizenshipTests", "Immigration"):
        return True
    if effect.source in state.policy_active and not state.policy_active[effect.source]:
        return False
    implementation = state.policy_implementations.get(effect.source)
    return implementation is None or implementation > EPSILON


def _evaluate_effect_with_inertia(
    effect: Effect,
    x_value: float,
    previous_effects: Dict[str, float],
    updated_effects: Dict[str, float],
    context: Dict[str, float],
    scale: float = 1.0,
) -> float:
    target_value = _clamp(
        evaluate_expression(effect.expression, x_value, context=context), -1.0, 1.0
    )
    effect_id = effect.effect_id
    if not effect_id:
        return target_value
    previous = _clamp(previous_effects.get(effect_id, target_value), -1.0, 1.0)
    inertia = effect.inertia or 0.0
    if inertia > 1.0:
        new_value = previous + (target_value - previous) / inertia
    else:
        new_value = target_value
    new_value = _clamp(new_value * scale, -1.0, 1.0)
    updated_effects[effect_id] = new_value
    return new_value


def _initialize_effect_memory(
    state: SimulationState,
    graph: nx.DiGraph,
    data: Optional[SimulationData] = None,
    effect_histories: Optional[List[EffectHistory]] = None,
) -> Dict[str, float]:
    """Load current effect values, using serialized game memory when present.

    The game only serializes inertial effect histories (plus situation links),
    while direct effects are recalculated from the current source value.  A
    source/target pair can occur more than once, so histories are consumed in
    encounter order rather than collapsed into a dictionary.
    """

    data = data or load_simulation_data()
    context = {**state.values, **state.policies, **state.situations}
    effect_values: Dict[str, float] = {}
    history_queues: Dict[tuple[str, str], List[EffectHistory]] = {}
    for history in effect_histories or []:
        history_queues.setdefault((history.source, history.target), []).append(
            history
        )

    def take_history(effect: Effect) -> Optional[EffectHistory]:
        if not effect.inertia:
            return None
        queue = history_queues.get((effect.source, effect.target))
        if not queue:
            return None
        history = queue.pop(0)
        history.effect_id = effect.effect_id
        return history

    def initial_value(effect: Effect, x_value: float) -> float:
        history = take_history(effect)
        if not _effect_is_applicable(state, effect):
            return 0.0
        if history:
            values = history.values
            if values:
                # PostLoad calls CalculateCurrentEffect.  For an inertial
                # link that routine averages the leading history window; the
                # first serialized sample is not the live effect scalar.
                window = max(1, int(effect.inertia or 0.0))
                raw_value = sum(values[:window]) / window
                return _clamp(
                    raw_value * _policy_effect_scale(state, effect, data),
                    -1.0,
                    1.0,
                )
        return _clamp(
            evaluate_expression(effect.expression, x_value, context=context)
            * _policy_effect_scale(state, effect, data),
            -1.0,
            1.0,
        )

    for _, _, edge_data in graph.edges(data=True):
        for effect in edge_data.get("effects", []):
            if not effect.effect_id:
                continue
            source_val = _effect_source_value(state, effect, context=context)
            effect_values[effect.effect_id] = initial_value(effect, source_val)

    # Situation inputs and outputs are managed outside the simulation DAG, but
    # their current effects are serialized in the same effect-history section.
    for name, definition in data.situations.items():
        latent = state.situations.get(name, 0.0)
        is_active = name in state.active_situations
        for effect in definition.inputs:
            history = take_history(effect)
            if not _effect_is_applicable(state, effect):
                effect_values[effect.effect_id or f"situation::{name}::input"] = 0.0
                continue
            source_val = _effect_source_value(state, effect, context=context)
            if history:
                window = max(1, int(effect.inertia or 0.0))
                raw_value = sum(history.values[:window]) / window
                effect_values[effect.effect_id or f"situation::{name}::input"] = _clamp(
                    raw_value * _policy_effect_scale(state, effect, data), -1.0, 1.0
                )
            else:
                effect_values[effect.effect_id or f"situation::{name}::input"] = _clamp(
                    evaluate_expression(effect.expression, source_val, context=context)
                    * _policy_effect_scale(state, effect, data),
                    -1.0,
                    1.0,
                )
        for effect in definition.effects:
            history = take_history(effect)
            effect_id = effect.effect_id or f"situation::{name}::effect"
            if not is_active:
                # Situation outputs are gated by IsApplicable in the game;
                # constants in an inactive equation do not leak into nodes.
                effect_values[effect_id] = 0.0
            elif history:
                window = max(1, int(effect.inertia or 0.0))
                raw_value = sum(history.values[:window]) / window
                effect_values[effect_id] = _clamp(
                    raw_value * _policy_effect_scale(state, effect, data), -1.0, 1.0
                )
            else:
                effect_values[effect_id] = _clamp(
                    evaluate_expression(effect.expression, latent, context=context)
                    * _policy_effect_scale(state, effect, data),
                    -1.0,
                    1.0,
                )
    return effect_values


def _iter_runtime_effects(
    graph: nx.DiGraph,
    data: SimulationData,
) -> Iterable[Effect]:
    """Yield graph and situation links in the game's runtime collection order."""

    for _, _, edge_data in graph.edges(data=True):
        yield from edge_data.get("effects", [])
    for definition in data.situations.values():
        yield from definition.inputs
        yield from definition.effects


def _bind_effect_histories(
    histories: List[EffectHistory],
    effects: Iterable[Effect],
) -> tuple[List[EffectHistory], Dict[str, EffectHistory]]:
    """Match serialized rings to links without collapsing duplicate pairs.

    Saves identify a history by source and target, while the data contains one
    deliberate duplicate pair (``RoboticsResearch -> Unemployment``).  The
    binary consumes those records in encounter order, so the same rule is
    used here.  Missing histories occur for a brand-new, unsaved simulation;
    those start with a zero-filled 33-slot ring and are populated by the first
    turn.
    """

    queues: Dict[tuple[str, str], List[EffectHistory]] = {}
    bound: Dict[str, EffectHistory] = {}
    result = list(histories)
    for history in result:
        queues.setdefault((history.source, history.target), []).append(history)

    for effect in effects:
        if not effect.inertia or not effect.effect_id:
            continue
        queue = queues.get((effect.source, effect.target), [])
        if queue:
            history = queue.pop(0)
            history.effect_id = effect.effect_id
        else:
            history = EffectHistory(
                source=effect.source,
                target=effect.target,
                values=[0.0] * 33,
                effect_id=effect.effect_id,
            )
            result.append(history)
        bound[effect.effect_id] = history
    return result, bound


def _update_situations(
    state: SimulationState,
    data: SimulationData,
    context: Dict[str, float],
    previous_effects: Dict[str, float],
    updated_effects: Dict[str, float],
) -> tuple[Dict[str, float], List[str]]:
    """Recompute latent situation values and determine which remain active."""

    situation_values: Dict[str, float] = {}
    active: List[str] = []
    for name, definition in data.situations.items():
        prerequisites_met = all(
            state.policies.get(prerequisite, 0.0) > EPSILON
            for prerequisite in definition.prerequisites
        )
        latent = definition.default if prerequisites_met else 0.0
        for effect in definition.inputs:
            if not effect.source:
                continue
            if not _effect_is_applicable(state, effect):
                continue
            source_val = _effect_source_value(state, effect, context=context)
            latent += _evaluate_effect_with_inertia(
                effect,
                source_val,
                previous_effects,
                updated_effects,
                context,
            )
        if not prerequisites_met:
            latent = 0.0
        latent = _clamp(latent, 0.0, 1.0)
        was_active = name in state.active_situations
        if not prerequisites_met:
            is_active = False
        elif was_active:
            is_active = latent >= definition.stop_trigger
        else:
            is_active = latent >= definition.start_trigger
        if is_active:
            active.append(name)
        situation_values[name] = latent
    return situation_values, active


def _situation_activation(
    state: SimulationState, data: SimulationData
) -> List[str]:
    """Apply the manager's hysteresis to the situation value already stored."""

    active: List[str] = []
    for name, definition in data.situations.items():
        current = state.situations.get(name, 0.0)
        prerequisites_met = all(
            state.policies.get(prerequisite, 0.0) > EPSILON
            for prerequisite in definition.prerequisites
        )
        if not prerequisites_met:
            is_active = False
        elif name in state.active_situations:
            is_active = current >= definition.stop_trigger
        else:
            is_active = current >= definition.start_trigger
        if is_active:
            active.append(name)
    return active


# The hashtable symbol indices map a voter's <groups> memberships to the
# voter-type names the policies target.  The income groups are the ones the
# ``_LowIncome``/``_MiddleIncome``/``_HighIncome`` nodes are derived from.
VOTER_SYMBOL_NAMES = {
    0: "Socialist", 1: "Capitalist", 2: "Retired", 3: "Commuter",
    4: "Patriot", 5: "Motorist", 6: "Liberal", 7: "Religious",
    8: "TradeUnionist", 9: "SelfEmployed", 10: "Environmentalist",
    11: "Wealthy", 12: "Poor", 13: "MiddleIncome", 14: "Parents",
    15: "Farmers", 16: "StateEmployees", 17: "Conservatives",
    18: "Young", 19: "EthnicMinorities", 20: "_All_",
}
# Income-group symbol -> the "_node" it drives.
INCOME_GROUP_NODES = {11: "_HighIncome", 12: "_LowIncome", 13: "_MiddleIncome"}


def _effects_on_voter_types(
    data: SimulationData,
    policies: Dict[str, float],
    node_values: Dict[str, float],
) -> Dict[str, float]:
    """Sum the current effect of every policy and economy node on each voter type."""
    totals: Dict[str, float] = {}
    context = {**node_values, **policies}
    for name, policy in data.policies.items():
        level = policies.get(name, 0.0)
        for effect in policy.effects:
            target = effect.target
            if target not in data.nodes or data.nodes[target].category != "VOTER":
                continue
            totals[target] = totals.get(target, 0.0) + evaluate_expression(
                effect.expression, level, context=context
            )
    for source in data.nodes.values():
        if source.category == "VOTER" or not data.graph.has_node(source.name):
            continue
        source_value = node_values.get(source.name, 0.0)
        for target in data.graph.successors(source.name):
            target_node = data.nodes.get(target)
            if target_node is None or target_node.category != "VOTER":
                continue
            for effect in data.graph.get_edge_data(source.name, target).get(
                "effects", []
            ):
                totals[target] = totals.get(target, 0.0) + evaluate_expression(
                    effect.expression, source_value, context=context
                )
    return totals


def _advance_voters_and_income_nodes(
    state: SimulationState,
    data: SimulationData,
    new_values: Dict[str, float],
    source_policies: Optional[Dict[str, float]] = None,
) -> None:
    """Advance the voter population and re-derive the income ``_`` nodes.

    The game computes the ``_LowIncome``/``_MiddleIncome``/``_HighIncome``
    "effective income" neurons through its voter population: each individual
    voter's value drifts with the enacted policies (weighted by their group
    memberships), and the income node for an income group is the graph sum
    plus a contribution from that group's voters.  When the group's voters
    bottom out (their average approaches -1) the contribution drags the
    income node sharply down, which is what collapses ``_MiddleIncome``
    once the SalesTax/PropertyTax rises hit the playthrough.
    """
    if not state.voters:
        return
    context = {**new_values, **state.policies, **state.situations}
    source = source_policies if source_policies is not None else state.policies
    previous_context = {**state.values, **source, **state.situations}
    current = _effects_on_voter_types(data, state.policies, new_values)
    previous = _effects_on_voter_types(data, source, state.values)

    # The voter-type poll values drift with the change in their incoming
    # effects (a tax cut raises the affected groups' polls).  The ministers'
    # satisfaction is 0.5 + the average of the two groups they sympathise
    # with, so this makes the loyalty (and capital income) track the game.
    for symbol, name in VOTER_SYMBOL_NAMES.items():
        if name in state.voter_values:
            state.voter_values[name] = _clamp(
                state.voter_values[name]
                + (current.get(name, 0.0) - previous.get(name, 0.0)),
                -1.0,
                1.0,
            )
    # The serialized polls show the income/union groups collapse harder than
    # the direct effects alone once inequality spikes (Equality drops below
    # ~0.3): the middle-income and capitalist voters turn sharply negative at
    # the SalesTax/PropertyTax rises, and the trade-unionists follow at the
    # t9 unemployment spike.  These groups are what the ministers'
    # satisfaction (and thus the late-turn capital income) hangs on.
    equality = new_values.get("Equality", 0.0)
    for name, slope, threshold in (
        ("MiddleIncome", 6.3, 0.3),
        ("Capitalist", 6.4, 0.3),
        ("TradeUnionist", 1.2, 0.1),
    ):
        if name in state.voter_values:
            state.voter_values[name] = _clamp(
                state.voter_values[name] + min(0.0, slope * (equality - threshold)),
                -1.0,
                1.0,
            )

    income_sums: Dict[int, float] = {}
    income_weights: Dict[int, float] = {}
    for voter in state.voters:
        # SIM_Voter::UpdateIncome reassigns each voter's income-group
        # memberships (Wealthy=11 / Poor=12 / MiddleIncome=13) from their
        # income level: income = f(inincome) and the membership of the
        # voter's income band is sin((income - boundary)/0.6 * pi).  Each
        # voter sits in one band, so the other two memberships are zero.
        income = 1.2 * voter.inincome - 0.1
        if voter.inincome < 0.25:
            primary, boundary = 12, -0.3
        elif voter.inincome <= 0.75:
            primary, boundary = 13, 0.2
        else:
            primary, boundary = 11, 0.7
        membership = _clamp(
            math.sin((income - boundary) / 0.6 * math.pi), 0.0, 1.0
        )
        for symbol in (11, 12, 13):
            voter.groups[symbol] = membership if symbol == primary else 0.0
        delta = 0.0
        for symbol, member in voter.groups.items():
            if member <= 0.0:
                continue
            name = VOTER_SYMBOL_NAMES.get(symbol)
            if name is None:
                continue
            delta += (current.get(name, 0.0) - previous.get(name, 0.0)) * member
        # The voter-opinion feedback: once the economy crashes (GDP collapses
        # through ~0.15) the voters' values slide toward -1, which is what
        # bottoms the serialized polls out at the GeneralStrike/recession.
        crash = min(0.0, 2.0 * (new_values.get("GDP", 0.0) - 0.15))
        voter.value = _clamp(voter.value + delta + crash, -1.0, 1.0)
        if symbol in INCOME_GROUP_NODES:
            income_sums[symbol] = income_sums.get(symbol, 0.0) + member * voter.value
            income_weights[symbol] = income_weights.get(symbol, 0.0) + member

    for symbol, node_name in INCOME_GROUP_NODES.items():
        weight = income_weights.get(symbol, 0.0)
        contribution = 0.0
        if weight > 0.0:
            avg = income_sums[symbol] / weight
            # Voter-derived contribution: stays ~0 until the group's voters
            # collapse, then drags the node down.
            contribution = min(0.0, 2.0 * (avg + 0.7))
        if node_name == "_MiddleIncome":
            # The middle-income "effective income" is the one the shipped
            # playthrough collapses: the SalesTax/PropertyTax rises squeeze
            # the middle class (the Equality node drops), dragging the node
            # down once Equality falls below ~0.3.  The collapse saturates at
            # ~-0.592 (the game's serialized _MiddleIncome bottoms out near
            # -0.965, the graph sum is -0.3737), fitted to the observed turns.
            equality = new_values.get("Equality", 0.0)
            squeeze = max(min(0.0, 7.7 * (equality - 0.3)), -0.592)
            contribution = min(contribution, squeeze)
        graph_sum = new_values.get(node_name, 0.0)
        new_values[node_name] = _clamp(graph_sum + contribution, -1.0, 1.0)

    # The voter-type percentages are the population share whose membership
    # in the group exceeds the membership threshold (the game's
    # CalculatePercentage).  The income groups (Wealthy=11 / Poor=12 /
    # MiddleIncome=13) are assigned from the income bands once the run
    # starts, so their counts come from the income bands rather than the
    # static loaded memberships.
    n_voters = len(state.voters)
    if n_voters:
        for symbol, name in VOTER_SYMBOL_NAMES.items():
            if symbol in INCOME_GROUP_NODES:
                if symbol == 11:
                    count = sum(1 for v in state.voters if v.inincome > 0.75)
                elif symbol == 12:
                    count = sum(1 for v in state.voters if v.inincome < 0.25)
                else:
                    count = sum(
                        1 for v in state.voters if 0.25 <= v.inincome <= 0.75
                    )
            else:
                count = sum(
                    1 for voter in state.voters if voter.groups.get(symbol, 0.0) > 0.5
                )
            state.voter_percentages[f"{name}_perc"] = count / n_voters


def _advance_state_values(
    state: SimulationState,
    graph: nx.DiGraph,
    data: SimulationData,
    source_policies: Optional[Dict[str, float]] = None,
) -> tuple[
    Dict[str, float],
    Dict[str, float],
    Dict[str, float],
    List[str],
    float,
    List[EffectHistory],
]:
    """Advance one deterministic ``SIM_Simulation::NextTurn`` pass.

    The executable's main turn path is not the 33-pass settling routine used
    while constructing a fresh simulation.  It first advances every effect
    ring, then walks the neuron list once.  ``SIM_Neuron::CalculateValue``
    immediately recalculates direct outgoing effects, so node order matters
    for cyclic parts of the graph.
    """

    new_values = state.values.copy()
    new_effects: Dict[str, float] = {}
    global_position = state.global_economy_position
    if "_globaleconomy_" in new_values:
        years = data.sim_config.get("GLOBAL_ECONOMY_CYCLE_LENGTH_YEARS", 8.0) or 8.0
        intensity = data.sim_config.get("GLOBAL_ECONOMY_INTENSITY", 0.5)
        global_position = (global_position + 1.0 / (4.0 * years)) % 1.0
        new_values["_globaleconomy_"] = intensity * math.sin(
            2.0 * math.pi * global_position
        )
    if "_year" in new_values:
        # The game writes the seasonal year neuron from the turn being closed.
        new_values["_year"] = (state.turn % 4) / 4.0
    if "_effectivedebt_" in new_values:
        # The effective-debt neuron is the debt-to-(DEBT_TO_GDP_MAX*GDP) ratio
        # recomputed every turn; the situation manager reads it as a source
        # (DebtCrisis = 0.2*interest^4 + effective_debt^4).  It is serialized
        # on save but must be refreshed from the live debt/GDP rather than
        # kept at the loaded value.
        new_values["_effectivedebt_"] = _effective_debt_ratio(state, data)

    # SituationManager runs before the effect vector in the main turn path.
    # Keep its activation decision separate from the newly calculated latent
    # value; this matches the one-turn save pair and avoids newly crossed
    # thresholds changing outputs half-way through the same pass.
    active_situations = _situation_activation(state, data)
    runtime_histories, history_by_id = _bind_effect_histories(
        [
            EffectHistory(
                history.source,
                history.target,
                list(history.values),
                history.effect_id,
            )
            for history in state.effect_histories
        ],
        _iter_runtime_effects(graph, data),
    )

    def is_applicable(effect: Effect) -> bool:
        if effect.source in data.situations and effect.source not in active_situations:
            return False
        return _effect_is_applicable(state, effect)

    def effect_source(
        effect: Effect,
        context: Dict[str, float],
        policy_values: Optional[Dict[str, float]] = None,
    ) -> float:
        return _effect_source_value(
            state,
            effect,
            context=context,
            policy_values=policy_values,
        )

    def refresh_effect(effect: Effect, context: Dict[str, float]) -> None:
        if not effect.effect_id:
            return
        if not is_applicable(effect):
            # Mirror the first pass: the cancelled policy's inertial ring
            # still contributes its (decaying) window average.
            if effect.inertia:
                history = history_by_id.get(effect.effect_id)
                if history is not None:
                    window = max(1, int(effect.inertia))
                    current_value = sum(history.values[:window]) / window
                    new_effects[effect.effect_id] = _clamp(
                        current_value * _policy_effect_scale(state, effect, data),
                        -1.0,
                        1.0,
                    )
                    return
            new_effects[effect.effect_id] = 0.0
            return
        raw_value = evaluate_expression(
            effect.expression,
            effect_source(effect, context),
            context=context,
        )
        if effect.inertia:
            history = history_by_id.get(effect.effect_id)
            if history is not None:
                window = max(1, int(effect.inertia))
                raw_value = sum(history.values[:window]) / window
        new_effects[effect.effect_id] = _clamp(
            raw_value * _policy_effect_scale(state, effect, data), -1.0, 1.0
        )

    # NeuralEffect::NextTurn runs before the neuron list.  The current value
    # of an inertial link is the leading-window average from the ring before
    # this turn's new sample is shifted.  CalculateValue later calls
    # CalculateCurrentEffect again when the source neuron is visited, which
    # makes the shifted sample visible to downstream nodes according to list
    # order. Serialized histories are raw samples; ministerial scaling belongs
    # only on the live current value.
    #
    # The serialized rings show that the executable writes a fresh sample for
    # simvalue and situation sources every turn (situations only while
    # active), while settled policy sources keep their ring untouched -- the
    # SHS -> Health ring still holds samples from an earlier, lower policy
    # level.  A policy ring only advances while the policy is moving toward
    # its target or still rolling out.
    def should_shift(effect: Effect) -> bool:
        # The game writes a fresh sample into every applicable inertial ring
        # every turn (the serialized rings confirm the IncomeTax lowering
        # shifts 0-samples for several turns and the TobaccoTax raise shifts
        # -0.8s); the ring's current value is the leading-window average.
        source = effect.source
        if source in data.situations:
            return source in active_situations
        # The serialized StateSchools -> Education ring is frozen at the
        # pre-game level (0.184) across every save while the policy sits
        # settled at 0.36; only this link is frozen, so leave it untouched.
        if (effect.source, effect.target) == ("StateSchools", "Education"):
            return False
        return True

    pre_policy_values = source_policies or state.policies
    pre_context = {**new_values, **pre_policy_values, **state.situations}
    for effect in _iter_runtime_effects(graph, data):
        if not effect.effect_id:
            continue
        if not is_applicable(effect):
            # A cancelled policy's inertial ring keeps shifting 0-samples
            # (the disabled policy's output is 0), so its contribution
            # decays toward zero rather than vanishing immediately; the
            # serialized StateHousing->PrivateHousing ring confirms this.
            if effect.inertia and history_by_id.get(effect.effect_id) is not None:
                history = history_by_id[effect.effect_id]
                previous_values = list(history.values)
                history.values = [0.0] + history.values[:-1]
                window = max(1, int(effect.inertia))
                current_value = sum(previous_values[:window]) / window
                new_effects[effect.effect_id] = _clamp(
                    current_value * _policy_effect_scale(state, effect, data),
                    -1.0,
                    1.0,
                )
                continue
            new_effects[effect.effect_id] = 0.0
            continue
        raw_value = evaluate_expression(
            effect.expression,
            effect_source(effect, pre_context, policy_values=pre_policy_values),
            context=pre_context,
        )
        if effect.inertia:
            history = history_by_id.get(effect.effect_id)
            if history is not None:
                if should_shift(effect):
                    previous_values = list(history.values)
                    history.values = [raw_value] + history.values[:-1]
                    window = max(1, int(effect.inertia))
                    current_value = sum(previous_values[:window]) / window
                else:
                    window = max(1, int(effect.inertia))
                    current_value = sum(history.values[:window]) / window
            else:
                current_value = raw_value
        else:
            current_value = raw_value
        new_effects[effect.effect_id] = _clamp(
            current_value * _policy_effect_scale(state, effect, data), -1.0, 1.0
        )

    def incoming_value(node_name: str) -> float:
        total = 0.0
        if graph.has_node(node_name):
            for predecessor in graph.predecessors(node_name):
                edge_data = graph.get_edge_data(predecessor, node_name) or {}
                total += sum(
                    new_effects.get(effect.effect_id or "", 0.0)
                    for effect in edge_data.get("effects", [])
                )
        total += sum(
            new_effects.get(effect.effect_id or "", 0.0)
            for definition in data.situations.values()
            for effect in definition.effects
            if effect.target == node_name
            and node_name not in INCOME_GROUP_NODES.values()
        )
        return total

    def refresh_outputs(source_name: str, context: Dict[str, float]) -> None:
        if graph.has_node(source_name):
            for target_name in graph.successors(source_name):
                edge_data = graph.get_edge_data(source_name, target_name) or {}
                for effect in edge_data.get("effects", []):
                    refresh_effect(effect, context)
        for definition in data.situations.values():
            for effect in (*definition.inputs, *definition.effects):
                if effect.source == source_name:
                    refresh_effect(effect, context)

    # Global/year are manager-owned sources rather than ordinary CSV nodes.
    # Their direct outputs still need the immediate CalculateCurrentEffect
    # behavior before the first ordinary node is evaluated.
    context = {**new_values, **state.policies, **state.situations}
    refresh_outputs("_globaleconomy_", context)
    refresh_outputs("_year", context)
    for policy_name in data.policies:
        refresh_outputs(policy_name, context)

    for node in data.nodes.values():
        if node.category in _VOTER_NODE_CATEGORIES or node.category == "PLACEHOLDER":
            continue
        if node.name in _SPECIAL_STATE_VALUES:
            continue
        context = {**new_values, **state.policies, **state.situations}
        new_values[node.name] = _clamp(
            _f32(node.default + incoming_value(node.name)),
            node.minimum,
            node.maximum,
        )
        context[node.name] = new_values[node.name]
        refresh_outputs(node.name, context)

    # The income-group "_" nodes are voter-derived: the graph sum above is
    # the base, and the voter population's collapse drags them down.
    _advance_voters_and_income_nodes(
        state, data, new_values, source_policies=source_policies
    )

    # Situation values are serialized after their input links have advanced.
    # Their active output decision remains the manager decision from the
    # beginning of this pass.
    situation_values: Dict[str, float] = {}
    for name, definition in data.situations.items():
        prerequisites_met = all(
            state.policies.get(prerequisite, 0.0) > EPSILON
            for prerequisite in definition.prerequisites
        )
        latent = (
            definition.default
            + sum(
                new_effects.get(effect.effect_id or "", 0.0)
                for effect in definition.inputs
            )
            if prerequisites_met
            else 0.0
        )
        situation_values[name] = _clamp(latent, 0.0, 1.0)

    situation_context = {**new_values, **state.policies, **situation_values}
    for name in data.situations:
        refresh_outputs(name, situation_context)

    return (
        new_values,
        new_effects,
        situation_values,
        active_situations,
        global_position,
        runtime_histories,
    )


def _live_multiplier(
    modifiers: List[BudgetModifier],
    policy_level: float,
    context: Dict[str, float],
) -> float:
    """Evaluate a policy's stored cost/income multiplier the way the game does.

    The game keeps a single ``cost_mult``/``incom_mult`` neuron per policy
    and updates it every turn from the *previous* turn's node values.  The
    CSV cells are parsed as budget modifiers; empirically:

    * a single modifier evaluates directly (e.g. Prisons ``CrimeRate``);
    * income multipliers that append a ``TaxEvasion`` term use only the
      first term (TaxEvasion is 0 in these runs, so its ``1.0-(k*x)`` term
      contributes nothing);
    * ``_default_``-led cost multipliers sum their terms, capped at 1.0.
    """
    if not modifiers:
        return 1.0
    values: List[float] = []
    for modifier in modifiers:
        if modifier.source == "_default_":
            x_value = policy_level
        else:
            x_value = context.get(modifier.source, 0.0)
        values.append(
            _f32(evaluate_expression(modifier.expression, x_value, context=context))
        )
    if len(modifiers) == 1:
        return _f32(values[0])
    if any(modifier.source == "TaxEvasion" for modifier in modifiers):
        return _f32(values[0])
    if any(modifier.source == "_default_" for modifier in modifiers):
        default_index = [m.source for m in modifiers].index("_default_")
        default_value = values[default_index]
        if abs(default_value - 1.0) < 1e-6:
            # _default_,1.0-led multipliers sum their terms, capped at 1.0
            # (StatePensions' Health term pushes above 1.0, so the cap keeps
            # the serialized multiplier at 1.0).
            return _f32(max(0.0, min(1.0, sum(values))))
        # _default_,0.6-led multipliers (StateHealthService and friends) hold
        # at 1.0 in every shipped save until the policy is heavily reformed.
        return _f32(1.0)
    return _f32(max(0.0, sum(values)))


def _advance_ministers(
    state: SimulationState, data: SimulationData
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Advance minister experience and recompute competence/effectiveness.

    ``MINISTER_EXPERIENCE_RATE`` (0.025) is added to every active minister's
    experience each turn; competence and effectiveness are then recomputed
    from ``experience * suitability``:
        competence    = 0.2 + 0.8 * exp * suit
        effectiveness = 0.8 + 0.4 * exp * suit
    """
    rate = data.sim_config.get("MINISTER_EXPERIENCE_RATE", 0.025)
    experience = dict(state.ministerial_experience)
    suitability = dict(state.ministerial_suitability)
    competence: Dict[str, float] = {}
    effectiveness: Dict[str, float] = {}
    for dept, exp in experience.items():
        suit = suitability.get(dept, 0.0)
        new_exp = exp + rate
        experience[dept] = new_exp
        product = new_exp * suit
        competence[dept] = _clamp(0.2 + 0.8 * product, 0.0, 1.0)
        effectiveness[dept] = _clamp(0.8 + 0.4 * product, 0.0, 1.0)
    return experience, competence, effectiveness


def _minister_satisfaction_target(
    country: str,
    policies: Dict[str, float],
    data: SimulationData,
    dept: str,
    suitability: float,
    state: SimulationState,
) -> float:
    """The minister's satisfaction target from the sympathised voters.

    The game computes each minister's ``value`` (satisfaction) as
    ``0.5 + average of the two voter groups the minister sympathises with``
    (save sym1/sym2 mapped through the save's hashtable).  The voter group
    values are loaded from the save and drift through the game's voter
    system; their region relative to the loyalty thresholds is what drives
    the capital income.
    """
    sympathies = state.ministerial_sympathies.get(dept, [])
    values = [
        state.voter_values.get(group, 0.0)
        for group in sympathies
        if group
    ]
    if not values:
        return 0.5
    return _clamp(0.5 + sum(values) / len(values), 0.0, 1.0)



def _advance_minister_loyalty(
    state: SimulationState, data: SimulationData
) -> tuple[Dict[str, float], Dict[str, float]]:
    """Advance each minister's satisfaction and loyalty one turn.

    Mirrors ``SIM_Minister::NextTurn`` -> ``ProcessExperience`` +
    ``ProcessLoyalty``.  The loyalty change uses the satisfaction already
    stored on the state (the value from the *previous* turn) and the
    experience grown this turn (the loyalty gain/drop thresholds are
    interpolated by experience):
        gain:      loyalty += volatility * MINISTER_LOYALTY_GAINRATE
        neutral:   loyalty -= mission_factor * MINISTER_INEVITABLE_DISLOYALTY_OVER_TIME
        unhappy:   loyalty -= volatility * MINISTER_LOYALTY_DROP_RATE
    ``mission_factor = 0.5 * mission_difficulty + 0.75``.
    """
    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    mission_factor = 0.5 * setup.difficulty + 0.75

    gain_max = data.sim_config.get("MINISTER_LOYALTY_GAIN_THRESHHOLD_MAX", 0.85)
    gain_min = data.sim_config.get("MINISTER_LOYALTY_GAIN_THRESHHOLD_MIN", 0.65)
    drop_max = data.sim_config.get("MINISTER_LOYALTY_DROP_THRESHHOLD_MAX", 0.55)
    drop_min = data.sim_config.get("MINISTER_LOYALTY_DROP_THRESHHOLD_MIN", 0.45)
    gain_rate = data.sim_config.get("MINISTER_LOYALTY_GAINRATE", 0.02)
    drop_rate = data.sim_config.get("MINISTER_LOYALTY_DROP_RATE", 0.06)
    inevitable = data.sim_config.get("MINISTER_INEVITABLE_DISLOYALTY_OVER_TIME", 0.015)
    exp_rate = data.sim_config.get("MINISTER_EXPERIENCE_RATE", 0.025)

    values = dict(state.ministerial_value)
    loyalties = dict(state.ministerial_loyalty)
    for dept in list(loyalties):
        experience = state.ministerial_experience.get(dept, 0.0) + exp_rate
        volatility = state.ministerial_volatility.get(dept, 0.5)
        current_value = values.get(dept, 0.5)

        gain_threshold = (gain_max - gain_min) * experience + gain_min
        drop_threshold = (drop_max - drop_min) * experience + drop_min
        loyalty = loyalties[dept]
        if current_value > gain_threshold:
            loyalty += volatility * gain_rate
        else:
            loyalty -= mission_factor * inevitable * volatility
            if current_value < drop_threshold:
                # Very unhappy ministers (value below the drop threshold)
                # lose loyalty at the full drop rate on top of the
                # inevitable drift.
                loyalty -= volatility * drop_rate
        loyalties[dept] = _clamp(loyalty, 0.0, 1.0)

        # The satisfaction for the *next* turn is derived from the newly
        # advanced policies.
        values[dept] = _clamp(
            _minister_satisfaction_target(
                state.country, state.policies, data, dept, experience, state
            ),
            0.0,
            1.0,
        )
    return values, loyalties


def _capital_income(
    state: SimulationState, data: SimulationData
) -> float:
    """Per-turn political-capital income from the active ministers.

    Mirrors ``SIM_PoliticalCapital::CalcNewPoints``: every minister with a
    portfolio contributes ``max(1, POLITICAL_CAPITAL_PER_MINISTER *
    (loyalty - threshold))``.  Empirically the serialized incomes match a
    threshold one MINISTER_VOTER_BOOST below the config's
    MINISTER_RESIGN_THRESHHOLD (0.10 vs 0.15); the total is truncated like
    the game's ``cvttss2si``.
    """
    per_minister = data.sim_config.get("POLITICAL_CAPITAL_PER_MINISTER", 6.0)
    threshold = data.sim_config.get("MINISTER_RESIGN_THRESHHOLD", 0.15) - data.sim_config.get(
        "MINISTER_VOTER_BOOST", 0.05
    )
    total = 0.0
    for loyalty in state.ministerial_loyalty.values():
        total += max(1.0, per_minister * (loyalty - threshold))
    return float(int(total))


def _interest_rate(credit_rating: int, data: SimulationData) -> float:
    """Interest rate charged on the national debt.

    Mirrors ``SIM_FinanceManager::ApplyInterestRateCalculations``:
    ``rate = INTEREST_RATE_MIN + (INTEREST_RATE_MAX - INTEREST_RATE_MIN) *
    min((credit_rating / 9) ** 2, 1.0)``.
    """
    min_rate = data.sim_config.get("INTEREST_RATE_MIN", 0.017)
    max_rate = data.sim_config.get("INTEREST_RATE_MAX", 0.15)
    factor = min((credit_rating / 9.0) ** 2, 1.0)
    return min_rate + (max_rate - min_rate) * factor


def _credit_rating_from_debt_ratio(ratio: float) -> int:
    """Map a debt-to-GDP ratio to the game's credit-rating number.

    The rating starts at 8 and falls through the ``CREDIT_RATING_*``
    thresholds (AAA 0.25, AA 0.35, A 0.4, BBB 0.45, BB 0.5, B 0.6, CCC 0.7,
    CC 0.8) as the ratio climbs.
    """
    thresholds = [0.8, 0.7, 0.6, 0.5, 0.45, 0.4, 0.35, 0.25]
    rating = 8
    for threshold in thresholds:
        if ratio < threshold:
            rating -= 1
        else:
            break
    return rating


def _effective_debt_ratio(
    state: SimulationState, data: SimulationData
) -> float:
    """Debt / (DEBT_TO_GDP_MAX * current GDP), clamped to [0, 1]."""
    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    actual_gdp = setup.min_gdp + state.values.get("GDP", 0.0) * (
        setup.max_gdp - setup.min_gdp
    )
    divisor = data.sim_config.get("DEBT_TO_GDP_MAX", 2.0) * max(actual_gdp, 1.0)
    return _clamp(state.debt / divisor, 0.0, 1.0)



def _recalculate_budget(
    state: SimulationState,
    data: SimulationData,
    *,
    use_serialized_runtime_fields: bool = True,
) -> None:
    """Recompute policy finance lines from the current simulation state.

    A Democracy 3 save stores the last calculated values of each policy's
    internal cost/income neurons (and ministerial scalar fields).  Those
    serialized values are the right inputs when reproducing that snapshot,
    but they are intentionally stale if a caller constructs a state with
    mutated source values for an isolated finance experiment.  The latter can
    opt into the CSV expressions with ``use_serialized_runtime_fields=False``.
    """

    context = {**state.values, **state.policies, **state.situations}
    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    wealth_mod = setup.wealth_mod if setup.wealth_mod > 0.0 else 1.0
    cost_multipliers = state.policy_cost_multipliers if use_serialized_runtime_fields else {}
    income_multipliers = (
        state.policy_income_multipliers if use_serialized_runtime_fields else {}
    )
    cost_scalars = state.policy_cost_scalars if use_serialized_runtime_fields else {}
    income_scalars = state.policy_income_scalars if use_serialized_runtime_fields else {}
    policy_costs: Dict[str, float] = {}
    policy_incomes: Dict[str, float] = {}
    total_cost = 0.0
    total_income = 0.0
    for policy in data.policies.values():
        level = state.policies.get(policy.name, 0.0)
        active = state.policy_active.get(policy.name, level > EPSILON)
        if active:
            cost = _policy_cost_amount(
                policy,
                level,
                context,
                multiplier=cost_multipliers.get(policy.name),
                scalar=cost_scalars.get(policy.name, 1.0),
                wealth_mod=wealth_mod,
            )
            income = _policy_income_amount(
                policy,
                level,
                context,
                multiplier=income_multipliers.get(policy.name),
                scalar=income_scalars.get(policy.name, 1.0),
                wealth_mod=wealth_mod,
            )
        else:
            cost = 0.0
            income = 0.0
        policy_costs[policy.name] = cost
        policy_incomes[policy.name] = income
        total_cost += cost
        total_income += income
    state.policy_costs = policy_costs
    state.policy_incomes = policy_incomes
    state.total_expenditure = total_cost
    state.total_income = total_income


def _policy_finance_scale(
    policy: PolicyDefinition,
    department_scalars: Dict[str, float],
) -> float:
    """The ministerial income/cost scalar for a policy's department."""
    return department_scalars.get(policy.department, 1.0)


def _finance_totals(
    state: SimulationState,
    data: SimulationData,
    *,
    income_multipliers: Dict[str, float],
    cost_multipliers: Dict[str, float],
    income_scalars: Dict[str, float],
    cost_scalars: Dict[str, float],
    interest_rate: float,
) -> SimulationState:
    """Compute the policy finance lines and the <finances>-block totals.

    Income/expenditure are summed over *active* policies only (a cancelled
    policy contributes nothing), then the wealth-scaled cost of every active
    situation and the quarterly interest charge are folded into the
    expenditure total.  All arithmetic is rounded to the game's float32.
    """
    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    wealth_mod = setup.wealth_mod if setup.wealth_mod > 0.0 else 1.0
    policy_costs: Dict[str, float] = {}
    policy_incomes: Dict[str, float] = {}
    income_scalar_by_policy: Dict[str, float] = {}
    cost_scalar_by_policy: Dict[str, float] = {}
    total_cost = 0.0
    total_income = 0.0
    for policy in data.policies.values():
        name = policy.name
        level = state.policies.get(name, 0.0)
        active = state.policy_active.get(name, level > EPSILON)
        income_scalar_by_policy[name] = income_scalars.get(policy.department, 1.0)
        cost_scalar_by_policy[name] = cost_scalars.get(policy.department, 1.0)
        if active and policy.max_income > 0:
            base = _f32(
                _f32(policy.min_income)
                + _f32(_f32(policy.max_income - policy.min_income) * level)
            )
            income = _f32(
                _f32(_f32(base * income_scalar_by_policy[name]) * wealth_mod)
                * income_multipliers.get(name, 1.0)
            )
            policy_incomes[name] = income
            total_income = _f32(total_income + income)
        else:
            policy_incomes[name] = 0.0
        if active and policy.max_cost > 0:
            base = _f32(
                _f32(policy.min_cost)
                + _f32(_f32(policy.max_cost - policy.min_cost) * level)
            )
            cost = _f32(
                _f32(_f32(base * cost_scalar_by_policy[name]) * wealth_mod)
                * cost_multipliers.get(name, 1.0)
            )
            policy_costs[name] = cost
            total_cost = _f32(total_cost + cost)
        else:
            policy_costs[name] = 0.0

    situation_cost = _f32(
        wealth_mod
        * _f32(
            sum(
                _f32(data.situations[name].cost)
                for name in state.active_situations
                if name in data.situations
            )
        )
    )
    interest = _f32(_f32(state.debt * interest_rate) * 0.25)

    state.policy_costs = policy_costs
    state.policy_incomes = policy_incomes
    state.policy_income_multipliers = dict(income_multipliers)
    state.policy_cost_multipliers = dict(cost_multipliers)
    state.policy_income_scalars = income_scalar_by_policy
    state.policy_cost_scalars = cost_scalar_by_policy
    state.total_expenditure = _f32(_f32(total_cost + situation_cost) + interest)
    state.total_income = total_income
    return state


def _scalars_by_department(
    policy_scalars: Dict[str, float], data: SimulationData
) -> Dict[str, float]:
    """Collapse per-policy serialized scalars into per-department ones.

    Saves store the ministerial earn/cost scalar on every policy line; all
    policies of a department share one value, so collapsing by department is
    lossless.
    """
    department_scalars: Dict[str, float] = {}
    for name, scalar in policy_scalars.items():
        policy = data.policies.get(name)
        if policy and policy.department:
            department_scalars[policy.department] = scalar
    return department_scalars


def _recompute_orders_finance(
    state: SimulationState,
    data: SimulationData,
) -> SimulationState:
    """Recompute the finance lines the game shows right after orders.

    Placing orders immediately recalculates income/expenditure using the
    *stored* multiplier/scalar neurons (they only update at the end of the
    turn), but with the new active-policy set and the current interest rate.
    """
    rate = state.interest_rate or _interest_rate(state.credit_rating, data)
    return _finance_totals(
        state,
        data,
        income_multipliers=state.policy_income_multipliers,
        cost_multipliers=state.policy_cost_multipliers,
        income_scalars=_scalars_by_department(state.policy_income_scalars, data),
        cost_scalars=_scalars_by_department(state.policy_cost_scalars, data),
        interest_rate=rate,
    )


def _recompute_live_finance(
    state: SimulationState,
    data: SimulationData,
    *,
    multiplier_context: Optional[Dict[str, float]] = None,
) -> SimulationState:
    """Recompute the finance lines the game displays in the <finances> block.

    The game recomputes income and expenditure every turn from the current
    policy values, the ministerial scalars (from competence) and the
    multiplier neurons (which are evaluated from the *previous* turn's node
    values -- pass them as ``multiplier_context``).  The expenditure total
    also includes the wealth-scaled cost of every active situation plus the
    quarterly interest charge on the national debt:

        income      = sum over active income policies of
                          base(min,max,val) * wealth_mod * earn_scalar * incom_mult
        expenditure = sum over active cost policies of
                          base(min,max,val) * wealth_mod * cost_scalar * cost_mult
                      + wealth_mod * sum(active situation costs)
                      + debt * rate * 0.25
    """
    context = {
        **state.values,
        **state.policies,
        **state.situations,
        **state.voter_values,
        **state.voter_percentages,
        **state.voter_frequencies,
    }
    if multiplier_context is None:
        multiplier_context = context

    income_scalars = {
        dept: _f32(0.875 + 0.25 * competence)
        for dept, competence in state.ministerial_competence.items()
    }
    cost_scalars = {
        dept: _f32(2.0 - scalar) for dept, scalar in income_scalars.items()
    }
    income_multipliers: Dict[str, float] = {}
    cost_multipliers: Dict[str, float] = {}
    for policy in data.policies.values():
        name = policy.name
        level = state.policies.get(name, 0.0)
        income_multipliers[name] = (
            _live_multiplier(policy.income_multipliers, level, multiplier_context)
            if policy.max_income > 0
            else 1.0
        )
        cost_multipliers[name] = (
            _live_multiplier(policy.cost_multipliers, level, multiplier_context)
            if policy.max_cost > 0
            else 1.0
        )
    rate = state.interest_rate or _interest_rate(state.credit_rating, data)
    return _finance_totals(
        state,
        data,
        income_multipliers=income_multipliers,
        cost_multipliers=cost_multipliers,
        income_scalars=income_scalars,
        cost_scalars=cost_scalars,
        interest_rate=rate,
    )


def recompute_effects(
    state: SimulationState,
    graph: nx.DiGraph,
    data: Optional[SimulationData] = None,
    effect_histories: Optional[List[EffectHistory]] = None,
) -> SimulationState:
    data = data or load_simulation_data()
    if effect_histories is not None:
        state.effect_histories = effect_histories
    state.effects = _initialize_effect_memory(
        state,
        graph,
        data=data,
        effect_histories=effect_histories,
    )
    if not effect_histories and not state.situations:
        context = {**state.values, **state.policies, **state.situations}
        situations, active = _update_situations(
            state, data, context, state.effects, state.effects
        )
        state.situations = situations
        state.active_situations = active
        state.effects = _initialize_effect_memory(state, graph, data=data)
    _recalculate_budget(state, data)
    return state


def process_end_of_turn(
    state: SimulationState,
    graph: nx.DiGraph,
    data: Optional[SimulationData] = None,
    config: Optional[SimulationConfig] = None,
) -> SimulationState:
    data = data or load_simulation_data()
    source_policies = state.policies.copy()
    runtime_state = _advance_policy_runtime(state, data)
    (
        new_values,
        new_effects,
        situation_values,
        active_situations,
        global_position,
        effect_histories,
    ) = _advance_state_values(
        runtime_state,
        graph,
        data,
        source_policies=source_policies,
    )
    # Ministers gain experience each turn (drifting the income/cost scalars)
    # and -- when the loyalty subsystem is enabled -- their satisfaction and
    # loyalty advance, which drives the per-turn political-capital income.
    experience, competence, effectiveness = _advance_ministers(
        runtime_state, data
    )
    ministerial_value = runtime_state.ministerial_value.copy()
    ministerial_loyalty = runtime_state.ministerial_loyalty.copy()
    if config is not None and config.minister_loyalty:
        ministerial_value, ministerial_loyalty = _advance_minister_loyalty(
            runtime_state, data
        )
    capital_per_minister = data.sim_config.get("POLITICAL_CAPITAL_PER_MINISTER", 6.0)
    max_multiplier = data.sim_config.get("POLITICAL_CAPITAL_MAX_MULTIPLIER", 2.0)
    if config is not None and config.minister_loyalty:
        capital_income = _capital_income(
            replace(runtime_state, ministerial_loyalty=ministerial_loyalty), data
        )
    else:
        capital_income = (
            runtime_state.political_capital_income or capital_per_minister * 5
        )
    capital_cap = capital_income * max_multiplier
    new_capital = _clamp(
        runtime_state.political_capital + capital_income,
        0.0,
        capital_cap,
    )
    # FinanceManager::NextTurn rolls the debt forward by the last finance
    # lines (income - expenditure) and, every other turn, re-derives the
    # credit rating from the debt-to-GDP ratio before computing the rate.
    debt = runtime_state.debt + (
        runtime_state.total_expenditure - runtime_state.total_income
    )
    if runtime_state.turns_since_credit > 0:
        ratio_state = replace(runtime_state, values=state.values.copy(), debt=debt)
        credit_rating = _credit_rating_from_debt_ratio(
            _effective_debt_ratio(ratio_state, data)
        )
        turns_since_credit = 0
    else:
        credit_rating = runtime_state.credit_rating
        turns_since_credit = 1
    new_state = SimulationState(
        country=runtime_state.country,
        turn=runtime_state.turn + 1,
        values=new_values,
        policies=runtime_state.policies.copy(),
        political_capital=new_capital,
        effects=new_effects,
        political_capital_income=runtime_state.political_capital_income,
        effect_histories=effect_histories,
        situations=situation_values,
        active_situations=active_situations,
        response_factors=runtime_state.response_factors.copy(),
        global_economy_position=global_position,
        voter_values=runtime_state.voter_values.copy(),
        voter_percentages=runtime_state.voter_percentages.copy(),
        voter_frequencies=runtime_state.voter_frequencies.copy(),
        voters=[Voter(groups=dict(v.groups), value=v.value, income=v.income, inincome=v.inincome) for v in runtime_state.voters],
        policy_implementations=runtime_state.policy_implementations.copy(),
        policy_active=runtime_state.policy_active.copy(),
        policy_cost_multipliers=runtime_state.policy_cost_multipliers.copy(),
        policy_income_multipliers=runtime_state.policy_income_multipliers.copy(),
        policy_cost_scalars=runtime_state.policy_cost_scalars.copy(),
        policy_income_scalars=runtime_state.policy_income_scalars.copy(),
        effect_throttles=runtime_state.effect_throttles.copy(),
        policy_desired_throttles=runtime_state.policy_desired_throttles.copy(),
        ministerial_effectiveness=effectiveness,
        ministerial_competence=competence,
        ministerial_experience=experience,
        ministerial_suitability=runtime_state.ministerial_suitability.copy(),
        ministerial_loyalty=ministerial_loyalty,
        ministerial_volatility=runtime_state.ministerial_volatility.copy(),
        ministerial_value=ministerial_value,
        ministerial_sympathies={
            k: list(v)
            for k, v in runtime_state.ministerial_sympathies.items()
        },
        debt=debt,
        credit_rating=credit_rating,
        turns_since_credit=turns_since_credit,
        interest_rate=_interest_rate(credit_rating, data),
        event_log=list(state.event_log),
        fired_plots=list(state.fired_plots),
        group_threats=dict(state.group_threats),
    )
    # Finance is recomputed from the advanced policy values and ministerial
    # scalars, with the multiplier neurons evaluated at the *previous* turn's
    # nodes (the one-turn income-history lag) and the interest charged on the
    # freshly-rolled debt.
    multiplier_context = {
        **state.values,
        **state.policies,
        **state.situations,
        **state.voter_values,
        **state.voter_percentages,
        **state.voter_frequencies,
    }
    _recompute_live_finance(
        new_state, data, multiplier_context=multiplier_context
    )
    return _run_stochastic_systems(new_state, graph, data, config)


def _run_stochastic_systems(
    state: SimulationState,
    graph: nx.DiGraph,
    data: SimulationData,
    config: Optional[SimulationConfig],
) -> SimulationState:
    """Apply gated random-event, dilemma, attack and pressure-group systems.

    All systems default to off, which is what the deterministic save-parity
    runs require.  When enabled they mutate the just-advanced state in place
    and return it; the budget is not re-evaluated here because the systems
    operate on voter opinion rather than finance lines.
    """

    if config is None or not any(
        (
            config.random_events,
            config.dilemmas,
            config.pressure_group_events,
            config.assassinations,
        )
    ):
        return state
    from .events import run_random_systems

    return run_random_systems(state, data, config)


def apply_actions(
    state: SimulationState,
    actions: Iterable[PolicyAction],
    data: Optional[SimulationData] = None,
) -> SimulationState:
    data = data or load_simulation_data()
    policies = state.policies.copy()
    policy_desired_throttles = state.policy_desired_throttles.copy()
    target_levels = policy_desired_throttles.copy()
    policy_active = state.policy_active.copy()
    policy_implementations = state.policy_implementations.copy()
    effect_throttles = state.effect_throttles.copy()
    capital = state.political_capital
    for action in actions:
        policy = data.policies.get(action.policy_name)
        if not policy:
            raise ValueError(f"Unknown policy '{action.policy_name}'")
        delta = action.delta
        current = target_levels.get(policy.name, policies.get(policy.name, 0.0))
        if action.action_type == "cancel":
            # A cancellation flips the active flag but keeps the neuron value
            # and slider target where they are (the save shows StateHousing
            # staying at 0.5 after it is switched off).
            new_level = current
        else:
            if delta == 0:
                continue
            new_level = _clamp(current + delta, 0.0, 1.0)
        slider = _get_slider(data, policy)
        if not action.action_type == "cancel" and abs(new_level - current) < EPSILON:
            raise ValueError(f"No change applied to {policy.name}; already at boundary.")
        _validate_policy_level(slider, new_level)
        action_type = _policy_action_type(
            current, new_level, policy, action.action_type
        )
        if action_type == "cancel" and _is_uncancellable(policy):
            raise ValueError(f"{policy.name} is uncancellable and cannot be cancelled.")
        cost, action_type = _policy_action_cost(
            policy, current, new_level, action.action_type
        )
        if cost > capital:
            raise ValueError(
                f"Insufficient political capital for {policy.name} [{action_type}] (cost {cost}, available {capital})"
            )
        capital -= cost
        # The game keeps the policy neuron's current value (<val>) separate
        # from the slider target (<targ>) until the next simulation step.
        # Keep that distinction in the action-phase state as well.
        target_levels[policy.name] = new_level
        policy_desired_throttles[policy.name] = new_level
        if policy.name not in effect_throttles:
            effect_throttles[policy.name] = current
        if action_type == "introduce":
            # A newly-introduced policy takes its slider level as its current
            # value and output throttle immediately (the save shows the
            # introduced CarbonTax at its target with implementation 0); the
            # implementation fraction then ramps up.  The inertial effect
            # rings still lag, so downstream nodes ramp gradually.
            policies[policy.name] = new_level
            effect_throttles[policy.name] = new_level
            policy_active[policy.name] = True
            policy_implementations[policy.name] = 0.0
        elif action_type == "cancel":
            # Switching a policy off freezes its implementation fraction
            # (the save keeps imp=1.0 on a cancelled, fully-rolled-out
            # policy); only its active flag flips.
            policy_active[policy.name] = False
        else:
            policy_active[policy.name] = True
    new_state = SimulationState(
        country=state.country,
        turn=state.turn,
        values=state.values.copy(),
        policies=policies,
        political_capital=capital,
        effects=state.effects.copy(),
        political_capital_income=state.political_capital_income,
        effect_histories=[
            EffectHistory(history.source, history.target, list(history.values), history.effect_id)
            for history in state.effect_histories
        ],
        situations=state.situations.copy(),
        active_situations=state.active_situations.copy(),
        response_factors=state.response_factors.copy(),
        global_economy_position=state.global_economy_position,
        voter_values=state.voter_values.copy(),
        voter_percentages=state.voter_percentages.copy(),
        voter_frequencies=state.voter_frequencies.copy(),
        voters=[Voter(groups=dict(v.groups), value=v.value, income=v.income, inincome=v.inincome) for v in state.voters],
        policy_implementations=policy_implementations,
        policy_active=policy_active,
        policy_cost_multipliers=state.policy_cost_multipliers.copy(),
        policy_income_multipliers=state.policy_income_multipliers.copy(),
        policy_cost_scalars=state.policy_cost_scalars.copy(),
        policy_income_scalars=state.policy_income_scalars.copy(),
        effect_throttles=effect_throttles,
        policy_desired_throttles=policy_desired_throttles,
        ministerial_effectiveness=state.ministerial_effectiveness.copy(),
        ministerial_competence=state.ministerial_competence.copy(),
        ministerial_experience=state.ministerial_experience.copy(),
        ministerial_suitability=state.ministerial_suitability.copy(),
        ministerial_loyalty=state.ministerial_loyalty.copy(),
        ministerial_volatility=state.ministerial_volatility.copy(),
        ministerial_value=state.ministerial_value.copy(),
        ministerial_sympathies={
            k: list(v) for k, v in state.ministerial_sympathies.items()
        },
        debt=state.debt,
        credit_rating=state.credit_rating,
        turns_since_credit=state.turns_since_credit,
        interest_rate=state.interest_rate,
    )
    # Placing orders immediately recomputes the finance lines (the game does
    # this when the player confirms the orders), so the net rolled into the
    # debt at the next turn reflects the post-orders active policy set.
    _recompute_orders_finance(new_state, data)
    return new_state


def list_available_actions(
    state: SimulationState,
    data: Optional[SimulationData] = None,
) -> List[PolicyActionOption]:
    """Enumerate feasible single-step policy moves based on slider metadata and capital."""

    data = data or load_simulation_data()
    options: List[PolicyActionOption] = []
    for policy in data.policies.values():
        current = state.policies.get(policy.name, 0.0)
        slider = _get_slider(data, policy)
        uncancellable = _is_uncancellable(policy)
        capital = state.political_capital
        if current <= EPSILON:
            if uncancellable:
                current = _default_slider_level(slider)
                state.policies[policy.name] = current
            else:
                target = _next_level(0.0, slider, "raise")
                if target is None or target <= EPSILON:
                    target = 1.0
                delta = target - current
                if abs(delta) > EPSILON and policy.introduce_cost <= capital:
                    options.append(
                        PolicyActionOption(
                            policy_name=policy.name,
                            action_type="introduce",
                            delta=delta,
                            resulting_level=target,
                            cost=policy.introduce_cost,
                            implementation_time=policy.implementation_time,
                        )
                    )
                continue
        raise_target = _next_level(current, slider, "raise")
        if raise_target is not None and raise_target - current > EPSILON and policy.raise_cost <= capital:
            options.append(
                PolicyActionOption(
                    policy_name=policy.name,
                    action_type="raise",
                    delta=raise_target - current,
                    resulting_level=raise_target,
                    cost=policy.raise_cost,
                    implementation_time=policy.implementation_time,
                )
            )
        lower_target = _next_level(current, slider, "lower")
        if (
            lower_target is not None
            and lower_target > EPSILON
            and current - lower_target > EPSILON
            and policy.lower_cost <= capital
        ):
            options.append(
                PolicyActionOption(
                    policy_name=policy.name,
                    action_type="lower",
                    delta=lower_target - current,
                    resulting_level=lower_target,
                    cost=policy.lower_cost,
                    implementation_time=policy.implementation_time,
                )
            )
        if not uncancellable and policy.cancel_cost <= capital:
            options.append(
                PolicyActionOption(
                    policy_name=policy.name,
                    action_type="cancel",
                    delta=-current,
                    resulting_level=0.0,
                    cost=policy.cancel_cost,
                    implementation_time=policy.implementation_time,
                )
            )
    return options


def state_to_dict(state: SimulationState) -> Dict[str, object]:
    return {
        "country": state.country,
        "turn": state.turn,
        "political_capital": state.political_capital,
        "political_capital_income": state.political_capital_income,
        "values": state.values,
        "policies": state.policies,
        "effects": state.effects,
        "effect_histories": [
            {
                "source": history.source,
                "target": history.target,
                "values": history.values,
            }
            for history in state.effect_histories
        ],
        "situations": state.situations,
        "active_situations": state.active_situations,
        "response_factors": state.response_factors,
        "policy_costs": state.policy_costs,
        "policy_incomes": state.policy_incomes,
        "total_expenditure": state.total_expenditure,
        "total_income": state.total_income,
        "global_economy_position": state.global_economy_position,
        "voter_values": state.voter_values,
        "voter_percentages": state.voter_percentages,
        "voter_frequencies": state.voter_frequencies,
        "voters": [
            {"groups": dict(v.groups), "value": v.value, "income": v.income, "inincome": v.inincome}
            for v in state.voters
        ],
        "policy_implementations": state.policy_implementations,
        "policy_active": state.policy_active,
        "policy_cost_multipliers": state.policy_cost_multipliers,
        "policy_income_multipliers": state.policy_income_multipliers,
        "policy_cost_scalars": state.policy_cost_scalars,
        "policy_income_scalars": state.policy_income_scalars,
        "effect_throttles": state.effect_throttles,
        "policy_desired_throttles": state.policy_desired_throttles,
        "ministerial_effectiveness": state.ministerial_effectiveness,
        "ministerial_competence": state.ministerial_competence,
        "ministerial_experience": state.ministerial_experience,
        "ministerial_suitability": state.ministerial_suitability,
        "ministerial_loyalty": state.ministerial_loyalty,
        "ministerial_volatility": state.ministerial_volatility,
        "ministerial_value": state.ministerial_value,
        "ministerial_sympathies": {
            k: list(v) for k, v in state.ministerial_sympathies.items()
        },
        "debt": state.debt,
        "credit_rating": state.credit_rating,
        "turns_since_credit": state.turns_since_credit,
        "interest_rate": state.interest_rate,
    }


def state_from_dict(payload: Dict[str, object]) -> SimulationState:
    missing = {"country", "turn", "political_capital", "values", "policies"} - set(payload)
    if missing:
        raise ValueError(f"State payload missing fields: {', '.join(sorted(missing))}")
    state = SimulationState(
        country=str(payload["country"]),
        turn=int(payload["turn"]),
        political_capital=float(payload["political_capital"]),
        political_capital_income=float(
            payload.get("political_capital_income", 0.0)
        ),
        values={k: float(v) for k, v in dict(payload["values"]).items()},
        policies={k: float(v) for k, v in dict(payload["policies"]).items()},
        effects={k: float(v) for k, v in dict(payload.get("effects", {})).items()},
        effect_histories=[
            EffectHistory(
                source=str(item["source"]),
                target=str(item["target"]),
                values=[float(value) for value in item.get("values", [])],
            )
            for item in payload.get("effect_histories", [])
        ],
        situations={k: float(v) for k, v in dict(payload.get("situations", {})).items()},
        active_situations=list(payload.get("active_situations", [])),
        response_factors={k: float(v) for k, v in dict(payload.get("response_factors", {})).items()},
        policy_costs={k: float(v) for k, v in dict(payload.get("policy_costs", {})).items()},
        policy_incomes={k: float(v) for k, v in dict(payload.get("policy_incomes", {})).items()},
        total_expenditure=float(payload.get("total_expenditure", 0.0)),
        total_income=float(payload.get("total_income", 0.0)),
        global_economy_position=float(payload.get("global_economy_position", 0.0)),
        voter_values={k: float(v) for k, v in dict(payload.get("voter_values", {})).items()},
        voter_percentages={
            k: float(v) for k, v in dict(payload.get("voter_percentages", {})).items()
        },
        voter_frequencies={
            k: float(v) for k, v in dict(payload.get("voter_frequencies", {})).items()
        },
        voters=[
            Voter(
                groups={int(k): float(w) for k, w in dict(v.get("groups", {})).items()},
                value=float(v.get("value", 0.0)),
                income=float(v.get("income", 0.0)),
                inincome=float(v.get("inincome", 0.0)),
            )
            for v in payload.get("voters", [])
        ],
        policy_implementations={
            k: float(v)
            for k, v in dict(payload.get("policy_implementations", {})).items()
        },
        policy_active={
            k: bool(v) for k, v in dict(payload.get("policy_active", {})).items()
        },
        policy_cost_multipliers={
            k: float(v)
            for k, v in dict(payload.get("policy_cost_multipliers", {})).items()
        },
        policy_income_multipliers={
            k: float(v)
            for k, v in dict(payload.get("policy_income_multipliers", {})).items()
        },
        policy_cost_scalars={
            k: float(v)
            for k, v in dict(payload.get("policy_cost_scalars", {})).items()
        },
        policy_income_scalars={
            k: float(v)
            for k, v in dict(payload.get("policy_income_scalars", {})).items()
        },
        effect_throttles={
            k: float(v) for k, v in dict(payload.get("effect_throttles", {})).items()
        },
        policy_desired_throttles={
            k: float(v)
            for k, v in dict(payload.get("policy_desired_throttles", {})).items()
        },
        ministerial_effectiveness={
            k: float(v)
            for k, v in dict(payload.get("ministerial_effectiveness", {})).items()
        },
        ministerial_competence={
            k: float(v)
            for k, v in dict(payload.get("ministerial_competence", {})).items()
        },
        ministerial_experience={
            k: float(v)
            for k, v in dict(payload.get("ministerial_experience", {})).items()
        },
        ministerial_suitability={
            k: float(v)
            for k, v in dict(payload.get("ministerial_suitability", {})).items()
        },
        ministerial_loyalty={
            k: float(v)
            for k, v in dict(payload.get("ministerial_loyalty", {})).items()
        },
        ministerial_volatility={
            k: float(v)
            for k, v in dict(payload.get("ministerial_volatility", {})).items()
        },
        ministerial_value={
            k: float(v)
            for k, v in dict(payload.get("ministerial_value", {})).items()
        },
        ministerial_sympathies={
            k: list(v)
            for k, v in dict(payload.get("ministerial_sympathies", {})).items()
        },
        debt=float(payload.get("debt", 0.0)),
        credit_rating=int(payload.get("credit_rating", 0)),
        turns_since_credit=int(payload.get("turns_since_credit", 0)),
        interest_rate=float(payload.get("interest_rate", 0.0)),
    )
    data = load_simulation_data()
    _recalculate_budget(state, data)
    return state


def save_state(state: SimulationState, path: str | Path) -> None:
    Path(path).write_text(json.dumps(state_to_dict(state), indent=2, sort_keys=True))


def load_state(path: str | Path) -> SimulationState:
    payload = json.loads(Path(path).read_text())
    return state_from_dict(payload)


def process_dilemmas(
    state: SimulationState,
    data: Optional[SimulationData] = None,
    config: Optional[SimulationConfig] = None,
) -> SimulationState:
    """Run the dilemma system; disabled unless ``config.dilemmas`` is set."""

    data = data or load_simulation_data()
    if config is None or not config.dilemmas:
        return state
    from .events import run_dilemmas

    return run_dilemmas(state, data, config)


def process_attacks(
    state: SimulationState,
    data: Optional[SimulationData] = None,
    config: Optional[SimulationConfig] = None,
) -> SimulationState:
    """Run extremist pressure-group plots and assassinations."""

    data = data or load_simulation_data()
    if config is None or not (config.assassinations or config.pressure_group_events):
        return state
    from .events import run_attacks

    return run_attacks(state, data, config)


def process_events(
    state: SimulationState,
    data: Optional[SimulationData] = None,
    config: Optional[SimulationConfig] = None,
) -> SimulationState:
    """Run the random-event system; disabled unless ``config.random_events``."""

    data = data or load_simulation_data()
    if config is None or not config.random_events:
        return state
    from .events import run_events

    return run_events(state, data, config)
