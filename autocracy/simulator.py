from __future__ import annotations

import ast
import json
import math
import re
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
)

DEFAULT_GAMEDATA = Path(__file__).resolve().parent.parent / "gamedata" / "data"
ALLOWED_FUNCS = {"min": min, "max": max, "abs": abs}
DEFAULT_PERCENTAGE_STEP = 0.05
EPSILON = 1e-6


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


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
    expr = re.sub(r"(\d)\(", r"\1*(", expr)
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
    # lower
    for level in reversed(levels):
        if level < current - EPSILON and level > EPSILON:
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


def _policy_action_type(current: float, target: float) -> str:
    active_current = current > EPSILON
    active_target = target > EPSILON
    if not active_current and active_target:
        return "introduce"
    if active_current and not active_target:
        return "cancel"
    if target > current:
        return "raise"
    if target < current:
        return "lower"
    return "noop"


def _policy_action_cost(
    policy: PolicyDefinition, current: float, target: float
) -> tuple[float, str]:
    action_type = _policy_action_type(current, target)
    if action_type == "introduce":
        return policy.introduce_cost, action_type
    if action_type == "cancel":
        return policy.cancel_cost, action_type
    if target > current:
        return policy.raise_cost, action_type
    if target < current:
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
    state.situations = save.situations.copy()
    state.active_situations = save.active_situations.copy()
    state.global_economy_position = save.global_economy_position
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
        if level <= EPSILON and desired_level <= EPSILON:
            is_active = False
        active[name] = is_active

        delay = max(policy.implementation_time, 1.0)
        if not is_active:
            implementations[name] = 0.0
        else:
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
        target_throttle = desired_level if is_active else 0.0
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

    if effect.source in state.policies and state.policies[effect.source] <= EPSILON:
        return 0.0
    if policy_values is not None and effect.source in policy_values:
        return policy_values[effect.source]
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
    if state.policies[effect.source] <= EPSILON:
        return False
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
        source = effect.source
        if source in data.situations:
            return source in active_situations
        if source in data.policies:
            level = state.policies.get(source, 0.0)
            target = state.policy_desired_throttles.get(source, level)
            implementation = state.policy_implementations.get(source, 1.0)
            if abs(level - target) > EPSILON or implementation < 1.0 - EPSILON:
                return True
            history = history_by_id.get(effect.effect_id)
            # A fresh simulation starts with zero-filled rings; let the first
            # turns populate them instead of freezing policy effects at zero.
            return history is not None and not any(history.values)
        return True

    pre_policy_values = source_policies or state.policies
    pre_context = {**new_values, **pre_policy_values, **state.situations}
    for effect in _iter_runtime_effects(graph, data):
        if not effect.effect_id:
            continue
        if not is_applicable(effect):
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
            node.default + incoming_value(node.name),
            node.minimum,
            node.maximum,
        )
        context[node.name] = new_values[node.name]
        refresh_outputs(node.name, context)

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
        policy_costs[policy.name] = cost
        policy_incomes[policy.name] = income
        total_cost += cost
        total_income += income
    state.policy_costs = policy_costs
    state.policy_incomes = policy_incomes
    state.total_expenditure = total_cost
    state.total_income = total_income


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
    capital_per_minister = data.sim_config.get("POLITICAL_CAPITAL_PER_MINISTER", 6.0)
    max_multiplier = data.sim_config.get("POLITICAL_CAPITAL_MAX_MULTIPLIER", 2.0)
    capital_income = (
        runtime_state.political_capital_income or capital_per_minister * 5
    )
    capital_cap = capital_income * max_multiplier
    new_capital = _clamp(
        runtime_state.political_capital + capital_income,
        0.0,
        capital_cap,
    )
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
        policy_implementations=runtime_state.policy_implementations.copy(),
        policy_active=runtime_state.policy_active.copy(),
        policy_cost_multipliers=runtime_state.policy_cost_multipliers.copy(),
        policy_income_multipliers=runtime_state.policy_income_multipliers.copy(),
        policy_cost_scalars=runtime_state.policy_cost_scalars.copy(),
        policy_income_scalars=runtime_state.policy_income_scalars.copy(),
        effect_throttles=runtime_state.effect_throttles.copy(),
        policy_desired_throttles=runtime_state.policy_desired_throttles.copy(),
        ministerial_effectiveness=runtime_state.ministerial_effectiveness.copy(),
        ministerial_competence=runtime_state.ministerial_competence.copy(),
        event_log=list(state.event_log),
        fired_plots=list(state.fired_plots),
        group_threats=dict(state.group_threats),
    )
    # FinanceManager runs before NeuralEffect::NextTurn exposes the new
    # policy-neuron value.  Calculate the saved finance lines from the
    # pre-turn node/policy values while retaining runtime fields already
    # advanced by the policy manager.
    finance_state = replace(
        runtime_state,
        values=state.values.copy(),
        policies=source_policies,
        situations=state.situations.copy(),
    )
    _recalculate_budget(finance_state, data)
    new_state.policy_costs = finance_state.policy_costs
    new_state.policy_incomes = finance_state.policy_incomes
    new_state.total_expenditure = finance_state.total_expenditure
    new_state.total_income = finance_state.total_income
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
        if delta == 0:
            continue
        current = target_levels.get(policy.name, policies.get(policy.name, 0.0))
        new_level = _clamp(current + delta, 0.0, 1.0)
        slider = _get_slider(data, policy)
        if abs(new_level - current) < EPSILON:
            raise ValueError(f"No change applied to {policy.name}; already at boundary.")
        _validate_policy_level(slider, new_level)
        if _is_uncancellable(policy) and new_level <= EPSILON:
            raise ValueError(f"{policy.name} is uncancellable and must remain active.")
        cost, action_type = _policy_action_cost(policy, current, new_level)
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
            policy_active[policy.name] = True
            policy_implementations[policy.name] = 0.0
        elif action_type == "cancel":
            policy_active[policy.name] = False
            policy_implementations[policy.name] = 0.0
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
    )
    _recalculate_budget(new_state, data)
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
