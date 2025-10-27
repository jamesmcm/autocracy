from __future__ import annotations

import ast
import json
import math
import re
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx

from . import data_loader
from .models import (
    BudgetModifier,
    CountrySetup,
    Effect,
    NodeDefinition,
    PolicyAction,
    PolicyActionOption,
    PolicyDefinition,
    SimulationData,
    SimulationState,
    SituationDefinition,
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
    expr = re.sub(r"0\.0\.(\d+)", r"0.\1", expr)
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
    policy: PolicyDefinition, level: float, context: Dict[str, float]
) -> float:
    if level <= EPSILON:
        return 0.0
    base = _budget_base_amount(policy.min_cost, policy.max_cost, level)
    multiplier = _evaluate_budget_modifiers(policy.cost_multipliers, level, context)
    return base * multiplier


def _policy_income_amount(
    policy: PolicyDefinition, level: float, context: Dict[str, float]
) -> float:
    if level <= EPSILON:
        return 0.0
    base = _budget_base_amount(policy.min_income, policy.max_income, level)
    multiplier = _evaluate_budget_modifiers(policy.income_multipliers, level, context)
    return base * multiplier


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
) -> bool:
    """If an initial save exists (e.g. uk0.xml) use it to seed node/policy values."""

    saves_root = data.gamedata_root.parent / "saves"
    save_path = saves_root / f"{state.country}0.xml"
    if not save_path.exists():
        return False
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
    return True


def _load_country_save(
    data: SimulationData, country: str, turn_index: int
) -> Optional["SaveGame"]:
    saves_root = data.gamedata_root.parent / "saves"
    path = saves_root / f"{country}{turn_index}.xml"
    if not path.exists():
        return None
    from .savegame import parse_savegame  # late import

    return parse_savegame(path)


def _calibrate_response_factors(
    state: SimulationState,
    graph: nx.DiGraph,
    data: SimulationData,
) -> None:
    baseline_save = _load_country_save(data, state.country, 1)
    if not baseline_save:
        return
    predicted_values, _, _, _ = _advance_state_values(state, graph, data)
    for name, observed in baseline_save.simvalues.items():
        current = state.values.get(name)
        predicted = predicted_values.get(name)
        if current is None or predicted is None:
            continue
        pred_delta = predicted - current
        obs_delta = observed - current
        if abs(pred_delta) < EPSILON:
            if abs(obs_delta) < EPSILON:
                continue
            factor = 0.0
        else:
            factor = obs_delta / pred_delta
        if not math.isfinite(factor):
            continue
        clamped = _clamp(factor, -3.0, 3.0)
        state.response_factors[name] = clamped


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
        )
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
    )
    seeded = _seed_state_from_initial_save(state, data)
    state.effects = _initialize_effect_memory(state, graph)
    context = {**state.values, **state.policies, **state.situations}
    situations, active = _update_situations(state, data, context, state.effects, state.effects)
    state.situations = situations
    state.active_situations = active
    if seeded:
        _calibrate_response_factors(state, graph, data)
    _recalculate_budget(state, data)
    return state, graph


def _source_value(state: SimulationState, source: str) -> float:
    if source in state.values:
        return state.values[source]
    return state.policies.get(source, 0.0)


def _evaluate_effect_with_inertia(
    effect: Effect,
    x_value: float,
    previous_effects: Dict[str, float],
    updated_effects: Dict[str, float],
    context: Dict[str, float],
) -> float:
    target_value = evaluate_expression(effect.expression, x_value, context=context)
    effect_id = effect.effect_id
    if not effect_id:
        return target_value
    previous = previous_effects.get(effect_id, target_value)
    inertia = effect.inertia or 0.0
    if inertia > 1.0:
        new_value = previous + (target_value - previous) / inertia
    else:
        new_value = target_value
    updated_effects[effect_id] = new_value
    return new_value


def _initialize_effect_memory(state: SimulationState, graph: nx.DiGraph) -> Dict[str, float]:
    context = {**state.values, **state.policies, **state.situations}
    effect_values: Dict[str, float] = {}
    for _, _, edge_data in graph.edges(data=True):
        for effect in edge_data.get("effects", []):
            if not effect.effect_id:
                continue
            source_val = _source_value(state, effect.source)
            effect_values[effect.effect_id] = evaluate_expression(effect.expression, source_val, context=context)
    return effect_values


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
        latent = 0.0
        for effect in definition.inputs:
            if not effect.source:
                continue
            source_val = _source_value(state, effect.source)
            latent += _evaluate_effect_with_inertia(
                effect,
                source_val,
                previous_effects,
                updated_effects,
                context,
            )
        latent = _clamp(latent, 0.0, 1.0)
        was_active = name in state.active_situations
        if was_active:
            is_active = latent >= definition.stop_trigger
        else:
            is_active = latent >= definition.start_trigger
        if is_active:
            active.append(name)
        situation_values[name] = latent
    return situation_values, active


def _advance_state_values(
    state: SimulationState,
    graph: nx.DiGraph,
    data: SimulationData,
    response_factors: Optional[Dict[str, float]] = None,
) -> tuple[Dict[str, float], Dict[str, float], Dict[str, float], List[str]]:
    new_values: Dict[str, float] = {}
    new_effects: Dict[str, float] = {}
    context = {**state.values, **state.policies, **state.situations}
    factors = response_factors or {}
    for node_name, node in data.nodes.items():
        incoming = 0.0
        if graph.has_node(node_name):
            for predecessor in graph.predecessors(node_name):
                edge_data = graph.get_edge_data(predecessor, node_name) or {}
                for effect in edge_data.get("effects", []):
                    source_val = _source_value(state, predecessor)
                    incoming += _evaluate_effect_with_inertia(
                        effect,
                        source_val,
                        state.effects,
                        new_effects,
                        context,
                    )
        factor = factors.get(node_name, 1.0)
        new_values[node_name] = state.values.get(node_name, node.default) + incoming * factor

    situation_values, active_situations = _update_situations(
        state,
        data,
        context,
        state.effects,
        new_effects,
    )

    for name, definition in data.situations.items():
        latent_value = situation_values.get(name, 0.0)
        x_value = latent_value if name in active_situations else 0.0
        for effect in definition.effects:
            if effect.target not in new_values:
                continue
            delta = _evaluate_effect_with_inertia(
                effect,
                x_value,
                state.effects,
                new_effects,
                context,
            )
            if name in active_situations:
                new_values[effect.target] += delta

    for node_name, node in data.nodes.items():
        new_values[node_name] = _clamp(
            new_values.get(node_name, node.default), node.minimum, node.maximum
        )
    return new_values, new_effects, situation_values, active_situations


def _recalculate_budget(state: SimulationState, data: SimulationData) -> None:
    context = {**state.values, **state.policies, **state.situations}
    policy_costs: Dict[str, float] = {}
    policy_incomes: Dict[str, float] = {}
    total_cost = 0.0
    total_income = 0.0
    for policy in data.policies.values():
        level = state.policies.get(policy.name, 0.0)
        cost = _policy_cost_amount(policy, level, context)
        income = _policy_income_amount(policy, level, context)
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
) -> SimulationState:
    data = data or load_simulation_data()
    state.effects = _initialize_effect_memory(state, graph)
    context = {**state.values, **state.policies, **state.situations}
    situations, active = _update_situations(state, data, context, state.effects, state.effects)
    state.situations = situations
    state.active_situations = active
    _recalculate_budget(state, data)
    return state


def process_end_of_turn(
    state: SimulationState,
    graph: nx.DiGraph,
    data: Optional[SimulationData] = None,
) -> SimulationState:
    data = data or load_simulation_data()
    new_values, new_effects, situation_values, active_situations = _advance_state_values(
        state, graph, data, response_factors=state.response_factors
    )
    capital_per_minister = data.sim_config.get("POLITICAL_CAPITAL_PER_MINISTER", 6.0)
    max_multiplier = data.sim_config.get("POLITICAL_CAPITAL_MAX_MULTIPLIER", 2.0)
    capital_cap = capital_per_minister * max_multiplier
    new_capital = _clamp(
        state.political_capital + capital_per_minister * 0.5, 0.0, capital_cap
    )
    new_state = SimulationState(
        country=state.country,
        turn=state.turn + 1,
        values=new_values,
        policies=state.policies.copy(),
        political_capital=new_capital,
        effects=new_effects,
        situations=situation_values,
        active_situations=active_situations,
        response_factors=state.response_factors.copy(),
    )
    _recalculate_budget(new_state, data)
    return new_state


def apply_actions(
    state: SimulationState,
    actions: Iterable[PolicyAction],
    data: Optional[SimulationData] = None,
) -> SimulationState:
    data = data or load_simulation_data()
    policies = state.policies.copy()
    capital = state.political_capital
    for action in actions:
        policy = data.policies.get(action.policy_name)
        if not policy:
            raise ValueError(f"Unknown policy '{action.policy_name}'")
        delta = action.delta
        if delta == 0:
            continue
        current = policies.get(policy.name, 0.0)
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
        policies[policy.name] = new_level
    new_state = SimulationState(
        country=state.country,
        turn=state.turn,
        values=state.values.copy(),
        policies=policies,
        political_capital=capital,
        effects=state.effects.copy(),
        situations=state.situations.copy(),
        active_situations=state.active_situations.copy(),
        response_factors=state.response_factors.copy(),
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
        "values": state.values,
        "policies": state.policies,
        "effects": state.effects,
        "situations": state.situations,
        "active_situations": state.active_situations,
        "response_factors": state.response_factors,
        "policy_costs": state.policy_costs,
        "policy_incomes": state.policy_incomes,
        "total_expenditure": state.total_expenditure,
        "total_income": state.total_income,
    }


def state_from_dict(payload: Dict[str, object]) -> SimulationState:
    missing = {"country", "turn", "political_capital", "values", "policies"} - set(payload)
    if missing:
        raise ValueError(f"State payload missing fields: {', '.join(sorted(missing))}")
    state = SimulationState(
        country=str(payload["country"]),
        turn=int(payload["turn"]),
        political_capital=float(payload["political_capital"]),
        values={k: float(v) for k, v in dict(payload["values"]).items()},
        policies={k: float(v) for k, v in dict(payload["policies"]).items()},
        effects={k: float(v) for k, v in dict(payload.get("effects", {})).items()},
        situations={k: float(v) for k, v in dict(payload.get("situations", {})).items()},
        active_situations=list(payload.get("active_situations", [])),
        response_factors={k: float(v) for k, v in dict(payload.get("response_factors", {})).items()},
        policy_costs={k: float(v) for k, v in dict(payload.get("policy_costs", {})).items()},
        policy_incomes={k: float(v) for k, v in dict(payload.get("policy_incomes", {})).items()},
        total_expenditure=float(payload.get("total_expenditure", 0.0)),
        total_income=float(payload.get("total_income", 0.0)),
    )
    data = load_simulation_data()
    _recalculate_budget(state, data)
    return state


def save_state(state: SimulationState, path: str | Path) -> None:
    Path(path).write_text(json.dumps(state_to_dict(state), indent=2, sort_keys=True))


def load_state(path: str | Path) -> SimulationState:
    payload = json.loads(Path(path).read_text())
    return state_from_dict(payload)


def process_dilemmas(*args, **kwargs):
    """Stub placeholder for dilemma handling (not yet implemented)."""

    raise NotImplementedError("Dilemma processing is not implemented yet.")


def process_attacks(*args, **kwargs):
    """Stub placeholder for security attack processing (not yet implemented)."""

    raise NotImplementedError("Attack processing is not implemented yet.")


def process_events(*args, **kwargs):
    """Stub placeholder for random events (not yet implemented)."""

    raise NotImplementedError("Random event processing is not implemented yet.")
