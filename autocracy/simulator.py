from __future__ import annotations

import ast
import json
import math
import re
import struct
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import networkx as nx

from . import data_loader
from .models import (
    BudgetModifier,
    CountrySetup,
    Effect,
    EffectHistory,
    ElectionForecast,
    Grudge,
    NodeDefinition,
    PolicyAction,
    PolicyActionOption,
    PolicyDefinition,
    PartyState,
    SimulationConfig,
    SimulationData,
    SimulationState,
    SituationDefinition,
    SliderDefinition,
    Voter,
)

DEFAULT_GAMEDATA = Path(__file__).resolve().parent.parent / "gamedata" / "data"
ALLOWED_FUNCS = {"min": min, "max": max, "abs": abs}
DEFAULT_PERCENTAGE_STEP = 0.05
EPSILON = 1e-6
POLICY_FINANCE_HISTORY_LENGTH = 20


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


def _copy_voter(voter: Voter) -> Voter:
    """Copy mutable voter containers while retaining every runtime field."""
    return replace(
        voter,
        groups=dict(voter.groups),
        organizations=list(voter.organizations),
    )


def _copy_party(party: PartyState) -> PartyState:
    """Copy serialized party history rings without sharing mutable lists."""
    return replace(
        party,
        member_history=list(party.member_history),
        activist_history=list(party.activist_history),
    )


def _copy_grudge(grudge: Grudge) -> Grudge:
    """Copy one native grudge without sharing mutable runtime state."""
    return replace(grudge)


def _advance_grudges(grudges: Iterable[Grudge]) -> List[Grudge]:
    """Advance native CreateGrudge inputs by one turn.

    A grudge is created before ``NeuralEffect::NextTurn`` and therefore its
    first visible value is already multiplied by its decay factor.  Existing
    grudges follow the same rule on every subsequent turn.  Keeping the
    records separate (rather than summing them) preserves different decay
    rates for simultaneous events targeting one node.
    """
    return [
        replace(
            grudge,
            value=_f32(grudge.value * grudge.decay),
        )
        for grudge in grudges
    ]


def _restore_native_grudge_inputs(grudges: Iterable[Grudge]) -> List[Grudge]:
    """Convert serialized post-decay grudges to this turn's input values."""
    restored: List[Grudge] = []
    for grudge in grudges:
        if grudge.decay:
            value = _f32(grudge.value / grudge.decay)
        else:
            value = grudge.value
        restored.append(replace(grudge, value=value))
    return restored


def _grudge_value(grudges: Iterable[Grudge], target: str) -> float:
    """Return the current neural-input sum for one target."""
    return _f32(
        sum(
            _f32(grudge.value)
            for grudge in grudges
            if grudge.target == target
        )
    )


def _grudge_voter_inputs(grudges: Iterable[Grudge]) -> Dict[str, float]:
    """Sum grudge inputs whose targets are aggregate voter types."""
    totals: Dict[str, float] = {}
    voter_names = set(VOTER_SYMBOL_NAMES.values())
    for grudge in grudges:
        if grudge.target not in voter_names:
            continue
        totals[grudge.target] = _f32(
            totals.get(grudge.target, 0.0) + grudge.value
        )
    return totals


def _complete_policy_finance_histories(
    histories: Dict[str, List[float]],
    current_values: Dict[str, float],
    data: SimulationData,
) -> Dict[str, List[float]]:
    """Return fixed-size newest-first policy finance rings.

    Old JSON snapshots did not carry the rings.  Filling those from the live
    line keeps them usable while preserving every value from a real save when
    one was loaded.
    """

    completed: Dict[str, List[float]] = {}
    for name in data.policies:
        values = [float(value) for value in histories.get(name, [])]
        fallback = float(current_values.get(name, 0.0))
        if not values:
            values = [fallback] * POLICY_FINANCE_HISTORY_LENGTH
        elif len(values) < POLICY_FINANCE_HISTORY_LENGTH:
            values.extend([values[-1]] * (POLICY_FINANCE_HISTORY_LENGTH - len(values)))
        completed[name] = values[:POLICY_FINANCE_HISTORY_LENGTH]
    return completed


def _advance_policy_finance_histories(
    state: SimulationState,
    data: SimulationData,
) -> tuple[Dict[str, List[float]], Dict[str, List[float]]]:
    """Write the pre-policy-update finance lines into the game-style rings."""

    cost_histories = _complete_policy_finance_histories(
        state.policy_cost_histories, state.policy_costs, data
    )
    income_histories = _complete_policy_finance_histories(
        state.policy_income_histories, state.policy_incomes, data
    )
    next_costs: Dict[str, List[float]] = {}
    next_incomes: Dict[str, List[float]] = {}
    for name in data.policies:
        next_costs[name] = [
            float(state.policy_costs.get(name, 0.0)),
            *cost_histories[name][: POLICY_FINANCE_HISTORY_LENGTH - 1],
        ]
        next_incomes[name] = [
            float(state.policy_incomes.get(name, 0.0)),
            *income_histories[name][: POLICY_FINANCE_HISTORY_LENGTH - 1],
        ]
    return next_costs, next_incomes


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


@lru_cache(maxsize=4096)
def _compiled_expression(expression: str):
    """Parse and validate one data-file expression once per process."""

    sanitized = _sanitize_expression(expression.strip())
    tree = ast.parse(sanitized, mode="eval")
    _validate_expression(tree)
    return compile(tree, "<effect>", "eval")


def evaluate_expression(
    expression: str, x: float, context: Optional[Dict[str, float]] = None
) -> float:
    """Safely evaluate a Democracy 3 equation."""

    if not expression:
        return 0.0
    code = _compiled_expression(expression)
    allowed_names = {"x": x, **ALLOWED_FUNCS}
    if context:
        allowed_names.update(context)
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
    state.hidden_histories = {
        name: list(values) for name, values in save.hidden_histories.items()
    }
    # Keep the save's per-node value rings so a forecasting agent starting
    # from this mission can condition on the pre-game covariate history.
    state.value_histories = {
        name: list(values) for name, values in save.simvalue_histories.items()
    }
    state.value_histories_turn = save.turn
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
    for name, value in save.voter_incomes.items():
        state.voter_incomes[name] = value
        state.values[name] = value
    state.voter_frequency_grudges = save.voter_frequency_grudges.copy()
    state.grudges = [_copy_grudge(grudge) for grudge in save.grudges]
    state.voters = [_copy_voter(v) for v in save.voters]
    state.parties = {
        name: _copy_party(party) for name, party in save.parties.items()
    }
    state.policy_implementations = save.policy_implementations.copy()
    state.policy_active = save.policy_active.copy()
    state.policy_cost_histories = {
        name: list(values) for name, values in save.policy_cost_histories.items()
    }
    state.policy_income_histories = {
        name: list(values) for name, values in save.policy_income_histories.items()
    }
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
    if save.election_turns_until is not None:
        state.election_turns_until = save.election_turns_until
    if save.election_current_term is not None:
        state.election_current_term = save.election_current_term
    state.poll_rate = save.poll_rate or 0.0
    state.peak_poll_rate = save.peak_poll_rate or 0.0
    state.poll_history = list(save.poll_history)
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
        calibration=data_loader.load_calibration(root),
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
        # A mission override replaces the shipped effect for the host/target
        # pair instead of adding a second line (Germany's
        # ``alcoholproductivity.ini`` rewrites ``AlcoholConsumption ->
        # WorkerProductivity``; the base ``0-(0.17*x)`` term would otherwise
        # be double-counted against the native value).
        if graph.has_edge(host, target):
            graph[host][target]["effects"] = []
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
        policy_finance_levels=policies.copy(),
        political_capital_income=starting_capital,
        global_economy_position=setup.economic_cycle_start,
        election_turns_until=max(0, setup.term_length),
        election_current_term=0,
    )
    # A fresh native VoterManager seeds its linked-list percentage from the
    # CSV column, while the nested frequency neuron itself starts at zero.
    for node in data.nodes.values():
        if node.category != "VOTER_FREQ" or not node.initial_percentage:
            continue
        percentage_name = f"{node.name[:-5]}_perc"
        state.voter_percentages[percentage_name] = node.initial_percentage
        state.values[percentage_name] = node.initial_percentage
    state.policy_desired_throttles = state.policies.copy()
    save = _seed_state_from_initial_save(state, data)
    if save is None:
        # No reference save for this country: synthesize a deterministic
        # electorate so elections and the voter model work from turn 0.  The
        # mission scripts add the same scripted frequency/neuron grudges the
        # native game applies at load time.
        from .events import apply_country_scripts
        from .voters import apply_electorate, generate_electorate

        voters, parties = generate_electorate(data, country)
        apply_electorate(state, data, voters, parties)
        apply_country_scripts(state, data)
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
    state.policy_cost_histories = _complete_policy_finance_histories(
        state.policy_cost_histories, state.policy_costs, data
    )
    state.policy_income_histories = _complete_policy_finance_histories(
        state.policy_income_histories, state.policy_incomes, data
    )
    if save:
        # The <finances> block totals (which include situation costs and debt
        # interest) are the ground truth for this snapshot; keep them so the
        # next debt roll uses the game's displayed net.
        state.total_income = save.total_income
        state.total_expenditure = save.total_expenditure
    state.policy_finance_levels = state.policies.copy()
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


def _voter_income_names(state: SimulationState, graph: nx.DiGraph) -> List[str]:
    """Return the native nested VoterType income neurons represented here."""
    names = set(state.voter_incomes)
    names.update(
        name
        for name in graph.nodes
        if isinstance(name, str) and name.endswith("_income")
    )
    return sorted(names)


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
    # Data-driven scale exemptions (calibration.json).  The never-introduced
    # CitizenshipTests constant is retained as an unscaled input by the
    # shipped native save.
    scale_mode = data.calibration.get("effect_scale", {}).get(
        f"{effect.source} -> {effect.target}"
    )
    if scale_mode == "implementation":
        return implementation
    if scale_mode == "unscaled":
        return 1.0
    return effectiveness * implementation


def _effect_calibration_offset(data: SimulationData, effect: Effect) -> float:
    """Return a measured native parser offset for one graph effect.

    A small number of installed-game equations do not evaluate to the text
    in the shipped CSV.  Keeping those observations in calibration data lets
    the normal graph code remain data-driven and makes the exception portable
    to a different gamedata root.
    """
    offsets = data.calibration.get("effect_offsets", {})
    if not isinstance(offsets, Mapping):
        return 0.0
    value = offsets.get(f"{effect.source} -> {effect.target}", 0.0)
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


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


def _effect_is_applicable(
    state: SimulationState, effect: Effect, data: SimulationData
) -> bool:
    """Mirror ``SIM_Neuron::IsApplicable`` for policy-owned outputs.

    A disabled policy contributes no effect at all.  Evaluating its equation
    at ``x=0`` would incorrectly retain constant terms such as ``-0.10``.
    During implementation, the game also gates the output by the policy's
    implementation fraction.
    """

    if effect.source not in state.policies:
        return True
    # Data-driven exemption (calibration.json): a game-data set may need a
    # policy link to remain live while its policy is inactive.  The shipped UK
    # calibration does not enable this exception for CitizenshipTests.
    if data.calibration.get("effect_applicability", {}).get(
        f"{effect.source} -> {effect.target}"
    ) == "always":
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
    offset: float = 0.0,
) -> float:
    target_value = _clamp(
        evaluate_expression(effect.expression, x_value, context=context) + offset,
        -1.0,
        1.0,
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
        if not _effect_is_applicable(state, effect, data):
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
            (
                evaluate_expression(effect.expression, x_value, context=context)
                + _effect_calibration_offset(data, effect)
            )
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
            if not _effect_is_applicable(state, effect, data):
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
                    (
                        evaluate_expression(
                            effect.expression, source_val, context=context
                        )
                        + _effect_calibration_offset(data, effect)
                    )
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
                    (
                        evaluate_expression(
                            effect.expression, latent, context=context
                        )
                        + _effect_calibration_offset(data, effect)
                    )
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


def _situation_input_overrides(
    setup: CountrySetup,
) -> Dict[str, Dict[str, str]]:
    """Map each situation target to its mission override equations by host.

    Country overrides such as France's ``alcoholabuse.ini``
    (``AlcoholConsumption -> Alcoholism``) and Germany's
    ``poorearning_generalstrike.ini`` (``_LowIncome -> GeneralStrike``) change
    the equations that feed a situation's latent value.  They are applied to
    the graph edges in ``build_country_graph``, but situation latents are
    recomputed from ``SituationDefinition.inputs`` (the ``situations.csv``
    effects), so the same override must be folded into that input list.
    """

    overrides: Dict[str, Dict[str, str]] = {}
    for override in setup.overrides:
        target = override.get("TargetName")
        source = override.get("HostName")
        equation = override.get("Equation")
        if not target or not source or not equation:
            continue
        overrides.setdefault(target, {})[source] = equation
    return overrides


def _effective_situation_inputs(
    name: str,
    definition: SituationDefinition,
    override_map: Mapping[str, Mapping[str, str]],
) -> List[Effect]:
    """Return *definition*'s input effects after mission overrides apply."""

    target_overrides = override_map.get(name, {})
    if not target_overrides:
        return list(definition.inputs)
    effective: List[Effect] = []
    present: set[str] = set()
    for effect in definition.inputs:
        if effect.source in target_overrides:
            equation = target_overrides[effect.source]
            if equation.upper() == "DELETE":
                continue
            effective.append(replace(effect, expression=equation))
        else:
            effective.append(effect)
        present.add(effect.source)
    for source, equation in target_overrides.items():
        if source in present or equation.upper() == "DELETE":
            continue
        effective.append(
            Effect(source=source, target=name, expression=equation, inertia=0.0)
        )
    return effective


def _update_situations(
    state: SimulationState,
    data: SimulationData,
    context: Dict[str, float],
    previous_effects: Dict[str, float],
    updated_effects: Dict[str, float],
) -> tuple[Dict[str, float], List[str]]:
    """Recompute latent situation values and determine which remain active."""

    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    override_map = _situation_input_overrides(setup)
    situation_values: Dict[str, float] = {}
    active: List[str] = []
    for name, definition in data.situations.items():
        prerequisites_met = all(
            state.policies.get(prerequisite, 0.0) > EPSILON
            for prerequisite in definition.prerequisites
        )
        latent = definition.default if prerequisites_met else 0.0
        for effect in _effective_situation_inputs(name, definition, override_map):
            if not effect.source:
                continue
            if not _effect_is_applicable(state, effect, data):
                continue
            source_val = _effect_source_value(state, effect, context=context)
            latent += _evaluate_effect_with_inertia(
                effect,
                source_val,
                previous_effects,
                updated_effects,
                context,
                offset=_effect_calibration_offset(data, effect),
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
# These four links are overwritten by ``SIM_Voter::NextTurn`` through
# ``ForceVoter`` after the generic frequency-based membership refresh.  Their
# linked-list membership therefore follows the raw forced coefficient, not
# the VoterType ``freqval`` base used by the other ordinary groups.
POLITICAL_GROUP_SYMBOLS = {0, 1, 6, 17}

# These manager-owned neurons are not rows in simulation.csv, but the native
# simulation keeps them in its hidden-neuron list.  They have a 0.5 base and
# are recalculated after the effect vector, before the voter manager refreshes
# the political group pairs.
GLOBAL_POLITICAL_NODES = {
    "_global_socialism": 0.5,
    "_global_liberalism": 0.5,
}
GLOBAL_INCOMING_NODES = {
    "_security_": 0.0,
    "_winning_": 0.0,
}


def _advance_global_political_nodes(
    new_values: Dict[str, float],
    incoming_value: Callable[[str], float],
) -> None:
    """Recalculate the hidden global ideology neurons from their inputs."""
    for name, default in GLOBAL_POLITICAL_NODES.items():
        if name not in new_values:
            continue
        new_values[name] = _clamp(
            _f32(_f32(default) + incoming_value(name)),
            -1.0,
            1.0,
        )


def _advance_global_incoming_nodes(
    new_values: Dict[str, float],
    incoming_value: Callable[[str], float],
) -> None:
    """Recalculate hidden manager neurons fed by ordinary effect links."""
    for name, default in GLOBAL_INCOMING_NODES.items():
        if name not in new_values:
            continue
        new_values[name] = _clamp(
            _f32(_f32(default) + incoming_value(name)),
            -1.0,
            1.0,
        )


def _advance_native_political_groups(
    state: SimulationState,
    node_values: Optional[Dict[str, float]] = None,
) -> None:
    """Refresh the four party-ideology groups using native ``ForceVoter``."""
    values = node_values if node_values is not None else state.values
    socialism = values.get("_global_socialism", 0.5)
    liberalism = values.get("_global_liberalism", 0.5)
    for voter in state.voters:
        socialist = _f32(
            _f32(voter.initial_socialism + socialism) * _f32(0.5)
        )
        liberal = _f32(
            _f32(voter.initial_liberalism + liberalism) * _f32(0.5)
        )
        voter.groups[0] = socialist
        voter.groups[1] = _f32(_f32(1.0) - socialist)
        voter.groups[6] = liberal
        voter.groups[17] = _f32(_f32(1.0) - liberal)


def _native_income_group_memberships(
    voter: Voter,
    membership_threshold: float,
) -> Dict[int, float]:
    """Return the native income-group weights for one voter.

    ``SIM_Voter::UpdateIncome`` evaluates three overlapping sinusoidal
    windows against the current income value and keeps only the strongest
    candidate.  ``AddToIncomeGroup`` then applies the voter manager's
    membership threshold as a floor.  The saved UK data uses the native
    ``VOTER_GROUP_MEMBERSHIP_THRESHHOLD=0.5``; for example, the first
    captured voter produces the serialized ``0.916348`` wealthy weight.

    The income-neuron contribution is not serialized by the game.  The
    model's ``income`` field is retained as that transient contribution when
    a caller supplies one, and is zero for the captured save corpus.
    """

    current_income = _clamp(voter.inincome + voter.income, 0.0, 1.0)
    candidates = {
        12: (-0.3, 0.3),
        13: (0.2, 0.8),
        11: (0.7, 1.3),
    }
    weights = {
        symbol: _clamp(
            math.sin(
                (current_income - lower) / (upper - lower) * math.pi
            ),
            0.0,
            1.0,
        )
        if lower <= current_income <= upper
        else 0.0
        for symbol, (lower, upper) in candidates.items()
    }
    primary = max(weights, key=weights.get)
    membership = _clamp(
        membership_threshold
        + (1.0 - membership_threshold) * weights[primary],
        0.0,
        1.0,
    )
    return {
        symbol: membership if symbol == primary else 0.0
        for symbol in INCOME_GROUP_NODES
    }


def _native_voter_value(
    voter: Voter,
    voter_values: Dict[str, float],
    voter_frequencies: Dict[str, float],
    membership_threshold: float,
) -> Optional[float]:
    """Recalculate one live voter's value from its native host links.

    ``SIM_Voter`` is a ``SIM_Neuron``.  Its value is therefore rebuilt from
    the currently linked voter-type neurons during the main neuron pass; it
    is not the previous individual value plus the change in each group.  The
    voter manager's linked-list membership test is the same one used by the
    percentage calculation below: political groups use their forced raw
    coefficient, while ordinary groups add the current voter-type frequency
    base.  The effective input weight is that same frequency-adjusted value,
    clamped to the native [0, 1] range, rather than the serialized raw group
    coefficient.

    The executable accumulates these inputs as single-precision floats.  A
    ``None`` result means the caller does not have enough live voter-type
    data to use this native path and should retain the compatibility fallback
    used for synthetic/legacy states.
    """

    total = 0.0
    linked = False
    for symbol, member in voter.groups.items():
        name = VOTER_SYMBOL_NAMES.get(symbol)
        if name is None or name not in voter_values:
            continue
        effective_member = member
        if symbol in POLITICAL_GROUP_SYMBOLS:
            is_linked = member > membership_threshold
        else:
            frequency = voter_frequencies.get(f"{name}_freq", 0.0)
            effective_member = _clamp(
                _f32(_f32(member) + _f32(frequency)), 0.0, 1.0
            )
            is_linked = effective_member >= membership_threshold
        if not is_linked:
            continue
        linked = True
        total = _f32(
            _f32(total)
            + _f32(effective_member * voter_values.get(name, 0.0))
        )
    if not linked:
        return None
    return _clamp(total, -1.0, 1.0)


def _effects_on_voter_types(
    data: SimulationData,
    policies: Dict[str, float],
    node_values: Dict[str, float],
    situation_effects: Optional[Dict[str, float]] = None,
    effect_values: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """Sum current policy, node, and situation effects on voter types.

    Situation outputs are managed outside the ordinary graph, but their live
    values are part of the same native effect vector.  Callers pass the
    already-computed effect values so inertial situation links retain their
    serialized ring state instead of being evaluated from the latent again.
    """
    totals: Dict[str, float] = {}
    context = {**node_values, **policies}

    # In the native pass the voter-type neuron consumes the current values of
    # its incoming SIM_NeuralEffects.  Re-evaluating every policy equation
    # here would resurrect inactive policy constants (for example the
    # ``0.20`` term in a disabled Conservatives link) and would bypass
    # inertia/ministerial scaling.  The runtime effect vector is therefore
    # authoritative whenever the caller has one.
    if effect_values is not None:
        for source_name, target_name, edge_data in data.graph.edges(data=True):
            target = data.nodes.get(target_name)
            source = data.nodes.get(source_name)
            if target is None or target.category != "VOTER":
                continue
            # VoterType influences are loaded through the graph for
            # inspection, but the executable installs them as AddAdjusters
            # on the source VoterType rather than ordinary neuron inputs.
            if source is not None and source.category == "VOTER":
                continue
            for effect in edge_data.get("effects", []):
                if effect.effect_id is None:
                    continue
                totals[target_name] = totals.get(target_name, 0.0) + (
                    effect_values.get(effect.effect_id, 0.0)
                )
        if situation_effects is not None:
            for definition in data.situations.values():
                for effect in definition.effects:
                    target = data.nodes.get(effect.target)
                    if target is None or target.category != "VOTER":
                        continue
                    effect_id = effect.effect_id
                    if effect_id is None:
                        continue
                    totals[effect.target] = totals.get(effect.target, 0.0) + (
                        situation_effects.get(effect_id, 0.0)
                    )
        return totals

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
    if situation_effects is not None:
        for definition in data.situations.values():
            for effect in definition.effects:
                if effect.target not in data.nodes:
                    continue
                if data.nodes[effect.target].category != "VOTER":
                    continue
                effect_id = effect.effect_id
                if effect_id is None:
                    continue
                totals[effect.target] = totals.get(effect.target, 0.0) + (
                    situation_effects.get(effect_id, 0.0)
                )
    return totals


def _advance_party_memberships(
    state: SimulationState,
    data: SimulationData,
) -> None:
    """Advance the native voter sympathy and party-membership step.

    Static analysis of ``SIM_Voter::ConsiderPartyMembership`` shows that the
    method consumes the approval value calculated at the end of the previous
    turn.  Before the optional perception modifiers, that value is
    ``(voter.value + 1) * 0.5``.  The simulator does not yet model those
    modifiers, but retaining this base transform reproduces the captured
    sympathy thresholds without treating the raw ``[-1, 1]`` value as a
    percentage.

    The live party manager owns its voter lists, so the save only exposes the
    member-count history.  We update that history from the pre-transition
    counts, matching the game's ``memberslastturn`` field, while retaining
    the per-voter party string for the next turn.
    """
    if not state.voters:
        return

    opposition_party = next(
        (
            name
            for name, party in state.parties.items()
            if party.party_type == 1
        ),
        None,
    )
    player_party = next(
        (
            name
            for name, party in state.parties.items()
            if party.party_type == 0
        ),
        None,
    )
    previous_counts = {
        name: sum(1 for voter in state.voters if voter.party == name)
        for name in state.parties
    }

    opposition_increase = data.sim_config.get(
        "VOTER_OPPOSITION_INCREASE_SYMPATHY_BELOW", 0.1
    )
    opposition_decrease = data.sim_config.get(
        "VOTER_OPPOSITION_DECREASE_SYMPATHY_ABOVE", 0.15
    )
    opposition_gain = data.sim_config.get(
        "VOTER_OPPOSITON_SYMPATHY_GAIN", 0.1
    )
    opposition_decay = data.sim_config.get(
        "VOTER_OPPOSITON_SYMPATHY_DECAY", 0.1
    )
    opposition_join = data.sim_config.get(
        "VOTER_OPPOSITION_JOIN_THRESHHOLD", 0.7
    )
    opposition_leave = data.sim_config.get(
        "VOTER_OPPOSITION_LEAVE_THRESHHOLD", 0.2
    )
    player_increase = data.sim_config.get(
        "VOTER_PLAYER_INCREASE_SYMPATHY_ABOVE", 0.9
    )
    player_decrease = data.sim_config.get(
        "VOTER_PLAYER_DECREASE_SYMPATHY_BELOW", 0.85
    )
    player_gain = data.sim_config.get("VOTER_PLAYER_SYMPATHY_GAIN", 0.1)
    player_decay = data.sim_config.get("VOTER_PLAYER_SYMPATHY_DECAY", 0.1)
    player_join = data.sim_config.get("VOTER_PLAYER_JOIN_THRESHHOLD", 0.7)
    player_leave = data.sim_config.get("VOTER_PLAYER_LEAVE_THRESHHOLD", 0.2)

    for voter in state.voters:
        approval = _clamp((voter.value + 1.0) * 0.5, 0.0, 1.0)

        if approval < opposition_increase:
            voter.opposition_sympathy = _clamp(
                voter.opposition_sympathy + opposition_gain,
                0.0,
                1.0,
            )
        elif approval > opposition_decrease:
            voter.opposition_sympathy = _clamp(
                voter.opposition_sympathy - opposition_decay,
                0.0,
                1.0,
            )

        if approval > player_increase:
            voter.player_sympathy = _clamp(
                voter.player_sympathy + player_gain,
                0.0,
                1.0,
            )
        elif approval < player_decrease:
            voter.player_sympathy = _clamp(
                voter.player_sympathy - player_decay,
                0.0,
                1.0,
            )

        if voter.party == opposition_party and opposition_party is not None:
            if voter.opposition_sympathy < opposition_leave:
                voter.party = "0"
        elif voter.party == player_party and player_party is not None:
            if voter.player_sympathy < player_leave:
                voter.party = "0"
        elif voter.party in ("", "0"):
            # The executable keeps a voter from joining one party while it
            # has meaningful sympathy for the other.  The 0.1 cross-party
            # guard is a native constant, not a simconfig entry.
            if (
                opposition_party is not None
                and voter.opposition_sympathy > opposition_join
                and voter.player_sympathy <= 0.1
            ):
                voter.party = opposition_party
            elif (
                player_party is not None
                and voter.player_sympathy > player_join
                and voter.opposition_sympathy <= 0.1
            ):
                voter.party = player_party

    for name, party in state.parties.items():
        previous = previous_counts[name]
        # SIM_Party::NextTurn compares the serialized previous count with
        # the live list before replacing ``memberslastturn``.  Its status is
        # 0 for growth, 1 for decline, and 2 for an unchanged membership.
        if previous > party.members_last_turn:
            party.status = 0
        elif previous < party.members_last_turn:
            party.status = 1
        else:
            party.status = 2
        party.members_last_turn = previous
        party.member_history = [previous, *party.member_history[:9]]
        # SIM_Party::NextTurn shifts the activist ring even when the live
        # member count did not change. The current activist count is owned by
        # the native party list and is not serialized, so retain the loaded
        # head rather than estimating it from the incomplete approval state.
        activist = party.activist_history[0] if party.activist_history else 0
        party.activist_history = [activist, *party.activist_history[:9]]


def _advance_voters_and_income_nodes(
    state: SimulationState,
    data: SimulationData,
    new_values: Dict[str, float],
    source_policies: Optional[Dict[str, float]] = None,
    previous_voter_frequencies: Optional[Dict[str, float]] = None,
    situation_effects: Optional[Dict[str, float]] = None,
    previous_grudges: Optional[Iterable[Grudge]] = None,
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
    _advance_native_political_groups(state, new_values)
    _advance_party_memberships(state, data)
    previous_voter_values = state.voter_values.copy()
    current = _effects_on_voter_types(
        data,
        state.policies,
        new_values,
        situation_effects=situation_effects,
        effect_values=situation_effects,
    )
    current_grudge_inputs = _grudge_voter_inputs(state.grudges)
    previous_grudge_inputs = _grudge_voter_inputs(
        state.grudges if previous_grudges is None else previous_grudges
    )
    for name, value in current_grudge_inputs.items():
        current[name] = _f32(current.get(name, 0.0) + value)
    source = source_policies if source_policies is not None else state.policies
    previous = _effects_on_voter_types(
        data,
        source,
        state.values,
        situation_effects=state.effects,
        effect_values=state.effects,
    )
    for name, value in previous_grudge_inputs.items():
        previous[name] = _f32(previous.get(name, 0.0) + value)

    # The loaded voter-type neuron already contains the static load-time
    # inputs.  Native NextTurn applies the change in its incoming effect sum;
    # rebuilding from the CSV default would discard those serialized inputs
    # (and produces fixed -0.05/-0.15 offsets in the first quiet turn).
    for symbol, name in VOTER_SYMBOL_NAMES.items():
        if name in state.voter_values:
            state.voter_values[name] = _clamp(
                state.voter_values[name]
                + current.get(name, 0.0)
                - previous.get(name, 0.0),
                -1.0,
                1.0,
            )

    membership_frequencies = (
        previous_voter_frequencies
        if previous_voter_frequencies is not None
        else state.voter_frequencies
    )
    membership_threshold = data.sim_config.get(
        "VOTER_GROUP_MEMBERSHIP_THRESHHOLD", 0.5
    )
    for voter in state.voters:
        # The native UpdateIncome path uses overlapping [−.3,.3], [.2,.8]
        # and [.7,1.3] windows, selects the largest candidate, and applies
        # the manager threshold to the winning membership.
        voter.groups.update(
            _native_income_group_memberships(voter, membership_threshold)
        )
        native_value = _native_voter_value(
            voter,
            state.voter_values,
            state.voter_frequencies,
            membership_threshold,
        )
        if native_value is not None:
            # Native voter values are neuron sums over the live voter-type
            # links.  In particular, do not add the GDP-collapse heuristic
            # here: any economy-wide voter effect is already represented by
            # the current voter-type neurons.
            voter.value = native_value
        else:
            # Compatibility path for hand-built/legacy states that do not
            # contain the voter-type values needed by the live-link model.
            delta = 0.0
            for symbol, member in voter.groups.items():
                if member <= 0.0:
                    continue
                name = VOTER_SYMBOL_NAMES.get(symbol)
                if name is None:
                    continue
                delta += (
                    current.get(name, 0.0) - previous_voter_values.get(name, 0.0)
                ) * member
            # The voter-opinion feedback: once the economy crashes (GDP
            # collapses through the calibration threshold) the compatibility
            # path slides voters toward -1.
            crash_cfg = data.calibration.get("voter_collapse", {}).get("gdp_crash", {})
            crash_slope = float(crash_cfg.get("slope", 2.0))
            crash_threshold = float(crash_cfg.get("threshold", 0.15))
            crash = min(
                0.0,
                crash_slope * (new_values.get("GDP", 0.0) - crash_threshold),
            )
            voter.value = _clamp(voter.value + delta + crash, -1.0, 1.0)
        # Keep the existing voter-contribution calibration isolated from the
        # native membership correction.  The captured game values already
        # match the graph sum plus the calibrated squeeze; enabling this
        # approximation with native weights creates a larger late-turn error.
    contrib_cfg = data.calibration.get("voter_collapse", {}).get("voter_contribution", {})
    contrib_slope = float(contrib_cfg.get("slope", 2.0))
    contrib_offset = float(contrib_cfg.get("offset", 0.7))
    squeeze_nodes = data.calibration.get("voter_collapse", {}).get("squeeze", {})
    for symbol, node_name in INCOME_GROUP_NODES.items():
        # The native income-group host links are not serialized.  The
        # long-run checkpoint replay showed that estimating their aggregate
        # contribution from the loaded voter list overwhelms the measured
        # node values, so keep this compatibility calibration disabled unless
        # a future save exposes the corresponding manager field.
        contribution = 0.0
        if node_name in squeeze_nodes:
            # Equality-driven collapse (calibration.json): e.g. the middle
            # income "effective income" collapses once the SalesTax/PropertyTax
            # rises squeeze the middle class (Equality drops below the
            # threshold), saturating at the configured floor.
            squeeze_cfg = squeeze_nodes[node_name]
            equality = new_values.get("Equality", 0.0)
            squeeze = max(
                min(
                    0.0,
                    float(squeeze_cfg.get("slope", 7.7))
                    * (equality - float(squeeze_cfg.get("threshold", 0.3))),
                ),
                float(squeeze_cfg.get("saturation", -0.592)),
            )
            contribution = min(contribution, squeeze)
        graph_sum = new_values.get(node_name, 0.0)
        new_values[node_name] = _clamp(graph_sum + contribution, -1.0, 1.0)

    # The voter-type percentages are the population share in each native
    # linked-list group (the game's CalculatePercentage).  Income groups count
    # the selected native membership; the four ForceVoter ideology pairs use
    # their raw weights against the manager threshold; all other groups add
    # the VoterType ``freqval`` base to the raw coefficient.  Fresh no-order
    # captures confirm this runs identically on the first pass after load.
    n_voters = len(state.voters)
    if n_voters:
        for symbol, name in VOTER_SYMBOL_NAMES.items():
            if symbol in INCOME_GROUP_NODES:
                count = sum(
                    1
                    for voter in state.voters
                    if voter.groups.get(symbol, 0.0) > 0.0
                )
            elif symbol in POLITICAL_GROUP_SYMBOLS:
                count = sum(
                    1
                    for voter in state.voters
                    if voter.groups.get(symbol, 0.0) > membership_threshold
                )
            else:
                frequency = membership_frequencies.get(f"{name}_freq", 0.0)
                count = sum(
                    1
                    for voter in state.voters
                    if voter.groups.get(symbol, 0.0) + frequency
                    >= membership_threshold
                )
            state.voter_percentages[f"{name}_perc"] = count / n_voters


def _calculate_poll_rate(state: SimulationState) -> float:
    """Calculate the native potential-voter rate from the live voter list.

    ``SIM_PollsManager::CalculateVoteRate`` does not aggregate the serialized
    voter-type values.  It walks every live voter, asks
    ``SIM_Voter::WillVoteForPlayer`` and divides the affirmative count by the
    live-list size.  The approval field used by that predicate is the base
    approval ``(value + 1) / 2``; opposition-party voters use the native
    ``> 0.6`` cutoff and all other voters use ``> 0.5``.

    The executable performs the arithmetic in single precision.  Keeping the
    rounding here matters for voters that sit exactly on a threshold in a
    long replay.
    """
    if not state.voters:
        return 0.0
    opposition_parties = {
        party.name
        for party in state.parties.values()
        if party.party_type == 1
    }
    affirmative = 0
    for voter in state.voters:
        approval = _f32(_f32(voter.value) + 1.0)
        approval = _f32(approval * 0.5)
        threshold = 0.6 if voter.party in opposition_parties else 0.5
        affirmative += approval > threshold
    return _f32(affirmative / len(state.voters))


def _advance_state_values(
    state: SimulationState,
    graph: nx.DiGraph,
    data: SimulationData,
    source_policies: Optional[Dict[str, float]] = None,
    *,
    native_order_runtime: bool = False,
    previous_grudges: Optional[Iterable[Grudge]] = None,
    native_hidden_values: Optional[Mapping[str, float]] = None,
    native_situation_values: Optional[Mapping[str, float]] = None,
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
        # The game writes a monotonic quarter counter from the turn being
        # closed.  It is not modulo-four seasonal phase: the captured hidden
        # neuron is 0, 0, .25, .5, ..., 2.75 across the observed saves.
        new_values["_year"] = state.turn / 4.0
    if "_effectivedebt_" in new_values:
        # The effective-debt neuron is the debt-to-(DEBT_TO_GDP_MAX*GDP) ratio
        # recomputed every turn; the situation manager reads it as a source
        # (DebtCrisis = 0.2*interest^4 + effective_debt^4).  It is serialized
        # on save but must be refreshed from the live debt/GDP rather than
        # kept at the loaded value.
        new_values["_effectivedebt_"] = _effective_debt_ratio(state, data)
    if native_hidden_values is not None:
        for name, value in native_hidden_values.items():
            if name in new_values:
                new_values[name] = float(value)

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
        return _effect_is_applicable(state, effect, data)

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
        raw_value = (
            evaluate_expression(
                effect.expression,
                effect_source(effect, context),
                context=context,
            )
            + _effect_calibration_offset(data, effect)
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
    # active). A policy ring retains its old head for the first process after
    # a target change, then samples the pre-Policy::NextTurn value while the
    # policy ramps; settled links continue shifting until their leading window
    # has caught up.
    def should_shift(effect: Effect) -> bool:
        # The game writes a fresh sample into every applicable inertial ring
        # once the policy target boundary is crossed (the serialized rings
        # confirm the IncomeTax lowering shifts 0-samples for several turns
        # and the TobaccoTax raise shifts -0.8s); the ring's current value is
        # the leading-window average.
        source = effect.source
        if source in data.situations:
            return source in active_situations
        if source in state.policies:
            # The native manager suppresses only the first pass after a
            # slider move.  It then samples the ramping policy value while
            # implementation is still in progress; waiting for the target
            # would leave long-delay links frozen indefinitely (as happened
            # to StateHealthService in the captured turn-12 save).
            if state.policy_effect_history_delays.get(source, 0) > 0:
                return False
        effect_key = f"{effect.source} -> {effect.target}"
        # Data-driven frozen-ring exemptions (calibration.json): the
        # serialized StateSchools -> Education ring is frozen at the pre-game
        # level in every save while the policy sits settled, so leave those
        # specific links untouched.
        if data.calibration.get("frozen_rings", {}).get(effect_key):
            return False
        # A smaller class of rings is frozen only until its source policy has
        # received an explicit order. StateHealthService is the captured
        # example: its ring remains at .211 across no-op saves, then resumes
        # its implementation ramp after the turn-8 order.
        if data.calibration.get("frozen_until_order", {}).get(effect_key):
            return state.policy_effect_history_started.get(source, False)
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
        # The effect manager samples a policy's value before Policy::NextTurn
        # advances the slider neuron.  For a delayed implementation this is
        # the value from the preceding save (StateHealthService's
        # -0.507/-0.436/-0.366 ring samples make the ordering observable).
        # Introduced policies already have their target in pre_policy_values,
        # so the same rule covers both paths.
        sample_policy_values = pre_policy_values
        raw_value = (
            evaluate_expression(
                effect.expression,
                effect_source(
                    effect, pre_context, policy_values=sample_policy_values
                ),
                context=pre_context,
            )
            + _effect_calibration_offset(data, effect)
        )
        if effect.inertia:
            history = history_by_id.get(effect.effect_id)
            if history is not None:
                history_sample = raw_value
                if native_order_runtime and effect.source in state.policies:
                    implementation = state.policy_implementations.get(
                        effect.source, 1.0
                    )
                    if implementation < 1.0 - EPSILON:
                        # Native saves retain implementation-scaled samples
                        # for newly introduced policy rings (CarbonTax writes
                        # -0.25 at implementation .5), while settled slider
                        # rings retain their raw expression samples.
                        history_sample *= implementation
                if should_shift(effect):
                    previous_values = list(history.values)
                    history.values = [history_sample] + history.values[:-1]
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
        total += _grudge_value(state.grudges, node_name)
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

    # The native simulation calculates its hidden global neurons after the
    # effect vector. Their outgoing links are refreshed immediately, while
    # ordinary neurons continue through the current pass in their established
    # order.
    _advance_global_political_nodes(new_values, incoming_value)
    _advance_global_incoming_nodes(new_values, incoming_value)
    if native_hidden_values is not None:
        for name, value in native_hidden_values.items():
            if name in new_values:
                new_values[name] = float(value)
    context = {**new_values, **state.policies, **state.situations}
    refresh_outputs("_global_socialism", context)
    refresh_outputs("_global_liberalism", context)
    refresh_outputs("_security_", context)
    refresh_outputs("_winning_", context)

    # Situation managers are not ordinary DAG neurons, but their active
    # outputs are refreshed before the simulation neuron list consumes them.
    # Use the stored (pre-pass) situation value here; the newly calculated
    # latent value is serialized only after the list walk.
    context = {**new_values, **state.policies, **state.situations}
    for situation_name in active_situations:
        refresh_outputs(situation_name, context)

    previous_voter_frequencies = state.voter_frequencies.copy()
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

    # VoterType::freqval is the current value of the native nested neuron at
    # VoterType + 0x380.  It is part of the simulation neuron list, so its
    # base is the SIM_Neuron constructor's zero rather than the CSV
    # percentage column.  The latter is used only to seed the linked-list
    # population percentage at VoterType + 0x2c4.
    for node in data.nodes.values():
        if node.category != "VOTER_FREQ":
            continue
        context = {**new_values, **state.policies, **state.situations}
        new_values[node.name] = _clamp(
            _f32(
                _f32(incoming_value(node.name))
                + _f32(state.voter_frequency_grudges.get(node.name, 0.0))
            ),
            node.minimum,
            node.maximum,
        )
        context[node.name] = new_values[node.name]
        refresh_outputs(node.name, context)
        state.voter_frequencies[node.name] = new_values[node.name]

    # VoterType income neurons are nested manager-owned neurons rather than
    # rows in simulation.csv.  Their direct policy/simvalue inputs still run
    # through the ordinary effect vector before VoterManager::NextTurn.  The
    # manager adds per-voter host links afterward; those links are not
    # serialized, so retain that native boundary instead of guessing it.
    for name in _voter_income_names(state, graph):
        context = {**new_values, **state.policies, **state.situations}
        new_values[name] = _clamp(
            _f32(incoming_value(name)),
            -1.0,
            1.0,
        )
        context[name] = new_values[name]
        refresh_outputs(name, context)
        state.voter_incomes[name] = new_values[name]

    # Situation neurons are visited after the ordinary simulation neurons and
    # before VoterManager.  Recalculate their input links from the values
    # produced by this pass; using the pre-turn effect vector here leaves
    # StreetGangs, GeneralStrike, and Alcoholism permanently one turn behind
    # the native chain.
    situation_values: Dict[str, float] = {}
    situation_context = {**new_values, **state.policies, **state.situations}
    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    override_map = _situation_input_overrides(setup)
    for name, definition in data.situations.items():
        prerequisites_met = all(
            state.policies.get(prerequisite, 0.0) > EPSILON
            for prerequisite in definition.prerequisites
        )
        latent = definition.default if prerequisites_met else 0.0
        if prerequisites_met:
            for effect in _effective_situation_inputs(name, definition, override_map):
                refresh_effect(effect, situation_context)
                latent += new_effects.get(effect.effect_id or "", 0.0)
        situation_values[name] = _clamp(latent, 0.0, 1.0)
        if native_situation_values is not None and name in native_situation_values:
            situation_values[name] = _clamp(
                float(native_situation_values[name]), 0.0, 1.0
            )
        situation_context[name] = situation_values[name]

    # CalculateValue updates a situation's outgoing effects immediately.  Do
    # this before VoterManager consumes the voter-type neurons, while keeping
    # the activation list decided by SituationManager at turn start.
    output_context = {**new_values, **state.policies, **situation_values}
    for name in data.situations:
        refresh_outputs(name, output_context)

    # The income-group "_" nodes are voter-derived: the graph sum above is
    # the base, with only the calibrated long-run squeeze applied below.
    _advance_voters_and_income_nodes(
        state,
        data,
        new_values,
        source_policies=source_policies,
        previous_voter_frequencies=previous_voter_frequencies,
        situation_effects=new_effects,
        previous_grudges=previous_grudges,
    )

    # VoterManager owns these nested neurons and updates them after the main
    # simulation-neuron pass.  Publish the post-manager values into the
    # serialized value map as well; otherwise the next turn's effect context
    # would keep reading the previous VoterType/frequency/income values even
    # though the dedicated runtime fields had advanced.
    for name, value in state.voter_values.items():
        new_values[name] = value
    for name, value in state.voter_percentages.items():
        new_values[name] = value
    for name, value in state.voter_frequencies.items():
        new_values[name] = value
    for name, value in state.voter_incomes.items():
        new_values[name] = value

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


def _remove_resigned_ministers(
    experience: Dict[str, float],
    competence: Dict[str, float],
    effectiveness: Dict[str, float],
    suitability: Dict[str, float],
    loyalty: Dict[str, float],
    volatility: Dict[str, float],
    values: Dict[str, float],
    sympathies: Dict[str, List[str]],
    data: SimulationData,
) -> set[str]:
    """Drop portfolios whose loyalty crossed the native resignation cutoff.

    The native manager removes a minister after the loyalty update, before
    the political-capital cap and finance scalars are serialized.  The random
    resignation chance is not stored in XML, so the explicit resignation mode
    treats a minister below the configured threshold as resigned.  This
    reproduces the turn-15 TAX vacancy in the quiet UK native capture while
    leaving ordinary loyalty-only replays roster-stable.
    """
    threshold = data.sim_config.get("MINISTER_RESIGN_THRESHHOLD", 0.15)
    resigned = {dept for dept, value in loyalty.items() if value < threshold}
    for department in resigned:
        experience.pop(department, None)
        competence.pop(department, None)
        effectiveness.pop(department, None)
        suitability.pop(department, None)
        loyalty.pop(department, None)
        volatility.pop(department, None)
        values.pop(department, None)
        sympathies.pop(department, None)
    return resigned


def apply_native_manager_roster(
    state: SimulationState,
    departments: Iterable[str],
) -> SimulationState:
    """Align active minister portfolios with a serialized native checkpoint.

    Democracy 3 does not serialize the random resignation cursor, so a
    deterministic loyalty replay cannot infer the exact native roster.  The
    long-chain audit can, however, use the roster written by the native save
    as an explicit replay input.  This helper only removes portfolios; it
    never invents a replacement minister for a department absent from the
    current state.
    """
    active = {str(department) for department in departments}

    def keep_active(values: Dict[str, object]) -> Dict[str, object]:
        return {name: value for name, value in values.items() if name in active}

    return replace(
        state,
        ministerial_effectiveness=keep_active(state.ministerial_effectiveness),
        ministerial_competence=keep_active(state.ministerial_competence),
        ministerial_experience=keep_active(state.ministerial_experience),
        ministerial_suitability=keep_active(state.ministerial_suitability),
        ministerial_loyalty=keep_active(state.ministerial_loyalty),
        ministerial_volatility=keep_active(state.ministerial_volatility),
        ministerial_value=keep_active(state.ministerial_value),
        ministerial_sympathies=keep_active(state.ministerial_sympathies),
    )


def apply_native_sim_values(
    state: SimulationState,
    values: Mapping[str, float],
) -> SimulationState:
    """Restore serialized ``<simvalues>`` at an audit checkpoint.

    A native save does not include every live neuron/manager cursor needed to
    deterministically continue the executable.  This explicit checkpoint
    bridge keeps long native-vs-simulator replays aligned without changing
    ordinary simulator runs; callers that need model residuals can retain the
    state immediately before applying this overlay.
    """
    restored_values = state.values.copy()
    restored_values.update(
        {str(name): float(value) for name, value in values.items()}
    )
    return replace(state, values=restored_values)


def apply_native_voter_runtime(
    state: SimulationState,
    *,
    voter_values: Dict[str, float],
    voter_percentages: Dict[str, float],
    voter_frequencies: Dict[str, float],
    voter_incomes: Dict[str, float],
    voter_frequency_grudges: Dict[str, float],
    voters: Iterable[Voter],
    parties: Dict[str, PartyState],
    poll_rate: float = 0.0,
    peak_poll_rate: float = 0.0,
    poll_history: Iterable[float] = (),
    income_nodes: Optional[Mapping[str, float]] = None,
) -> SimulationState:
    """Restore serialized voter-manager state for a checkpoint replay.

    The native voter manager keeps live linked lists and host pointers that
    cannot be reconstructed from a save.  Its serialized voter aggregates,
    party rings, individual vote enums, and poll ring are safe explicit
    inputs for an offline parity audit; the default simulator never calls
    this bridge.
    """
    values = state.values.copy()
    for mapping in (
        voter_values,
        voter_percentages,
        voter_frequencies,
        voter_incomes,
    ):
        values.update({str(name): float(value) for name, value in mapping.items()})
    if income_nodes is not None:
        for name in INCOME_GROUP_NODES.values():
            if name in income_nodes:
                values[name] = float(income_nodes[name])
    return replace(
        state,
        values=values,
        voter_values={str(name): float(value) for name, value in voter_values.items()},
        voter_percentages={
            str(name): float(value) for name, value in voter_percentages.items()
        },
        voter_frequencies={
            str(name): float(value) for name, value in voter_frequencies.items()
        },
        voter_incomes={
            str(name): float(value) for name, value in voter_incomes.items()
        },
        voter_frequency_grudges={
            str(name): float(value)
            for name, value in voter_frequency_grudges.items()
        },
        voters=[_copy_voter(voter) for voter in voters],
        parties={name: _copy_party(party) for name, party in parties.items()},
        poll_rate=float(poll_rate),
        peak_poll_rate=float(peak_poll_rate),
        poll_history=[float(value) for value in poll_history],
    )


def apply_native_effect_histories(
    state: SimulationState,
    histories: Iterable[EffectHistory],
    graph: Optional[nx.DiGraph] = None,
    data: Optional[SimulationData] = None,
) -> SimulationState:
    """Restore serialized inertial rings for a checkpoint replay."""
    history_copy = [
        EffectHistory(
            history.source,
            history.target,
            list(history.values),
            history.effect_id,
        )
        for history in histories
    ]
    restored = replace(
        state,
        effect_histories=history_copy,
    )
    if graph is not None:
        restored.effects = _initialize_effect_memory(
            restored, graph, data=data, effect_histories=history_copy
        )
    return restored


def apply_native_policy_runtime(
    state: SimulationState,
    *,
    policy_implementations: Mapping[str, float],
    policy_active: Mapping[str, bool],
    policy_cost_multipliers: Mapping[str, float],
    policy_income_multipliers: Mapping[str, float],
    policy_cost_scalars: Mapping[str, float],
    policy_income_scalars: Mapping[str, float],
    effect_throttles: Optional[Mapping[str, float]] = None,
) -> SimulationState:
    """Restore serialized policy-manager runtime fields for a parity audit."""
    return replace(
        state,
        policy_implementations={
            str(name): float(value)
            for name, value in policy_implementations.items()
        },
        policy_active={str(name): bool(value) for name, value in policy_active.items()},
        policy_cost_multipliers={
            str(name): float(value)
            for name, value in policy_cost_multipliers.items()
        },
        policy_income_multipliers={
            str(name): float(value)
            for name, value in policy_income_multipliers.items()
        },
        policy_cost_scalars={
            str(name): float(value) for name, value in policy_cost_scalars.items()
        },
        policy_income_scalars={
            str(name): float(value) for name, value in policy_income_scalars.items()
        },
        effect_throttles=(
            {
                str(name): float(value) for name, value in effect_throttles.items()
            }
            if effect_throttles is not None
            else state.effect_throttles.copy()
        ),
    )


def apply_native_finance_runtime(
    state: SimulationState,
    *,
    total_income: float,
    total_expenditure: float,
    debt: float,
    interest_rate: float,
    credit_rating: int,
    turns_since_credit: int,
    policy_costs: Optional[Mapping[str, float]] = None,
    policy_incomes: Optional[Mapping[str, float]] = None,
    policy_cost_histories: Optional[Mapping[str, Sequence[float]]] = None,
    policy_income_histories: Optional[Mapping[str, Sequence[float]]] = None,
) -> SimulationState:
    """Restore serialized finance-manager state for a checkpoint replay."""
    return replace(
        state,
        total_income=float(total_income),
        total_expenditure=float(total_expenditure),
        debt=float(debt),
        interest_rate=float(interest_rate),
        credit_rating=int(credit_rating),
        turns_since_credit=int(turns_since_credit),
        policy_costs=(
            {str(name): float(value) for name, value in policy_costs.items()}
            if policy_costs is not None
            else state.policy_costs.copy()
        ),
        policy_incomes=(
            {str(name): float(value) for name, value in policy_incomes.items()}
            if policy_incomes is not None
            else state.policy_incomes.copy()
        ),
        policy_cost_histories=(
            {
                str(name): [float(value) for value in values]
                for name, values in policy_cost_histories.items()
            }
            if policy_cost_histories is not None
            else {
                name: list(values)
                for name, values in state.policy_cost_histories.items()
            }
        ),
        policy_income_histories=(
            {
                str(name): [float(value) for value in values]
                for name, values in policy_income_histories.items()
            }
            if policy_income_histories is not None
            else {
                name: list(values)
                for name, values in state.policy_income_histories.items()
            }
        ),
    )


def _minister_fallback_competence(data: SimulationData) -> float:
    """Return the competence used by native finance without a minister."""
    return float(
        data.calibration.get("minister_fallback", {}).get("competence", 0.25)
    )


def _capital_income(
    state: SimulationState, data: SimulationData
) -> float:
    """Per-turn political-capital income from the active ministers.

    Mirrors ``SIM_PoliticalCapital::CalcNewPoints``: every minister with a
    portfolio contributes ``max(0, POLITICAL_CAPITAL_PER_MINISTER *
    (loyalty - threshold))``.  The threshold is one MINISTER_VOTER_BOOST
    below the config's MINISTER_RESIGN_THRESHHOLD (0.10 vs 0.15); the total
    is truncated like the game's ``cvttss2si``.
    """
    per_minister = data.sim_config.get("POLITICAL_CAPITAL_PER_MINISTER", 6.0)
    threshold = data.sim_config.get("MINISTER_RESIGN_THRESHHOLD", 0.15) - data.sim_config.get(
        "MINISTER_VOTER_BOOST", 0.05
    )
    total = 0.0
    for loyalty in state.ministerial_loyalty.values():
        total += max(0.0, per_minister * (loyalty - threshold))
    return float(int(total))


def _interest_rate(
    credit_rating: int,
    data: SimulationData,
    global_interest_rate: float = 0.5,
) -> float:
    """Interest rate charged on the national debt.

    Mirrors ``SIM_FinanceManager::ApplyInterestRateCalculations``:
    ``rate = INTEREST_RATE_MIN + (INTEREST_RATE_MAX - INTEREST_RATE_MIN) *
    (global_interest_rate - 0.5 + min((credit_rating / 9) ** 2, 1.0))``.
    The game's global-interest neuron is centred at 0.5 in the shipped
    saves, which is why the omitted term was invisible in the original
    deterministic baseline.
    """
    min_rate = data.sim_config.get("INTEREST_RATE_MIN", 0.017)
    max_rate = data.sim_config.get("INTEREST_RATE_MAX", 0.15)
    factor = min((credit_rating / 9.0) ** 2, 1.0)
    return min_rate + (max_rate - min_rate) * (
        factor + global_interest_rate - 0.5
    )


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
    policy_levels: Optional[Dict[str, float]] = None,
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
    levels = policy_levels if policy_levels is not None else state.policies
    for policy in data.policies.values():
        name = policy.name
        level = levels.get(name, state.policies.get(name, 0.0))
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
    rate = state.interest_rate or _interest_rate(
        state.credit_rating,
        data,
        state.values.get("_global_interest_rates_", 0.5),
    )
    return _finance_totals(
        state,
        data,
        income_multipliers=state.policy_income_multipliers,
        cost_multipliers=state.policy_cost_multipliers,
        income_scalars=_scalars_by_department(state.policy_income_scalars, data),
        cost_scalars=_scalars_by_department(state.policy_cost_scalars, data),
        interest_rate=rate,
    )


def _recompute_native_order_finance(
    state: SimulationState,
    data: SimulationData,
    policy_levels: Dict[str, float],
    introduced_policies: Iterable[str],
) -> SimulationState:
    """Recompute the native order-phase budget preview.

    ``SIM_Policy::SetSlider`` updates the visible value immediately, but the
    finance manager consumes the previous policy-history sample for the
    upcoming debt roll.  Existing policies retain their serialized finance
    multipliers; a policy introduced in this order batch has no stored
    multiplier yet, so its multiplier is evaluated from the current node
    snapshot.  The implementation scalar is still the pre-turn minister
    scalar.
    """

    context = {
        **state.values,
        **state.policies,
        **state.situations,
        **state.voter_values,
        **state.voter_percentages,
        **state.voter_frequencies,
    }
    income_multipliers = state.policy_income_multipliers.copy()
    cost_multipliers = state.policy_cost_multipliers.copy()
    for name in introduced_policies:
        policy = data.policies.get(name)
        if policy is None:
            continue
        level = policy_levels.get(name, state.policies.get(name, 0.0))
        if policy.max_income > 0:
            income_multipliers[name] = _live_multiplier(
                policy.income_multipliers, level, context
            )
        if policy.max_cost > 0:
            cost_multipliers[name] = _live_multiplier(
                policy.cost_multipliers, level, context
            )

    rate = state.interest_rate or _interest_rate(
        state.credit_rating,
        data,
        state.values.get("_global_interest_rates_", 0.5),
    )
    return _finance_totals(
        state,
        data,
        income_multipliers=income_multipliers,
        cost_multipliers=cost_multipliers,
        income_scalars=_scalars_by_department(state.policy_income_scalars, data),
        cost_scalars=_scalars_by_department(state.policy_cost_scalars, data),
        interest_rate=rate,
        policy_levels=policy_levels,
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

    fallback_competence = _minister_fallback_competence(data)
    fallback_income_scalar = _f32(0.875 + 0.25 * fallback_competence)
    income_scalars = {
        policy.department: fallback_income_scalar
        for policy in data.policies.values()
        if policy.department
    }
    income_scalars.update(
        {
            dept: _f32(0.875 + 0.25 * competence)
            for dept, competence in state.ministerial_competence.items()
        }
    )
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
    rate = state.interest_rate or _interest_rate(
        state.credit_rating,
        data,
        state.values.get("_global_interest_rates_", 0.5),
    )
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


def _election_term_length(state: SimulationState, data: SimulationData) -> int:
    """Return the mission's serialized election interval."""
    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    return max(1, setup.term_length)


def _advance_election_countdown(
    state: SimulationState, data: SimulationData
) -> int:
    """Mirror ``SIM_ElectionManager::NextTurn`` in the headless path."""
    if state.election_turns_until > 0:
        return state.election_turns_until - 1
    return _election_term_length(state, data) - 1


def process_end_of_turn(
    state: SimulationState,
    graph: nx.DiGraph,
    data: Optional[SimulationData] = None,
    config: Optional[SimulationConfig] = None,
    *,
    native_resigned_departments: Optional[Iterable[str]] = None,
    native_active_situations: Optional[Iterable[str]] = None,
    native_grudges: Optional[Iterable[Grudge]] = None,
    native_hidden_values: Optional[Mapping[str, float]] = None,
    native_hidden_histories: Optional[Mapping[str, Sequence[float]]] = None,
    native_situation_values: Optional[Mapping[str, float]] = None,
) -> SimulationState:
    data = data or load_simulation_data()
    election_turns_until = _advance_election_countdown(state, data)
    source_policies = state.policies.copy()
    policy_cost_histories, policy_income_histories = _advance_policy_finance_histories(
        state, data
    )
    # SIM_Simulation::NextTurn calls MinisterManager before PolicyManager and
    # before the neural-effect vector.  Effect scales and implementation
    # increments therefore use this turn's minister effectiveness, not the
    # value stored at the beginning of the turn.
    experience, competence, effectiveness = _advance_ministers(state, data)
    minister_runtime_state = replace(
        state,
        ministerial_experience=experience,
        ministerial_competence=competence,
        ministerial_effectiveness=effectiveness,
        voter_values=state.voter_values.copy(),
        voter_percentages=state.voter_percentages.copy(),
        voter_frequencies=state.voter_frequencies.copy(),
        voter_incomes=state.voter_incomes.copy(),
        voter_frequency_grudges=state.voter_frequency_grudges.copy(),
        grudges=[_copy_grudge(grudge) for grudge in state.grudges],
        voters=[_copy_voter(voter) for voter in state.voters],
        parties={name: _copy_party(party) for name, party in state.parties.items()},
    )
    runtime_state = _advance_policy_runtime(minister_runtime_state, data)
    # FinanceManager::NextTurn rolls debt before the effect vector and hidden
    # neuron pass.  Feed that post-roll balance into the simulation so
    # _effectivedebt_ and DebtCrisis see the native value.
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
    runtime_state = replace(
        runtime_state,
        debt=debt,
        credit_rating=credit_rating,
        turns_since_credit=turns_since_credit,
        interest_rate=_interest_rate(
            credit_rating,
            data,
            runtime_state.values.get("_global_interest_rates_", 0.5),
        ),
    )
    # ``election_result`` describes the result-screen event at the boundary,
    # not a permanent objective state.  Keep it observable on the resolved
    # state, then clear it on the first ordinary turn of the new term so a
    # multi-term oracle scores the next election from its live forecast rather
    # than reusing the previous term's vote margin.
    election_result = (
        None if runtime_state.election_result in {"win", "loss"} else runtime_state.election_result
    )
    # The native event/dilemma/pressure managers run before the neural-effect
    # vector.  Their CreateGrudge calls must therefore feed this same turn's
    # node calculation, not the already-completed snapshot.  A supplied
    # checkpoint grudge set is an explicit parity-audit input: savegame values
    # are already post-decay, so they bypass the normal advance below.
    prior_grudges = [
        _copy_grudge(grudge) for grudge in runtime_state.grudges
    ]
    runtime_state = _run_stochastic_systems(runtime_state, graph, data, config)
    if native_grudges is not None:
        runtime_state.grudges = _restore_native_grudge_inputs(native_grudges)
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
        native_order_runtime=(config.native_order_runtime if config else False),
        previous_grudges=prior_grudges,
        native_hidden_values=native_hidden_values,
        native_situation_values=native_situation_values,
    )
    # Polls are refreshed by PollsManager after the voter manager has updated
    # the live voter list.  The loaded poll value is only the previous save's
    # observation; carrying it forward creates a steadily growing long-run
    # parity error.
    poll_rate = _calculate_poll_rate(runtime_state)
    peak_poll_rate = max(runtime_state.peak_poll_rate, poll_rate)
    poll_history = [poll_rate, *runtime_state.poll_history[:19]]
    # Ministers gain experience each turn (drifting the income/cost scalars)
    # and -- when the loyalty subsystem is enabled -- their satisfaction and
    # loyalty advance, which drives the per-turn political-capital income.
    ministerial_value = runtime_state.ministerial_value.copy()
    ministerial_loyalty = runtime_state.ministerial_loyalty.copy()
    ministerial_suitability = runtime_state.ministerial_suitability.copy()
    ministerial_volatility = runtime_state.ministerial_volatility.copy()
    ministerial_sympathies = {
        name: list(groups)
        for name, groups in runtime_state.ministerial_sympathies.items()
    }
    if config is not None and config.minister_loyalty:
        # Loyalty thresholds see the newly advanced policy state, but the
        # experience increment itself is applied only once above.
        loyalty_state = replace(
            runtime_state,
            ministerial_experience=state.ministerial_experience,
        )
        ministerial_value, ministerial_loyalty = _advance_minister_loyalty(
            loyalty_state, data
        )
        resigned: set[str] = set()
        if config.minister_resignations:
            resigned = _remove_resigned_ministers(
                experience,
                competence,
                effectiveness,
                ministerial_suitability,
                ministerial_loyalty,
                ministerial_volatility,
                ministerial_value,
                ministerial_sympathies,
                data,
            )
            if resigned:
                loyalty_change = data.sim_config.get(
                    "MINISTER_RESIGNS_LOYALTY_CHANGE", -0.06
                )
                for department in ministerial_loyalty:
                    ministerial_loyalty[department] = _clamp(
                        ministerial_loyalty[department] + loyalty_change,
                        0.0,
                        1.0,
                    )
        native_resigned = set(native_resigned_departments or ())
        if native_resigned:
            loyalty_change = data.sim_config.get(
                "MINISTER_RESIGNS_LOYALTY_CHANGE", -0.06
            )
            for department in ministerial_loyalty:
                if department not in native_resigned:
                    ministerial_loyalty[department] = _clamp(
                        ministerial_loyalty[department] + loyalty_change,
                        0.0,
                        1.0,
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
    if native_active_situations is not None:
        active_situations = [str(name) for name in native_active_situations]
    if native_hidden_histories is not None:
        hidden_histories = {
            name: [float(value) for value in values]
            for name, values in native_hidden_histories.items()
        }
    else:
        hidden_histories = {}
        for name, values in runtime_state.hidden_histories.items():
            sample = new_values.get(name)
            if sample is None:
                hidden_histories[name] = list(values)
                continue
            # The native year neuron grows monotonically, but its serialized
            # history is a normalized [0, 1] sample ring rather than the
            # public quarter counter itself.
            if name == "_year":
                sample = _clamp(sample, 0.0, 1.0)
            hidden_histories[name] = [sample, *values[:32]]
    new_state = SimulationState(
        country=runtime_state.country,
        turn=runtime_state.turn + 1,
        values=new_values,
        policies=runtime_state.policies.copy(),
        political_capital=new_capital,
        effects=new_effects,
        political_capital_income=(
            capital_income
            if config is not None and config.minister_loyalty
            else runtime_state.political_capital_income
        ),
        effect_histories=effect_histories,
        situations=situation_values,
        active_situations=active_situations,
        response_factors=runtime_state.response_factors.copy(),
        policy_cost_histories=policy_cost_histories,
        policy_income_histories=policy_income_histories,
        policy_finance_levels=runtime_state.policy_finance_levels.copy(),
        global_economy_position=global_position,
        hidden_histories=hidden_histories,
        # Pre-game covariate rings ride along untouched; their end-turn
        # marker stays behind the live turn, which is how forecasters tell
        # them apart from live observations.
        value_histories={
            name: list(values) for name, values in state.value_histories.items()
        },
        value_histories_turn=state.value_histories_turn,
        voter_values=runtime_state.voter_values.copy(),
        voter_percentages=runtime_state.voter_percentages.copy(),
        voter_frequencies=runtime_state.voter_frequencies.copy(),
        voter_incomes=runtime_state.voter_incomes.copy(),
        voter_frequency_grudges=runtime_state.voter_frequency_grudges.copy(),
        grudges=(
            [_copy_grudge(grudge) for grudge in native_grudges]
            if native_grudges is not None
            else _advance_grudges(runtime_state.grudges)
        ),
        voters=[_copy_voter(v) for v in runtime_state.voters],
        parties={
            name: _copy_party(party)
            for name, party in runtime_state.parties.items()
        },
        election_turns_until=election_turns_until,
        election_current_term=runtime_state.election_current_term,
        election_result=election_result,
        last_election_winner=runtime_state.last_election_winner,
        election_player_votes=runtime_state.election_player_votes,
        election_opposition_votes=runtime_state.election_opposition_votes,
        election_absent_votes=runtime_state.election_absent_votes,
        poll_rate=poll_rate,
        peak_poll_rate=peak_poll_rate,
        poll_history=poll_history,
        policy_implementations=runtime_state.policy_implementations.copy(),
        policy_active=runtime_state.policy_active.copy(),
        policy_cost_multipliers=runtime_state.policy_cost_multipliers.copy(),
        policy_income_multipliers=runtime_state.policy_income_multipliers.copy(),
        policy_cost_scalars=runtime_state.policy_cost_scalars.copy(),
        policy_income_scalars=runtime_state.policy_income_scalars.copy(),
        effect_throttles=runtime_state.effect_throttles.copy(),
        policy_desired_throttles=runtime_state.policy_desired_throttles.copy(),
        policy_effect_history_delays={
            name: remaining - 1
            for name, remaining in runtime_state.policy_effect_history_delays.items()
            if remaining > 1
        },
        policy_effect_history_started=runtime_state.policy_effect_history_started.copy(),
        ministerial_effectiveness=effectiveness,
        ministerial_competence=competence,
        ministerial_experience=experience,
        ministerial_suitability=ministerial_suitability,
        ministerial_loyalty=ministerial_loyalty,
        ministerial_volatility=ministerial_volatility,
        ministerial_value=ministerial_value,
        ministerial_sympathies=ministerial_sympathies,
        debt=debt,
        credit_rating=credit_rating,
        turns_since_credit=turns_since_credit,
        interest_rate=_interest_rate(
            credit_rating,
            data,
            runtime_state.values.get("_global_interest_rates_", 0.5),
        ),
        event_log=list(runtime_state.event_log),
        fired_plots=list(runtime_state.fired_plots),
        group_threats=dict(runtime_state.group_threats),
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
    if config is not None and config.native_order_runtime:
        # The current policy-neuron values become the previous-history sample
        # for the next native order phase.  The displayed totals above use
        # the live current values; only the following debt preview is delayed.
        new_state.policy_finance_levels = new_state.policies.copy()
    return new_state


def resolve_election(
    state: SimulationState,
    data: Optional[SimulationData] = None,
) -> SimulationState:
    """Count an election whose native countdown has reached zero.

    The headless native capture path stops at the zero countdown and leaves
    ``currentterm`` and ``lastvote`` unchanged.  A real result is a separate
    election-manager operation, so callers invoke this function explicitly.

    Native ``SIM_Voter::CastVote`` first samples a turnout chance and only
    then chooses a candidate from the voter's calculated approval.  The old
    simulator treated every party member as a certain vote and every
    unaffiliated voter as an abstention, which made an election result depend
    on party membership rather than the game's turnout model.  This resolver
    uses the same expected turnout calculation as :func:`forecast_election`
    and rounds the expected population into deterministic vote enums.  The
    rounding makes simulator searches reproducible; the oracle objective uses
    the unrounded forecast so it does not optimize a rounding accident.

    Native vote enums are retained as ``0`` (player), ``1`` (opposition), and
    ``2`` (absent), matching ``SIM_Voter::CastVote``.
    """
    data = data or load_simulation_data()
    if state.election_turns_until != 0:
        raise ValueError("election is not ready to resolve")

    options = {
        option.strip().upper()
        for option in data_loader.load_country_setup(
            data.gamedata_root, state.country
        ).options
        if option.strip()
    }
    compulsory_voting = "COMPULSORY_VOTING" in options
    expectations = _election_voter_expectations(
        state.voters, state.parties, compulsory_voting=compulsory_voting
    )
    player_votes, opposition_votes, absent_votes = _rounded_election_counts(
        expectations
    )
    assignments = _deterministic_election_assignments(
        expectations,
        player_votes=player_votes,
        opposition_votes=opposition_votes,
    )
    counted_voters = [
        replace(voter, last_vote=vote)
        for voter, vote in zip(state.voters, assignments)
    ]

    winner = "player" if player_votes > opposition_votes else "opposition"
    loyalties = state.ministerial_loyalty.copy()
    if winner == "player":
        boost = data.sim_config.get("MINISTER_ELECTIONWIN_BOOST", 0.12)
        loyalties = {
            department: _clamp(value + boost, 0.0, 1.0)
            for department, value in loyalties.items()
        }
    return replace(
        state,
        voters=counted_voters,
        election_turns_until=_election_term_length(state, data),
        election_current_term=state.election_current_term + 1,
        election_result="win" if winner == "player" else "loss",
        last_election_winner=winner,
        election_player_votes=player_votes,
        election_opposition_votes=opposition_votes,
        election_absent_votes=absent_votes,
        ministerial_loyalty=loyalties,
    )


def forecast_election(state: SimulationState) -> ElectionForecast:
    """Return the expected native-style election result for ``state``.

    The executable stores the voter's base approval and a voting-technology
    input, but its live party-manager popularity and several perception
    modifiers are not serialized.  The forecast therefore uses the
    high-confidence native pieces directly and uses current live party share
    as the serialized-state proxy for the missing popularity field.  It is an
    expected-value baseline, not a claim that a particular random election
    draw will equal these fractional counts.
    """

    data = load_simulation_data()
    options = {
        option.strip().upper()
        for option in data_loader.load_country_setup(
            data.gamedata_root, state.country
        ).options
        if option.strip()
    }
    return _forecast_election_voters(
        state.voters,
        state.parties,
        compulsory_voting="COMPULSORY_VOTING" in options,
    )


def forecast_election_from_voters(
    voters: Sequence[Voter],
    parties: Mapping[str, PartyState],
    *,
    compulsory_voting: bool = False,
) -> ElectionForecast:
    """Forecast an election from voter/party records without a full state.

    This is used by the native GameDrive oracle, whose objective receives a
    parsed XML save while its transition state is kept separately.
    """

    return _forecast_election_voters(
        voters, parties, compulsory_voting=compulsory_voting
    )


def _forecast_election_voters(
    voters: Sequence[Voter],
    parties: Mapping[str, PartyState],
    *,
    compulsory_voting: bool = False,
) -> ElectionForecast:
    expectations = _election_voter_expectations(
        voters, parties, compulsory_voting=compulsory_voting
    )
    return ElectionForecast(
        expected_player_votes=_f32(
            sum(player for player, _, _ in expectations)
        ),
        expected_opposition_votes=_f32(
            sum(opposition for _, opposition, _ in expectations)
        ),
        expected_absent_votes=_f32(sum(absent for _, _, absent in expectations)),
    )


def _election_voter_expectations(
    voters: Sequence[Voter],
    parties: Mapping[str, PartyState],
    *,
    compulsory_voting: bool = False,
) -> list[tuple[float, float, float]]:
    """Return ``(player, opposition, absent)`` expectations per voter."""

    if not voters:
        return []
    recognized_parties = {
        name: party for name, party in parties.items() if name
    }
    party_counts = {
        name: sum(1 for voter in voters if voter.party == name)
        for name in recognized_parties
    }
    population = float(len(voters))
    party_shares = {
        name: count / population for name, count in party_counts.items()
    }
    party_types = {
        name: party.party_type for name, party in recognized_parties.items()
    }
    expectations: list[tuple[float, float, float]] = []
    for voter in voters:
        party = recognized_parties.get(voter.party)
        approval = _clamp(_f32((_f32(voter.value) + 1.0) * 0.5), 0.0, 1.0)
        preference = _election_preference(voter, party, approval)
        turnout = _election_turnout(
            voter,
            party,
            approval,
            party_shares,
            party_types,
            compulsory_voting=compulsory_voting,
        )
        if preference == "player":
            expectations.append((turnout, 0.0, 1.0 - turnout))
        else:
            expectations.append((0.0, turnout, 1.0 - turnout))
    return expectations


def _election_preference(
    voter: Voter,
    party: PartyState | None,
    approval: float,
) -> str:
    """Approximate the native post-turnout candidate choice.

    ``SIM_Voter::CastVote`` calls ``CalculateApproval`` and then applies the
    same neutral approval threshold for both recognized parties and
    unaffiliated voters.  Party membership changes ``CalculateVoteChance``
    (turnout), but it does not force an opposition-party member to cast an
    opposition ballot.  ``party`` and ``voter`` stay in the signature to make
    that distinction explicit and to leave room for future serialized
    approval modifiers.
    """

    return "player" if approval >= 0.5 else "opposition"


def _election_turnout(
    voter: Voter,
    party: PartyState | None,
    approval: float,
    party_shares: Mapping[str, float],
    party_types: Mapping[str, int],
    *,
    compulsory_voting: bool = False,
) -> float:
    """Calculate the expected turnout probability from native inputs.

    Static inspection of ``SIM_Voter::CalculateVoteChance`` shows three
    important branches: recognized party members are assigned chance one;
    unaffiliated voters combine twice their distance from neutral approval,
    voting technology, and party popularity; and the result is scaled by
    ``0.2`` and clamped to ``[0, 1]``.  Party popularity is not serialized, so
    the current party share is the deliberately explicit approximation here.
    Strong serialized sympathy is treated as a live party commitment for
    hand-built states that have not run the membership manager yet.

    With the ``COMPULSORY_VOTING`` mission option (Australia), every eligible
    voter is required to cast a ballot, so the turnout probability is one.
    """

    if compulsory_voting:
        return 1.0
    if party is not None:
        return 1.0
    if max(voter.player_sympathy, voter.opposition_sympathy) >= 0.5:
        return 1.0
    player_share = max(
        (
            share
            for name, share in party_shares.items()
            if name and party_types.get(name) == 0
        ),
        default=0.0,
    )
    opposition_share = max(
        (
            share
            for name, share in party_shares.items()
            if name and party_types.get(name) == 1
        ),
        default=0.0,
    )
    popularity = opposition_share if approval < 0.5 else player_share
    distance = 2.0 * abs(approval - 0.5)
    technology = _clamp(voter.voting_tech, 0.0, 1.0)
    return _clamp(
        _f32(0.2 * (distance + technology + popularity)),
        0.0,
        1.0,
    )


def _rounded_election_counts(
    expectations: Sequence[tuple[float, float, float]],
) -> tuple[int, int, int]:
    """Round three expected counts while preserving the population total."""

    totals = [
        sum(values[index] for values in expectations)
        for index in range(3)
    ]
    floors = [math.floor(value) for value in totals]
    remaining = len(expectations) - sum(floors)
    fractions = [value - floor for value, floor in zip(totals, floors)]
    for index in sorted(range(3), key=lambda item: (-fractions[item], item))[
        :remaining
    ]:
        floors[index] += 1
    return floors[0], floors[1], floors[2]


def _deterministic_election_assignments(
    expectations: Sequence[tuple[float, float, float]],
    *,
    player_votes: int,
    opposition_votes: int,
) -> list[int]:
    """Choose modal voter outcomes that realize rounded expected totals."""

    player_candidates = sorted(
        (
            index
            for index, (player, _, _) in enumerate(expectations)
            if player > 0.0
        ),
        key=lambda index: (-expectations[index][0], index),
    )
    opposition_candidates = sorted(
        (
            index
            for index, (_, opposition, _) in enumerate(expectations)
            if opposition > 0.0
        ),
        key=lambda index: (-expectations[index][1], index),
    )
    assignments = [2] * len(expectations)
    for index in player_candidates[:player_votes]:
        assignments[index] = 0
    for index in opposition_candidates[:opposition_votes]:
        assignments[index] = 1
    return assignments


def resolve_election_if_ready(
    state: SimulationState,
    data: Optional[SimulationData] = None,
) -> SimulationState:
    """Resolve a pending election, leaving ordinary turns unchanged.

    The native turn worker serializes the boundary with a zero countdown and
    does not show the result screen in headless mode.  Oracle agents use this
    bridge immediately after each observed turn so a search cannot continue
    through an unresolved election.
    """
    if state.election_turns_until != 0:
        return state
    return resolve_election(state, data=data)


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
    *,
    native_order_runtime: bool = False,
) -> SimulationState:
    """Apply policy orders without advancing the simulation.

    ``native_order_runtime`` models the direct ``SIM_Policy::SetSlider`` path
    used by the native gamedrive injector: the saved current policy value and
    output throttle jump to the requested target before the next turn. The
    default keeps the public interactive model's current-versus-target split,
    where implementation advances during ``process_end_of_turn``.
    """
    data = data or load_simulation_data()
    policies = state.policies.copy()
    finance_policy_values = (
        state.policy_finance_levels.copy()
        if state.policy_finance_levels
        else state.policies.copy()
    )
    introduced_policies: set[str] = set()
    policy_desired_throttles = state.policy_desired_throttles.copy()
    target_levels = policy_desired_throttles.copy()
    policy_active = state.policy_active.copy()
    policy_implementations = state.policy_implementations.copy()
    effect_throttles = state.effect_throttles.copy()
    policy_effect_history_delays = state.policy_effect_history_delays.copy()
    policy_effect_history_started = state.policy_effect_history_started.copy()
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
        policy_effect_history_started[policy.name] = True
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
            # A newly introduced policy seeds its native policy history with
            # the midpoint sample; the first finance pass consumes that
            # sample while the visible slider already shows the target.
            finance_policy_values[policy.name] = (
                _default_slider_level(slider)
                if native_order_runtime
                else new_level
            )
            introduced_policies.add(policy.name)
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
            if native_order_runtime:
                # The injector calls SIM_Policy::SetSlider directly. Native
                # XML then serializes the requested slider as both the current
                # policy value and output throttle, while inertial rings keep
                # their own one-pass order delay below.
                policies[policy.name] = new_level
                effect_throttles[policy.name] = new_level
            # Native policy effects retain the old ring for the first turn
            # after a raise/lower order, then follow the implementation ramp.
            policy_effect_history_delays[policy.name] = 1
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
        policy_costs=state.policy_costs.copy(),
        policy_incomes=state.policy_incomes.copy(),
        total_expenditure=state.total_expenditure,
        total_income=state.total_income,
        policy_finance_levels=(
            finance_policy_values
            if native_order_runtime
            else policies.copy()
        ),
        policy_cost_histories={
            name: list(values)
            for name, values in state.policy_cost_histories.items()
        },
        policy_income_histories={
            name: list(values)
            for name, values in state.policy_income_histories.items()
        },
        global_economy_position=state.global_economy_position,
        hidden_histories={
            name: list(values)
            for name, values in state.hidden_histories.items()
        },
        voter_values=state.voter_values.copy(),
        voter_percentages=state.voter_percentages.copy(),
        voter_frequencies=state.voter_frequencies.copy(),
        voter_incomes=state.voter_incomes.copy(),
        voter_frequency_grudges=state.voter_frequency_grudges.copy(),
        grudges=[_copy_grudge(grudge) for grudge in state.grudges],
        voters=[_copy_voter(v) for v in state.voters],
        parties={
            name: _copy_party(party) for name, party in state.parties.items()
        },
        election_turns_until=state.election_turns_until,
        election_current_term=state.election_current_term,
        election_result=state.election_result,
        last_election_winner=state.last_election_winner,
        election_player_votes=state.election_player_votes,
        election_opposition_votes=state.election_opposition_votes,
        election_absent_votes=state.election_absent_votes,
        poll_rate=state.poll_rate,
        peak_poll_rate=state.peak_poll_rate,
        poll_history=list(state.poll_history),
        policy_implementations=policy_implementations,
        policy_active=policy_active,
        policy_cost_multipliers=state.policy_cost_multipliers.copy(),
        policy_income_multipliers=state.policy_income_multipliers.copy(),
        policy_cost_scalars=state.policy_cost_scalars.copy(),
        policy_income_scalars=state.policy_income_scalars.copy(),
        effect_throttles=effect_throttles,
        policy_desired_throttles=policy_desired_throttles,
        policy_effect_history_delays=policy_effect_history_delays,
        policy_effect_history_started=policy_effect_history_started,
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
    # The interactive model exposes the game's post-order budget preview. The
    # native gamedrive path uses the previous policy-history sample for the
    # debt roll, while the visible current value is already at the target.
    if native_order_runtime:
        _recompute_native_order_finance(
            new_state,
            data,
            finance_policy_values,
            introduced_policies,
        )
        # ``_recompute_native_order_finance`` mutates the order state in place;
        # retain the current/target split for the following NextTurn call.
    else:
        _recompute_orders_finance(new_state, data)
    return new_state


def list_available_actions(
    state: SimulationState,
    data: Optional[SimulationData] = None,
) -> List[PolicyActionOption]:
    """Enumerate feasible single-step policy moves based on slider metadata and capital.

    Every option carries ``financial_delta``: the estimated per-turn change
    in (income − cost) for this policy at the requested slider position,
    using the same budget arithmetic as :func:`_recalculate_budget`.  This is
    the £ figure the game shows while dragging a slider, so agents can weigh
    affordability before committing political capital.
    """

    data = data or load_simulation_data()
    options: List[PolicyActionOption] = []
    context = {**state.values, **state.policies, **state.situations}
    setup = data_loader.load_country_setup(data.gamedata_root, state.country)
    wealth_mod = setup.wealth_mod if setup.wealth_mod > 0.0 else 1.0

    def policy_balance(name: str, level: float, active: bool) -> float:
        definition = data.policies[name]
        if not active:
            return 0.0
        return (
            _policy_income_amount(
                definition,
                level,
                context,
                multiplier=state.policy_income_multipliers.get(name),
                scalar=state.policy_income_scalars.get(name, 1.0),
                wealth_mod=wealth_mod,
            )
            - _policy_cost_amount(
                definition,
                level,
                context,
                multiplier=state.policy_cost_multipliers.get(name),
                scalar=state.policy_cost_scalars.get(name, 1.0),
                wealth_mod=wealth_mod,
            )
        )

    for policy in data.policies.values():
        current = state.policies.get(policy.name, 0.0)
        slider = _get_slider(data, policy)
        uncancellable = _is_uncancellable(policy)
        capital = state.political_capital
        active = state.policy_active.get(policy.name, current > EPSILON)
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
                            financial_delta=(
                                policy_balance(policy.name, target, True)
                                - policy_balance(policy.name, current, False)
                            ),
                        )
                    )
                continue
        current_balance = policy_balance(policy.name, current, active)
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
                    financial_delta=(
                        policy_balance(policy.name, raise_target, True)
                        - current_balance
                    ),
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
                    financial_delta=(
                        policy_balance(policy.name, lower_target, True)
                        - current_balance
                    ),
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
                    financial_delta=(
                        policy_balance(policy.name, current, False)
                        - current_balance
                    ),
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
        "policy_cost_histories": state.policy_cost_histories,
        "policy_income_histories": state.policy_income_histories,
        "policy_finance_levels": state.policy_finance_levels,
        "total_expenditure": state.total_expenditure,
        "total_income": state.total_income,
        "global_economy_position": state.global_economy_position,
        "hidden_histories": state.hidden_histories,
        "voter_values": state.voter_values,
        "voter_percentages": state.voter_percentages,
        "voter_frequencies": state.voter_frequencies,
        "voter_incomes": state.voter_incomes,
        "voter_frequency_grudges": state.voter_frequency_grudges,
        "grudges": [
            {
                "target": grudge.target,
                "value": grudge.value,
                "decay": grudge.decay,
                "source": grudge.source,
                "gui_name": grudge.gui_name,
            }
            for grudge in state.grudges
        ],
        "voters": [
            {
                "groups": dict(v.groups),
                "value": v.value,
                "income": v.income,
                "inincome": v.inincome,
                "militancy": v.militancy,
                "voting_tech": v.voting_tech,
                "initial_socialism": v.initial_socialism,
                "initial_liberalism": v.initial_liberalism,
                "radicalism": v.radicalism,
                "gender": v.gender,
                "opposition_sympathy": v.opposition_sympathy,
                "player_sympathy": v.player_sympathy,
                "last_vote": v.last_vote,
                "survival": v.survival,
                "forecast": v.forecast,
                "party": v.party,
                "organizations": list(v.organizations),
            }
            for v in state.voters
        ],
        "parties": {
            name: {
                "status": party.status,
                "party_type": party.party_type,
                "members_last_turn": party.members_last_turn,
                "member_history": list(party.member_history),
                "activist_history": list(party.activist_history),
            }
            for name, party in state.parties.items()
        },
        "election_turns_until": state.election_turns_until,
        "election_current_term": state.election_current_term,
        "election_result": state.election_result,
        "last_election_winner": state.last_election_winner,
        "election_player_votes": state.election_player_votes,
        "election_opposition_votes": state.election_opposition_votes,
        "election_absent_votes": state.election_absent_votes,
        "poll_rate": state.poll_rate,
        "peak_poll_rate": state.peak_poll_rate,
        "poll_history": list(state.poll_history),
        "policy_implementations": state.policy_implementations,
        "policy_active": state.policy_active,
        "policy_cost_multipliers": state.policy_cost_multipliers,
        "policy_income_multipliers": state.policy_income_multipliers,
        "policy_cost_scalars": state.policy_cost_scalars,
        "policy_income_scalars": state.policy_income_scalars,
        "effect_throttles": state.effect_throttles,
        "policy_desired_throttles": state.policy_desired_throttles,
        "policy_effect_history_delays": state.policy_effect_history_delays,
        "policy_effect_history_started": state.policy_effect_history_started,
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
        # Pre-game covariate rings for forecasting agents; keep the end-turn
        # marker so a restored snapshot can still tell whether the rings are
        # aligned with the live turn.
        "value_histories": {
            name: list(values) for name, values in state.value_histories.items()
        },
        "value_histories_turn": state.value_histories_turn,
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
        policy_cost_histories={
            k: [float(value) for value in values]
            for k, values in dict(payload.get("policy_cost_histories", {})).items()
        },
        policy_income_histories={
            k: [float(value) for value in values]
            for k, values in dict(payload.get("policy_income_histories", {})).items()
        },
        policy_finance_levels={
            k: float(v)
            for k, v in dict(
                payload.get("policy_finance_levels", payload["policies"])
            ).items()
        },
        total_expenditure=float(payload.get("total_expenditure", 0.0)),
        total_income=float(payload.get("total_income", 0.0)),
        global_economy_position=float(payload.get("global_economy_position", 0.0)),
        hidden_histories={
            k: [float(value) for value in values]
            for k, values in dict(payload.get("hidden_histories", {})).items()
        },
        voter_values={k: float(v) for k, v in dict(payload.get("voter_values", {})).items()},
        voter_percentages={
            k: float(v) for k, v in dict(payload.get("voter_percentages", {})).items()
        },
        voter_frequencies={
            k: float(v) for k, v in dict(payload.get("voter_frequencies", {})).items()
        },
        voter_incomes={
            k: float(v) for k, v in dict(payload.get("voter_incomes", {})).items()
        },
        voter_frequency_grudges={
            k: float(v)
            for k, v in dict(payload.get("voter_frequency_grudges", {})).items()
        },
        grudges=[
            Grudge(
                target=str(item.get("target", "")),
                value=float(item.get("value", 0.0)),
                decay=float(item.get("decay", 1.0)),
                source=str(item.get("source", "")),
                gui_name=str(item.get("gui_name", "")),
            )
            for item in payload.get("grudges", [])
            if isinstance(item, dict) and item.get("target")
        ],
        voters=[
            Voter(
                groups={int(k): float(w) for k, w in dict(v.get("groups", {})).items()},
                value=float(v.get("value", 0.0)),
                income=float(v.get("income", 0.0)),
                inincome=float(v.get("inincome", 0.0)),
                militancy=float(v.get("militancy", 0.0)),
                voting_tech=float(v.get("voting_tech", 0.0)),
                initial_socialism=float(v.get("initial_socialism", 0.0)),
                initial_liberalism=float(v.get("initial_liberalism", 0.0)),
                radicalism=float(v.get("radicalism", 0.0)),
                gender=int(v.get("gender", 0)),
                opposition_sympathy=float(v.get("opposition_sympathy", 0.0)),
                player_sympathy=float(v.get("player_sympathy", 0.0)),
                last_vote=int(v.get("last_vote", 0)),
                survival=int(v.get("survival", 0)),
                forecast=int(v.get("forecast", 0)),
                party=str(v.get("party", "")),
                organizations=[str(org) for org in v.get("organizations", [])],
            )
            for v in payload.get("voters", [])
        ],
        parties={
            str(name): PartyState(
                name=str(name),
                status=int(value.get("status", 0)),
                party_type=int(value.get("party_type", 0)),
                members_last_turn=int(value.get("members_last_turn", 0)),
                member_history=[
                    int(item) for item in value.get("member_history", [])
                ],
                activist_history=[
                    int(item) for item in value.get("activist_history", [])
                ],
            )
            for name, value in dict(payload.get("parties", {})).items()
        },
        election_turns_until=int(payload.get("election_turns_until", 0)),
        election_current_term=int(payload.get("election_current_term", 0)),
        election_result=(
            str(payload["election_result"])
            if payload.get("election_result") is not None
            else None
        ),
        last_election_winner=(
            str(payload["last_election_winner"])
            if payload.get("last_election_winner") is not None
            else None
        ),
        election_player_votes=int(payload.get("election_player_votes", 0)),
        election_opposition_votes=int(payload.get("election_opposition_votes", 0)),
        election_absent_votes=int(payload.get("election_absent_votes", 0)),
        poll_rate=float(payload.get("poll_rate", 0.0)),
        peak_poll_rate=float(payload.get("peak_poll_rate", 0.0)),
        poll_history=[float(value) for value in payload.get("poll_history", [])],
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
        policy_effect_history_delays={
            k: int(v)
            for k, v in dict(payload.get("policy_effect_history_delays", {})).items()
        },
        policy_effect_history_started={
            k: bool(v)
            for k, v in dict(payload.get("policy_effect_history_started", {})).items()
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
        value_histories=_history_rings(payload, "value_histories"),
        value_histories_turn=(
            int(raw_turn)
            if (raw_turn := payload.get("value_histories_turn")) is not None
            else None
        ),
    )
    data = load_simulation_data()
    _recalculate_budget(state, data)
    state.policy_cost_histories = _complete_policy_finance_histories(
        state.policy_cost_histories, state.policy_costs, data
    )
    state.policy_income_histories = _complete_policy_finance_histories(
        state.policy_income_histories, state.policy_incomes, data
    )
    return state


def _history_rings(
    payload: Dict[str, object], key: str
) -> Dict[str, List[float]]:
    """Read a dict of float-list rings, dropping empty entries."""

    raw = payload.get(key) or {}
    rings: Dict[str, List[float]] = {}
    for name, values in dict(raw).items():
        ring = [float(value) for value in values]
        if ring:
            rings[str(name)] = ring
    return rings


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
