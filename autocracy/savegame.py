from __future__ import annotations

from dataclasses import dataclass, field
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import networkx as nx

from . import simulator
from .models import EffectHistory, SimulationData, SimulationState

ENCODING = "latin-1"
DEFAULT_TOLERANCE = 1e-3
NUMERIC_TAG_RE = re.compile(r"<(/?)(\d[\w]*)>")


@dataclass(slots=True)
class SaveGame:
    country: str
    turn: int
    simvalues: Dict[str, float]
    policies: Dict[str, float]
    policy_desired_throttles: Dict[str, float]
    policy_costs: Dict[str, float]
    policy_incomes: Dict[str, float]
    total_expenditure: float
    total_income: float
    effect_histories: List[EffectHistory]
    situations: Dict[str, float]
    active_situations: List[str]
    global_economy_position: float
    global_economy_years: float
    global_economy_intensity: float
    hidden_values: Dict[str, float]
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
    ministerial_effectiveness: Dict[str, float] = field(default_factory=dict)
    ministerial_competence: Dict[str, float] = field(default_factory=dict)
    political_capital: float = 0.0
    # The save's <inherited> block holds the simvalues as of two turns ago.
    # The game uses the current-vs-inherited comparison to decide which
    # inertial effect rings receive a new sample each turn.
    inherited_values: Dict[str, float] = field(default_factory=dict)


@dataclass(slots=True)
class DiffEntry:
    name: str
    simulator: float
    savegame: float

    @property
    def delta(self) -> float:
        return self.simulator - self.savegame


@dataclass(slots=True)
class StateComparison:
    value_diffs: List[DiffEntry]
    policy_diffs: List[DiffEntry]
    missing_values: List[str]
    missing_policies: List[str]
    cost_diffs: List[DiffEntry]
    income_diffs: List[DiffEntry]
    missing_costs: List[str]
    missing_incomes: List[str]
    budget_diffs: List[DiffEntry]

    def has_differences(self) -> bool:
        return bool(
            self.value_diffs
            or self.policy_diffs
            or self.missing_values
            or self.missing_policies
            or self.cost_diffs
            or self.income_diffs
            or self.missing_costs
            or self.missing_incomes
            or self.budget_diffs
        )


def _clean_xml(text: str) -> str:
    stripped = text.strip()
    if stripped.endswith("</xml>"):
        stripped = stripped[: stripped.rfind("</xml>")]
    sanitized = NUMERIC_TAG_RE.sub(lambda m: f"<{m.group(1)}n_{m.group(2)}>", stripped)
    return f"<savegame>{sanitized}</savegame>"


def _first_history_value(text: Optional[str]) -> float:
    if not text:
        return 0.0
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            return float(chunk)
        except ValueError:
            continue
    return 0.0


def _history_values(text: Optional[str]) -> List[float]:
    if not text:
        return []
    values: List[float] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            values.append(float(chunk))
        except ValueError:
            continue
    return values


def parse_savegame(path: str | Path) -> SaveGame:
    raw_text = Path(path).read_text(encoding=ENCODING)
    wrapped = _clean_xml(raw_text)
    root = ET.fromstring(wrapped)
    country = (root.findtext("mission/name") or "").strip() or "uk"
    turns = root.findall("compass_progress/en/turn")
    try:
        turn = int(float(turns[-1].text or "0")) if turns else 0
    except (TypeError, ValueError):
        turn = 0
    simvalues: Dict[str, float] = {}
    simvalues_elem = root.find("simvalues")
    if simvalues_elem is not None:
        for simvalue in simvalues_elem.findall("simvalue"):
            name = simvalue.findtext("name")
            value = simvalue.findtext("value")
            if not name or value is None:
                continue
            try:
                simvalues[name.strip()] = float(value)
            except ValueError:
                continue
    policies: Dict[str, float] = {}
    policy_desired_throttles: Dict[str, float] = {}
    policy_costs: Dict[str, float] = {}
    policy_incomes: Dict[str, float] = {}
    policy_implementations: Dict[str, float] = {}
    policy_active: Dict[str, bool] = {}
    policy_cost_multipliers: Dict[str, float] = {}
    policy_income_multipliers: Dict[str, float] = {}
    policy_cost_scalars: Dict[str, float] = {}
    policy_income_scalars: Dict[str, float] = {}
    effect_throttles: Dict[str, float] = {}
    policies_elem = root.find("policies")
    if policies_elem is not None:
        for policy in policies_elem.findall("policy"):
            name = policy.findtext("name")
            value = policy.findtext("val")
            if not name or value is None:
                continue
            try:
                normalized = name.strip()
                policies[normalized] = float(value)
                policy_desired_throttles[normalized] = float(
                    policy.findtext("targ", default=value)
                )
                policy_implementations[normalized] = float(
                    policy.findtext("imp", default="0")
                )
                policy_active[normalized] = (
                    policy.findtext("active", default="0").strip() == "1"
                )
                policy_cost_multipliers[normalized] = float(
                    policy.findtext("cost_mult", default="1")
                )
                policy_income_multipliers[normalized] = float(
                    policy.findtext("incom_mult", default="1")
                )
                policy_income_scalars[normalized] = float(
                    policy.findtext("earn_scalar", default="1")
                )
                policy_cost_scalars[normalized] = float(
                    policy.findtext("cost_scalar", default="1")
                )
                cost_history = policy.findtext("costhistory")
                income_history = policy.findtext("incomehistory")
                policy_costs[normalized] = _first_history_value(cost_history)
                policy_incomes[normalized] = _first_history_value(income_history)
            except ValueError:
                continue
    effects_elem = root.find("effects")
    if effects_elem is not None:
        for effect in effects_elem.findall("effect"):
            raw_throttle = effect.findtext("throttle") or ""
            parts = [part.strip() for part in raw_throttle.split(",")]
            if len(parts) != 3:
                continue
            try:
                effect_throttles[parts[1]] = float(parts[2])
            except ValueError:
                continue
    total_expenditure = sum(policy_costs.values())
    total_income = sum(policy_incomes.values())
    political_capital = 0.0
    political_capital_elem = root.find("politicalcapital")
    if political_capital_elem is not None:
        try:
            political_capital = float(
                political_capital_elem.findtext("points", default="0")
            )
        except ValueError:
            political_capital = 0.0
    effect_histories: List[EffectHistory] = []
    effect_histories_elem = root.find("effecthistories")
    if effect_histories_elem is not None:
        for effect in effect_histories_elem.findall("effect"):
            raw_history = effect.findtext("effecthistory")
            if not raw_history:
                continue
            parts = [part.strip() for part in raw_history.split(",")]
            if len(parts) < 3:
                continue
            effect_histories.append(
                EffectHistory(
                    source=parts[0],
                    target=parts[1],
                    values=_history_values(",".join(parts[2:])),
                )
            )
    situations: Dict[str, float] = {}
    active_situations: List[str] = []
    situations_elem = root.find("situations")
    if situations_elem is not None:
        for situation in situations_elem.findall("situation"):
            name = situation.findtext("name")
            if not name:
                continue
            try:
                value = float(situation.findtext("val", default="0"))
            except ValueError:
                value = 0.0
            normalized = name.strip()
            situations[normalized] = value
            if situation.findtext("active", default="0").strip() == "1":
                active_situations.append(normalized)
    global_economy_position = 0.0
    global_economy_years = 0.0
    global_economy_intensity = 0.0
    global_economy_elem = root.find("globaleconomy")
    if global_economy_elem is not None:
        global_economy_position = _first_history_value(global_economy_elem.findtext("pos"))
        global_economy_years = _first_history_value(global_economy_elem.findtext("years"))
        global_economy_intensity = _first_history_value(global_economy_elem.findtext("intens"))
    hidden_values: Dict[str, float] = {}
    simulation_elem = root.find("simulation")
    if simulation_elem is not None:
        # These are the fixed global neurons created by the game before it
        # loads simulation.csv.  The unnamed slots are intentionally omitted.
        hidden_names = {
            "n_0": "_global_socialism",
            "n_1": "_global_liberalism",
            "n_2": "_security_",
            "n_3": "_winning_",
            "n_5": "_effectivedebt_",
            "n_6": "_globaleconomy_",
            "n_7": "_global_interest_rates_",
            "n_8": "_year",
        }
        for slot, name in hidden_names.items():
            raw_value = simulation_elem.findtext(slot)
            if raw_value is None:
                continue
            try:
                hidden_values[name] = float(raw_value)
            except ValueError:
                continue
    voter_values: Dict[str, float] = {}
    voter_percentages: Dict[str, float] = {}
    voter_frequencies: Dict[str, float] = {}
    votertypes_elem = root.find("votertypes")
    if votertypes_elem is not None:
        for votertype in votertypes_elem.findall("votertype"):
            name = (votertype.findtext("name") or "").strip()
            if not name:
                continue
            try:
                voter_values[name] = float(votertype.findtext("value", default="0"))
            except ValueError:
                pass
            percentage = _first_history_value(votertype.findtext("perc_history"))
            voter_percentages[f"{name}_perc"] = percentage
            try:
                voter_frequencies[f"{name}_freq"] = float(
                    votertype.findtext("freqval", default="0")
                )
            except ValueError:
                pass
    ministerial_effectiveness: Dict[str, float] = {}
    ministerial_competence: Dict[str, float] = {}
    ministers_elem = root.find("ministers")
    if ministers_elem is not None:
        for minister in ministers_elem.findall("minister"):
            job = (minister.findtext("job") or "").strip()
            if not job or job.lower() == "none":
                continue
            try:
                experience = float(minister.findtext("exp", default="0"))
            except ValueError:
                continue
            suitability = 0.0
            for suit in minister.findall("suits/suit"):
                if (suit.findtext("grp") or "").strip() != job:
                    continue
                try:
                    suitability = float(suit.findtext("val", default="0"))
                except ValueError:
                    suitability = 0.0
                break
            product = experience * suitability
            ministerial_competence[job] = max(0.0, min(1.0, 0.2 + 0.8 * product))
            ministerial_effectiveness[job] = max(0.0, min(1.0, 0.8 + 0.4 * product))
    inherited_values: Dict[str, float] = {}
    inherited_elem = root.find("inherited")
    if inherited_elem is not None:
        for inherited in inherited_elem.findall("ival"):
            name = inherited.findtext("name")
            value = inherited.findtext("val")
            if not name or value is None:
                continue
            try:
                inherited_values[name.strip()] = float(value)
            except ValueError:
                continue
    return SaveGame(
        country=country.lower(),
        turn=turn,
        simvalues=simvalues,
        policies=policies,
        policy_desired_throttles=policy_desired_throttles,
        policy_costs=policy_costs,
        policy_incomes=policy_incomes,
        total_expenditure=total_expenditure,
        total_income=total_income,
        political_capital=political_capital,
        effect_histories=effect_histories,
        situations=situations,
        active_situations=active_situations,
        global_economy_position=global_economy_position,
        global_economy_years=global_economy_years,
        global_economy_intensity=global_economy_intensity,
        hidden_values=hidden_values,
        voter_values=voter_values,
        voter_percentages=voter_percentages,
        voter_frequencies=voter_frequencies,
        policy_implementations=policy_implementations,
        policy_active=policy_active,
        policy_cost_multipliers=policy_cost_multipliers,
        policy_income_multipliers=policy_income_multipliers,
        policy_cost_scalars=policy_cost_scalars,
        policy_income_scalars=policy_income_scalars,
        effect_throttles=effect_throttles,
        ministerial_effectiveness=ministerial_effectiveness,
        ministerial_competence=ministerial_competence,
        inherited_values=inherited_values,
    )


def state_from_savegame(
    save: SaveGame,
    data: Optional[SimulationData] = None,
) -> tuple[SimulationState, nx.DiGraph]:
    data = data or simulator.load_simulation_data()
    state, graph = simulator.get_initial_state(save.country, data.gamedata_root)
    state.turn = save.turn
    state.political_capital = save.political_capital
    for name, value in save.simvalues.items():
        if name in state.values:
            node = data.nodes.get(name)
            if node:
                clamped = max(node.minimum, min(node.maximum, value))
            else:
                clamped = value
            state.values[name] = clamped
    for name, value in save.policies.items():
        if name in state.policies:
            state.policies[name] = max(0.0, min(1.0, value))
    state.policy_desired_throttles = save.policy_desired_throttles.copy()
    for name, value in save.hidden_values.items():
        state.values[name] = value
    for name, value in save.voter_values.items():
        state.voter_values[name] = value
        state.values[name] = value
    for name, value in save.voter_percentages.items():
        state.voter_percentages[name] = value
        # Percentage sources are exposed to situation equations under their
        # serialized names (for example ``TradeUnionist_perc``).
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
    state.ministerial_effectiveness = save.ministerial_effectiveness.copy()
    state.ministerial_competence = save.ministerial_competence.copy()
    state.situations = save.situations.copy()
    state.active_situations = save.active_situations.copy()
    state.global_economy_position = save.global_economy_position
    state.effect_histories = [
        EffectHistory(history.source, history.target, list(history.values))
        for history in save.effect_histories
    ]
    simulator.recompute_effects(state, graph, effect_histories=state.effect_histories)
    return state, graph


def load_state_from_savegame(
    path: str | Path,
    data: Optional[SimulationData] = None,
) -> tuple[SimulationState, nx.DiGraph]:
    save = parse_savegame(path)
    return state_from_savegame(save, data=data)


def compare_state_to_savegame(
    state: SimulationState,
    save: SaveGame,
    tolerance: float = DEFAULT_TOLERANCE,
) -> StateComparison:
    value_diffs: List[DiffEntry] = []
    missing_values: List[str] = []
    for name, save_val in save.simvalues.items():
        sim_val = state.values.get(name)
        if sim_val is None:
            missing_values.append(name)
            continue
        if abs(sim_val - save_val) > tolerance:
            value_diffs.append(DiffEntry(name=name, simulator=sim_val, savegame=save_val))
    policy_diffs: List[DiffEntry] = []
    missing_policies: List[str] = []
    for name, save_val in save.policies.items():
        sim_val = state.policies.get(name)
        if sim_val is None:
            missing_policies.append(name)
            continue
        if abs(sim_val - save_val) > tolerance:
            policy_diffs.append(DiffEntry(name=name, simulator=sim_val, savegame=save_val))
    cost_diffs, missing_costs = _diff_budget_maps(state.policy_costs, save.policy_costs, tolerance)
    income_diffs, missing_incomes = _diff_budget_maps(state.policy_incomes, save.policy_incomes, tolerance)
    budget_diffs: List[DiffEntry] = []
    if abs(state.total_income - save.total_income) > tolerance:
        budget_diffs.append(
            DiffEntry(name="Total Income", simulator=state.total_income, savegame=save.total_income)
        )
    if abs(state.total_expenditure - save.total_expenditure) > tolerance:
        budget_diffs.append(
            DiffEntry(name="Total Expenditure", simulator=state.total_expenditure, savegame=save.total_expenditure)
        )
    value_diffs.sort(key=lambda diff: abs(diff.delta), reverse=True)
    policy_diffs.sort(key=lambda diff: abs(diff.delta), reverse=True)
    cost_diffs.sort(key=lambda diff: abs(diff.delta), reverse=True)
    income_diffs.sort(key=lambda diff: abs(diff.delta), reverse=True)
    budget_diffs.sort(key=lambda diff: abs(diff.delta), reverse=True)
    return StateComparison(
        value_diffs=value_diffs,
        policy_diffs=policy_diffs,
        missing_values=missing_values,
        missing_policies=missing_policies,
        cost_diffs=cost_diffs,
        income_diffs=income_diffs,
        missing_costs=missing_costs,
        missing_incomes=missing_incomes,
        budget_diffs=budget_diffs,
    )


def _diff_budget_maps(
    simulator_map: Dict[str, float],
    save_map: Dict[str, float],
    tolerance: float,
) -> Tuple[List[DiffEntry], List[str]]:
    diffs: List[DiffEntry] = []
    missing: List[str] = []
    for name, save_val in save_map.items():
        sim_val = simulator_map.get(name)
        if sim_val is None:
            missing.append(name)
            continue
        if abs(sim_val - save_val) > tolerance:
            diffs.append(DiffEntry(name=name, simulator=sim_val, savegame=save_val))
    return diffs, missing
