from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import xml.etree.ElementTree as ET

import networkx as nx

from . import simulator
from .models import SimulationData, SimulationState

ENCODING = "latin-1"
DEFAULT_TOLERANCE = 1e-3
NUMERIC_TAG_RE = re.compile(r"<(/?)(\d[\w]*)>")


@dataclass(slots=True)
class SaveGame:
    country: str
    turn: int
    simvalues: Dict[str, float]
    policies: Dict[str, float]
    policy_costs: Dict[str, float]
    policy_incomes: Dict[str, float]
    total_expenditure: float
    total_income: float


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


def parse_savegame(path: str | Path) -> SaveGame:
    raw_text = Path(path).read_text(encoding=ENCODING)
    wrapped = _clean_xml(raw_text)
    root = ET.fromstring(wrapped)
    country = (root.findtext("mission/name") or "").strip() or "uk"
    try:
        turn = int(float(root.findtext("compass_progress/en/turn", default="0")))
    except ValueError:
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
    policy_costs: Dict[str, float] = {}
    policy_incomes: Dict[str, float] = {}
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
                cost_history = policy.findtext("costhistory")
                income_history = policy.findtext("incomehistory")
                policy_costs[normalized] = _first_history_value(cost_history)
                policy_incomes[normalized] = _first_history_value(income_history)
            except ValueError:
                continue
    total_expenditure = sum(policy_costs.values())
    total_income = sum(policy_incomes.values())
    return SaveGame(
        country=country.lower(),
        turn=turn,
        simvalues=simvalues,
        policies=policies,
        policy_costs=policy_costs,
        policy_incomes=policy_incomes,
        total_expenditure=total_expenditure,
        total_income=total_income,
    )


def state_from_savegame(
    save: SaveGame,
    data: Optional[SimulationData] = None,
) -> tuple[SimulationState, nx.DiGraph]:
    data = data or simulator.load_simulation_data()
    state, graph = simulator.get_initial_state(save.country, data.gamedata_root)
    state.turn = save.turn
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
    simulator.recompute_effects(state, graph)
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
