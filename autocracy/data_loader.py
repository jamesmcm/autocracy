from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .models import (
    BudgetModifier,
    CountrySetup,
    Effect,
    NodeDefinition,
    PolicyDefinition,
    SliderDefinition,
    SituationDefinition,
)

ENCODING = "latin-1"


def _safe_float(value: str, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _parse_effect_cell(source: str, cell: str) -> Optional[Effect]:
    cell = cell.strip()
    if not cell or cell in {"#", "#Effects"}:
        return None
    parts = [p.strip() for p in cell.split(",") if p.strip()]
    if not parts:
        return None
    target = parts[0]
    expression = parts[1] if len(parts) > 1 else "0"
    inertia = _safe_float(parts[2], default=0.0) if len(parts) > 2 else None
    if inertia == 0:
        inertia = None
    return Effect(source=source, target=target, expression=expression, inertia=inertia)


def _parse_budget_modifiers(raw: Optional[str]) -> List[BudgetModifier]:
    if not raw:
        return []
    text = raw.strip().strip('"')
    if not text:
        return []
    modifiers: List[BudgetModifier] = []
    for chunk in text.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [part.strip() for part in chunk.split(",") if part.strip()]
        if not parts:
            continue
        source = parts[0]
        expression = parts[1] if len(parts) > 1 else "0"
        modifiers.append(BudgetModifier(source=source, expression=expression))
    return modifiers


def load_simulation_nodes(root: Path) -> Tuple[Dict[str, NodeDefinition], List[Effect]]:
    nodes: Dict[str, NodeDefinition] = {}
    effects: List[Effect] = []
    path = root / "simulation" / "simulation.csv"
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)  # skip header row
        for row in reader:
            if len(row) < 5 or not row[1]:
                continue
            name = row[1].strip()
            nodes[name] = NodeDefinition(
                name=name,
                display_name=row[2].strip(),
                description=row[3].strip(),
                category=row[4].strip(),
                default=_safe_float(row[5], default=0.0),
                minimum=_safe_float(row[6], default=0.0),
                maximum=_safe_float(row[7], default=1.0),
                emotion=row[8].strip(),
                icon=row[9].strip(),
            )
            try:
                input_marker_idx = row.index("#", 10)
                output_marker_idx = row.index("#", input_marker_idx + 1)
            except ValueError:
                continue

            for cell in row[input_marker_idx + 1 : output_marker_idx]:
                cell = cell.strip()
                if not cell:
                    continue
                parts = [p.strip() for p in cell.split(",") if p.strip()]
                if not parts:
                    continue
                source = parts[0]
                expression = parts[1] if len(parts) > 1 else "0"
                inertia = _safe_float(parts[2], default=0.0) if len(parts) > 2 else None
                if inertia == 0:
                    inertia = None
                effects.append(Effect(source=source, target=name, expression=expression, inertia=inertia))

            for cell in row[output_marker_idx + 1 :]:
                effect = _parse_effect_cell(name, cell)
                if effect:
                    effects.append(effect)
    return nodes, effects


def load_voter_types(root: Path) -> Tuple[Dict[str, NodeDefinition], List[Effect]]:
    nodes: Dict[str, NodeDefinition] = {}
    effects: List[Effect] = []
    path = root / "simulation" / "votertypes.csv"
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        header = next(reader, [])
        influence_index = None
        for idx, column in enumerate(header):
            if column.strip().lower() == "influences":
                influence_index = idx
                break
        for row in reader:
            if len(row) < 2 or not row[1]:
                continue
            name = row[1].strip()
            default = _safe_float(row[6], default=0.0)
            percentage = _safe_float(row[7], default=0.0)
            nodes[name] = NodeDefinition(
                name=name,
                display_name=row[2].strip(),
                description=row[8].strip(),
                category="VOTER",
                default=default,
                minimum=-1.0,
                maximum=1.0,
            )
            freq_name = f"{name}_freq"
            nodes[freq_name] = NodeDefinition(
                name=freq_name,
                display_name=f"{row[2].strip()} Membership",
                description=f"Membership share for {row[2].strip()}",
                category="VOTER_FREQ",
                # The CSV percentage seeds the native linked-list membership
                # count, not the nested SIM_Neuron's base value. The latter
                # is constructed with a zero default and receives effects
                # during the simulation pass.
                default=0.0,
                # Native SIM_VoterType stores this as the current value of
                # its nested ``initial_voter_freq`` neuron.  That neuron is
                # a normal [-1, 1] SIM_Neuron with a zero base; it is not
                # the non-negative membership percentage from CSV.
                minimum=-1.0,
                maximum=1.0,
                initial_percentage=percentage,
            )
            if influence_index is not None:
                for cell in row[influence_index:]:
                    cell = cell.strip()
                    if not cell:
                        continue
                    effect = _parse_effect_cell(name, cell)
                    if effect:
                        effects.append(effect)
    return nodes, effects


def load_policies(root: Path) -> Dict[str, PolicyDefinition]:
    policies: Dict[str, PolicyDefinition] = {}
    path = root / "simulation" / "policies.csv"
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 4 or not row[1]:
                continue
            name = row[1].strip()
            slider = row[3].strip()
            description = row[4].strip()
            flags = [flag.strip() for flag in row[5].split("|") if flag.strip()] if row[5] else []
            introduce = _safe_float(row[6], default=0.0)
            cancel = _safe_float(row[7], default=0.0)
            raise_cost = _safe_float(row[8], default=0.0)
            lower_cost = _safe_float(row[9], default=0.0)
            department = row[10].strip()
            min_cost = _safe_float(row[11], default=0.0)
            max_cost = _safe_float(row[12], default=0.0)
            raw_cost_multiplier = row[13].strip() if len(row) > 13 else ""
            cost_multipliers = _parse_budget_modifiers(raw_cost_multiplier)
            implementation_time = _safe_float(row[14], default=0.0)
            min_income = _safe_float(row[15], default=0.0)
            max_income = _safe_float(row[16], default=0.0)
            raw_income_multiplier = row[17].strip() if len(row) > 17 else ""
            income_multipliers = _parse_budget_modifiers(raw_income_multiplier)
            try:
                effect_index = row.index("#Effects")
            except ValueError:
                effect_index = len(row)
            effects: List[Effect] = []
            for cell in row[effect_index + 1 :]:
                effect = _parse_effect_cell(name, cell)
                if effect:
                    effects.append(effect)
            policies[name] = PolicyDefinition(
                name=name,
                display_name=row[2].strip(),
                description=description,
                slider=slider or "default",
                introduce_cost=introduce,
                cancel_cost=cancel,
                raise_cost=raise_cost,
                lower_cost=lower_cost,
                department=department,
                flags=flags,
                min_cost=min_cost,
                max_cost=max_cost,
                cost_multiplier=raw_cost_multiplier,
                implementation_time=implementation_time,
                min_income=min_income,
                max_income=max_income,
                income_multiplier=raw_income_multiplier,
                cost_multipliers=cost_multipliers,
                income_multipliers=income_multipliers,
                effects=effects,
            )
    return policies


def load_sliders(root: Path) -> Dict[str, SliderDefinition]:
    sliders: Dict[str, SliderDefinition] = {}
    path = root / "simulation" / "sliders.csv"
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if len(row) < 3 or not row[1]:
                continue
            name = row[1].strip()
            kind = (row[2].strip() or "DISCRETE").upper()
            extra = [segment.strip() for segment in row[3:] if segment.strip()]
            labels: List[str] = []
            min_value = 0.0
            max_value = 1.0
            if kind == "PERCENTAGE":
                if extra:
                    min_value = _safe_float(extra[0], default=0.0)
                if len(extra) > 1:
                    max_value = _safe_float(extra[1], default=100.0)
            else:
                if extra and _is_number(extra[0]):
                    extra = extra[1:]
                labels = extra
            sliders[name] = SliderDefinition(
                name=name,
                kind=kind,
                labels=labels,
                min_value=min_value,
                max_value=max_value,
            )
    return sliders


def load_situations(root: Path) -> Dict[str, SituationDefinition]:
    situations: Dict[str, SituationDefinition] = {}
    path = root / "simulation" / "situations.csv"
    if not path.exists():
        return situations
    with path.open(encoding=ENCODING, newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if len(row) < 5 or not row[1]:
                continue
            name = row[1].strip()
            if not name:
                continue
            display_name = row[2].strip()
            description = row[3].strip()
            category = row[4].strip()
            icon = row[5].strip() if len(row) > 5 else ""
            positive = bool(int(row[8].strip())) if len(row) > 8 and row[8].strip() else False
            start_trigger = _safe_float(row[9], default=0.6) if len(row) > 9 else 0.6
            stop_trigger = _safe_float(row[10], default=0.4) if len(row) > 10 else 0.4
            cost = _safe_float(row[11], default=0.0) if len(row) > 11 else 0.0
            split_index = None
            for idx, cell in enumerate(row):
                if cell.strip() == "#" and idx >= 13:
                    split_index = idx
                    break
            if split_index is None:
                split_index = len(row)
            inputs_raw = row[13:split_index] if len(row) > 13 else []
            effects_raw = row[split_index + 1 :] if split_index + 1 < len(row) else []
            default = 0.0
            prerequisites: List[str] = []
            inputs: List[Effect] = []
            for idx, cell in enumerate(inputs_raw):
                parsed = _parse_effect_cell(name, cell)
                if not parsed or not parsed.target:
                    continue
                if parsed.target == "_default_":
                    # The _default_ cell is a small expression (e.g.
                    # ``0.8+(0*x)``), not necessarily a bare number.
                    from .simulator import evaluate_expression  # late import

                    default = evaluate_expression(
                        parsed.expression, 0.0, context={}
                    )
                    continue
                if parsed.target == "_prereq_":
                    prerequisite = parsed.expression.strip()
                    if prerequisite:
                        prerequisites.append(prerequisite)
                    continue
                effect = Effect(
                    source=parsed.target,
                    target=name,
                    expression=parsed.expression,
                    inertia=parsed.inertia,
                    effect_id=f"situation::{name}::input::{idx}",
                )
                inputs.append(effect)
            effects: List[Effect] = []
            for idx, cell in enumerate(effects_raw):
                parsed = _parse_effect_cell(name, cell)
                if not parsed or not parsed.target:
                    continue
                parsed.effect_id = f"situation::{name}::effect::{idx}"
                effects.append(parsed)
            situations[name] = SituationDefinition(
                name=name,
                display_name=display_name,
                description=description,
                category=category,
                icon=icon,
                positive=positive,
                start_trigger=start_trigger,
                stop_trigger=stop_trigger,
                cost=cost,
                default=default,
                prerequisites=prerequisites,
                inputs=inputs,
                effects=effects,
            )
    return situations


def load_sim_config(root: Path) -> Dict[str, float]:
    config: Dict[str, float] = {}
    path = root / "simconfig.txt"
    with path.open(encoding=ENCODING) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("["):
                continue
            if "=" not in line:
                continue
            key, value = [chunk.strip() for chunk in line.split("=", 1)]
            config[key] = _safe_float(value, default=0.0)
    return config


def _parse_section_line(line: str) -> Optional[Tuple[str, str]]:
    if ": =" in line:
        line = line.replace(": =", "=")
    if "=" not in line:
        return None
    key, value = [chunk.strip() for chunk in line.split("=", 1)]
    return key, value.strip().strip('"')


def load_country_setup(root: Path, country: str) -> CountrySetup:
    mission_dir = root / "missions" / country
    mission_file = mission_dir / f"{country}.txt"
    if not mission_file.exists():
        raise FileNotFoundError(f"Mission file not found for country '{country}'")
    section = None
    config_data: Dict[str, str] = {}
    options: List[str] = []
    stats: Dict[str, str] = {}
    policies: Dict[str, float] = {}
    with mission_file.open(encoding=ENCODING) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("[") and line.endswith("]"):
                section = line.strip("[]").lower()
                continue
            if section == "options":
                options.append(line)
                continue
            parsed = _parse_section_line(line)
            if not parsed:
                continue
            key, value = parsed
            if section == "config":
                config_data[key.lower()] = value
            elif section == "stats":
                stats[key] = value
            elif section == "policies":
                policies[key] = _safe_float(value, default=0.0)
    overrides = load_country_overrides(mission_dir)
    return CountrySetup(
        name=country,
        currency=config_data.get("currency", "$"),
        description=config_data.get("description", ""),
        policy_levels=policies,
        options=options,
        stats=stats,
        overrides=overrides,
        economic_cycle_start=_safe_float(config_data.get("economic_cycle_start", "0")),
        wealth_mod=_safe_float(config_data.get("wealth_mod", "1"), default=1.0),
        difficulty=_safe_float(config_data.get("difficulty", "0.5"), default=0.5),
        min_income=_safe_float(config_data.get("min_income", "0")),
        max_income=_safe_float(config_data.get("max_income", "0")),
        min_gdp=_safe_float(config_data.get("min_gdp", "0")),
        max_gdp=_safe_float(config_data.get("max_gdp", "0")),
        starting_debt=_safe_float(config_data.get("starting_debt", "0")),
        term_length=int(_safe_float(config_data.get("term_length", "16"), default=16)),
        max_terms=int(_safe_float(config_data.get("max_terms", "-1"), default=-1)),
    )


def load_country_overrides(mission_dir: Path) -> List[dict]:
    overrides: List[dict] = []
    overrides_dir = mission_dir / "overrides"
    if not overrides_dir.exists():
        return overrides
    for ini_file in overrides_dir.glob("*.ini"):
        override_data: Dict[str, str] = {}
        with ini_file.open(encoding=ENCODING) as handle:
            section = None
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                if line.startswith("[") and line.endswith("]"):
                    section = line.strip("[]").lower()
                    continue
                if section != "override":
                    continue
                parsed = _parse_section_line(line)
                if parsed:
                    override_data[parsed[0]] = parsed[1]
        if override_data:
            overrides.append(override_data)
    return overrides


_CALIBRATION_PATH = Path(__file__).with_name("calibration.json")


def load_calibration(root: Path) -> Dict[str, object]:
    """Load the parity calibration, defaulting to the shipped values.

    The default calibration lives in ``autocracy/calibration.json`` and
    reproduces the shipped UK playthrough.  A ``calibration.json`` placed in
    the gamedata root overrides it, so a different country/gamedata set can be
    reproduced without editing the simulator.
    """
    import json

    defaults: Dict[str, object] = {}
    if _CALIBRATION_PATH.exists():
        try:
            loaded = json.loads(_CALIBRATION_PATH.read_text(encoding=ENCODING))
            if isinstance(loaded, dict):
                defaults = loaded
        except (ValueError, OSError):
            defaults = {}
    def _deep_merge(base: object, extra: object) -> object:
        if isinstance(base, dict) and isinstance(extra, dict):
            merged = dict(base)
            for key, value in extra.items():
                merged[key] = _deep_merge(merged.get(key), value)
            return merged
        return extra

    result = json.loads(json.dumps(defaults))
    path = root / "calibration.json"
    if path.exists():
        try:
            overrides = json.loads(path.read_text(encoding=ENCODING))
            if isinstance(overrides, dict):
                result = _deep_merge(result, overrides)  # type: ignore[arg-type]
        except (ValueError, OSError):
            pass
    return result
