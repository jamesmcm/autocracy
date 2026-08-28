"""Data-driven random event, dilemma, attack and pressure-group systems.

``gamedata/data/simulation`` ships the game's original ``events/*.txt``,
``dilemmas/*.txt`` and ``attacks/*.txt`` files plus ``pressuregroups.csv``.
This module parses those files and applies them to a ``SimulationState``
when the corresponding ``SimulationConfig`` toggle is enabled.

Every system defaults to **off**, which is what the deterministic save-parity
runs require; enabling any of them switches that part of the turn over to a
seeded, reproducible random stream.  The implementations are deliberately
approximate: the Democracy 3 executable keeps unstored random-system state
(event cooldowns, pressure-group strength), so live-game timing cannot be
reproduced exactly.  The important contract is that disabled systems are
bit-for-bit no-ops.
"""

from __future__ import annotations

import csv
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from . import data_loader
from .models import Grudge, SimulationConfig, SimulationData, SimulationState
from .simulator import DEFAULT_GAMEDATA, _clamp, evaluate_expression

SCRIPT_CALL_RE = re.compile(r"(\w+)\s*\(([^)]*)\)")


@dataclass(slots=True)
class ScriptAction:
    """One call parsed from an ``OnImplement``/``OnSuccess`` script."""

    name: str
    args: List[str]


@dataclass(slots=True)
class Influence:
    """A chance/trigger modifier: evaluate ``expression`` at ``source``."""

    source: str
    expression: str


@dataclass(slots=True)
class EventDefinition:
    name: str
    gui_name: str
    description: str
    influences: List[Influence]
    on_implement: List[ScriptAction]
    prereqs: List[str]


@dataclass(slots=True)
class DilemmaOption:
    name: str
    description: str
    on_implement: List[ScriptAction]


@dataclass(slots=True)
class DilemmaDefinition:
    name: str
    gui_name: str
    description: str
    influences: List[Influence]
    options: List[DilemmaOption]
    prereqs: List[str]


@dataclass(slots=True)
class AttackDefinition:
    name: str
    gui_name: str
    used_by: str
    attack_type: str  # PLOT | ASSASSINATION
    min_strength: float
    success_chance: float
    on_success: List[ScriptAction]
    on_failure: List[ScriptAction]
    prereqs: List[str]


@dataclass(slots=True)
class PressureGroup:
    name: str
    group_type: str  # PROTEST | EXTREMIST
    base_threat: float
    low_threat: float
    medium_threat: float
    high_threat: float
    required: List[str]
    radicalisation_rate: float
    deradicalisation_rate: float


def _parse_ini(text: str) -> Dict[str, Dict[str, str]]:
    """Parse the game's ``[section]`` / ``key = value`` task files."""

    sections: Dict[str, Dict[str, str]] = {}
    current: Optional[str] = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith(";"):
            continue
        if line.startswith("[") and line.endswith("]"):
            current = line[1:-1].strip()
            sections.setdefault(current, {})
            continue
        if current is None or "=" not in line:
            continue
        key, _, value = line.partition("=")
        sections[current][key.strip()] = value.strip().strip('"')
    return sections


def _parse_scripts(script: str) -> List[ScriptAction]:
    if not script:
        return []
    actions: List[ScriptAction] = []
    for match in SCRIPT_CALL_RE.finditer(script):
        name = match.group(1)
        args = [arg.strip() for arg in match.group(2).split(",") if arg.strip()]
        actions.append(ScriptAction(name=name, args=args))
    return actions


def _parse_influences(section: Optional[Dict[str, str]]) -> List[Influence]:
    influences: List[Influence] = []
    if not section:
        return influences
    for key in sorted(section, key=lambda k: (len(k), k)):
        value = section[key]
        if key.startswith("_"):
            continue  # e.g. _default_ handled separately by dilemmas
        parts = [part.strip() for part in value.split(",") if part.strip()]
        if not parts:
            continue
        source = parts[0]
        expression = parts[1] if len(parts) > 1 else "0"
        influences.append(Influence(source=source, expression=expression))
    return influences


def _parse_prereqs(text: str) -> List[str]:
    """Parse the bare ``[prereqs]`` tokens that gate event/dilemma firing.

    The section lists required mission options such as ``MONARCHY``,
    ``FOXES``, ``HURRICANES`` or ``EARTHQUAKES``; ``_parse_ini`` skips lines
    without ``=``, so the tokens are read directly from the raw file text.
    """

    prereqs: List[str] = []
    in_section = False
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[") and line.endswith("]"):
            in_section = line[1:-1].strip().lower() == "prereqs"
            continue
        if not in_section or "=" in line:
            continue
        prereqs.append(line)
    return prereqs


def _load_events(root: Path) -> Dict[str, EventDefinition]:
    events: Dict[str, EventDefinition] = {}
    for path in sorted((root / "simulation" / "events").glob("*.txt")):
        text = path.read_text(encoding="latin-1")
        sections = _parse_ini(text)
        config = sections.get("config", {})
        name = config.get("Name") or path.stem
        events[name] = EventDefinition(
            name=name,
            gui_name=config.get("GUIName", name),
            description=config.get("Description", ""),
            influences=_parse_influences(sections.get("influences")),
            on_implement=_parse_scripts(config.get("OnImplement", "")),
            prereqs=_parse_prereqs(text),
        )
    return events


def _load_dilemmas(root: Path) -> Dict[str, DilemmaDefinition]:
    dilemmas: Dict[str, DilemmaDefinition] = {}
    for path in sorted((root / "simulation" / "dilemmas").glob("*.txt")):
        text = path.read_text(encoding="latin-1")
        sections = _parse_ini(text)
        config = sections.get("dilemma", {})
        name = config.get("name") or path.stem
        options: List[DilemmaOption] = []
        for index in range(4):
            option = sections.get(f"option{index}")
            if not option:
                continue
            options.append(
                DilemmaOption(
                    name=option.get("Name", f"Option {index}"),
                    description=option.get("Description", ""),
                    on_implement=_parse_scripts(option.get("OnImplement", "")),
                )
            )
        dilemmas[name] = DilemmaDefinition(
            name=name,
            gui_name=config.get("guiname", name),
            description=config.get("description", ""),
            influences=_parse_influences(sections.get("influences")),
            options=options,
            prereqs=_parse_prereqs(text),
        )
    return dilemmas


def _load_attacks(root: Path) -> Dict[str, AttackDefinition]:
    attacks: Dict[str, AttackDefinition] = {}
    for path in sorted((root / "simulation" / "attacks").glob("*.txt")):
        text = path.read_text(encoding="latin-1")
        sections = _parse_ini(text)
        config = sections.get("config", {})
        name = config.get("Name") or path.stem
        # Attack prereqs name the plot that must have fired first (for
        # example an assassination requires its matching plot); they are bare
        # tokens, so they use the same raw-section parse as events/dilemmas.
        prereqs = _parse_prereqs(text)
        attacks[name] = AttackDefinition(
            name=name,
            gui_name=config.get("GUIName", name),
            used_by=config.get("UsedBy", ""),
            attack_type=config.get("Type", "PLOT").upper(),
            min_strength=_parse_float(config.get("MinStrength", 0.0)),
            success_chance=_parse_float(config.get("SuccessChance", 1.0)),
            on_success=_parse_scripts(config.get("OnSuccess", "")),
            on_failure=_parse_scripts(config.get("OnFailure", "")),
            prereqs=prereqs,
        )
    return attacks


def _load_pressure_groups(root: Path) -> Dict[str, PressureGroup]:
    groups: Dict[str, PressureGroup] = {}
    path = root / "simulation" / "pressuregroups.csv"
    if not path.exists():
        return groups
    with path.open(encoding="latin-1", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 8 or not row[1] or row[0].strip() == "token":
                continue
            required = [part.strip() for part in row[11].split(",") if part.strip()]
            groups[row[1].strip()] = PressureGroup(
                name=row[1].strip(),
                group_type=row[2].strip().upper(),
                base_threat=_parse_float(row[7]),
                low_threat=_parse_float(row[8]),
                medium_threat=_parse_float(row[9]),
                high_threat=_parse_float(row[10]),
                required=required,
                radicalisation_rate=_parse_float(row[12]),
                deradicalisation_rate=_parse_float(row[13]),
            )
    return groups


def _parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _country_options(data: SimulationData, country: str) -> set[str]:
    """Return the mission's enabled ``[options]`` as an uppercase set.

    The Democracy 3 mission files gate option-scoped events and dilemmas
    through these tokens (``MONARCHY``, ``FOXES``, ``HURRICANES``,
    ``EARTHQUAKES``).  Events without prereqs fire for every country.
    """

    setup = data_loader.load_country_setup(data.gamedata_root, country)
    return {option.strip().upper() for option in setup.options if option.strip()}


def _prereqs_satisfied(
    prereqs: Sequence[str], options: set[str]
) -> bool:
    return all(prereq.upper() in options for prereq in prereqs)


def _load_country_scripts(root: Path, country: str) -> List[ScriptAction]:
    """Parse the mission's ``scripts/*.txt`` into ``CreateGrudge`` actions.

    These scripted frequency/neuron offsets are the game's ``CreateGrudge``
    calls applied at mission load.  They are already baked into a serialized
    initial save, so ``apply_country_scripts`` only feeds them to states that
    are synthesized without a reference save.
    """

    scripts_dir = root / "missions" / country / "scripts"
    if not scripts_dir.is_dir():
        return []
    actions: List[ScriptAction] = []
    for path in sorted(scripts_dir.glob("*.txt")):
        actions.extend(_parse_scripts(path.read_text(encoding="latin-1")))
    return actions


def apply_country_scripts(
    state: SimulationState, data: SimulationData
) -> SimulationState:
    """Apply the mission's scripted grudges to a synthesized initial state."""

    actions = _load_country_scripts(data.gamedata_root, state.country)
    log: List[str] = []
    for action in actions:
        if action.name == "CreateGrudge":
            _apply_grudge(state, action.args, SimulationConfig(), log)
    return state


def _seed_int(seed: int, state: SimulationState, salt: int) -> int:
    """Derive a per-turn RNG seed that is stable across processes.

    ``hash()`` is salted per interpreter, so it cannot appear in the seed.
    """

    country = sum(ord(ch) for ch in state.country)
    return (seed * 7919) ^ (country << 8) ^ ((state.turn & 0xFFFF) << 16) ^ salt


def _event_chance(definition: EventDefinition, state: SimulationState) -> float:
    """Approximate the trigger chance: base ``_random_`` constant plus the
    evaluated condition influences, clamped to [0, 1]."""

    base = 0.0
    modifiers = 0.0
    context = {**state.values, **state.policies, **state.situations}
    for influence in definition.influences:
        if influence.source == "_random_":
            base = _parse_float(influence.expression)
            continue
        try:
            value = evaluate_expression(
                influence.expression,
                context.get(influence.source, 0.0),
                context=context,
            )
        except Exception:
            value = 0.0
        modifiers += value
    return _clamp(base + modifiers, 0.0, 1.0)


def _voter_target(state: SimulationState, target: str) -> Optional[str]:
    """Resolve a grudge target to a voter field, if it is one."""

    if target in state.voter_values:
        return "voter_values"
    if target in state.voter_frequencies:
        return "voter_frequencies"
    if target in state.voter_percentages:
        return "voter_percentages"
    return None


def _apply_grudge(
    state: SimulationState,
    args: List[str],
    config: SimulationConfig,
    log: List[str],
) -> None:
    """Record one ``CreateGrudge(guiname, id, target, value, decay)`` call.

    Native grudges are neural inputs with their own decay ring.  Recording the
    input here lets the simulator apply it during the same turn's neural pass
    and preserves it for later turns.  The old direct-value mutation made
    event effects disappear as soon as the target neuron was recalculated.
    """

    if len(args) < 4:
        return
    target = args[2]
    try:
        value = float(args[3])
    except (TypeError, ValueError):
        return
    try:
        decay = float(args[4]) if len(args) > 4 else 1.0
    except (TypeError, ValueError):
        decay = 1.0
    source = args[1] if len(args) > 1 else ""
    gui_name = args[0] if args else ""
    normalized_target = "_All_" if target == "_all_" else target
    if normalized_target.endswith("_freq"):
        # Frequency neurons retain the historical aggregate input path.  The
        # ordinary grudge list intentionally excludes these records so the
        # simulator does not add them twice.
        state.voter_frequency_grudges[normalized_target] = (
            state.voter_frequency_grudges.get(normalized_target, 0.0) + value
        )
        log.append(f"grudge {normalized_target} {value:+.3f}")
        return
    state.grudges.append(
        Grudge(
            target=normalized_target,
            value=value,
            decay=decay,
            source=source,
            gui_name=gui_name,
        )
    )
    log.append(f"grudge {normalized_target} {value:+.3f}")


def _run_script_actions(
    state: SimulationState,
    actions: List[ScriptAction],
    config: SimulationConfig,
    log: List[str],
    events: Optional[Dict[str, EventDefinition]] = None,
) -> bool:
    """Run an OnImplement/OnSuccess/OnFailure script.  Returns True if the
    game should stop (GameOver)."""

    if events is None:
        events = _load_events(Path(DEFAULT_GAMEDATA))
    for action in actions:
        if action.name == "CreateGrudge":
            _apply_grudge(state, action.args, config, log)
        elif action.name == "TriggerEvent":
            event = events.get(action.args[0]) if action.args else None
            if event:
                _run_script_actions(state, event.on_implement, config, log, events)
        elif action.name == "GameOver":
            log.append(f"GAME OVER: {action.args[0] if action.args else 'unknown'}")
            return True
        elif action.name in ("CreatePolitician", "FireMinister"):
            # Minister lifecycle scripts are stubbed: they alter the cabinet
            # roster, which this simulator does not model.
            continue
    return False


def run_events(
    state: SimulationState,
    data: SimulationData,
    config: SimulationConfig,
) -> SimulationState:
    """Roll each random event and apply its OnImplement effects."""

    rng = random.Random(_seed_int(config.random_seed, state, 0xE9))
    events = _load_events(data.gamedata_root)
    options = _country_options(data, state.country)
    for event in events.values():
        if not _prereqs_satisfied(event.prereqs, options):
            continue
        chance = _event_chance(event, state)
        if chance <= 0.0 or rng.random() >= chance:
            continue
        state.event_log.append(f"event {event.name} ({chance:.2f})")
        _run_script_actions(state, event.on_implement, config, state.event_log, events)
    return state


def run_dilemmas(
    state: SimulationState,
    data: SimulationData,
    config: SimulationConfig,
) -> SimulationState:
    """Trigger dilemmas whose latent influence sum crosses the threshold and
    resolve them with a seeded choice between the available options."""

    rng = random.Random(_seed_int(config.random_seed, state, 0xD1))
    dilemmas = _load_dilemmas(data.gamedata_root)
    options = _country_options(data, state.country)
    for dilemma in dilemmas.values():
        if not dilemma.options:
            continue
        if not _prereqs_satisfied(dilemma.prereqs, options):
            continue
        trigger = 0.0
        context = {**state.values, **state.policies, **state.situations}
        for influence in dilemma.influences:
            if influence.source == "_default_":
                try:
                    trigger += evaluate_expression(
                        influence.expression, 0.0, context=context
                    )
                except Exception:
                    pass
                continue
            try:
                trigger += evaluate_expression(
                    influence.expression,
                    context.get(influence.source, 0.0),
                    context=context,
                )
            except Exception:
                pass
        if trigger < 0.5:
            continue
        option = dilemma.options[rng.randrange(len(dilemma.options))]
        state.event_log.append(f"dilemma {dilemma.name}: {option.name}")
        _run_script_actions(state, option.on_implement, config, state.event_log)
    return state


def _group_strength(state: SimulationState, required: List[str]) -> float:
    """Approximate an extremist group's strength (0-100) from the support of
    the voter types it draws on."""

    strength = 0.0
    for voter in required:
        frequency = state.voter_frequencies.get(f"{voter}_freq")
        if frequency is None:
            frequency = max(0.0, state.voter_values.get(voter, 0.0))
        strength = max(strength, _clamp(frequency, 0.0, 1.0) * 100.0)
    return strength


def run_attacks(
    state: SimulationState,
    data: SimulationData,
    config: SimulationConfig,
) -> SimulationState:
    """Extremist pressure groups launch plots and, once a plot has fired,
    assassination attempts against the government."""

    if not (config.assassinations or config.pressure_group_events):
        return state
    rng = random.Random(_seed_int(config.random_seed, state, 0xA7))
    groups = _load_pressure_groups(data.gamedata_root)
    attacks = _load_attacks(data.gamedata_root)
    fired_plots = set(getattr(state, "fired_plots", []))
    for attack in sorted(attacks.values(), key=lambda a: a.name):
        group = groups.get(attack.used_by)
        if group is None:
            continue
        strength = _group_strength(state, group.required)
        if strength < attack.min_strength:
            continue
        if attack.attack_type == "ASSASSINATION":
            if not attack.prereqs or not all(prereq in fired_plots for prereq in attack.prereqs):
                continue
            if rng.random() >= attack.success_chance:
                state.event_log.append(
                    f"attack {attack.name} FAILED (strength {strength:.0f})"
                )
                _run_script_actions(state, attack.on_failure, config, state.event_log)
                continue
            state.event_log.append(f"attack {attack.name} SUCCEEDED")
            _run_script_actions(state, attack.on_success, config, state.event_log)
        else:  # PLOT
            state.event_log.append(f"plot {attack.name} (strength {strength:.0f})")
            fired_plots.add(attack.name)
            _run_script_actions(state, attack.on_success, config, state.event_log)
    state.fired_plots = sorted(fired_plots)
    return state


def run_pressure_groups(
    state: SimulationState,
    data: SimulationData,
    config: SimulationConfig,
) -> SimulationState:
    """Evolve protest-group threat from support and apply opinion drift when
    a group turns militant."""

    if not config.pressure_group_events:
        return state
    rng = random.Random(_seed_int(config.random_seed, state, 0x70))
    groups = _load_pressure_groups(data.gamedata_root)
    threats = dict(getattr(state, "group_threats", {}))
    for group in groups.values():
        if group.group_type != "PROTEST":
            continue
        support = _group_strength(state, group.required) / 100.0
        threat = threats.get(group.name, group.base_threat)
        threat += group.radicalisation_rate * support
        threat -= group.deradicalisation_rate * threat
        threat = _clamp(threat, 0.0, 2.0)
        threats[group.name] = threat
        if threat >= group.high_threat and rng.random() < 0.5:
            for voter in group.required:
                if voter in state.voter_values:
                    state.voter_values[voter] = _clamp(
                        state.voter_values[voter] - 0.01, -1.0, 1.0
                    )
                    state.values[voter] = state.voter_values[voter]
            state.event_log.append(f"pressure {group.name} militant (threat {threat:.2f})")
    state.group_threats = threats
    return state


def run_random_systems(
    state: SimulationState,
    data: SimulationData,
    config: SimulationConfig,
) -> SimulationState:
    """Apply every enabled stochastic system in the game's rough turn order."""

    if config.random_events:
        run_events(state, data, config)
    if config.dilemmas:
        run_dilemmas(state, data, config)
    if config.pressure_group_events:
        run_pressure_groups(state, data, config)
    if config.assassinations:
        run_attacks(state, data, config)
    return state
