"""Machine-readable snapshots used to validate Democracy 3 parity.

The XML save files remain the source of truth.  This module turns the parts of
those saves that affect a simulation turn into a stable JSON-shaped snapshot,
so parity cases can be reviewed and consumed without teaching every caller
the save format.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from .models import EffectHistory
from .savegame import SaveGame, parse_savegame

PARITY_SCHEMA_VERSION = 1


def _sorted_values(values: Dict[str, Any]) -> Dict[str, Any]:
    return {name: values[name] for name in sorted(values)}


def _history_snapshot(history: EffectHistory) -> Dict[str, object]:
    return {
        "source": history.source,
        "target": history.target,
        "values": list(history.values),
    }


def snapshot_from_savegame(save: SaveGame) -> Dict[str, object]:
    """Return all currently extracted turn-state data from a save.

    In particular, ``effect_memory`` preserves the complete serialized ring
    for every effect.  The simulator may use a reduced current-effect view
    while its turn kernel is being brought to parity, but the extraction
    corpus must not throw away the information needed to finish that work.
    """

    return {
        "schema_version": PARITY_SCHEMA_VERSION,
        "country": save.country,
        "turn": save.turn,
        "political_capital": save.political_capital,
        "simvalues": _sorted_values(save.simvalues),
        "policies": _sorted_values(save.policies),
        "finance": {
            "policy_costs": _sorted_values(save.policy_costs),
            "policy_incomes": _sorted_values(save.policy_incomes),
            "total_expenditure": save.total_expenditure,
            "total_income": save.total_income,
        },
        "situations": {
            "values": _sorted_values(save.situations),
            "active": sorted(save.active_situations),
        },
        "global_economy": {
            "position": save.global_economy_position,
            "years": save.global_economy_years,
            "intensity": save.global_economy_intensity,
        },
        "hidden_values": _sorted_values(save.hidden_values),
        "inherited": _sorted_values(save.inherited_values),
        "voters": {
            "values": _sorted_values(save.voter_values),
            "percentages": _sorted_values(save.voter_percentages),
            "frequencies": _sorted_values(save.voter_frequencies),
        },
        "parties": {
            name: {
                "status": party.status,
                "party_type": party.party_type,
                "members_last_turn": party.members_last_turn,
                "member_history": list(party.member_history),
                "activist_history": list(party.activist_history),
            }
            for name, party in sorted(save.parties.items())
        },
        "policy_runtime": {
            "implementations": _sorted_values(save.policy_implementations),
            "active": _sorted_values(save.policy_active),
            "cost_multipliers": _sorted_values(save.policy_cost_multipliers),
            "income_multipliers": _sorted_values(save.policy_income_multipliers),
            "cost_scalars": _sorted_values(save.policy_cost_scalars),
            "income_scalars": _sorted_values(save.policy_income_scalars),
            "effect_throttles": _sorted_values(save.effect_throttles),
            "desired_throttles": _sorted_values(save.policy_desired_throttles),
            "ministerial_effectiveness": _sorted_values(
                save.ministerial_effectiveness
            ),
            "ministerial_competence": _sorted_values(save.ministerial_competence),
        },
        "effect_memory": {
            "record_count": len(save.effect_histories),
            "history_lengths": sorted(
                {len(history.values) for history in save.effect_histories}
            ),
            "records": [_history_snapshot(history) for history in save.effect_histories],
        },
    }


def extract_savegame_snapshot(path: str | Path) -> Dict[str, object]:
    """Parse a Democracy 3 save and return its parity snapshot."""

    return snapshot_from_savegame(parse_savegame(path))


def load_parity_corpus(path: str | Path) -> Dict[str, object]:
    """Load and minimally validate a JSON parity corpus."""

    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Parity corpus root must be a JSON object")
    if payload.get("schema_version") != PARITY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported parity corpus schema: {payload.get('schema_version')!r}"
        )
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError("Parity corpus must contain a non-empty 'cases' list")
    return payload


def _main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Extract a Democracy 3 save snapshot")
    parser.add_argument("save", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rendered = json.dumps(extract_savegame_snapshot(args.save), indent=2, sort_keys=True)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    _main()
