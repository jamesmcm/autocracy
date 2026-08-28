"""Analyze campaign traces: actions, debt, and crisis behaviour.

Aggregates the recorded per-turn actions and end-of-turn metrics from
``reports/campaigns/<mode>/<country>/<seed>`` into a compact JSON and prints
a per-country comparison of the actions each agent chooses, the debt/crisis
trajectory, and (for chronos) the per-seed election margins.
"""

from __future__ import annotations

import gzip
import json
from collections import Counter
from pathlib import Path

CHRONOS_ROOT = Path("reports/campaigns/chronos")
ORACLE_ROOT = Path("reports/campaigns/oracle")
NOOP_ROOT = Path("reports/campaigns/noop")
CACHE_ROOT = Path("reports/campaigns/_cache")
COUNTRIES = ("uk", "australia", "canada", "france", "germany", "usa")


def _seed_dirs(root: Path, country: str) -> list[Path]:
    base = root / country
    if not base.exists():
        return []
    return sorted(
        (
            path
            for path in base.iterdir()
            if path.is_dir() and (path / "summary.json").exists()
        ),
        key=lambda p: int(p.name),
    )


def _life_analysis(root: Path) -> dict:
    """Return a small per-life summary without re-parsing the big states."""

    cache_path = (
        CACHE_ROOT / root.relative_to(Path("reports/campaigns")) / "life.json"
    )
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    actions: Counter[str] = Counter()
    action_ups: Counter[str] = Counter()
    action_downs: Counter[str] = Counter()
    debt_by_turn: list[float] = []
    poll_by_turn: list[float] = []
    capital_end_by_turn: list[float] = []
    expenditure_by_turn: list[float] = []
    election_turns: list[dict] = []
    with gzip.open(str(root / "turns.jsonl.gz"), "rt") as handle:
        for line in handle:
            record = json.loads(line)
            for action in record.get("actions", []):
                name = action["policy_name"]
                actions[name] += 1
                if action["delta"] > 0:
                    action_ups[name] += 1
                elif action["delta"] < 0:
                    action_downs[name] += 1
            debt_by_turn.append(float(record.get("state", {}).get("debt", 0.0)))
            poll_by_turn.append(float(record.get("poll_end", 0.0)))
            capital_end_by_turn.append(float(record.get("capital_end", 0.0)))
            expenditure_by_turn.append(float(record.get("expenditure_end", 0.0)))
            if record.get("election_result"):
                election_turns.append(
                    {
                        "turn": int(record["turn"]),
                        "result": record["election_result"],
                        "player": int(record.get("player_votes", 0)),
                        "opposition": int(record.get("opposition_votes", 0)),
                    }
                )
    analysis = {
        "actions": dict(actions),
        "action_ups": dict(action_ups),
        "action_downs": dict(action_downs),
        "debt_start": debt_by_turn[0] if debt_by_turn else 0.0,
        "debt_end": debt_by_turn[-1] if debt_by_turn else 0.0,
        "debt_max": max(debt_by_turn) if debt_by_turn else 0.0,
        "mean_poll": sum(poll_by_turn) / len(poll_by_turn) if poll_by_turn else 0.0,
        "mean_capital_end": (
            sum(capital_end_by_turn) / len(capital_end_by_turn)
            if capital_end_by_turn
            else 0.0
        ),
        "mean_expenditure_end": (
            sum(expenditure_by_turn) / len(expenditure_by_turn)
            if expenditure_by_turn
            else 0.0
        ),
        "elections": election_turns,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(analysis), encoding="utf-8")
    return analysis


def top_actions(analysis: dict, n: int = 12) -> list[tuple[str, int]]:
    return Counter(analysis["actions"]).most_common(n)


def main() -> None:
    for country in COUNTRIES:
        print(f"\n===== {country.upper()} =====")
        for mode in ("chronos", "oracle", "noop"):
            root = CHRONOS_ROOT if mode == "chronos" else (
                ORACLE_ROOT if mode == "oracle" else NOOP_ROOT
            )
            dirs = _seed_dirs(root, country)
            if not dirs:
                continue
            analyses = [_life_analysis(d) for d in dirs]
            total_actions = sum(sum(a["actions"].values()) for a in analyses)
            mean_debt_end = sum(a["debt_end"] for a in analyses) / len(analyses)
            mean_debt_max = sum(a["debt_max"] for a in analyses) / len(analyses)
            mean_poll = sum(a["mean_poll"] for a in analyses) / len(analyses)
            crisis_terms = sum(
                s["term_crisis"].count(True)
                for d in dirs
                for s in [json.loads((d / "summary.json").read_text(encoding="utf-8"))]
            )
            print(
                f"[{mode:<8}] seeds={len(dirs)} actions={total_actions} "
                f"mean_poll={mean_poll:.3f} debt_end={mean_debt_end:,.0f} "
                f"debt_max={mean_debt_max:,.0f} crises={crisis_terms} terms"
            )
            combined = Counter()
            for a in analyses:
                combined.update(a["actions"])
            for name, count in combined.most_common(10):
                ups = sum(a["action_ups"].get(name, 0) for a in analyses)
                downs = sum(a["action_downs"].get(name, 0) for a in analyses)
                print(f"    {name:<28} {count:>4}  (up {ups}, down {downs})")


if __name__ == "__main__":
    main()