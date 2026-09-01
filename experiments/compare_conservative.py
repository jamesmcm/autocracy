"""Compare the conservative no-op-gate chronos agent against references.

Reads the per-turn traces under ``reports/campaigns`` and prints, per country,
a per-seed table of election win rate, mean poll, action count, and DebtCrisis
exposure for three profiles:

* ``chronos/conservative`` — the status-quo gate profile (experiment 1);
* ``chronos`` — the ungated baseline;
* ``noop`` — the passive reference.

Usage::

    uv run python experiments/compare_conservative.py [--elections 20]
"""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter
from pathlib import Path

ROOT = Path("reports/campaigns")
COUNTRIES = ("uk", "australia", "canada", "france", "germany", "usa")
PROFILES = (
    ("chronos/conservative", "gate"),
    ("chronos", "baseline"),
    ("noop", "noop"),
)


def _seed_dirs(root: Path, country: str) -> list[Path]:
    base = root / country
    if not base.exists():
        return []
    return sorted(
        (p for p in base.iterdir() if (p / "turns.jsonl.gz").is_file()),
        key=lambda p: int(p.name),
    )


def _life(path: Path, elections: int) -> dict | None:
    """Aggregate one life; return None for truncated (in-flight) traces."""

    try:
        return _life_inner(path, elections)
    except EOFError:
        return None


def _life_inner(path: Path, elections: int) -> dict:
    """Aggregate one life: wins, polls, actions, crises, margins."""

    wins = 0
    losses = 0
    margins: list[float] = []
    polls: list[float] = []
    actions = 0
    per_policy: Counter[str] = Counter()
    crisis_turns = 0
    max_crisis_streak = 0
    streak = 0
    with gzip.open(path, "rt") as handle:
        for line in handle:
            record = json.loads(line)
            polls.append(float(record["poll_end"]))
            for action in record.get("actions", ()):
                actions += 1
                sign = "+" if float(action["delta"]) >= 0 else "-"
                per_policy[f"{action['policy_name']} {sign}"] += 1
            state = record.get("state", {})
            if "DebtCrisis" in state.get("active_situations", []):
                crisis_turns += 1
                streak += 1
                max_crisis_streak = max(max_crisis_streak, streak)
            else:
                streak = 0
            if "election_result" in record:
                if record["election_result"] == "win":
                    wins += 1
                else:
                    losses += 1
                margins.append(
                    float(record["player_votes"]) - float(record["opposition_votes"])
                )
    return {
        "wins": wins,
        "losses": losses,
        "elections": wins + losses,
        "mean_poll": sum(polls) / len(polls) if polls else 0.0,
        "final_poll": polls[-1] if polls else 0.0,
        "actions": actions,
        "actions_per_term": actions / max(wins + losses, 1),
        "crisis_turns": crisis_turns,
        "max_crisis_streak": max_crisis_streak,
        "margins": margins[:elections],
        "top_actions": per_policy.most_common(5),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--elections", type=int, default=20)
    args = parser.parse_args()

    for country in COUNTRIES:
        rows = []
        for profile, label in PROFILES:
            for seed_dir in _seed_dirs(ROOT / profile, country):
                summary = _life(seed_dir / "turns.jsonl.gz", args.elections)
                if summary is not None:
                    rows.append((label, seed_dir.name, summary))
        if not rows:
            continue
        print(f"\n== {country}")
        # Per-profile aggregates.
        for label in {label for label, _x, _s in rows}:
            lives = [summary for lab, _seed, summary in rows if lab == label]
            n = len(lives)
            wins = sum(summary["wins"] for summary in lives)
            total = sum(summary["elections"] for summary in lives)
            mean_poll = sum(summary["mean_poll"] for summary in lives) / n
            acts = sum(summary["actions"] for summary in lives) / n
            crisis = sum(summary["crisis_turns"] for summary in lives) / n
            print(
                f"  {label:9s} seeds={n} wins={wins}/{total}"
                f" mean_poll={mean_poll:.3f} actions/life={acts:.0f}"
                f" crisis_turns/life={crisis:.1f}"
            )
        # Per-seed detail.
        for label, seed, s in rows:
            margin_txt = (
                f" margin_mean={sum(s['margins']) / len(s['margins']):+.0f}"
                if s["margins"]
                else ""
            )
            print(
                f"    {label:9s} {seed}: {s['wins']}/{s['elections']}"
                f" poll={s['mean_poll']:.3f} final={s['final_poll']:.3f}"
                f" actions={s['actions']} crisis={s['crisis_turns']}"
                f" streak_max={s['max_crisis_streak']}{margin_txt}"
            )


if __name__ == "__main__":
    main()
