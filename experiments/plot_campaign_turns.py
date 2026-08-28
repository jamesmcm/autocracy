"""Plot campaign performance per turn for each country.

Reads ``reports/campaigns/chronos/<country>/<seed>/turns.jsonl.gz`` for the
forecaster lives and ``reports/campaigns/oracle/<country>/<seed>`` for the
perfect-case reference, then draws each country's poll rate against the turn
count with vertical lines at the recorded election boundaries.

Usage::

    uv run --with matplotlib python experiments/plot_campaign_turns.py
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHRONOS_ROOT = Path("reports/campaigns/chronos")
ORACLE_ROOT = Path("reports/campaigns/oracle")
OUT_DIR = Path("reports/plots")
DEFAULT_COUNTRIES = ("australia", "canada", "france", "germany", "usa")


def _load_turns(root: Path) -> list[dict]:
    if not root.exists():
        return []
    try:
        with gzip.open(str(root / "turns.jsonl.gz"), "rt") as handle:
            return [json.loads(line) for line in handle]
    except (EOFError, OSError, ValueError):
        # A life may still be streaming its gzip output; skip it rather than
        # failing the whole figure.
        return []


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


def _polls_and_elections(records: list[dict]) -> tuple[list[float], list[int]]:
    polls: list[float] = []
    election_turns: list[int] = []
    for record in records:
        polls.append(float(record.get("poll_end", 0.0)))
        if record.get("election_result"):
            election_turns.append(int(record["turn"]))
    return polls, election_turns


def _mean_series(series_list: list[list[float]]) -> list[float]:
    if not series_list:
        return []
    length = max(len(series) for series in series_list)
    return [
        sum(series[turn] for series in series_list if turn < len(series))
        / sum(1 for series in series_list if turn < len(series))
        for turn in range(length)
    ]


def plot_countries(countries: list[str], out_path: Path) -> None:
    rows = 2
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(16, 9), squeeze=False)
    for index, country in enumerate(countries):
        ax = axes[index // cols][index % cols]
        chronos_dirs = _seed_dirs(CHRONOS_ROOT, country)
        oracle_dirs = _seed_dirs(ORACLE_ROOT, country)

        all_polls: list[list[float]] = []
        colours = plt.get_cmap("tab10")
        for seed_index, life_dir in enumerate(chronos_dirs):
            records = _load_turns(life_dir)
            polls, election_turns = _polls_and_elections(records)
            all_polls.append(polls)
            turns = range(1, len(polls) + 1)
            ax.plot(
                turns,
                polls,
                color=colours(seed_index % 10),
                alpha=0.4,
                linewidth=0.9,
            )
            for election_turn in election_turns:
                ax.axvline(
                    election_turn,
                    color="grey",
                    linewidth=0.5,
                    alpha=0.5,
                    linestyle="--",
                )

        mean_polls = _mean_series(all_polls)
        if mean_polls:
            ax.plot(
                range(1, len(mean_polls) + 1),
                mean_polls,
                color="black",
                linewidth=2.4,
                label=f"chronos-2 mean ({len(chronos_dirs)} seeds)",
            )

        for life_dir in oracle_dirs:
            records = _load_turns(life_dir)
            polls, election_turns = _polls_and_elections(records)
            ax.plot(
                range(1, len(polls) + 1),
                polls,
                color="gold",
                linewidth=2.6,
                label="oracle",
            )
            for election_turn in election_turns:
                ax.axvline(
                    election_turn,
                    color="gold",
                    linewidth=0.8,
                    alpha=0.7,
                    linestyle="--",
                )

        ax.axhline(0.0, color="black", linewidth=0.6, linestyle=":", alpha=0.4)
        ax.set_title(country.upper(), fontsize=12)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Poll rate")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8, loc="upper right")

    for index in range(len(countries), rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.suptitle(
        "20-term campaigns: chronos-2 vs oracle, poll rate per turn "
        "(dashed = election boundaries)",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--countries",
        nargs="+",
        default=list(DEFAULT_COUNTRIES),
        help="Countries to plot (default: all five new countries).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "campaign_20terms_chronos2_vs_oracle_by_turn.png",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_countries(list(args.countries), args.out)
    print(f"wrote {args.out}")
    for country in args.countries:
        chronos = _seed_dirs(CHRONOS_ROOT, country)
        oracle = _seed_dirs(ORACLE_ROOT, country)
        total_terms = sum(
            len(json.loads((d / "summary.json").read_text(encoding="utf-8"))["margins"])
            for d in chronos
        )
        wins = 0
        for d in chronos:
            summary = json.loads((d / "summary.json").read_text(encoding="utf-8"))
            wins += sum(1 for margin in summary["margins"] if margin > 0)
        print(
            f"{country}: chronos {len(chronos)} seeds, {wins}/{total_terms} term-wins; "
            f"oracle {len(oracle)} run(s)"
        )


if __name__ == "__main__":
    main()