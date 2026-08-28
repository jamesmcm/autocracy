"""Plot campaign performance per turn for every country.

Reads the chronos-2 lives, the simulator oracle, and the passive no-op
baseline from ``reports/campaigns/<mode>/<country>/<seed>`` (UK lives live
under ``reports/campaigns/<mode>/uk/``) and draws each country's poll rate
against the turn count with:

* vertical dashed lines at the recorded election boundaries;
* a horizontal dashed line at the 0.5 poll-rate win barrier;
* the oracle (gold), the passive no-op baseline (green), and the chronos-2
  per-seed lines plus the bold seed mean.

Usage::

    uv run --with matplotlib python experiments/plot_campaign_compare.py
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
NOOP_ROOT = Path("reports/campaigns/noop")
CACHE_ROOT = Path("reports/campaigns/_cache")
OUT_DIR = Path("reports/plots")
DEFAULT_COUNTRIES = ("uk", "australia", "canada", "france", "germany", "usa")


def _load_turns(root: Path) -> list[dict]:
    if not root.exists():
        return []
    try:
        with gzip.open(str(root / "turns.jsonl.gz"), "rt") as handle:
            return [json.loads(line) for line in handle]
    except (EOFError, OSError, ValueError):
        return []


def _polls_series(root: Path) -> list[dict]:
    """Return a lightweight ``[{"turn", "poll_end", "election_result"}]`` trace.

    The full per-turn states are large (thousands of voters), so cache the
    small trace beside the campaigns and only decompress once.
    """

    cache_path = CACHE_ROOT / root.relative_to(Path("reports/campaigns")) / "polls.json"
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    records = _load_turns(root)
    trace = [
        {
            "turn": int(record["turn"]),
            "poll_end": float(record.get("poll_end", 0.0)),
            "election_result": record.get("election_result"),
        }
        for record in records
    ]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(trace), encoding="utf-8")
    return trace


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


def _polls_and_elections(trace: list[dict]) -> tuple[list[float], list[int]]:
    polls: list[float] = []
    election_turns: list[int] = []
    for record in trace:
        polls.append(record["poll_end"])
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
    fig, axes = plt.subplots(rows, cols, figsize=(17, 10), squeeze=False)
    for index, country in enumerate(countries):
        ax = axes[index // cols][index % cols]
        chronos_dirs = _seed_dirs(CHRONOS_ROOT, country)
        oracle_dirs = _seed_dirs(ORACLE_ROOT, country)
        noop_dirs = _seed_dirs(NOOP_ROOT, country)

        all_polls: list[list[float]] = []
        colours = plt.get_cmap("tab10")
        for seed_index, life_dir in enumerate(chronos_dirs):
            polls, election_turns = _polls_and_elections(_polls_series(life_dir))
            all_polls.append(polls)
            turns = range(1, len(polls) + 1)
            ax.plot(
                turns,
                polls,
                color=colours(seed_index % 10),
                alpha=0.35,
                linewidth=0.9,
            )
            for election_turn in election_turns:
                ax.axvline(
                    election_turn,
                    color="grey",
                    linewidth=0.4,
                    alpha=0.4,
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
            polls, election_turns = _polls_and_elections(_polls_series(life_dir))
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
                    alpha=0.6,
                    linestyle="--",
                )

        for life_dir in noop_dirs:
            polls, _ = _polls_and_elections(_polls_series(life_dir))
            ax.plot(
                range(1, len(polls) + 1),
                polls,
                color="green",
                linewidth=1.8,
                linestyle="--",
                label="no-op baseline",
            )

        ax.axhline(
            0.5,
            color="red",
            linewidth=1.1,
            linestyle="--",
            alpha=0.7,
            label="win barrier",
        )
        ax.set_title(country.upper(), fontsize=12)
        ax.set_xlabel("Turn")
        ax.set_ylabel("Poll rate")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7, loc="upper right")

    for index in range(len(countries), rows * cols):
        axes[index // cols][index % cols].axis("off")

    fig.suptitle(
        "20-term campaigns: chronos-2 vs oracle vs no-op, poll rate per turn "
        "(dashed verticals = election boundaries, dashed horizontal = win barrier)",
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
        help="Countries to plot (default: UK plus the five new countries).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUT_DIR / "campaign_20terms_noop_oracle_by_turn.png",
    )
    args = parser.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_countries(list(args.countries), args.out)
    print(f"wrote {args.out}")
    for country in args.countries:
        chronos = _seed_dirs(CHRONOS_ROOT, country)
        oracle = _seed_dirs(ORACLE_ROOT, country)
        noop = _seed_dirs(NOOP_ROOT, country)

        def wins(dirs):
            return sum(
                1
                for d in dirs
                for m in json.loads((d / "summary.json").read_text(encoding="utf-8"))[
                    "margins"
                ]
                if m > 0
            )

        total = sum(
            len(json.loads((d / "summary.json").read_text(encoding="utf-8"))["margins"])
            for d in chronos
        )
        print(
            f"{country}: chronos {wins(chronos)}/{total} wins, "
            f"oracle {wins(oracle)}/{max(len(oracle)*20,1)}, "
            f"noop {wins(noop)}/{max(len(noop)*20,1)}"
        )


if __name__ == "__main__":
    main()