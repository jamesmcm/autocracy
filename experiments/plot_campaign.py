"""Plot the 20-election full-trace campaigns.

Reads ``reports/campaigns/chronos/<seed>/summary.json`` for every seed and
``reports/campaigns/oracle/<seed>/summary.json`` for the perfect-case run,
then draws:

* election margins by election index (thin per-seed lines, bold mean, and
  the oracle line overlaid as the reference);
* mean poll per election, same layout.

Usage::

    uv run --with matplotlib --extra chronos python experiments/plot_campaign.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CHRONOS_ROOT = Path("reports/campaigns/chronos")
ORACLE_ROOT = Path("reports/campaigns/oracle")
OUT_DIR = Path("reports/plots")


def _load_summaries(root: Path) -> list[dict]:
    if not root.exists():
        return []
    return [
        json.loads((summary_dir / "summary.json").read_text(encoding="utf-8"))
        for summary_dir in sorted(root.iterdir(), key=lambda p: int(p.name))
        if (summary_dir / "summary.json").exists()
    ]


def plot_campaigns(chronos: list[dict], oracle: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)

    margins_ax, polls_ax = axes
    colours = plt.get_cmap("tab10")

    max_terms = max(
        (len(summary["margins"]) for summary in chronos), default=1
    )
    for index, summary in enumerate(chronos):
        terms = range(1, len(summary["margins"]) + 1)
        margins_ax.plot(
            terms,
            summary["margins"],
            color=colours(index),
            alpha=0.45,
            linewidth=1.1,
        )
        polls_ax.plot(
            terms,
            summary["term_mean_polls"],
            color=colours(index),
            alpha=0.45,
            linewidth=1.1,
        )

    mean_margins = [
        sum(s["margins"][term] for s in chronos if term < len(s["margins"]))
        / sum(1 for s in chronos if term < len(s["margins"]))
        for term in range(max_terms)
    ]
    mean_polls = [
        sum(s["term_mean_polls"][term] for s in chronos if term < len(s["term_mean_polls"]))
        / sum(1 for s in chronos if term < len(s["term_mean_polls"]))
        for term in range(max_terms)
    ]
    margins_ax.plot(
        range(1, max_terms + 1),
        mean_margins,
        color="black",
        linewidth=2.6,
        label=f"chronos-2 (mean, {len(chronos)} seeds)",
    )
    polls_ax.plot(
        range(1, max_terms + 1),
        mean_polls,
        color="black",
        linewidth=2.6,
    )

    for summary in oracle:
        terms = range(1, len(summary["margins"]) + 1)
        margins_ax.plot(
            terms,
            summary["margins"],
            color="gold",
            linewidth=2.8,
            linestyle="-",
            marker="o",
            markersize=4,
            label="simulator oracle (perfect case)",
        )
        polls_ax.plot(
            terms,
            summary["term_mean_polls"],
            color="gold",
            linewidth=2.8,
            linestyle="-",
            marker="o",
            markersize=4,
        )

    for ax, ylabel, title in (
        (margins_ax, "Election margin (player − opposition votes)", "Election margins"),
        (polls_ax, "Mean poll rate within the term", "Mean poll rate"),
    ):
        ax.axhline(0.0, color="black", linewidth=0.7, linestyle="--", alpha=0.5)
        ax.grid(alpha=0.25)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)
    polls_ax.set_xlabel("Election index (20 elections × 16 turns)")
    margins_ax.legend(fontsize=9, loc="lower right")
    fig.suptitle(
        "20-election campaigns: chronos-2 (120M) across 10 seeds vs simulator oracle",
        fontsize=13,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    chronos = _load_summaries(CHRONOS_ROOT)
    oracle = _load_summaries(ORACLE_ROOT)
    if not chronos:
        raise SystemExit("no chronos campaign summaries found")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "campaign_20elections_chronos2_vs_oracle.png"
    plot_campaigns(chronos, oracle, out_path)
    print(f"wrote {out_path}")
    total_terms = sum(len(s["margins"]) for s in chronos)
    total_wins = sum(
        1 for s in chronos for margin in s["margins"] if margin > 0
    )
    print(
        f"chronos: {len(chronos)} seeds, {total_wins}/{total_terms} term-wins"
    )
    if oracle:
        o = oracle[0]
        print(
            f"oracle: {len(o['margins'])} elections, "
            f"wins={sum(1 for m in o['margins'] if m > 0)}"
        )


if __name__ == "__main__":
    main()