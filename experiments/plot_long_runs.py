"""Render the long-term keep-playing runs as plots.

Reads the four JSON files produced by running ``chronos_learning.py``
with ``--terms 10 --keep-playing --warmup-size 8`` for both forecasters
(``--model autogluon/chronos-2-small`` and ``autogluon/chronos-2``,
seeds 20260813-16 and 20260920-21) and draws:

* ``long_term_recovery_small.png`` - per-term election margins, one line
  per life, win/loss markers, showing the recovery arcs;
* ``long_term_small_vs_base.png`` - both forecasters side by side, thin
  per-seed lines under a bold cross-seed mean.

Usage::

    uv run --with matplotlib --extra chronos python \
        experiments/plot_long_runs.py [--plots-dir reports/plots]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL_FILES = {
    "chronos-2-small": [
        "reports/long_small_a.json",
        "reports/long_small_b.json",
    ],
    "chronos-2": [
        "reports/long_base_a.json",
        "reports/long_base_b.json",
    ],
}


def load_lives(model: str) -> list[dict]:
    lives: list[dict] = []
    for name in MODEL_FILES[model]:
        path = Path(name)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        lives.extend(payload["runs"])
    lives.sort(key=lambda life: life["seed"])
    return lives


def plot_recovery(lives: list[dict], out_path: Path, model: str) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))
    colours = plt.get_cmap("tab10")
    for index, life in enumerate(lives):
        terms = range(1, len(life["margins"]) + 1)
        wins = [m > 0 for m in life["margins"]]
        ax.plot(
            terms,
            life["margins"],
            color=colours(index),
            alpha=0.75,
            linewidth=1.6,
            label=f"seed {life['seed']} ({sum(wins)}W-{len(wins) - sum(wins)}L)",
        )
        ax.scatter(
            [t for t, w in zip(terms, wins) if w],
            [m for m, w in zip(life["margins"], wins) if w],
            color=colours(index),
            marker="o",
            s=28,
            zorder=3,
        )
        ax.scatter(
            [t for t, w in zip(terms, wins) if not w],
            [m for m, w in zip(life["margins"], wins) if not w],
            color=colours(index),
            marker="x",
            s=42,
            zorder=3,
        )
    ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.set_xlabel("Term (election boundary)")
    ax.set_ylabel("Election margin (player − opposition votes)")
    ax.set_title(
        f"Long-run keep-playing campaigns — {model}\n"
        "o won · x lost · gentle diverse warm-up, losses are non-terminal"
    )
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_model_comparison(
    lives_by_model: dict[str, list[dict]], out_path: Path
) -> None:
    models = list(lives_by_model)
    fig, axes = plt.subplots(
        1,
        len(models),
        figsize=(7 * len(models), 6),
        sharey=True,
    )
    if len(models) == 1:
        axes = [axes]
    for ax, model in zip(axes, models):
        lives = lives_by_model[model]
        colours = plt.get_cmap("tab10")
        max_len = max(len(life["margins"]) for life in lives)
        means = []
        for term in range(max_len):
            values = [
                life["margins"][term]
                for life in lives
                if term < len(life["margins"])
            ]
            means.append(sum(values) / len(values))
        for index, life in enumerate(lives):
            ax.plot(
                range(1, len(life["margins"]) + 1),
                life["margins"],
                color=colours(index),
                alpha=0.4,
                linewidth=1.2,
            )
        ax.plot(
            range(1, len(means) + 1),
            means,
            color="black",
            linewidth=2.8,
            label="cross-seed mean",
        )
        total_wins = sum(
            1 for life in lives for margin in life["margins"] if margin > 0
        )
        total_terms = sum(len(life["margins"]) for life in lives)
        first_wins = sum(
            1
            for life in lives
            if life["margins"][:1] and life["margins"][0] > 0
        )
        ax.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
        ax.set_xlabel("Term (election boundary)")
        ax.set_title(
            f"{model} — {total_wins}/{total_terms} terms won, "
            f"{first_wins}/{len(lives)} first-election wins"
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize=9, loc="lower right")
    axes[0].set_ylabel("Election margin (player − opposition votes)")
    fig.suptitle(
        "Long-run campaigns by forecaster (thin lines: seeds, bold: mean)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plots-dir", type=Path, default=Path("reports/plots"))
    args = parser.parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    lives_by_model = {model: load_lives(model) for model in MODEL_FILES}
    for model, lives in lives_by_model.items():
        if not lives:
            raise SystemExit(f"no long-run data found for {model}")
        out = args.plots_dir / (
            "long_term_recovery_"
            + model.replace("/", "_").replace("-", "_")
            + ".png"
        )
        plot_recovery(lives, out, model)
        print(f"wrote {out}")
    comparison = args.plots_dir / "long_term_small_vs_base.png"
    plot_model_comparison(lives_by_model, comparison)
    print(f"wrote {comparison}")

    for model, lives in lives_by_model.items():
        total_wins = sum(
            1 for life in lives for margin in life["margins"] if margin > 0
        )
        total_terms = sum(len(life["margins"]) for life in lives)
        print(
            f"{model}: {len(lives)} lives, {total_wins}/{total_terms} "
            "term-wins"
        )


if __name__ == "__main__":
    main()
