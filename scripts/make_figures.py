"""Render the README's results figures from the committed result files in `docs/results/`.

    .venv/bin/python scripts/make_figures.py          # writes docs/images/results/*.png

To regenerate the result files themselves (default preset, seeds 7, 13, 21; ~10 minutes, the
learning curve is the heavy part):

    fraudrings sweep --data-dir /tmp/r  && cp /tmp/r/sweep.json docs/results/
    fraudrings grade --data-dir /tmp/r  && cp /tmp/r/rule_grades.json docs/results/
    fraudrings rank  --data-dir /tmp/r  && cp /tmp/r/concentration.json docs/results/
    fraudrings curve --level sophisticated --asymmetric --data-dir /tmp/r \\
        && cp /tmp/r/learning_curve.json docs/results/

Needs the [docs] extra (matplotlib). Colors are the first two slots of a validated categorical
palette (CVD-safe), on a light surface; every panel has one y-axis and at most two series.
"""

from __future__ import annotations

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "docs", "results")
OUT = os.path.join(ROOT, "docs", "images", "results")

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#e4e3de"
BLUE = "#2a78d6"    # categorical slot 1
ORANGE = "#eb6834"  # categorical slot 2
LEVELS = ["naive", "moderate", "sophisticated"]


def _load(name: str) -> dict | list:
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)


def _style(ax, title: str, ylabel: str = "") -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=11, color=INK, fontweight="bold", pad=10)
    ax.set_ylabel(ylabel, color=INK_2, fontsize=9)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def _line(ax, xs, ys, color: str, label: str, end_label: bool = True) -> None:
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=2, label=label, zorder=3)
    if end_label:  # direct label at the line end, only where curves don't converge
        ax.annotate(label, (xs[-1], ys[-1]), xytext=(8, 0), textcoords="offset points",
                    va="center", fontsize=9, color=INK_2)


def fig_dial() -> str:
    sweep = {r["level"]: r for r in _load("sweep.json")["levels"]}
    grade = {r["level"]: r for r in _load("rule_grades.json")["levels"]}
    rank = {r["level"]: r for r in _load("concentration.json")}
    x = list(range(len(LEVELS)))

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), facecolor=SURFACE)

    ax = axes[0]
    _style(ax, "Seeded engine: ranking vs. what you'd action")
    _line(ax, x, [sweep[lv]["mean_ap"] for lv in LEVELS], BLUE, "per-ring AP")
    _line(ax, x, [sweep[lv]["operational_precision"] for lv in LEVELS], ORANGE, "operational precision")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower left", frameon=False, fontsize=9, labelcolor=INK_2)

    ax = axes[1]
    _style(ax, "Recall on held-out rings")
    w = 0.36
    base = [grade[lv]["baseline_recall_at_k"] for lv in LEVELS]
    rule = [grade[lv]["recall"] for lv in LEVELS]
    b1 = ax.bar([i - w / 2 - 0.01 for i in x], base, w, color=BLUE, label="hand-weighted score", zorder=3)
    b2 = ax.bar([i + w / 2 + 0.01 for i in x], rule, w, color=ORANGE, label="learned rule", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.2f}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                        xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8, color=INK_2)
    ax.set_ylim(0, 1.3)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=9, labelcolor=INK_2)

    ax = axes[2]
    _style(ax, "Seedless ranker: ring targets near the head")
    vals = [rank[lv]["recall_at_k"] for lv in LEVELS]
    bars = ax.bar(x, vals, 0.55, color=BLUE, zorder=3)
    for bar in bars:
        ax.annotate(f"{bar.get_height():.0%}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, color=INK_2)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("share in top k (k = 2 × #targets)", color=INK_2, fontsize=9)

    for ax in axes:
        ax.set_xticks(x, LEVELS)
        ax.set_xlim(-0.5, len(LEVELS) - 0.5 + (0.6 if ax is axes[0] else 0))
    fig.suptitle("Turning the evasion dial  ·  default preset, seeds 7, 13, 21", x=0.01, ha="left",
                 fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT, "dial.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def fig_learning_curve() -> str:
    curve = _load("learning_curve.json")
    by = {(s["strategy"], s["noise"]): s for s in curve["series"]}
    t = curve["targets"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), facecolor=SURFACE)
    series = [(("random", "ε=0"), BLUE, "random labels"), (("uncertainty", "ε=0"), ORANGE, "uncertainty sampling")]

    ax = axes[0]
    _style(ax, f"Recall vs. labels (bar: ≥ {t['recall']:.0%})")
    for key, color, label in series:
        pts = by[key]["points"]
        _line(ax, [p["n_labels"] for p in pts], [p["recall"] for p in pts], color, label, end_label=False)
    ax.axhline(t["recall"], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_ylim(0, 1.05)

    ax = axes[1]
    _style(ax, f"Legit false positives vs. labels (bar: ≤ {t['legit_fp']:.0%})")
    for key, color, label in series:
        pts = by[key]["points"]
        _line(ax, [p["n_labels"] for p in pts], [p["legit_fp_rate"] for p in pts], color, label, end_label=False)
    ax.axhline(t["legit_fp"], color=INK_2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_ylim(0, 0.3)

    for ax in axes:
        ax.set_xscale("log")
        ax.set_xlabel("oracle labels used to train the rule", color=INK_2, fontsize=9)
        ax.legend(loc="center right", frameon=False, fontsize=9, labelcolor=INK_2)
    fig.suptitle(f"Learning curve at {curve['level']}  ·  held-out rings, perfect oracle  ·  "
                 f"stable crossing: {by[('random', 'ε=0')]['labels_to_target']} labels (random)",
                 x=0.01, ha="left", fontsize=10, color=INK_2)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out = os.path.join(OUT, "learning_curve.png")
    fig.savefig(out, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return out


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    for path in (fig_dial(), fig_learning_curve()):
        print(f"wrote {os.path.relpath(path, ROOT)}")


if __name__ == "__main__":
    main()
