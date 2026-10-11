"""Export the app's Plotly figures as JSON for the static site: `docs/charts/<name>.json`.

    .venv/bin/python scripts/export_charts.py                  # default preset, seed 7 (the walkthrough world)
    .venv/bin/python scripts/export_charts.py --preset smoke --out /tmp/charts   # fast, for tests

Each file is `fig.to_json()` plus `{"plotlyjs": <version the export targets>, "source": <what it was
drawn from>}`. The world is built in a temp dir, so the export is deterministic and leaves no data
behind. The app's charts call the same data functions and chart builders the app calls, with the same
arguments; only the two results charts are new, and they are drawn from `docs/results/*.json`.

Needs the [viz] extra. The site build (`build_site.py`) checks each file against the pinned plotly.js.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import plotly.graph_objects as go
from plotly.offline import get_plotlyjs_version
from plotly.subplots import make_subplots

from fraudrings.config import ConcentrationConfig, preset
from fraudrings.viewer import charts as C
from fraudrings.viewer import data as D
from fraudrings.viewer import data_triage as R
from fraudrings.viewer.app import CHART_SET_CAP, _shared_window
from fraudrings.viewer.plotly_theme import NEUTRAL, RING, TEXT_DIM, register_plotly_template

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs" / "results"
LEVELS = ["naive", "moderate", "sophisticated"]
ORANGE = "#e0a05c"
BUCKET_SECONDS = 5 * 60  # the app's default time bucket
NAMES = [
    "triage-axes", "triage-contributors", "triage-weekly",
    "investigate-linkage", "investigate-actionmix",
    "temporal-occupancy", "results-dial", "results-learning-curve",
]


def _write(out: Path, name: str, fig: go.Figure, source: str) -> None:
    doc = json.loads(fig.to_json())
    doc.update({"plotlyjs": get_plotlyjs_version(), "source": source})
    path = out / f"{name}.json"
    path.write_text(json.dumps(doc, separators=(",", ":")) + "\n")
    print(f"wrote {path.name}  ({path.stat().st_size / 1000:.0f} KB)")


def _world(preset_name: str, level: str, seed: int, base: str, concurrency: int | None = None) -> D.World:
    return D.load_world(preset_name, level, seed, cotarget_on=True, base_dir=base,  # type: ignore[arg-type]
                        concurrency=concurrency)


# --- charts drawn from a seeded world ----------------------------------------------------------


def triage_and_investigate(out: Path, preset_name: str, seed: int, base: str) -> None:
    """The naive world, rank-1 suspect, and the creator-1 handoff seeds (app_triage.py, app.py)."""
    world = _world(preset_name, "naive", seed, base)
    tag = f"{preset_name}/naive seed={seed}"
    cfg = ConcentrationConfig(sort_by="suspicion", cap=10,  # type: ignore[arg-type]
                              impact_floor=preset(preset_name).concentration.impact_floor)
    s = R.suspects(world.con, cfg)[0]  # rank 1, as in `?rank=1`

    _write(out, "triage-axes", C.suspect_axis_figure(s.pct_neff, s.pct_booster, s.pct_burst7),
           f"{tag}, rank 1 (creator {s.target_id})")
    _write(out, "triage-contributors",
           C.suspect_contributor_figure(R.top_contributors(world.con, s.target_id, 20), R.account_ring(world)),
           f"{tag}, rank 1 (creator {s.target_id}), ring coloring on")
    _write(out, "triage-weekly", C.suspect_weekly_figure(R.weekly_qv(world.con, s.target_id)),
           f"{tag}, rank 1 (creator {s.target_id})")

    # `?creator=<rank 1>`: the dedicated core's first five accounts become the seeds.
    seeds = list(s.core)[:5]
    cands = D.investigate(world, seeds)
    set_ids = list(dict.fromkeys([*seeds, *(c.account_id for c in cands)]))[:CHART_SET_CAP]
    src = f"{tag}, creator {s.target_id} handoff ({len(set_ids)} accounts)"
    nodes, edges = D.linkage_graph(world, set_ids)
    _write(out, "investigate-linkage", C.linkage_graph_figure(nodes, edges), src)
    _write(out, "investigate-actionmix", C.action_composition_figure(D.action_composition(world, set_ids)), src)


def temporal(out: Path, preset_name: str, seed: int, base: str) -> None:
    """The sophisticated, conc=4 world, ring 1 beside a bandwagon cluster on one shared time axis
    (app.py `_render_temporal`, ground truth on). The app draws two panels; here they share one plot."""
    world = _world(preset_name, "sophisticated", seed, base, concurrency=4)
    ring_id = D.rings(world)[0].ring_id if preset_name != "default" else 1
    ring_mem = D.ring_members(world, ring_id)
    clusters = D.bandwagon_clusters(world)
    if not clusters and preset_name == "default":
        raise SystemExit("the default world should have a bandwagon cluster to compare against")
    members = {f"ring {ring_id} (serialized)": (ring_mem, RING)}
    if clusters:
        members[f"bandwagon {clusters[0].cluster_id} (uncapped)"] = (
            D.bandwagon_members(world, clusters[0].cluster_id), NEUTRAL)
    t0, t1 = _shared_window(world, *(m for m, _ in members.values())) if clusters else (None, None)
    fig = go.Figure()
    for i, (title, (accounts, color)) in enumerate(members.items()):
        rd = D.account_raster(world, accounts, bucket_seconds=BUCKET_SECONDS, t0=t0, t1=t1)
        panel = C.occupancy_figure(rd, world.concurrency, color)
        trace = panel.data[0]
        trace.name, trace.showlegend = title, True
        fig.add_trace(trace)
        if i == 0:
            fig.layout.shapes, fig.layout.annotations = panel.layout.shapes, panel.layout.annotations
            fig.layout.xaxis.title, fig.layout.yaxis.title = panel.layout.xaxis.title, panel.layout.yaxis.title
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=30), showlegend=True,
                      legend=dict(orientation="h", y=1.1, x=0))
    _write(out, "temporal-occupancy", fig, f"{preset_name}/sophisticated conc=4 seed={seed}, ring {ring_id}")


# --- charts drawn from committed results -------------------------------------------------------


def _load(name: str):
    return json.loads((RESULTS / name).read_text())


def _bars(fig: go.Figure, row: int, col: int, x: list[str], series: list[tuple[str, list[float], str]],
          fmt: str) -> None:
    for name, ys, color in series:
        fig.add_trace(go.Bar(x=x, y=ys, name=name, marker_color=color, text=[format(y, fmt) for y in ys],
                             textposition="outside", cliponaxis=False,
                             hovertemplate=f"{name}: %{{y:{fmt}}}<extra>%{{x}}</extra>",
                             showlegend=row == 1 and col == 2 and len(series) > 1), row=row, col=col)


def results_dial() -> go.Figure:
    """Same data and panels as `make_figures.fig_dial`: `sweep.json`, `rule_grades.json`, `concentration.json`."""
    sweep = {r["level"]: r for r in _load("sweep.json")["levels"]}
    grade = {r["level"]: r for r in _load("rule_grades.json")["levels"]}
    rank = {r["level"]: r for r in _load("concentration.json")}
    fig = make_subplots(rows=1, cols=3, horizontal_spacing=0.07, subplot_titles=(
        "Ranking vs. what you'd action", "Recall on held-out rings", "Seedless: targets in the top k"))
    lines = (("per-ring AP", "mean_ap", NEUTRAL), ("operational precision", "operational_precision", ORANGE))
    for name, key, color in lines:
        fig.add_trace(go.Scatter(
            x=LEVELS, y=[sweep[lv][key] for lv in LEVELS], name=name, mode="lines+markers",
            line=dict(color=color, width=2), hovertemplate=f"{name}: %{{y:.2f}}<extra>%{{x}}</extra>"), row=1, col=1)
    _bars(fig, 1, 2, LEVELS, [
        ("hand-weighted score", [grade[lv]["baseline_recall_at_k"] for lv in LEVELS], NEUTRAL),
        ("learned rule", [grade[lv]["recall"] for lv in LEVELS], ORANGE)], ".2f")
    _bars(fig, 1, 3, LEVELS, [("share in top k", [rank[lv]["recall_at_k"] for lv in LEVELS], NEUTRAL)], ".0%")
    fig.update_yaxes(range=[0, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.3], tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0], row=1, col=2)
    fig.update_yaxes(range=[0, 1.15], tickformat=".0%", row=1, col=3)
    fig.update_annotations(font_size=12)  # the subplot titles
    fig.update_layout(height=420, barmode="group", margin=dict(l=40, r=10, t=50, b=40),
                      legend=dict(orientation="h", y=-0.15, x=0))
    return fig


def results_learning_curve() -> go.Figure:
    """Mirrors `make_figures.fig_learning_curve`: `learning_curve.json`."""
    curve = _load("learning_curve.json")
    by = {(s["strategy"], s["noise"]): s for s in curve["series"]}
    t = curve["targets"]
    fig = make_subplots(rows=1, cols=2, horizontal_spacing=0.08, subplot_titles=(
        f"Recall vs. labels (bar: ≥ {t['recall']:.0%})",
        f"Legit false positives vs. labels (bar: ≤ {t['legit_fp']:.0%})"))
    for key, color, label in ((("random", "ε=0"), NEUTRAL, "random labels"),
                              (("uncertainty", "ε=0"), ORANGE, "uncertainty sampling")):
        pts = by[key]["points"]
        for col, field in ((1, "recall"), (2, "legit_fp_rate")):
            fig.add_trace(go.Scatter(
                x=[p["n_labels"] for p in pts], y=[p[field] for p in pts], name=label, mode="lines+markers",
                line=dict(color=color, width=2), showlegend=col == 1,
                hovertemplate=f"{label}: %{{y:.1%}} at %{{x}} labels<extra></extra>"), row=1, col=col)
    for col, level in ((1, t["recall"]), (2, t["legit_fp"])):
        fig.add_hline(y=level, line=dict(color=TEXT_DIM, dash="dash", width=1), row=1, col=col)
    fig.update_annotations(font_size=12)
    fig.update_xaxes(type="log", title_text="oracle labels used to train the rule")
    fig.update_yaxes(range=[0, 1.05], tickformat=".0%", row=1, col=1)
    fig.update_yaxes(range=[0, 0.3], tickformat=".0%", row=1, col=2)
    fig.update_layout(height=400, margin=dict(l=40, r=10, t=50, b=50), legend=dict(orientation="h", y=-0.22, x=0))
    return fig


def export(out: Path, preset_name: str = "default", seed: int = 7) -> list[str]:
    if get_plotlyjs_version() != "4.1.1":
        print(f"note: this plotly bundles plotly.js {get_plotlyjs_version()}; "
              "site/site.toml's plotly_version must match")
    register_plotly_template()
    out.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="fraudrings-charts-") as base:
        triage_and_investigate(out, preset_name, seed, os.path.join(base, "data"))
        temporal(out, preset_name, seed, os.path.join(base, "data"))
    _write(out, "results-dial", results_dial(), "docs/results/sweep.json, rule_grades.json, concentration.json")
    _write(out, "results-learning-curve", results_learning_curve(), "docs/results/learning_curve.json")
    return NAMES


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preset", default="default")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "charts")
    args = ap.parse_args()
    export(args.out, args.preset, args.seed)


if __name__ == "__main__":
    main()
