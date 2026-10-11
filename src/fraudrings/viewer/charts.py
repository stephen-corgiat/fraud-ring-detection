"""Plotly chart builders for the viewer (no streamlit import — unit-testable).

Each function turns a presentation-only data structure from `viewer.data` into a Plotly
`go.Figure`. The figures are the *spine*: a single charting library (Plotly) subsumes the
occupancy strip, the axis explainers, the EDA substrate (account age, registration burst,
cadence), and the lasso surface for select-to-reseed.

Anti-tautology boundary: charts are presentation-only — no
chart-derived signal feeds scoring. Per-account selectable surfaces carry the account id in
each point's `customdata` so a lasso/box selection maps straight back to account ids.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import networkx as nx
import plotly.graph_objects as go

from fraudrings.viewer.data import FUNNEL_ACTIONS
from fraudrings.viewer.raster import RasterData

# Palette shared with the rest of the viewer (ring = red, neutral = blue).
_RING = "#d65f5f"
_NEUTRAL = "#4c78a8"
_BASELINE = "#bbbbbb"
# Funnel actions get a colour ramp from "intent" (search) through "engaged" (save).
_ACTION_COLORS = {
    "search": "#9c755f", "view": "#4c78a8", "complete": "#59a14f",
    "skip": "#e8a87c", "save": "#b07aa1",
}


def occupancy_figure(rd: RasterData, cap: int | None, color: str = _NEUTRAL) -> go.Figure:
    """Occupancy strip: simultaneous active accounts per bucket, with the concurrency cap as a
    dashed rule. A serialized ring hugs the cap (flat plateau); a bandwagon spikes above it.
    (migrated from Altair so the whole spine is single-library.)"""
    minutes = [b * rd.bucket_seconds / 60.0 for b in range(rd.n_buckets)]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=minutes, y=rd.occupancy, fill="tozeroy", mode="lines",
            line=dict(color=color, width=1), fillcolor=color, opacity=0.6,
            name="simultaneous accounts", hovertemplate="%{x:.0f} min · %{y} active<extra></extra>",
        )
    )
    if cap is not None:
        fig.add_hline(
            y=cap, line=dict(color="red", dash="dash", width=1),
            annotation_text=f"cap C={cap}", annotation_position="top left",
        )
    fig.update_layout(
        height=180, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
        xaxis_title="minutes from window start", yaxis_title="simultaneous accounts",
    )
    return fig


def daily_activity_figure(rows: Sequence[Mapping[str, float]]) -> go.Figure:
    """Per-day views-vs-skips bars (day 0 = world's first event). Folded into the Plotly spine so
    the profile's activity chart is themed like the rest."""
    days = [r["day"] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=days, y=[r.get("views", 0) for r in rows], name="views",
                         marker_color="#5b8fd6"))
    fig.add_trace(go.Bar(x=days, y=[r.get("skips", 0) for r in rows], name="skips",
                         marker_color="#34465c"))
    fig.update_layout(
        barmode="stack", height=200, margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="day", yaxis_title="events",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return fig


def action_composition_figure(rows: list[dict[str, int]]) -> go.Figure:
    """Stacked per-account funnel composition (explains the action-mix axis). One bar per
    account; each segment is a funnel action. Selectable: every segment carries its account id."""
    accounts = [r["account_id"] for r in rows]
    labels = [str(a) for a in accounts]
    fig = go.Figure()
    for act in FUNNEL_ACTIONS:
        fig.add_trace(
            go.Bar(
                x=labels, y=[r.get(act, 0) for r in rows], name=act,
                marker_color=_ACTION_COLORS[act], customdata=accounts,
                hovertemplate=f"account %{{customdata}} · {act}=%{{y}}<extra></extra>",
            )
        )
    fig.update_layout(
        barmode="stack", height=320, margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="account", yaxis_title="event count",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    fig.update_xaxes(type="category")
    return fig


def position_histogram_figure(
    candidate: list[float], population: list[float], threshold: float
) -> go.Figure:
    """Reported view-position histogram with the payout threshold as a vline. The
    candidate set overlays a faint population baseline; a farming ring stacks just past the line."""
    fig = go.Figure()
    if population:
        fig.add_trace(
            go.Histogram(
                x=population, name="population", histnorm="probability density",
                marker_color=_BASELINE, opacity=0.55, nbinsx=40,
            )
        )
    fig.add_trace(
        go.Histogram(
            x=candidate, name="candidate set", histnorm="probability density",
            marker_color=_RING, opacity=0.65, nbinsx=40,
        )
    )
    fig.add_vline(
        x=threshold, line=dict(color="#8b929c", dash="dash", width=1),
        annotation_text=f"qualified ≥ {threshold:g}", annotation_position="top right",
    )
    fig.update_layout(
        barmode="overlay", height=300, margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="reported position %", yaxis_title="density",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return fig


def cotarget_heatmap_figure(
    accounts: list[int], creators: list[int], matrix: list[list[int]]
) -> go.Figure:
    """Account × top-creator qualified-view heatmap. A co-targeting ring lights a dense
    vertical block on the shared target creator(s); genuine fans scatter."""
    fig = go.Figure(
        go.Heatmap(
            z=matrix, x=[f"creator {a}" for a in creators], y=[str(a) for a in accounts],
            # dark at zero so empty cells recede into the dark theme; red where views concentrate
            colorscale=[[0.0, "#161b22"], [0.02, "#3a1d22"], [0.35, "#8c2f35"], [1.0, "#ff6b6b"]],
            colorbar=dict(title="q-views"),
            hovertemplate="account %{y} · %{x} · %{z} q-views<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(260, 22 * len(accounts) + 80), margin=dict(l=10, r=10, t=10, b=60),
        xaxis_title="top creators in the set", yaxis_title="account",
    )
    fig.update_yaxes(type="category")
    return fig


def linkage_graph_figure(nodes: list[int], edges: list[dict]) -> go.Figure:
    """Force-directed account–account linkage subgraph. Nodes are accounts; an edge is a
    shared in-gate rare identifier, weighted by summed rarity. Nodes carry their account id
    (selectable). A ring shows as a dense rare-infra clump; organic accounts stay isolated."""
    g = nx.Graph()
    g.add_nodes_from(nodes)
    for e in edges:
        g.add_edge(e["a"], e["b"], weight=e["weight"])
    # Deterministic layout (fixed seed) so the same world always renders the same shape. Linked
    # accounts are laid out on their own: left in, a few isolated accounts would claim the layout's
    # scale and crush a dense ring into a single dot. Isolates then sit evenly on an outer circle.
    linked = [n for n in nodes if g.degree(n) > 0]
    isolates = [n for n in nodes if g.degree(n) == 0]
    pos = {}
    if linked:
        sub = g.subgraph(linked)
        pos.update(nx.spring_layout(sub, seed=42, k=2.0 / max(len(linked), 1) ** 0.5, iterations=200))
    for i, n in enumerate(isolates):
        angle = 2 * math.pi * i / max(len(isolates), 1)
        pos[n] = (1.35 * math.cos(angle), 1.35 * math.sin(angle))

    fig = go.Figure()
    edge_x: list[float | None] = []
    edge_y: list[float | None] = []
    for e in edges:
        x0, y0 = pos[e["a"]]
        x1, y1 = pos[e["b"]]
        edge_x += [x0, x1, None]
        edge_y += [y0, y1, None]
    if edge_x:
        fig.add_trace(
            go.Scatter(
                x=edge_x, y=edge_y, mode="lines", line=dict(color="rgba(160,170,185,0.18)", width=1),
                hoverinfo="skip", showlegend=False,
            )
        )
    degree = dict(g.degree())
    max_deg = max(degree.values(), default=0) or 1
    labelled = len(nodes) <= 25  # per-node id labels only while they stay legible
    fig.add_trace(
        go.Scatter(
            x=[pos[n][0] for n in nodes], y=[pos[n][1] for n in nodes],
            mode="markers+text" if labelled else "markers",
            text=[str(n) for n in nodes], textposition="top center",
            textfont=dict(size=9), customdata=nodes,
            marker=dict(
                # sqrt-scaled and capped: a 50-member ring must stay a readable clump, not a blob
                size=[8 + 14 * (degree.get(n, 0) / max_deg) ** 0.5 for n in nodes],
                color=[degree.get(n, 0) for n in nodes], colorscale="Reds",
                line=dict(color="#333333", width=1),
            ),
            hovertemplate="account %{customdata} · %{marker.color} shared-id link(s)<extra></extra>",
            showlegend=False,
        )
    )
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
    )
    return fig


def registration_burst_figure(rows: list[dict[str, float]]) -> go.Figure:
    """In-window registration-offset histogram. A freshly-minted ring co-registers in a
    tight spike; aged accounts sit at large negative offsets (registered before the window)."""
    offsets = [r["offset_days"] for r in rows]
    in_win = [r["offset_days"] for r in rows if r["in_window"]]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(x=offsets, name="all accounts", marker_color=_NEUTRAL, opacity=0.55, nbinsx=40)
    )
    if in_win:
        fig.add_trace(
            go.Histogram(x=in_win, name="in-window (burst)", marker_color=_RING, opacity=0.8, nbinsx=40)
        )
    fig.add_vline(x=0.0, line=dict(color="#8b929c", dash="dash", width=1),
                  annotation_text="window start", annotation_position="top")
    fig.update_layout(
        barmode="overlay", height=300, margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="registration offset (days from window start)", yaxis_title="accounts",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    return fig


def account_age_strip_figure(
    candidate: list[dict[str, float]], population: list[dict[str, float]]
) -> go.Figure:
    """Account-age strip: one selectable point per candidate account over a population violin
    (the age distribution chart). A fresh ring clusters at low age + low variance; legit spreads
    broad and old. Candidate points carry their account id (selectable)."""
    fig = go.Figure()
    if population:
        fig.add_trace(
            go.Violin(
                y=[r["age"] for r in population], name="population", line_color=_BASELINE,
                points=False, meanline_visible=True, hoverinfo="skip", width=0.8,
            )
        )
    ids = [int(r["account_id"]) for r in candidate]
    fig.add_trace(
        go.Scatter(
            x=["population"] * len(candidate), y=[r["age"] for r in candidate],
            mode="markers", name="candidate set", customdata=ids,
            marker=dict(color=_RING, size=9, line=dict(color="#333333", width=1)),
            hovertemplate="account %{customdata} · age %{y:.0f}d<extra></extra>",
        )
    )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=10, b=30), showlegend=False,
        yaxis_title="account age (days)", xaxis=dict(showticklabels=False),
    )
    return fig


def gap_histogram_figure(candidate: list[float], population: list[float]) -> go.Figure:
    """Inter-event gap histogram on a log-second axis (cadence comb). A cadenced ring burst
    piles up at a near-fixed gap (a sharp tooth); organic viewing smears across minutes-to-days."""
    cand = [g for g in candidate if g > 0]
    pop = [g for g in population if g > 0]
    fig = go.Figure()
    if pop:
        fig.add_trace(
            go.Histogram(
                x=pop, name="population", histnorm="probability density",
                marker_color=_BASELINE, opacity=0.55, nbinsx=50,
            )
        )
    fig.add_trace(
        go.Histogram(
            x=cand, name="candidate set", histnorm="probability density",
            marker_color=_RING, opacity=0.65, nbinsx=50,
        )
    )
    fig.update_layout(
        barmode="overlay", height=300, margin=dict(l=10, r=10, t=10, b=40),
        xaxis_title="inter-event gap (seconds, log scale)", yaxis_title="density",
        legend=dict(orientation="h", y=1.02, yanchor="bottom"),
    )
    fig.update_xaxes(type="log")
    return fig


# --- triage panel (data_triage → app_triage) ----------------------------------------------------
# Presentation-only evidence charts for one ranked suspect creator. With the truth overlay on, ring
# accounts among the top contributors are tinted red.
_HOT = "#d65f5f"
_USER = "#4c78a8"


def suspect_contributor_figure(rows: list[dict], ring_of: dict[int, int | None] | None = None) -> go.Figure:
    """Top contributors to a creator by qualified views (the visual behind `n_eff`). A few tall bars
    = concentrated (few-/single-source); a flat spread of many short bars = diffuse."""
    rows = list(reversed(rows))  # highest at top of the horizontal bar
    labels = [f"acct {r['account_id']}" for r in rows]
    colors = [_HOT if ring_of is not None and ring_of.get(r["account_id"]) is not None else _USER for r in rows]
    fig = go.Figure(
        go.Bar(
            x=[r["qv"] for r in rows], y=labels, orientation="h", marker_color=colors,
            hovertemplate="%{y}: %{x} qualified views<extra></extra>",
        )
    )
    fig.update_layout(
        height=max(180, 22 * len(rows)), margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="qualified views", yaxis_title="", showlegend=False,
    )
    return fig


def suspect_weekly_figure(rows: list[dict]) -> go.Figure:
    """Qualified views per tumbling 7-day window (the visual behind `burst7`). A campaign packs into
    one window (a tall lone tooth); an organic release smears across many weeks."""
    weeks = [r["week"] for r in rows]
    qvs = [r["qv"] for r in rows]
    peak = max(qvs) if qvs else 0
    fig = go.Figure(
        go.Bar(
            x=weeks, y=qvs,
            marker_color=[_HOT if q == peak else _NEUTRAL for q in qvs],
            hovertemplate="week %{x}: %{y} qualified views<extra></extra>",
        )
    )
    fig.update_layout(
        height=200, margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="week (0 = creator's first qualified view)", yaxis_title="qualified views",
        showlegend=False,
    )
    return fig


def suspect_axis_figure(pct_neff: float, pct_booster: float, pct_burst7: float) -> go.Figure:
    """The three entry-axis percentiles — *why* this target is on the shortlist. The tallest bar is
    the axis that fired; `suspicion` is the max of the three."""
    axes = ["contributor-conc. (1/n_eff)", "booster-count", "temporal (burst7)"]
    vals = [pct_neff, pct_booster, pct_burst7]
    top = max(vals) if vals else 0
    fig = go.Figure(
        go.Bar(
            x=vals, y=axes, orientation="h",
            marker_color=[_RING if v == top else _NEUTRAL for v in vals],
            hovertemplate="%{y}: %{x:.1%} percentile<extra></extra>",
        )
    )
    fig.update_layout(
        height=170, margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="percentile within the impact-floored pool", yaxis_title="", showlegend=False,
    )
    fig.update_xaxes(range=[0, 1], tickformat=".0%")
    return fig
