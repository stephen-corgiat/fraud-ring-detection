"""The Plotly chart spine — chart substrate (`viewer.data`) + Plotly builders (`viewer.charts`).

Exercised directly (no Streamlit runtime) against a freshly built smoke world. The chart
substrate is presentation-only and event-derived — nothing here is read by the detector or
feeds scoring — so reading ground truth to pick a ring is allowed (eval side of the line).
"""

from __future__ import annotations

from typing import Any, cast

import plotly.graph_objects as go

from fraudrings.schema import EvasionLevel
from fraudrings.viewer import charts as C
from fraudrings.viewer import data as D


def _customdata(trace: object) -> tuple:
    """Account ids a trace carries for select-to-reseed (plotly types `.customdata` loosely)."""
    return tuple(cast(Any, getattr(trace, "customdata", None)) or ())


def _traces(fig: go.Figure) -> tuple:
    """`fig.data` as a concrete tuple (plotly types the property loosely for `len`/iteration)."""
    return tuple(cast(Any, fig.data))


def _world(tmp_path, level: EvasionLevel = "naive", seed=7):
    return D.load_world("smoke", level, seed, cotarget_on=True, base_dir=str(tmp_path))


def _candidate_set(w, cap=40):
    """Mirror the app: seeds + recovered candidates, capped — the on-the-fly chart surface."""
    ring0 = D.rings(w)[0].ring_id
    seeds = D.default_seeds(w, ring0, k=3)
    cands = D.investigate(w, seeds)
    set_ids = list(dict.fromkeys([*seeds, *(c.account_id for c in cands)]))[:cap]
    return ring0, seeds, set_ids


# --- chart substrate (viewer.data) -----------------------------------------


def test_action_composition_one_row_per_account_in_order(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    rows = D.action_composition(w, set_ids)
    assert [r["account_id"] for r in rows] == set_ids  # preserves input order
    for r in rows:
        assert set(D.FUNNEL_ACTIONS) <= set(r)
        assert all(isinstance(r[a], int) and r[a] >= 0 for a in D.FUNNEL_ACTIONS)
    assert D.action_composition(w, []) == []


def test_position_samples_candidate_subset_and_population_baseline(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    cand = D.position_samples(w, set_ids)
    pop = D.position_samples(w)
    assert cand and pop
    assert len(pop) >= len(cand)
    assert all(0.0 <= p <= 1.0 for p in cand)
    assert D.position_samples(w, []) == []


def test_gap_samples_positive_and_population_capped(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    cand = D.gap_samples(w, set_ids)
    pop = D.gap_samples(w)
    assert pop  # population has gaps
    assert all(g >= 0 for g in cand)
    assert len(pop) <= D._BASELINE_SAMPLE_CAP


def test_cotarget_matrix_shape_is_consistent(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    accounts, creators, matrix = D.cotarget_matrix(w, set_ids, top_n=8)
    assert accounts == set_ids
    assert len(creators) <= 8
    assert len(matrix) == len(accounts)
    assert all(len(row) == len(creators) for row in matrix)
    assert D.cotarget_matrix(w, []) == ([], [], [])


def test_registration_offsets_flags_in_window(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    rows = D.registration_offsets(w, set_ids)
    assert rows
    for r in rows:
        assert {"account_id", "registration_ts", "offset_days", "in_window"} <= set(r)
        # in_window <=> registered at or after the window start (offset >= 0).
        assert bool(r["in_window"]) == (r["offset_days"] >= 0.0)


def test_account_age_samples_population_superset_of_candidate(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    cand = D.account_age_samples(w, set_ids)
    pop = D.account_age_samples(w)
    assert {r["account_id"] for r in cand} <= {r["account_id"] for r in pop}
    assert all(r["age"] >= 0.0 for r in pop)


def test_linkage_graph_recovers_ring_infra_edges(tmp_path) -> None:
    # A naive ring shares rare infra -> its members must form at least one in-gate edge.
    w = _world(tmp_path, level="naive")
    ring0 = D.rings(w)[0].ring_id
    members = D.ring_members(w, ring0)
    nodes, edges = D.linkage_graph(w, members)
    assert nodes == members
    assert edges, "naive ring members should link via shared rare identifiers"
    for e in edges:
        assert e["a"] < e["b"] and e["weight"] > 0 and e["n_shared"] >= 1
    # fewer than two accounts -> no edges
    assert D.linkage_graph(w, members[:1]) == (members[:1], [])


# --- Plotly builders (viewer.charts) ---------------------------------------


def test_action_composition_figure_traces_and_customdata(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    fig = C.action_composition_figure(D.action_composition(w, set_ids))
    assert isinstance(fig, go.Figure)
    traces = _traces(fig)
    assert len(traces) == len(D.FUNNEL_ACTIONS)  # one stacked trace per funnel action
    assert fig.layout.barmode == "stack"
    assert all(_customdata(tr) == tuple(set_ids) for tr in traces)  # ids for select-to-reseed


def test_position_histogram_figure_has_threshold_line(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    fig = C.position_histogram_figure(
        D.position_samples(w, set_ids), D.position_samples(w), 0.3
    )
    assert isinstance(fig, go.Figure)
    # threshold vline lands as a layout shape at x=0.3
    assert any(getattr(s, "x0", None) == 0.3 for s in fig.layout.shapes)


def test_linkage_graph_figure_node_customdata(tmp_path) -> None:
    w = _world(tmp_path, level="naive")
    members = D.ring_members(w, D.rings(w)[0].ring_id)
    nodes, edges = D.linkage_graph(w, members)
    fig = C.linkage_graph_figure(nodes, edges)
    assert isinstance(fig, go.Figure)
    node_trace = fig.data[-1]  # nodes drawn last, over the edges
    assert _customdata(node_trace) == tuple(nodes)


def test_account_age_strip_figure_carries_candidate_ids(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    fig = C.account_age_strip_figure(
        D.account_age_samples(w, set_ids), D.account_age_samples(w)
    )
    assert isinstance(fig, go.Figure)
    scatter = [tr for tr in fig.data if isinstance(tr, go.Scatter)][-1]
    assert _customdata(scatter) == tuple(set_ids)


def test_cotarget_heatmap_and_registration_and_gap_figures(tmp_path) -> None:
    w = _world(tmp_path)
    _, _, set_ids = _candidate_set(w)
    accs, arts, mat = D.cotarget_matrix(w, set_ids)
    assert isinstance(C.cotarget_heatmap_figure(accs, arts, mat), go.Figure)
    assert isinstance(C.registration_burst_figure(D.registration_offsets(w, set_ids)), go.Figure)
    assert isinstance(
        C.gap_histogram_figure(D.gap_samples(w, set_ids), D.gap_samples(w)), go.Figure
    )


def test_occupancy_figure_draws_cap_line(tmp_path) -> None:
    w = D.load_world("smoke", "naive", 7, base_dir=str(tmp_path), concurrency=12)
    ring0 = D.rings(w)[0].ring_id
    rd = D.account_raster(w, D.ring_members(w, ring0), bucket_seconds=300)
    fig = C.occupancy_figure(rd, cap=12)
    assert isinstance(fig, go.Figure)
    assert any(getattr(s, "y0", None) == 12 for s in fig.layout.shapes)  # cap rule
    assert isinstance(C.occupancy_figure(rd, cap=None), go.Figure)
