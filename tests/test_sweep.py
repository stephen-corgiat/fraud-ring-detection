"""The sweep harness + the two curves.

The headline deliverable. The dial *moves* the metrics on one world; the sweep turns it
across every level and emits the degradation curve — and,
crucially, separates the two stories the single AP number conflates:

  * **per-ring ranking** (mean AP / Stage-1 recall) degrades with sophistication;
  * the **operational false-positive line on legit holds** — the precision loss is the
    engine confusing one fraud ring for *another* (still fraud), not flagging real
    viewers. The benign-coordination control never cracks the flagged set.

These tests pin both: the operational decomposition the sweep reports (this-ring /
other-ring-fraud / legit) and the sweep's monotonic-degradation-with-held-legit-line
shape. The detector still never sees `evasion_level` (anti-tautology).
"""

from __future__ import annotations

import json

import polars as pl

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import evaluate
from fraudrings.eval.sweep import sweep
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel


def _evaluate_world(tmp_path, level: EvasionLevel, *, seed: int = 7):
    cfg = preset("smoke")
    cfg.data_dir = str(tmp_path / f"{level}-{seed}")
    cfg.sim.evasion_level = level
    cfg.sim.seed = seed
    events_path, gt_path = generate(cfg.sim, cfg.data_dir)
    con = build_db(events_path, cfg.engine, db_path=str(tmp_path / f"{level}-{seed}" / "world.duckdb"))
    return evaluate(con, gt_path, cfg.engine, cfg.eval)


def test_flagged_decomposition_is_internally_consistent(tmp_path) -> None:
    """Every flagged slot lands in exactly one bucket: this-ring / other-ring-fraud / legit.
    The three counts must partition the flagged total, per ring and in aggregate."""
    result = _evaluate_world(tmp_path, "moderate")
    assert result.rings, "no rings evaluated"

    for r in result.rings:
        assert r.flagged_this_ring + r.flagged_other_ring + r.flagged_legit == r.flagged_total
        # flagged set is the top-k (k = #held-out) of the ranked candidates
        assert r.flagged_total <= r.n_held_out

    assert result.flagged_total == sum(r.flagged_total for r in result.rings)
    # operational precision = fraction of flagged slots that are fraud at all (any ring)
    op = result.operational_precision
    assert 0.0 <= op <= 1.0
    legit = sum(r.flagged_legit for r in result.rings)
    assert result.legit_fp_rate == (legit / result.flagged_total if result.flagged_total else 0.0)


def test_naive_world_has_no_legit_or_cross_ring_leak(tmp_path) -> None:
    """On the naive adversary recovery is trivially perfect: every flagged slot is a
    this-ring true positive — no cross-ring bleed, no legit false positives."""
    result = _evaluate_world(tmp_path, "naive")
    assert result.flagged_total > 0
    assert result.legit_fp_rate == 0.0
    assert all(r.flagged_other_ring == 0 and r.flagged_legit == 0 for r in result.rings)
    assert result.operational_precision == 1.0


def test_sweep_reproduces_the_curve_and_holds_the_legit_line(tmp_path) -> None:
    """The success check. Sweeping the dial must reproduce the headline:
    per-ring ranking (AP, Stage-1 recall) slopes DOWN with sophistication while the
    operational precision on legit stays ~1.0 (the precision loss is cross-ring)."""
    result = sweep(
        levels=["naive", "moderate", "sophisticated"],
        seeds=[7, 13],
        preset_name="smoke",
        data_dir=str(tmp_path),
    )

    by_level = {s.level: s for s in result.levels}
    naive, mod, soph = by_level["naive"], by_level["moderate"], by_level["sophisticated"]

    # naive is the trivial-recovery regime
    assert naive.mean_ap >= 0.97 and naive.mean_stage1_recall >= 0.97

    # ranking degrades monotonically with sophistication
    assert naive.mean_ap > mod.mean_ap > soph.mean_ap, (
        naive.mean_ap, mod.mean_ap, soph.mean_ap
    )
    assert soph.mean_ap <= 0.80
    assert naive.mean_stage1_recall > soph.mean_stage1_recall

    # ...but the operational false-positive line on legit holds at every level
    for s in result.levels:
        assert s.operational_precision >= 0.95, (s.level, s.operational_precision)
        assert s.legit_fp_rate <= 0.05, (s.level, s.legit_fp_rate)

    # the degradation that DOES happen is cross-ring bleed, surfacing as sophistication rises
    assert soph.cross_ring_frac >= naive.cross_ring_frac


def test_sweep_result_serializes_round_trip(tmp_path) -> None:
    """The sweep emits a serializable artifact (JSON) + CSV rows for the curve."""
    result = sweep(
        levels=["naive", "sophisticated"],
        seeds=[7],
        preset_name="smoke",
        data_dir=str(tmp_path),
    )

    blob = json.dumps(result.to_dict())
    reloaded = json.loads(blob)
    assert {lvl["level"] for lvl in reloaded["levels"]} == {"naive", "sophisticated"}
    assert reloaded["preset"] == "smoke"
    assert reloaded["seeds"] == [7]

    rows = result.to_rows()
    assert len(rows) == 2
    for row in rows:
        assert {"level", "mean_ap", "mean_stage1_recall", "operational_precision", "legit_fp_rate"} <= set(row)

    # the curve persists to disk under data_dir for downstream plotting/reporting
    written = pl.read_csv(str(tmp_path / "sweep.csv"))
    assert written.height == 2
    assert "mean_ap" in written.columns
