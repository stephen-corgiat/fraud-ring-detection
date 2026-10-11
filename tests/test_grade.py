"""Grade the supervised detector against the success targets.

Locks the headline claim end-to-end: across the evasion dial, on HELD-OUT rings, the
co-target candidate-gen + learned basket rule beats the hand-weighted ranking and holds the
operational precision / legit-FP line — the loop works even as the adversary turns sophisticated.
"""

from __future__ import annotations

import json

import polars as pl

from fraudrings.eval.grade import grade


def test_grade_beats_v1_and_holds_the_line_across_the_dial(tmp_path) -> None:
    result = grade(
        levels=["naive", "moderate", "sophisticated"], seeds=[7], preset_name="smoke", data_dir=str(tmp_path)
    )
    by = {g.level: g for g in result.levels}
    assert set(by) == {"naive", "moderate", "sophisticated"}

    # naive: linkage retrieval is near-perfect, but with always-widen the candidate pool now also
    # holds the benign fan-club, so naive is honest classification (not a trivial flag-all). It
    # still clears the bar with a perfect precision / zero-FP line.
    assert by["naive"].meets, (by["naive"].recall, by["naive"].operational_precision)
    assert by["naive"].operational_precision == 1.0 and by["naive"].legit_fp_rate == 0.0

    # the learned rule lifts actioned recall over the hand-weighted baseline at every non-trivial level
    assert by["moderate"].recall_lift > 0.0
    assert by["sophisticated"].recall_lift > 0.0, (by["sophisticated"].recall, by["sophisticated"].baseline_recall_at_k)

    # ...and the operational line holds under sophistication
    soph = by["sophisticated"]
    assert soph.operational_precision >= 0.90, soph.operational_precision
    assert soph.legit_fp_rate <= 0.05, soph.legit_fp_rate


def test_grade_reports_per_seed_variance(tmp_path) -> None:
    """Hardening: 'meets' must not hinge on one noisy world — the grade exposes per-seed values and
    a variance-aware count of how many seeds individually clear the bar (plus worst-case FP)."""
    result = grade(
        levels=["sophisticated"], seeds=[7, 13], preset_name="smoke", data_dir=str(tmp_path)
    )
    g = result.levels[0]
    assert len(g.recall_seeds) == 2 and len(g.fp_seeds) == 2 and len(g.precision_seeds) == 2
    # the headline numbers are the means of the per-seed values
    assert abs(g.recall - sum(g.recall_seeds) / 2) < 1e-9
    # variance-aware accounting is consistent and worst-case FP >= the mean
    assert 0 <= g.n_seeds_meeting <= 2
    assert g.fp_max >= g.legit_fp_rate
    assert g.meets_all_seeds == (g.n_seeds_meeting == 2)

    reloaded = json.loads(json.dumps(result.to_dict()))
    lvl = reloaded["levels"][0]
    assert len(lvl["fp_per_seed"]) == 2 and "n_seeds_meeting" in lvl


def test_grade_serializes_and_persists(tmp_path) -> None:
    result = grade(levels=["naive", "sophisticated"], seeds=[7], preset_name="smoke", data_dir=str(tmp_path))

    reloaded = json.loads(json.dumps(result.to_dict()))
    assert reloaded["targets"]["recall"] == 0.80
    assert {lvl["level"] for lvl in reloaded["levels"]} == {"naive", "sophisticated"}

    written = pl.read_csv(str(tmp_path / "rule_grades.csv"))
    assert written.height == 2
    assert {"recall", "operational_precision", "meets_targets", "recall_lift"} <= set(written.columns)
