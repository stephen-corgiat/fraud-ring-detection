"""The learning curve (metric vs. #labels) + label-noise sensitivity.

These pin:

  * **Continuity anchor:** at ε=0 / full budget a curve point reproduces the grade exactly
    — the curve is the grade, sub-sampled (the easiest correctness guarantee).
  * the random-baseline curve **rises to clear the bar**, and `labels_to_target` is the stable
    crossing;
  * **label noise recedes the target** — more labels (or never) to clear the bar as ε rises;
  * **uncertainty sampling's machinery is sound** — it converges to random/the grade once the whole
    pool is labeled. (Whether AL *wins* is scale-dependent: it halves the clean-label cost at
    default scale but loses to random at smoke scale, so we deliberately don't assert a win here.)
  * the per-level `labels_to_target` summary and the JSON/CSV artifacts round-trip.

Smoke scale, with `n_rings` bumped to 6 for leave-one-ring-out stability (as `test_supervised._world`
does). The crossing tests use the *moderate* level, which reliably clears the bar at smoke;
sophisticated smoke does not always reach recall ≥ 0.80, so it is used only for the (meet-agnostic)
reconciliation anchor.
"""

from __future__ import annotations

import json

import numpy as np
import polars as pl

from fraudrings.config import RunConfig, preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.curve import (
    NoiseSpec,
    _eval_point,
    _random_order,
    _World,
    learning_curve,
    write_curve,
)
from fraudrings.eval.supervised import (
    DEFAULT_TARGETS,
    HEADLINE_FEATURES,
    _matrix,
    collect_labeled_candidates,
    evaluate_supervised,
)
from fraudrings.generator import generate


def _smoke6(cfg: RunConfig) -> None:
    """Bump smoke to 6 rings / 400 legit — leave-one-ring-out needs enough rings."""
    cfg.sim.n_rings = 6
    cfg.sim.n_legit = 400


def test_full_clean_point_reconciles_with_grade(tmp_path) -> None:
    """The anchor: at ε=0 and the full budget a curve point equals `evaluate_supervised` *exactly*
    — noise and budget are perturbations that vanish in that corner. We evaluate both on the SAME
    candidate pool (the curve sorts its full selection back to the original row order, so the
    logistic fit is bit-identical — not merely close)."""
    cfg = preset("smoke")
    _smoke6(cfg)
    cfg.sim.evasion_level = "sophisticated"
    cfg.sim.seed = 7
    cfg.engine.cotarget_candidate_gen = True
    cfg.engine.cotarget_always = True
    cfg.data_dir = str(tmp_path / "direct")
    events_path, gt_path = generate(cfg.sim, cfg.data_dir)
    con = build_db(events_path, cfg.engine, db_path=str(tmp_path / "direct" / "world.duckdb"))
    rows, held = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)
    ref = evaluate_supervised(rows, held, features=HEADLINE_FEATURES, model="logreg")

    world = _World(
        seed=7,
        level="sophisticated",
        x=_matrix(rows, HEADLINE_FEATURES),
        label=np.array([r.label for r in rows], dtype=int),
        group=np.array([r.ring_id for r in rows], dtype=int),
        fraud_any=np.array([1 if r.is_fraud_any else 0 for r in rows], dtype=int),
        held_counts=held,
        features=HEADLINE_FEATURES,
    )
    orders = {h: _random_order(world, h) for h in world.held_counts}
    recall, op, fp = _eval_point(
        world, world.label, orders, world.max_train_pool(), DEFAULT_TARGETS, "logreg"
    )
    assert recall == ref.recall, (recall, ref.recall)
    assert op == ref.operational_precision, (op, ref.operational_precision)
    assert fp == ref.legit_fp_rate, (fp, ref.legit_fp_rate)


def test_random_curve_rises_and_crosses(tmp_path) -> None:
    result = learning_curve(
        seeds=[7, 13],
        preset_name="smoke",
        level="moderate",
        strategies=("random",),
        epsilons=(0.0,),
        summary_levels=("moderate",),
        data_dir=str(tmp_path),
        configure=_smoke6,
    )
    pts = result.series[0].points
    assert pts[0].n_labels < pts[-1].n_labels  # budget grid is increasing
    assert pts[-1].is_full
    assert pts[-1].recall >= pts[0].recall  # more labels never lose recall, broadly
    assert pts[-1].meets(), (pts[-1].recall, pts[-1].operational_precision, pts[-1].legit_fp_rate)
    assert result.series[0].labels_to_target is not None  # a stable crossing exists


def test_noise_recedes_the_target(tmp_path) -> None:
    result = learning_curve(
        seeds=[7, 13],
        preset_name="smoke",
        level="moderate",
        strategies=("random",),
        epsilons=(0.0, 0.20),
        summary_levels=("moderate",),
        data_dir=str(tmp_path),
        configure=_smoke6,
    )
    by_eps = {s.noise.fraud_to_legit: s for s in result.series}
    clean = by_eps[0.0].labels_to_target
    noisy = by_eps[0.20].labels_to_target
    assert clean is not None
    # heavy oracle noise costs at least as many labels to clear the bar — or never clears it.
    assert noisy is None or noisy >= clean, (clean, noisy)


def test_uncertainty_is_computed_and_converges_at_full_budget(tmp_path) -> None:
    """AL's payoff is scale-dependent and we refuse to bake a flaky win into a smoke test: at default
    scale uncertainty sampling can first clear the bar at about half random's budget (though not
    stably), but at *this* smoke scale it
    underperforms random (small pool ⇒ boundary-focused labels are unrepresentative). So we assert
    only the scale-agnostic invariant: at the *full* budget both strategies have labeled the entire
    pool, so the two curves must converge to the same point (and to the grade)."""
    result = learning_curve(
        seeds=[7, 13],
        preset_name="smoke",
        level="sophisticated",
        strategies=("random", "uncertainty"),
        epsilons=(0.0,),
        summary_levels=("sophisticated",),
        data_dir=str(tmp_path),
        configure=_smoke6,
    )
    by = {s.strategy: s for s in result.series}
    assert len(by["uncertainty"].points) == len(by["random"].points)
    rnd_full, al_full = by["random"].points[-1], by["uncertainty"].points[-1]
    assert rnd_full.is_full and al_full.is_full
    # full pool labeled by either route -> identical training set -> identical held-out metrics
    assert abs(al_full.recall - rnd_full.recall) < 1e-9, (al_full.recall, rnd_full.recall)
    assert abs(al_full.operational_precision - rnd_full.operational_precision) < 1e-9
    # AL still yields a (possibly None) stable crossing without error
    assert by["uncertainty"].labels_to_target is None or by["uncertainty"].labels_to_target > 0


def test_exploit_is_computed_and_converges_at_full_budget(tmp_path) -> None:
    """The exploit comparator (label the clearest-fraud cases) is wired in like uncertainty. As with
    AL, whether it *wins* is scale-dependent, so we assert only the scale-agnostic invariant: at the
    full budget it has labeled the whole pool and converges to random (and the grade)."""
    result = learning_curve(
        seeds=[7, 13],
        preset_name="smoke",
        level="sophisticated",
        strategies=("random", "exploit"),
        epsilons=(0.0,),
        summary_levels=("sophisticated",),
        data_dir=str(tmp_path),
        configure=_smoke6,
    )
    by = {s.strategy: s for s in result.series}
    assert "exploit" in by
    rnd_full, exp_full = by["random"].points[-1], by["exploit"].points[-1]
    assert rnd_full.is_full and exp_full.is_full
    assert abs(exp_full.recall - rnd_full.recall) < 1e-9, (exp_full.recall, rnd_full.recall)
    assert abs(exp_full.operational_precision - rnd_full.operational_precision) < 1e-9


def test_asymmetric_noise_isolates_precision(tmp_path) -> None:
    """The realistic-noise finding: oracle error is
    missed-fraud-dominant, and that is the *benign* direction for the success bar. Missed-fraud noise
    (fraud→legit) costs recall at worst; it's the false-confirmations (legit→fraud) that teach the
    rule to flag innocent super-fans and blow the ≤5% legit-FP line. So at the full budget the
    legit-FP under missed-fraud-only noise must be below that under false-confirm-only noise — the
    fragile target is precision, driven by false confirmations."""
    result = learning_curve(
        seeds=[7, 13],
        preset_name="smoke",
        level="sophisticated",
        strategies=("random",),
        epsilons=(0.0,),
        asymmetric=(NoiseSpec(0.20, 0.0), NoiseSpec(0.0, 0.20)),
        summary_levels=("sophisticated",),
        data_dir=str(tmp_path),
        configure=_smoke6,
    )
    by = {s.noise: s for s in result.series}
    clean_fp = by[NoiseSpec(0.0, 0.0)].points[-1].legit_fp_rate
    missed_fraud_fp = by[NoiseSpec(0.20, 0.0)].points[-1].legit_fp_rate
    false_confirm_fp = by[NoiseSpec(0.0, 0.20)].points[-1].legit_fp_rate
    # false confirmations blow the FP line; missed fraud barely moves it (stays near the clean line).
    assert false_confirm_fp > missed_fraud_fp, (false_confirm_fp, missed_fraud_fp)
    assert missed_fraud_fp <= clean_fp + 0.02, (missed_fraud_fp, clean_fp)


def test_per_level_summary_and_artifacts(tmp_path) -> None:
    result = learning_curve(
        seeds=[7],
        preset_name="smoke",
        level="sophisticated",
        strategies=("random",),
        epsilons=(0.0,),
        summary_levels=("naive", "moderate", "sophisticated"),
        data_dir=str(tmp_path),
        configure=_smoke6,
    )
    summ = result.labels_to_target_by_level
    assert set(summ) == {"naive", "moderate", "sophisticated"}
    assert summ["moderate"] is not None  # moderate reliably crosses at smoke

    write_curve(result, str(tmp_path))
    reloaded = json.loads(json.dumps(result.to_dict()))
    assert reloaded["level"] == "sophisticated"
    assert "labels_to_target_by_level" in reloaded
    assert reloaded["targets"]["recall"] == 0.80

    written = pl.read_csv(str(tmp_path / "learning_curve.csv"))
    assert {"strategy", "noise", "fraud_to_legit", "legit_to_fraud", "n_labels", "recall", "meets_targets"} <= set(
        written.columns
    )
