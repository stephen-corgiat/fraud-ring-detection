"""The distilled supervised rule, graded on held-out rings.

These pin the headline claims:

  * a learned rule over the weak-signal *basket* (cover-view breadth, timing spread, co-target,
    residual linkage) lifts actioned recall on held-out rings well past the hand-weighted
    score, while holding the operational precision / legit-FP line;
  * the near-perfect `challenge_fail_rate` tell is NOT what carries it — ablating it barely moves
    the basket rule (we grade the basket honestly);
  * the rule is trained and tested on DISJOINT rings and never sees ground truth as a feature
    (anti-tautology).
"""

from __future__ import annotations

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import evaluate
from fraudrings.eval.supervised import (
    CHALLENGE_FEATURE,
    HEADLINE_FEATURES,
    Targets,
    collect_labeled_candidates,
    evaluate_supervised,
)
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel


def _world(tmp_path, level: EvasionLevel, *, seed: int = 7, benign_cluster_size: int | None = None):
    """A multi-ring world (enough rings for leave-one-ring-out) at smoke scale for speed."""
    cfg = preset("smoke")
    cfg.sim.n_rings = 6
    cfg.sim.n_legit = 400
    if benign_cluster_size is not None:
        cfg.sim.benign_cluster_size = benign_cluster_size
    cfg.sim.evasion_level = level
    cfg.sim.seed = seed
    cfg.engine.cotarget_candidate_gen = True
    cfg.engine.cotarget_always = True  # supervised rule always-widens (see grade.py / EngineConfig)
    cfg.data_dir = str(tmp_path / f"{level}-{seed}")
    events_path, gt_path = generate(cfg.sim, cfg.data_dir)
    con = build_db(events_path, cfg.engine, db_path=str(tmp_path / f"{level}-{seed}" / "world.duckdb"))
    return cfg, con, gt_path


def test_collect_labels_retrieves_ring_members_with_features(tmp_path) -> None:
    cfg, con, gt_path = _world(tmp_path, "sophisticated")
    rows, held_counts = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)
    assert held_counts and all(c > 0 for c in held_counts.values())
    assert any(r.label == 1 for r in rows), "no true ring members retrieved as candidates"
    # features are populated for both the score components and the behavioral basket
    sample = next(r for r in rows if r.label == 1)
    for f in HEADLINE_FEATURES:
        assert f in sample.feats


def test_rule_lifts_recall_and_holds_the_line_on_held_out_rings(tmp_path) -> None:
    """The headline: the learned basket rule roughly doubles actioned recall vs the hand-weighted
    ranking on held-out rings, while keeping operational precision high and the legit-FP line ≤ 5%."""
    cfg, con, gt_path = _world(tmp_path, "sophisticated")
    rows, held_counts = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)

    base = evaluate(con, gt_path, cfg.engine, cfg.eval)  # hand-weighted Stage 2, actioned top-k
    rule = evaluate_supervised(rows, held_counts, features=HEADLINE_FEATURES, model="logreg")

    assert rule.recall > base.mean_recall_at_k, (rule.recall, base.mean_recall_at_k)
    assert rule.recall >= 0.55, rule.recall
    assert rule.operational_precision >= 0.90, rule.operational_precision
    assert rule.legit_fp_rate <= 0.05, rule.legit_fp_rate


def test_moderate_rule_meets_all_targets(tmp_path) -> None:
    """At moderate sophistication the learned rule should clear the full bar (0.90p / 0.80r / 5%FP)."""
    cfg, con, gt_path = _world(tmp_path, "moderate")
    rows, held_counts = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)
    rule = evaluate_supervised(rows, held_counts, features=HEADLINE_FEATURES, model="logreg")
    assert rule.meets(Targets()), (rule.recall, rule.operational_precision, rule.legit_fp_rate)


def test_moderate_meets_when_superfan_confound_is_in_training(tmp_path) -> None:
    """Fairness, graded honestly: a held-out ring that targets the same
    creator as a benign super-fan club must not blow the legit-FP line.

    With a single fan-club (on target_creators[0]) only ONE ring collides with it, and under
    leave-one-ring-out that ring is always the *test* fold — the rule meets the ring-vs-superfan
    boundary for the first time on held-out data and over-flags the fans (seed 13: FP ~0.14).
    With one cohort PER target creator, the confound also appears in the training folds, so the rule
    can learn to reject super-fans and holds the line on the held-out ring. A large fan-club
    (benign_cluster_size=60) and seed 13 make the pre-fix miss unambiguous."""
    cfg, con, gt_path = _world(tmp_path, "moderate", seed=13, benign_cluster_size=60)
    rows, held_counts = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)
    rule = evaluate_supervised(rows, held_counts, features=HEADLINE_FEATURES, model="logreg")
    assert rule.meets(Targets()), (rule.recall, rule.operational_precision, rule.legit_fp_rate)


def test_challenge_ablation_barely_moves_the_basket(tmp_path) -> None:
    """The honest-thesis check: the basket already separates ring from fan-club, so adding the
    near-perfect challenge tell changes recall only marginally — the basket carries the rule."""
    cfg, con, gt_path = _world(tmp_path, "sophisticated")
    rows, held_counts = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)

    basket = evaluate_supervised(rows, held_counts, features=HEADLINE_FEATURES, model="logreg")
    plus = evaluate_supervised(
        rows, held_counts, features=HEADLINE_FEATURES + (CHALLENGE_FEATURE,), model="logreg"
    )
    assert abs(plus.recall - basket.recall) <= 0.10, (basket.recall, plus.recall)


def test_features_never_include_ground_truth() -> None:
    """Anti-tautology (invariant #1): no oracle column may be a feature."""
    forbidden = {"evasion_level", "ring_id", "is_fraud", "label", "account_id"}
    assert forbidden.isdisjoint(set(HEADLINE_FEATURES) | {CHALLENGE_FEATURE})
