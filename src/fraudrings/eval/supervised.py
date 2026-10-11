"""The distilled supervised rule.

The engine ranks candidates with a hand-set linear score over linkage, behavioral, co-target and
action-mix evidence. Under a sophisticated adversary that's not enough: the members co-target
candidate generation retrieves carry no linkage evidence, so the hand-weighted score buries them.
A *learned* rule can weight the weak-signal basket — cover-view breadth, timing spread,
co-target, residual linkage — to lift actioned recall while holding the legit false-positive line.

Design:
  * Interpretable first — **logistic regression** is the headline ("the model is the rule"; its
    coefficients are inspectable). LightGBM is offered as a stronger, less-transparent comparator.
  * **Train/test split by RING**: leave-one-ring-out CV, so we never test on a ring
    we trained on — the honest "catch a *held-out* ring" question.
  * Labels are oracle-provided (this module lives in eval/, the oracle). Features are derived from
    events only — never `evasion_level`/`ring_id`/`is_fraud` (anti-tautology).
  * The near-perfect `challenge_fail_rate` tell is kept OUT of the headline feature set and used
    only as an ablation / operational lever — we grade the weak-signal basket honestly.

Operating point: on each training fold, pick the highest-recall probability threshold that still
holds precision ≥ target and legit-FP ≤ target; apply it to the held-out ring. Precision is
*operational* (a flagged account is a TP if it is fraud in ANY ring — cross-ring bleed is still
fraud); a false positive is only a genuine legit/benign account.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np
import polars as pl

from fraudrings.config import EngineConfig, EvalConfig
from fraudrings.detector.engine import run_investigation

# Seed-relative score components (from the engine's Candidate) + the raw behavioral basket. The basket
# is what separates a ring member from a benign super-fan: cover-view breadth (n_creators), timing
# spread (gap_std/gap_mean), target concentration (top_creator_share), spoof tell (qualified_rate).
SCORE_FEATURES: tuple[str, ...] = ("linkage", "behavioral", "cotarget")
BASKET_FEATURES: tuple[str, ...] = (
    "n_creators",
    "top_creator_share",
    "gap_std",
    "gap_mean",
    "qualified_rate",
    "skip_rate",
    "views_per_session",
    "active_hours",
    "n_ips",
    "n_asns",
)
HEADLINE_FEATURES: tuple[str, ...] = SCORE_FEATURES + BASKET_FEATURES
# challenge_fail_rate is excluded from HEADLINE_FEATURES on purpose: it is a near-perfect tell, and
# the point is to grade the weak-signal basket honestly. Callers opt in for the ablation by appending it.
CHALLENGE_FEATURE: str = "challenge_fail_rate"


@dataclass(frozen=True)
class Targets:
    """The success bar (user-chosen): precision ≥ 0.90 (operational), recall ≥ 0.80, legit-FP ≤ 5%."""

    precision: float = 0.90
    recall: float = 0.80
    legit_fp: float = 0.05


# Frozen → safe to share as a default-argument singleton.
DEFAULT_TARGETS = Targets()


@dataclass
class CandidateRow:
    ring_id: int  # the investigated ring this candidate was retrieved for
    account_id: int
    label: int  # 1 iff a held-out member of THIS ring (the per-ring recovery target)
    is_fraud_any: bool  # fraud in any ring — the operational TP definition
    feats: dict[str, float]


@dataclass
class SupervisedResult:
    model: str
    features: tuple[str, ...]
    recall: float  # held-out ring members flagged / all held-out members (ceiling-aware)
    operational_precision: float  # flagged that are fraud (any ring) / flagged
    legit_fp_rate: float  # flagged that are legit / flagged
    flagged_total: int
    n_positives: int  # total held-out members across rings (the recall denominator)

    def meets(self, t: Targets) -> bool:
        return (
            self.recall >= t.recall
            and self.operational_precision >= t.precision
            and self.legit_fp_rate <= t.legit_fp
        )


def _fetch_features(con: duckdb.DuckDBPyConnection, ids: list[int]) -> dict[int, dict[str, float]]:
    if not ids:
        return {}
    cols = (*BASKET_FEATURES, CHALLENGE_FEATURE)
    clause = "(" + ",".join(str(int(i)) for i in ids) + ")"
    rows = con.execute(
        f"SELECT account_id, {', '.join(cols)} FROM account_features WHERE account_id IN {clause}"
    ).fetchall()
    out: dict[int, dict[str, float]] = {}
    for r in rows:
        out[int(r[0])] = {c: float(r[i + 1] or 0.0) for i, c in enumerate(cols)}
    return out


def collect_labeled_candidates(
    con: duckdb.DuckDBPyConnection,
    ground_truth_path: str,
    engine: EngineConfig,
    eval_cfg: EvalConfig,
) -> tuple[list[CandidateRow], dict[int, int]]:
    """Run a seeded investigation per ring and label every retrieved candidate.

    Returns the candidate rows (features + labels) and, per ring, the number of held-out members
    (the recall denominator — a member Stage-1 never retrieved still counts against recall).
    """
    gt = pl.read_parquet(ground_truth_path)
    fraud = gt.filter(pl.col("is_fraud") & pl.col("ring_id").is_not_null())
    fraud_any = {int(a) for a in gt.filter(pl.col("is_fraud"))["account_id"].to_list()}
    ring_ids = sorted({int(r) for r in fraud["ring_id"].to_list()})

    rows: list[CandidateRow] = []
    held_counts: dict[int, int] = {}
    for ring_id in ring_ids:
        members = sorted(
            int(a) for a in fraud.filter(pl.col("ring_id") == ring_id)["account_id"].to_list()
        )
        k = eval_cfg.seeds_per_ring
        if len(members) <= k + 1:
            continue
        seeds, positives = members[:k], set(members[k:])
        held_counts[ring_id] = len(positives)

        candidates = run_investigation(con, seeds, engine)
        feats = _fetch_features(con, [c.account_id for c in candidates])
        for c in candidates:
            row_feats = {"linkage": c.linkage, "behavioral": c.behavioral, "cotarget": c.cotarget}
            row_feats.update(feats.get(c.account_id, {}))
            rows.append(
                CandidateRow(
                    ring_id=ring_id,
                    account_id=c.account_id,
                    label=int(c.account_id in positives),
                    is_fraud_any=c.account_id in fraud_any,
                    feats=row_feats,
                )
            )
    return rows, held_counts


def _matrix(rows: list[CandidateRow], features: tuple[str, ...]) -> np.ndarray:
    return np.array([[r.feats.get(f, 0.0) for f in features] for r in rows], dtype=float)


def _fit_predict(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, model: str
) -> tuple[np.ndarray, np.ndarray]:
    """Fit `model` on train, return (train_probs, test_probs). Logistic regression is standardized
    (its coefficients then read as signed signal weights); LightGBM is scale-invariant."""
    if model == "logreg":
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        scaler = StandardScaler().fit(x_tr)
        xs_tr, xs_te = scaler.transform(x_tr), scaler.transform(x_te)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(xs_tr, y_tr)
        return np.asarray(clf.predict_proba(xs_tr))[:, 1], np.asarray(clf.predict_proba(xs_te))[:, 1]
    if model == "lgbm":
        import warnings

        import lightgbm as lgb

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # cosmetic sklearn-wrapper feature-name notices
            clf = lgb.LGBMClassifier(
                n_estimators=200, num_leaves=15, min_child_samples=5,
                class_weight="balanced", verbose=-1,
            ).fit(x_tr, y_tr)
            return np.asarray(clf.predict_proba(x_tr))[:, 1], np.asarray(clf.predict_proba(x_te))[:, 1]
    raise ValueError(f"unknown model {model!r} (expected 'logreg' or 'lgbm')")


def _choose_threshold(
    probs: np.ndarray, label: np.ndarray, fraud_any: np.ndarray, targets: Targets
) -> float:
    """Highest-recall probability threshold (on the training fold) that still holds precision and
    FP within target. Lowering the threshold flags more → higher recall but worse precision/FP."""
    n_pos = max(1, int(label.sum()))
    best_thr, best_recall = float("inf"), -1.0
    for thr in sorted(set(probs.tolist()), reverse=True):
        flag = probs >= thr
        n_flag = int(flag.sum())
        if n_flag == 0:
            continue
        precision = float((flag & (fraud_any == 1)).sum()) / n_flag
        fp_rate = float((flag & (fraud_any == 0)).sum()) / n_flag
        if precision >= targets.precision and fp_rate <= targets.legit_fp:
            recall = float((flag & (label == 1)).sum()) / n_pos
            if recall > best_recall:
                best_thr, best_recall = thr, recall
    return best_thr  # +inf when no point clears the bar -> flag nothing on the held-out ring


def evaluate_supervised(
    rows: list[CandidateRow],
    held_counts: dict[int, int],
    *,
    features: tuple[str, ...] = HEADLINE_FEATURES,
    model: str = "logreg",
    targets: Targets = DEFAULT_TARGETS,
) -> SupervisedResult:
    """Leave-one-ring-out evaluation of the distilled rule against the held-out rings."""
    x = _matrix(rows, features)
    label = np.array([r.label for r in rows], dtype=int)
    group = np.array([r.ring_id for r in rows], dtype=int)
    fraud_any = np.array([1 if r.is_fraud_any else 0 for r in rows], dtype=int)

    flagged_total = flagged_fraud = flagged_legit = recalled = 0
    for held in sorted(held_counts):
        tr, te = group != held, group == held
        if int(te.sum()) == 0:
            continue
        train_classes = set(label[tr].tolist())
        if len(train_classes) < 2:
            # Degenerate fold: the training candidate pool is single-class. At naive, infra linkage
            # retrieves ONLY true members (no legit/other-ring), so retrieval itself is the rule —
            # flag every retrieved candidate. If somehow no members were retrieved anywhere, flag none.
            flag = np.ones(int(te.sum()), dtype=bool) if 1 in train_classes else np.zeros(int(te.sum()), dtype=bool)
        else:
            p_tr, p_te = _fit_predict(x[tr], label[tr], x[te], model)
            thr = _choose_threshold(p_tr, label[tr], fraud_any[tr], targets)
            flag = p_te >= thr
        flagged_total += int(flag.sum())
        flagged_fraud += int((flag & (fraud_any[te] == 1)).sum())
        flagged_legit += int((flag & (fraud_any[te] == 0)).sum())
        recalled += int((flag & (label[te] == 1)).sum())

    n_pos = sum(held_counts.values())
    return SupervisedResult(
        model=model,
        features=features,
        recall=recalled / n_pos if n_pos else 0.0,
        operational_precision=flagged_fraud / flagged_total if flagged_total else 0.0,
        legit_fp_rate=flagged_legit / flagged_total if flagged_total else 0.0,
        flagged_total=flagged_total,
        n_positives=n_pos,
    )
