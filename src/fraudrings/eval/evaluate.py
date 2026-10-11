"""Evaluation protocol.

For each injected ring: hand the loop K labeled seeds, ask it to recover the rest,
and measure ranking quality against the hidden ground truth. Without the simulator's
hidden labels this number cannot be produced at all (no labels => no measurement).

Reported per ring and averaged:
  * stage1_recall  — fraction of held-out ring members Stage-1 even retrieved
                     (a ceiling on recall; the rarity gate trades this for precision)
  * ap             — average precision of the Stage-2 ranking over retrieved candidates
  * recall@k       — held-out members found in the top-k (k = #held-out)
  * precision@k    — fraction of the top-k that are truly in the ring
"""

from __future__ import annotations

from dataclasses import dataclass, field

import duckdb
import numpy as np
import polars as pl

from fraudrings.config import EngineConfig, EvalConfig
from fraudrings.detector.engine import Candidate, run_investigation


def average_precision(y_score: np.ndarray, y_true: np.ndarray) -> float:
    if y_true.sum() == 0:
        return 0.0
    order = np.argsort(-y_score, kind="stable")
    yt = y_true[order]
    cum_tp = np.cumsum(yt)
    ranks = np.arange(1, len(yt) + 1)
    precision_at = cum_tp / ranks
    return float((precision_at * yt).sum() / yt.sum())


@dataclass
class RingResult:
    ring_id: int
    n_members: int
    n_seeds: int
    n_held_out: int
    n_candidates: int
    stage1_recall: float
    ap: float
    recall_at_k: float
    precision_at_k: float
    # Operational decomposition of the FLAGGED set (the top-k candidates, k = #held-out,
    # that a real investigation would action). Each slot lands in exactly one bucket so the
    # three partition `flagged_total`. The split tells apart the two stories the single AP
    # number conflates: cross-ring bleed is a per-ring miss but NOT a legit false positive,
    # whereas `flagged_legit` is the false-positive line that actually matters operationally.
    flagged_total: int = 0
    flagged_this_ring: int = 0  # true positive for THIS ring
    flagged_other_ring: int = 0  # a different ring's fraud — still fraud, operationally a TP
    flagged_legit: int = 0  # a real viewer / benign-coordination control — the real FP
    top: list[Candidate] = field(default_factory=list)


@dataclass
class EvalResult:
    rings: list[RingResult]

    @property
    def mean_ap(self) -> float:
        return float(np.mean([r.ap for r in self.rings])) if self.rings else 0.0

    @property
    def mean_stage1_recall(self) -> float:
        return float(np.mean([r.stage1_recall for r in self.rings])) if self.rings else 0.0

    @property
    def mean_recall_at_k(self) -> float:
        return float(np.mean([r.recall_at_k for r in self.rings])) if self.rings else 0.0

    @property
    def mean_precision_at_k(self) -> float:
        return float(np.mean([r.precision_at_k for r in self.rings])) if self.rings else 0.0

    # --- Operational curve: fraud-vs-legit, summed over flagged slots ----------
    @property
    def flagged_total(self) -> int:
        return sum(r.flagged_total for r in self.rings)

    @property
    def operational_precision(self) -> float:
        """Fraction of flagged slots that are fraud at ALL (this ring or another). The
        operational success line — distinct from per-ring AP, which cross-ring bleed deflates."""
        total = self.flagged_total
        if not total:
            return 0.0
        legit = sum(r.flagged_legit for r in self.rings)
        return (total - legit) / total

    @property
    def legit_fp_rate(self) -> float:
        """Fraction of flagged slots that are real viewers — the false-positive line."""
        total = self.flagged_total
        return sum(r.flagged_legit for r in self.rings) / total if total else 0.0

    @property
    def cross_ring_frac(self) -> float:
        """Fraction of flagged slots that are a *different* ring's fraud (the bleed that
        deflates per-ring AP without being a legit FP)."""
        total = self.flagged_total
        return sum(r.flagged_other_ring for r in self.rings) / total if total else 0.0


def evaluate(
    con: duckdb.DuckDBPyConnection,
    ground_truth_path: str,
    engine: EngineConfig,
    eval_cfg: EvalConfig,
) -> EvalResult:
    gt = pl.read_parquet(ground_truth_path)
    fraud = gt.filter(pl.col("is_fraud") & pl.col("ring_id").is_not_null())

    # Oracle-only: the full fraud roster, so a flagged candidate can be
    # bucketed this-ring / other-ring-fraud / legit. The detector never sees this.
    all_fraud_ids = {int(a) for a in gt.filter(pl.col("is_fraud"))["account_id"].to_list()}

    ring_ids = sorted({int(r) for r in fraud["ring_id"].to_list()})
    results: list[RingResult] = []

    for ring_id in ring_ids:
        members = sorted(
            int(a) for a in fraud.filter(pl.col("ring_id") == ring_id)["account_id"].to_list()
        )
        k = eval_cfg.seeds_per_ring
        if len(members) <= k + 1:
            continue
        seeds = members[:k]
        positives = set(members[k:])
        n_pos = len(positives)

        candidates = run_investigation(con, seeds, engine)
        ranked_ids = [c.account_id for c in candidates]
        scores = np.array([c.score for c in candidates], dtype=float)
        y_true = np.array([1 if a in positives else 0 for a in ranked_ids], dtype=float)

        retrieved_pos = int(y_true.sum())
        stage1_recall = retrieved_pos / n_pos if n_pos else 0.0
        ap = average_precision(scores, y_true) if len(scores) else 0.0
        topk = y_true[:n_pos]
        recall_at_k = float(topk.sum() / n_pos) if n_pos else 0.0
        precision_at_k = float(topk.sum() / len(topk)) if len(topk) else 0.0

        # Operational decomposition of the flagged set = top-k candidates (k = #held-out).
        flagged = ranked_ids[:n_pos]
        this_ring = sum(a in positives for a in flagged)
        other_ring = sum(a not in positives and a in all_fraud_ids for a in flagged)
        legit = len(flagged) - this_ring - other_ring

        results.append(
            RingResult(
                ring_id=ring_id,
                n_members=len(members),
                n_seeds=k,
                n_held_out=n_pos,
                n_candidates=len(candidates),
                stage1_recall=stage1_recall,
                ap=ap,
                recall_at_k=recall_at_k,
                precision_at_k=precision_at_k,
                flagged_total=len(flagged),
                flagged_this_ring=this_ring,
                flagged_other_ring=other_ring,
                flagged_legit=legit,
                top=candidates[:8],
            )
        )

    return EvalResult(rings=results)
