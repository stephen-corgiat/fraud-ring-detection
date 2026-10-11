"""Score the seedless concentration ranker against the hidden ground truth.

The ranker (`detector.concentration.generate_suspects`) reads only `events` and emits a shortlist of
suspect *creators*. Eval knows which creator each ring was paid to boost: the creator its members
pour the most qualified views into. This module reports where those ring targets land on the
shortlist. A ring target that lands near the head is a seed source the investigator can start from.

Reported per world:
  * target_ranks    — 1-based shortlist rank of each ring target (absent targets are omitted)
  * recall_at_k     — fraction of ring targets in the top k (k = 2 × #ring targets by default)
  * mean_rank       — mean rank of the ring targets that made the shortlist
  * benign_above    — shortlisted non-target creators ranked above the worst-ranked ring target
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Sequence
from dataclasses import dataclass

import duckdb
import polars as pl

from fraudrings.config import ConcentrationConfig, preset
from fraudrings.detector.concentration import SuspectTarget, generate_suspects
from fraudrings.detector.ingest import build_db
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel


@dataclass(frozen=True)
class ConcentrationResult:
    n_targets: int
    n_suspects: int
    target_ranks: tuple[int, ...]
    k: int
    recall_at_k: float
    mean_rank: float
    benign_above: int


def ring_target_map(events_path: str, gt_path: str) -> dict[int, list[int]]:
    """creator_id -> the rings boosting it. Each ring's target is the creator its members put the
    most views into; several rings may share one target."""
    ev = pl.read_parquet(events_path).filter(pl.col("creator_id").is_not_null())
    gt = pl.read_parquet(gt_path).filter(pl.col("is_fraud"))
    per = (
        ev.join(gt.select("account_id", "ring_id"), on="account_id")
        .group_by("ring_id", "creator_id")
        .len()
        .sort(["ring_id", "len", "creator_id"], descending=[False, True, False])
    )
    out: dict[int, list[int]] = {}
    for ring_id, creator_id in per.group_by("ring_id", maintain_order=True).first().select(
        "ring_id", "creator_id"
    ).iter_rows():
        out.setdefault(int(creator_id), []).append(int(ring_id))
    return out


def ring_targets(events_path: str, gt_path: str) -> set[int]:
    """The set of creators some ring was paid to boost."""
    return set(ring_target_map(events_path, gt_path))


def score_shortlist(suspects: list[SuspectTarget], targets: set[int], k: int | None = None) -> ConcentrationResult:
    """Grade a shortlist against the known ring targets. Pure; no database access."""
    ids = [s.target_id for s in suspects]
    pos = {t: i + 1 for i, t in enumerate(ids)}
    ranks = tuple(sorted(pos[t] for t in targets if t in pos))
    k = k if k is not None else 2 * len(targets)
    worst = max(ranks) if ranks else 0
    return ConcentrationResult(
        n_targets=len(targets),
        n_suspects=len(ids),
        target_ranks=ranks,
        k=k,
        recall_at_k=sum(r <= k for r in ranks) / len(targets) if targets else 0.0,
        mean_rank=sum(ranks) / len(ranks) if ranks else float("nan"),
        benign_above=sum(1 for t in ids[:worst] if t not in targets),
    )


def evaluate_concentration(
    con: duckdb.DuckDBPyConnection,
    events_path: str,
    gt_path: str,
    cfg: ConcentrationConfig | None = None,
) -> ConcentrationResult:
    """Run the ranker on a built world (events only) and grade it against the ring targets."""
    return score_shortlist(generate_suspects(con, cfg), ring_targets(events_path, gt_path))


@dataclass(frozen=True)
class ConcentrationLevel:
    """The ranker's grade at one evasion level, averaged over seeds."""

    level: str
    seeds: tuple[int, ...]
    recall_at_k: float       # mean over seeds
    mean_rank: float         # mean over seeds of the mean ring-target rank
    worst_rank: int          # worst ring-target rank seen at this level (across seeds)
    n_suspects: float        # mean shortlist length

    def to_dict(self) -> dict:
        return {
            "level": self.level, "seeds": list(self.seeds),
            "recall_at_k": round(self.recall_at_k, 4), "mean_rank": round(self.mean_rank, 2),
            "worst_rank": self.worst_rank, "n_suspects": round(self.n_suspects, 1),
        }


def concentration_sweep(
    levels: Sequence[EvasionLevel],
    seeds: Sequence[int],
    preset_name: str = "default",
    data_dir: str = "data",
    cfg: ConcentrationConfig | None = None,
) -> list[ConcentrationLevel]:
    """Grade the seedless ranker across the evasion dial; writes data_dir/concentration.{json,csv}."""
    out: list[ConcentrationLevel] = []
    for level in levels:
        results: list[ConcentrationResult] = []
        for seed in seeds:
            run_cfg = preset(preset_name)
            run_cfg.sim.evasion_level = level
            run_cfg.sim.seed = seed
            run_dir = os.path.join(data_dir, "concentration", f"{level}-{seed}")
            events_path, gt_path = generate(run_cfg.sim, run_dir)
            con = build_db(events_path, run_cfg.engine, db_path=os.path.join(run_dir, "world.duckdb"))
            results.append(evaluate_concentration(con, events_path, gt_path, cfg or run_cfg.concentration))
            con.close()
        out.append(ConcentrationLevel(
            level=level, seeds=tuple(seeds),
            recall_at_k=sum(r.recall_at_k for r in results) / len(results),
            mean_rank=sum(r.mean_rank for r in results) / len(results),
            worst_rank=max((max(r.target_ranks) for r in results if r.target_ranks), default=0),
            n_suspects=sum(r.n_suspects for r in results) / len(results),
        ))
    os.makedirs(data_dir, exist_ok=True)
    rows = [lv.to_dict() for lv in out]
    with open(os.path.join(data_dir, "concentration.json"), "w") as f:
        json.dump(rows, f, indent=2)
    with open(os.path.join(data_dir, "concentration.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows({**r, "seeds": " ".join(map(str, r["seeds"]))} for r in rows)
    return out
