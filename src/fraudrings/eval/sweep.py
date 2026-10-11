"""The sweep harness and the two curves.

The evasion dial *moves* the metrics on a single world; this sweep turns the dial across
`naive → moderate → sophisticated` (re-generating a fresh world per level, optionally
averaging a few seeds to tighten the curve) and emits **two** curves vs. sophistication:

  (a) **per-ring ranking** — mean AP / Stage-1 recall / recall@k / precision@k. This is the
      headline that degrades as the adversary blends its shared-infrastructure footprint into
      the legit population.
  (b) **operational fraud-vs-legit** — of the flagged slots a real investigation would action,
      what fraction are fraud at all (operational precision) vs. real viewers (the legit
      false-positive line). The `eval/evaluate.py` decomposition is the kernel.

The load-bearing finding: (a) slopes down while (b) holds ~1.0 — the precision
loss under evasion is the engine confusing one fraud ring for *another* (still fraud), not
flagging legit users. The sweep reports that decomposition honestly rather than letting the
single AP number imply the operational line moved.

The detector still never sees `evasion_level` — it lives only in ground_truth, read by the
oracle (`eval/`). Anti-tautology isolation is intact.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass

import numpy as np

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import EvalResult, evaluate
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel

# Default dial positions, in increasing sophistication — one world (one curve point) each.
LEVELS: tuple[EvasionLevel, ...] = ("naive", "moderate", "sophisticated")


@dataclass
class LevelSummary:
    """One point on both curves: the metrics at a single sophistication level, aggregated
    over the swept seeds. Ranking metrics are *averaged* over seeds; the operational
    flagged-slot counts are *summed* (then turned into fractions) so a level's false-positive
    line reflects every slot the loop would have actioned across the seeds."""

    level: EvasionLevel
    seeds: list[int]

    # (a) per-ring ranking, mean over rings then mean over seeds
    mean_ap: float
    mean_stage1_recall: float
    mean_recall_at_k: float
    mean_precision_at_k: float

    # (b) operational decomposition, summed flagged slots over rings and seeds
    flagged_total: int
    flagged_this_ring: int
    flagged_other_ring: int
    flagged_legit: int

    @property
    def operational_precision(self) -> float:
        """Flagged slots that are fraud at all (this ring or another) / total."""
        return (self.flagged_total - self.flagged_legit) / self.flagged_total if self.flagged_total else 0.0

    @property
    def legit_fp_rate(self) -> float:
        """Flagged slots that are real viewers / total — the false-positive line."""
        return self.flagged_legit / self.flagged_total if self.flagged_total else 0.0

    @property
    def cross_ring_frac(self) -> float:
        """Flagged slots that are a *different* ring's fraud / total — the bleed that deflates
        per-ring AP without being a legit false positive."""
        return self.flagged_other_ring / self.flagged_total if self.flagged_total else 0.0

    def to_row(self) -> dict[str, object]:
        """A flat, CSV/table-friendly view (rounded floats) — the curve as one row."""
        return {
            "level": self.level,
            "n_seeds": len(self.seeds),
            "mean_ap": round(self.mean_ap, 4),
            "mean_stage1_recall": round(self.mean_stage1_recall, 4),
            "mean_recall_at_k": round(self.mean_recall_at_k, 4),
            "mean_precision_at_k": round(self.mean_precision_at_k, 4),
            "operational_precision": round(self.operational_precision, 4),
            "legit_fp_rate": round(self.legit_fp_rate, 4),
            "cross_ring_frac": round(self.cross_ring_frac, 4),
            "flagged_total": self.flagged_total,
        }

    def to_dict(self) -> dict[str, object]:
        """Full serializable view: the flat row plus the raw seeds and slot counts."""
        d = self.to_row()
        d["seeds"] = list(self.seeds)
        d["flagged_this_ring"] = self.flagged_this_ring
        d["flagged_other_ring"] = self.flagged_other_ring
        d["flagged_legit"] = self.flagged_legit
        return d


@dataclass
class SweepResult:
    preset: str
    seeds: list[int]
    levels: list[LevelSummary]

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "seeds": list(self.seeds),
            "levels": [s.to_dict() for s in self.levels],
        }

    def to_rows(self) -> list[dict[str, object]]:
        return [s.to_row() for s in self.levels]


def _summarize(level: EvasionLevel, seeds: list[int], results: list[EvalResult]) -> LevelSummary:
    """Collapse the per-seed EvalResults at one level into a single curve point."""
    return LevelSummary(
        level=level,
        seeds=list(seeds),
        mean_ap=float(np.mean([r.mean_ap for r in results])),
        mean_stage1_recall=float(np.mean([r.mean_stage1_recall for r in results])),
        mean_recall_at_k=float(np.mean([r.mean_recall_at_k for r in results])),
        mean_precision_at_k=float(np.mean([r.mean_precision_at_k for r in results])),
        flagged_total=sum(r.flagged_total for r in results),
        flagged_this_ring=sum(rr.flagged_this_ring for r in results for rr in r.rings),
        flagged_other_ring=sum(rr.flagged_other_ring for r in results for rr in r.rings),
        flagged_legit=sum(rr.flagged_legit for r in results for rr in r.rings),
    )


def _write(result: SweepResult, data_dir: str) -> tuple[str, str]:
    """Persist the curve for downstream reporting/plotting: JSON (full) + CSV (flat rows)."""
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "sweep.json")
    csv_path = os.path.join(data_dir, "sweep.csv")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    rows = result.to_rows()
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def sweep(
    levels: list[EvasionLevel] | tuple[EvasionLevel, ...] = LEVELS,
    seeds: list[int] | tuple[int, ...] = (7, 13, 21),
    preset_name: str = "default",
    data_dir: str = "data",
) -> SweepResult:
    """Run `generate → build_db → evaluate` across `levels` × `seeds` and collect both curves.

    A fresh world is generated per (level, seed) — one level per world,
    so the sweep is the only place sophistication varies. Each run gets its own subdirectory so
    the worlds and DuckDB files don't collide. The aggregated `SweepResult` is also written to
    `data_dir/sweep.{json,csv}`.
    """
    seeds = list(seeds)
    summaries: list[LevelSummary] = []
    for level in levels:
        per_seed: list[EvalResult] = []
        for seed in seeds:
            run_dir = os.path.join(data_dir, f"{level}-{seed}")
            cfg = preset(preset_name)
            cfg.data_dir = run_dir
            cfg.sim.evasion_level = level
            cfg.sim.seed = seed
            events_path, gt_path = generate(cfg.sim, run_dir)
            con = build_db(events_path, cfg.engine, db_path=os.path.join(run_dir, "world.duckdb"))
            per_seed.append(evaluate(con, gt_path, cfg.engine, cfg.eval))
        summaries.append(_summarize(level, seeds, per_seed))

    result = SweepResult(preset=preset_name, seeds=seeds, levels=summaries)
    _write(result, data_dir)
    return result
