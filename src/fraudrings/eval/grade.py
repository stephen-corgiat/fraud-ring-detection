"""Grade the learned rule against the success targets.

Sweeps the evasion dial (re-generating per level/seed) and, on each world, grades the supervised
detector — Stage-1b co-target candidate generation + the learned basket rule — on **held-out
rings** against a fixed bar: precision ≥ 0.90 (operational), recall ≥ 0.80, legit-FP ≤ 5%.

Each level row carries the honest comparisons:
  * `baseline_recall_at_k` — the hand-weighted ranking (candidate-gen OFF), the baseline to beat;
  * `recall` / `operational_precision` / `legit_fp_rate` — the headline logreg basket rule;
  * `challenge_recall` — the same rule WITH the near-perfect challenge tell (ablation: how little it adds);
  * `lgbm_recall` — a gradient-boosted comparator (weaker here — overfits the small per-ring folds).

Candidate-gen is enabled only inside this supervised path (the hand-weighted path can't rank out
the genuine fans the wider net retrieves); the global engine default stays OFF.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field

import numpy as np

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import evaluate
from fraudrings.eval.supervised import (
    CHALLENGE_FEATURE,
    DEFAULT_TARGETS,
    HEADLINE_FEATURES,
    Targets,
    collect_labeled_candidates,
    evaluate_supervised,
)
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel

LEVELS: tuple[EvasionLevel, ...] = ("naive", "moderate", "sophisticated")


@dataclass
class LevelGrade:
    level: EvasionLevel
    seeds: list[int]
    recall: float  # headline logreg basket rule, held-out rings (ceiling-aware) — mean over seeds
    operational_precision: float
    legit_fp_rate: float
    baseline_recall_at_k: float  # baseline: hand-weighted ranking, candidate-gen OFF
    challenge_recall: float  # basket + challenge_fail (ablation)
    lgbm_recall: float  # gradient-boosted comparator
    targets: Targets
    # Per-seed values (one entry per seed, same order as `seeds`) so "meets" isn't decided by a
    # single noisy world — the headline numbers above are their means.
    recall_seeds: list[float] = field(default_factory=list)
    precision_seeds: list[float] = field(default_factory=list)
    fp_seeds: list[float] = field(default_factory=list)

    @property
    def meets(self) -> bool:
        return (
            self.recall >= self.targets.recall
            and self.operational_precision >= self.targets.precision
            and self.legit_fp_rate <= self.targets.legit_fp
        )

    def _seed_meets(self, r: float, p: float, fp: float) -> bool:
        return r >= self.targets.recall and p >= self.targets.precision and fp <= self.targets.legit_fp

    @property
    def n_seeds_meeting(self) -> int:
        """How many individual seeds clear the full bar (the honest, variance-aware 'meets')."""
        return sum(
            self._seed_meets(r, p, fp)
            for r, p, fp in zip(self.recall_seeds, self.precision_seeds, self.fp_seeds, strict=True)
        )

    @property
    def meets_all_seeds(self) -> bool:
        return bool(self.recall_seeds) and self.n_seeds_meeting == len(self.recall_seeds)

    @property
    def fp_max(self) -> float:  # worst-case legit-FP across seeds — the line to defend
        return max(self.fp_seeds) if self.fp_seeds else self.legit_fp_rate

    @property
    def recall_min(self) -> float:
        return min(self.recall_seeds) if self.recall_seeds else self.recall

    @property
    def recall_lift(self) -> float:
        return self.recall - self.baseline_recall_at_k

    def to_row(self) -> dict[str, object]:
        return {
            "level": self.level,
            "recall": round(self.recall, 4),
            "operational_precision": round(self.operational_precision, 4),
            "legit_fp_rate": round(self.legit_fp_rate, 4),
            "meets_targets": self.meets,
            "n_seeds_meeting": self.n_seeds_meeting,
            "n_seeds": len(self.seeds),
            "recall_min": round(self.recall_min, 4),
            "fp_max": round(self.fp_max, 4),
            "baseline_recall_at_k": round(self.baseline_recall_at_k, 4),
            "recall_lift": round(self.recall_lift, 4),
            "challenge_recall": round(self.challenge_recall, 4),
            "lgbm_recall": round(self.lgbm_recall, 4),
        }


@dataclass
class GradeResult:
    preset: str
    seeds: list[int]
    targets: Targets
    levels: list[LevelGrade]

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "seeds": list(self.seeds),
            "targets": {
                "precision": self.targets.precision,
                "recall": self.targets.recall,
                "legit_fp": self.targets.legit_fp,
            },
            "levels": [
                {
                    **lg.to_row(),
                    "seeds": list(lg.seeds),
                    "recall_per_seed": [round(x, 4) for x in lg.recall_seeds],
                    "precision_per_seed": [round(x, 4) for x in lg.precision_seeds],
                    "fp_per_seed": [round(x, 4) for x in lg.fp_seeds],
                }
                for lg in self.levels
            ],
        }

    def to_rows(self) -> list[dict[str, object]]:
        return [lg.to_row() for lg in self.levels]


def _grade_level(
    level: EvasionLevel, seeds: list[int], preset_name: str, data_dir: str, targets: Targets
) -> LevelGrade:
    base, rule, chal, lgbm = [], [], [], []
    op, fp = [], []
    for seed in seeds:
        run_dir = os.path.join(data_dir, f"{level}-{seed}")
        cfg = preset(preset_name)
        cfg.data_dir = run_dir
        cfg.sim.evasion_level = level
        cfg.sim.seed = seed
        events_path, gt_path = generate(cfg.sim, run_dir)
        con = build_db(events_path, cfg.engine, db_path=os.path.join(run_dir, "world.duckdb"))

        # baseline: hand-weighted ranking with candidate-gen OFF
        cfg.engine.cotarget_candidate_gen = False
        base.append(evaluate(con, gt_path, cfg.engine, cfg.eval).mean_recall_at_k)

        # supervised: candidate-gen ON (always-widen, so the super-fan confound is in the training folds
        # too — not only the held-out ring), learned rule on held-out rings.
        cfg.engine.cotarget_candidate_gen = True
        cfg.engine.cotarget_always = True
        rows, held = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)
        head = evaluate_supervised(rows, held, features=HEADLINE_FEATURES, model="logreg", targets=targets)
        rule.append(head.recall)
        op.append(head.operational_precision)
        fp.append(head.legit_fp_rate)
        chal.append(
            evaluate_supervised(
                rows, held, features=HEADLINE_FEATURES + (CHALLENGE_FEATURE,), model="logreg", targets=targets
            ).recall
        )
        lgbm.append(evaluate_supervised(rows, held, features=HEADLINE_FEATURES, model="lgbm", targets=targets).recall)

    mean = lambda xs: float(np.mean(xs)) if xs else 0.0  # noqa: E731
    return LevelGrade(
        level=level,
        seeds=list(seeds),
        recall=mean(rule),
        operational_precision=mean(op),
        legit_fp_rate=mean(fp),
        baseline_recall_at_k=mean(base),
        challenge_recall=mean(chal),
        lgbm_recall=mean(lgbm),
        targets=targets,
        recall_seeds=rule,
        precision_seeds=op,
        fp_seeds=fp,
    )


def _write(result: GradeResult, data_dir: str) -> tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "rule_grades.json")
    csv_path = os.path.join(data_dir, "rule_grades.csv")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    rows = result.to_rows()
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path


def grade(
    levels: list[EvasionLevel] | tuple[EvasionLevel, ...] = LEVELS,
    seeds: list[int] | tuple[int, ...] = (7, 13, 21),
    preset_name: str = "default",
    data_dir: str = "data",
    targets: Targets = DEFAULT_TARGETS,
) -> GradeResult:
    """Grade the learned rule across the evasion dial; also writes data_dir/rule_grades.{json,csv}."""
    seeds = list(seeds)
    grades = [_grade_level(lvl, seeds, preset_name, data_dir, targets) for lvl in levels]
    result = GradeResult(preset=preset_name, seeds=seeds, targets=targets, levels=grades)
    _write(result, data_dir)
    return result
