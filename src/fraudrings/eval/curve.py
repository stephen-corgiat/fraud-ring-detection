"""The learning curve (metric vs. #labels) + label-noise sensitivity.

The grade (`eval/grade.py`) answers "does the loop work?" This answers the question that is
impossible without ground truth: **what does it cost?** — *how many oracle labels* must a human
bench confirm before the learned rule clears the bar, and *how robust* is that to oracle error.

Design:

  * **x-axis** = number of **labeled candidates** the oracle confirms to *train* the rule, drawn
    from a candidate pool that is **frozen** (built once per world in the supervised regime:
    `cotarget_candidate_gen=True, cotarget_always=True`). The budget buys *training labels*, not
    retrieval (closed-loop re-retrieval is out of scope).
  * Budget is **per-fold**: under leave-one-ring-out, each held-out ring is graded by a rule
    trained on a budget of labels drawn from *that fold's* training pool.
  * Three **query strategies**: `random` (the baseline), `uncertainty` (batched active learning —
    label what the model is least sure about), and `exploit` (label what it thinks is most likely
    fraud). At default scale uncertainty halves the clean-label cost; exploit, by labeling the
    easy/clearest cases, carries less information per label.
  * **Label noise**: the oracle (`eval/oracle.py`) corrupts **training labels only** (held-out
    grading always uses true ground truth). The symmetric headline sweep is ε ∈ {0, .05, .10, .20};
    `asymmetric` profiles model the realistic missed-fraud-dominant case, which isolates the finding
    that the fragile target is *precision* (driven by false confirmations), not recall.
  * **labels_to_target** = the *stable crossing*: the smallest budget at which all three targets
    hold *and keep holding* for every larger budget tested, on the seed-averaged curve.

**Continuity anchor (the key correctness guarantee):** at ε=0 and the full budget, every fold
trains on its entire clean pool, so a curve point reduces *exactly* to `evaluate_supervised` — the
top of the curve reproduces the grade. Noise and budget are perturbations that vanish in
that corner; `tests/test_curve.py` locks the reconciliation.
"""

from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from fraudrings.config import RunConfig, preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.oracle import noisy_labels
from fraudrings.eval.supervised import (
    DEFAULT_TARGETS,
    HEADLINE_FEATURES,
    Targets,
    _choose_threshold,
    _fit_predict,
    _matrix,
    collect_labeled_candidates,
)
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel

LEVELS: tuple[EvasionLevel, ...] = ("naive", "moderate", "sophisticated")
DEFAULT_EPSILONS: tuple[float, ...] = (0.0, 0.05, 0.10, 0.20)
# random first (the baseline), then the two label-spending strategies:
#   uncertainty — label what the model is least sure about (the hard boundary);
#   exploit     — label what the model thinks is most likely fraud (clearest cases).
DEFAULT_STRATEGIES: tuple[str, ...] = ("random", "uncertainty", "exploit")
DEFAULT_BATCH: int = 8  # active-learning query batch size


@dataclass(frozen=True)
class NoiseSpec:
    """An oracle error profile: the two class-conditional flip rates the labeler applies to TRAINING
    labels. The symmetric headline sweep sets both equal to ε; the asymmetric profiles model the
    realistic case where missed fraud dominates false confirmations (accounts that age quietly look
    legit until they act)."""

    fraud_to_legit: float  # a true ring member the analyst wrongly CLEARS (the realistic, dominant error)
    legit_to_fraud: float  # an innocent account the analyst wrongly CONFIRMS as fraud

    @property
    def is_clean(self) -> bool:
        return self.fraud_to_legit == 0.0 and self.legit_to_fraud == 0.0

    @property
    def is_symmetric(self) -> bool:
        return self.fraud_to_legit == self.legit_to_fraud

    @property
    def label(self) -> str:
        if self.is_symmetric:
            return f"ε={self.fraud_to_legit:g}"
        return f"f→l={self.fraud_to_legit:g}/l→f={self.legit_to_fraud:g}"


def _symmetric(epsilon: float) -> NoiseSpec:
    return NoiseSpec(fraud_to_legit=epsilon, legit_to_fraud=epsilon)


CLEAN = _symmetric(0.0)
# Realistic asymmetric profiles (missed-fraud dominant): a pure missed-fraud run isolates the effect
# on recall, and a realistic blend keeps false-confirmations low — the test of whether the fragile
# precision line survives when the oracle errs the way real analysts do.
DEFAULT_ASYMMETRIC: tuple[NoiseSpec, ...] = (
    NoiseSpec(fraud_to_legit=0.20, legit_to_fraud=0.0),
    NoiseSpec(fraud_to_legit=0.20, legit_to_fraud=0.05),
)


# --------------------------------------------------------------------------- world cache


@dataclass
class _World:
    """A frozen candidate pool for one (level, seed) — the supervised arrays `evaluate_supervised`
    consumes, cached so every strategy/ε/budget reuses the same retrieval (Q2: retrieval is frozen)."""

    seed: int
    level: EvasionLevel
    x: np.ndarray
    label: np.ndarray  # per-candidate: held-out-member-of-this-ring (the recall target / noise victim)
    group: np.ndarray  # ring_id (the leave-one-ring-out fold key)
    fraud_any: np.ndarray  # fraud in ANY ring — the clean operational-precision basis
    held_counts: dict[int, int]
    features: tuple[str, ...]

    def max_train_pool(self) -> int:
        """Largest leave-one-ring-out training pool — the top of the per-fold budget grid."""
        return max(int((self.group != held).sum()) for held in self.held_counts)


def _load_world(
    seed: int,
    level: EvasionLevel,
    preset_name: str,
    data_dir: str,
    configure: Callable[[RunConfig], None] | None,
) -> _World:
    cfg = preset(preset_name)
    run_dir = os.path.join(data_dir, f"{level}-{seed}")
    cfg.data_dir = run_dir
    cfg.sim.evasion_level = level
    cfg.sim.seed = seed
    # Supervised regime: widen retrieval for every investigation so the benign super-fan confound is
    # in the training folds, not only the held-out one (see grade.py).
    cfg.engine.cotarget_candidate_gen = True
    cfg.engine.cotarget_always = True
    if configure is not None:
        configure(cfg)
    events_path, gt_path = generate(cfg.sim, run_dir)
    con = build_db(events_path, cfg.engine, db_path=os.path.join(run_dir, "world.duckdb"))
    rows, held_counts = collect_labeled_candidates(con, gt_path, cfg.engine, cfg.eval)
    return _World(
        seed=seed,
        level=level,
        x=_matrix(rows, HEADLINE_FEATURES),
        label=np.array([r.label for r in rows], dtype=int),
        group=np.array([r.ring_id for r in rows], dtype=int),
        fraud_any=np.array([1 if r.is_fraud_any else 0 for r in rows], dtype=int),
        held_counts=held_counts,
        features=HEADLINE_FEATURES,
    )


# --------------------------------------------------------------------------- seeding helpers


def _rng_seed(seed: int, held: int) -> int:
    """A reproducible per-(world, fold) RNG seed for label-selection order (no wall-clock)."""
    return int(seed) * 7919 + int(held) * 31 + 1


def _noise_seed(seed: int, noise: NoiseSpec) -> int:
    """A reproducible per-(world, noise-profile) RNG seed for the oracle's flips (no wall-clock)."""
    return (
        int(seed) * 1_000_003
        + int(round(noise.fraud_to_legit * 100_000)) * 131
        + int(round(noise.legit_to_fraud * 100_000))
    )


# --------------------------------------------------------------------------- label-query strategies


def _random_order(world: _World, held: int) -> np.ndarray:
    """A reproducible random permutation of the fold's training-pool row indices. A budget of B
    labels is then the first B of this order — so larger budgets *nest* the smaller ones (less curve
    noise) while staying a genuine random sample."""
    tr_idx = np.flatnonzero(world.group != held)
    return np.random.default_rng(_rng_seed(world.seed, held)).permutation(tr_idx)


# Acquisition criteria for the batched active-learning strategies: given the model's fraud
# probabilities on the unlabeled pool, return the indices to query *first* (most informative first).
#   uncertainty — smallest |p−0.5| (the model is least sure → the hard fan/member boundary);
#   exploit     — largest p (the model is most sure it's fraud → label the clearest cases).
_ACQUIRE = {
    "uncertainty": lambda p: np.argsort(np.abs(p - 0.5)),
    "exploit": lambda p: np.argsort(p)[::-1],
}


def _active_order(
    world: _World, held: int, noisy_label: np.ndarray, batch: int, model: str, criterion: str
) -> np.ndarray:
    """Batched active learning. Bootstrap a random first batch (the model needs *some* labels before
    it can rank anything), then repeatedly train, score the remaining pool, and acquire the next batch
    by `criterion`. Returns the full acquisition order; a budget of B is its first B (so budgets nest,
    exactly as for random)."""
    rank = _ACQUIRE[criterion]
    tr_idx = np.flatnonzero(world.group != held)
    perm = np.random.default_rng(_rng_seed(world.seed, held)).permutation(tr_idx)
    order: list[int] = list(perm[:batch])
    remaining: list[int] = list(perm[batch:])
    x = world.x
    while remaining:
        sel = np.sort(np.array(order, dtype=int))
        y = noisy_label[sel]
        if len(set(y.tolist())) < 2:
            # The model can't rank without both classes yet — keep drawing randomly (the bootstrap
            # continues) until a positive turns up.
            take = remaining[:batch]
        else:
            rem = np.array(remaining, dtype=int)
            _, p_rem = _fit_predict(x[sel], y, x[rem], model)
            take = [remaining[i] for i in rank(p_rem)[:batch].tolist()]
        order.extend(take)
        taken = set(take)
        remaining = [r for r in remaining if r not in taken]
    return np.array(order, dtype=int)


def _fold_orders(
    world: _World, strategy: str, noisy_label: np.ndarray, batch: int, model: str
) -> dict[int, np.ndarray]:
    if strategy == "random":
        return {held: _random_order(world, held) for held in world.held_counts}
    if strategy in _ACQUIRE:
        return {
            held: _active_order(world, held, noisy_label, batch, model, strategy)
            for held in world.held_counts
        }
    raise ValueError(f"unknown strategy {strategy!r} (expected one of random/{'/'.join(_ACQUIRE)})")


# --------------------------------------------------------------------------- one curve point


def _eval_point(
    world: _World,
    noisy_label: np.ndarray,
    orders: dict[int, np.ndarray],
    budget: int,
    targets: Targets,
    model: str,
) -> tuple[float, float, float]:
    """Leave-one-ring-out evaluation under a per-fold label budget. Mirrors `evaluate_supervised`'s
    fold body so that at the full budget with clean labels it reproduces it exactly: the only changes
    are (a) the rule trains on the *budgeted* subset of rows and (b) on their *oracle* (possibly
    flipped) labels. Threshold-selection precision uses the CLEAN `fraud_any`, and the held-out fold
    is always graded against clean ground truth."""
    x, label, group, fraud_any = world.x, world.label, world.group, world.fraud_any
    flagged_total = flagged_fraud = flagged_legit = recalled = 0
    for held in sorted(world.held_counts):
        te = group == held
        n_te = int(te.sum())
        if n_te == 0:
            continue
        # The budgeted rows. Sorting the selection (it stays the same SET) makes the full-budget case
        # bit-identical to evaluate_supervised's x[tr] order — the exact reconciliation anchor.
        sel = np.sort(orders[held][:budget])
        y_sel = noisy_label[sel]
        classes = set(y_sel.tolist())
        if len(classes) < 2:
            # Single-class budget: same degenerate guard as evaluate_supervised — flag all of the
            # held-out fold if the only label the oracle gave is fraud, else flag none.
            flag = np.ones(n_te, dtype=bool) if 1 in classes else np.zeros(n_te, dtype=bool)
        else:
            p_tr, p_te = _fit_predict(x[sel], y_sel, x[te], model)
            thr = _choose_threshold(p_tr, y_sel, fraud_any[sel], targets)
            flag = p_te >= thr
        flagged_total += int(flag.sum())
        flagged_fraud += int((flag & (fraud_any[te] == 1)).sum())
        flagged_legit += int((flag & (fraud_any[te] == 0)).sum())
        recalled += int((flag & (label[te] == 1)).sum())

    n_pos = sum(world.held_counts.values())
    return (
        recalled / n_pos if n_pos else 0.0,
        flagged_fraud / flagged_total if flagged_total else 0.0,
        flagged_legit / flagged_total if flagged_total else 0.0,
    )


# --------------------------------------------------------------------------- result types


@dataclass
class CurvePoint:
    n_labels: int  # per-fold label budget (the full point = the largest training pool)
    is_full: bool
    recall: float  # seed-averaged held-out metrics
    operational_precision: float
    legit_fp_rate: float
    targets: Targets
    recall_seeds: list[float] = field(default_factory=list)
    precision_seeds: list[float] = field(default_factory=list)
    fp_seeds: list[float] = field(default_factory=list)

    def meets(self) -> bool:
        return (
            self.recall >= self.targets.recall
            and self.operational_precision >= self.targets.precision
            and self.legit_fp_rate <= self.targets.legit_fp
        )

    def to_row(self) -> dict[str, object]:
        return {
            "n_labels": self.n_labels,
            "is_full": self.is_full,
            "recall": round(self.recall, 4),
            "operational_precision": round(self.operational_precision, 4),
            "legit_fp_rate": round(self.legit_fp_rate, 4),
            "meets_targets": self.meets(),
        }


@dataclass
class CurveSeries:
    strategy: str
    noise: NoiseSpec
    level: EvasionLevel
    points: list[CurvePoint]
    labels_to_target: int | None  # stable crossing (n_labels), None if never stably crossed

    def _noise_cols(self) -> dict[str, object]:
        return {
            "noise": self.noise.label,
            "fraud_to_legit": self.noise.fraud_to_legit,
            "legit_to_fraud": self.noise.legit_to_fraud,
        }

    def to_rows(self) -> list[dict[str, object]]:
        return [{"strategy": self.strategy, **self._noise_cols(), **p.to_row()} for p in self.points]

    def to_dict(self) -> dict[str, object]:
        return {
            "strategy": self.strategy,
            **self._noise_cols(),
            "level": self.level,
            "labels_to_target": self.labels_to_target,
            "points": [
                {
                    **p.to_row(),
                    "recall_per_seed": [round(x, 4) for x in p.recall_seeds],
                    "precision_per_seed": [round(x, 4) for x in p.precision_seeds],
                    "fp_per_seed": [round(x, 4) for x in p.fp_seeds],
                }
                for p in self.points
            ],
        }


@dataclass
class CurveResult:
    preset: str
    seeds: list[int]
    targets: Targets
    level: EvasionLevel  # the level the full (strategy × noise) curves are drawn at
    series: list[CurveSeries]
    labels_to_target_by_level: dict[str, int | None]  # cheap per-level summary (random, clean labels)

    def to_rows(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for s in self.series:
            rows.extend(s.to_rows())
        return rows

    def to_dict(self) -> dict[str, object]:
        return {
            "preset": self.preset,
            "seeds": list(self.seeds),
            "level": self.level,
            "targets": {
                "precision": self.targets.precision,
                "recall": self.targets.recall,
                "legit_fp": self.targets.legit_fp,
            },
            "labels_to_target_by_level": self.labels_to_target_by_level,
            "series": [s.to_dict() for s in self.series],
        }


def _stable_crossing(points: list[CurvePoint], targets: Targets) -> int | None:
    """Smallest n_labels that meets all targets and stays met for every larger budget (Q5). Points
    are assumed sorted by budget ascending (the full point last)."""
    for i, p in enumerate(points):
        if all(points[j].meets() for j in range(i, len(points))):
            return p.n_labels
    return None


def _budget_grid(max_pool: int, start: int = 5, factor: int = 2) -> list[int]:
    """Geometric grid 5/10/20/40/… capped below `max_pool`, with the full pool as the top point."""
    grid: list[int] = []
    b = start
    while b < max_pool:
        grid.append(b)
        b *= factor
    grid.append(max_pool)
    return grid


def _mean(xs: list[float]) -> float:
    return float(np.mean(xs)) if xs else 0.0


def _build_series(
    worlds: dict[int, _World],
    seeds: list[int],
    grid: list[int],
    max_pool: int,
    strategy: str,
    noise: NoiseSpec,
    batch: int,
    model: str,
    targets: Targets,
) -> CurveSeries:
    # Per seed: the (possibly-noisy) oracle labels and the strategy's acquisition order, computed
    # once and reused across every budget on the grid.
    prepared: list[tuple[_World, np.ndarray, dict[int, np.ndarray]]] = []
    for seed in seeds:
        w = worlds[seed]
        noisy = noisy_labels(
            w.label,
            fraud_to_legit=noise.fraud_to_legit,
            legit_to_fraud=noise.legit_to_fraud,
            seed=_noise_seed(seed, noise),
        )
        prepared.append((w, noisy, _fold_orders(w, strategy, noisy, batch, model)))

    points: list[CurvePoint] = []
    for budget in grid:
        rs, ops, fps = [], [], []
        for w, noisy, orders in prepared:
            r, op, fp = _eval_point(w, noisy, orders, budget, targets, model)
            rs.append(r)
            ops.append(op)
            fps.append(fp)
        points.append(
            CurvePoint(
                n_labels=budget,
                is_full=budget == max_pool,
                recall=_mean(rs),
                operational_precision=_mean(ops),
                legit_fp_rate=_mean(fps),
                targets=targets,
                recall_seeds=rs,
                precision_seeds=ops,
                fp_seeds=fps,
            )
        )
    return CurveSeries(
        strategy=strategy,
        noise=noise,
        level=worlds[seeds[0]].level,
        points=points,
        labels_to_target=_stable_crossing(points, targets),
    )


def _level_crossing(worlds: dict[int, _World], seeds: list[int], targets: Targets, model: str) -> int | None:
    """The cheap per-level summary number (Q6): random strategy, clean labels only."""
    max_pool = max(w.max_train_pool() for w in worlds.values())
    series = _build_series(
        worlds, seeds, _budget_grid(max_pool), max_pool, "random", CLEAN, DEFAULT_BATCH, model, targets
    )
    return series.labels_to_target


def learning_curve(
    *,
    seeds: list[int] | tuple[int, ...] = (7, 13, 21),
    preset_name: str = "default",
    data_dir: str = "data",
    level: EvasionLevel = "sophisticated",
    strategies: tuple[str, ...] = DEFAULT_STRATEGIES,
    epsilons: tuple[float, ...] = DEFAULT_EPSILONS,
    asymmetric: tuple[NoiseSpec, ...] = (),
    summary_levels: tuple[EvasionLevel, ...] = LEVELS,
    batch: int = DEFAULT_BATCH,
    model: str = "logreg",
    targets: Targets = DEFAULT_TARGETS,
    configure: Callable[[RunConfig], None] | None = None,
) -> CurveResult:
    """Build the learning curve: the full strategy × noise family at `level`, plus a per-level
    labels-to-target summary. `epsilons` are the symmetric headline sweep; `asymmetric` adds the
    realistic missed-fraud-dominant profiles. Worlds (frozen candidate pools) are generated once per
    (level, seed) and reused across every strategy/noise/budget."""
    seeds = list(seeds)
    noise_specs = [_symmetric(e) for e in epsilons] + list(asymmetric)

    # The full (strategy × noise) curve is drawn at one level (the headline: sophisticated).
    worlds = {seed: _load_world(seed, level, preset_name, data_dir, configure) for seed in seeds}
    max_pool = max(w.max_train_pool() for w in worlds.values())
    grid = _budget_grid(max_pool)
    series = [
        _build_series(worlds, seeds, grid, max_pool, strategy, noise, batch, model, targets)
        for strategy in strategies
        for noise in noise_specs
    ]

    # Cheap per-level crossing summary (random, clean labels). Reuse the loaded worlds for `level`.
    by_level: dict[str, int | None] = {}
    for lvl in summary_levels:
        lvl_worlds = worlds if lvl == level else {
            seed: _load_world(seed, lvl, preset_name, data_dir, configure) for seed in seeds
        }
        by_level[lvl] = _level_crossing(lvl_worlds, seeds, targets, model)

    return CurveResult(
        preset=preset_name,
        seeds=seeds,
        targets=targets,
        level=level,
        series=series,
        labels_to_target_by_level=by_level,
    )


def write_curve(result: CurveResult, data_dir: str) -> tuple[str, str]:
    os.makedirs(data_dir, exist_ok=True)
    json_path = os.path.join(data_dir, "learning_curve.json")
    csv_path = os.path.join(data_dir, "learning_curve.csv")
    with open(json_path, "w") as f:
        json.dump(result.to_dict(), f, indent=2)
    rows = result.to_rows()
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    return json_path, csv_path
