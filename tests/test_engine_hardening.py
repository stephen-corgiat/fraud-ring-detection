"""Engine hardening: the two config-gated, default-OFF Stage-2 deltas.

Discipline: with each flag OFF the engine is byte-identical to the baseline, so every existing
result is unchanged by construction; the rest of the suite
(test_candidate_gen, test_actionmix, test_supervised, …) running green on the default config IS
that regression guard. Here we (a) pin the mechanism of each flag in isolation, (b) assert OFF is
identity and ON is wired, and (c) measure the lift the flag buys when ON.

  * robust_centroid       — median seed centroid resists a contaminated/atypical seed.
  * cotarget_concentration — the co-target term rewards devotion TO the target (booster/core
                             signature) instead of mere repertoire overlap (Jaccard).
"""

from __future__ import annotations

import os

import duckdb
import numpy as np
import polars as pl
import pytest

from fraudrings.config import EngineConfig, preset
from fraudrings.detector.engine import (
    _centroid,
    _seed_target_concentration,
    _seed_targets,
    run_investigation,
)
from fraudrings.detector.ingest import build_db
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel

DAY = 86_400


def _world(tmp_path, level: EvasionLevel, *, seed: int, engine: EngineConfig):
    """Build one synthetic world + its DuckDB; return (con, ground_truth_path)."""
    cfg = preset("smoke")
    cfg.sim.evasion_level = level
    cfg.sim.seed = seed
    d = str(tmp_path / f"{level}-{seed}")
    ep, gp = generate(cfg.sim, d)
    con = build_db(ep, engine, db_path=os.path.join(d, "world.duckdb"))
    return con, gp


def _rings(gp: str) -> dict[int, list[int]]:
    gt = pl.read_parquet(gp)
    fraud = gt.filter(pl.col("is_fraud") & pl.col("ring_id").is_not_null())
    return {
        int(rid): sorted(int(a) for a in fraud.filter(pl.col("ring_id") == rid)["account_id"])
        for rid in sorted({int(r) for r in fraud["ring_id"]})
    }


def _legit_outlier(gp: str) -> int:
    """A legit account with an extreme behavioral profile — the worst-case contaminated seed."""
    gt = pl.read_parquet(gp)
    legit = {int(a) for a in gt.filter(~pl.col("is_fraud"))["account_id"]}
    return max(legit)  # deterministic; any legit account is off-ring, which is the point


def _auc(scores: list[float], labels: list[int]) -> float:
    """Rank-AUC: P(a positive outscores a negative). NaN when a fold is single-class."""
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (y == 0).sum()))


def _axis_auc(con, gp, engine, *, axis, seeds_of, k: int) -> float:
    """Mean AUC of one Stage-2 axis at separating a held-out ring member from any other retrieved
    candidate, over every ring — the intrinsic discriminative power of that axis. `seeds_of(members)`
    picks the seed set; the held-out positives are the ring members not used as seeds."""
    aucs: list[float] = []
    for members in _rings(gp).values():
        if len(members) <= k + 1:
            continue
        seeds = seeds_of(members)
        positives = set(members) - set(seeds)
        cands = run_investigation(con, seeds, engine)
        y = [1 if c.account_id in positives else 0 for c in cands]
        if sum(y) == 0:
            continue
        aucs.append(_auc([axis(c) for c in cands], y))
    return float(np.nanmean(aucs)) if aucs else float("nan")


# --------------------------------------------------------------------------- robust_centroid


def test_centroid_median_resists_outlier_while_mean_is_dragged() -> None:
    """The mechanism: with two tight vectors and one far outlier, the median centroid stays on the
    tight pair (the outlier is the middle-excluded extreme per feature) while the mean is pulled a
    third of the way toward it."""
    vecs = {1: np.array([0.0, 0.0]), 2: np.array([0.0, 0.0]), 3: np.array([9.0, 9.0])}  # 3 = outlier

    mean = _centroid(vecs, 2, robust=False)
    median = _centroid(vecs, 2, robust=True)

    assert mean == pytest.approx([3.0, 3.0])       # dragged 1/3 toward the outlier
    assert median == pytest.approx([0.0, 0.0])     # sits on the true pair, outlier ignored
    assert _centroid({}, 2, robust=True) == pytest.approx([0.0, 0.0])  # empty ⇒ origin


def test_robust_centroid_off_is_identity_and_on_is_wired(tmp_path) -> None:
    """OFF (default) reproduces the mean-centroid baseline exactly; ON changes the ranking (so it is truly
    wired, not a no-op). The OFF path being byte-identical is what leaves every other result intact."""
    off = EngineConfig()
    on = EngineConfig(robust_centroid=True)
    con, gp = _world(tmp_path, "naive", seed=7, engine=off)
    ring = next(iter(_rings(gp).values()))
    seeds = ring[:3]

    a = run_investigation(con, seeds, off)
    b = run_investigation(con, seeds, off)
    assert [(c.account_id, c.score) for c in a] == [(c.account_id, c.score) for c in b]  # deterministic

    robust = run_investigation(con, seeds, on)
    # same candidate SET (retrieval is centroid-independent), but the behavioral axis — hence the
    # score/order — moves, proving the flag reaches Stage-2.
    assert {c.account_id for c in robust} == {c.account_id for c in a}
    assert [c.behavioral for c in robust] != [c.behavioral for c in a]


def test_robust_centroid_lifts_the_behavioral_axis_under_a_contaminated_seed(tmp_path) -> None:
    """The lift, isolated to the axis the centroid drives. When one of the K seeds is a contaminant (a
    legit account mislabelled into the seed set), the mean centroid is dragged toward it and the
    behavioral axis separates true members from the widened confounders worse; the median centroid
    ignores the outlier and holds discrimination. Widening is on so the candidate pool has confounders
    to rank past; averaged over rings against smoke noise."""
    off = EngineConfig(cotarget_candidate_gen=True, cotarget_always=True)
    on = off.model_copy(update={"robust_centroid": True})
    con, gp = _world(tmp_path, "naive", seed=7, engine=off)
    outlier = _legit_outlier(gp)

    def contaminated(members: list[int]) -> list[int]:
        return [members[0], members[1], outlier]   # 2 true + 1 contaminant

    auc_off = _axis_auc(con, gp, off, axis=lambda c: c.behavioral, seeds_of=contaminated, k=2)
    auc_on = _axis_auc(con, gp, on, axis=lambda c: c.behavioral, seeds_of=contaminated, k=2)
    assert auc_on > auc_off, (auc_off, auc_on)   # median centroid discriminates better than the mean


# ------------------------------------------------------------------ cotarget_concentration


def _events_db(rows: list[tuple[int, int, int, str, float]]) -> duckdb.DuckDBPyConnection:
    """An events-only DuckDB — the whole world the co-target helpers read."""
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE events "
        "(ts BIGINT, account_id BIGINT, creator_id BIGINT, action VARCHAR, reported_position_pct DOUBLE)"
    )
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?)", rows)
    return con


def _qv(account: int, target: int, day: int, n: int, *, pct: float = 0.95) -> list[tuple]:
    return [(day * DAY + i, account, target, "view", pct) for i in range(n)]


def test_seed_targets_is_high_agreement_target_not_scattered_cover() -> None:
    """The target is the creator the seeds AGREE on (all three qualified-view it), not the higher-volume
    but idiosyncratic cover each seed views alone."""
    rows = (
        _qv(1, 500, 0, 5) + _qv(2, 500, 0, 5) + _qv(3, 500, 0, 5)   # target: 3 seeds agree
        + _qv(1, 600, 0, 40)                                          # seed 1's private cover (high vol, 1 acct)
        + _qv(2, 601, 0, 40) + _qv(3, 602, 0, 40)                     # seeds 2,3 private cover
    )
    con = _events_db(rows)
    targets = _seed_targets(con, [1, 2, 3], EngineConfig(cotarget_n_targets=1))
    assert [t[0] for t in targets] == [500]


def test_seed_target_concentration_is_qv_share_on_the_target() -> None:
    """Devotion, not overlap: an all-on-target account scores 1.0; a half-elsewhere account 0.5; an
    account with no qualified views is absent (→ 0 at the call site)."""
    rows = (
        _qv(10, 500, 0, 20)                       # 20/20 on target → 1.0
        + _qv(11, 500, 0, 5) + _qv(11, 700, 0, 5)  # 5/10 on target → 0.5
        + [(0, 12, 800, "view", 0.1)]              # acct 12: only a NON-qualified view (below threshold)
    )
    con = _events_db(rows)
    conc = _seed_target_concentration(con, [10, 11, 12], [500], EngineConfig())
    assert conc[10] == pytest.approx(1.0)
    assert conc[11] == pytest.approx(0.5)
    assert 12 not in conc                          # no qualified views → absent (scores 0.0 in the loop)


def test_cotarget_concentration_off_is_deterministic_on_is_wired(tmp_path) -> None:
    """OFF is the baseline Jaccard path (deterministic — the byte-identical guard). ON replaces the axis with
    the target-share, so the `cotarget` values change AND equal `_seed_target_concentration` end-to-end."""
    off = EngineConfig(cotarget_candidate_gen=True, cotarget_always=True)
    on = off.model_copy(update={"cotarget_concentration": True})
    con, gp = _world(tmp_path, "sophisticated", seed=7, engine=off)
    ring = max(_rings(gp).values(), key=len)
    seeds = ring[:3]

    j1 = run_investigation(con, seeds, off)
    j2 = run_investigation(con, seeds, off)
    assert [(c.account_id, c.cotarget) for c in j1] == [(c.account_id, c.cotarget) for c in j2]

    conc_cands = run_investigation(con, seeds, on)
    assert {c.account_id for c in conc_cands} == {c.account_id for c in j1}  # same retrieval
    assert [c.cotarget for c in conc_cands] != [c.cotarget for c in j1]      # axis genuinely swapped

    target_ids = [t[0] for t in _seed_targets(con, seeds, on)]
    expected = _seed_target_concentration(con, [c.account_id for c in conc_cands], target_ids, on)
    for c in conc_cands:
        assert c.cotarget == pytest.approx(expected.get(c.account_id, 0.0))
        assert 0.0 <= c.cotarget <= 1.0


def test_cotarget_concentration_is_a_valid_axis_and_wins_at_naive(tmp_path) -> None:
    """The concentration axis is a valid, above-chance discriminator (the swap does not break Stage-2),
    and it BEATS the baseline Jaccard axis exactly where the ring concentrates hardest and infra retrieval is
    complete — naive, where the candidate pool is the true ring plus the co-target fans the widened net
    pulls in. Devotion-to-target then separates ring from fan better than mere repertoire overlap.

    On synthetic the two axes only TRADE by regime (Jaccard wins at moderate), which is precisely why
    the flag stays default-OFF: the concentration term's decisive payoff is a few-dominant core,
    a shape the simulated rings (many similar members) under-represent.
    Averaged over seeds against smoke noise; retrieval is flag-independent, so one world serves both."""
    seeds_list = [7, 13, 21]
    jaccard = EngineConfig(cotarget_candidate_gen=True, cotarget_always=True)
    concentration = jaccard.model_copy(update={"cotarget_concentration": True})

    def axis_aucs(level: EvasionLevel) -> tuple[float, float]:
        jac, conc = [], []
        for s in seeds_list:
            con, gp = _world(tmp_path, level, seed=s, engine=jaccard)
            jac.append(_axis_auc(con, gp, jaccard, axis=lambda c: c.cotarget, seeds_of=lambda m: m[:3], k=3))
            conc.append(_axis_auc(con, gp, concentration, axis=lambda c: c.cotarget, seeds_of=lambda m: m[:3], k=3))
        return float(np.mean(jac)), float(np.mean(conc))

    jac_naive, conc_naive = axis_aucs("naive")
    _, conc_soph = axis_aucs("sophisticated")

    assert conc_naive >= 0.60, conc_naive              # a valid, above-chance discriminator…
    assert conc_soph >= 0.60, conc_soph                # …at both extremes (the swap doesn't break it)
    assert conc_naive > jac_naive, (jac_naive, conc_naive)  # and it beats Jaccard where the ring concentrates
