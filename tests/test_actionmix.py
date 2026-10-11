"""The action-mix scoring axis.

The engine gains a fourth top-level term — a seed-centroid proximity over the funnel-shape /
account-age sub-features — re-weighted into the hand-weighted conjunction (0.45/0.25/0.15/0.15). These pin:
  1. every candidate carries an `actionmix` axis and the conjunction weights still sum to 1;
  2. the axis is a strong independent discriminator at naive/moderate and FADES at sophisticated
     (the evasion-dial convergence — not a "truncated funnel => fraud" tautology);
  3. the benign lurker confound HOLDS the legit-FP line when the axis is switched on.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl

from fraudrings.config import EngineConfig, preset
from fraudrings.detector.engine import run_investigation
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import evaluate
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel


def _con(tmp_path, level: EvasionLevel, *, seed: int, engine: EngineConfig):
    cfg = preset("smoke")
    cfg.sim.evasion_level = level
    cfg.sim.seed = seed
    d = str(tmp_path / f"{level}-{seed}")
    ep, gp = generate(cfg.sim, d)
    con = build_db(ep, engine, db_path=os.path.join(d, "world.duckdb"))
    return cfg, con, gp


def _auc(scores: list[float], labels: list[int]) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, int)
    if y.sum() == 0 or y.sum() == len(y):
        return float("nan")
    order = np.argsort(s)
    ranks = np.empty(len(s))
    ranks[order] = np.arange(1, len(s) + 1)
    return float((ranks[y == 1].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (y == 0).sum()))


def _axis_auc(con, gp, cfg, engine) -> float:
    """Mean AUC of the action-mix axis for separating a held-out ring member from any other
    retrieved candidate, over all rings (the intrinsic discriminative power of the axis)."""
    gt = pl.read_parquet(gp)
    fraud = gt.filter(pl.col("is_fraud") & pl.col("ring_id").is_not_null())
    aucs = []
    for rid in sorted({int(r) for r in fraud["ring_id"]}):
        members = sorted(int(a) for a in fraud.filter(pl.col("ring_id") == rid)["account_id"])
        k = cfg.eval.seeds_per_ring
        if len(members) <= k + 1:
            continue
        seeds, pos = members[:k], set(members[k:])
        cands = run_investigation(con, seeds, engine)
        y = [1 if c.account_id in pos else 0 for c in cands]
        if sum(y) == 0:
            continue
        aucs.append(_auc([c.actionmix for c in cands], y))
    return float(np.nanmean(aucs)) if aucs else float("nan")


def test_candidate_carries_actionmix_and_weights_sum_to_one(tmp_path) -> None:
    e = EngineConfig()
    assert abs(e.w_linkage + e.w_behavioral + e.w_cotarget + e.w_actionmix - 1.0) < 1e-9
    cfg, con, gp = _con(tmp_path, "naive", seed=7, engine=e)
    gt = pl.read_parquet(gp)
    fraud = gt.filter(pl.col("is_fraud") & pl.col("ring_id").is_not_null())
    rid = sorted({int(r) for r in fraud["ring_id"]})[0]
    members = sorted(int(a) for a in fraud.filter(pl.col("ring_id") == rid)["account_id"])
    cands = run_investigation(con, members[: cfg.eval.seeds_per_ring], e)
    assert cands
    c = cands[0]
    assert 0.0 <= c.actionmix <= 1.0
    # the reported score is exactly the re-weighted conjunction of its four axes
    expect = (
        e.w_linkage * c.linkage + e.w_behavioral * c.behavioral
        + e.w_cotarget * c.cotarget + e.w_actionmix * c.actionmix
    )
    assert abs(c.score - expect) < 1e-9


def test_action_mix_axis_discriminates_and_fades_with_sophistication(tmp_path) -> None:
    """A strong tell at naive/moderate that fades at sophisticated (the convergence). Averaged
    over seeds to defend against smoke-scale single-seed noise."""
    e = EngineConfig()
    e.cotarget_candidate_gen = True
    e.cotarget_always = True  # widen so the candidate set holds confounders to rank past
    seeds = [7, 13, 21]

    def mean_auc(level: EvasionLevel) -> float:
        vals = []
        for s in seeds:
            cfg, con, gp = _con(tmp_path, level, seed=s, engine=e)
            vals.append(_axis_auc(con, gp, cfg, e))
        return float(np.mean(vals))

    by = {level: mean_auc(level) for level in ("naive", "moderate", "sophisticated")}
    assert by["naive"] >= 0.85, by
    assert by["moderate"] >= 0.80, by
    # the fade: sophisticated is materially weaker than naive (the funnels converge)
    assert by["sophisticated"] < by["naive"] - 0.1, by


def test_lurker_confound_holds_the_legit_fp_line_when_axis_is_on(tmp_path) -> None:
    """The action-mix axis must not over-flag the funnel-truncated lurkers/legit: turning it on holds
    (or improves) the operational false-positive line vs the old three-axis basket."""
    seeds = [7, 13, 21]
    on = EngineConfig()  # actionmix on (default re-weighted conjunction)
    off = EngineConfig(w_linkage=0.50, w_behavioral=0.30, w_cotarget=0.20, w_actionmix=0.0)
    for eng in (on, off):
        eng.cotarget_candidate_gen = True
        eng.cotarget_always = True
    fp_on, fp_off = [], []
    for s in seeds:
        cfg, con, gp = _con(tmp_path, "sophisticated", seed=s, engine=on)
        fp_on.append(evaluate(con, gp, on, cfg.eval).legit_fp_rate)
        cfg2, con2, gp2 = _con(tmp_path, "sophisticated", seed=s, engine=off)
        fp_off.append(evaluate(con2, gp2, off, cfg2.eval).legit_fp_rate)
    # the line holds: the axis does not worsen the legit false-positive rate, and stays within target
    assert np.mean(fp_on) <= np.mean(fp_off) + 1e-9, (np.mean(fp_on), np.mean(fp_off))
    assert np.mean(fp_on) <= 0.05
