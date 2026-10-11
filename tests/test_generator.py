"""Fairness: the benign super-fan confound is spread across targets.

A fraud ring and a benign "super-fan club" can target the SAME creator — the hardest legit
negative (genuine, enthusiastic fans of the target the ring is pumping). The generator builds
this confound on purpose. For the leave-one-ring-out grade to be FAIR, the confound must appear
on more than one target creator: otherwise the single colliding ring always lands in the test fold
and the rule meets the ring-vs-superfan boundary for the first time on held-out data.

These pin that the generator emits one super-fan cohort per target creator (capped by
`n_benign_clusters`), each colliding with a ring's target — so the confound is represented in the
training folds too. The detector still reads only events (anti-tautology); ground truth is oracle.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from fraudrings.config import SimConfig
from fraudrings.generator import generate


def _world(tmp_path, **overrides):
    cfg = SimConfig(
        n_legit=200, n_rings=6, ring_size_min=12, ring_size_max=20, n_creators=40,
        span_days=14, legit_events_mean=40, fraud_events_mean=140, evasion_level="moderate",
        seed=7, **overrides,
    )
    events_path, gt_path = generate(cfg, str(tmp_path / "w"))
    return cfg, pl.read_parquet(events_path), pl.read_parquet(gt_path)


def _fanclub_dominant_creators(cfg: SimConfig, events: pl.DataFrame, gt: pl.DataFrame) -> list[int]:
    """Each super-fan cohort is a trailing contiguous block of accounts that almost exclusively
    view one target creator; recover each account's dominant creator over that block."""
    n_clusters = min(cfg.n_benign_clusters, cfg.n_target_creators)
    max_aid = int(cast(int, gt["account_id"].max()))
    block = set(range(max_aid - n_clusters * cfg.benign_cluster_size + 1, max_aid + 1))
    fan_events = events.filter(pl.col("account_id").is_in(list(block)))
    top = (
        fan_events.group_by("account_id", "creator_id").len()
        .sort("len", descending=True)
        .unique(subset="account_id", keep="first")
    )
    return sorted({int(cast(int, a)) for a in top["creator_id"].to_list() if a is not None})


def test_super_fan_cohorts_span_multiple_target_creators(tmp_path) -> None:
    cfg, events, gt = _world(tmp_path, n_benign_clusters=3, benign_cluster_size=30)
    creators = _fanclub_dominant_creators(cfg, events, gt)
    # one cohort per target creator 0..k-1 — multiple distinct targets, all in the target range
    assert creators == [0, 1, 2], creators
    assert all(a < cfg.n_target_creators for a in creators)


def test_super_fan_targets_collide_with_multiple_rings(tmp_path) -> None:
    """The property LORO needs: >= 2 rings target a creator that also has a super-fan cohort, so
    holding one such ring out still leaves the confound in the training folds."""
    cfg, events, gt = _world(tmp_path, n_benign_clusters=3, benign_cluster_size=30)
    fan_creators = set(_fanclub_dominant_creators(cfg, events, gt))
    ring_ids = sorted({int(r) for r in gt.filter(pl.col("ring_id").is_not_null())["ring_id"].to_list()})
    ring_targets = {r: r % cfg.n_target_creators for r in ring_ids}
    colliding = [r for r, t in ring_targets.items() if t in fan_creators]
    assert len(colliding) >= 2, (colliding, fan_creators, ring_targets)


def test_default_world_has_more_than_one_super_fan_cohort(tmp_path) -> None:
    """The fairness fix is the DEFAULT (the grade must be fair without opt-in)."""
    cfg, events, gt = _world(tmp_path)  # default n_benign_clusters
    assert cfg.n_benign_clusters > 1
    assert len(_fanclub_dominant_creators(cfg, events, gt)) > 1
