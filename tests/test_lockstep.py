"""The lock-step simulation scheduler.

`SimConfig.operator_concurrency` adds a finite-concurrency operator that SERIALIZES each ring:
its members' sessions are packed onto C lanes inside a burst window, so only <=C accounts active
at once and the pool cycles (they take turns). The benign bandwagon uses the SAME session
primitive and the SAME window but UNCAPPED — fans pile on. The concurrency cap is the *sole*
difference between ring and bandwagon, so any downstream temporal-overlap signal has to earn the
separation honestly instead of assuming it (anti-tautology).

These pin four properties of the world the scheduler builds:
  1. identity — operator_concurrency=None is a pinned, deterministic world;
  2. serialization binds — ring max simultaneous overlap <=C and materially below the bandwagon's;
  3. determinism — same seed -> identical events;
  4. anti-tautology parity — ring vs bandwagon per-account gap_mean / active_hours / n_ips are
     statistically indistinguishable WITHIN the lock-step world (only cross-account overlap differs).
"""

from __future__ import annotations

import hashlib
from typing import cast

import numpy as np
import polars as pl

from fraudrings.config import preset
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel

# Content snapshot (canonical CSV of the event table, not parquet file bytes — those drift with the
# parquet writer version) of the identity branch (operator_concurrency=None), smoke worlds, seed 7. Pins that
# the branch stays deterministic and reproducible. Re-snapshot here ONLY alongside an intentional
# generator change (and say why in the commit).
_IDENTITY_SHA256 = {
    "naive": "c05a62c54b1e604fd129b1e832ae7542470c07fb47540127b6c6e8525ed53a95",
    "moderate": "ab99898763b3d17f4233649c452dc4fd3f40de3bf0248395dbbc04cb542e45c8",
    "sophisticated": "f71de3a732e506f09c85e0b2152268cb1c619cc6aa70756efc0ab82edeb5acdb",
}

_C = 30  # operator concurrency (lanes) used by the lock-step worlds below


def _lockstep_world(tmp_path, *, seed: int = 7):
    """A sophisticated lock-step world sized so the cap binds (rings > C) and the bandwagon cohorts
    exceed C (so 'piles on' is observable). 3 benign cohorts co-located with the 3 rings' windows."""
    cfg = preset("smoke")
    cfg.sim.evasion_level = "sophisticated"
    cfg.sim.seed = seed
    cfg.sim.operator_concurrency = _C
    cfg.sim.n_benign_clusters = 3
    cfg.sim.benign_cluster_size = 80
    events_path, gt_path = generate(cfg.sim, str(tmp_path / f"ls-{seed}"))
    return cfg, pl.read_parquet(events_path), pl.read_parquet(gt_path)


def _bandwagon_cohorts(cfg, gt: pl.DataFrame) -> list[set[int]]:
    """The bandwagon is the trailing contiguous block of accounts (generated after the rings),
    one cohort of `benign_cluster_size` per cluster — same convention as test_generator.py."""
    n_clusters = min(cfg.sim.n_benign_clusters, cfg.sim.n_target_creators)
    size = cfg.sim.benign_cluster_size
    max_aid = int(cast(int, gt["account_id"].max()))
    first = max_aid - n_clusters * size + 1
    return [set(range(first + i * size, first + (i + 1) * size)) for i in range(n_clusters)]


def _max_simultaneous(events: pl.DataFrame, accounts: set[int]) -> int:
    """Peak number of distinct ACCOUNTS active at the same instant — the design's serialization
    claim ('<=C accounts take turns'), not merely concurrent sessions. Each account's session
    intervals [first_view, last_view] are merged into disjoint busy intervals first, so one account
    counts at most +1 even when two of its sessions land on different lanes and overlap in time.
    Sweep line over (open:+1, close:-1); at a tie opens precede closes (touching = overlapping)."""
    sub = events.filter(pl.col("account_id").is_in(list(accounts)))
    seg = sub.group_by("account_id", "session_id").agg(
        pl.col("ts").min().alias("s"), pl.col("ts").max().alias("e")
    )
    by_account: dict[int, list[tuple[int, int]]] = {}
    for a, s, e in zip(seg["account_id"], seg["s"], seg["e"], strict=True):
        by_account.setdefault(int(a), []).append((int(s), int(e)))

    deltas: list[tuple[int, int]] = []
    for intervals in by_account.values():
        intervals.sort()
        cur_s, cur_e = intervals[0]
        for s, e in intervals[1:]:
            if s <= cur_e:  # overlap or touch -> merge (same account, count once)
                cur_e = max(cur_e, e)
            else:
                deltas += [(cur_s, 1), (cur_e, -1)]
                cur_s, cur_e = s, e
        deltas += [(cur_s, 1), (cur_e, -1)]

    deltas.sort(key=lambda d: (d[0], -d[1]))  # at a tie, opens (+1) before closes (-1)
    cur = peak = 0
    for _, d in deltas:
        cur += d
        peak = max(peak, cur)
    return peak


def _feature(events: pl.DataFrame, accounts: set[int], name: str) -> np.ndarray:
    """Per-account value of one of the three features the parity guard pins. Definitions MATCH
    detector/ingest.py: gap_mean = mean inter-event gap; active_hours = distinct hour-of-day;
    n_ips = distinct IPs."""
    sub = events.filter(pl.col("account_id").is_in(list(accounts))).sort("account_id", "ts")
    if name == "gap_mean":
        gaps = sub.with_columns((pl.col("ts") - pl.col("ts").shift(1).over("account_id")).alias("gap"))
        out = gaps.group_by("account_id").agg(pl.col("gap").mean().alias("v"))["v"].drop_nulls()
    elif name == "active_hours":
        hod = sub.with_columns(((pl.col("ts") // 3600) % 24).alias("hod"))
        out = hod.group_by("account_id").agg(pl.col("hod").n_unique().alias("v"))["v"]
    elif name == "n_ips":
        out = sub.group_by("account_id").agg(pl.col("ip").n_unique().alias("v"))["v"]
    else:  # pragma: no cover - guard
        raise ValueError(name)
    return out.to_numpy().astype(float)


def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = len(a), len(b)
    va, vb = a.var(ddof=1), b.var(ddof=1)
    pooled = np.sqrt(((na - 1) * va + (nb - 1) * vb) / max(na + nb - 2, 1))
    return float(abs(a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


# --- 1. identity ------------------------------------------------------------
def test_operator_concurrency_none_is_byte_identical(tmp_path) -> None:
    """operator_concurrency=None is a deterministic, reproducible code path: the smoke worlds hash to
    a fixed snapshot. See _IDENTITY_SHA256 above."""
    for level, expected in _IDENTITY_SHA256.items():
        cfg = preset("smoke")
        cfg.sim.evasion_level = cast(EvasionLevel, level)
        cfg.sim.seed = 7
        assert cfg.sim.operator_concurrency is None  # the default
        events_path, _ = generate(cfg.sim, str(tmp_path / level))
        got = hashlib.sha256(pl.read_parquet(events_path).write_csv().encode()).hexdigest()
        assert got == expected, (level, got)


# --- 2. serialization binds -------------------------------------------------
def test_concurrency_cap_serializes_the_ring_below_the_bandwagon(tmp_path) -> None:
    """The headline property: each ring's simultaneous overlap is capped at C (the operator's lanes),
    the cap actually BINDS (overlap reaches ~C, not idle), and every bandwagon cohort — same window,
    uncapped — piles MATERIALLY above it. Computed eval-side; the detector never sees this yet."""
    cfg, events, gt = _lockstep_world(tmp_path)

    ring_ids = sorted({int(r) for r in gt.filter(pl.col("ring_id").is_not_null())["ring_id"]})
    ring_overlaps = [
        _max_simultaneous(events, {int(a) for a in gt.filter(pl.col("ring_id") == r)["account_id"]})
        for r in ring_ids
    ]
    cohort_overlaps = [_max_simultaneous(events, c) for c in _bandwagon_cohorts(cfg, gt)]

    # the cap is a hard ceiling on every ring (distinct active accounts <= concurrent sessions <= C)
    assert all(o <= _C + 1 for o in ring_overlaps), ring_overlaps
    # ...and it BINDS — saturated lanes drive overlap up to ~C (else there is no signal to detect)
    assert max(ring_overlaps) >= 0.7 * _C, ring_overlaps
    # the bandwagon piles on: every cohort exceeds C and sits materially above the serialized ring
    assert all(o >= 1.1 * _C for o in cohort_overlaps), cohort_overlaps
    assert min(cohort_overlaps) >= 1.2 * max(ring_overlaps), (ring_overlaps, cohort_overlaps)


# --- 3. determinism ---------------------------------------------------------
def test_lockstep_is_deterministic(tmp_path) -> None:
    """Seed-driven, no wall-clock: the same config produces byte-identical events."""
    _, ev1, _ = _lockstep_world(tmp_path / "a")
    _, ev2, _ = _lockstep_world(tmp_path / "b")
    assert ev1.equals(ev2)


# --- 4b. eval-side bandwagon tag --------------------------------------------
def test_bandwagon_cluster_id_tags_only_the_uncapped_cohort(tmp_path) -> None:
    """The eval-only tag: every bandwagon fan carries a
    `bandwagon_cluster_id`; ring members and legit do not. Without it the bandwagon is
    indistinguishable from generic legit in GT, so the 2AFC grading could not find it. The detector
    never reads ground truth, so this stays anti-tautology-safe."""
    cfg, _events, gt = _lockstep_world(tmp_path)
    assert "bandwagon_cluster_id" in gt.columns
    assert gt["bandwagon_cluster_id"].dtype == pl.Int64  # typed even when (elsewhere) all-null

    tagged = gt.filter(pl.col("bandwagon_cluster_id").is_not_null())
    # exactly the bandwagon block, every one benign, grouped into the expected cohorts
    assert (~tagged["is_fraud"]).all()
    assert tagged["ring_id"].is_null().all()
    n_clusters = min(cfg.sim.n_benign_clusters, cfg.sim.n_target_creators)
    assert sorted(tagged["bandwagon_cluster_id"].unique().to_list()) == list(range(n_clusters))
    assert tagged.height == n_clusters * cfg.sim.benign_cluster_size
    # ring members are NOT tagged
    assert gt.filter(pl.col("is_fraud"))["bandwagon_cluster_id"].is_null().all()


def test_identity_world_has_a_typed_but_empty_bandwagon_tag(tmp_path) -> None:
    """operator_concurrency=None (identity branch) emits the column as all-null Int64 — schema is
    stable across both branches, and nothing is mislabeled a bandwagon."""
    cfg = preset("smoke")
    cfg.sim.evasion_level = "naive"
    cfg.sim.seed = 7
    _events_path, gt_path = generate(cfg.sim, str(tmp_path / "identity"))
    gt = pl.read_parquet(gt_path)
    assert "bandwagon_cluster_id" in gt.columns
    assert gt["bandwagon_cluster_id"].dtype == pl.Int64
    assert gt["bandwagon_cluster_id"].null_count() == gt.height


# --- 4. anti-tautology parity ----------------------------------------------
def test_ring_and_bandwagon_are_indistinguishable_on_per_account_features(tmp_path) -> None:
    """The guard step 2 then exploits: WITHIN the lock-step world, ring vs bandwagon per-account
    gap_mean / active_hours / n_ips are statistically indistinguishable — only the cross-account
    temporal overlap (test 2) is allowed to separate them. If any of these split the populations,
    a downstream classifier could cheat without ever reading the serialization signal."""
    cfg, events, gt = _lockstep_world(tmp_path)
    ring = {int(a) for a in gt.filter(pl.col("is_fraud"))["account_id"]}
    band: set[int] = set().union(*_bandwagon_cohorts(cfg, gt))

    for name in ("gap_mean", "active_hours", "n_ips"):
        r = _feature(events, ring, name)
        b = _feature(events, band, name)
        rel_mean_diff = abs(r.mean() - b.mean()) / max(abs(r.mean()), abs(b.mean()), 1e-9)
        # centered the same (<15%) AND no more than a small distributional effect (Cohen's d < 0.5)
        assert rel_mean_diff < 0.15, (name, r.mean(), b.mean())
        assert _cohens_d(r, b) < 0.5, (name, _cohens_d(r, b))
