"""The temporal raster/occupancy renderer + its membership plumbing.

Exercises `viewer.raster` and the lock-step-aware additions to `viewer.data` (the eval-side
bandwagon tag, the bandwagon-cohort helpers, the concurrency-keyed world cache). These pin the
*renderer's* contract — the analyst aid for lock-step serialization — not the
detector (which never reads any of this). The headline property: the occupancy strip reproduces the
serialization shape (ring hugs the cap C; the uncapped bandwagon piles above it).
"""

from __future__ import annotations

import polars as pl
import pytest

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel
from fraudrings.viewer import data as D
from fraudrings.viewer.raster import build_raster

_C = 12  # concurrency for the smoke lock-step worlds below


def _lockstep_smoke(tmp_path, *, concurrency=_C, level: EvasionLevel = "sophisticated", seed=7) -> D.World:
    return D.load_world("smoke", level, seed, base_dir=str(tmp_path), concurrency=concurrency)


def _sized_world(tmp_path, *, concurrency: int):
    """A lock-step world sized so serialization is unmistakable: ring M ≫ C and a bandwagon cohort
    large enough to pile above the cap. Generated directly (bypassing the smoke preset's tiny
    cohort) so the 'piles on' property is observable."""
    cfg = preset("smoke")
    cfg.sim.evasion_level = "sophisticated"
    cfg.sim.seed = 7
    cfg.sim.operator_concurrency = concurrency
    cfg.sim.n_benign_clusters = 3
    cfg.sim.benign_cluster_size = 120
    cfg.sim.ring_size_min = 30
    cfg.sim.ring_size_max = 45  # soph ×3.5 → M ≈ 105–157 ≫ C
    wd = str(tmp_path / f"sized-c{concurrency}")
    ep, gp = generate(cfg.sim, wd)
    con = build_db(ep, cfg.engine, db_path=f"{wd}/world.duckdb")
    return cfg, con, pl.read_parquet(gp)


# --- core renderer contract ------------------------------------------------


def test_row_order_is_activity_onset_not_ground_truth(tmp_path) -> None:
    """Rows MUST be ordered by event-derivable activity onset (first-event ts), never GT membership
    — handing the eye a GT-sorted raster would leak the answer (a fairness constraint)."""
    w = _lockstep_smoke(tmp_path)
    members = D.ring_members(w, D.rings(w)[0].ring_id)
    rd = D.account_raster(w, members, bucket_seconds=300)
    onsets = {
        int(a): int(t)
        for a, t in w.con.execute(
            f"SELECT account_id, min(ts) FROM events WHERE account_id IN ({','.join(map(str, members))}) "
            "GROUP BY account_id"
        ).fetchall()
    }
    ordered = [onsets[a] for a in rd.accounts]
    assert ordered == sorted(ordered), "rows must be ascending in activity onset"


def test_ring_occupancy_hugs_the_cap(tmp_path) -> None:
    """The serialized ring's peak simultaneity is bounded by C and actually reaches it (the cap
    binds) — the flat plateau that is the signal."""
    w = _lockstep_smoke(tmp_path, concurrency=_C)
    rings = sorted(D.rings(w), key=lambda r: r.size, reverse=True)
    # the largest ring (M ≫ C) must saturate its lanes
    rd = D.account_raster(w, D.ring_members(w, rings[0].ring_id), bucket_seconds=300)
    assert rd.peak_occupancy <= _C + 1, rd.peak_occupancy
    assert rd.peak_occupancy >= 0.7 * _C, rd.peak_occupancy


def test_bandwagon_piles_above_the_serialized_ring(tmp_path) -> None:
    """The discriminating shape: with a cohort large enough to pile, every bandwagon's peak
    occupancy sits materially above the co-located ring's capped plateau."""
    cfg, con, gt = _sized_world(tmp_path, concurrency=_C)
    ring_ids = sorted({int(r) for r in gt.filter(pl.col("ring_id").is_not_null())["ring_id"]})
    bw_col = gt.filter(pl.col("bandwagon_cluster_id").is_not_null())["bandwagon_cluster_id"]
    cluster_ids = sorted({int(c) for c in bw_col})
    for rid in (r for r in ring_ids if r in cluster_ids):
        ring_mem = sorted(int(a) for a in gt.filter(pl.col("ring_id") == rid)["account_id"])
        band_mem = sorted(int(a) for a in gt.filter(pl.col("bandwagon_cluster_id") == rid)["account_id"])
        ring = build_raster(con, ring_mem, bucket_seconds=300)
        band = build_raster(con, band_mem, bucket_seconds=300)
        assert ring.peak_occupancy <= _C + 1, (rid, ring.peak_occupancy)
        assert band.peak_occupancy > ring.peak_occupancy, (rid, ring.peak_occupancy, band.peak_occupancy)
    con.close()


def test_matrix_and_points_account_for_every_event(tmp_path) -> None:
    w = _lockstep_smoke(tmp_path)
    members = D.ring_members(w, D.rings(w)[0].ring_id)
    rd = D.account_raster(w, members, bucket_seconds=300)
    row = w.con.execute(
        f"SELECT count(*) FROM events WHERE account_id IN ({','.join(map(str, members))})"
    ).fetchone()
    assert row is not None
    n_events = int(row[0])
    m = rd.matrix()
    assert m.shape == (rd.n_accounts, rd.n_buckets)
    assert int(m.sum()) == n_events
    assert sum(c for _r, _b, c in rd.points) == n_events  # long-form points agree with the matrix


def test_empty_set_and_bucket_validation(tmp_path) -> None:
    w = _lockstep_smoke(tmp_path)
    empty = D.account_raster(w, [])
    assert empty.n_accounts == 0 and empty.n_buckets == 0 and empty.peak_occupancy == 0
    with pytest.raises(ValueError):
        D.account_raster(w, D.ring_members(w, D.rings(w)[0].ring_id), bucket_seconds=0)


def test_shared_window_clips_to_requested_bounds(tmp_path) -> None:
    """Passing t0/t1 fixes the axis (so two cohorts can share it) and clips events into range."""
    w = _lockstep_smoke(tmp_path)
    members = D.ring_members(w, D.rings(w)[0].ring_id)
    full = D.account_raster(w, members, bucket_seconds=300)
    mid = full.t0 + (full.n_buckets // 2) * full.bucket_seconds
    clipped = D.account_raster(w, members, bucket_seconds=300, t0=full.t0, t1=mid)
    assert clipped.t0 == full.t0
    assert clipped.n_buckets <= full.n_buckets // 2 + 1


# --- membership plumbing ---------------------------------------------------


def test_bandwagon_helpers_present_in_lockstep_absent_in_identity(tmp_path) -> None:
    ls = _lockstep_smoke(tmp_path, concurrency=_C)
    clusters = D.bandwagon_clusters(ls)
    assert clusters, "lock-step world should expose bandwagon cohorts"
    members = D.bandwagon_members(ls, clusters[0].cluster_id)
    assert len(members) == clusters[0].size
    # every bandwagon member is is_fraud=False in GT (the benign control)
    fraud = set(ls.gt.filter(pl.col("is_fraud"))["account_id"].to_list())
    assert not (set(members) & fraud)

    identity = D.load_world("smoke", "sophisticated", 7, base_dir=str(tmp_path / "id"))
    assert identity.concurrency is None
    assert D.bandwagon_clusters(identity) == []  # identity worlds carry no bandwagon tag


def test_concurrency_keys_the_world_dir_without_disturbing_identity_paths(tmp_path) -> None:
    """Lock-step worlds get a `-c{C}` suffix; identity worlds keep their historical path so existing
    caches/tests are untouched."""
    identity = D.load_world("smoke", "naive", 7, base_dir=str(tmp_path))
    lock = D.load_world("smoke", "naive", 7, base_dir=str(tmp_path), concurrency=30)
    assert identity.world_dir.endswith("smoke-naive-7")
    assert lock.world_dir.endswith("smoke-naive-7-c30")
    assert identity.world_dir != lock.world_dir
