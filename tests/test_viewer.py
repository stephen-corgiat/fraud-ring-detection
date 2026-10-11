"""The Investigate page's query layer + label scoring.

These exercise `viewer.data` and `viewer.labels` directly (no Streamlit runtime), against
a freshly built smoke world. The viewer sits on the eval side of the anti-tautology line, so
reading ground truth here is allowed — but the *detector* path it wraps (`run_investigation`)
still only ever sees events.
"""

from __future__ import annotations

import duckdb
import polars as pl

from fraudrings.detector.ingest import SCHEMA_VERSION
from fraudrings.schema import EvasionLevel
from fraudrings.viewer import data as D
from fraudrings.viewer import labels as L


def _world(tmp_path, level: EvasionLevel = "sophisticated", seed=7):
    return D.load_world("smoke", level, seed, cotarget_on=True, base_dir=str(tmp_path))


def test_load_world_is_idempotent_and_cached(tmp_path) -> None:
    w1 = D.load_world("smoke", "naive", 7, base_dir=str(tmp_path))
    assert w1.n_accounts > 0 and w1.n_fraud > 0
    # second load must NOT regenerate (reuse parquet + duckdb on disk)
    db_mtime = (tmp_path / "viewer" / "smoke-naive-7" / "world.duckdb").stat().st_mtime
    w2 = D.load_world("smoke", "naive", 7, base_dir=str(tmp_path))
    assert w2.n_accounts == w1.n_accounts
    assert (tmp_path / "viewer" / "smoke-naive-7" / "world.duckdb").stat().st_mtime == db_mtime


def test_stale_world_auto_migrates_on_load(tmp_path) -> None:
    """A cache from an older schema (no/older `schema_meta` stamp, missing the Stage-1 tables) is
    rebuilt on load — so `fraudrings view` never trips over missing db columns."""
    w = D.load_world("smoke", "naive", 7, base_dir=str(tmp_path))
    db_path = str(tmp_path / "viewer" / "smoke-naive-7" / "world.duckdb")
    assert D._db_schema_version(db_path) == SCHEMA_VERSION
    w.con.close()

    # Downgrade the cache to look like an older schema: drop the version stamp + the accounts table.
    con = duckdb.connect(db_path)
    con.execute("DROP TABLE schema_meta")
    con.execute("DROP TABLE accounts")
    con.close()
    assert D._db_schema_version(db_path) is None  # now reads as stale

    # Reloading triggers the auto-migration: stamp restored, Stage-1 schema back.
    w2 = D.load_world("smoke", "naive", 7, base_dir=str(tmp_path))
    assert D._db_schema_version(db_path) == SCHEMA_VERSION
    tabs = {r[0] for r in w2.con.execute("SHOW TABLES").fetchall()}
    assert {"accounts", "schema_meta"} <= tabs
    cols = {r[1] for r in w2.con.execute("PRAGMA table_info('account_features')").fetchall()}
    assert "account_age_days" in cols


def test_rings_and_seeds(tmp_path) -> None:
    w = _world(tmp_path, level="naive")
    rings = D.rings(w)
    assert rings, "expected fraud rings in the world"
    # sorted largest-first
    assert all(rings[i].size >= rings[i + 1].size for i in range(len(rings) - 1))
    ring0 = rings[0].ring_id
    members = D.ring_members(w, ring0)
    seeds = D.default_seeds(w, ring0, k=3)
    assert seeds == members[:3]
    assert all(D.account_exists(w, m) for m in members)


def test_profile_overview_and_anomalies(tmp_path) -> None:
    w = _world(tmp_path, level="naive")
    ring0 = D.rings(w)[0].ring_id
    member = D.ring_members(w, ring0)[0]

    ov = D.account_overview(w, member)
    assert ov is not None and ov["n_events"] > 0
    assert D.account_overview(w, -999) is None  # missing account

    pop = D.population_stats(w)
    assert set(pop) == set(D.PROFILE_FEATURES)
    anoms = D.anomalies(ov, pop)
    assert len(anoms) == len(D.PROFILE_FEATURES)
    # sorted by |z| descending
    zs = [abs(a.z) for a in anoms]
    assert zs == sorted(zs, reverse=True)


def test_timeline_and_top_creators(tmp_path) -> None:
    w = _world(tmp_path, level="naive")
    member = D.ring_members(w, D.rings(w)[0].ring_id)[0]
    act = D.daily_activity(w, member)
    assert act and all({"day", "views", "skips"} <= set(d) for d in act)
    arts = D.top_creators(w, member, n=5)
    assert arts and len(arts) <= 5
    # views come back descending
    views = [a["views"] for a in arts]
    assert views == sorted(views, reverse=True)


def test_truth_tag_distinguishes_this_ring_from_legit(tmp_path) -> None:
    w = _world(tmp_path, level="naive")
    ring0 = D.rings(w)[0].ring_id
    member = D.ring_members(w, ring0)[0]
    assert D.truth_tag(w, member, ring0) == "this-ring"
    # a legit account: one with no fraud label
    legit_ids = w.gt.filter(~pl.col("is_fraud"))["account_id"].to_list()
    assert D.truth_tag(w, int(legit_ids[0]), ring0) == "legit"


def test_investigation_retrieves_ring_members_with_linkage_evidence(tmp_path) -> None:
    w = _world(tmp_path, level="naive")  # naive: infra linkage retrieves the ring
    ring0 = D.rings(w)[0].ring_id
    members = set(D.ring_members(w, ring0))
    seeds = D.default_seeds(w, ring0, k=3)

    cands = D.investigate(w, seeds)
    assert cands, "expected related candidates"
    retrieved = {c.account_id for c in cands}
    # at least one held-out member of the ring is recovered
    assert (members - set(seeds)) & retrieved

    # the top candidate that is a held-out member shares an in-gate rare identifier with the seeds
    member_cands = [c for c in cands if c.account_id in (members - set(seeds))]
    assert member_cands
    shared = D.shared_with_group(w, member_cands[0].account_id, seeds)
    assert any(s.in_gate for s in shared), "a recovered ring member should share a rare linkage id"


def test_label_store_roundtrip_and_scoring(tmp_path) -> None:
    w = _world(tmp_path, level="naive")
    store = L.store_for_world(w.world_dir)

    fraud_ids = w.gt.filter(pl.col("is_fraud"))["account_id"].to_list()
    legit_ids = w.gt.filter(~pl.col("is_fraud"))["account_id"].to_list()

    store.set(int(fraud_ids[0]), "fraud")
    store.set(int(fraud_ids[1]), "legit")          # a miss -> FN
    store.set(int(legit_ids[0]), "fraud")          # a false alarm -> FP
    store.set(int(legit_ids[1]), "legit")          # correct -> TN
    store.set(int(legit_ids[2]), "unsure")         # excluded from scoring

    # persisted to disk and reloadable
    reloaded = L.store_for_world(w.world_dir)
    assert reloaded.counts() == {"fraud": 2, "legit": 2, "unsure": 1}

    s = reloaded.score(w.gt)
    assert (s.tp, s.fp, s.tn, s.fn) == (1, 1, 1, 1)
    assert s.n_decided == 4
    assert s.precision == 0.5 and s.recall == 0.5 and s.accuracy == 0.5

    reloaded.clear(int(legit_ids[2]))
    assert L.store_for_world(w.world_dir).counts()["unsure"] == 0
