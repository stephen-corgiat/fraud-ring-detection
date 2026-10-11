"""Viewer query layer (no streamlit import — unit-testable on its own).

Loads/builds a world keyed by (preset, evasion level, seed), then answers the
questions a profile view asks: who is this account, how anomalous is it vs the
population, what did it do over time, who is it related to (the engine's own
seed-anchored investigation), and what shared rare identifiers link them.

The world's DuckDB depends only on (preset, level, seed) — the co-target toggle
changes `run_investigation` behaviour, not the precomputed tables — so a load is
safely cached by those three keys.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import duckdb
import polars as pl

from fraudrings.config import EngineConfig, RunConfig, preset
from fraudrings.detector.engine import Candidate, run_investigation
from fraudrings.detector.ingest import SCHEMA_VERSION, build_db
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel
from fraudrings.viewer.raster import RasterData, build_raster

# Features surfaced on a profile, in display order. A superset of the engine's
# behavioral basket plus the volume/active-probe signals an investigator wants.
PROFILE_FEATURES: tuple[str, ...] = (
    "n_views",
    "skip_rate",
    "qualified_rate",
    "views_per_session",
    "gap_mean",
    "gap_std",
    "active_hours",
    "n_creators",
    "top_creator_share",
    "n_ips",
    "n_asns",
    "challenge_fail_rate",
)


@dataclass
class World:
    """A loaded synthetic world: the detector tables + the (eval-only) ground truth."""

    con: duckdb.DuckDBPyConnection
    engine: EngineConfig
    gt: pl.DataFrame
    world_dir: str
    preset_name: str
    level: EvasionLevel
    seed: int
    concurrency: int | None = None  # operator_concurrency: lock-step world when set, else identity

    @property
    def n_accounts(self) -> int:
        return int(self.gt.height)

    @property
    def n_fraud(self) -> int:
        return int(self.gt.filter(pl.col("is_fraud")).height)


@dataclass
class RingInfo:
    ring_id: int
    size: int
    evasion_level: str


@dataclass
class ClusterInfo:
    """A benign bandwagon cohort (lock-step worlds only) — the negative control for serialization."""

    cluster_id: int
    size: int


@dataclass
class Anomaly:
    feature: str
    value: float
    pop_mean: float
    pop_std: float

    @property
    def z(self) -> float:
        return (self.value - self.pop_mean) / self.pop_std if self.pop_std else 0.0


@dataclass
class SharedId:
    id_type: str
    id_value: str
    rarity: float
    distinct_accounts: int
    in_gate: bool  # within the Stage-1 rarity gate (i.e. this edge can actually link them)


def _world_dir(base_dir: str, preset_name: str, level: str, seed: int, concurrency: int | None) -> str:
    # The `-c{C}` suffix is appended ONLY for lock-step worlds, so identity worlds keep their
    # historical path (`{preset}-{level}-{seed}`) — existing caches and test paths are unchanged.
    suffix = "" if concurrency is None else f"-c{concurrency}"
    return os.path.join(base_dir, "viewer", f"{preset_name}-{level}-{seed}{suffix}")


def resolve_config(
    preset_name: str, level: EvasionLevel, seed: int, cotarget_on: bool, concurrency: int | None = None
) -> RunConfig:
    """Build the RunConfig for a world. Co-target widening is on by default here so the
    sophisticated regime (where infra linkage is dead) actually retrieves a ring to inspect.
    `concurrency` (operator_concurrency) selects the lock-step generator branch when set."""
    cfg = preset(preset_name)
    cfg.sim.seed = seed
    cfg.sim.evasion_level = level
    cfg.sim.operator_concurrency = concurrency
    cfg.engine.cotarget_candidate_gen = cotarget_on
    cfg.engine.cotarget_always = cotarget_on
    return cfg


def _db_schema_version(db_path: str) -> int | None:
    """The schema version stamped into a built world, or None for a pre-versioning / unreadable DB.

    A None (or mismatching) version means the cached world predates a generator/ingest change and
    must be rebuilt — otherwise the new columns/tables the viewer queries are simply absent.
    """
    try:
        con = duckdb.connect(db_path, read_only=True)
        try:
            row = con.execute("SELECT version FROM schema_meta").fetchone()
            return int(row[0]) if row and row[0] is not None else None
        finally:
            con.close()
    except Exception:  # noqa: BLE001 — missing schema_meta table, or a DB an older engine wrote
        return None


def load_world(
    preset_name: str,
    level: EvasionLevel,
    seed: int,
    *,
    cotarget_on: bool = True,
    base_dir: str = "data",
    rebuild: bool = False,
    concurrency: int | None = None,
) -> World:
    """Generate (if missing) and ingest a world, returning queryable handles.

    Idempotent: events/ground-truth are generated once per (preset, level, seed, concurrency); the
    DuckDB is built once. `rebuild=True` forces regeneration (e.g. after a generator change).
    `concurrency` set => lock-step world (serialized rings + uncapped bandwagon cohorts).

    **Auto-migration:** a cached world built by an older schema (no/older `schema_meta` stamp — e.g.
    one that lacks the `accounts` table or the action-mix columns) is rebuilt
    from the current generator before use, so `fraudrings view` never trips over missing db columns.
    """
    cfg = resolve_config(preset_name, level, seed, cotarget_on, concurrency)
    wdir = _world_dir(base_dir, preset_name, level, seed, concurrency)
    events_path = os.path.join(wdir, "events.parquet")
    gt_path = os.path.join(wdir, "ground_truth.parquet")
    db_path = os.path.join(wdir, "world.duckdb")

    cached_version = _db_schema_version(db_path) if os.path.exists(db_path) else None
    stale = os.path.exists(db_path) and cached_version != SCHEMA_VERSION
    if stale and not rebuild:
        print(
            f"[fraudrings] migrating cached world '{os.path.basename(wdir)}': schema "
            f"v{cached_version} → v{SCHEMA_VERSION} (regenerating events + features)…"
        )
    do_rebuild = rebuild or stale

    if do_rebuild or not (os.path.exists(events_path) and os.path.exists(gt_path)):
        events_path, gt_path = generate(cfg.sim, wdir)
    if do_rebuild and os.path.exists(db_path):
        os.remove(db_path)
    if not os.path.exists(db_path):
        build_db(events_path, cfg.engine, db_path=db_path).close()

    con = duckdb.connect(db_path, read_only=True)
    gt = pl.read_parquet(gt_path)
    return World(
        con=con, engine=cfg.engine, gt=gt, world_dir=wdir,
        preset_name=preset_name, level=level, seed=seed, concurrency=concurrency,
    )


# --- roster / navigation ---------------------------------------------------


def rings(world: World) -> list[RingInfo]:
    """Fraud rings in the world, largest first — the natural investigation starting points."""
    fraud = world.gt.filter(pl.col("is_fraud") & pl.col("ring_id").is_not_null())
    if fraud.height == 0:
        return []
    grouped = (
        fraud.group_by("ring_id")
        .agg(pl.len().alias("size"), pl.col("evasion_level").first().alias("evasion_level"))
        .sort("size", descending=True)
    )
    return [
        RingInfo(ring_id=int(r["ring_id"]), size=int(r["size"]), evasion_level=str(r["evasion_level"]))
        for r in grouped.iter_rows(named=True)
    ]


def ring_members(world: World, ring_id: int) -> list[int]:
    members = world.gt.filter(pl.col("is_fraud") & (pl.col("ring_id") == ring_id))["account_id"].to_list()
    return sorted(int(a) for a in members)


def default_seeds(world: World, ring_id: int, k: int = 3) -> list[int]:
    """The first k members — mirrors the eval protocol's seed choice (evaluate.py)."""
    return ring_members(world, ring_id)[:k]


def bandwagon_clusters(world: World) -> list[ClusterInfo]:
    """Benign bandwagon cohorts in the world (lock-step worlds only), largest first.

    The bandwagon is the eval-side negative control: co-target and time-correlated with a ring but
    *uncapped* (fans pile on -> a tall occupancy peak). Empty for identity worlds, which have no tag.
    """
    if "bandwagon_cluster_id" not in world.gt.columns:
        return []
    bw = world.gt.filter(pl.col("bandwagon_cluster_id").is_not_null())
    if bw.height == 0:
        return []
    grouped = bw.group_by("bandwagon_cluster_id").agg(pl.len().alias("size"))
    # cohorts often tie on size, and group_by order is not stable: break ties by id so the first
    # cohort is the same on every run (the walkthrough and the site chart both show it)
    grouped = grouped.sort(["size", "bandwagon_cluster_id"], descending=[True, False])
    return [
        ClusterInfo(cluster_id=int(r["bandwagon_cluster_id"]), size=int(r["size"]))
        for r in grouped.iter_rows(named=True)
    ]


def bandwagon_members(world: World, cluster_id: int) -> list[int]:
    """Oracle-clean account ids of one bandwagon cohort (eval-only; used to seed 2AFC pairs)."""
    members = world.gt.filter(pl.col("bandwagon_cluster_id") == cluster_id)["account_id"].to_list()
    return sorted(int(a) for a in members)


def account_exists(world: World, account_id: int) -> bool:
    row = world.con.execute(
        "SELECT 1 FROM account_features WHERE account_id = ? LIMIT 1", [account_id]
    ).fetchone()
    return row is not None


# --- profile ---------------------------------------------------------------


def account_overview(world: World, account_id: int) -> dict[str, float] | None:
    """The full account_features row as a dict, or None if the account has no events."""
    cols = ["account_id", *PROFILE_FEATURES, "n_events", "n_skips", "n_sessions", "n_items"]
    sel = ", ".join(cols)
    row = world.con.execute(
        f"SELECT {sel} FROM account_features WHERE account_id = ?", [account_id]
    ).fetchone()
    if row is None:
        return None
    return {c: (float(v) if v is not None else 0.0) for c, v in zip(cols, row, strict=True)}


def population_stats(world: World) -> dict[str, tuple[float, float]]:
    """(mean, std) per profile feature over all accounts — the anomaly baseline."""
    parts = [f"avg({f}) AS {f}_m, coalesce(stddev_samp({f}), 0) AS {f}_s" for f in PROFILE_FEATURES]
    row = world.con.execute(f"SELECT {', '.join(parts)} FROM account_features").fetchone()
    out: dict[str, tuple[float, float]] = {}
    assert row is not None
    for i, f in enumerate(PROFILE_FEATURES):
        out[f] = (float(row[2 * i] or 0.0), float(row[2 * i + 1] or 0.0))
    return out


def anomalies(overview: dict[str, float], pop: dict[str, tuple[float, float]]) -> list[Anomaly]:
    """Per-feature value vs population, sorted by |z| (most anomalous first)."""
    out = [
        Anomaly(feature=f, value=overview.get(f, 0.0), pop_mean=pop[f][0], pop_std=pop[f][1])
        for f in PROFILE_FEATURES
    ]
    out.sort(key=lambda a: abs(a.z), reverse=True)
    return out


def daily_activity(world: World, account_id: int) -> list[dict[str, int]]:
    """Views/skips per day, indexed off the world's first event so a burst's position is visible."""
    rows = world.con.execute(
        """
        WITH base AS (SELECT min(ts) AS t0 FROM events)
        SELECT CAST(floor((ts - (SELECT t0 FROM base)) / 86400) AS INTEGER) AS day,
               count(*) FILTER (WHERE action IN ('view','complete')) AS views,
               count(*) FILTER (WHERE action = 'skip')              AS skips
        FROM events WHERE account_id = ?
        GROUP BY day ORDER BY day
        """,
        [account_id],
    ).fetchall()
    return [{"day": int(r[0]), "views": int(r[1]), "skips": int(r[2])} for r in rows]


def top_creators(world: World, account_id: int, n: int = 8) -> list[dict[str, int]]:
    thr = world.engine.qualified_view_threshold
    rows = world.con.execute(
        f"""
        SELECT creator_id,
               count(*) FILTER (WHERE action IN ('view','complete'))                       AS views,
               count(*) FILTER (WHERE action IN ('view','complete')
                                  AND reported_position_pct >= {thr})                       AS qualified
        FROM events
        WHERE account_id = ? AND creator_id IS NOT NULL
        GROUP BY creator_id ORDER BY views DESC LIMIT ?
        """,
        [account_id, n],
    ).fetchall()
    return [{"creator_id": int(r[0]), "views": int(r[1]), "qualified": int(r[2])} for r in rows]


# --- ground truth (eval-side reveal) ---------------------------------------


def ground_truth_of(world: World, account_id: int) -> dict | None:
    row = world.gt.filter(pl.col("account_id") == account_id)
    if row.height == 0:
        return None
    r = row.row(0, named=True)
    return {
        "is_fraud": bool(r["is_fraud"]),
        "ring_id": None if r["ring_id"] is None else int(r["ring_id"]),
        "evasion_level": str(r["evasion_level"]),
    }


def truth_tag(world: World, account_id: int, subject_ring_id: int | None) -> str:
    """A flag's operational verdict relative to the investigation's ring:
    this-ring TP, other-ring fraud (still fraud), or legit (the real false positive)."""
    gt = ground_truth_of(world, account_id)
    if gt is None or not gt["is_fraud"]:
        return "legit"
    if subject_ring_id is not None and gt["ring_id"] == subject_ring_id:
        return "this-ring"
    return "other-ring"


# --- investigation / linkage evidence --------------------------------------


def investigate(world: World, seeds: list[int]) -> list[Candidate]:
    """Run the engine's seed-anchored link-then-score and return ranked related accounts."""
    return run_investigation(world.con, seeds, world.engine)


# --- temporal raster -------------------------------------------------------


def account_raster(
    world: World,
    account_ids: list[int],
    *,
    bucket_seconds: int = 300,
    t0: int | None = None,
    t1: int | None = None,
) -> RasterData:
    """Build the event-raster + occupancy strip for an account set (see `viewer.raster`).

    Presentation-only and event-derived: nothing here is read by the detector or feeds scoring.
    Pass a set's oracle-clean membership (a ring or a bandwagon cohort) for the Stage-1 ceiling
    view, or a retrieved/polluted candidate set for the Stage-2 segmentation view.
    """
    return build_raster(world.con, account_ids, bucket_seconds=bucket_seconds, t0=t0, t1=t1)


def shared_identifiers(world: World, a: int, b: int) -> list[SharedId]:
    """Rare identifiers both accounts touch — the actual linkage evidence behind a candidate.

    `in_gate` marks the ones inside the Stage-1 rarity gate (shared by >=expand_min and
    <=expand_max accounts): only those can carry linkage score; the rest are too common
    (e.g. a big mobile ASN) to be evidence.
    """
    types = "(" + ",".join(f"'{t}'" for t in world.engine.link_id_types) + ")"
    lo, hi = world.engine.expand_min_accounts, world.engine.expand_max_accounts
    rows = world.con.execute(
        f"""
        WITH shared AS (
            SELECT id_type, id_value, count(DISTINCT account_id) AS n
            FROM assoc
            WHERE account_id IN (?, ?) AND id_type IN {types}
            GROUP BY id_type, id_value
            HAVING count(DISTINCT account_id) = 2
        )
        SELECT s.id_type, s.id_value, st.rarity, st.distinct_accounts
        FROM shared s JOIN identifier_stats st USING (id_type, id_value)
        ORDER BY st.rarity DESC
        """,
        [a, b],
    ).fetchall()
    return [
        SharedId(
            id_type=str(r[0]), id_value=str(r[1]), rarity=float(r[2]),
            distinct_accounts=int(r[3]), in_gate=(lo <= int(r[3]) <= hi),
        )
        for r in rows
    ]


def shared_with_group(world: World, account_id: int, group: list[int]) -> list[SharedId]:
    """Rare identifiers `account_id` shares with *any* member of a seed group — the linkage
    evidence behind a candidate's rank, since the engine links to the seed set as a whole."""
    others = [g for g in group if g != account_id]
    if not others:
        return []
    types = "(" + ",".join(f"'{t}'" for t in world.engine.link_id_types) + ")"
    grp = "(" + ",".join(str(int(g)) for g in others) + ")"
    lo, hi = world.engine.expand_min_accounts, world.engine.expand_max_accounts
    rows = world.con.execute(
        f"""
        WITH cand AS (
            SELECT DISTINCT id_type, id_value FROM assoc
            WHERE account_id = ? AND id_type IN {types}
        ),
        grp AS (
            SELECT DISTINCT id_type, id_value FROM assoc
            WHERE account_id IN {grp} AND id_type IN {types}
        )
        SELECT c.id_type, c.id_value, st.rarity, st.distinct_accounts
        FROM cand c JOIN grp g USING (id_type, id_value)
                    JOIN identifier_stats st USING (id_type, id_value)
        ORDER BY st.rarity DESC
        """,
        [account_id],
    ).fetchall()
    return [
        SharedId(
            id_type=str(r[0]), id_value=str(r[1]), rarity=float(r[2]),
            distinct_accounts=int(r[3]), in_gate=(lo <= int(r[3]) <= hi),
        )
        for r in rows
    ]


# --- chart substrate (the Plotly chart spine) ------------------------------
#
# These power the Plotly chart spine in `viewer.charts`. All are presentation-only and
# event-derived: nothing returned here is read by the detector or feeds scoring. They run
# on-the-fly over a passed account set (the seed-anchored candidate set), or — with
# `account_ids=None`, where supported — over the whole loaded world to compute a baseline.

# Funnel actions surfaced by the action-composition chart, in funnel order.
FUNNEL_ACTIONS: tuple[str, ...] = ("search", "view", "complete", "skip", "save")
# Cap on raw rows pulled for a population baseline distribution — a histogram only
# needs a representative slice, and the browser shouldn't render the full default-preset stream.
_BASELINE_SAMPLE_CAP = 20_000


def _ids_clause(account_ids: list[int]) -> str:
    return "(" + ",".join(str(int(a)) for a in account_ids) + ")"


def action_composition(world: World, account_ids: list[int]) -> list[dict[str, int]]:
    """Per-account counts of each funnel action (explains the action-mix axis).

    A naive ring is glaringly truncated (direct target views, ~no search/save, near-100%
    `complete`); a legit/lurker mix carries the full funnel. Returned in `account_ids` order.
    """
    if not account_ids:
        return []
    parts = [f"count(*) FILTER (WHERE action='{a}') AS {a}" for a in FUNNEL_ACTIONS]
    rows = world.con.execute(
        f"SELECT account_id, {', '.join(parts)} FROM events "
        f"WHERE account_id IN {_ids_clause(account_ids)} GROUP BY account_id"
    ).fetchall()
    by_id = {int(r[0]): r for r in rows}
    out: list[dict[str, int]] = []
    for a in account_ids:
        r = by_id.get(a)
        rec: dict[str, int] = {"account_id": a}
        for i, act in enumerate(FUNNEL_ACTIONS, start=1):
            rec[act] = int(r[i]) if r is not None else 0
        out.append(rec)
    return out


def position_samples(world: World, account_ids: list[int] | None = None) -> list[float]:
    """Reported view positions. `account_ids=None` -> a population-baseline sample.

    The threshold to overlay is `world.engine.qualified_view_threshold`: a payout-farming
    ring stacks mass just above it; legit media events spread across the [0,1] range.
    """
    where = "reported_position_pct IS NOT NULL AND action IN ('view','complete','skip')"
    if account_ids is not None:
        if not account_ids:
            return []
        where += f" AND account_id IN {_ids_clause(account_ids)}"
        limit = ""
    else:
        limit = f" LIMIT {_BASELINE_SAMPLE_CAP}"
    rows = world.con.execute(
        f"SELECT reported_position_pct FROM events WHERE {where}{limit}"
    ).fetchall()
    return [float(r[0]) for r in rows]


def gap_samples(world: World, account_ids: list[int] | None = None) -> list[float]:
    """Inter-event gaps (seconds) between successive media events, per account (cadence comb).

    A cadenced ring burst piles up at a near-fixed gap (a sharp comb tooth); organic viewing
    smears across minutes-to-days. `account_ids=None` -> a population-baseline sample.
    """
    where = ""
    limit = ""
    if account_ids is not None:
        if not account_ids:
            return []
        where = f" AND account_id IN {_ids_clause(account_ids)}"
    else:
        limit = f" LIMIT {_BASELINE_SAMPLE_CAP}"
    rows = world.con.execute(
        f"""
        WITH g AS (
            SELECT ts - lag(ts) OVER (PARTITION BY account_id ORDER BY ts) AS gap
            FROM events WHERE action IN ('view','complete','skip'){where}
        )
        SELECT gap FROM g WHERE gap IS NOT NULL{limit}
        """
    ).fetchall()
    return [float(r[0]) for r in rows]


def cotarget_matrix(
    world: World, account_ids: list[int], top_n: int = 12
) -> tuple[list[int], list[int], list[list[int]]]:
    """Account × top-creator qualified-view matrix (heatmap).

    A co-targeting ring lights up a dense block on the shared rare target creator(s); genuine
    fans scatter. Columns are the `top_n` creators by qualified-view volume across the set.
    Returns (accounts in input order, creator_ids, matrix[acc][creator]).
    """
    if not account_ids:
        return [], [], []
    thr = world.engine.qualified_view_threshold
    rows = world.con.execute(
        f"""
        SELECT account_id, creator_id, count(*) AS c
        FROM events
        WHERE account_id IN {_ids_clause(account_ids)} AND creator_id IS NOT NULL
          AND action IN ('view','complete') AND reported_position_pct >= {thr}
        GROUP BY account_id, creator_id
        """
    ).fetchall()
    counts: dict[tuple[int, int], int] = {}
    creator_tot: dict[int, int] = {}
    for aid, art, c in rows:
        counts[(int(aid), int(art))] = int(c)
        creator_tot[int(art)] = creator_tot.get(int(art), 0) + int(c)
    creators = [a for a, _ in sorted(creator_tot.items(), key=lambda kv: (-kv[1], kv[0]))[:top_n]]
    matrix = [[counts.get((aid, art), 0) for art in creators] for aid in account_ids]
    return list(account_ids), creators, matrix


def registration_offsets(world: World, account_ids: list[int]) -> list[dict[str, float]]:
    """Each account's registration time relative to the observation window (burst histogram).

    `offset_days` = days after the window's first event (negative => registered before the
    window, i.e. an aged account). A freshly-minted ring co-registers in a tight in-window
    spike; aged legit/lurker accounts sit at large negative offsets. `in_window` flags the spike.
    """
    if not account_ids:
        return []
    rows = world.con.execute(
        f"""
        WITH w AS (SELECT min(ts) AS t0 FROM events)
        SELECT ac.account_id, ac.registration_ts,
               (ac.registration_ts - (SELECT t0 FROM w)) / 86400.0 AS offset_days,
               (ac.registration_ts >= (SELECT t0 FROM w))         AS in_window
        FROM accounts ac WHERE ac.account_id IN {_ids_clause(account_ids)}
        """
    ).fetchall()
    return [
        {
            "account_id": int(r[0]), "registration_ts": int(r[1]),
            "offset_days": float(r[2]), "in_window": bool(r[3]),
        }
        for r in rows
    ]


def account_age_samples(world: World, account_ids: list[int] | None = None) -> list[dict[str, float]]:
    """`account_age_days` per account (the account-age distribution chart).

    A fresh ring clusters at low age with low variance; legit/lurker accounts spread broad and
    old. `account_ids=None` -> the whole population (the baseline cloud).
    """
    if account_ids is not None and not account_ids:
        return []
    where = "" if account_ids is None else f" WHERE account_id IN {_ids_clause(account_ids)}"
    rows = world.con.execute(
        f"SELECT account_id, account_age_days FROM account_features{where}"
    ).fetchall()
    if account_ids is None:
        return [{"account_id": int(r[0]), "age": float(r[1])} for r in rows]
    # Preserve input order (mirrors action_composition) so a point's customdata is predictable;
    # silently drop any selected id with no events (not in account_features).
    by_id = {int(r[0]): float(r[1]) for r in rows}
    return [{"account_id": a, "age": by_id[a]} for a in account_ids if a in by_id]


def linkage_graph(world: World, account_ids: list[int]) -> tuple[list[int], list[dict]]:
    """Account–account linkage subgraph over the set (node-link).

    An edge joins two accounts that share an in-rarity-gate identifier (IP/subnet/ASN/fp/cookie);
    its weight is the summed rarity (IDF) of those shared identifiers — exactly what the linkage
    axis scores on. A ring shows as a dense rare-infra clump; organic accounts stay isolated.
    Returns (nodes = all set accounts, edges = [{a, b, weight, n_shared}]).
    """
    nodes = [int(a) for a in account_ids]
    if len(nodes) < 2:
        return nodes, []
    types = "(" + ",".join(f"'{t}'" for t in world.engine.link_id_types) + ")"
    lo, hi = world.engine.expand_min_accounts, world.engine.expand_max_accounts
    rows = world.con.execute(
        f"""
        WITH sa AS (
            SELECT a.account_id, a.id_type, a.id_value, st.rarity
            FROM assoc a JOIN identifier_stats st USING (id_type, id_value)
            WHERE a.account_id IN {_ids_clause(nodes)} AND a.id_type IN {types}
              AND st.distinct_accounts BETWEEN {lo} AND {hi}
        ),
        shared AS (
            SELECT id_type, id_value FROM sa
            GROUP BY id_type, id_value HAVING count(DISTINCT account_id) >= 2
        )
        SELECT sa.id_type, sa.id_value, sa.rarity, sa.account_id
        FROM sa JOIN shared s USING (id_type, id_value)
        ORDER BY sa.id_type, sa.id_value, sa.account_id
        """
    ).fetchall()
    members: dict[tuple[str, str], tuple[float, list[int]]] = {}
    for id_type, id_value, rarity, aid in rows:
        key = (str(id_type), str(id_value))
        rar, accs = members.get(key, (float(rarity), []))
        accs.append(int(aid))
        members[key] = (float(rarity), accs)
    edges: dict[tuple[int, int], dict] = {}
    for rarity, accs in members.values():
        for i in range(len(accs)):
            for j in range(i + 1, len(accs)):
                a, b = sorted((accs[i], accs[j]))
                e = edges.get((a, b))
                if e is None:
                    edges[(a, b)] = {"a": a, "b": b, "weight": rarity, "n_shared": 1}
                else:
                    e["weight"] += rarity
                    e["n_shared"] += 1
    return nodes, list(edges.values())
