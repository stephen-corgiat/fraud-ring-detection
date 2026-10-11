"""Batch pre-compute: from the event stream, build the tables the
detector queries — `account_features`, `assoc` (the graph at rest), `identifier_stats`
(rarity / IDF weights).

`window` is parameterized so the same code drives a single batch *or* time-ordered
micro-batches (the incremental path). The pipeline calls it once.
"""

from __future__ import annotations

import os

import duckdb

from fraudrings.config import EngineConfig

# Bump when a generator/ingest change makes an older cached world's DuckDB unreadable by the
# current queries. `build_db` stamps it into a `schema_meta` table; `viewer.data.load_world`
# rebuilds any world whose stamp differs — a lightweight migration (no migration framework here;
# worlds are regenerated from the deterministic generator). A pre-versioning DB has no `schema_meta`
# table (treated as v1). v2 = the `accounts` table + the action-mix columns
# (completion/search/save rates, funnel_completeness, account_age_days).
# v3 = the creator_id / item_id columns and the `view` action.
SCHEMA_VERSION = 3


def build_db(
    events_path: str,
    engine: EngineConfig,
    *,
    db_path: str = ":memory:",
    window: tuple[int, int] | None = None,
    accounts_path: str | None = None,
) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(db_path)
    where_ts = ""
    if window is not None:
        where_ts = f"AND ts BETWEEN {int(window[0])} AND {int(window[1])}"

    # events: only the detector-visible stream (ground_truth is never loaded here).
    con.execute(
        f"""
        CREATE OR REPLACE TABLE events AS
        SELECT * FROM read_parquet('{events_path}')
        WHERE account_id IS NOT NULL {where_ts};
        """
    )

    # accounts registry: a detector-READABLE side table (registration dates,
    # NOT ground truth) -> `account_age_days`. Defaults to accounts.parquet beside events.parquet.
    # Absent (an older data dir) -> an empty table, and account_age_days falls back to 0.
    if accounts_path is None:
        accounts_path = os.path.join(os.path.dirname(events_path), "accounts.parquet")
    if os.path.exists(accounts_path):
        con.execute(
            f"CREATE OR REPLACE TABLE accounts AS SELECT * FROM read_parquet('{accounts_path}');"
        )
    else:
        con.execute(
            "CREATE OR REPLACE TABLE accounts (account_id BIGINT, registration_ts BIGINT);"
        )

    # --- association edge table (the "graph" at rest) ----------------------
    con.execute(
        """
        CREATE OR REPLACE TABLE assoc AS
        SELECT account_id, 'ip' AS id_type, ip AS id_value, count(*) AS cnt FROM events GROUP BY 1, 3
        UNION ALL
        SELECT account_id, 'subnet', subnet, count(*) FROM events GROUP BY 1, 3
        UNION ALL
        SELECT account_id, 'asn', CAST(asn AS VARCHAR), count(*) FROM events GROUP BY 1, 3
        UNION ALL
        SELECT account_id, 'fp', fingerprint, count(*) FROM events GROUP BY 1, 3
        UNION ALL
        SELECT account_id, 'cookie', session_id, count(*) FROM events GROUP BY 1, 3
        UNION ALL
        SELECT account_id, 'creator', CAST(creator_id AS VARCHAR), count(*)
            FROM events WHERE creator_id IS NOT NULL GROUP BY 1, 3;
        """
    )

    # --- identifier rarity (IDF): surprise of sharing an identifier --------
    n_accounts = con.execute("SELECT count(DISTINCT account_id) FROM events").fetchone()
    total_accounts = int(n_accounts[0]) if n_accounts else 0
    con.execute(
        f"""
        CREATE OR REPLACE TABLE identifier_stats AS
        SELECT id_type, id_value,
               count(DISTINCT account_id) AS distinct_accounts,
               sum(cnt)                    AS total_events,
               ln({max(total_accounts, 1)}.0 / count(DISTINCT account_id)) AS rarity
        FROM assoc
        GROUP BY 1, 2;
        """
    )

    # --- per-account features ----------------------------------------------
    # The behavioral basket is computed over MEDIA events only
    # (view/complete/skip) so the funnel events (search/save/register/login) cannot perturb it —
    # `complete` folds back into `n_views`, so splitting view->complete is metric-neutral and the
    # behavioral features are independent of them. The funnel events feed only the action-mix sub-features
    # (completion/search/save rates, funnel_completeness) + `account_age_days` (the new axis).
    now_row = con.execute("SELECT max(ts) FROM events").fetchone()
    now_ts = int(now_row[0]) if now_row and now_row[0] is not None else 0
    con.execute(
        f"""
        CREATE OR REPLACE TABLE account_features AS
        WITH media AS (
            SELECT * FROM events WHERE action IN ('view','complete','skip')
        ),
        gaps AS (
            SELECT account_id,
                   ts - lag(ts) OVER (PARTITION BY account_id ORDER BY ts) AS gap
            FROM media
        ),
        gap_stats AS (
            SELECT account_id, avg(gap) AS gap_mean, coalesce(stddev_samp(gap), 0) AS gap_std
            FROM gaps WHERE gap IS NOT NULL GROUP BY account_id
        ),
        creator_views AS (
            SELECT account_id, creator_id, count(*) AS c
            FROM media WHERE action IN ('view','complete') AND creator_id IS NOT NULL
            GROUP BY 1, 2
        ),
        top_creator AS (
            SELECT account_id, max(c) AS top_c, sum(c) AS tot_c FROM creator_views GROUP BY account_id
        ),
        -- funnel-event tallies (over the WHOLE stream) for the action-mix axis.
        funnel AS (
            SELECT account_id,
                count(*) FILTER (WHERE action='search') AS n_search,
                count(*) FILTER (WHERE action='save')   AS n_save
            FROM events GROUP BY account_id
        ),
        base AS (
            SELECT l.account_id,
                count(*)                                   AS n_media,
                count(*) FILTER (WHERE action IN ('view','complete')) AS n_views,
                count(*) FILTER (WHERE action='skip')      AS n_skips,
                count(*) FILTER (WHERE action='complete')  AS n_complete,
                count(*) FILTER (WHERE action IN ('view','complete')
                                   AND reported_position_pct >= {engine.qualified_view_threshold})
                                                           AS n_qualified,
                count(DISTINCT session_id)                 AS n_sessions,
                count(DISTINCT item_id)                   AS n_items,
                count(DISTINCT creator_id)                  AS n_creators,
                count(DISTINCT ip)                         AS n_ips,
                count(DISTINCT asn)                        AS n_asns,
                count(DISTINCT hour(to_timestamp(ts)))     AS active_hours,
                count(*) FILTER (WHERE challenge_result='fail') AS n_chal_fail,
                count(*) FILTER (WHERE challenge_result='pass') AS n_chal_pass
            FROM media l GROUP BY l.account_id
        ),
        -- n_events keeps its whole-stream meaning (the viewer's "X events" overview).
        ev AS (SELECT account_id, count(*) AS n_events FROM events GROUP BY account_id)
        SELECT
            b.account_id,
            e.n_events, b.n_views, b.n_skips, b.n_sessions,
            b.n_items, b.n_creators, b.n_ips, b.n_asns, b.active_hours,
            coalesce(g.gap_mean, 0.0) AS gap_mean,
            coalesce(g.gap_std, 0.0)  AS gap_std,
            CASE WHEN b.n_views > 0 THEN b.n_skips    * 1.0 / b.n_views    ELSE 0 END AS skip_rate,
            CASE WHEN b.n_views > 0 THEN b.n_qualified* 1.0 / b.n_views    ELSE 0 END AS qualified_rate,
            CASE WHEN b.n_sessions>0 THEN b.n_views   * 1.0 / b.n_sessions ELSE 0 END AS views_per_session,
            CASE WHEN t.tot_c   > 0 THEN t.top_c      * 1.0 / t.tot_c      ELSE 0 END AS top_creator_share,
            -- active-challenge outcome (the challenge feedback loop). Bots fail / never pass; legit
            -- pass. A near-perfect tell, so it's kept OUT of the headline rule and used only as an
            -- ablation / operational lever (grade the weak-signal basket honestly).
            CASE WHEN (b.n_chal_fail + b.n_chal_pass) > 0
                 THEN b.n_chal_fail * 1.0 / (b.n_chal_fail + b.n_chal_pass) ELSE 0 END AS challenge_fail_rate,
            -- --- action-mix axis sub-features -----------------------------------------------------
            -- completion_rate: share of non-skip media events that ran to completion (naive ring spoofs
            -- ~1.0; a sophisticated ring softens it toward legit). search/save rates + funnel
            -- coverage read the funnel SHAPE. account_age_days from the registry (recency).
            CASE WHEN b.n_views > 0 THEN b.n_complete * 1.0 / b.n_views ELSE 0 END AS completion_rate,
            CASE WHEN b.n_media > 0 THEN coalesce(f.n_search,0) * 1.0 / b.n_media ELSE 0 END AS search_rate,
            CASE WHEN b.n_media > 0 THEN coalesce(f.n_save,0)   * 1.0 / b.n_media ELSE 0 END AS save_rate,
            (CASE WHEN coalesce(f.n_search,0) > 0 THEN 1 ELSE 0 END
             + CASE WHEN coalesce(f.n_save,0) > 0 THEN 1 ELSE 0 END
             + CASE WHEN b.n_complete > 0 THEN 1 ELSE 0 END) / 3.0 AS funnel_completeness,
            CASE WHEN ac.registration_ts IS NOT NULL
                 THEN ({now_ts} - ac.registration_ts) / 86400.0 ELSE 0.0 END AS account_age_days
        FROM base b
        LEFT JOIN gap_stats g USING (account_id)
        LEFT JOIN top_creator t USING (account_id)
        LEFT JOIN funnel f USING (account_id)
        LEFT JOIN ev e USING (account_id)
        LEFT JOIN accounts ac USING (account_id);
        """
    )

    # Stamp the schema version so a stale cached world can be detected and rebuilt (see load_world).
    con.execute("CREATE OR REPLACE TABLE schema_meta (version INTEGER);")
    con.execute("INSERT INTO schema_meta VALUES (?);", [SCHEMA_VERSION])
    return con
