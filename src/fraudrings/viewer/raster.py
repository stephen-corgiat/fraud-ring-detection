"""Temporal raster + occupancy renderer.

A *presentation-only*, streamlit-free data layer for the lock-step temporal views. Lock-step
serialization turned out not to be recoverable by any per-account scalar (see
`docs/adr/`), so this shows a human the shape directly. It renders the two things the failed `peak/size` MAX
statistic threw away:

  * **Event raster** — rows = accounts, X = absolute time, each cell the event count in a time
    bucket. Per-account identity is preserved (so a human can circle a ring core in Stage 2).
  * **Occupancy strip** — sharing the X axis: the count of accounts *simultaneously active* per
    bucket. A serialized ring is a **flat plateau hugging the concurrency cap C**; an uncapped
    bandwagon is a **tall transient peak**. That shape is the signal.

Hard fairness constraints baked in:
  1. **Row order is event-derivable only** — accounts are ordered by activity-onset time, NEVER by
     ground-truth membership. Handing the eye a GT-sorted raster would leak the answer.
  2. This module reads only the event stream (`events` table). It never touches ground truth, so it
     stays an analyst aid, not a detector feature: no value here feeds scoring.

Keep this layer streamlit-free and unit-tested (the `data.py` / `app.py` split).
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np


@dataclass
class RasterData:
    """Render-ready temporal data for one account set, both views sharing the X axis.

    `points` is long-form ``(row, bucket, count)`` so a renderer can draw dots or densify to a
    matrix; `occupancy[b]` is the number of distinct accounts active in bucket ``b`` (session
    intervals merged, so an account with overlapping sessions still counts once per bucket).
    """

    accounts: list[int]  # row order: activity-onset ascending (event-derivable, never GT)
    bucket_seconds: int
    t0: int  # absolute epoch of bucket 0's left edge
    n_buckets: int
    points: list[tuple[int, int, int]]  # (row_index, bucket_index, event_count)
    occupancy: list[int]  # len == n_buckets

    @property
    def n_accounts(self) -> int:
        return len(self.accounts)

    @property
    def peak_occupancy(self) -> int:
        """Max simultaneous active accounts — the set-level statistic the cap caps for a ring."""
        return max(self.occupancy) if self.occupancy else 0

    def matrix(self) -> np.ndarray:
        """Densify `points` to an ``(n_accounts, n_buckets)`` event-count matrix (row 0 = top)."""
        m = np.zeros((max(self.n_accounts, 1), max(self.n_buckets, 1)), dtype=np.int64)
        for row, bucket, count in self.points:
            m[row, bucket] = count
        return m


def _fetch_events(
    con: duckdb.DuckDBPyConnection, account_ids: list[int]
) -> list[tuple[int, str, int]]:
    if not account_ids:
        return []
    clause = "(" + ",".join(str(int(a)) for a in account_ids) + ")"
    return [
        (int(a), str(s), int(t))
        for a, s, t in con.execute(
            f"SELECT account_id, session_id, ts FROM events WHERE account_id IN {clause}"
        ).fetchall()
    ]


def build_raster(
    con: duckdb.DuckDBPyConnection,
    account_ids: list[int],
    *,
    bucket_seconds: int = 300,
    t0: int | None = None,
    t1: int | None = None,
) -> RasterData:
    """Build the raster + occupancy for `account_ids` at `bucket_seconds` granularity.

    Time window defaults to the set's own [min, max] event ts (the lock-step burst dominates a
    ring/bandwagon account's activity); pass `t0`/`t1` to clip to a fixed window (e.g. align two
    cohorts on one axis for a 2AFC pair). Buckets containing no event are still emitted (occupancy
    can be > 0 there if a session spans the gap), so the plateau/peak shape is faithful.

    Occupancy is the **peak instantaneous** simultaneity within each bucket: each (account, session)
    collapses to ``[first_event, last_event]``, an account's sessions are merged into disjoint busy
    intervals (so the same account never counts twice when two of its sessions overlap), and a
    sweep line gives the number of distinct accounts active at every instant. Per bucket we keep the
    MAX of that step function — so a serialized ring reads as a flat plateau at ~C (never more than
    C active at any instant, even though far more than C *touch* a 5-min bucket as lanes take turns),
    while the uncapped bandwagon spikes above C. Counting accounts that merely *touch* a bucket would
    wash this out (a lane serves several sessions per bucket), so the MAX-of-instantaneous is load-
    bearing: it matches `tests/test_lockstep._max_simultaneous`, not a per-bucket headcount.
    """
    if bucket_seconds < 1:
        raise ValueError(f"bucket_seconds must be >= 1, got {bucket_seconds}")
    rows = _fetch_events(con, account_ids)
    if not rows:
        return RasterData(
            accounts=[], bucket_seconds=bucket_seconds, t0=t0 or 0, n_buckets=0, points=[], occupancy=[]
        )

    lo = t0 if t0 is not None else min(t for _a, _s, t in rows)
    hi = t1 if t1 is not None else max(t for _a, _s, t in rows)
    if hi < lo:
        lo, hi = hi, lo
    n_buckets = int((hi - lo) // bucket_seconds) + 1

    def bucket_of(ts: int) -> int:
        return min(max((ts - lo) // bucket_seconds, 0), n_buckets - 1)

    # Row order = activity onset (first event ts), ascending — event-derivable, never GT. Ties
    # broken by account_id for a stable, reproducible ordering.
    onset: dict[int, int] = {}
    sessions: dict[tuple[int, str], list[int]] = {}
    for a, s, t in rows:
        if a not in onset or t < onset[a]:
            onset[a] = t
        sessions.setdefault((a, s), [t, t])
        iv = sessions[(a, s)]
        if t < iv[0]:
            iv[0] = t
        if t > iv[1]:
            iv[1] = t
    accounts = sorted(onset, key=lambda a: (onset[a], a))
    row_of = {a: i for i, a in enumerate(accounts)}

    # Event-count cells for the raster.
    cell_counts: dict[tuple[int, int], int] = {}
    for a, _s, t in rows:
        key = (row_of[a], bucket_of(t))
        cell_counts[key] = cell_counts.get(key, 0) + 1
    points = [(r, b, c) for (r, b), c in sorted(cell_counts.items())]

    occupancy = _peak_instantaneous_occupancy(sessions, lo=lo, hi=hi, bucket_seconds=bucket_seconds,
                                               n_buckets=n_buckets)

    return RasterData(
        accounts=accounts,
        bucket_seconds=bucket_seconds,
        t0=lo,
        n_buckets=n_buckets,
        points=points,
        occupancy=occupancy,
    )


def _merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge an account's session intervals; touching (s <= cur_e) counts as overlapping."""
    intervals.sort()
    merged: list[tuple[int, int]] = []
    cur_s, cur_e = intervals[0]
    for s, e in intervals[1:]:
        if s <= cur_e:
            cur_e = max(cur_e, e)
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))
    return merged


def _peak_instantaneous_occupancy(
    sessions: dict[tuple[int, str], list[int]], *, lo: int, hi: int, bucket_seconds: int, n_buckets: int
) -> list[int]:
    """Per-bucket MAX of the instantaneous distinct-active-account count.

    Sweep line over per-account *merged* busy intervals (so one account counts at most +1 at any
    instant), clipped to [lo, hi]; at a tie opens precede closes (touching = overlapping), matching
    `tests/test_lockstep._max_simultaneous`. The step function is constant between event times, so
    each constant segment updates the running max of every display bucket it overlaps.
    """
    by_account: dict[int, list[tuple[int, int]]] = {}
    for (a, _s), (s_ts, e_ts) in sessions.items():
        s_c, e_c = max(int(s_ts), lo), min(int(e_ts), hi)
        if e_c >= s_c:
            by_account.setdefault(a, []).append((s_c, e_c))

    deltas: list[tuple[int, int]] = []
    for intervals in by_account.values():
        for s, e in _merge_intervals(intervals):
            deltas.append((s, 1))
            deltas.append((e, -1))
    bucket_max = [0] * max(n_buckets, 1)
    if not deltas:
        return bucket_max[:n_buckets]

    deltas.sort(key=lambda d: (d[0], -d[1]))  # opens (+1) before closes (-1) at a tie
    # Collapse to breakpoints: occupancy == occ on [t, next_t).
    breakpoints: list[tuple[int, int]] = []
    cur = i = 0
    n = len(deltas)
    while i < n:
        t = deltas[i][0]
        while i < n and deltas[i][0] == t:
            cur += deltas[i][1]
            i += 1
        breakpoints.append((t, cur))

    for k, (t_k, occ) in enumerate(breakpoints):
        if occ <= 0:
            continue
        t_next = breakpoints[k + 1][0] if k + 1 < len(breakpoints) else hi + 1
        b_lo = min(max((t_k - lo) // bucket_seconds, 0), n_buckets - 1)
        # the segment is half-open [t_k, t_next); the last bucket it touches holds the instant before t_next
        b_hi = min(max((t_next - 1 - lo) // bucket_seconds, 0), n_buckets - 1)
        for b in range(b_lo, b_hi + 1):
            if occ > bucket_max[b]:
                bucket_max[b] = occ
    return bucket_max[:n_buckets]
