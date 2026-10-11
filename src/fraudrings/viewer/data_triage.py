"""Triage-page data layer: the seedless suspect shortlist and its evidence. No streamlit import.

The triage page runs the concentration ranker (`fraudrings.detector.concentration`) over a loaded
world and shows, for each suspect creator, the evidence behind its rank: which entry axis fired,
how concentrated its contributors are, when its views landed, and who the dedicated core is. The
question the page puts to a human: with this evidence, can you separate a coordinated ring from a
benign superfan cohort?

The ranker and the drill-down queries read only `events`. The ground-truth helpers at the bottom
read the world's hidden labels, so the page can grade a human's calls. They sit on the eval side of
the anti-tautology line and never feed scoring.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal

import duckdb
import polars as pl

from fraudrings.config import ConcentrationConfig
from fraudrings.detector.concentration import SuspectTarget, generate_suspects
from fraudrings.eval.concentration import ring_target_map
from fraudrings.viewer.data import World

QUAL = ConcentrationConfig().qualified_view_threshold

SuspectVerdict = Literal["ring", "legit", "unsure", "escalate"]
SUSPECT_VERDICTS: tuple[SuspectVerdict, ...] = ("ring", "legit", "unsure", "escalate")


def suspects(con: duckdb.DuckDBPyConnection, cfg: ConcentrationConfig | None = None) -> list[SuspectTarget]:
    """The ranked shortlist (reads only `events`)."""
    return generate_suspects(con, cfg)


def top_contributors(con: duckdb.DuckDBPyConnection, target_id: int, limit: int = 20) -> list[dict]:
    """Top contributors to a creator by qualified views — the visual behind `n_eff`."""
    rows = con.execute(
        f"""
        SELECT account_id, count(*) AS qv
        FROM events
        WHERE creator_id = ? AND action IN ('view','complete') AND reported_position_pct >= {QUAL}
        GROUP BY 1 ORDER BY qv DESC, account_id LIMIT {int(limit)}
        """,
        [int(target_id)],
    ).fetchall()
    return [{"account_id": int(a), "qv": int(q)} for a, q in rows]


def weekly_qv(con: duckdb.DuckDBPyConnection, target_id: int) -> list[dict]:
    """Qualified views per tumbling 7-day window — the visual behind `burst7`."""
    rows = con.execute(
        f"""
        SELECT CAST(ts / 604800 AS INTEGER) AS wk, count(*) AS qv
        FROM events
        WHERE creator_id = ? AND action IN ('view','complete') AND reported_position_pct >= {QUAL}
        GROUP BY 1 ORDER BY 1
        """,
        [int(target_id)],
    ).fetchall()
    if not rows:
        return []
    w0 = rows[0][0]
    return [{"week": int(w - w0), "qv": int(q)} for w, q in rows]


def core_stats(con: duckdb.DuckDBPyConnection, core_ids: tuple[int, ...], target_id: int) -> list[dict]:
    """Per-core-account evidence: qualified views on this creator, total, single-target share,
    number of creators viewed, and the account's infrastructure footprint (distinct subnets/ASNs,
    modal ASN).

    High share and few other creators marks a dedicated booster; broad viewing at low share is the
    diffuse profile of a benign viewer. The infra columns corroborate: a ring tends to share an ASN
    or subnet across its core, while a genuine superfan sits on one or two residential subnets. The
    ranker never reads infra; it is evidence for the human only.
    """
    if not core_ids:
        return []
    ids = ",".join(str(int(a)) for a in core_ids)
    rows = con.execute(
        f"""
        WITH pa AS (  -- qualified views per (account, creator)
            SELECT account_id AS acct, creator_id AS tgt, count(*) AS qv
            FROM events
            WHERE action IN ('view','complete') AND reported_position_pct >= {QUAL}
              AND account_id IN ({ids})
            GROUP BY 1, 2
        ),
        qv AS (
            SELECT acct,
                   coalesce(sum(qv) FILTER (WHERE tgt = ?), 0) AS qv_on_target,
                   sum(qv)              AS total_qv,
                   count(DISTINCT tgt)  AS n_targets
            FROM pa GROUP BY acct
        ),
        infra AS (  -- infra footprint over ALL of the account's events (not just qualified)
            SELECT account_id AS acct,
                   count(DISTINCT subnet) AS n_subnet,
                   count(DISTINCT asn)    AS n_asn,
                   mode(asn)              AS top_asn
            FROM events WHERE account_id IN ({ids}) GROUP BY 1
        )
        SELECT q.acct, q.qv_on_target, q.total_qv, q.n_targets, i.n_subnet, i.n_asn, i.top_asn
        FROM qv q JOIN infra i USING (acct)
        ORDER BY q.qv_on_target DESC, q.acct
        """,
        [int(target_id)],
    ).fetchall()
    return [
        {
            "account_id": int(acct), "qv_on_target": int(qvt), "total_qv": int(tot),
            "n_targets": int(nt), "share": (qvt / tot) if tot else 0.0,
            "n_subnet": int(n_subnet), "n_asn": int(n_asn),
            "top_asn": (int(top_asn) if top_asn is not None else None),
        }
        for acct, qvt, tot, nt, n_subnet, n_asn, top_asn in rows
    ]


# --- ground truth (eval side: grading only, never scoring) -----------------------------------


@dataclass(frozen=True)
class TargetTruth:
    """What the hidden labels say about one creator."""

    ring_ids: tuple[int, ...]   # rings paid to boost this creator (empty = not a ring target)
    fraud_qv_share: float       # share of its qualified views that came from ring accounts

    @property
    def is_ring_target(self) -> bool:
        return bool(self.ring_ids)


def _paths(world: World) -> tuple[str, str]:
    return (os.path.join(world.world_dir, "events.parquet"),
            os.path.join(world.world_dir, "ground_truth.parquet"))


def target_truth(world: World) -> dict[int, TargetTruth]:
    """Per-creator truth for the whole world: which rings boost it, and how much of its qualified
    viewing is ring traffic."""
    events_path, gt_path = _paths(world)
    targets = ring_target_map(events_path, gt_path)
    ev = pl.read_parquet(events_path).filter(
        pl.col("creator_id").is_not_null()
        & pl.col("action").is_in(["view", "complete"])
        & (pl.col("reported_position_pct") >= QUAL)
    )
    fraud = world.gt.select("account_id", "is_fraud")
    share = (
        ev.join(fraud, on="account_id", how="left")
        .group_by("creator_id")
        .agg(pl.col("is_fraud").fill_null(False).mean().alias("s"))
    )
    out = {int(c): TargetTruth(ring_ids=(), fraud_qv_share=float(s)) for c, s in share.iter_rows()}
    for creator, rings in targets.items():
        prev = out.get(creator, TargetTruth((), 0.0))
        out[creator] = TargetTruth(ring_ids=tuple(sorted(rings)), fraud_qv_share=prev.fraud_qv_share)
    return out


def account_ring(world: World) -> dict[int, int | None]:
    """account_id -> ring_id for ring members (None for everyone else)."""
    fr = world.gt.filter(pl.col("is_fraud"))
    return {int(a): (int(r) if r is not None else None) for a, r in fr.select("account_id", "ring_id").iter_rows()}


# --- target-level label store ----------------------------------------------------------------


@dataclass(frozen=True)
class TriageScore:
    n_decided: int
    tp: int
    fp: int
    tn: int
    fn: int

    @property
    def precision(self) -> float:
        return self.tp / (self.tp + self.fp) if self.tp + self.fp else 0.0

    @property
    def recall(self) -> float:
        return self.tp / (self.tp + self.fn) if self.tp + self.fn else 0.0

    @property
    def accuracy(self) -> float:
        return (self.tp + self.tn) / self.n_decided if self.n_decided else 0.0


class SuspectLabelStore:
    """Human creator-level verdicts (ring/legit/unsure/escalate) in a JSON file in the world dir.

    Mirrors `viewer.labels.LabelStore` but keyed by creator. The `suspicion` and `rank` captured at
    label time let a later pass compute precision-at-K over the human's calls.

    Pass `backing` (a dict) for an in-memory store that never touches disk (the hosted demo).
    """

    def __init__(self, path: str, backing: dict[int, dict] | None = None) -> None:
        self.path = path
        self._memory = backing is not None
        self._labels: dict[int, dict] = backing if backing is not None else {}
        if not self._memory and os.path.exists(path):
            with open(path) as f:
                self._labels = {int(k): v for k, v in json.load(f).items()}

    def _flush(self) -> None:
        if self._memory:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump({str(k): v for k, v in self._labels.items()}, f, indent=2, sort_keys=True)

    def set(self, target_id: int, verdict: SuspectVerdict, *, note: str = "",
            suspicion: float | None = None, rank: int | None = None) -> None:
        if verdict not in SUSPECT_VERDICTS:
            raise ValueError(f"unknown verdict {verdict!r}; expected one of {SUSPECT_VERDICTS}")
        self._labels[int(target_id)] = {
            "verdict": verdict, "note": note, "suspicion": suspicion, "rank": rank,
            "updated": datetime.now(UTC).isoformat(timespec="seconds"),
        }
        self._flush()

    def clear(self, target_id: int) -> None:
        if int(target_id) in self._labels:
            del self._labels[int(target_id)]
            self._flush()

    def get(self, target_id: int) -> dict | None:
        return self._labels.get(int(target_id))

    def counts(self) -> dict[str, int]:
        out = {v: 0 for v in SUSPECT_VERDICTS}
        for rec in self._labels.values():
            out[rec["verdict"]] = out.get(rec["verdict"], 0) + 1
        return out

    def score(self, truth: dict[int, TargetTruth]) -> TriageScore:
        """Grade ring/legit calls against which creators the rings actually boosted."""
        tp = fp = tn = fn = 0
        for tgt, rec in self._labels.items():
            is_ring = truth.get(tgt, TargetTruth((), 0.0)).is_ring_target
            if rec["verdict"] == "ring":
                tp, fp = tp + is_ring, fp + (not is_ring)
            elif rec["verdict"] == "legit":
                fn, tn = fn + is_ring, tn + (not is_ring)
        return TriageScore(n_decided=tp + fp + tn + fn, tp=tp, fp=fp, tn=tn, fn=fn)


def store_for_world(world_dir: str) -> SuspectLabelStore:
    return SuspectLabelStore(os.path.join(world_dir, "suspect_labels.json"))
