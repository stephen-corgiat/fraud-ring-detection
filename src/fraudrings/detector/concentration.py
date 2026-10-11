"""Seedless concentration ranker: shortlist suspect creators from the event stream alone.

A boost ring shows up as a **target** creator whose qualified views *concentrate* — on a few
accounts (low `n_eff`), on many dedicated single-target accounts (high booster-count), or in a
short window (high `burst7`). The ranker scores every target that clears an impact floor on two
first-class keys: `suspicion` (max percentile across those three entry axes — pure concentration)
and `rank_score` (suspicion × capped-qv — concentration weighted by payout impact). It emits a
ranked shortlist of **suspects**.

The default order is `suspicion` (`cfg.sort_by`): a human
triaging the shortlist wants "how coordinated does this look", so the concentrated few-dominant core
heads the list and a merely-popular high-volume target can't float above it on volume alone (impact
only breaks ties within an equal-suspicion band). `sort_by="rank_score"` gives the impact-first view.

It reads only `events` — never ground truth, never `cfg`-of-the-world (anti-tautology). The metrics
are one DuckDB pass; ranking/tagging are thin Python.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Sequence
from dataclasses import dataclass

import duckdb

from fraudrings.config import ConcentrationConfig, EngineConfig

# The seedless -> seeded seam: reuse the engine's rarity-gated cohesion check as the expansion gate,
# and its seeded loop unchanged, so the engine stays one regression-guarded implementation.
from fraudrings.detector.engine import Candidate, run_investigation, seed_cohesion


@dataclass(frozen=True)
class SuspectTarget:
    """One ranked suspect: a target and the evidence behind its rank. Never a verdict."""

    target_id: int
    # --- ranking (suspicion is impact-independent; rank_score points it at payout harm) ---------
    rank_score: float             # suspicion × log10(capped_qv) — the shortlist order
    suspicion: float              # max percentile across the three entry axes (in the pool)
    pct_neff: float               # this target's percentile on the concentration axis (1/n_eff)
    pct_booster: float            # …on the booster-count axis
    pct_burst7: float             # …on the temporal axis (burst7) — which axis fired drives review
    # --- impact ---------------------------------------------------------------------------------
    capped_qv: int    # payout-relevant impact: Σ over (account, day) of min(daily qv, CAP)
    tot_qv: int       # raw qualified-view volume
    # --- entry-axis metrics + contributor shape -------------------------------------------------
    n_eff: float      # effective # contributors (inverse-HHI of qv across accounts)
    booster_count: int            # # accounts dedicated to this target (the booster entry axis)
    burst7: float     # largest share of qv in any tumbling 7-day window (temporal entry axis)
    top1_share: float             # qv share of the single biggest contributor
    top3_share: float             # qv share of the top three contributors
    # --- the dedicated subset + its observed shape ----------------------------------------------
    core: tuple[int, ...]         # those dedicated contributor account ids — what triage surfaces
    tags: tuple[str, ...]         # observed-topology tags — a description, not a verdict


def _percentile(sorted_vals: list[float], x: float) -> float:
    """Midpoint-rank percentile of `x` within `sorted_vals` (ties get the average rank)."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    lo = bisect.bisect_left(sorted_vals, x)
    hi = bisect.bisect_right(sorted_vals, x)
    return (lo + (hi - lo) / 2.0) / n


def topology_tags(
    *,
    top1_share: float,
    top3_share: float,
    booster_count: int,
    burst7: float,
    cfg: ConcentrationConfig,
) -> tuple[str, ...]:
    """Name a target's observed contributor shape. A pure function of the metrics.

    `single-source`/`few-source` are mutually exclusive (one actor vs a handful dominating);
    `multi-source` (many dedicated contributors — the coordination signal, independent of `n_eff`)
    and `campaign` (temporal) are independent modifiers. An
    empty tuple is honest — a mid-concentration suspect simply carries no strong topology signal.
    """
    tags: list[str] = []
    if top1_share >= cfg.tag_single_source_share:
        tags.append("single-source")
    elif top3_share >= cfg.tag_few_source_share:
        tags.append("few-source")
    if booster_count >= cfg.tag_multi_source_boosters:
        tags.append("multi-source")
    if burst7 >= cfg.tag_campaign_burst7:
        tags.append("campaign")
    return tuple(tags)


def generate_suspects(
    con: duckdb.DuckDBPyConnection, cfg: ConcentrationConfig | None = None
) -> list[SuspectTarget]:
    """Rank every impact-floored target as a suspect. Order = `cfg.sort_by` (default `suspicion`,
    pure concentration; `rank_score` for the impact-first view). Reads only `events`."""
    cfg = cfg or ConcentrationConfig()
    rows = con.execute(
        f"""
        WITH qvd AS (  -- qualified views per (target, account, day) — the cap bucket
            SELECT creator_id AS tgt, account_id AS acct,
                   CAST(ts / 86400 AS INTEGER) AS dy, count(*) AS qvd
            FROM events
            WHERE action IN ('view', 'complete')
              AND reported_position_pct >= {cfg.qualified_view_threshold}
              AND creator_id IS NOT NULL
            GROUP BY 1, 2, 3
        ),
        pa AS (  -- per (target, account): raw qv and capped qv
            SELECT tgt, acct,
                   sum(qvd)                  AS qv,
                   sum(LEAST(qvd, {cfg.cap})) AS cqv
            FROM qvd
            GROUP BY 1, 2
        ),
        atot AS (  -- per-account total qv across ALL targets (for single-target share)
            SELECT acct, sum(qv) AS acct_tot FROM pa GROUP BY 1
        ),
        pab AS (  -- flag each (target, account) as a booster + rank within the target by qv
            SELECT pa.tgt, pa.acct, pa.qv, pa.cqv,
                   (pa.qv >= {cfg.booster_min_qv}
                    AND pa.qv * 1.0 / atot.acct_tot >= {cfg.booster_min_share}) AS is_booster,
                   row_number() OVER (PARTITION BY pa.tgt ORDER BY pa.qv DESC) AS rk
            FROM pa JOIN atot USING (acct)
        ),
        wk AS (  -- qv per (target, tumbling 7-day window) for burst7
            SELECT tgt, CAST(dy / 7 AS INTEGER) AS wk, sum(qvd) AS wq
            FROM qvd GROUP BY 1, 2
        ),
        burst AS (
            SELECT tgt, max(wq) * 1.0 / sum(wq) AS burst7 FROM wk GROUP BY 1
        )
        SELECT p.tgt,
               sum(p.qv)        AS tot_qv,
               sum(p.cqv)       AS capped_qv,
               sum(p.qv * p.qv) AS sumsq,
               count(*) FILTER (WHERE p.is_booster)       AS booster_count,
               list(p.acct)     FILTER (WHERE p.is_booster) AS core,
               any_value(b.burst7) AS burst7,
               max(p.qv)                              AS top1_qv,
               sum(p.qv) FILTER (WHERE p.rk <= 3)     AS top3_qv
        FROM pab p JOIN burst b USING (tgt)
        GROUP BY p.tgt
        HAVING sum(p.qv) >= {cfg.impact_floor}
        """
    ).fetchall()

    # First pass: scalar metrics per impact-floored target.
    recs = []
    for tgt, tot_qv, capped_qv, sumsq, booster_count, core, burst7, top1_qv, top3_qv in rows:
        tot = float(tot_qv)
        recs.append(
            dict(
                target_id=int(tgt), tot_qv=int(tot_qv), capped_qv=int(capped_qv),
                n_eff=tot * tot / float(sumsq), booster_count=int(booster_count),
                core=tuple(sorted(int(a) for a in (core or []))), burst7=float(burst7),
                top1_share=float(top1_qv) / tot, top3_share=float(top3_qv) / tot,
            )
        )

    # The three entry axes, scored as percentiles within the impact-floored pool (so a target is
    # "extreme on any axis" relative to its peers). Higher 1/n_eff, booster-count, burst7 == more
    # suspect; suspicion is the MAX across them, so each axis catches an archetype the others miss.
    inv_neff = sorted(1.0 / r["n_eff"] for r in recs)
    boosters = sorted(float(r["booster_count"]) for r in recs)
    bursts = sorted(r["burst7"] for r in recs)

    suspects = []
    for r in recs:
        pct_neff = _percentile(inv_neff, 1.0 / r["n_eff"])
        pct_booster = _percentile(boosters, float(r["booster_count"]))
        pct_burst7 = _percentile(bursts, r["burst7"])
        suspicion = max(pct_neff, pct_booster, pct_burst7)
        suspects.append(
            SuspectTarget(
                rank_score=suspicion * math.log10(max(r["capped_qv"], 1)),
                suspicion=suspicion, pct_neff=pct_neff, pct_booster=pct_booster, pct_burst7=pct_burst7,
                tags=topology_tags(
                    top1_share=r["top1_share"], top3_share=r["top3_share"],
                    booster_count=r["booster_count"], burst7=r["burst7"], cfg=cfg,
                ),
                **{k: r[k] for k in (
                    "target_id", "tot_qv", "capped_qv", "n_eff", "booster_count",
                    "burst7", "top1_share", "top3_share", "core",
                )},
            )
        )

    # Default order is pure concentration (`suspicion`); `rank_score` gives the impact-first view.
    # Ties on the primary key break by capped-qv (more payout harm first) then target_id (stable).
    primary = (lambda s: s.suspicion) if cfg.sort_by == "suspicion" else (lambda s: s.rank_score)
    suspects.sort(key=lambda s: (primary(s), s.capped_qv, -s.target_id), reverse=True)
    return suspects


def maybe_expand(
    con: duckdb.DuckDBPyConnection,
    core: Sequence[int],
    engine: EngineConfig,
    cfg: ConcentrationConfig | None = None,
) -> list[Candidate] | None:
    """Optionally hand a suspect's core to the seeded engine, gated on infra cohesion.

    On naive / device-farm cores the members share rare infrastructure, so the engine's rarity-gated
    expansion can recover the rest of the ring — we hand the core to `run_investigation`. On a
    sophisticated (residential, infra-diverse) core that linkage is dead, so expansion would only fan
    out to thousands of false candidates; we decline and the **core itself is the output**. Returns
    the ranked candidates when it expands, or `None` when it does not (do not auto-expand).
    """
    cfg = cfg or ConcentrationConfig()
    seeds = [int(a) for a in core]
    if len(seeds) < 2:  # no member-to-member cohesion to assess
        return None
    if seed_cohesion(con, seeds, engine) < cfg.expand_cohesion_trigger:
        return None
    return run_investigation(con, seeds, engine)
