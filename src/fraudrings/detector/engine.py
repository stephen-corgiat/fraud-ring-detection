"""The two-stage 'link-then-score' engine.

Stage 1 — seed-anchored, rarity-gated expansion over `assoc` (no global graph).
Stage 2 — seed-relative evidence score: linkage-IDF + behavioral homogeneity +
          co-target overlap + action mix. Interpretable and label-free.
"""

from __future__ import annotations

from dataclasses import dataclass

import duckdb
import numpy as np

from fraudrings.config import EngineConfig


@dataclass
class Candidate:
    account_id: int
    score: float
    linkage: float  # summed IDF of shared rare identifiers
    behavioral: float  # similarity to ring centroid in standardized feature space
    cotarget: float  # co-target evidence: Jaccard of viewed creators with the ring (baseline), or the
    # candidate's qualified-view share on the seed target (`cotarget_concentration`)
    actionmix: float  # similarity to ring centroid over the action-funnel / account-age sub-features


def _clause(ids: list[int]) -> str:
    return "(" + ",".join(str(int(i)) for i in ids) + ")"


def _standardizer(con: duckdb.DuckDBPyConnection, cols: tuple[str, ...]) -> tuple[np.ndarray, np.ndarray]:
    sel = ", ".join(f"avg({c}) AS {c}_m, coalesce(stddev_samp({c}), 0) AS {c}_s" for c in cols)
    row = con.execute(f"SELECT {sel} FROM account_features").fetchall()[0]
    means = np.array([float(row[2 * i] or 0.0) for i in range(len(cols))], dtype=float)
    stds = np.array([float(row[2 * i + 1] or 0.0) for i in range(len(cols))], dtype=float)
    stds[stds == 0.0] = 1.0
    return means, stds


def _feature_vectors(
    con: duckdb.DuckDBPyConnection, ids: list[int], cols: tuple[str, ...],
    means: np.ndarray, stds: np.ndarray,
) -> dict[int, np.ndarray]:
    if not ids:
        return {}
    rows = con.execute(
        f"SELECT account_id, {', '.join(cols)} FROM account_features WHERE account_id IN {_clause(ids)}"
    ).fetchall()
    out: dict[int, np.ndarray] = {}
    for r in rows:
        vec = np.array([float(x or 0.0) for x in r[1:]], dtype=float)
        out[int(r[0])] = (vec - means) / stds
    return out


def _centroid(vecs: dict[int, np.ndarray], dim: int, *, robust: bool) -> np.ndarray:
    """The seed cohort's reference point in standardized feature space.

    `robust=False` (baseline): the per-feature MEAN — the plain centroid. `robust=True`: the per-feature
    MEDIAN, which ignores an atypical or mislabelled seed instead of being dragged toward it (one
    outlier among K seeds shifts the mean by 1/K per feature but leaves the median untouched). An empty
    seed set has no reference, so both collapse to the origin (every candidate is then equidistant).
    """
    if not vecs:
        return np.zeros(dim)
    matrix = np.stack(list(vecs.values()))
    return np.median(matrix, axis=0) if robust else np.mean(matrix, axis=0)


def _creator_sets(con: duckdb.DuckDBPyConnection, ids: list[int]) -> dict[int, set[str]]:
    if not ids:
        return {}
    rows = con.execute(
        f"""SELECT account_id, id_value FROM assoc
            WHERE id_type='creator' AND cnt >= 2 AND account_id IN {_clause(ids)}"""
    ).fetchall()
    out: dict[int, set[str]] = {}
    for r in rows:
        out.setdefault(int(r[0]), set()).add(str(r[1]))
    return out


def seed_cohesion(con: duckdb.DuckDBPyConnection, seeds: list[int], engine: EngineConfig) -> float:
    """Fraction of seeds that interlink with another seed via a rarity-gated identifier.

    This is a *retrieval-coverage* check, not a confidence score: a low value means the seeds
    barely share infrastructure (the sophisticated regime), so infra linkage has likely failed
    to retrieve the rest of the ring — the cue to widen with co-target. ~1.0 at naive (everyone
    on the same rare datacenter IPs), low at sophisticated (residential proxies, 1-2 IPs each).
    """
    if len(seeds) < 2:
        return 1.0
    seed_clause = _clause(seeds)
    types_clause = "(" + ",".join(f"'{t}'" for t in engine.link_id_types) + ")"
    row = con.execute(
        f"""
        WITH seed_ids AS (
            SELECT a.account_id, a.id_type, a.id_value
            FROM assoc a JOIN identifier_stats s USING (id_type, id_value)
            WHERE a.account_id IN {seed_clause}
              AND a.id_type IN {types_clause}
              AND s.distinct_accounts BETWEEN {engine.expand_min_accounts} AND {engine.expand_max_accounts}
        )
        SELECT count(DISTINCT s1.account_id)
        FROM seed_ids s1 JOIN seed_ids s2
          ON s1.id_type = s2.id_type AND s1.id_value = s2.id_value AND s1.account_id <> s2.account_id
        """
    ).fetchone()
    linked = int(row[0]) if row and row[0] is not None else 0
    return linked / len(seeds)


def _seed_targets(
    con: duckdb.DuckDBPyConnection, seeds: list[int], engine: EngineConfig
) -> list[tuple[int, int, int]]:
    """The seed cohort's target creator(s): `(creator_id, first_ts, last_ts)`, up to `cotarget_n_targets`.

    Rank creators by cross-seed agreement (#seeds that qualified-viewed it), then by seed qualified-view
    *volume*. Cover views are individually random → low agreement; the target is the high-agreement,
    high-volume creator. NB we do NOT weight by global rarity: the ring *inflates* its target's account
    count, so the true target is not rare — rarity would perversely demote it and promote rare cover
    views. Concentration, not rarity, is the discriminator (the ring pours qualified views into the
    target for payouts). Label-free and event-only. Shared by the co-target candidate-gen (which
    expands ON these targets) and the co-target concentration term (which scores devotion TO them).
    """
    seed_clause = _clause(seeds)
    thr = engine.qualified_view_threshold
    rows = con.execute(
        f"""
        SELECT creator_id, min(ts) AS t0, max(ts) AS t1
        FROM events
        WHERE account_id IN {seed_clause} AND creator_id IS NOT NULL
          AND action IN ('view','complete') AND reported_position_pct >= {thr}
        GROUP BY creator_id
        ORDER BY count(DISTINCT account_id) DESC, count(*) DESC, creator_id ASC
        LIMIT {engine.cotarget_n_targets}
        """
    ).fetchall()
    return [(int(r[0]), int(r[1]), int(r[2])) for r in rows]


def _seed_target_concentration(
    con: duckdb.DuckDBPyConnection,
    cand_ids: list[int],
    target_ids: list[int],
    engine: EngineConfig,
) -> dict[int, float]:
    """Each candidate's DEVOTION to the seed target(s): its qualified views ON the target(s) ÷ its
    total qualified views, in [0, 1] (the single-target-share / booster signature).

    A core member pours its views into the target → share near 1; a broad-taste legit viewer who
    merely *also* views the target has a low share. This is the concentration analogue of the baseline
    Jaccard repertoire-overlap term: it rewards concentration, not catalog breadth. Event-only,
    label-free. Candidates absent from the result (no qualified views at all) score 0.
    """
    if not cand_ids or not target_ids:
        return {}
    thr = engine.qualified_view_threshold
    rows = con.execute(
        f"""
        SELECT account_id,
               count(*) FILTER (WHERE creator_id IN {_clause(target_ids)}) AS on_target,
               count(*) AS total
        FROM events
        WHERE account_id IN {_clause(cand_ids)} AND creator_id IS NOT NULL
          AND action IN ('view','complete') AND reported_position_pct >= {thr}
        GROUP BY account_id
        """
    ).fetchall()
    return {int(r[0]): (float(r[1]) / float(r[2]) if r[2] else 0.0) for r in rows}


def _cotarget_candidates(
    con: duckdb.DuckDBPyConnection, seeds: list[int], engine: EngineConfig
) -> set[int]:
    """Retrieve accounts that share the seeds' rare target creator(s), in the campaign window.

    Identify the target(s) via `_seed_targets`, then expand on them, restricted to the seed-derived
    campaign window (the rarity trough) so a later legit bandwagon on a since-popular creator is
    excluded by construction. Both steps are label-free and event-only.
    """
    seed_clause = _clause(seeds)
    thr = engine.qualified_view_threshold
    targets = _seed_targets(con, seeds, engine)
    if not targets:
        return set()
    target_ids = [t[0] for t in targets]
    t0 = min(t[1] for t in targets)
    t1 = max(t[2] for t in targets)
    pad = int((t1 - t0) * engine.cotarget_window_pad_frac)
    rows = con.execute(
        f"""
        SELECT account_id
        FROM events
        WHERE creator_id IN {_clause(target_ids)}
          AND action IN ('view','complete') AND reported_position_pct >= {thr}
          AND ts BETWEEN {t0 - pad} AND {t1 + pad}
          AND account_id NOT IN {seed_clause}
        GROUP BY account_id
        HAVING count(*) >= {engine.cotarget_min_qviews}
        """
    ).fetchall()
    return {int(r[0]) for r in rows}


def run_investigation(
    con: duckdb.DuckDBPyConnection, seeds: list[int], engine: EngineConfig
) -> list[Candidate]:
    """Given a seed of known/suspected accounts, return related candidates, ranked."""
    if not seeds:
        return []
    seed_clause = _clause(seeds)
    types_clause = "(" + ",".join(f"'{t}'" for t in engine.link_id_types) + ")"

    # --- Stage 1 + linkage score (one query) -------------------------------
    rows = con.execute(
        f"""
        WITH seed_ids AS (
            SELECT DISTINCT a.id_type, a.id_value, s.rarity
            FROM assoc a JOIN identifier_stats s USING (id_type, id_value)
            WHERE a.account_id IN {seed_clause}
              AND a.id_type IN {types_clause}
              AND s.distinct_accounts BETWEEN {engine.expand_min_accounts} AND {engine.expand_max_accounts}
        )
        SELECT a.account_id AS account_id, sum(si.rarity) AS linkage
        FROM assoc a JOIN seed_ids si ON a.id_type = si.id_type AND a.id_value = si.id_value
        WHERE a.account_id NOT IN {seed_clause}
        GROUP BY a.account_id
        """
    ).fetchall()
    linkage: dict[int, float] = {int(r[0]): float(r[1]) for r in rows}

    # --- Stage 1b: co-target widening ------------------------------------
    # Fires either always (`cotarget_always`, the supervised rule's regime) or adaptively when infra
    # linkage is too thin to have retrieved the ring. Co-target-only candidates carry no linkage
    # evidence (0.0); Stage-2 must earn their rank on behavioral + co-target.
    if engine.cotarget_candidate_gen and (
        engine.cotarget_always
        or seed_cohesion(con, seeds, engine) < engine.cotarget_seed_cohesion_trigger
    ):
        for aid in _cotarget_candidates(con, seeds, engine):
            linkage.setdefault(aid, 0.0)

    if not linkage:
        return []
    cand_ids = list(linkage.keys())

    # --- Stage 2 inputs ----------------------------------------------------
    means, stds = _standardizer(con, engine.behavioral_features)
    seed_vecs = _feature_vectors(con, seeds, engine.behavioral_features, means, stds)
    cand_vecs = _feature_vectors(con, cand_ids, engine.behavioral_features, means, stds)
    centroid = _centroid(seed_vecs, len(engine.behavioral_features), robust=engine.robust_centroid)

    # Action-mix axis: a SECOND seed-centroid proximity, over the funnel-shape /
    # account-age sub-features, computed exactly like the behavioral term but kept as its own axis so
    # it diagnoses separately. A naive/moderate ring's truncated funnel + fresh
    # accounts sit far from a legit candidate's, lifting true members; the term fades as a
    # sophisticated ring fakes the funnel and ages its accounts (the candidate funnels converge).
    am_means, am_stds = _standardizer(con, engine.actionmix_features)
    am_seed_vecs = _feature_vectors(con, seeds, engine.actionmix_features, am_means, am_stds)
    am_cand_vecs = _feature_vectors(con, cand_ids, engine.actionmix_features, am_means, am_stds)
    am_centroid = _centroid(am_seed_vecs, len(engine.actionmix_features), robust=engine.robust_centroid)

    # Co-target evidence. Baseline (default): Jaccard of viewed-creator REPERTOIRES. Alternative
    # (cotarget_concentration): each candidate's qualified-view SHARE on the seed target(s) — devotion
    # to the target, not catalog breadth. Compute only the side the flag selects (the OFF path is the
    # baseline code verbatim, so Stage 2 is byte-identical when the flag is off).
    seed_creators: set[str] = set()
    cand_creators: dict[int, set[str]] = {}
    concentration: dict[int, float] = {}
    if engine.cotarget_concentration:
        target_ids = [t[0] for t in _seed_targets(con, seeds, engine)]
        concentration = _seed_target_concentration(con, cand_ids, target_ids, engine)
    else:
        for s in _creator_sets(con, seeds).values():
            seed_creators |= s
        cand_creators = _creator_sets(con, cand_ids)

    # normalize linkage across the candidate set -> [0,1]
    lk = np.array([linkage[c] for c in cand_ids], dtype=float)
    lo, hi = float(lk.min()), float(lk.max())
    lk_norm = np.ones_like(lk) if hi == lo else (lk - lo) / (hi - lo)

    out: list[Candidate] = []
    for i, c in enumerate(cand_ids):
        behav = 1.0 / (1.0 + float(np.linalg.norm(cand_vecs[c] - centroid))) if c in cand_vecs else 0.0
        amix = (
            1.0 / (1.0 + float(np.linalg.norm(am_cand_vecs[c] - am_centroid)))
            if c in am_cand_vecs else 0.0
        )
        if engine.cotarget_concentration:
            cot = concentration.get(c, 0.0)
        else:
            ca = cand_creators.get(c, set())
            union = seed_creators | ca
            cot = len(seed_creators & ca) / len(union) if union else 0.0
        score = (
            engine.w_linkage * float(lk_norm[i])
            + engine.w_behavioral * behav
            + engine.w_cotarget * cot
            + engine.w_actionmix * amix
        )
        out.append(Candidate(account_id=c, score=score, linkage=float(lk_norm[i]),
                             behavioral=behav, cotarget=cot, actionmix=amix))

    out.sort(key=lambda x: x.score, reverse=True)
    return out
