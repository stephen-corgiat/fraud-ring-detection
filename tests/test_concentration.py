"""The seedless concentration ranker, label-free.

The ranker emits a ranked shortlist of **suspect target creators** from the event stream alone —
never ground truth, never `cfg`. A target ranks high if it is extreme on *any* of three entry axes
(`n_eff`, booster-count, `burst7`). See `GLOSSARY.md` for the vocabulary.

Most tests pin the module on tiny hand-built fixtures; the last ones grade it on simulated worlds
against the hidden ring targets (`eval/concentration.py`).
"""

from __future__ import annotations

from typing import cast

import duckdb
import polars as pl
import pytest

from fraudrings.config import EngineConfig, preset
from fraudrings.detector.concentration import (
    ConcentrationConfig,
    SuspectTarget,
    generate_suspects,
    maybe_expand,
    topology_tags,
)
from fraudrings.detector.engine import Candidate
from fraudrings.detector.ingest import build_db
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel

DAY = 86_400


def _events_db(rows: list[tuple[int, int, int, str, float]]) -> duckdb.DuckDBPyConnection:
    """An events-ONLY DuckDB (no accounts, no ground_truth) — the generator's whole world.

    rows: (ts, account_id, creator_id, action, reported_position_pct).
    """
    con = duckdb.connect(":memory:")
    con.execute(
        "CREATE TABLE events "
        "(ts BIGINT, account_id BIGINT, creator_id BIGINT, action VARCHAR, reported_position_pct DOUBLE)"
    )
    con.executemany("INSERT INTO events VALUES (?,?,?,?,?)", rows)
    return con


def _qv(account: int, target: int, day: int, n: int, *, pct: float = 0.95) -> list[tuple]:
    """`n` qualified views by `account` on `target`, all within `day` (distinct seconds)."""
    return [(day * DAY + i, account, target, "view", pct) for i in range(n)]


def _by_id(suspects: list[SuspectTarget]) -> dict[int, SuspectTarget]:
    return {s.target_id: s for s in suspects}


def test_surfaces_few_dominant_target_with_correct_concentration() -> None:
    """Tracer bullet: a target dominated by one account is surfaced with the right `n_eff`/`tot_qv`."""
    rows = _qv(1, 100, 0, 30) + _qv(2, 100, 0, 5) + _qv(3, 100, 0, 5)
    con = _events_db(rows)

    suspects = generate_suspects(con, ConcentrationConfig(impact_floor=10, booster_min_qv=5))

    s = _by_id(suspects)[100]
    assert s.tot_qv == 40
    # inverse-HHI: tot_qv^2 / Σ qv_i^2  = 1600 / (900+25+25) = 1.684…
    assert s.n_eff == pytest.approx(1600 / 950, rel=1e-9)


def test_capped_qv_caps_each_account_day() -> None:
    """Impact = Σ over (account, day) of min(daily qv, CAP). A lone cap-violator's wasted
    over-cap views do not count toward payout harm; a per-day cap resets each day."""
    rows = (
        _qv(10, 200, 0, 30)   # 30 in one day → capped to 10
        + _qv(10, 200, 1, 30)  # another 30 next day → another 10 (cap is per-DAY)
        + _qv(11, 200, 0, 8)   # 8 < cap → counts in full
    )
    con = _events_db(rows)

    s = _by_id(generate_suspects(con, ConcentrationConfig(impact_floor=10, cap=10)))[200]

    assert s.tot_qv == 68            # raw: 30 + 30 + 8
    assert s.capped_qv == 28         # min(30,10) + min(30,10) + min(8,10) = 10 + 10 + 8


def test_booster_count_and_core_are_single_target_dedicated_accounts() -> None:
    """A booster is an account dedicated to ONE target (single-target share ≥ thr, at volume). The
    core is exactly that dedicated set — the booster-count entry axis the `n_eff` axis can miss."""
    rows = (
        _qv(41, 400, 0, 8)               # all 8 on target 400 → share 1.0, dedicated
        + _qv(42, 400, 0, 8) + _qv(42, 401, 0, 8)  # split 400/401 → share 0.5, NOT dedicated
        + _qv(43, 400, 0, 6)             # all 6 on target 400 → share 1.0, dedicated
    )
    con = _events_db(rows)

    cfg = ConcentrationConfig(impact_floor=10, booster_min_qv=5, booster_min_share=0.80)
    s = _by_id(generate_suspects(con, cfg))[400]

    assert s.booster_count == 2
    assert set(s.core) == {41, 43}


def test_burst7_is_densest_tumbling_week_share() -> None:
    """burst7 = a target's largest share of qualified views in any single tumbling 7-day window —
    the temporal entry axis that catches a campaign spread across many accounts (high `n_eff`)."""
    rows = _qv(31, 300, 0, 10) + _qv(31, 300, 7, 30)  # week 0: 10, week 1: 30
    con = _events_db(rows)

    s = _by_id(generate_suspects(con, ConcentrationConfig(impact_floor=10)))[300]

    assert s.burst7 == pytest.approx(30 / 40, rel=1e-9)


def test_contributor_shares() -> None:
    """top1/top3 share describe the contributor shape."""
    rows = (
        _qv(51, 500, 0, 20)    # top1
        + _qv(54, 500, 0, 10)
        + _qv(52, 500, 0, 6)
        + _qv(53, 500, 0, 4)
    )
    con = _events_db(rows)

    s = _by_id(generate_suspects(con, ConcentrationConfig(impact_floor=10)))[500]

    assert s.top1_share == pytest.approx(20 / 40, rel=1e-9)        # 0.5
    assert s.top3_share == pytest.approx((20 + 10 + 6) / 40, rel=1e-9)  # 0.9


def test_topology_tags_name_the_observed_contributor_shape() -> None:
    """Tags are a pure function of the metrics — the observed shape, never a verdict.
    single/few-source are mutually exclusive; multi-source/campaign are independent."""
    cfg = ConcentrationConfig()  # default thresholds: single .60, few .60, multi 3, campaign .50

    def tags(**m):
        base = dict(top1_share=0.0, top3_share=0.0, booster_count=0, burst7=0.0)
        return set(topology_tags(cfg=cfg, **{**base, **m}))

    assert tags(top1_share=0.8, top3_share=0.95) == {"single-source"}    # one actor dominates
    assert tags(top1_share=0.4, top3_share=0.7) == {"few-source"}        # a handful dominate
    assert tags(booster_count=5, top1_share=0.1) == {"multi-source"}     # many dedicated, high n_eff
    assert "campaign" in tags(burst7=0.6)                                # temporal concentration
    assert tags(top1_share=0.4, top3_share=0.7, burst7=0.6) == {"few-source", "campaign"}
    assert tags() == set()  # a mid-concentration suspect carries no topology tag — honest, not "diffuse"


def test_suspect_carries_its_tags() -> None:
    """The generator attaches the tags to each suspect (end-to-end)."""
    rows = _qv(61, 600, 0, 40) + _qv(62, 600, 0, 3) + _qv(63, 600, 0, 3)  # top1_share ~0.87
    con = _events_db(rows)
    s = _by_id(generate_suspects(con, ConcentrationConfig(impact_floor=10)))[600]
    assert "single-source" in s.tags


def _diffuse_rows(target: int, accounts: range, day: int = 0) -> list[tuple]:
    """A diffuse, high-`n_eff` target: many accounts, one qv each — a legit-looking body."""
    return [r for a in accounts for r in _qv(a, target, day, 1)]


def test_rank_only_keeps_diffuse_targets_and_orders_by_concentration() -> None:
    """No hard `n_eff` cut: every impact-floored target appears, *ranked*. A concentrated
    target outranks a diffuse one because its suspicion (axis percentile) is higher — under either
    sort key (the concentrated target has both higher suspicion AND higher rank_score here)."""
    import math

    rows: list[tuple] = []
    # one concentrated target: a dominant booster + a small contributor (low n_eff, booster=1)
    rows += _qv(71, 700, 0, 9) + _qv(71, 700, 1, 9) + _qv(71, 700, 2, 9) + _qv(71, 700, 3, 9) + _qv(71, 700, 4, 9)
    rows += _qv(72, 700, 0, 5)
    # five diffuse targets: 50 accounts, one qv each (high n_eff, no boosters)
    for t in range(710, 715):
        rows += _diffuse_rows(t, range(1000, 1050))
    con = _events_db(rows)

    suspects = generate_suspects(con, ConcentrationConfig(impact_floor=10, cap=10, booster_min_qv=5))

    ids = [s.target_id for s in suspects]
    assert set(ids) == {700, 710, 711, 712, 713, 714}        # rank-only: nothing dropped at an n_eff cut
    assert ids[0] == 700                                       # the concentrated target ranks first

    top = suspects[0]
    assert top.suspicion == pytest.approx(max(top.pct_neff, top.pct_booster, top.pct_burst7))
    assert top.rank_score == pytest.approx(top.suspicion * math.log10(max(top.capped_qv, 1)))

    # rank_score is a first-class alternate view: order by it and the list is sorted by rank_score.
    impact_cfg = ConcentrationConfig(impact_floor=10, cap=10, booster_min_qv=5, sort_by="rank_score")
    by_impact = generate_suspects(con, impact_cfg)
    assert by_impact == sorted(by_impact, key=lambda s: (s.rank_score, s.capped_qv, -s.target_id), reverse=True)


def test_default_order_is_suspicion_not_impact_weighted() -> None:
    """The DEFAULT shortlist order is `suspicion` (pure
    concentration). A highly-concentrated but modest-impact core heads the list, while a diffuse
    high-volume target (much larger payout impact, but not concentrated) sits far below it — the
    OPPOSITE of impact-weighted `rank_score`. This is the core-vs-superfan confound in miniature:
    impact-weighting buries the concentrated core beneath a merely-popular target."""
    cfg = ConcentrationConfig(impact_floor=50, cap=10, booster_min_qv=15)
    rows: list[tuple] = []
    # A concentrated few-dominant core (target 900): two dedicated accounts, low n_eff — but the cap
    # wastes its lone-heavy volume, so its capped impact is tiny.
    rows += _qv(90_001, 900, 0, 40) + _qv(90_002, 900, 0, 20)          # capped_qv = 10 + 10 = 20
    # A diffuse, popular, HIGH-volume target (target 901): 300 accounts × 4 qv — huge capped impact,
    # very high n_eff, zero boosters (nobody dedicated at volume) → concentration axes all cold.
    rows += [r for a in range(91_000, 91_300) for r in _qv(a, 901, 0, 4)]   # capped_qv = 300 × 4 = 1200
    # Twenty moderate background targets so the axis-percentiles are meaningful (a populated pool).
    for t in range(910, 930):
        rows += _qv(t * 100, t, 0, 20) + [r for a in range(t * 100 + 1, t * 100 + 15) for r in _qv(a, t, 0, 5)]
    con = _events_db(rows)

    by_suspicion = generate_suspects(con, cfg)
    by_impact = generate_suspects(con, cfg.model_copy(update={"sort_by": "rank_score"}))

    conc, popular = _by_id(by_suspicion)[900], _by_id(by_suspicion)[901]
    assert conc.suspicion > popular.suspicion         # the core is unmistakably more concentrated…
    assert popular.capped_qv > conc.capped_qv         # …but the popular target does far more raw harm

    # default (suspicion): the concentrated core heads the list despite its smaller impact
    assert by_suspicion[0].target_id == 900
    # impact-weighted (rank_score): the popular high-volume target floats to the top instead
    assert by_impact[0].target_id == 901


def test_capped_impact_ranks_lone_violator_below_a_many_account_ring() -> None:
    """The cap deprioritizes a lone cap-violator (most of its views are wasted over the cap) and
    elevates the economically-real many-account ring of equal RAW volume (each under the cap)."""
    rows: list[tuple] = []
    rows += _qv(81, 800, 0, 300)                              # lone violator: 300 raw → capped 10
    for a in range(8100, 8130):                              # many-account ring: 30 × 10 = 300 raw
        rows += _qv(a, 810, 0, 10)                            #   → capped 300 (each account under cap)
    for t in range(820, 824):                                # background so percentiles are meaningful
        rows += [r for a in range(9000 + t, 9000 + t + 60, 2) for r in _qv(a, t, 0, 2)]
    con = _events_db(rows)

    suspects = generate_suspects(con, ConcentrationConfig(impact_floor=50, cap=10, booster_min_qv=5))
    ids = [s.target_id for s in suspects]
    lone = _by_id(suspects)[800]
    ring = _by_id(suspects)[810]

    assert lone.tot_qv == ring.tot_qv == 300       # equal RAW volume
    assert ring.capped_qv == 300 and lone.capped_qv == 10
    assert ids.index(810) < ids.index(800)         # the many-account ring outranks the lone violator


def test_reads_only_events_never_ground_truth() -> None:
    """Anti-tautology: the ranker's whole world is `events`. Its output must be
    invariant to the presence/content of any ground-truth or features table, and its query must
    name only `events` among known tables."""
    import inspect

    from fraudrings.detector import concentration

    rows = _qv(1, 100, 0, 30) + _qv(2, 100, 0, 5) + _qv(3, 100, 0, 5)
    cfg = ConcentrationConfig(impact_floor=10)

    clean = generate_suspects(_events_db(rows), cfg)

    poisoned = _events_db(rows)
    poisoned.execute("CREATE TABLE ground_truth (account_id BIGINT, is_fraud BOOLEAN, ring_id BIGINT)")
    poisoned.execute("INSERT INTO ground_truth VALUES (1, true, 99), (2, true, 99), (3, false, NULL)")
    assert generate_suspects(poisoned, cfg) == clean  # ground truth changes nothing

    src = inspect.getsource(concentration.generate_suspects)
    assert "FROM events" in src
    for forbidden in ("ground_truth", "is_fraud", "ring_id", "evasion_level", "account_features"):
        assert forbidden not in src


def _core_of_first_ring(tmp_path, level, *, k=4):
    """Generate a world; return (con, the first fraud ring's first k members as a core)."""
    cfg = preset("smoke")
    cfg.sim.evasion_level = cast(EvasionLevel, level)
    cfg.data_dir = str(tmp_path / level)
    events_path, gt_path = generate(cfg.sim, cfg.data_dir)
    con = build_db(events_path, cfg.engine, db_path=str(tmp_path / level / "world.duckdb"))
    gt = pl.read_parquet(gt_path).filter(pl.col("is_fraud"))
    ring_id = gt["ring_id"][0]
    core = gt.filter(pl.col("ring_id") == ring_id)["account_id"].to_list()[:k]
    return con, [int(a) for a in core]


def test_maybe_expand_fires_only_when_core_shares_rare_infra(tmp_path) -> None:
    """Expansion gate: a core whose members interlink on rare infra (naive/device-farm) is handed to the engine;
    a sophisticated core (residential, no shared infra) is NOT auto-expanded — the core IS the output."""
    naive_con, naive_core = _core_of_first_ring(tmp_path, "naive")
    soph_con, soph_core = _core_of_first_ring(tmp_path, "sophisticated")
    eng = EngineConfig()

    expanded = maybe_expand(naive_con, naive_core, eng)
    assert expanded is not None
    assert all(isinstance(c, Candidate) for c in expanded)

    assert maybe_expand(soph_con, soph_core, eng) is None


def test_maybe_expand_needs_at_least_two_core_members(tmp_path) -> None:
    """A single-account core has no member-to-member cohesion to assess — don't auto-expand."""
    con, core = _core_of_first_ring(tmp_path, "naive")
    assert maybe_expand(con, core[:1], EngineConfig()) is None


def test_injected_few_dominant_ring_is_recovered_atop_a_diffuse_legit_body() -> None:
    """End-to-end recall: a few-dominant ring injected onto one host creator inside a diffuse legit
    body is surfaced at the head of the shortlist."""
    rows: list[tuple] = []
    # diffuse legit body: 25 targets, each 25 accounts × 4 qv (high n_eff, no boosters)
    for t in range(1, 26):
        rows += [r for i in range(25) for r in _qv(t * 1000 + i, t, 0, 4)]
    # inject a few-dominant ring onto host target 13: one dominant account + two small boosters
    host = 13
    rows += _qv(990_001, host, 0, 150) + _qv(990_002, host, 0, 20) + _qv(990_003, host, 0, 20)
    con = _events_db(rows)

    suspects = generate_suspects(con, ConcentrationConfig(impact_floor=50, cap=10, booster_min_qv=5))

    assert suspects[0].target_id == host                 # the ring is recovered at rank #1
    top = suspects[0]
    assert top.n_eff < 5.0                                # its concentration is unmistakable
    assert 990_001 in top.core                            # the dominant account is in the surfaced core
    assert "multi-source" in top.tags                     # 3 dedicated contributors → coordination


def _graded_world(tmp_path, level: str):
    from fraudrings.eval.concentration import evaluate_concentration

    cfg = preset("smoke")
    cfg.sim.evasion_level = cast(EvasionLevel, level)
    events_path, gt_path = generate(cfg.sim, str(tmp_path / level))
    con = build_db(events_path, cfg.engine, db_path=str(tmp_path / level / "world.duckdb"))
    return evaluate_concentration(con, events_path, gt_path, cfg.concentration)


def test_ring_targets_head_the_shortlist_on_a_naive_world(tmp_path) -> None:
    """Seedless recovery on a simulated world: every ring's target creator is shortlisted, and all
    of them rank in the top k = 2 × #targets, above the benign superfan and lurker creators."""
    r = _graded_world(tmp_path, "naive")
    assert len(r.target_ranks) == r.n_targets
    assert r.recall_at_k == 1.0


def test_sophistication_pushes_ring_targets_down_the_shortlist(tmp_path) -> None:
    """The evasion dial dilutes concentration: sophisticated rings spread over cover creators, so
    their targets rank no better on average than naive ones."""
    naive = _graded_world(tmp_path, "naive")
    soph = _graded_world(tmp_path, "sophisticated")
    assert soph.mean_rank >= naive.mean_rank


def test_score_shortlist_counts_benign_creators_above_the_worst_target() -> None:
    from fraudrings.eval.concentration import score_shortlist

    fake = [SuspectTarget.__new__(SuspectTarget) for _ in range(5)]
    for i, t in enumerate([10, 99, 11, 98, 97]):
        object.__setattr__(fake[i], "target_id", t)
    r = score_shortlist(fake, {10, 11}, k=2)
    assert r.target_ranks == (1, 3)
    assert r.recall_at_k == 0.5
    assert r.benign_above == 1  # creator 99 sits between the two ring targets


def test_concentration_sweep_writes_a_row_per_level(tmp_path) -> None:
    import json

    from fraudrings.eval.concentration import concentration_sweep

    rows = concentration_sweep(["naive", "sophisticated"], [7], preset_name="smoke", data_dir=str(tmp_path))
    assert [r.level for r in rows] == ["naive", "sophisticated"]
    assert rows[0].recall_at_k == 1.0
    saved = json.loads((tmp_path / "concentration.json").read_text())
    assert [r["level"] for r in saved] == ["naive", "sophisticated"]
