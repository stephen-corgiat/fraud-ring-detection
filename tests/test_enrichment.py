"""World enrichment: the action funnel, the accounts registry, the lurker confound, and the
load-bearing rng-isolation guarantee.

These pin the generator side:
  1. all five funnel actions are emitted; `complete` is a relabel (not a new event);
  2. the detector-readable `accounts.parquet` registry covers every account and validates;
  3. windowed truncation — `register` is sparse, fresh rings co-register in-window, sophisticated
     rings age their accounts (evasion convergence);
  4. the benign lurker segment is honest + funnel-truncated (the anti-tautology confound);
  5. RNG ISOLATION: enrichment randomness lives on a separate stream, so the core
     MEDIA stream — everything the old behavioral basket reads — is byte-identical regardless of the
     enrichment knobs. That is *why* the behavioral results can't move when the funnel knobs do.
"""

from __future__ import annotations

import os
from typing import cast

import polars as pl

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.generator import generate
from fraudrings.schema import ACCOUNT_COLUMNS, EVENT_COLUMNS, Account, EvasionLevel, Event


def _mean(series: pl.Series) -> float:
    """Series mean as a float (polars types `mean()` as an optional literal)."""
    return float(cast(float, series.mean()))


def _world(tmp_path, level="naive", *, seed=7, **sim):
    cfg = preset("smoke")
    cfg.sim.evasion_level = cast(EvasionLevel, level)
    cfg.sim.seed = seed
    for k, v in sim.items():
        setattr(cfg.sim, k, v)
    d = str(tmp_path)
    ep, gp = generate(cfg.sim, d)
    return cfg, ep, gp, d


def _segment(cfg, account_id: int) -> str:
    """Account-id block convention: legit, then lurkers, then rings/clusters."""
    nl, nlur = cfg.sim.n_legit, cfg.sim.n_lurkers
    if account_id < nl:
        return "legit"
    if account_id < nl + nlur:
        return "lurker"
    return "ringblock"


def test_all_funnel_actions_emitted_and_schema_holds(tmp_path) -> None:
    cfg, ep, gp, d = _world(tmp_path)
    ev = pl.read_parquet(ep)
    assert tuple(ev.columns) == EVENT_COLUMNS
    Event.model_validate(ev.row(0, named=True))
    actions = set(ev["action"].unique().to_list())
    # the enrichment turns the two-action stream into the full funnel
    assert {"register", "login", "search", "view", "complete", "skip", "save"} <= actions
    assert ev["ts"].is_sorted()  # still time-ordered (incremental-path precondition)


def test_accounts_registry_covers_every_account_and_validates(tmp_path) -> None:
    cfg, ep, gp, d = _world(tmp_path)
    acc_path = os.path.join(d, "accounts.parquet")
    assert os.path.exists(acc_path)
    acc = pl.read_parquet(acc_path)
    assert tuple(acc.columns) == ACCOUNT_COLUMNS
    Account.model_validate(acc.row(0, named=True))
    gt = pl.read_parquet(gp)
    # one registry row per ground-truth account (a 1:1 detector-readable side table)
    assert sorted(acc["account_id"].to_list()) == sorted(gt["account_id"].to_list())
    assert acc["registration_ts"].null_count() == 0


def test_complete_is_a_relabel_not_a_new_event(tmp_path) -> None:
    """`complete` is a relabel: every complete is a high-position media event, and folding view+complete back
    together reproduces the non-skip view count the old basket reads as `n_views`."""
    cfg, ep, gp, d = _world(tmp_path)
    ev = pl.read_parquet(ep)
    completes = ev.filter(pl.col("action") == "complete")
    assert completes.height > 0
    assert (completes["reported_position_pct"] >= cfg.sim.complete_threshold).all()
    assert (~completes["skip_flag"]).all()  # a completion is never a skip
    # n_views in ingest folds complete back in -> == non-skip view events
    con = build_db(ep, cfg.engine, db_path=os.path.join(d, "a.duckdb"))
    row = con.execute("SELECT sum(n_views) FROM account_features").fetchone()
    assert row is not None
    n_views = row[0]
    non_skip_views = ev.filter(pl.col("action").is_in(["view", "complete"])).height
    assert int(n_views) == non_skip_views


def test_register_is_sparse_and_rings_age_under_sophistication(tmp_path) -> None:
    """Windowed truncation + evasion convergence: a naive ring is freshly co-registered (many
    in-window `register` events, low account age); a sophisticated ring ages its accounts (few
    in-window registers, legit-like age)."""
    _, ep_n, _, _ = _world(tmp_path / "n", level="naive")
    _, ep_s, _, _ = _world(tmp_path / "s", level="sophisticated")
    n_reg = pl.read_parquet(ep_n).filter(pl.col("action") == "register").height
    s_reg = pl.read_parquet(ep_s).filter(pl.col("action") == "register").height
    # register is sparse in both (most accounts registered before the window) ...
    assert n_reg < pl.read_parquet(ep_n).height * 0.05
    # ... and the naive fresh-ring burst means materially MORE in-window registers than aged soph.
    assert n_reg > s_reg


def test_lurker_segment_is_benign_aged_and_funnel_truncated(tmp_path) -> None:
    """The anti-tautology confound: lurkers are honest (not fraud), aged (high account_age), and
    funnel-truncated (rarely search/save) — funnel-thin like a naive ring but NOT fraud."""
    cfg, ep, gp, d = _world(tmp_path)
    gt = pl.read_parquet(gp)
    nl, nlur = cfg.sim.n_legit, cfg.sim.n_lurkers
    lurkers = set(range(nl, nl + nlur))
    lr = gt.filter(pl.col("account_id").is_in(list(lurkers)))
    assert lr.height == nlur
    assert (~lr["is_fraud"]).all() and lr["ring_id"].is_null().all()  # benign
    con = build_db(ep, cfg.engine, db_path=os.path.join(d, "a.duckdb"))
    feats = con.sql(
        "SELECT account_id, search_rate, save_rate, account_age_days FROM account_features"
    ).pl()
    lf = feats.filter(pl.col("account_id").is_in(list(lurkers)))
    assert _mean(lf["search_rate"]) < 0.05 and _mean(lf["save_rate"]) < 0.05
    assert _mean(lf["account_age_days"]) > 60.0  # aged (not a fresh ring)


def test_enrichment_rng_is_isolated_from_the_core_media_stream(tmp_path) -> None:
    """RNG isolation, made exact: enrichment randomness (here the `login_prob` knob) lives on a
    dedicated rng, so the MEDIA events (view/complete/skip) — and therefore every OLD-basket feature
    computed over them — are byte-identical regardless of the enrichment knobs. Only funnel events
    differ."""
    cfg_a, ep_a, _, da = _world(tmp_path / "a", login_prob=0.0)
    cfg_b, ep_b, _, db = _world(tmp_path / "b", login_prob=0.9)
    # The lurker segment is itself a new population on the enrichment stream, so its media events DO move
    # with the knob. The guarantee is about the CORE populations (legit / rings / fan-clubs) — the
    # ones the behavioral tests grade. n_lurkers is fixed here, so the lurker block is a stable
    # account-id range we exclude.
    nl, nlur = cfg_a.sim.n_legit, cfg_a.sim.n_lurkers
    core = ~pl.col("account_id").is_between(nl, nl + nlur - 1)
    media = ["view", "complete", "skip"]
    la = pl.read_parquet(ep_a).filter(pl.col("action").is_in(media) & core)
    lb = pl.read_parquet(ep_b).filter(pl.col("action").is_in(media) & core)
    assert la.equals(lb), "the core media stream must not depend on an enrichment knob"

    # ...and so the old 10-feature basket is identical for every core account (ids are unshifted here).
    old_basket = list(cfg_a.engine.behavioral_features)
    sel = "account_id, " + ", ".join(old_basket)
    where = f"WHERE account_id < {nl} OR account_id >= {nl + nlur}"
    ca = build_db(ep_a, cfg_a.engine, db_path=os.path.join(da, "a.duckdb")).sql(
        f"SELECT {sel} FROM account_features {where} ORDER BY account_id"
    ).pl()
    cb = build_db(ep_b, cfg_b.engine, db_path=os.path.join(db, "b.duckdb")).sql(
        f"SELECT {sel} FROM account_features {where} ORDER BY account_id"
    ).pl()
    assert ca.equals(cb), "the old behavioral basket leaked an enrichment knob"


def test_actionmix_features_separate_naive_ring_from_benign(tmp_path) -> None:
    """The action-mix sub-features land on `account_features` and a naive ring is a glaring outlier:
    perfect completion spoof, no search/save, fresh accounts — far from legit/lurker."""
    cfg, ep, gp, d = _world(tmp_path, level="naive")
    gt = pl.read_parquet(gp)
    con = build_db(ep, cfg.engine, db_path=os.path.join(d, "a.duckdb"))
    cols = ["completion_rate", "search_rate", "save_rate", "funnel_completeness", "account_age_days"]
    feats = con.sql(f"SELECT account_id, {', '.join(cols)} FROM account_features").pl()
    j = feats.join(gt.select(["account_id", "is_fraud"]), on="account_id")
    ring = j.filter(pl.col("is_fraud"))
    legit = j.filter((~pl.col("is_fraud")) & (pl.col("account_id") < cfg.sim.n_legit))
    # naive ring: spoofed completion ~1, no funnel breadth, fresh accounts
    assert _mean(ring["completion_rate"]) > 0.9
    assert _mean(ring["search_rate"]) < 0.02
    assert _mean(ring["funnel_completeness"]) < _mean(legit["funnel_completeness"])
    assert _mean(ring["account_age_days"]) < _mean(legit["account_age_days"]) / 2


def test_evasion_convergence_fades_the_action_mix_tell(tmp_path) -> None:
    """The anti-tautology convergence: a sophisticated ring fakes the full funnel on
    aged accounts, so its action-mix signature moves toward legit — the funnel gap that is glaring at
    naive nearly closes at sophisticated."""
    def funnel_gap(level):
        cfg, ep, gp, d = _world(tmp_path / level, level=level)
        gt = pl.read_parquet(gp)
        con = build_db(ep, cfg.engine, db_path=os.path.join(d, "a.duckdb"))
        f = con.sql(
            "SELECT account_id, funnel_completeness, account_age_days FROM account_features"
        ).pl().join(gt.select(["account_id", "is_fraud"]), on="account_id")
        ring = f.filter(pl.col("is_fraud"))
        legit = f.filter((~pl.col("is_fraud")) & (pl.col("account_id") < cfg.sim.n_legit))
        return (
            _mean(legit["funnel_completeness"]) - _mean(ring["funnel_completeness"]),
            _mean(legit["account_age_days"]) - _mean(ring["account_age_days"]),
        )

    naive_fgap, naive_agap = funnel_gap("naive")
    soph_fgap, soph_agap = funnel_gap("sophisticated")
    # both the funnel-completeness gap and the account-age gap shrink dramatically with sophistication
    assert soph_fgap < naive_fgap * 0.5, (naive_fgap, soph_fgap)
    assert soph_agap < naive_agap * 0.5, (naive_agap, soph_agap)
