"""The evasion dial.

The simulator gains a tunable `evasion_level` (naive|moderate|sophisticated). These
tests pin the *observable* contract through the public `generate` interface:

  * the configured level is recorded in ground truth (plumbing);
  * raising the level degrades the ring's shared-infrastructure footprint
    (the thing Stage-1 linkage keys on) and its behavioral tell-tales;
  * end-to-end, the detector must finally *discriminate* — AP falls off 1.0 and
    legit accounts start leaking into ring candidate sets.

The detector never sees `evasion_level` (it lives only in ground_truth) — the
anti-tautology isolation still holds.
"""

from __future__ import annotations

from typing import cast

import polars as pl

from fraudrings.config import SimConfig, preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import evaluate
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel


def _generate_world(tmp_path, level: EvasionLevel, *, seed: int = 7) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Generate one world at `level`; return (events, ground_truth) as Polars frames.

    Tests are the oracle — they alone may read ground_truth to map
    accounts back to rings. The detector never does.
    """
    cfg = preset("smoke")
    cfg.data_dir = str(tmp_path / level)
    cfg.sim.evasion_level = level
    cfg.sim.seed = seed
    events_path, gt_path = generate(cfg.sim, cfg.data_dir)
    return pl.read_parquet(events_path), pl.read_parquet(gt_path)


def _fraud_ids(gt: pl.DataFrame) -> list[int]:
    return gt.filter(pl.col("is_fraud"))["account_id"].to_list()


def _legit_ids(gt: pl.DataFrame) -> list[int]:
    return gt.filter(~pl.col("is_fraud"))["account_id"].to_list()


def test_evasion_level_propagates_to_ground_truth(tmp_path) -> None:
    cfg = preset("smoke")
    cfg.data_dir = str(tmp_path)
    cfg.sim.evasion_level = "moderate"

    _events_path, gt_path = generate(cfg.sim, cfg.data_dir)
    gt = pl.read_parquet(gt_path)

    rings = gt.filter(pl.col("is_fraud"))
    assert rings.height > 0, "no fraud rows generated"
    assert set(rings["evasion_level"].to_list()) == {"moderate"}

    # legit + benign-coordination controls stay labelled "none"
    legit = gt.filter(~pl.col("is_fraud"))
    assert set(legit["evasion_level"].to_list()) == {"none"}


def test_default_evasion_level_is_naive() -> None:
    assert SimConfig().evasion_level == "naive"


def test_sophisticated_moves_ring_asns_onto_legit_infra(tmp_path) -> None:
    """Naive bots sit on distinctive datacenter ASNs (a rare, strong link). Sophisticated
    bots route through residential proxies on the SAME ASNs legit users have — killing the
    ASN linkage signal."""

    def asn_overlap(events: pl.DataFrame, gt: pl.DataFrame) -> set[int]:
        fraud, legit = _fraud_ids(gt), _legit_ids(gt)
        fraud_asns = set(events.filter(pl.col("account_id").is_in(fraud))["asn"].unique().to_list())
        legit_asns = set(events.filter(pl.col("account_id").is_in(legit))["asn"].unique().to_list())
        return fraud_asns & legit_asns

    ev_naive, gt_naive = _generate_world(tmp_path, "naive")
    ev_soph, gt_soph = _generate_world(tmp_path, "sophisticated")

    assert asn_overlap(ev_naive, gt_naive) == set(), "naive rings must sit on distinctive datacenter ASNs"
    assert asn_overlap(ev_soph, gt_soph) != set(), "sophisticated rings must blend onto legit ASNs"


def test_higher_evasion_degrades_behavioral_telltales(tmp_path) -> None:
    """As sophistication rises the ring's behavioral signature should approach legit:
    timing becomes more jittered (gap_std up) and view activity less concentrated on the
    target creator (top-creator share down). Monotonic across the dial."""

    def ring_means(level: EvasionLevel) -> tuple[float, float]:
        events, gt = _generate_world(tmp_path, level)
        ring = events.filter(pl.col("account_id").is_in(_fraud_ids(gt))).sort(["account_id", "ts"])
        # per-member timing jitter (std of inter-event gaps)
        gaps = ring.with_columns(
            (pl.col("ts") - pl.col("ts").shift(1).over("account_id")).alias("gap")
        ).drop_nulls("gap")
        gap_std = gaps.group_by("account_id").agg(pl.col("gap").std().alias("s"))["s"].mean()
        # per-member concentration on the single most-viewed creator
        views = ring.filter(pl.col("creator_id").is_not_null())
        per_creator = views.group_by(["account_id", "creator_id"]).agg(pl.len().alias("c"))
        share = per_creator.group_by("account_id").agg(
            (pl.col("c").max() / pl.col("c").sum()).alias("top_share")
        )["top_share"].mean()
        # polars scalar-mean stubs return a wide union; coerce explicitly.
        return float(cast(float, gap_std)), float(cast(float, share))

    naive_jit, naive_share = ring_means("naive")
    mod_jit, mod_share = ring_means("moderate")
    soph_jit, soph_share = ring_means("sophisticated")

    assert naive_jit < mod_jit < soph_jit, (naive_jit, mod_jit, soph_jit)
    assert naive_share > mod_share > soph_share, (naive_share, mod_share, soph_share)


def test_evasion_dial_forces_discrimination_but_holds_the_legit_line(tmp_path) -> None:
    """The headline effect end-to-end. On a naive adversary the loop is trivially perfect
    (AP=recall=1.0). As sophistication rises the ranking must finally work for
    its supper: AP and Stage-1 recall both fall off 1.0.

    Crucially, the precision loss is the engine confusing rings with *each other* (still fraud),
    NOT flagging real viewers: operational precision (fraction of flagged accounts that are
    fraud at all) stays ~1.0 at every level. The benign-coordination control + ordinary legit
    never crack the flagged set — the false-positive line that matters holds."""

    def metrics(level: EvasionLevel) -> tuple[float, float, float]:
        cfg = preset("smoke")
        cfg.data_dir = str(tmp_path / level)
        cfg.sim.evasion_level = level
        events_path, gt_path = generate(cfg.sim, cfg.data_dir)
        con = build_db(events_path, cfg.engine, db_path=str(tmp_path / level / "world.duckdb"))
        result = evaluate(con, gt_path, cfg.engine, cfg.eval)

        gt = pl.read_parquet(gt_path)
        is_fraud = {int(r["account_id"]): bool(r["is_fraud"]) for r in gt.iter_rows(named=True)}
        flagged = [c.account_id for r in result.rings for c in r.top[: r.n_held_out]]
        op_precision = sum(is_fraud.get(a, False) for a in flagged) / len(flagged) if flagged else 0.0
        return result.mean_ap, result.mean_stage1_recall, op_precision

    ap_naive, rec_naive, op_naive = metrics("naive")
    ap_mod, _rec_mod, op_mod = metrics("moderate")
    ap_soph, rec_soph, op_soph = metrics("sophisticated")

    # naive is the trivial-recovery regime
    assert ap_naive >= 0.97 and rec_naive >= 0.97, (ap_naive, rec_naive)
    # sophistication monotonically degrades the per-ring ranking and retrieval
    assert ap_naive > ap_mod > ap_soph, (ap_naive, ap_mod, ap_soph)
    assert ap_soph <= 0.80, ap_soph
    assert rec_naive > rec_soph, (rec_naive, rec_soph)
    # ...but real viewers stay out of the flagged set at every level
    assert min(op_naive, op_mod, op_soph) >= 0.95, (op_naive, op_mod, op_soph)
