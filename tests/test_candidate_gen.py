"""Stage 1b: behavioral (co-target) candidate generation.

When a sophisticated adversary blends its shared infrastructure onto legit infra, Stage-1
infra linkage stops retrieving the whole ring — recall caps at ~0.57 (default) / ~0.71
(smoke). No Stage-2 rule can rank a member it never retrieved, so to lift recall we must
widen *retrieval*.

The economic invariant the adversary can't escape: a ring exists to pump qualified views of
a *target creator* for payouts. So when infra linkage is thin (seeds barely interlink), we
expand on the rare target creator the seeds qualified-viewed *in common*, time-localized to the
ring's campaign window. The expansion is ADAPTIVE — it fires only when linkage cohesion is low,
so the pristine naive/moderate regimes are untouched.

Precision (keeping genuine fans of the target out of the flagged set) is the learned Stage-2
rule's job (`test_supervised.py`) — here we only assert retrieval widens where it should and nowhere it shouldn't.
The detector still reads only events; `evasion_level` never enters (anti-tautology).
"""

from __future__ import annotations

from fraudrings.config import preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.evaluate import EvalResult, evaluate
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel


def _eval(tmp_path, level: EvasionLevel, *, cotarget: bool, seed: int = 7) -> EvalResult:
    cfg = preset("smoke")
    sub = f"{level}-{cotarget}-{seed}"
    cfg.data_dir = str(tmp_path / sub)
    cfg.sim.evasion_level = level
    cfg.sim.seed = seed
    cfg.engine.cotarget_candidate_gen = cotarget
    events_path, gt_path = generate(cfg.sim, cfg.data_dir)
    con = build_db(events_path, cfg.engine, db_path=str(tmp_path / sub / "world.duckdb"))
    return evaluate(con, gt_path, cfg.engine, cfg.eval)


def test_cotarget_candidate_gen_lifts_recall_at_sophisticated(tmp_path) -> None:
    """The headline: co-target retrieval breaks the linkage-only Stage-1 ceiling."""
    base = _eval(tmp_path, "sophisticated", cotarget=False)
    cot = _eval(tmp_path, "sophisticated", cotarget=True)
    assert cot.mean_stage1_recall > base.mean_stage1_recall, (
        base.mean_stage1_recall, cot.mean_stage1_recall
    )


def test_cotarget_is_adaptive_and_does_not_disturb_naive(tmp_path) -> None:
    """At naive, infra linkage already retrieves the whole ring (cohesion high), so the
    co-target net must NOT fire — results are byte-identical and the legit line stays clean."""
    base = _eval(tmp_path, "naive", cotarget=False)
    cot = _eval(tmp_path, "naive", cotarget=True)
    assert cot.mean_stage1_recall == base.mean_stage1_recall == 1.0
    assert cot.flagged_total == base.flagged_total
    assert cot.legit_fp_rate == 0.0


def test_cotarget_widens_the_candidate_pool_at_sophisticated(tmp_path) -> None:
    """Retrieval genuinely grows: the union pulls in members linkage alone missed."""
    base = _eval(tmp_path, "sophisticated", cotarget=False)
    cot = _eval(tmp_path, "sophisticated", cotarget=True)
    base_cands = sum(r.n_candidates for r in base.rings)
    cot_cands = sum(r.n_candidates for r in cot.rings)
    assert cot_cands > base_cands, (base_cands, cot_cands)
