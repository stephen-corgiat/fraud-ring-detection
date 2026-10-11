"""Orchestration + CLI.

    generate (world @ evasion level) -> ingest (DuckDB tables) -> investigate seeds -> measure

Run:
    fraudrings run                            # 'default' preset, one world
    fraudrings run --preset smoke             # tiny
    fraudrings run --seed 13
    fraudrings sweep                          # dial naive->moderate->sophisticated, both curves
    fraudrings sweep --preset smoke --seeds 7,13,21
    fraudrings grade                          # learned rule vs the success bar, held-out rings
    fraudrings curve --preset smoke           # learning curve: metric vs #labels, with label noise
    fraudrings rank                           # seedless ranker: ring targets on the shortlist, per level
    fraudrings view                           # the Triage + Investigate UI ([viz] extra)
"""

from __future__ import annotations

import argparse
import os

import polars as pl

from fraudrings.config import RunConfig, preset
from fraudrings.detector.ingest import build_db
from fraudrings.eval.concentration import ConcentrationLevel, concentration_sweep
from fraudrings.eval.curve import DEFAULT_ASYMMETRIC, CurveResult, learning_curve, write_curve
from fraudrings.eval.evaluate import EvalResult, evaluate
from fraudrings.eval.grade import GradeResult, grade
from fraudrings.eval.sweep import LEVELS, SweepResult, sweep
from fraudrings.generator import generate
from fraudrings.schema import EvasionLevel


def run(cfg: RunConfig) -> EvalResult:
    os.makedirs(cfg.data_dir, exist_ok=True)
    print(f"[1/3] generating synthetic world (seed={cfg.sim.seed}) ...")
    events_path, gt_path = generate(cfg.sim, cfg.data_dir)

    events = pl.read_parquet(events_path)
    gt = pl.read_parquet(gt_path)
    n_fraud = int(gt.filter(pl.col("is_fraud")).height)
    print(
        f"      {events.height:,} events | {gt.height:,} accounts "
        f"({n_fraud} fraud across {cfg.sim.n_rings} rings, {gt.height - n_fraud} legit/benign)"
    )

    print("[2/3] ingest -> features / assoc / rarity (DuckDB) ...")
    con = build_db(events_path, cfg.engine, db_path=os.path.join(cfg.data_dir, "world.duckdb"))

    print("[3/3] seed-anchored link-then-score + measure ...")
    result = evaluate(con, gt_path, cfg.engine, cfg.eval)
    _report(result, gt)
    return result


def _report(result: EvalResult, gt: pl.DataFrame) -> None:
    truth = {int(r["account_id"]): bool(r["is_fraud"]) for r in gt.iter_rows(named=True)}

    print("\n=== per-ring recovery ===")
    print(f"{'ring':>4} {'size':>4} {'cand':>5} {'s1_recall':>9} {'AP':>6} {'R@k':>6} {'P@k':>6}")
    for r in result.rings:
        print(
            f"{r.ring_id:>4} {r.n_members:>4} {r.n_candidates:>5} "
            f"{r.stage1_recall:>9.2f} {r.ap:>6.2f} {r.recall_at_k:>6.2f} {r.precision_at_k:>6.2f}"
        )

    print("\n=== headline ===")
    print(f"  mean Stage-1 recall : {result.mean_stage1_recall:.3f}   (retrieval ceiling)")
    print(f"  mean AP (ranking)   : {result.mean_ap:.3f}")
    print(f"  mean recall@k       : {result.mean_recall_at_k:.3f}")
    print(f"  mean precision@k    : {result.mean_precision_at_k:.3f}")

    if result.rings:
        r = result.rings[0]
        print(f"\n=== sample investigation (ring {r.ring_id}, top candidates) ===")
        print(f"  {'account':>8} {'score':>6} {'link':>5} {'behav':>6} {'cotgt':>6} {'amix':>6}  truth")
        for c in r.top:
            tag = "FRAUD" if truth.get(c.account_id) else "legit"
            print(
                f"  {c.account_id:>8} {c.score:>6.2f} {c.linkage:>5.2f} "
                f"{c.behavioral:>6.2f} {c.cotarget:>6.2f} {c.actionmix:>6.2f}  {tag}"
            )


def run_sweep(
    levels: list[EvasionLevel], seeds: list[int], preset_name: str, data_dir: str
) -> SweepResult:
    print(
        f"sweeping evasion dial {list(levels)} over seeds {seeds} "
        f"(preset={preset_name}) — re-generating a world per (level, seed) ..."
    )
    result = sweep(levels=levels, seeds=seeds, preset_name=preset_name, data_dir=data_dir)
    _report_sweep(result, data_dir)
    return result


def _report_sweep(result: SweepResult, data_dir: str) -> None:
    # (a) per-ring ranking curve — the headline that degrades under evasion.
    print("\n=== sweep (a): per-ring ranking vs. sophistication ===")
    print(f"{'level':>14} {'AP':>6} {'s1_recall':>9} {'R@k':>6} {'P@k':>6}")
    for s in result.levels:
        print(
            f"{s.level:>14} {s.mean_ap:>6.2f} {s.mean_stage1_recall:>9.2f} "
            f"{s.mean_recall_at_k:>6.2f} {s.mean_precision_at_k:>6.2f}"
        )

    # (b) operational fraud-vs-legit curve — the false-positive line that should HOLD.
    print("\n=== sweep (b): operational fraud-vs-legit (flagged slots) ===")
    print(f"{'level':>14} {'op_prec':>8} {'legit_FP':>9} {'cross_ring':>11} {'flagged':>8}")
    for s in result.levels:
        print(
            f"{s.level:>14} {s.operational_precision:>8.2f} {s.legit_fp_rate:>9.2f} "
            f"{s.cross_ring_frac:>11.2f} {s.flagged_total:>8}"
        )

    print(
        "\nreading: ranking (a) slopes down as the adversary blends onto legit infra; the "
        "operational line (b) holds — the\nprecision loss is cross-ring bleed (still fraud), "
        "not legit false positives."
    )
    print(f"\nwrote {os.path.join(data_dir, 'sweep.json')} and {os.path.join(data_dir, 'sweep.csv')}")


def run_grade(
    levels: list[EvasionLevel], seeds: list[int], preset_name: str, data_dir: str
) -> GradeResult:
    print(
        f"grading the learned rule (co-target candidate-gen + learned basket) on held-out rings "
        f"across {list(levels)}, seeds {seeds} (preset={preset_name}) ..."
    )
    result = grade(levels=levels, seeds=seeds, preset_name=preset_name, data_dir=data_dir)
    _report_grade(result, data_dir)
    return result


def _report_grade(result: GradeResult, data_dir: str) -> None:
    t = result.targets
    print(
        f"\n=== rule grade vs targets (precision>={t.precision}, recall>={t.recall}, "
        f"legit-FP<={t.legit_fp}); held-out rings ==="
    )
    print(f"{'level':>14} {'recall':>7} {'op_prec':>8} {'legitFP':>8} {'meets':>6} "
          f"{'seeds✓':>7} {'fp_max':>7} {'base@k':>6} {'lift':>6} {'+chal':>6} {'lgbm':>6}")
    for g in result.levels:
        print(
            f"{g.level:>14} {g.recall:>7.2f} {g.operational_precision:>8.2f} {g.legit_fp_rate:>8.2f} "
            f"{('YES' if g.meets else 'no'):>6} {f'{g.n_seeds_meeting}/{len(g.seeds)}':>7} "
            f"{g.fp_max:>7.2f} {g.baseline_recall_at_k:>6.2f} {g.recall_lift:>+6.2f} "
            f"{g.challenge_recall:>6.2f} {g.lgbm_recall:>6.2f}"
        )
    print(
        "\nreading: headline cols are means over seeds; 'seeds✓' = how many individual seeds clear the "
        "full bar\n(variance-aware 'meets'), 'fp_max' = worst-case legit-FP across seeds (the line to "
        "defend). 'lift' = recall\ngained over the hand-weighted baseline ranking; '+chal' adds the "
        "near-perfect challenge tell (barely moves the\nbasket — the weak-signal basket carries the "
        "rule); 'lgbm' overfits the small per-ring folds."
    )
    print(f"\nwrote {os.path.join(data_dir, 'rule_grades.json')} and {os.path.join(data_dir, 'rule_grades.csv')}")


def run_curve(
    seeds: list[int], preset_name: str, data_dir: str, level: EvasionLevel, asymmetric: bool
) -> CurveResult:
    extra = DEFAULT_ASYMMETRIC if asymmetric else ()
    print(
        f"building the learning curve at '{level}' (random/uncertainty/exploit × "
        f"{'symmetric+asymmetric' if asymmetric else 'symmetric'} noise sweep) plus a per-level "
        f"labels-to-target summary, seeds {seeds} (preset={preset_name}) — re-generating a frozen "
        f"candidate pool per (level, seed); this is the heavy run, background it for the default preset ..."
    )
    result = learning_curve(
        seeds=seeds, preset_name=preset_name, data_dir=data_dir, level=level, asymmetric=extra
    )
    write_curve(result, data_dir)
    _report_curve(result, data_dir)
    return result


def _report_curve(result: CurveResult, data_dir: str) -> None:
    t = result.targets
    print(
        f"\n=== learning curve @ {result.level} (held-out rings; target precision>={t.precision}, "
        f"recall>={t.recall}, legit-FP<={t.legit_fp}) ==="
    )
    # Full strategy × noise curves at the headline level.
    for s in result.series:
        cross = "never" if s.labels_to_target is None else str(s.labels_to_target)
        print(f"\n-- strategy={s.strategy:<11} {s.noise.label:<14} labels_to_target={cross}")
        print(f"   {'#labels':>8} {'recall':>7} {'op_prec':>8} {'legitFP':>8} {'meets':>6}")
        for p in s.points:
            tag = "full" if p.is_full else ""
            print(
                f"   {p.n_labels:>8} {p.recall:>7.2f} {p.operational_precision:>8.2f} "
                f"{p.legit_fp_rate:>8.2f} {('YES' if p.meets() else 'no'):>6} {tag}"
            )

    # Cheap per-level crossing summary (random, ε=0) — how the human cost rises with sophistication.
    print("\n=== labels-to-target by evasion level (random query, ε=0) ===")
    for lvl, n in result.labels_to_target_by_level.items():
        print(f"   {lvl:>14}: {'never reached' if n is None else f'{n} labels'}")

    print(
        "\nreading: each curve is the seed-averaged held-out metric vs. the number of oracle labels "
        "used to TRAIN the\nrule on a frozen candidate pool; 'labels_to_target' is the smallest budget "
        "that clears the bar and STAYS\ncleared (the stable crossing). Noise (ε) corrupts training "
        "labels only — held-out grading is always clean;\nas ε rises the legit-FP line drifts onto "
        "the 5% bar, so the STABLE target can slip out of reach even\nwhen recall recovers (precision "
        "is the fragile constraint, not recall). NB: uncertainty sampling can\nclear the bar early "
        "but is non-monotonic — it falls back over the line at larger budgets, loses to random\nat "
        "smoke scale, and adds no robustness under noise."
    )
    print(f"\nwrote {os.path.join(data_dir, 'learning_curve.json')} and {os.path.join(data_dir, 'learning_curve.csv')}")


def run_rank(levels: list[EvasionLevel], seeds: list[int], preset_name: str, data_dir: str) -> list[ConcentrationLevel]:
    print(f"grading the seedless concentration ranker across {list(levels)}, seeds {seeds} "
          f"(preset={preset_name}) ...")
    result = concentration_sweep(levels, seeds, preset_name=preset_name, data_dir=data_dir)
    print("\n=== seedless ranker: where the ring-target creators land on the shortlist ===")
    print(f"{'level':>14} {'recall@k':>9} {'mean_rank':>10} {'worst_rank':>11} {'shortlist':>10}")
    for r in result:
        print(f"{r.level:>14} {r.recall_at_k:>9.2f} {r.mean_rank:>10.1f} {r.worst_rank:>11} {r.n_suspects:>10.0f}")
    print("\nreading: k = 2 x #ring targets. Concentration finds naive and moderate rings near the head; "
          "sophisticated\nrings spread their views over cover creators and some targets sink into the tail.")
    print(f"\nwrote {os.path.join(data_dir, 'concentration.json')} and {os.path.join(data_dir, 'concentration.csv')}")
    return result


def run_view(data_dir: str, rebuild: bool = False) -> None:
    """Launch the Streamlit viewer (`fraudrings view`): Triage + Investigate. Needs the [viz] extra.

    `rebuild=True` force-regenerates every world loaded this session; stale worlds from an older
    schema auto-rebuild regardless (see `viewer.data.load_world`).
    """
    import importlib.util
    import subprocess
    import sys

    if importlib.util.find_spec("streamlit") is None:
        raise SystemExit(
            "the viewer needs the [viz] extra — install it with:\n"
            "  .venv/bin/python -m pip install -e \".[viz]\"\n"
            "then re-run `fraudrings view`."
        )
    app_path = os.path.join(os.path.dirname(__file__), "viewer", "main.py")
    print(f"launching the fraudrings viewer (data-dir={data_dir}) — open the URL Streamlit prints ...")
    # `--` separates Streamlit's args from the app's own (the app parses --data-dir / --rebuild).
    app_args = ["--data-dir", data_dir] + (["--rebuild"] if rebuild else [])
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--", *app_args],
        check=False,
    )


def _parse_levels(spec: str) -> list[EvasionLevel]:
    valid = set(LEVELS)
    levels = [p.strip() for p in spec.split(",") if p.strip()]
    bad = [lvl for lvl in levels if lvl not in valid]
    if bad:
        raise SystemExit(f"unknown evasion level(s) {bad}; expected a subset of {list(LEVELS)}")
    return [lvl for lvl in LEVELS if lvl in levels]  # canonical (increasing) order


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="fraudrings", description="Simulate, detect and measure coordinated fraud rings"
    )
    parser.add_argument(
        "command", nargs="?", default="run",
        choices=["run", "sweep", "grade", "curve", "rank", "view"],
    )
    parser.add_argument("--preset", default="default", choices=["smoke", "default"])
    parser.add_argument("--seed", type=int, default=None, help="single seed (run)")
    parser.add_argument(
        "--seeds", default="7,13,21", help="comma-separated seeds to average (sweep)"
    )
    parser.add_argument(
        "--levels", default=",".join(LEVELS), help="comma-separated evasion levels (sweep)"
    )
    parser.add_argument(
        "--level",
        default="sophisticated",
        choices=list(LEVELS),
        help="evasion level for the full learning curve (curve)",
    )
    parser.add_argument(
        "--asymmetric",
        action="store_true",
        help="also sweep realistic missed-fraud-dominant noise profiles (curve)",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="force-rebuild cached worlds before viewing (view); stale-schema worlds rebuild anyway",
    )
    args = parser.parse_args()

    if args.command == "view":
        run_view(args.data_dir, args.rebuild)
        return

    if args.command in ("sweep", "grade", "curve", "rank"):
        seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
        if args.command == "curve":
            run_curve(seeds, args.preset, args.data_dir, args.level, args.asymmetric)
            return
        levels = _parse_levels(args.levels)
        if args.command == "sweep":
            run_sweep(levels, seeds, args.preset, args.data_dir)
        elif args.command == "rank":
            run_rank(levels, seeds, args.preset, args.data_dir)
        else:
            run_grade(levels, seeds, args.preset, args.data_dir)
        return

    cfg = preset(args.preset)
    if args.seed is not None:
        cfg.sim.seed = args.seed
    cfg.data_dir = args.data_dir
    run(cfg)


if __name__ == "__main__":
    main()
