# 0004 — Lock-step serialization is real, but not recoverable from timing

**Status:** accepted — a negative result. The lock-step *generator* ships. No lock-step *detector
feature* does.

## Context

A finite operator can only drive so many accounts at once. With an account pool much larger than
its concurrency C, a ring's members take turns: at most C are active at any instant. That is a pure
timestamp property, so it looked like a signal that would survive every infra evasion — the
lock-step idea behind SynchroTrap and CopyCatch, inverted (serialized rather than synchronized).

To test it honestly, the generator builds a **lock-step world** where the concurrency cap is the
*only* difference between a ring and its benign **bandwagon**: same session primitive, same burst
window, same target creator, matched per-account density. A detector that separates them has to
earn it.

## What was measured

These are exploratory probes; their scripts are not shipped. Every number is reproducible from
`SimConfig.operator_concurrency` on the default preset with the method described.

1. **Per-account co-occupancy is null.** Sweeping bucket {1–15 min} × C {4, 12, 30, 60} ×
   aggregation gave ring-vs-bandwagon AUC ≈ 0.50 everywhere. The cap constrains the *set*, but its
   effect on any one (seed, candidate) pair is O(1/M) — invisible at ring size M ≈ 140 with K = 3
   seeds.
2. **The set-level signal exists only while the cap binds hard.** The peak simultaneous accounts
   for the ring vs. the bandwagon: 4 vs 11 at C=4; 12 vs 21 at C=12; 30 vs 36 at C=30; 53 vs 56 at
   C=60.
3. **A per-cluster gate (`peak / size`) fails on realistic retrieval.** It is a MAX statistic, so a
   few percent of contamination in the retrieved set erases the gap — worst exactly where the cap
   bites hardest.
4. **A human reading per-account rasters does better, but only at low C.** Per-account co-activity
   separates ring from piled-on fans at AUC 0.98 with 6% fan contamination at C=4, and stays ≥ 0.87
   out to 40%. It fades as C grows (0.71 at C=12, ~0.5 by C=60).
5. **A seedless time-synchrony front-end is infeasible.** Without seeds, the candidate set the ring
   lands in is ~15–19% pure: the bandwagon is co-target and co-window *by construction*. The best
   separation was AUC 0.63–0.67, even on the positive control.

## Decision

- **No lock-step feature in the engine.** The signal works only where it is redundant: at naive,
  infra linkage already separates the ring. It fails where it is needed: at sophisticated, the
  retrieved membership is too polluted.
- **Keep the generator and the renderer.** The lock-step world is a tested, deterministic capability.
  The temporal raster and occupancy strip in the Investigate page are an *analyst aid* for
  confirming serialization on a near-clean candidate set at low C. They never feed scoring.
- **Re-open only with new data,** not new statistics: a source where one operator's many accounts are
  observable at population scale, so the cap shows up without per-investigation retrieval.

## Why this ADR matters

Reporting this as "not met" instead of manufacturing a number is the point of the harness. The
benchmark exists to say where a method breaks.

## References

- Cao, Yang, Yu, Palow, *Uncovering Large Groups of Active Malicious Accounts in Online Social
  Networks* (SynchroTrap), CCS 2014.
- Beutel et al., *CopyCatch: Stopping Group Attacks by Spotting Lockstep Behavior in Social
  Networks*, WWW 2013.
- Kang & Kim, *Quick and easy game bot detection based on action time interval estimation*, ETRI
  Journal 2023 — a timing detector that works on metronomic bots and concedes the one-line
  random-delay evasion.
- `tests/test_lockstep.py`, `tests/test_raster.py`, `src/fraudrings/viewer/raster.py`.
