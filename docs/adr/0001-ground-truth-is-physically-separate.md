# 0001 — Ground truth is physically separate from detector input

**Status:** accepted

## Context

A synthetic benchmark invites the most common failure in applied fraud ML: *detecting exactly what
you injected*. If the generator's knobs, the ring ids, or the evasion level can reach the detector
by any path — a shared table, a feature computed from config, a test fixture that peeks — the
reported precision and recall measure the plumbing, not the method. Arp et al. catalogue this family
of mistakes (data snooping, spurious correlations, inappropriate baselines) and find it in a large
share of published security-ML work.

## Decision

Enforce the boundary structurally, not by convention:

- `generator/` writes **two files**: `events.parquet` (what a collector would see) and
  `ground_truth.parquet` (`is_fraud`, `ring_id`, `evasion_level`, bandwagon cohort). A third,
  `accounts.parquet`, holds only what a real platform knows (registration time).
- `detector/` reads `events` and `accounts` only. It never opens ground truth and never reads the
  world's config. `evasion_level` exists only in ground truth.
- `eval/` is the only package that joins the two — it plays the oracle.
- The viewer sits on the eval side (it may reveal truth behind a toggle) but every chart is
  presentation-only: nothing a chart computes feeds scoring (see [0007](0007-plotly-single-chart-library.md)).
- **Benign look-alikes are first-class populations**, so no single cue is a free answer:
  - *superfan cohorts* concentrate on the same creators rings target;
  - *lurkers* have truncated funnels without being fraud;
  - *bandwagon* crowds pile onto a ring's burst window;
  - *household IP sharers and VPN users* create coincidental infra links.

## Consequences

- Tests enforce it: the concentration ranker's test poisons the database with a `ground_truth`
  table and asserts the output is unchanged, and inspects the query source for forbidden names.
- Every headline number has an honest denominator: legit false positives include the confounders
  built to fool the detector.
- The cost is realism. A simulator encodes its author's threat model; results characterize the
  method *against that model*. The evasion dial ([0002](0002-adversary-is-a-dial.md)) exists to
  make that model explicit and adjustable rather than implicit.

## References

- Arp et al., *Dos and Don'ts of Machine Learning in Computer Security*, USENIX Security 2022.
- [GLOSSARY.md](../../GLOSSARY.md) — *anti-tautology*, *ground truth*, *superfan cohort*, *lurker*.
