# 0008 — Isolated RNG streams for additive generator features

**Status:** accepted

## Context

A simulator grows. Every new population or event type changes the random draws that follow it, so
every downstream number shifts — and then nobody can tell whether a metric moved because the
*method* changed or because the *dice* did. With a test suite that pins dozens of measured results,
"re-baseline everything" after each generator change would make those tests meaningless.

## Decision

- Draw every **additive** generator feature from its own RNG stream, seeded at
  `cfg.seed + offset`. This covers registration, lifecycle events, funnel search/save presence and
  the whole lurker population.
- The **core content stream** — infra, timing, views, skips, reported positions, everything the
  behavioral basket reads — draws only from the main RNG, so it is independent of the additive knobs.
- New event types are **added**, never interleaved into existing draws. `complete` is a relabel of an
  existing view event, not a new event, so session and gap structure are untouched.
- Pin it with exact tests, not statistical ones:
  - `test_enrichment` changes a funnel knob and asserts the core media events are byte-identical;
  - `test_lockstep` pins a content hash of the identity world (a canonical CSV of the events, not
    parquet bytes, which drift with the writer version).

## Consequences

- Features ship without moving existing results, so the sweep, grade and curve tests stayed fixed
  through the funnel, lurker and lock-step additions.
- The same discipline made the vocabulary refactor provable: mapping the renamed columns back
  reproduced the previous content hash exactly.
- The cost is a little bookkeeping: each new stochastic feature has to pick its stream deliberately.
