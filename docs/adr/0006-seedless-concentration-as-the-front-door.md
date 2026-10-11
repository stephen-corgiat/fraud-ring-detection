# 0006 — Seedless concentration as the front door; suspects, not verdicts

**Status:** accepted

## Context

The seeded engine answers "given a few accounts we already suspect, find the rest of the ring."
Someone has to supply those first accounts. In practice nobody hands you seeds. You need a way in
that reads only the event stream.

A ring exists to inflate one creator's payouts, so that creator's qualified views *concentrate*
in ways organic popularity rarely does:

- on a few accounts (low effective contributor count, `n_eff`),
- on many accounts dedicated to it alone (high **booster** count),
- or inside one short window (high `burst7`).

Each shape catches an archetype the others miss. Dense-block methods (FRAUDAR, fBox) formalize
related intuitions at the graph level. Concentration is the cheapest, most interpretable member of
that family, and it needs no labels.

## Decision

- `detector/concentration.py` scores every creator above an **impact floor** on three **entry axes**
  as percentiles within the pool. `suspicion` is the max across them. `rank_score` adds payout impact
  (capped qualified views). The default order is `suspicion`, because a human triaging wants "how
  coordinated does this look", and a merely popular creator must not float up on volume alone.
- Output is **ranked suspects for human review, never verdicts.** Every suspect carries **topology
  tags** that describe its observed contributor shape (`single-source`, `few-source`,
  `multi-source`, `campaign`). Coordination is inferred only at `multi-source`. The benign twin of a
  booster is a superfan, and the two can be behaviorally identical.
- **The expansion gate.** A suspect's core is handed to the seeded engine only when its members
  interlink on rare infrastructure. Without shared infra, expansion fans out into noise, and the
  core *is* the output.
- The **Triage** page puts this in front of a human: the queue, the three axes, contributor and
  temporal concentration, the core with infra evidence, the expansion verdict, and a one-click
  handoff of the core into **Investigate** as seeds.

## Consequences

`fraudrings rank` grades the ranker against the hidden ring targets on the default preset, seeds
7, 13, 21 (k = 2 × number of ring targets):

| level | ring targets in top k | mean rank | worst rank | shortlist |
|---|---|---|---|---|
| naive | 100% | 6.9 | 15 | 40 |
| moderate | 92% | 9.5 | 17 | 123 |
| sophisticated | 50% | 46.7 | 129 | 152 |

The same degradation shape as the seeded engine: concentration finds naive and moderate rings at
the head; sophisticated rings spread views across cover creators and half their targets sink. Small,
loyal-audience creators share the head with ring targets at naive — they rank on `n_eff` alone, carry
no tags, and are exactly the case triage exists to dismiss.

## References

- Hooi et al., *FRAUDAR: Bounding Graph Fraud in the Face of Camouflage*, KDD 2016.
- Shah et al., *Spotting Suspicious Link Behavior with fBox*, ICDM 2014.
- `src/fraudrings/detector/concentration.py`, `src/fraudrings/eval/concentration.py`,
  `tests/test_concentration.py`, `tests/test_triage.py`.
