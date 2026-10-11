# 0005 — What over when: action mix over timing

**Status:** accepted

## Context

Behavioral features fall on a durability ladder. **Timing statistics** (inter-event gaps, cadence,
active hours) are the cheapest thing for an adversary to perturb: one line of random delay defeats
a timing-marginal detector, and its own authors say so (Kang & Kim 2023). [0004](0004-lockstep-serialization-is-not-recoverable.md)
found the same wall for cross-account timing. **What** an account does — the composition and ratios
of its actions — is costlier to fake, because faking it means doing the work. Lugoloobi et al.
(*Known By Their Actions*, 2026) find the same split from the attacker's side when fingerprinting
LLM browser agents: action composition carries identity that timing alone does not.

## Decision

- Add an **action funnel** to the world: `search → view → complete → save`, plus `register` and
  `login`. Engaged legit viewers exercise all of it. A naive ring views its target directly: no
  search, no save, a perfect completion spoof, on freshly co-registered accounts.
- Add an **action-mix axis** to Stage 2. It is a seed-centroid proximity over completion rate,
  search rate, save rate, funnel completeness and account age, kept as its own term so it diagnoses
  separately. Weights: 0.45 linkage + 0.25 behavioral + 0.15 co-target + 0.15 action-mix.
- **Make the axis earn its keep.** The *lurker* confound is a funnel-truncated honest population, so
  "truncated funnel ⇒ fraud" can never be a free answer. A *sophisticated* ring fakes the whole funnel
  on aged accounts, so the axis fades exactly as it should.
- **Isolate the new randomness.** Every funnel draw comes from a separate RNG stream. The core media
  stream is independent of the funnel knobs, so adding the funnel provably could not move any
  existing behavioral result ([0008](0008-rng-isolation-for-additive-features.md)).

## Consequences

Per-population means (smoke preset, seed 7):

| population | completion rate | search rate | funnel completeness | account age (days) |
|---|---|---|---|---|
| naive ring | 1.00 | 0.00 | 0.33 | ~13 |
| sophisticated ring | 0.55 | 0.30 | 1.00 | ~394 |
| lurker (benign) | 0.11 | 0.02 | 0.78 | ~429 |
| legit | 0.11 | 0.30 | 0.99 | ~385 |

The axis is a strong independent discriminator at naive and moderate, and converges to legit at
sophisticated. `tests/test_actionmix.py` pins both halves, plus the legit-FP line with the axis on.

## References

- Lugoloobi et al., *Known By Their Actions: Fingerprinting LLM Browser Agents via UI Traces*, arXiv
  2605.14786, 2026.
- Kang & Kim, ETRI Journal 2023.
- Kadel et al., *BOTracle*, arXiv 2412.02266, 2024 — behavior from navigation shape; faking a human
  walk forces human pace, which destroys the bot's economics.
