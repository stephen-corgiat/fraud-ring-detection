# Measuring a fraud-ring detector when you have no labels

Coordinated fraud has a measurement problem before it has a modeling problem. Confirmed fraud is
scarce, comes from whatever an investigator happened to look at, and arrives with no matching set of
confirmed-honest accounts. "Our detector flags rings" is easy to say and nearly impossible to check.
Most of this repository is the machinery for checking it honestly. This page explains that machinery
end to end. The decision records in [`docs/adr/`](../adr/) hold the details.

## 1. Build a world you know the answer to

The simulator (`fraudrings.generator`) produces a platform: accounts, creators, items, and a
time-ordered event stream of searches, views, completions, skips, saves, registrations and logins,
each carrying session, IP, subnet, ASN, fingerprint and the client-reported progress that payouts
depend on. A world is fully determined by a config and a seed, so every number in this repository
can be regenerated exactly.

Into that world it injects **fraud rings** — groups of accounts run by one operator to inflate one
target creator's qualified views. It also injects the populations that make a naive detector look
good and a real one earn its keep:

| population | why it exists |
|---|---|
| **superfan cohorts** | genuine fans of the very creators rings target: concentrated, coordinated-looking, honest |
| **lurkers** | passive viewers with truncated funnels: no searches, no saves, high completion, and honest |
| **bandwagon crowds** | benign accounts that pile onto a lock-step ring's burst window |
| **household and VPN sharers** | legit accounts that coincidentally share infrastructure |

The answer key goes to a separate file the detector never opens
([ADR 0001](../adr/0001-ground-truth-is-physically-separate.md)).

## 2. Make the adversary a dial

One world proves nothing about a detector, because a detector can be tuned to one world. So every
world is generated at one **evasion level**, and one small table turns that level into concrete
behavior:

- **naive** — shared datacenter IPs, metronomic timing, perfect completion spoofs, a direct path to
  the target, freshly minted accounts;
- **moderate** — partial rotation and jitter, some cover traffic;
- **sophisticated** — residential IPs pinned per account, human-like timing, cover traffic across
  other creators, a faked search-and-save funnel, aged accounts, and the same total yield split
  across many more low-volume accounts.

Results are always reported as **curves over the dial**
([ADR 0002](../adr/0002-adversary-is-a-dial.md)). The point of a benchmark is to show *where a
method breaks*, not to produce one flattering number.

## 3. Report the number that matters, not the one that's easy

Average precision per ring is the standard ranking metric. On its own it is misleading here, because
it can't tell two very different failures apart:

- flagging a member of **another ring** — still fraud, still worth actioning;
- flagging a **genuine user** — the false positive that actually costs you.

So every investigation's flagged set is split three ways — this ring, other-ring fraud, legit — and
reported as **operational precision**, **legit false-positive rate** and **cross-ring bleed** next to
the ranking curve. At the sophisticated level, per-ring AP drops to about 0.5 while operational
precision stays around 0.95. The loss is almost entirely rings being confused with each other.

## 4. Grade on rings the model never saw

The learned rule is a logistic regression over weak signals: cover-view breadth, timing spread,
target concentration, residual linkage. Its coefficients are inspectable. It is trained with
**leave-one-ring-out** cross-validation, because members of one ring share infrastructure and
behavior, and a row-level split would grade memorization. The headline also excludes the one
near-perfect tell (failing an active challenge), which appears only as an ablation. The weak signals
have to carry the result on their own.

The success bar was fixed before looking: **operational precision ≥ 0.90, recall ≥ 0.80, legit false
positives ≤ 5%**, met per seed, not just on average. The learned rule clears it at every level on
every seed, lifting recall at the sophisticated level from 0.37 (hand-weighted ranking) to 1.00
([ADR 0003](../adr/0003-held-out-rings-and-a-noisy-oracle.md)).

## 5. Price the human

A loop that needs labels has to answer *how many*, and *what if the labeler is wrong*. The oracle in
`fraudrings.eval.oracle` flips training labels at two independent, seeded rates: missed fraud and
false confirmations. Grading always uses clean truth. The learning curve then sweeps the label budget
and reports **labels-to-target**: the smallest budget that clears the bar and *keeps* clearing it.
With a perfect oracle that is 10 labels at naive and 80 at sophisticated. Under noise, precision is
the fragile constraint: the legit-FP line drifts onto the 5% bar long before recall suffers.

A continuity test pins the whole construction. At zero noise and full budget, a curve point must
reproduce the held-out grade bit for bit.

## 6. Find somewhere to start

Seeds have to come from somewhere. The seedless ranker (`fraudrings.detector.concentration`) reads
only events and shortlists creators whose qualified views *concentrate*: on few accounts, on many
dedicated accounts, or inside one week. It outputs **suspects, not verdicts**, each tagged with its
observed shape. It is graded the same way as everything else: where do the creators the rings were
paid to boost land on the list, at each evasion level
([ADR 0006](../adr/0006-seedless-concentration-as-the-front-door.md))?

## 7. Write down what didn't work

The simulator also builds **lock-step worlds**, where a ring is an operator with a hard concurrency
limit. The cap is the *only* difference between the ring and a benign bandwagon on the same creator
in the same window. Serialization is real in those worlds, but no event-derived statistic recovers
it where it would matter. A human reading per-account rasters can confirm it at low concurrency.
That result is recorded, with its numbers, as a negative finding
([ADR 0004](../adr/0004-lockstep-serialization-is-not-recoverable.md)), and the raster ships as an
analyst aid rather than a feature.

## 8. Keep the experiment honest as it grows

A simulator that changes under its own tests is worthless. New stochastic features draw from
**isolated random streams**, so adding a population or event type cannot move any existing result
([ADR 0008](../adr/0008-rng-isolation-for-additive-features.md)). Tests pin the measured claims
across the dial, so a regression in the method shows up as a failing test, not a quietly worse number.

## What this does not claim

A simulator encodes its author's threat model. These results describe how the method behaves against
*that* model, made explicit and adjustable by the dial. They don't say how often each adversary tier
appears in the wild, and they are not a substitute for measuring base rates on real traffic. The
approach was also exercised against real production traffic; what that taught is written up,
generically, in [lessons.md](lessons.md).
