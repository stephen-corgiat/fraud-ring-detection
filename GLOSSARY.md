# Glossary

The project's shared language: the terms code, docs and decision records use consistently. The
domain is a **content platform where accounts engage with items published by creators, and creators
are paid per qualified engagement** — the shape behind view farming, click fraud, creator-fund
abuse and fake engagement generally.

This is a glossary only: definitions, not implementation. Where a term has a tempting synonym that
means something subtly different, the synonym is listed under _Avoid_.

## The world

**World**:
One seeded, simulated platform: accounts, creators, items, a time-ordered event stream, and a
separate ground-truth file. Fully determined by its config and seed.
_Avoid_: dataset, scenario

**Creator**:
The account that publishes items and receives payouts for qualified views of them. The unit a ring
is paid to boost.
_Avoid_: owner, channel

**Item**:
One piece of content a creator publishes. Views happen on items; payouts accrue to creators.
_Avoid_: post, asset

**Ring**:
The true set of accounts one fraud operator controls and coordinates to inflate a target creator's
payouts. Generally only partly observable: a careful ring's cover traffic looks like ordinary diffuse
viewing.
_Avoid_: cluster, group, network

**Operator**:
The person or organization behind a ring. One operator runs many accounts; how many it can drive at
once is its *concurrency*.
_Avoid_: fraudster, attacker, bad actor

**Superfan cohort**:
A benign group of genuine fans who concentrate their viewing on one creator — often the very creator
a ring targets. The hardest negative: behaviorally close to a ring, but honest.
_Avoid_: fan club (fine in prose, not as a defined term), whale

**Lurker**:
A benign passive viewer who rarely searches or saves and completes most of what autoplay feeds
them. Funnel-truncated without being fraud, so "truncated funnel ⇒ fraud" can never be a free
signal.
_Avoid_: bot-like user

**Bandwagon**:
A benign crowd that piles onto the same creator in the same burst window as a lock-step ring,
uncapped. The negative control for any timing-overlap signal.
_Avoid_: flash crowd (fine in prose)

**Evasion level**:
How hard every ring in a world tries to look organic: `naive`, `moderate` or `sophisticated`.
Resolved into concrete knobs (infra rotation, timing jitter, spoof fidelity, funnel faking, account
aging, fixed-yield splitting) by one table. One level per world.
_Avoid_: difficulty, sophistication score

**Lock-step world**:
A world where each ring is a finite operator with C concurrent lanes, so at most C members are
active at once and they take turns. Built to test whether serialization is detectable.
_Avoid_: synchronized ring (they are serialized, not synchronized)

## Events

**Event**:
One detector-visible record: timestamp, account, session, infrastructure (IP, subnet, ASN,
fingerprint), action, item, creator, reported position, skip flag, challenge result.
_Avoid_: log line, hit

**Media event**:
Any consumption event — `view`, `complete` or `skip`. The behavioral basket is computed over media
events only, so funnel events cannot perturb it.
_Avoid_: hit, impression

**View**:
A non-skipped media event. A view at or above the payout threshold is a **qualified view**.
_Avoid_: hit, impression

**Qualified view** (qv):
A view whose client-reported position clears the payout threshold — the unit creators are paid on.
Spoofable: the position is reported by the client.
_Avoid_: valid view, paid view

**Funnel**:
The action sequence `search → view → complete → save`. Engaged legit viewers exercise all of it; a
naive ring views the target directly. Its shape is the action-mix axis.
_Avoid_: journey, flow

**Payout cap**:
The per-(account, day) ceiling on qualified views that earn money. **Capped qv** sums
`min(daily qv, cap)` — payout-relevant harm, rather than raw volume.
_Avoid_: rate limit

## The seeded engine (find the rest of the ring)

**Seed**:
An account the investigator already suspects, handed to the engine as input. Seeds are inputs, not
results.
_Avoid_: label, positive

**Stage 1 — retrieval**:
Seed-anchored expansion over shared identifiers, rarity-gated: only identifiers shared by few enough
accounts count as links. Sets the recall ceiling — a member never retrieved can never be ranked.
_Avoid_: candidate generation (that is Stage 1b specifically)

**Rarity gate**:
The Stage 1 rule that an identifier links accounts only if between 2 and a cap of accounts share it.
A coffee-shop IP shared by hundreds is not evidence; a datacenter IP shared by twelve is.
_Avoid_: threshold (ambiguous)

**Stage 1b — co-target candidate generation**:
Widened retrieval on the rare target creator the seeds qualified-viewed *in common*, time-localized
to the campaign window. The economic invariant a ring cannot escape: it exists to pump its target.
_Avoid_: behavioral retrieval

**Seed cohesion**:
The fraction of seeds that interlink through a rarity-gated identifier. High at naive (shared
datacenter infra), low at sophisticated (residential proxies). The cue to widen with co-target.
_Avoid_: confidence

**Stage 2 — scoring**:
Seed-relative evidence score over four axes — linkage, behavioral, co-target, action-mix — with
hand-set weights; or the learned rule.
_Avoid_: classification (it ranks; it does not decide)

**Linkage**:
Summed rarity (IDF) of the identifiers a candidate shares with the seeds. The primary evidence at
naive; dead at sophisticated.
_Avoid_: similarity

**Behavioral** (axis):
Proximity to the seed centroid in standardized behavioral-feature space (skip rate, cadence, active
hours, creator breadth, infra spread).
_Avoid_: anomaly score

**Co-target** (axis):
How much a candidate's viewing overlaps the seeds' target — repertoire overlap (Jaccard, the
default) or devotion to the target (single-target share, opt-in).
_Avoid_: collaborative filtering

**Action-mix** (axis):
Proximity to the seed centroid over funnel shape and account age. Strong at naive/moderate; fades
as a sophisticated ring fakes the funnel and ages its accounts.
_Avoid_: engagement score

**Learned rule**:
Logistic regression over the weak-signal basket, trained with leave-one-ring-out cross-validation
on oracle labels. Its coefficients are the rule.
_Avoid_: the model (ambiguous)

## The seedless ranker (find a place to start)

**Target**:
The creator a ring boosts — and the unit the seedless ranker scores.
_Avoid_: victim

**Entry axis**:
One of three independent ways a target reaches the suspect shortlist: contributor concentration
(`n_eff`), booster count, or temporal concentration (`burst7`). Extreme on *any* axis qualifies;
each catches an archetype the others miss.
_Avoid_: feature, signal (when you mean one of the three admission axes)

**n_eff**:
A target's effective number of contributors — inverse-HHI of its qualified views across accounts.
Low `n_eff` ⇒ a few accounts supply most views.
_Avoid_: diversity, HHI (bare)

**Booster**:
An account dedicated to one target: high single-target share at volume. A neutral suspect role, not
a proven bot — its benign twin is the superfan. Counted per target as **booster count**.
_Avoid_: bot, fan

**Single-target share**:
The account-level measure behind "booster": the account's qualified views on this target ÷ its
total qualified views.
_Avoid_: loyalty

**burst7**:
A target's largest share of qualified views falling in any one 7-day window. High `burst7` ⇒ a
time-localized push (the `campaign` tag).
_Avoid_: burstiness, spike

**Core**:
A target's boosters — the concentration-detectable, dedicated subset of a ring, and what triage
surfaces. **The ranker detects cores, not whole rings.**
_Avoid_: cohort, clique

**Topology tag**:
A description of a target's observed contributor shape — `single-source`, `few-source`,
`multi-source`, `campaign`. Coordination is *inferred* only at `multi-source` (many dedicated
contributors). Tags describe; they never decide.
_Avoid_: verdict, label

**Suspect**:
A ranked target on the shortlist. The ranker emits **suspects for human review**, never verdicts.
_Avoid_: detection, positive

**Impact floor**:
The minimum qualified-view volume for a target to enter the shortlist at all. Below it, the
concentration signal is noise and the payout harm is negligible.
_Avoid_: threshold (ambiguous)

**Expansion gate**:
Hand a suspect's core to the seeded engine only when the core interlinks on rare infrastructure.
Without shared infra, expansion fans out into noise and the core itself is the output.
_Avoid_: auto-expand

## Measurement

**Ground truth**:
The hidden per-account labels a world is generated with (`is_fraud`, `ring_id`, evasion level,
bandwagon cohort). Written to a separate file the detector never opens.
_Avoid_: labels (labels are what a human or oracle produces)

**Anti-tautology**:
The rule that no signal may be measured against the thing that generated it. The detector reads
only events; ground truth lives on the eval side; benign look-alikes exist so that no single cue is a
free answer. The project's central methodological commitment.
_Avoid_: leakage prevention (narrower)

**Oracle**:
The labeler the eval harness simulates — perfect by default, or with seeded two-rate label noise
(missed fraud, false confirmations). Corrupts training labels only; grading is always against clean
truth.
_Avoid_: annotator

**Leave-one-ring-out**:
Cross-validation that holds out whole rings, so the rule is always graded on a ring it never saw.
The honest version of "can we catch a new ring?"
_Avoid_: k-fold (rows from one ring must never straddle train and test)

**Operational precision**:
Of the accounts an investigation would action, the fraction that are fraud in *any* ring.
**Cross-ring bleed** — flagging a member of another ring — costs per-ring ranking, not operational
precision.
_Avoid_: precision (bare, when you mean the per-ring number)

**Legit false-positive rate**:
The fraction of actioned accounts that are genuine users. The line that actually matters
operationally.
_Avoid_: FPR (ambiguous)

**Labels-to-target**:
The smallest oracle-label budget at which the learned rule clears the success bar *and keeps
clearing it* for every larger budget tested.
_Avoid_: sample efficiency (vague)

**Success bar**:
Precision ≥ 0.90 (operational), recall ≥ 0.80, legit false positives ≤ 5%, on held-out rings.
_Avoid_: target (overloaded — a target is a creator)
