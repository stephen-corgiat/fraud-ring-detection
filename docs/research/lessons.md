# Lessons

Working rules distilled from building this harness, reading the literature
([literature-review.md](literature-review.md)), and exercising the approach against real production
traffic. They are written to apply to any platform with coordinated abuse; nothing here depends on
the simulator's domain.

## About signals

**1. The harness travels; features don't.**
Features are tied to their platform: what a "qualified" action is, which identifiers exist, what
honest behavior looks like. What carries over to a new abuse type is the machinery: a separated
answer key, an adversary dial, held-out groups, a noisy oracle, and a label store that records how
each label was selected. Build that first. A new domain then means new features plugged into the
same harness, not a new project.

**2. What over when.**
Rank every proposed signal on a durability ladder. *Timing* statistics are the cheapest for an
adversary to perturb; one line of random delay defeats them. *Spatial and structural* choices — which
infrastructure, which items, which creators — sit in the middle. The *composition* of what an
account does is the most durable, because faking it means doing the work. Timing can be a
precision-first floor, never the strategy ([ADR 0005](../adr/0005-what-over-when.md)).

**3. Audit perturbation durability before trusting a feature.**
Before claiming a behavioral feature works, simulate the adversary's *cheapest* perturbation of it,
retrain, and report what still carries the performance. If the answer is "nothing", you measured the
adversary's laziness, not the feature.

**4. Prefer signals bound to physics and money.**
An adversary can hide *who* it is cheaply, and *what* it is less cheaply. It can't hide costs it must
actually pay: bandwidth that must be moved, latency that betrays location, payouts that must land
somewhere. The unit that collects the money is often a better unit of detection than the accounts
that generate it.

**5. Coherence is a collection-design question.**
When you design telemetry, ask which independently emitted claims you can require to agree (protocol
layers, client-reported progress versus server-observed transfer), and which physically entailed
event cascades you can require to be complete. Rules built on coherence last; signatures of specific
tools are perishable.

## About evidence

**6. Stress-test base rates before trusting a rate.**
A feature that separates classes on a balanced sample can drown at the real prevalence. Plug in the
base rate before celebrating. A signal that is "3× more common in fraud" is useless if honest users
outnumber fraud a thousand to one.

**7. Don't report balanced AUC as the headline.**
Report the operating point a human would act on: precision at the flagged set, legit false positives,
and cross-group bleed, at realistic class balance.

**8. Every number carries its label provenance.**
"Precision against analyst labels", "precision against datacenter-IP proxies" and "precision per
vendor telemetry" are three different claims. State which one you mean. Vendor numbers are
directional, never evidence.

**9. Split by whole identifiers, never by rows.**
Any evaluation of identifier-level or group-level features holds out whole groups — rings, subnets,
devices, accounts. Rows from one group straddling train and test grade memorization
([ADR 0003](../adr/0003-held-out-rings-and-a-noisy-oracle.md)).

**10. Census the benign look-alikes first.**
Before a rule ships on a surface, measure the honest population that mimics the positive class:
password managers on login forms, accessibility tools on input, devoted fans on a creator, passive
viewers on a funnel. Every surface has one, and the ranked queue won't reveal it. A human working
through cases will.

**11. Under-linkage is never exculpatory.**
Failing to find shared infrastructure doesn't clear an account; sophisticated operators are built to
leave none. And IP-derived evidence should only discount suspicion, never convict on its own. Shared
IPs are often carrier NAT, and proxy exits are real subscribers' devices.

## About decisions

**12. Suspects, not verdicts.**
A coordinated ring and a devoted fan base can produce identical contribution shapes. Rank, describe
the observed shape, show the evidence, and let a human decide
([ADR 0006](../adr/0006-seedless-concentration-as-the-front-door.md)).

**13. Precision-first floors are a tax, not a strategy.**
A cheap detector tuned for zero false positives at modest recall is worth running against the naive
tier. Document it as evadable by one line. Its value is forcing the adversary to write that line.

**14. Score the economy before the graph.**
Before building a graph detector, spend thirty minutes on the platform's economics. Does the payout
rule force fraud into *width* (many accounts, each capped), or let a few *stars* carry it? Is fraud
sold as a service, with shared account pools? The answers decide whether dense-block methods can
work at all.

**15. Report what didn't work, with its numbers.**
A negative result with a reproducible measurement is worth more than a positive one without it. The
lock-step finding ([ADR 0004](../adr/0004-lockstep-serialization-is-not-recoverable.md)) keeps the
next person from spending a month on the same wall.

**16. Pre-register the bar.**
Fix the success criteria and kill conditions before looking at results. A gate that can be moved
after the fact isn't a gate.
