# Decisions

The design decisions behind the simulator, detector and evaluation, written down as they were made:
the context, what was decided, and what it costs. Most of them protect one thing, the honesty of
the measurements. [Measuring a fraud-ring detector when you have no labels](../research/method.md)
shows how they fit together.

| # | Decision | In one line |
|---|---|---|
| 0001 | [Ground truth is physically separate](0001-ground-truth-is-physically-separate.md) | The detector can't read the answer key by any path, so the scores measure the method, not the plumbing. |
| 0002 | [The adversary is a dial](0002-adversary-is-a-dial.md) | Every result is a curve over how hard the rings try to look organic, never one number. |
| 0003 | [Held-out rings, and a noisy oracle](0003-held-out-rings-and-a-noisy-oracle.md) | Learn from some rings, score on others, and price the labels, including when the humans are wrong. |
| 0004 | [Lock-step serialization is not recoverable](0004-lockstep-serialization-is-not-recoverable.md) | A negative result: the lock-step generator ships, but it isn't recoverable from timing, so no detector feature does. |
| 0005 | [What over when](0005-what-over-when.md) | What an account does is costlier to fake than when it does it, so action mix beats timing. |
| 0006 | [Seedless concentration as the front door](0006-seedless-concentration-as-the-front-door.md) | Triage starts from the event stream alone and offers suspects, not verdicts. |
| 0007 | [One chart library](0007-plotly-single-chart-library.md) | Plotly for every view, and charts never feed scoring. |
| 0008 | [Isolated RNG streams](0008-rng-isolation-for-additive-features.md) | New generator features draw from their own stream, so pinned results don't move. |
