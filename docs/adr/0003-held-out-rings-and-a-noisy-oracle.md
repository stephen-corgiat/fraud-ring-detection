# 0003 — Held-out rings, and an oracle that can be wrong

**Status:** accepted

## Context

Two questions decide whether a fraud loop is worth running, and both are unanswerable without labels:

1. Does a rule learned from known rings catch a ring it has **never seen**?
2. How many human-confirmed labels does it **cost** to get there — and what happens when the humans
   are wrong some of the time?

Row-level k-fold answers neither. Members of one ring share infrastructure and behavior, so a random
split leaks the ring across train and test and grades memorization.

## Decision

- **Leave-one-ring-out.** The learned rule (`eval/supervised.py`, logistic regression over a
  weak-signal basket) is always graded on a whole ring held out of training. Features come from
  events only.
- **The headline excludes the near-perfect tell.** Active-challenge failure would make the problem
  trivial; it is reported only as an ablation (`+chal`), so the grade measures the weak signals.
- **A fixed success bar,** decided before looking: operational precision ≥ 0.90, recall ≥ 0.80,
  legit false positives ≤ 5%, on held-out rings, per seed (`n_seeds_meeting`) as well as on average.
- **A simulated oracle with two error rates** (`eval/oracle.py`): missed fraud (fraud → legit) and
  false confirmation (legit → fraud), seeded and reproducible. Noise corrupts **training labels only**;
  grading always uses clean truth.
- **The learning curve** (`eval/curve.py`): metric vs. label budget on a frozen candidate pool, with
  `random`, `uncertainty` and `exploit` query strategies. **Labels-to-target** is the *stable*
  crossing — the smallest budget that clears the bar and keeps clearing it.
- **A continuity anchor:** at zero noise and full budget, a curve point must reproduce the grade
  bit-for-bit. A test enforces it, so the curve can never silently drift from the grade it extends.

## Consequences

At default scale over seeds 7, 13, 21, the learned rule clears the bar at every level on all three
seeds:

| level | recall | operational precision | legit FP | hand-weighted recall@k | lift |
|---|---|---|---|---|---|
| naive | 0.99 | 1.00 | 0.00 | 1.00 | −0.01 |
| moderate | 1.00 | 1.00 | 0.00 | 0.78 | +0.22 |
| sophisticated | 1.00 | 0.96 | 0.04 | 0.37 | +0.63 |

The learning curve (`fraudrings curve`, sophisticated level, default preset, seeds 7, 13, 21) puts a
price on it. Labels-to-target with random sampling and a perfect oracle:

| level | labels to clear the bar and stay cleared |
|---|---|
| naive | 10 |
| moderate | 80 |
| sophisticated | 80 |

What the noisy oracle made measurable:

- **Precision is the fragile constraint.** At 5% symmetric label error the stable crossing holds at
  80 labels. At 10%, recall recovers by ~320 labels but the legit-FP line sits *on* the 5% bar
  (0.04–0.05), so the stable crossing slips to the full pool. The rare false confirmations
  (legit → fraud) are what push it over.
- **Missed fraud is the benign direction.** With 20% of fraud labels flipped to legit and no false
  confirmations (`fraudrings curve --asymmetric`), the rule still reaches the stable bar, at 160
  labels instead of 80. It costs labels, not the target. That matters because it is the realistic
  error: accounts that age quietly look legit until they act.
- **Active learning helps first, not durably.** Uncertainty sampling first clears the bar at 40
  labels, half of random's 80, but its legit-FP line wobbles back over 5% at larger budgets, so its
  *stable* crossing comes much later. It also loses to random at smoke scale. The tests assert only
  that it converges to random at full budget, not that it wins.

## References

- Elkan & Noto, *Learning Classifiers from Only Positive and Unlabeled Data*, KDD 2008.
- Northcutt, Jiang & Chuang, *Confident Learning*, JAIR 2021.
- Arp et al., USENIX Security 2022 — pitfalls P1 (sampling bias) and P3 (data snooping).
- `tests/test_supervised.py`, `tests/test_grade.py`, `tests/test_curve.py`, `tests/test_oracle.py`.
