"""The oracle labeler, with a tunable error rate.

A perfect oracle reads the manufactured ground truth and returns it verbatim (it lives in `eval/`,
the only place allowed to read the answer key). The learning curve (`eval/curve.py`) asks a
sharper question: how robust is the loop when the oracle is **wrong** some of the time, as a
real bench of human analysts inevitably is?

We model oracle error as a **two-rate label flip**:
  * `fraud_to_legit` — a true ring member the analyst *misses* (clears as legit). The realistic,
    dominant error in this domain: accounts that age quietly look legit until they act.
  * `legit_to_fraud` — an innocent account the analyst *wrongly confirms* as fraud.

The **headline sweep is symmetric** (`symmetric_noisy_labels`, both rates = ε ∈ {0, .05, .10, .20})
so the curve reads as one knob; the two-rate form is kept for the realistic asymmetric story.

Noise corrupts **training labels only** — the held-out grading in `curve.py` always uses true
ground truth. And the noise is fully reproducible: it is a deterministic function of (labels, rates,
seed) via a seeded numpy Generator — no wall-clock. At ε=0 it is the identity, which is what lets
the curve's top point reconcile exactly with the grade.
"""

from __future__ import annotations

import numpy as np


def noisy_labels(
    labels: np.ndarray, *, fraud_to_legit: float, legit_to_fraud: float, seed: int
) -> np.ndarray:
    """Return a copy of binary `labels` (0/1) with class-conditional flips.

    Each true-fraud (1) is flipped to 0 with probability `fraud_to_legit`; each true-legit (0) is
    flipped to 1 with probability `legit_to_fraud`. Deterministic given `seed`; never mutates the
    input. Both rates 0 ⇒ identity.
    """
    if not (0.0 <= fraud_to_legit <= 1.0 and 0.0 <= legit_to_fraud <= 1.0):
        raise ValueError(f"flip rates must be in [0,1] (got {fraud_to_legit}, {legit_to_fraud})")
    out = np.asarray(labels, dtype=int).copy()
    if fraud_to_legit == 0.0 and legit_to_fraud == 0.0:
        return out
    # One uniform draw per element. Each element belongs to exactly one class, so comparing that
    # element's draw to its own class's rate yields the correct independent per-element flip
    # probability while keeping the whole vector a single deterministic RNG sequence.
    u = np.random.default_rng(seed).random(out.shape[0])
    # Mask on the ORIGINAL classes, not the partially-mutated output — otherwise a fraud just
    # flipped to 0 would be a candidate for the 0→1 flip and (with equal rates) flipped right back.
    is_fraud, is_legit = out == 1, out == 0
    out[is_fraud & (u < fraud_to_legit)] = 0
    out[is_legit & (u < legit_to_fraud)] = 1
    return out


def symmetric_noisy_labels(labels: np.ndarray, *, epsilon: float, seed: int) -> np.ndarray:
    """The headline knob: flip in either direction with the same probability ε. ε=0 is identity."""
    return noisy_labels(labels, fraud_to_legit=epsilon, legit_to_fraud=epsilon, seed=seed)
