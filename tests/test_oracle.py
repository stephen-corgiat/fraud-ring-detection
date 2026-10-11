"""The (im)perfect oracle.

The learning curve asks "how many oracle labels to reach the bar?" — and then "how robust is
that when the oracle is *wrong* sometimes?" These pin the noisy labeler that answers the second
question:

  * a two-rate flip model (fraud→legit, legit→fraud) so we can model the realistic asymmetry as
    well as the symmetric headline sweep;
  * fully reproducible — identical seed ⇒ identical output (no wall-clock);
  * ε=0 is the identity (the continuity anchor: at zero noise the curve must reproduce the
    grade exactly).
"""

from __future__ import annotations

import numpy as np

from fraudrings.eval.oracle import noisy_labels, symmetric_noisy_labels


def test_zero_rate_is_identity() -> None:
    labels = np.array([0, 1, 1, 0, 1, 0, 0, 1], dtype=int)
    out = symmetric_noisy_labels(labels, epsilon=0.0, seed=7)
    assert np.array_equal(out, labels)
    # the two-rate form with both rates zero is also identity
    assert np.array_equal(noisy_labels(labels, fraud_to_legit=0.0, legit_to_fraud=0.0, seed=7), labels)


def test_identical_seed_identical_output() -> None:
    labels = (np.arange(1000) % 2).astype(int)
    a = symmetric_noisy_labels(labels, epsilon=0.1, seed=13)
    b = symmetric_noisy_labels(labels, epsilon=0.1, seed=13)
    assert np.array_equal(a, b)
    # a different seed gives a (statistically certain to be) different flip pattern
    c = symmetric_noisy_labels(labels, epsilon=0.1, seed=14)
    assert not np.array_equal(a, c)


def test_does_not_mutate_input() -> None:
    labels = np.array([0, 1, 0, 1], dtype=int)
    snapshot = labels.copy()
    symmetric_noisy_labels(labels, epsilon=0.5, seed=1)
    assert np.array_equal(labels, snapshot)


def test_symmetric_flip_rate_is_statistically_correct() -> None:
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 2, size=200_000)
    eps = 0.1
    out = symmetric_noisy_labels(labels, epsilon=eps, seed=99)

    frauds = labels == 1
    legits = labels == 0
    fraud_flip = float((out[frauds] != labels[frauds]).mean())
    legit_flip = float((out[legits] != labels[legits]).mean())
    assert abs(fraud_flip - eps) < 0.01, fraud_flip
    assert abs(legit_flip - eps) < 0.01, legit_flip


def test_two_rate_only_flips_the_named_direction() -> None:
    rng = np.random.default_rng(1)
    labels = rng.integers(0, 2, size=200_000)
    # only confirm-as-fraud noise on legits; frauds are never lost
    out = noisy_labels(labels, fraud_to_legit=0.0, legit_to_fraud=0.2, seed=5)
    frauds = labels == 1
    legits = labels == 0
    assert np.array_equal(out[frauds], labels[frauds]), "frauds must be untouched at fraud_to_legit=0"
    legit_flip = float((out[legits] != labels[legits]).mean())
    assert abs(legit_flip - 0.2) < 0.01, legit_flip

    # ...and the mirror: only missed-fraud noise, legits untouched
    out2 = noisy_labels(labels, fraud_to_legit=0.3, legit_to_fraud=0.0, seed=5)
    assert np.array_equal(out2[legits], labels[legits])
    fraud_flip = float((out2[frauds] != labels[frauds]).mean())
    assert abs(fraud_flip - 0.3) < 0.01, fraud_flip
