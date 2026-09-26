"""Distribution log-prob / entropy correctness."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from dbn.reinforcement.common import (
    categorical_entropy,
    categorical_log_prob,
    gaussian_entropy,
    gaussian_log_prob,
)
from tests.references import (
    reference_categorical_entropy,
    reference_categorical_log_prob,
    reference_gaussian_entropy,
    reference_gaussian_log_prob,
)


def test_categorical_log_prob_matches_reference():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(16, 5)).astype(np.float32)
    actions = rng.integers(0, 5, size=(16,)).astype(np.int32)
    got = categorical_log_prob(mx.array(logits), mx.array(actions))
    mx.eval(got)
    expected = reference_categorical_log_prob(logits, actions)
    np.testing.assert_allclose(np.array(got), expected, rtol=1e-5, atol=1e-6)


def test_categorical_entropy_matches_reference():
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(16, 4)).astype(np.float32)
    got = categorical_entropy(mx.array(logits))
    mx.eval(got)
    expected = reference_categorical_entropy(logits)
    np.testing.assert_allclose(np.array(got), expected, rtol=1e-5, atol=1e-6)


def test_gaussian_log_prob_matches_reference():
    rng = np.random.default_rng(2)
    mean = rng.normal(size=(16, 3)).astype(np.float32)
    log_std = rng.normal(size=(3,)).astype(np.float32) * 0.1
    actions = rng.normal(size=(16, 3)).astype(np.float32)
    got = gaussian_log_prob(mx.array(mean), mx.array(log_std), mx.array(actions))
    mx.eval(got)
    expected = reference_gaussian_log_prob(mean, log_std, actions)
    np.testing.assert_allclose(np.array(got), expected, rtol=1e-5, atol=1e-6)


def test_gaussian_entropy_matches_reference():
    log_std = np.array([0.0, -0.5, 0.25], dtype=np.float32)
    got = gaussian_entropy(mx.array(log_std))
    mx.eval(got)
    expected = reference_gaussian_entropy(log_std)
    np.testing.assert_allclose(float(got), expected, rtol=1e-5, atol=1e-6)


def test_batch_one_and_many_shapes():
    logits = mx.zeros((1, 3))
    actions = mx.array([1], dtype=mx.int32)
    lp = categorical_log_prob(logits, actions)
    mx.eval(lp)
    assert lp.shape == (1,)

    logits = mx.zeros((8, 3))
    actions = mx.zeros((8,), dtype=mx.int32)
    lp = categorical_log_prob(logits, actions)
    mx.eval(lp)
    assert lp.shape == (8,)
