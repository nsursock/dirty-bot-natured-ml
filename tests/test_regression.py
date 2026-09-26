"""Regression: minimal reproductions for fixed bugs.

Add a test here whenever a bug is discovered — keeps AI edits from resurfacing.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from dbn.reinforcement.common import make_gae
from dbn.reinforcement.losses import ppo_ratio


def test_regression_gae_returns_equal_adv_plus_values():
    """Invariant: returns = advantages + values (bootstrap GAE definition)."""
    T, N = 5, 2
    rewards = np.random.randn(T, N).astype(np.float32)
    values = np.random.randn(T, N).astype(np.float32)
    dones = np.zeros((T, N), dtype=np.float32)
    last = np.zeros((N,), dtype=np.float32)
    adv, ret = make_gae(0.99, 0.95)(
        mx.array(rewards), mx.array(values), mx.array(dones), mx.array(last)
    )
    mx.eval(adv, ret)
    np.testing.assert_allclose(np.array(ret), np.array(adv) + values, rtol=1e-5, atol=1e-6)


def test_regression_ratio_identity_when_logprobs_equal():
    lp = mx.array([0.1, -0.2, 0.3])
    ratio = ppo_ratio(lp, lp)
    mx.eval(ratio)
    np.testing.assert_allclose(np.array(ratio), np.ones(3), rtol=1e-5, atol=1e-6)
