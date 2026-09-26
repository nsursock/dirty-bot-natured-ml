"""Network forward shapes and gradient finiteness."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from dbn.reinforcement.common import (
    DeterministicActor,
    MlpPolicy,
    SquashedGaussianActor,
    TwinQ,
)
from tests.references import all_finite_tree


def test_mlp_policy_discrete_shapes():
    policy = MlpPolicy(4, 2, continuous=False, net_arch=(16, 16))
    obs = mx.zeros((8, 4))
    logits, values = policy(obs)
    mx.eval(logits, values)
    assert logits.shape == (8, 2)
    assert values.shape == (8,)


def test_mlp_policy_continuous_shapes():
    policy = MlpPolicy(3, 1, continuous=True, net_arch=(16,))
    obs = mx.zeros((5, 3))
    mean, values = policy(obs)
    mx.eval(mean, values, policy.log_std)
    assert mean.shape == (5, 1)
    assert values.shape == (5,)
    assert policy.log_std.shape == (1,)


def test_mlp_policy_batch_one():
    policy = MlpPolicy(4, 2, continuous=False)
    logits, values = policy(mx.zeros((1, 4)))
    mx.eval(logits, values)
    assert logits.shape == (1, 2)
    assert values.shape == (1,)


def test_twin_q_shapes():
    q = TwinQ(3, 1, net_arch=(8, 8))
    q1, q2 = q(mx.zeros((4, 3)), mx.zeros((4, 1)))
    mx.eval(q1, q2)
    assert q1.shape == (4,)
    assert q2.shape == (4,)


def test_actors_output_in_bounds():
    scale = mx.array([2.0])
    bias = mx.array([0.0])
    det = DeterministicActor(3, 1, net_arch=(8,), action_scale=scale, action_bias=bias)
    squ = SquashedGaussianActor(3, 1, net_arch=(8,), action_scale=scale, action_bias=bias)
    obs = mx.random.normal(shape=(16, 3))
    a_det = det(obs)
    a_stoch, logp = squ(obs, mx.random.key(0), deterministic=False)
    a_mean, _ = squ(obs, None, deterministic=True)
    mx.eval(a_det, a_stoch, a_mean, logp)
    for a in (a_det, a_stoch, a_mean):
        arr = np.array(a)
        assert arr.shape == (16, 1)
        assert np.all(arr >= -2.0 - 1e-5) and np.all(arr <= 2.0 + 1e-5)
    assert logp.shape == (16,)


def test_policy_gradients_finite_and_shaped():
    from dbn.reinforcement.common import tree_flatten

    policy = MlpPolicy(4, 2, continuous=False, net_arch=(8,))
    obs = mx.random.normal(shape=(4, 4))

    def loss(model):
        logits, values = model(obs)
        return mx.mean(logits**2) + mx.mean(values**2)

    loss_and_grad = nn.value_and_grad(policy, loss)
    l, grads = loss_and_grad(policy)
    mx.eval(l)
    assert np.isfinite(float(l))
    assert all_finite_tree(grads)
    g_leaves = tree_flatten(grads)
    p_leaves = tree_flatten(policy.parameters())
    assert len(g_leaves) == len(p_leaves)
    for g, p in zip(g_leaves, p_leaves):
        assert g.shape == p.shape
