"""Core math: GAE, returns, PPO/SAC/TD3 losses vs NumPy references."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from dbn.reinforcement.common import (
    make_gae,
    normalize_advantages,
    ppo_policy_loss,
    ppo_ratio,
    ppo_value_loss,
    sac_actor_loss,
    sac_alpha_loss,
    sac_bellman_target,
    sac_critic_loss,
    soft_update,
    td3_bellman_target,
    td3_smooth_target_action,
)
from tests.references import (
    all_finite_tree,
    reference_gae,
    reference_ppo_policy_loss,
    reference_soft_update,
)


def test_gae_matches_reference():
    rng = np.random.default_rng(7)
    T, N = 16, 3
    rewards = rng.normal(size=(T, N)).astype(np.float32)
    values = rng.normal(size=(T, N)).astype(np.float32)
    dones = (rng.random(size=(T, N)) < 0.1).astype(np.float32)
    last_values = rng.normal(size=(N,)).astype(np.float32)
    gamma, lam = 0.99, 0.95

    expected_adv, expected_ret = reference_gae(
        rewards, values, dones, last_values, gamma, lam
    )
    gae_fn = make_gae(gamma, lam)
    zeros = np.zeros_like(dones)
    got_adv, got_ret = gae_fn(
        mx.array(rewards),
        mx.array(values),
        mx.array(dones),
        mx.array(zeros),
        mx.array(zeros),
        mx.array(last_values),
    )
    mx.eval(got_adv, got_ret)
    np.testing.assert_allclose(np.array(got_adv), expected_adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(got_ret), expected_ret, rtol=1e-5, atol=1e-6)


def test_gae_handles_all_dones():
    T, N = 8, 2
    rewards = np.ones((T, N), dtype=np.float32)
    values = np.zeros((T, N), dtype=np.float32)
    dones = np.ones((T, N), dtype=np.float32)
    last = np.zeros((N,), dtype=np.float32)
    expected_adv, expected_ret = reference_gae(rewards, values, dones, last, 0.99, 0.95)
    gae_fn = make_gae(0.99, 0.95)
    zeros = np.zeros_like(dones)
    got_adv, got_ret = gae_fn(
        mx.array(rewards), mx.array(values), mx.array(dones), mx.array(zeros), mx.array(zeros), mx.array(last)
    )
    mx.eval(got_adv, got_ret)
    np.testing.assert_allclose(np.array(got_adv), expected_adv, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(np.array(got_ret), expected_ret, rtol=1e-5, atol=1e-6)


def test_ppo_ratio_and_clipped_policy_loss():
    rng = np.random.default_rng(1)
    new_lp = rng.normal(size=(32,)).astype(np.float32)
    old_lp = rng.normal(size=(32,)).astype(np.float32)
    adv = rng.normal(size=(32,)).astype(np.float32)
    clip = 0.2

    ratio = ppo_ratio(mx.array(new_lp), mx.array(old_lp))
    mx.eval(ratio)
    np.testing.assert_allclose(np.array(ratio), np.exp(new_lp - old_lp), rtol=1e-5, atol=1e-6)

    loss = ppo_policy_loss(mx.array(new_lp), mx.array(old_lp), mx.array(adv), clip)
    mx.eval(loss)
    expected = reference_ppo_policy_loss(new_lp, old_lp, adv, clip)
    np.testing.assert_allclose(float(loss), expected, rtol=1e-5, atol=1e-6)


def test_ppo_clip_range_respected():
    # Extreme ratio should be clipped in surrogate
    new_lp = mx.array([10.0, -10.0])
    old_lp = mx.array([0.0, 0.0])
    adv = mx.array([1.0, 1.0])
    clip = 0.2
    ratio = np.array(ppo_ratio(new_lp, old_lp))
    assert ratio[0] > 1.0 + clip
    assert ratio[1] < 1.0 - clip
    loss = float(ppo_policy_loss(new_lp, old_lp, adv, clip))
    # ratio>>1 → clip to 1.2; ratio<<1 with A>0 → unclipped wins (≈0)
    # -mean([1.2, 0]) = -0.6
    np.testing.assert_allclose(loss, -0.6, rtol=1e-4, atol=1e-4)


def test_ppo_value_loss_unclipped_and_clipped():
    values = mx.array([1.0, 2.0, 3.0])
    returns = mx.array([1.5, 2.5, 2.0])
    unclipped = ppo_value_loss(values, returns)
    mx.eval(unclipped)
    expected = float(np.mean((np.array([1.5, 2.5, 2.0]) - np.array([1.0, 2.0, 3.0])) ** 2))
    np.testing.assert_allclose(float(unclipped), expected, rtol=1e-5, atol=1e-6)

    old = mx.array([1.0, 2.0, 3.0])
    clipped = ppo_value_loss(values, returns, old, clip_range_vf=0.1)
    mx.eval(clipped)
    assert np.isfinite(float(clipped))


def test_normalize_advantages():
    adv = mx.array([1.0, 2.0, 3.0, 4.0])
    norm = normalize_advantages(adv)
    mx.eval(norm)
    arr = np.array(norm)
    np.testing.assert_allclose(arr.mean(), 0.0, atol=1e-6)
    np.testing.assert_allclose(arr.std(), 1.0, atol=1e-5)


def test_sac_bellman_and_twin_q_min():
    rewards = mx.array([1.0, 0.5])
    dones = mx.array([0.0, 1.0])
    q1 = mx.array([2.0, 3.0])
    q2 = mx.array([1.5, 4.0])
    logp = mx.array([-0.1, -0.2])
    alpha = 0.2
    gamma = 0.99
    target = sac_bellman_target(rewards, dones, q1, q2, logp, alpha, gamma)
    mx.eval(target)
    # env0: uses min(q1,q2)=1.5; env1 done → just reward
    expected0 = 1.0 + gamma * (1.5 - alpha * (-0.1))
    expected1 = 0.5
    np.testing.assert_allclose(np.array(target), [expected0, expected1], rtol=1e-5, atol=1e-6)

    loss = sac_critic_loss(q1, q2, target)
    mx.eval(loss)
    assert np.isfinite(float(loss))


def test_sac_actor_and_alpha_loss():
    logp = mx.array([-1.0, -2.0])
    q1 = mx.array([0.5, 1.0])
    q2 = mx.array([0.4, 1.5])
    alpha = 0.2
    actor = sac_actor_loss(logp, q1, q2, alpha)
    mx.eval(actor)
    expected = float(np.mean(0.2 * np.array([-1.0, -2.0]) - np.minimum([0.5, 1.0], [0.4, 1.5])))
    np.testing.assert_allclose(float(actor), expected, rtol=1e-5, atol=1e-6)

    log_alpha = mx.array(0.0)
    aloss = sac_alpha_loss(log_alpha, logp, target_entropy=-1.0)
    mx.eval(aloss)
    assert np.isfinite(float(aloss))


def test_td3_target_smoothing_and_noise_clip():
    action = mx.array([[0.0], [1.0]])
    noise = mx.array([[10.0], [-10.0]])  # huge → must clip
    scale = mx.array([2.0])
    bias = mx.array([0.0])
    out = td3_smooth_target_action(
        action, noise, noise_clip=0.5, action_scale=scale, action_bias=bias
    )
    mx.eval(out)
    arr = np.array(out)
    # noise clipped to ±0.5 then * scale=2 → ±1.0; then action clipped to [-2, 2]
    np.testing.assert_allclose(arr[0], [1.0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(arr[1], [0.0], rtol=1e-5, atol=1e-6)
    assert np.all(arr >= -2.0 - 1e-6) and np.all(arr <= 2.0 + 1e-6)


def test_td3_bellman_uses_min_q():
    rewards = mx.array([1.0])
    dones = mx.array([0.0])
    target = td3_bellman_target(rewards, dones, mx.array([3.0]), mx.array([2.0]), 0.99)
    mx.eval(target)
    np.testing.assert_allclose(float(target), 1.0 + 0.99 * 2.0, rtol=1e-5, atol=1e-6)


def test_polyak_soft_update():
    class Tiny:
        def __init__(self, v):
            self._p = {"w": mx.array(v)}

        def parameters(self):
            return self._p

        def update(self, p):
            self._p = p

    target = Tiny(0.0)
    source = Tiny(1.0)
    soft_update(target, source, tau=0.1)
    mx.eval(target.parameters()["w"])
    expected = reference_soft_update(np.array(0.0), np.array(1.0), 0.1)
    np.testing.assert_allclose(float(target.parameters()["w"]), float(expected), rtol=1e-5, atol=1e-6)


def test_loss_gradients_are_finite():
    new_lp = mx.array(np.random.randn(8).astype(np.float32))
    old_lp = mx.array(np.random.randn(8).astype(np.float32))
    adv = mx.array(np.random.randn(8).astype(np.float32))

    def loss(nlp):
        return ppo_policy_loss(nlp, old_lp, adv, 0.2)

    grads = mx.grad(loss)(new_lp)
    mx.eval(grads)
    assert all_finite_tree(grads)
    assert grads.shape == new_lp.shape
