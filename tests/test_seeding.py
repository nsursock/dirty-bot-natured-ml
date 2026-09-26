"""Seeding / reproducibility contracts."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from dbn.reinforcement.common import categorical_sample, gaussian_sample
from dbn.reinforcement.algos.ppo import PPO
from dbn.reinforcement.envs import CartPole


def test_env_same_seed_reproducible():
    a = CartPole(seed=42)
    b = CartPole(seed=42)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=42)
    np.testing.assert_array_equal(oa, ob)
    for _ in range(10):
        oa, ra, ta, tra, _ = a.step(1)
        ob, rb, tb, trb, _ = b.step(1)
        np.testing.assert_array_equal(oa, ob)
        assert ra == rb
        assert ta == tb
        assert tra == trb


def test_env_different_seeds_diverge():
    a = CartPole(seed=42)
    b = CartPole(seed=43)
    oa, _ = a.reset(seed=42)
    ob, _ = b.reset(seed=43)
    assert not np.array_equal(oa, ob)


def test_action_sampling_seeded():
    logits = mx.zeros((8, 4))
    key = mx.random.key(0)
    a1 = categorical_sample(logits, key)
    a2 = categorical_sample(logits, key)
    mx.eval(a1, a2)
    np.testing.assert_array_equal(np.array(a1), np.array(a2))

    key_other = mx.random.key(1)
    a3 = categorical_sample(logits, key_other)
    mx.eval(a3)
    # extremely unlikely to match all 8 with different keys
    assert not np.array_equal(np.array(a1), np.array(a3))


def test_gaussian_sample_seeded():
    mean = mx.zeros((4, 2))
    log_std = mx.zeros((2,))
    key = mx.random.key(5)
    s1 = gaussian_sample(mean, log_std, key)
    s2 = gaussian_sample(mean, log_std, key)
    mx.eval(s1, s2)
    np.testing.assert_allclose(np.array(s1), np.array(s2), rtol=0, atol=0)


def test_ppo_same_seed_same_first_actions():
    def first_actions(seed: int):
        env = CartPole(seed=seed)
        model = PPO(
            "MlpPolicy",
            env,
            n_steps=32,
            batch_size=16,
            n_epochs=1,
            policy_kwargs={"net_arch": (8,)},
            seed=seed,
        )
        obs, _ = env.reset(seed=seed)
        outs = []
        for _ in range(5):
            action, _ = model.predict(obs, deterministic=True)
            outs.append(int(action))
            obs, _, term, trunc, _ = env.step(action)
            if term or trunc:
                obs, _ = env.reset()
        return outs

    assert first_actions(123) == first_actions(123)


def test_ppo_different_seeds_can_differ():
    from dbn.reinforcement.common import tree_flatten

    def init_weight_norm(seed: int) -> float:
        env = CartPole(seed=seed)
        model = PPO(
            "MlpPolicy",
            env,
            policy_kwargs={"net_arch": (8,)},
            seed=seed,
        )
        leaves = tree_flatten(model.policy.parameters())
        return float(mx.sum(leaves[0] ** 2))

    assert init_weight_norm(1) != init_weight_norm(2)


def test_replay_sampling_seeded():
    from dbn.reinforcement.common import ReplayBuffer

    def sample_once(seed: int):
        np.random.seed(seed)
        buf = ReplayBuffer(50, 3, 1)
        for i in range(20):
            buf.add(np.full(3, i), np.array([0.0]), float(i), np.full(3, i), 0.0)
        return np.array(buf.sample(5)["rewards"])

    np.testing.assert_array_equal(sample_once(7), sample_once(7))
    assert not np.array_equal(sample_once(7), sample_once(8))
