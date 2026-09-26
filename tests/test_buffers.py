"""Replay and rollout buffer shape / contract tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np

from dbn.reinforcement.common import ReplayBuffer, RolloutBuffer


def test_replay_buffer_add_sample_shapes():
    buf = ReplayBuffer(capacity=100, obs_dim=4, action_dim=2)
    for i in range(10):
        buf.add(
            np.ones(4) * i,
            np.array([0.1, -0.1]),
            1.0,
            np.ones(4) * (i + 1),
            float(i % 2),
        )
    assert buf.size == 10
    batch = buf.sample(5)
    assert batch["obs"].shape == (5, 4)
    assert batch["actions"].shape == (5, 2)
    assert batch["rewards"].shape == (5,)
    assert batch["next_obs"].shape == (5, 4)
    assert batch["dones"].shape == (5,)


def test_replay_buffer_wraps_capacity():
    buf = ReplayBuffer(capacity=5, obs_dim=2, action_dim=1)
    for i in range(12):
        buf.add(np.zeros(2), np.zeros(1), float(i), np.zeros(2), 0.0)
    assert buf.size == 5
    assert buf.pos == 2


def test_replay_buffer_add_batch_and_wrap():
    buf = ReplayBuffer(capacity=10, obs_dim=3, action_dim=1)
    obs = np.arange(24, dtype=np.float32).reshape(8, 3)
    actions = np.ones((8, 1), dtype=np.float32)
    rewards = np.arange(8, dtype=np.float32)
    next_obs = obs + 1
    dones = np.zeros(8, dtype=np.float32)
    buf.add_batch(obs, actions, rewards, next_obs, dones)
    assert buf.size == 8
    assert buf.pos == 8
    # wrap
    buf.add_batch(obs[:5], actions[:5], rewards[:5], next_obs[:5], dones[:5])
    assert buf.size == 10
    assert buf.pos == 3
    batch = buf.sample(4)
    assert batch["obs"].shape == (4, 3)


def test_box_batch_sample():
    from dbn.reinforcement.envs.spaces import Box

    space = Box(low=-1.0, high=1.0, shape=(2,))
    one = space.sample()
    assert one.shape == (2,)
    batch = space.sample(n=16)
    assert batch.shape == (16, 2)
    assert np.all(batch >= -1.0) and np.all(batch <= 1.0)


def test_rollout_buffer_gae_shapes_discrete():
    T, N = 8, 2
    buf = RolloutBuffer(T, N, (4,), action_dim=2, continuous=False, gamma=0.99, gae_lambda=0.95)
    for t in range(T):
        buf.add(
            mx.zeros((N, 4)),
            mx.zeros((N,), dtype=mx.int32),
            mx.ones((N,)),
            mx.zeros((N,)),
            mx.zeros((N,)),
            mx.zeros((N,)),
        )
    buf.compute_returns_and_advantage(mx.zeros((N,)))
    assert buf.obs.shape == (T, N, 4)
    assert buf.advantages.shape == (T, N)
    assert buf.returns.shape == (T, N)
    batches = list(buf.get(batch_size=4))
    assert len(batches) == (T * N) // 4
    b0 = batches[0]
    assert b0["obs"].shape == (4, 4)
    assert b0["actions"].shape == (4,)
    assert b0["advantages"].shape == (4,)


def test_rollout_buffer_continuous_action_shape():
    T, N = 4, 1
    buf = RolloutBuffer(T, N, (3,), action_dim=1, continuous=True)
    for _ in range(T):
        buf.add(
            mx.zeros((N, 3)),
            mx.zeros((N, 1)),
            mx.zeros((N,)),
            mx.zeros((N,)),
            mx.zeros((N,)),
            mx.zeros((N,)),
        )
    buf.compute_returns_and_advantage(mx.zeros((N,)))
    b = next(buf.get(batch_size=2))
    assert b["actions"].shape == (2, 1)
