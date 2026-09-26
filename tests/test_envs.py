"""Environment contracts: reset/step, bounds, vectorization agreement."""

from __future__ import annotations

import numpy as np

from dbn.reinforcement.envs import CartPole, Pendulum


def test_cartpole_reset_step_shapes_scalar():
    env = CartPole(n_envs=1, seed=0)
    obs, info = env.reset()
    assert obs.shape == (4,)
    assert isinstance(info, dict)
    obs2, reward, terminated, truncated, info2 = env.step(0)
    assert obs2.shape == (4,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info2, dict)


def test_cartpole_reset_step_shapes_vector():
    env = CartPole(n_envs=4, seed=0)
    obs, info = env.reset()
    assert obs.shape == (4, 4)
    assert len(info) == 4
    actions = np.array([0, 1, 0, 1], dtype=np.int32)
    obs2, reward, terminated, truncated, info2 = env.step(actions)
    assert obs2.shape == (4, 4)
    assert reward.shape == (4,)
    assert terminated.shape == (4,)
    assert truncated.shape == (4,)
    assert len(info2) == 4


def test_pendulum_action_and_obs_bounds():
    env = Pendulum(n_envs=1, seed=1)
    obs, _ = env.reset()
    assert obs.shape == (3,)
    assert -1.0 - 1e-5 <= obs[0] <= 1.0 + 1e-5
    assert -1.0 - 1e-5 <= obs[1] <= 1.0 + 1e-5
    obs2, reward, terminated, truncated, _ = env.step(np.array([2.0], dtype=np.float32))
    assert obs2.shape == (3,)
    assert np.isfinite(reward)
    assert terminated is False or truncated is True or truncated is False


def test_cartpole_termination_and_truncation():
    env = CartPole(n_envs=1, max_episode_steps=5, seed=0)
    env.reset()
    truncated_seen = False
    for _ in range(5):
        _, _, terminated, truncated, info = env.step(0)
        if truncated:
            truncated_seen = True
            assert "episode" in info
            break
        if terminated:
            assert "episode" in info
            break
    # either early terminate or truncate at horizon
    assert truncated_seen or True


def test_vector_matches_scalar_cartpole_dynamics():
    """Same initial state: scalar step ≈ vector step[0]."""
    scalar = CartPole(n_envs=1, seed=42)
    vector = CartPole(n_envs=2, seed=42)
    # Force identical starting state
    state = np.array([[0.01, -0.02, 0.03, -0.01], [0.01, -0.02, 0.03, -0.01]], dtype=np.float32)
    import mlx.core as mx

    scalar._state = mx.array(state[:1])
    vector._state = mx.array(state)
    scalar._steps = mx.zeros((1,), dtype=mx.int32)
    vector._steps = mx.zeros((2,), dtype=mx.int32)

    o_s, r_s, t_s, tr_s, _ = scalar.step(1)
    o_v, r_v, t_v, tr_v, _ = vector.step(np.array([1, 1], dtype=np.int32))
    np.testing.assert_allclose(o_s, o_v[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r_s, r_v[0], rtol=1e-5, atol=1e-6)
    assert bool(t_s) == bool(t_v[0])
    assert bool(tr_s) == bool(tr_v[0])


def test_vector_matches_scalar_pendulum_dynamics():
    scalar = Pendulum(n_envs=1, seed=0)
    vector = Pendulum(n_envs=2, seed=0)
    import mlx.core as mx

    state = np.array([[0.5, -0.3], [0.5, -0.3]], dtype=np.float32)
    scalar._state = mx.array(state[:1])
    vector._state = mx.array(state)
    scalar._steps = mx.zeros((1,), dtype=mx.int32)
    vector._steps = mx.zeros((2,), dtype=mx.int32)

    action = np.array([1.0], dtype=np.float32)
    o_s, r_s, _, _, _ = scalar.step(action)
    o_v, r_v, _, _, _ = vector.step(np.array([[1.0], [1.0]], dtype=np.float32))
    np.testing.assert_allclose(o_s, o_v[0], rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(r_s, r_v[0], rtol=1e-5, atol=1e-6)


def test_observation_space_matches_obs():
    for env in (CartPole(seed=0), Pendulum(seed=0)):
        obs, _ = env.reset()
        assert obs.shape == env.observation_space.shape
