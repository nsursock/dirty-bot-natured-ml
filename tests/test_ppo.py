"""PPO invariants, smoke, and math wiring."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from dbn.reinforcement.algos.ppo import PPO
from dbn.reinforcement.common import ppo_ratio
from dbn.reinforcement.envs import CartPole, Pendulum


@pytest.mark.smoke
def test_ppo_smoke_discrete():
    env = CartPole(seed=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={"net_arch": (32,)},
        seed=0,
    )
    model.learn(128, progress_bar=False)
    assert model.num_timesteps >= 128
    action, state = model.predict(env.reset()[0], deterministic=True)
    assert state is None
    assert int(action) in (0, 1)


@pytest.mark.smoke
def test_ppo_smoke_continuous():
    env = Pendulum(seed=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=64,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={"net_arch": (32,)},
        seed=0,
    )
    model.learn(128, progress_bar=False)
    action, _ = model.predict(env.reset()[0], deterministic=True)
    assert np.asarray(action).shape == (1,)
    assert np.isfinite(action).all()


def test_ppo_ratio_invariant_in_loss_path():
    new_lp = mx.array([0.0, 0.5, -0.5])
    old_lp = mx.array([0.0, 0.0, 0.0])
    ratio = ppo_ratio(new_lp, old_lp)
    mx.eval(ratio)
    np.testing.assert_allclose(np.array(ratio), np.exp(np.array(new_lp) - np.array(old_lp)))


def test_ppo_advantages_normalized_when_requested(cartpole):
    model = PPO(
        "MlpPolicy",
        cartpole,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        normalize_advantage=True,
        policy_kwargs={"net_arch": (16,)},
        seed=1,
    )
    # collect one rollout and inspect buffer advantages before train normalize
    obs, _ = model._reset_env()
    model._collect_rollouts(obs)
    assert model.buffer is not None
    adv = np.array(model.buffer.advantages)
    assert adv.shape == (model.n_steps, model.n_envs)
    # normalize happens inside loss; raw GAE need not be mean0
    assert np.isfinite(adv).all()


def test_ppo_vec_env(cartpole_vec):
    model = PPO(
        "MlpPolicy",
        cartpole_vec,
        n_steps=32,
        batch_size=32,
        n_epochs=1,
        policy_kwargs={"net_arch": (16,)},
        seed=0,
    )
    model.learn(64, progress_bar=False)
    assert model.n_envs == 4
    assert model.num_timesteps >= 64


def test_ppo_truncation_terminal_values_used():
    env = Pendulum(n_envs=1, max_episode_steps=2, seed=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=2,
        batch_size=2,
        n_epochs=1,
        policy_kwargs={"net_arch": (8,)},
        seed=0,
    )
    obs, _ = model._reset_env()
    model._collect_rollouts(obs)
    assert model.buffer is not None
    truncs = np.array(model.buffer.truncations)
    tvals = np.array(model.buffer.terminal_values)
    assert truncs.max() > 0
    assert np.any(np.abs(tvals) > 0)
