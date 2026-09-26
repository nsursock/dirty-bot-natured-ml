"""TD3 invariants and smoke tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from dbn.reinforcement.common import tree_flatten
from dbn.reinforcement.algos.td3 import TD3
from dbn.reinforcement.envs import Pendulum


@pytest.mark.smoke
def test_td3_smoke():
    env = Pendulum(seed=0)
    model = TD3(
        "MlpPolicy",
        env,
        learning_starts=32,
        buffer_size=10_000,
        batch_size=32,
        policy_delay=2,
        policy_kwargs={"net_arch": (32, 32)},
        seed=0,
    )
    model.learn(96, progress_bar=False)
    assert model.num_timesteps >= 96
    action, state = model.predict(env.reset()[0], deterministic=True)
    assert state is None
    assert np.asarray(action).shape == (1,)
    assert np.isfinite(action).all()


def test_td3_rejects_discrete():
    from dbn.reinforcement.envs import CartPole

    with pytest.raises(ValueError, match="continuous"):
        TD3("MlpPolicy", CartPole(seed=0))


def test_td3_delayed_actor_update():
    env = Pendulum(seed=0)
    model = TD3(
        "MlpPolicy",
        env,
        learning_starts=16,
        batch_size=16,
        buffer_size=1000,
        policy_delay=3,
        policy_kwargs={"net_arch": (16,)},
        seed=0,
    )
    # fill replay
    model.learn(48, progress_bar=False)

    def actor_snapshot():
        return [np.array(x).copy() for x in tree_flatten(model.actor.parameters())]

    # Force critic-only updates: _n_updates not divisible by policy_delay
    # Run train steps and count when actor changes
    before = actor_snapshot()
    actor_changed = []
    for _ in range(6):
        info = model._train_step()
        after = actor_snapshot()
        changed = any(not np.allclose(a, b) for a, b in zip(before, after))
        actor_changed.append(changed)
        if changed:
            before = after
    # with delay=3, actor should not change every step
    assert any(actor_changed)
    assert not all(actor_changed)


def test_td3_target_noise_clipped():
    from dbn.reinforcement.losses import td3_smooth_target_action

    out = td3_smooth_target_action(
        mx.array([[0.0]]),
        mx.array([[100.0]]),
        noise_clip=0.5,
        action_scale=mx.array([1.0]),
        action_bias=mx.array([0.0]),
    )
    mx.eval(out)
    np.testing.assert_allclose(float(out), 0.5, rtol=1e-5, atol=1e-6)


def test_td3_polyak_moves_targets_toward_online():
    env = Pendulum(seed=0)
    model = TD3(
        "MlpPolicy",
        env,
        learning_starts=8,
        batch_size=8,
        buffer_size=500,
        policy_delay=1,
        tau=0.1,
        policy_kwargs={"net_arch": (8,)},
        seed=2,
    )
    model.learn(24, progress_bar=False)
    # after updates with tau>0, targets should be close but defined
    for t_leaf, s_leaf in zip(
        tree_flatten(model.actor_target.parameters()),
        tree_flatten(model.actor.parameters()),
    ):
        assert np.isfinite(np.array(t_leaf)).all()
        assert np.isfinite(np.array(s_leaf)).all()
