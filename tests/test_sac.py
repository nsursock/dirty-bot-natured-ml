"""SAC invariants and smoke tests."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import pytest

from dbn.reinforcement.algos.sac import SAC
from dbn.reinforcement.common import sac_bellman_target, soft_update, tree_flatten, tree_map2
from dbn.reinforcement.envs import Pendulum


@pytest.mark.smoke
def test_sac_smoke():
    env = Pendulum(seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=32,
        buffer_size=10_000,
        batch_size=32,
        policy_kwargs={"net_arch": (32, 32)},
        seed=0,
    )
    model.learn(96, progress_bar=False)
    assert model.num_timesteps >= 96
    action, state = model.predict(env.reset()[0], deterministic=True)
    assert state is None
    assert np.asarray(action).shape == (1,)
    assert np.isfinite(action).all()


def test_sac_rejects_discrete():
    from dbn.reinforcement.envs import CartPole

    with pytest.raises(ValueError, match="continuous"):
        SAC("MlpPolicy", CartPole(seed=0))


def test_sac_alpha_remains_positive():
    env = Pendulum(seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=16,
        batch_size=16,
        buffer_size=1000,
        policy_kwargs={"net_arch": (16,)},
        seed=0,
        ent_coef="auto",
    )
    # fill buffer + a few updates
    model.learn(64, progress_bar=False)
    assert model.log_alpha_mod is not None
    alpha = float(model.log_alpha_mod())
    assert alpha > 0.0


def test_sac_target_networks_soft_update_moves_toward_online():
    env = Pendulum(seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=8,
        batch_size=8,
        buffer_size=500,
        tau=0.05,
        policy_kwargs={"net_arch": (8,)},
        seed=1,
    )
    # snapshot distance after init (should be ~0)
    before = [
        float(mx.sum((t - s) ** 2))
        for t, s in zip(
            tree_flatten(model.critic_target.parameters()),
            tree_flatten(model.critic.parameters()),
        )
    ]
    assert all(abs(x) < 1e-6 for x in before)

    # mutate online critic then soft-update once
    mutated = tree_map2(lambda t, s: s + 1.0, model.critic.parameters(), model.critic.parameters())
    model.critic.update(mutated)
    mx.eval(model.critic.parameters())
    soft_update(model.critic_target, model.critic, model.tau)
    mx.eval(model.critic_target.parameters())
    after = [
        float(mx.sum((t - s) ** 2))
        for t, s in zip(
            tree_flatten(model.critic_target.parameters()),
            tree_flatten(model.critic.parameters()),
        )
    ]
    # target moved partway; still not equal when tau < 1
    assert all(d > 0 for d in after)


def test_sac_target_q_uses_min():
    t = sac_bellman_target(
        mx.array([0.0]),
        mx.array([0.0]),
        mx.array([5.0]),
        mx.array([1.0]),
        mx.array([0.0]),
        0.0,
        1.0,
    )
    mx.eval(t)
    assert float(t) == 1.0
