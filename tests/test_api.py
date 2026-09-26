"""Public SB3-shaped API contract tests."""

from __future__ import annotations

import numpy as np

from dbn import PPO, SAC, TD3, CartPole, Pendulum


def test_exports():
    assert callable(PPO)
    assert callable(SAC)
    assert callable(TD3)
    assert callable(CartPole)
    assert callable(Pendulum)


def test_ppo_api_learn_predict():
    env = CartPole(seed=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": (8,)},
        seed=0,
    )
    out = model.learn(32, progress_bar=False)
    assert out is model
    action, state = model.predict(env.reset()[0])
    assert state is None
    assert np.asarray(action).shape in ((), (1,))


def test_sac_api_learn_predict():
    env = Pendulum(seed=0)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=16,
        batch_size=16,
        buffer_size=1000,
        policy_kwargs={"net_arch": (8,)},
        seed=0,
    )
    out = model.learn(48, progress_bar=False)
    assert out is model
    action, state = model.predict(env.reset()[0])
    assert state is None
    assert np.asarray(action).shape == (1,)


def test_td3_api_learn_predict():
    env = Pendulum(seed=0)
    model = TD3(
        "MlpPolicy",
        env,
        learning_starts=16,
        batch_size=16,
        buffer_size=1000,
        policy_kwargs={"net_arch": (8,)},
        seed=0,
    )
    out = model.learn(48, progress_bar=False)
    assert out is model
    action, state = model.predict(env.reset()[0])
    assert state is None
    assert np.asarray(action).shape == (1,)


def test_space_info_discrete_and_continuous():
    from dbn.reinforcement.common import space_info

    od, ad, cont = space_info(CartPole())
    assert od == 4 and ad == 2 and cont is False
    od, ad, cont = space_info(Pendulum())
    assert od == 3 and ad == 1 and cont is True
