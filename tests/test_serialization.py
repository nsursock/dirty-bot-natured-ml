"""Save / load contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from dbn.reinforcement.algos.ppo import PPO
from dbn.reinforcement.envs import CartPole


def test_ppo_save_load_preserves_deterministic_predictions(tmp_path: Path):
    env = CartPole(seed=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=32,
        batch_size=16,
        n_epochs=1,
        policy_kwargs={"net_arch": (16,)},
        seed=0,
    )
    model.learn(64, progress_bar=False)

    obs, _ = env.reset(seed=1)
    before = [model.predict(obs, deterministic=True)[0] for _ in range(5)]
    # re-reset same obs trajectory start
    obs, _ = env.reset(seed=1)
    path = tmp_path / "ppo_cartpole"
    model.save(str(path))

    env2 = CartPole(seed=0)
    loaded = PPO.load(str(path), env2)
    obs2, _ = env2.reset(seed=1)
    after = [loaded.predict(obs2, deterministic=True)[0] for _ in range(5)]
    for a, b in zip(before, after):
        assert int(a) == int(b)


def test_ppo_save_writes_npz(tmp_path: Path):
    env = CartPole(seed=0)
    model = PPO("MlpPolicy", env, policy_kwargs={"net_arch": (8,)}, seed=0)
    out = tmp_path / "weights"
    model.save(str(out))
    assert (tmp_path / "weights.npz").exists()
    data = np.load(tmp_path / "weights.npz")
    assert "_obs_dim" in data.files
    assert any(not k.startswith("_") for k in data.files)
