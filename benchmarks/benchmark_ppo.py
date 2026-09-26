"""Benchmark PPO — NOT part of pytest CI.

Run: python benchmarks/benchmark_ppo.py
"""

from __future__ import annotations

import time

from dbn.reinforcement.algos.ppo import PPO
from dbn.reinforcement.envs import CartPole


def main():
    env = CartPole(n_envs=8, seed=0)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=512,
        batch_size=256,
        n_epochs=4,
        policy_kwargs={"net_arch": (64, 64)},
        seed=0,
    )
    steps = 10_000
    t0 = time.perf_counter()
    model.learn(steps, progress_bar=True)
    elapsed = time.perf_counter() - t0
    fps = model.num_timesteps / elapsed
    print(f"PPO train_fps={fps:.1f} steps={model.num_timesteps} elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
