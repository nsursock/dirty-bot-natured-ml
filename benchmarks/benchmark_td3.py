"""Benchmark TD3 — NOT part of pytest CI.

Run: python benchmarks/benchmark_td3.py
"""

from __future__ import annotations

import time

from dbn.reinforcement.algos.td3 import TD3
from dbn.reinforcement.envs import Pendulum


def main():
    env = Pendulum(n_envs=1, seed=0)
    model = TD3(
        "MlpPolicy",
        env,
        learning_starts=1000,
        buffer_size=100_000,
        batch_size=256,
        policy_kwargs={"net_arch": (256, 256)},
        seed=0,
    )
    steps = 5_000
    t0 = time.perf_counter()
    model.learn(steps, progress_bar=True)
    elapsed = time.perf_counter() - t0
    fps = model.num_timesteps / elapsed
    print(f"TD3 train_fps={fps:.1f} steps={model.num_timesteps} elapsed_s={elapsed:.2f}")


if __name__ == "__main__":
    main()
