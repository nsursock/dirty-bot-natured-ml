# Dirty Bot Natured — ML library (JAX/MLX) for trading bots.

No Stable-Baselines3 or Gymnasium dependency — envs and algos are native.

## Install

```bash
pip install -e ".[dev]"
```

## Layout

```
dbn/reinforcement/
  algos/   common.py, losses.py, ppo.py, sac.py, td3.py
  envs/    cartpole.py, pendulum.py, spaces.py
tests/     correctness / contracts / regression (pytest)
benchmarks/  FPS / scaling (not in CI)
```

## Examples

```python
from dbn import PPO, SAC, TD3, CartPole, Pendulum

# discrete
env = CartPole()
PPO("MlpPolicy", env, tensorboard_log="runs/").learn(50_000)

# continuous
env = Pendulum()
SAC("MlpPolicy", env, tensorboard_log="runs/").learn(50_000)
TD3("MlpPolicy", env, tensorboard_log="runs/").learn(50_000)
```

Logs: TensorBoard events + `progress.csv` under `runs/<Algo>_1/`. Training uses `tqdm`.

## Tests

```bash
pytest                 # math, shapes, seeds, envs, smoke
python benchmarks/benchmark_ppo.py
```

Pytest covers correctness and API contracts — not performance.
