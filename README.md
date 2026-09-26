# dbn

Dirty Bot Natured — ML library (JAX/MLX) for trading bots.

No Stable-Baselines3 or Gymnasium dependency — envs and algos are native.

## Install

```bash
pip install -e .
```

## Layout

```
dbn/reinforcement/
  algos/   common.py, ppo.py, sac.py, td3.py
  envs/    cartpole.py, pendulum.py, spaces.py
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
