"""Shared RL primitives: spaces helpers, policy, buffer, logging, trees."""

from __future__ import annotations

import csv
import math
import time
from pathlib import Path
from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def as_numpy(x) -> np.ndarray:
    if isinstance(x, mx.array):
        return np.array(x)
    return np.asarray(x)


def space_info(env) -> tuple[int, int, bool]:
    obs_space = getattr(env, "observation_space", None)
    act_space = getattr(env, "action_space", None)
    if obs_space is not None and hasattr(obs_space, "shape"):
        obs_dim = int(np.prod(obs_space.shape))
    else:
        o, _ = env.reset()
        obs_dim = int(np.asarray(o).reshape(-1).shape[0])
    if act_space is None:
        raise ValueError("env must expose action_space with .n or .shape")
    if hasattr(act_space, "n"):
        return obs_dim, int(act_space.n), False
    return obs_dim, int(np.prod(act_space.shape)), True


def explained_variance(y_pred: mx.array, y_true: mx.array) -> float:
    var_y = mx.var(y_true)
    return float(1.0 - mx.var(y_true - y_pred) / (var_y + 1e-8)) if float(var_y) > 0 else 0.0


def tree_flatten(tree) -> list[mx.array]:
    if isinstance(tree, mx.array):
        return [tree]
    if isinstance(tree, dict):
        out: list[mx.array] = []
        for k in sorted(tree.keys(), key=str):
            out.extend(tree_flatten(tree[k]))
        return out
    if isinstance(tree, (list, tuple)):
        out: list[mx.array] = []
        for x in tree:
            out.extend(tree_flatten(x))
        return out
    return []


def tree_map(fn, tree):
    if isinstance(tree, mx.array):
        return fn(tree)
    if isinstance(tree, dict):
        return {k: tree_map(fn, v) for k, v in tree.items()}
    if isinstance(tree, list):
        return [tree_map(fn, v) for v in tree]
    if isinstance(tree, tuple):
        return tuple(tree_map(fn, v) for v in tree)
    return tree


def tree_map2(fn, a, b):
    if isinstance(a, mx.array):
        return fn(a, b)
    if isinstance(a, dict):
        return {k: tree_map2(fn, a[k], b[k]) for k in a}
    if isinstance(a, list):
        return [tree_map2(fn, x, y) for x, y in zip(a, b)]
    if isinstance(a, tuple):
        return tuple(tree_map2(fn, x, y) for x, y in zip(a, b))
    return fn(a, b)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    target.update(tree_map2(lambda t, s: t * (1.0 - tau) + s * tau, target.parameters(), source.parameters()))


def action_scale_bias(env) -> tuple[mx.array, mx.array]:
    space = env.action_space
    high = np.asarray(space.high, dtype=np.float32).reshape(-1)
    low = np.asarray(space.low, dtype=np.float32).reshape(-1)
    scale = mx.array((high - low) / 2.0)
    bias = mx.array((high + low) / 2.0)
    return scale, bias


def tree_flatten_dict(tree, prefix="") -> dict[str, mx.array]:
    if isinstance(tree, mx.array):
        return {prefix: tree}
    out: dict[str, mx.array] = {}
    if isinstance(tree, dict):
        for k, v in tree.items():
            out.update(tree_flatten_dict(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(tree, (list, tuple)):
        for i, v in enumerate(tree):
            out.update(tree_flatten_dict(v, f"{prefix}.{i}" if prefix else str(i)))
    return out


def clip_grad_norm(grads, max_norm: float):
    leaves = tree_flatten(grads)
    total = mx.sqrt(sum(mx.sum(g.astype(mx.float32) ** 2) for g in leaves) + 1e-8)
    scale = mx.minimum(mx.array(1.0), mx.array(max_norm) / total)
    return tree_map(lambda g: g * scale, grads)


# ---------------------------------------------------------------------------
# logging — TensorBoard events + progress.csv
# ---------------------------------------------------------------------------

class Logger:
    """Writes scalar metrics to TensorBoard and a CSV progress file."""

    def __init__(self, log_dir: str | Path):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter

        self._Event = Event
        self._Summary = Summary
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self._writer = EventFileWriter(str(self.log_dir))
        self._csv_path = self.log_dir / "progress.csv"
        self._csv_file = self._csv_path.open("w", newline="")
        self._csv_writer: csv.DictWriter | None = None
        self._fieldnames: list[str] | None = None

    def record(self, metrics: dict[str, float], step: int) -> None:
        for tag, value in metrics.items():
            summary = self._Summary(
                value=[self._Summary.Value(tag=tag, simple_value=float(value))]
            )
            self._writer.add_event(
                self._Event(wall_time=time.time(), step=int(step), summary=summary)
            )
        row = {"timesteps": int(step), **{k: float(v) for k, v in metrics.items()}}
        if self._csv_writer is None:
            self._fieldnames = list(row.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._fieldnames)
            self._csv_writer.writeheader()
        # stable columns: drop unknown keys / fill missing
        assert self._fieldnames is not None
        self._csv_writer.writerow({k: row.get(k, "") for k in self._fieldnames})

    def flush(self) -> None:
        self._writer.flush()
        self._csv_file.flush()

    def close(self) -> None:
        self._writer.close()
        self._csv_file.close()


# ---------------------------------------------------------------------------
# policy
# ---------------------------------------------------------------------------

def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.Tanh())
    return nn.Sequential(*layers)


class MlpPolicy(nn.Module):
    """Shared-trunk actor-critic. Discrete → logits; continuous → mean + log_std."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        continuous: bool = False,
        net_arch: Sequence[int] = (64, 64),
        log_std_init: float = 0.0,
    ):
        super().__init__()
        self.continuous = continuous
        self.action_dim = action_dim
        hidden = list(net_arch)
        self.trunk = _mlp([obs_dim, *hidden])
        last = hidden[-1] if hidden else obs_dim
        self.actor = nn.Linear(last, action_dim)
        self.critic = nn.Linear(last, 1)
        if continuous:
            self.log_std = mx.full((action_dim,), float(log_std_init))

    def __call__(self, obs: mx.array) -> tuple[mx.array, mx.array]:
        h = self.trunk(obs)
        return self.actor(h), mx.squeeze(self.critic(h), axis=-1)

    def value(self, obs: mx.array) -> mx.array:
        return mx.squeeze(self.critic(self.trunk(obs)), axis=-1)


def _log_softmax(x: mx.array) -> mx.array:
    return x - mx.logsumexp(x, axis=-1, keepdims=True)


def categorical_sample(logits: mx.array, key: mx.array) -> mx.array:
    u = mx.random.uniform(key=key, shape=logits.shape)
    return mx.argmax(logits - mx.log(-mx.log(u + 1e-8) + 1e-8), axis=-1)


def categorical_log_prob(logits: mx.array, actions: mx.array) -> mx.array:
    log_probs = _log_softmax(logits)

    def _one(lp: mx.array, a: mx.array) -> mx.array:
        return lp[a.astype(mx.int32)]

    return mx.vmap(_one)(log_probs, actions)


def categorical_entropy(logits: mx.array) -> mx.array:
    log_p = _log_softmax(logits)
    p = mx.exp(log_p)
    return -mx.sum(p * log_p, axis=-1)


def gaussian_sample(mean: mx.array, log_std: mx.array, key: mx.array) -> mx.array:
    eps = mx.random.normal(key=key, shape=mean.shape)
    return mean + mx.exp(log_std) * eps


def gaussian_log_prob(mean: mx.array, log_std: mx.array, actions: mx.array) -> mx.array:
    var = mx.exp(2.0 * log_std)
    log_scale = log_std + 0.5 * math.log(2.0 * math.pi)
    return -0.5 * mx.sum(((actions - mean) ** 2) / var, axis=-1) - mx.sum(log_scale, axis=-1)


def gaussian_entropy(log_std: mx.array) -> mx.array:
    return 0.5 * math.log(2.0 * math.pi * math.e) * log_std.shape[-1] + mx.sum(log_std)


# ---------------------------------------------------------------------------
# rollout buffer + fused GAE
# ---------------------------------------------------------------------------

def make_gae(gamma: float, gae_lambda: float):
    """Build fused vmap(GAE) ∘ compile; gamma/λ captured as constants."""

    def _gae_one_env(
        rewards: mx.array,
        values: mx.array,
        terminations: mx.array,
        truncations: mx.array,
        terminal_values: mx.array,
        last_value: mx.array,
    ) -> tuple[mx.array, mx.array]:
        T = int(rewards.shape[0])
        adv_rev: list[mx.array] = []
        gae = mx.array(0.0, dtype=rewards.dtype)
        next_value = last_value
        for t in range(T - 1, -1, -1):
            next_value = mx.where(truncations[t] > 0, terminal_values[t], next_value)
            nonterminal = 1.0 - terminations[t]
            delta = rewards[t] + gamma * next_value * nonterminal - values[t]
            gae = delta + gamma * gae_lambda * nonterminal * gae
            adv_rev.append(gae)
            next_value = values[t]
        advantages = mx.stack(adv_rev[::-1])
        return advantages, advantages + values

    return mx.compile(mx.vmap(_gae_one_env, in_axes=(1, 1, 1, 1, 1, 0), out_axes=(1, 1)))


class RolloutBuffer:
    """On-policy buffer: collects steps then stacks to (T, N, ...)."""

    def __init__(
        self,
        n_steps: int,
        n_envs: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        *,
        continuous: bool,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        self.n_steps = n_steps
        self.n_envs = n_envs
        self.continuous = continuous
        self.obs_shape = obs_shape
        self.action_dim = action_dim
        self._gae = make_gae(gamma, gae_lambda)
        self.reset()

    def reset(self) -> None:
        self._obs: list[mx.array] = []
        self._actions: list[mx.array] = []
        self._rewards: list[mx.array] = []
        self._dones: list[mx.array] = []
        self._terminations: list[mx.array] = []
        self._truncations: list[mx.array] = []
        self._terminal_values: list[mx.array] = []
        self._values: list[mx.array] = []
        self._log_probs: list[mx.array] = []
        self.obs = self.actions = self.rewards = None  # type: ignore
        self.dones = self.values = self.log_probs = None  # type: ignore
        self.terminations = self.truncations = self.terminal_values = None  # type: ignore
        self.advantages = self.returns = None  # type: ignore

    def add(self, obs, actions, rewards, dones, terminations, truncations, terminal_values, values, log_probs) -> None:
        self._obs.append(obs)
        self._actions.append(actions)
        self._rewards.append(rewards)
        self._dones.append(dones)
        self._terminations.append(terminations)
        self._truncations.append(truncations)
        self._terminal_values.append(terminal_values)
        self._values.append(values)
        self._log_probs.append(log_probs)

    def compute_returns_and_advantage(self, last_values: mx.array) -> None:
        self.obs = mx.stack(self._obs)
        self.actions = mx.stack(self._actions)
        self.rewards = mx.stack(self._rewards)
        self.dones = mx.stack(self._dones)
        self.terminations = mx.stack(self._terminations)
        self.truncations = mx.stack(self._truncations)
        self.terminal_values = mx.stack(self._terminal_values)
        self.values = mx.stack(self._values)
        self.log_probs = mx.stack(self._log_probs)
        adv, ret = self._gae(self.rewards, self.values, self.terminations, self.truncations, self.terminal_values, last_values)
        self.advantages, self.returns = adv, ret
        mx.eval(self.advantages, self.returns)

    def get(self, batch_size: int):
        T, N = self.n_steps, self.n_envs
        n = T * N
        obs = self.obs.reshape((n, *self.obs_shape))
        actions = (
            self.actions.reshape((n, self.action_dim))
            if self.continuous
            else self.actions.reshape((n,))
        )
        flat = {
            "obs": obs,
            "actions": actions,
            "old_log_prob": self.log_probs.reshape((n,)),
            "advantages": self.advantages.reshape((n,)),
            "returns": self.returns.reshape((n,)),
            "old_values": self.values.reshape((n,)),
        }
        idx = mx.random.permutation(n)
        for start in range(0, n, batch_size):
            bi = idx[start : start + batch_size]
            yield {k: v[bi] for k, v in flat.items()}


# ---------------------------------------------------------------------------
# off-policy: replay + continuous actors / twin Q
# ---------------------------------------------------------------------------

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = int(capacity)
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.next_obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity,), dtype=np.float32)
        self.dones = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0
        self.size = 0

    def add(self, obs, action, reward, next_obs, done) -> None:
        obs = np.asarray(obs, dtype=np.float32).reshape(self.obs.shape[-1])
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(self.next_obs.shape[-1])
        action = np.asarray(action, dtype=np.float32).reshape(self.actions.shape[-1])
        i = self.pos
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i] = float(reward)
        self.dones[i] = float(done)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def add_batch(self, obs, actions, rewards, next_obs, dones) -> None:
        """Vectorized insert of a batch of transitions (handles wraparound)."""
        obs = np.asarray(obs, dtype=np.float32).reshape(-1, self.obs.shape[-1])
        next_obs = np.asarray(next_obs, dtype=np.float32).reshape(-1, self.next_obs.shape[-1])
        actions = np.asarray(actions, dtype=np.float32).reshape(-1, self.actions.shape[-1])
        rewards = np.asarray(rewards, dtype=np.float32).reshape(-1)
        dones = np.asarray(dones, dtype=np.float32).reshape(-1)
        n = int(obs.shape[0])
        if n == 0:
            return
        if n > self.capacity:
            # keep only the most recent `capacity` transitions
            obs, next_obs = obs[-self.capacity :], next_obs[-self.capacity :]
            actions, rewards, dones = actions[-self.capacity :], rewards[-self.capacity :], dones[-self.capacity :]
            n = self.capacity

        end = self.pos + n
        if end <= self.capacity:
            sl = slice(self.pos, end)
            self.obs[sl] = obs
            self.next_obs[sl] = next_obs
            self.actions[sl] = actions
            self.rewards[sl] = rewards
            self.dones[sl] = dones
        else:
            first = self.capacity - self.pos
            self.obs[self.pos :] = obs[:first]
            self.next_obs[self.pos :] = next_obs[:first]
            self.actions[self.pos :] = actions[:first]
            self.rewards[self.pos :] = rewards[:first]
            self.dones[self.pos :] = dones[:first]
            rest = n - first
            self.obs[:rest] = obs[first:]
            self.next_obs[:rest] = next_obs[first:]
            self.actions[:rest] = actions[first:]
            self.rewards[:rest] = rewards[first:]
            self.dones[:rest] = dones[first:]
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size: int) -> dict[str, mx.array]:
        idx = np.random.randint(0, self.size, size=batch_size)
        return {
            "obs": mx.array(self.obs[idx]),
            "actions": mx.array(self.actions[idx]),
            "rewards": mx.array(self.rewards[idx]),
            "next_obs": mx.array(self.next_obs[idx]),
            "dones": mx.array(self.dones[idx]),
        }


class DeterministicActor(nn.Module):
    """tanh → scaled continuous actions (TD3)."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        net_arch: Sequence[int] = (256, 256),
        action_scale: mx.array | float = 1.0,
        action_bias: mx.array | float = 0.0,
    ):
        super().__init__()
        hidden = list(net_arch)
        self.body = _mlp([obs_dim, *hidden, action_dim])
        self.action_scale = (
            action_scale if isinstance(action_scale, mx.array) else mx.array(float(action_scale))
        )
        self.action_bias = (
            action_bias if isinstance(action_bias, mx.array) else mx.array(float(action_bias))
        )

    def __call__(self, obs: mx.array) -> mx.array:
        scale = mx.stop_gradient(self.action_scale)
        bias = mx.stop_gradient(self.action_bias)
        return mx.tanh(self.body(obs)) * scale + bias


class SquashedGaussianActor(nn.Module):
    """Reparameterized tanh-Gaussian (SAC)."""

    LOG_STD_MIN = -20.0
    LOG_STD_MAX = 2.0

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        *,
        net_arch: Sequence[int] = (256, 256),
        action_scale: mx.array | float = 1.0,
        action_bias: mx.array | float = 0.0,
    ):
        super().__init__()
        hidden = list(net_arch)
        last = hidden[-1] if hidden else obs_dim
        self.trunk = _mlp([obs_dim, *hidden]) if hidden else _mlp([obs_dim, obs_dim])
        self.mean_head = nn.Linear(last if hidden else obs_dim, action_dim)
        self.log_std_head = nn.Linear(last if hidden else obs_dim, action_dim)
        self.action_scale = (
            action_scale if isinstance(action_scale, mx.array) else mx.array(float(action_scale))
        )
        self.action_bias = (
            action_bias if isinstance(action_bias, mx.array) else mx.array(float(action_bias))
        )

    def _params(self, obs: mx.array) -> tuple[mx.array, mx.array]:
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = mx.clip(self.log_std_head(h), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def __call__(
        self, obs: mx.array, key: mx.array | None = None, *, deterministic: bool = False
    ) -> tuple[mx.array, mx.array | None]:
        mean, log_std = self._params(obs)
        scale = mx.stop_gradient(self.action_scale)
        bias = mx.stop_gradient(self.action_bias)
        if deterministic:
            return mx.tanh(mean) * scale + bias, None
        assert key is not None
        std = mx.exp(log_std)
        u = mean + std * mx.random.normal(key=key, shape=mean.shape)
        action = mx.tanh(u) * scale + bias
        log_prob_u = -0.5 * (
            ((u - mean) / (std + 1e-8)) ** 2 + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        log_prob_u = mx.sum(log_prob_u, axis=-1)
        log_det = mx.sum(mx.log(scale * (1.0 - mx.tanh(u) ** 2) + 1e-6), axis=-1)
        return action, log_prob_u - log_det


class TwinQ(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, *, net_arch: Sequence[int] = (256, 256)):
        super().__init__()
        sizes = [obs_dim + action_dim, *list(net_arch), 1]
        self.q1 = _mlp(sizes)
        self.q2 = _mlp(sizes)

    def __call__(self, obs: mx.array, action: mx.array) -> tuple[mx.array, mx.array]:
        x = mx.concatenate([obs, action], axis=-1)
        return mx.squeeze(self.q1(x), axis=-1), mx.squeeze(self.q2(x), axis=-1)


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------

def evaluate_policy(
    model,
    env,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
) -> tuple[float, float, int]:
    """Run deterministic policy evaluation, return (mean, std, n_episodes)."""
    n_envs = int(getattr(env, "num_envs", getattr(env, "n_envs", 1)))
    obs, _ = env.reset()
    episode_returns: list[float] = []
    current_returns = np.zeros(n_envs, dtype=np.float64)
    while len(episode_returns) < n_eval_episodes:
        action, _ = model.predict(obs, deterministic=deterministic)
        step_out = env.step(action)
        if len(step_out) == 5:
            next_obs, rewards, terminations, truncations, _ = step_out
            dones = np.logical_or(terminations, truncations)
        else:
            next_obs, rewards, dones, _ = step_out
        rewards = np.asarray(rewards).reshape(n_envs)
        dones = np.asarray(dones).reshape(n_envs)
        current_returns += rewards
        obs = next_obs
        for i in np.flatnonzero(dones):
            episode_returns.append(float(current_returns[i]))
            current_returns[i] = 0.0
    returns = np.asarray(episode_returns[:n_eval_episodes], dtype=np.float64)
    mean = float(np.mean(returns))
    std = float(np.std(returns, ddof=1)) if len(returns) > 1 else 0.0
    return mean, std, len(returns)


# ---------------------------------------------------------------------------
# losses
# ---------------------------------------------------------------------------

def ppo_ratio(new_log_prob: mx.array, old_log_prob: mx.array) -> mx.array:
    return mx.exp(new_log_prob - old_log_prob)


def ppo_policy_loss(
    new_log_prob: mx.array,
    old_log_prob: mx.array,
    advantages: mx.array,
    clip_range: float,
) -> mx.array:
    """Clipped surrogate: -mean(min(r*A, clip(r)*A))."""
    ratio = ppo_ratio(new_log_prob, old_log_prob)
    pg1 = ratio * advantages
    pg2 = mx.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    return -mx.mean(mx.minimum(pg1, pg2))


def ppo_value_loss(
    values: mx.array,
    returns: mx.array,
    old_values: mx.array | None = None,
    clip_range_vf: float | None = None,
) -> mx.array:
    if clip_range_vf is None or old_values is None:
        return mx.mean((returns - values) ** 2)
    v_clipped = old_values + mx.clip(values - old_values, -clip_range_vf, clip_range_vf)
    return 0.5 * mx.mean(
        mx.maximum((returns - values) ** 2, (returns - v_clipped) ** 2)
    )


def ppo_entropy_loss(entropy: mx.array) -> mx.array:
    """Entropy bonus term as used in PPO total loss (negative mean entropy)."""
    return -mx.mean(entropy)


def normalize_advantages(advantages: mx.array, eps: float = 1e-8) -> mx.array:
    return (advantages - mx.mean(advantages)) / (mx.std(advantages) + eps)


def sac_bellman_target(
    rewards: mx.array,
    dones: mx.array,
    next_q1: mx.array,
    next_q2: mx.array,
    next_log_prob: mx.array,
    alpha: mx.array | float,
    gamma: float,
) -> mx.array:
    """Soft Bellman target using twin-Q minimum."""
    return rewards + (1.0 - dones) * gamma * (
        mx.minimum(next_q1, next_q2) - alpha * next_log_prob
    )


def sac_critic_loss(q1: mx.array, q2: mx.array, target_q: mx.array) -> mx.array:
    return 0.5 * (mx.mean((q1 - target_q) ** 2) + mx.mean((q2 - target_q) ** 2))


def sac_actor_loss(
    log_prob: mx.array,
    q1: mx.array,
    q2: mx.array,
    alpha: mx.array | float,
) -> mx.array:
    return mx.mean(alpha * log_prob - mx.minimum(q1, q2))


def sac_alpha_loss(
    log_alpha: mx.array,
    log_prob: mx.array,
    target_entropy: float,
) -> mx.array:
    return -mx.mean(log_alpha * mx.stop_gradient(log_prob + target_entropy))


def td3_smooth_target_action(
    actor_action: mx.array,
    noise: mx.array,
    *,
    noise_clip: float,
    action_scale: mx.array,
    action_bias: mx.array,
) -> mx.array:
    """Target policy smoothing: clip noise, then clip action to bounds."""
    clipped_noise = mx.clip(noise, -noise_clip, noise_clip) * action_scale
    low = action_bias - action_scale
    high = action_bias + action_scale
    return mx.clip(actor_action + clipped_noise, low, high)


def td3_bellman_target(
    rewards: mx.array,
    dones: mx.array,
    next_q1: mx.array,
    next_q2: mx.array,
    gamma: float,
) -> mx.array:
    return rewards + (1.0 - dones) * gamma * mx.minimum(next_q1, next_q2)


def td3_critic_loss(q1: mx.array, q2: mx.array, target_q: mx.array) -> mx.array:
    return 0.5 * (mx.mean((q1 - target_q) ** 2) + mx.mean((q2 - target_q) ** 2))


def td3_actor_loss(q_value: mx.array) -> mx.array:
    return -mx.mean(q_value)


# ---------------------------------------------------------------------------
# spaces
# ---------------------------------------------------------------------------

class Box:
    def __init__(self, low, high, shape=None, dtype=np.float32):
        if shape is None:
            low_a = np.asarray(low, dtype=dtype)
            high_a = np.asarray(high, dtype=dtype)
            shape = low_a.shape
            self.low = low_a
            self.high = high_a
        else:
            self.low = np.full(shape, low, dtype=dtype)
            self.high = np.full(shape, high, dtype=dtype)
        self.shape = tuple(shape)
        self.dtype = dtype
        self._rng = np.random.default_rng()

    def sample(self, rng: np.random.Generator | None = None, n: int | None = None):
        """Sample one action, or a batch of ``n`` actions with shape (n, *shape)."""
        rng = rng or self._rng
        if n is None:
            return rng.uniform(self.low, self.high).astype(self.dtype)
        size = (int(n),) + self.shape
        return rng.uniform(self.low, self.high, size=size).astype(self.dtype)


class Discrete:
    def __init__(self, n: int):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.int64
        self._rng = np.random.default_rng()

    def sample(self, rng: np.random.Generator | None = None, n: int | None = None):
        rng = rng or self._rng
        if n is None:
            return int(rng.integers(0, self.n))
        return rng.integers(0, self.n, size=int(n), dtype=self.dtype)
