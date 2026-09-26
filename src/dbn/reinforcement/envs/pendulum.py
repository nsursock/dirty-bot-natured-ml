"""Pendulum (classic control) in MLX — Gymnasium-like API, no gym dependency."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from dbn.reinforcement.envs.spaces import Box

_MAX_SPEED = 8.0
_MAX_TORQUE = 2.0
_DT = 0.05
_G = 10.0
_M = 1.0
_L = 1.0


def _pendulum_step(state: mx.array, action: mx.array) -> tuple[mx.array, mx.array]:
    """state (N,2)=[theta, theta_dot], action (N,1) → next_state, reward."""
    theta, theta_dot = state[:, 0], state[:, 1]
    u = mx.clip(action[:, 0], -_MAX_TORQUE, _MAX_TORQUE)
    reward = -(theta**2 + 0.1 * theta_dot**2 + 0.001 * (u**2))
    new_dot = theta_dot + (
        3.0 * _G / (2.0 * _L) * mx.sin(theta) + 3.0 / (_M * _L**2) * u
    ) * _DT
    new_dot = mx.clip(new_dot, -_MAX_SPEED, _MAX_SPEED)
    new_theta = theta + new_dot * _DT
    next_state = mx.stack([new_theta, new_dot], axis=-1)
    return next_state, reward


_step_compiled = mx.compile(_pendulum_step)


def _obs_from_state(state: mx.array) -> mx.array:
    theta, theta_dot = state[:, 0], state[:, 1]
    return mx.stack([mx.cos(theta), mx.sin(theta), theta_dot], axis=-1)


_EMPTY_INFO: dict = {}


class Pendulum:
    """
    Gymnasium-like Pendulum-v1.

    Observation: [cosθ, sinθ, θ̇]  Action: torque ∈ [-2, 2]
    """

    def __init__(self, n_envs: int = 1, max_episode_steps: int = 200, seed: int | None = None):
        self.n_envs = int(n_envs)
        self.num_envs = self.n_envs
        self.max_episode_steps = int(max_episode_steps)
        high = np.array([1.0, 1.0, _MAX_SPEED], dtype=np.float32)
        self.observation_space = Box(low=-high, high=high)
        self.action_space = Box(
            low=-_MAX_TORQUE, high=_MAX_TORQUE, shape=(1,), dtype=np.float32
        )
        self._key = mx.random.key(seed if seed is not None else 0)
        self._state = mx.zeros((self.n_envs, 2))
        self._steps = mx.zeros((self.n_envs,), dtype=mx.int32)
        self._ep_ret = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)

    def _split_key(self) -> mx.array:
        self._key, k = mx.random.split(self._key)
        return k

    def _sample_state(self) -> mx.array:
        # θ ~ U[-π, π], θ̇ ~ U[-1, 1]
        k1, k2 = mx.random.split(self._split_key())
        theta = mx.random.uniform(key=k1, shape=(self.n_envs,), low=-math.pi, high=math.pi)
        theta_dot = mx.random.uniform(key=k2, shape=(self.n_envs,), low=-1.0, high=1.0)
        return mx.stack([theta, theta_dot], axis=-1)

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._key = mx.random.key(seed)
        self._state = self._sample_state()
        self._steps = mx.zeros((self.n_envs,), dtype=mx.int32)
        self._ep_ret[:] = 0.0
        self._ep_len[:] = 0
        obs = _obs_from_state(self._state)
        mx.eval(self._state, obs)
        obs_np = np.array(obs, dtype=np.float32)
        if self.n_envs == 1:
            return obs_np[0], {}
        return obs_np, [{} for _ in range(self.n_envs)]

    def step(self, action):
        a = np.asarray(action, dtype=np.float32)
        if a.ndim == 0:
            a = a.reshape(1, 1)
        elif a.ndim == 1:
            a = a.reshape(self.n_envs, 1)
        action_mx = mx.array(a)
        next_state, reward = _step_compiled(self._state, action_mx)
        # keep θ unwrapped for dynamics continuity (obs uses cos/sin)
        self._steps = self._steps + 1
        truncated = (self._steps >= self.max_episode_steps).astype(mx.float32)
        terminated = mx.zeros((self.n_envs,), dtype=mx.float32)  # no early terminate
        done = truncated

        mx.eval(next_state, reward, done)
        rew_np = np.array(reward, dtype=np.float32)
        trunc_np = np.array(truncated) > 0
        done_np = np.array(done) > 0
        term_np = np.zeros(self.n_envs, dtype=bool)

        self._ep_ret += rew_np
        self._ep_len += 1

        if self.n_envs == 1:
            info: dict = {}
            if done_np[0]:
                info["episode"] = {"r": float(self._ep_ret[0]), "l": int(self._ep_len[0])}
                self._ep_ret[0] = 0.0
                self._ep_len[0] = 0
            infos: list[dict] | dict = info
        else:
            infos = [_EMPTY_INFO] * self.n_envs
            if done_np.any():
                infos = list(infos)
                for i in np.flatnonzero(done_np):
                    infos[i] = {"episode": {"r": float(self._ep_ret[i]), "l": int(self._ep_len[i])}}
                self._ep_ret[done_np] = 0.0
                self._ep_len[done_np] = 0

        if np.any(done_np):
            fresh = self._sample_state()
            mask = mx.array(done_np.astype(np.float32))[:, None]
            next_state = next_state * (1.0 - mask) + fresh * mask
            self._steps = mx.where(mx.array(done_np), mx.zeros_like(self._steps), self._steps)
            mx.eval(next_state, self._steps)

        self._state = next_state
        obs = np.array(_obs_from_state(self._state), dtype=np.float32)

        if self.n_envs == 1:
            return obs[0], float(rew_np[0]), bool(term_np[0]), bool(trunc_np[0]), infos
        return obs, rew_np, term_np, trunc_np, infos
