"""CartPole (classic control) in MLX — Gymnasium-like API, no gym dependency."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np

from dbn.reinforcement.envs.spaces import Box, Discrete

# Classic CartPole-v1 constants
_GRAVITY = 9.8
_MASSCART = 1.0
_MASSPOLE = 0.1
_TOTAL_MASS = _MASSCART + _MASSPOLE
_LENGTH = 0.5  # half-pole
_POLEMASS_LENGTH = _MASSPOLE * _LENGTH
_FORCE_MAG = 10.0
_TAU = 0.02
_THETA_THRESHOLD = 12 * 2 * math.pi / 360
_X_THRESHOLD = 2.4


def _cartpole_step(state: mx.array, action: mx.array) -> tuple[mx.array, mx.array, mx.array]:
    """Vectorized dynamics. state (N,4), action (N,) int → next_state, reward, terminated."""
    x, x_dot, theta, theta_dot = state[:, 0], state[:, 1], state[:, 2], state[:, 3]
    force = mx.where(action == 1, _FORCE_MAG, -_FORCE_MAG)
    costheta = mx.cos(theta)
    sintheta = mx.sin(theta)
    temp = (force + _POLEMASS_LENGTH * theta_dot**2 * sintheta) / _TOTAL_MASS
    thetaacc = (_GRAVITY * sintheta - costheta * temp) / (
        _LENGTH * (4.0 / 3.0 - _MASSPOLE * costheta**2 / _TOTAL_MASS)
    )
    xacc = temp - _POLEMASS_LENGTH * thetaacc * costheta / _TOTAL_MASS
    x = x + _TAU * x_dot
    x_dot = x_dot + _TAU * xacc
    theta = theta + _TAU * theta_dot
    theta_dot = theta_dot + _TAU * thetaacc
    next_state = mx.stack([x, x_dot, theta, theta_dot], axis=-1)
    terminated = (
        (mx.abs(x) > _X_THRESHOLD)
        | (mx.abs(theta) > _THETA_THRESHOLD)
    ).astype(mx.float32)
    reward = mx.ones((state.shape[0],), dtype=mx.float32)
    return next_state, reward, terminated


# Fuse + compile dynamics for repeated steps (vectorized over envs).
_step_compiled = mx.compile(_cartpole_step)


_EMPTY_INFO: dict = {}


class CartPole:
    """
    Gymnasium-like CartPole.

    reset() -> (obs, info)
    step(action) -> (obs, reward, terminated, truncated, info)

    Set ``n_envs`` > 1 for independent parallel carts (batched MLX state).
    """

    def __init__(self, n_envs: int = 1, max_episode_steps: int = 500, seed: int | None = None):
        self.n_envs = int(n_envs)
        self.num_envs = self.n_envs
        self.max_episode_steps = int(max_episode_steps)
        self.observation_space = Box(
            low=np.array([-4.8, -np.inf, -_THETA_THRESHOLD * 2, -np.inf], dtype=np.float32),
            high=np.array([4.8, np.inf, _THETA_THRESHOLD * 2, np.inf], dtype=np.float32),
        )
        self.action_space = Discrete(2)
        self._key = mx.random.key(seed if seed is not None else 0)
        self._state = mx.zeros((self.n_envs, 4))
        self._steps = mx.zeros((self.n_envs,), dtype=mx.int32)
        self._ep_ret = np.zeros(self.n_envs, dtype=np.float64)
        self._ep_len = np.zeros(self.n_envs, dtype=np.int64)

    def _split_key(self) -> mx.array:
        self._key, k = mx.random.split(self._key)
        return k

    def _sample_state(self) -> mx.array:
        return mx.random.uniform(
            key=self._split_key(),
            shape=(self.n_envs, 4),
            low=-0.05,
            high=0.05,
        )

    def reset(self, *, seed: int | None = None):
        if seed is not None:
            self._key = mx.random.key(seed)
        self._state = self._sample_state()
        self._steps = mx.zeros((self.n_envs,), dtype=mx.int32)
        self._ep_ret[:] = 0.0
        self._ep_len[:] = 0
        mx.eval(self._state)
        obs = np.array(self._state, dtype=np.float32)
        if self.n_envs == 1:
            return obs[0], {}
        return obs, [{} for _ in range(self.n_envs)]

    def step(self, action):
        a = np.atleast_1d(np.asarray(action, dtype=np.int32)).reshape(self.n_envs)
        action_mx = mx.array(a)
        next_state, reward, terminated = _step_compiled(self._state, action_mx)
        self._steps = self._steps + 1
        truncated = (self._steps >= self.max_episode_steps).astype(mx.float32)
        # don't truncate if already terminated
        truncated = truncated * (1.0 - terminated)
        done = ((terminated + truncated) > 0).astype(mx.float32)

        mx.eval(next_state, reward, terminated, truncated, done)
        rew_np = np.array(reward, dtype=np.float32)
        term_np = np.array(terminated) > 0
        trunc_np = np.array(truncated) > 0
        done_np = np.array(done) > 0

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
            # Shared empty sentinel — avoid allocating n_envs dicts every step.
            infos = [_EMPTY_INFO] * self.n_envs
            if done_np.any():
                infos = list(infos)
                for i in np.flatnonzero(done_np):
                    infos[i] = {"episode": {"r": float(self._ep_ret[i]), "l": int(self._ep_len[i])}}
                self._ep_ret[done_np] = 0.0
                self._ep_len[done_np] = 0

        # auto-reset finished envs (VecEnv-style)
        if np.any(done_np):
            fresh = self._sample_state()
            mask = mx.array(done_np.astype(np.float32))[:, None]
            next_state = next_state * (1.0 - mask) + fresh * mask
            self._steps = mx.where(mx.array(done_np), mx.zeros_like(self._steps), self._steps)
            mx.eval(next_state, self._steps)

        self._state = next_state
        obs = np.array(self._state, dtype=np.float32)

        if self.n_envs == 1:
            return obs[0], float(rew_np[0]), bool(term_np[0]), bool(trunc_np[0]), infos
        return obs, rew_np, term_np, trunc_np, infos
