"""SAC (soft actor-critic) in MLX — continuous actions."""

from __future__ import annotations

from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Callable
import time

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from dbn.reinforcement.common import (
    Logger,
    ReplayBuffer,
    SquashedGaussianActor,
    TwinQ,
    action_scale_bias,
    as_numpy,
    sac_actor_loss,
    sac_alpha_loss,
    sac_bellman_target,
    sac_critic_loss,
    soft_update,
    space_info,
)


class _LogAlpha(nn.Module):
    def __init__(self, init: float = 0.0):
        super().__init__()
        self.log_alpha = mx.array(float(init))

    def __call__(self) -> mx.array:
        return mx.exp(self.log_alpha)


class SAC:
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 3e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        ent_coef: str | float = "auto",
        target_entropy: str | float = "auto",
        tensorboard_log: str | None = None,
        policy_kwargs: dict | None = None,
        verbose: int = 0,
        seed: int | None = None,
    ):
        del policy
        if seed is not None:
            mx.random.seed(seed)
            np.random.seed(seed)

        self.env = env
        self.n_envs = int(getattr(env, "num_envs", getattr(env, "n_envs", 1)))
        self.learning_rate = learning_rate
        self.learning_starts = learning_starts
        self.batch_size = batch_size
        self.tau = tau
        self.gamma = gamma
        self.train_freq = train_freq
        self.gradient_steps = gradient_steps
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs or {}

        self.obs_dim, self.action_dim, continuous = space_info(env)
        if not continuous:
            raise ValueError("SAC requires a continuous action space")
        scale, bias = action_scale_bias(env)

        self._auto_ent = ent_coef == "auto"
        self._fixed_alpha = 0.0 if self._auto_ent else float(ent_coef)
        te = target_entropy
        self._target_entropy = float(-self.action_dim) if te == "auto" else float(te)

        net_arch = tuple(self.policy_kwargs.get("net_arch", (256, 256)))
        self.actor = SquashedGaussianActor(
            self.obs_dim, self.action_dim, net_arch=net_arch, action_scale=scale, action_bias=bias
        )
        self.critic = TwinQ(self.obs_dim, self.action_dim, net_arch=net_arch)
        self.critic_target = TwinQ(self.obs_dim, self.action_dim, net_arch=net_arch)
        mx.eval(self.actor.parameters(), self.critic.parameters())
        self.critic_target.update(self.critic.parameters())
        mx.eval(self.critic_target.parameters())

        self.actor_opt = optim.Adam(learning_rate=learning_rate)
        self.critic_opt = optim.Adam(learning_rate=learning_rate)
        self.log_alpha_mod = _LogAlpha(0.0) if self._auto_ent else None
        self.alpha_opt = optim.Adam(learning_rate=learning_rate) if self._auto_ent else None
        if self.log_alpha_mod is not None:
            mx.eval(self.log_alpha_mod.parameters())

        self.replay = ReplayBuffer(buffer_size, self.obs_dim, self.action_dim)
        self._logger: Logger | None = None
        self.num_timesteps = 0
        self._env_steps = 0
        self._n_updates = 0
        self._ep_info_buffer: deque = deque(maxlen=100)
        self._key = mx.random.key(seed if seed is not None else 0)
        self._compiled: Callable | None = None
        self._build_compiled()

    def _next_key(self) -> mx.array:
        self._key, k = mx.random.split(self._key)
        return k

    def _build_compiled(self) -> None:
        actor, critic, critic_target = self.actor, self.critic, self.critic_target
        actor_opt, critic_opt = self.actor_opt, self.critic_opt
        gamma, tau = self.gamma, self.tau
        auto_ent = self._auto_ent
        target_entropy = self._target_entropy
        fixed_alpha = self._fixed_alpha
        log_alpha_mod = self.log_alpha_mod
        alpha_opt = self.alpha_opt

        state: list[Any] = [actor, critic, critic_target, actor_opt.state, critic_opt.state]
        if auto_ent:
            state.extend([log_alpha_mod, alpha_opt.state])

        def critic_loss_fn(model, obs, actions, target_q):
            q1, q2 = model(obs, actions)
            return sac_critic_loss(q1, q2, target_q)

        def actor_loss_fn(model, obs, alpha, key):
            actions, logp = model(obs, key, deterministic=False)
            q1, q2 = critic(obs, actions)
            return sac_actor_loss(logp, q1, q2, alpha), logp

        def alpha_loss_fn(model, logp):
            return sac_alpha_loss(model.log_alpha, logp, target_entropy)

        critic_vag = nn.value_and_grad(critic, critic_loss_fn)
        actor_vag = nn.value_and_grad(actor, actor_loss_fn)
        alpha_vag = nn.value_and_grad(log_alpha_mod, alpha_loss_fn) if auto_ent else None

        @partial(mx.compile, inputs=state, outputs=state)
        def update_step(obs, actions, rewards, next_obs, dones, key):
            k1, k2 = mx.random.split(key)
            alpha = log_alpha_mod() if auto_ent else mx.array(fixed_alpha)

            next_a, next_logp = actor(next_obs, k1, deterministic=False)
            q1_t, q2_t = critic_target(next_obs, next_a)
            target_q = sac_bellman_target(
                rewards, dones, q1_t, q2_t, next_logp, alpha, gamma
            )

            crit_loss, grads_c = critic_vag(critic, obs, actions, mx.stop_gradient(target_q))
            critic_opt.update(critic, grads_c)

            (actor_loss, logp), grads_a = actor_vag(actor, obs, mx.stop_gradient(alpha), k2)
            actor_opt.update(actor, grads_a)

            alpha_loss = mx.array(0.0)
            if auto_ent:
                alpha_loss, grads_al = alpha_vag(log_alpha_mod, logp)
                alpha_opt.update(log_alpha_mod, grads_al)
                alpha = log_alpha_mod()

            soft_update(critic_target, critic, tau)

            # Diagnostics (no gradient contribution).
            q1_b, q2_b = critic(obs, actions)
            td_err1 = mx.abs(q1_b - mx.stop_gradient(target_q))
            td_err2 = mx.abs(q2_b - mx.stop_gradient(target_q))
            td_error = mx.concatenate([td_err1, td_err2])
            entropy = -mx.mean(logp)

            return crit_loss, actor_loss, alpha_loss, alpha, entropy, q1_b, q2_b, target_q, td_error

        self._compiled = update_step

    def _obs_np(self, obs) -> np.ndarray:
        return as_numpy(obs).astype(np.float32).reshape(self.n_envs, self.obs_dim)

    def predict(self, obs, deterministic: bool = True):
        x = mx.array(np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim))
        actions, _ = self.actor(
            x, None if deterministic else self._next_key(), deterministic=deterministic
        )
        mx.eval(actions)
        a = as_numpy(actions)
        return (a[0] if a.shape[0] == 1 else a), None

    def _train_step(self) -> dict[str, float]:
        b = self.replay.sample(self.batch_size)
        (
            crit_loss,
            actor_loss,
            alpha_loss,
            alpha,
            entropy,
            q1_b,
            q2_b,
            target_q,
            td_error,
        ) = self._compiled(
            b["obs"], b["actions"], b["rewards"], b["next_obs"], b["dones"], self._next_key()
        )
        outs = [
            crit_loss,
            actor_loss,
            alpha_loss,
            alpha,
            entropy,
            q1_b,
            q2_b,
            target_q,
            td_error,
            self.actor.parameters(),
            self.critic.parameters(),
        ]
        if self.log_alpha_mod is not None:
            outs.append(self.log_alpha_mod.parameters())
        mx.eval(*outs)
        self._n_updates += 1

        q_values = mx.concatenate([q1_b, q2_b])
        return {
            "actor_loss": float(actor_loss),
            "critic_loss": float(crit_loss),
            "ent_coef": float(alpha),
            "ent_coef_loss": float(alpha_loss),
            "entropy": float(entropy),
            "q_value_mean": float(mx.mean(q_values)),
            "q_value_std": float(mx.sqrt(mx.mean((q_values - mx.mean(q_values)) ** 2))),
            "target_q_mean": float(mx.mean(target_q)),
            "td_error_abs_mean": float(mx.mean(td_error)),
            "td_error_std": float(mx.sqrt(mx.mean((td_error - mx.mean(td_error)) ** 2))),
        }

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 4,
        tb_log_name: str = "SAC",
        reset_num_timesteps: bool = True,
        progress_bar: bool = True,
    ):
        if reset_num_timesteps:
            self.num_timesteps = 0
        if self.tensorboard_log is not None and self._logger is None:
            self._logger = Logger(Path(self.tensorboard_log) / f"{tb_log_name}_1")

        train_start = time.time()

        out = self.env.reset()
        obs = self._obs_np(out[0] if isinstance(out, tuple) else out)
        train_info: dict[str, float] = {}
        pbar = tqdm(
            total=total_timesteps, desc="SAC", unit="step", disable=not progress_bar, dynamic_ncols=True
        )
        try:
            while self.num_timesteps < total_timesteps:
                if self.num_timesteps < self.learning_starts:
                    actions = np.asarray(
                        self.env.action_space.sample(n=self.n_envs), dtype=np.float32
                    )
                    if actions.ndim == 1:
                        actions = actions.reshape(self.n_envs, -1)
                else:
                    a_mx, _ = self.actor(mx.array(obs), self._next_key(), deterministic=False)
                    mx.eval(a_mx)
                    actions = as_numpy(a_mx).astype(np.float32)

                step_out = self.env.step(actions[0] if self.n_envs == 1 else actions)
                if len(step_out) == 5:
                    next_obs, rewards, terminations, truncations, infos = step_out
                else:
                    next_obs, rewards, dones, infos = step_out
                    terminations = dones
                    truncations = np.zeros(self.n_envs, dtype=dones.dtype)
                next_obs_np = self._obs_np(next_obs)
                trunc_np = np.atleast_1d(np.asarray(truncations, dtype=bool)).reshape(self.n_envs)
                for i in np.flatnonzero(trunc_np):
                    term_obs = infos[i].get("terminal_observation") if isinstance(infos, (list, tuple)) and i < len(infos) else infos.get("terminal_observation") if isinstance(infos, dict) else None
                    if term_obs is not None:
                        next_obs_np[i] = np.asarray(term_obs, dtype=np.float32).reshape(self.obs_dim)
                dones = np.atleast_1d(np.asarray(terminations, dtype=np.float32))
                rewards = np.atleast_1d(np.asarray(rewards, dtype=np.float32))

                self.replay.add_batch(obs, actions, rewards, next_obs_np, dones)
                if isinstance(infos, dict):
                    if "episode" in infos:
                        self._ep_info_buffer.append(infos["episode"])
                else:
                    for info in infos:
                        if info and "episode" in info:
                            self._ep_info_buffer.append(info["episode"])

                obs = next_obs_np
                self.num_timesteps += self.n_envs
                self._env_steps += 1
                pbar.update(self.n_envs)

                if self.num_timesteps >= self.learning_starts and self._env_steps % self.train_freq == 0:
                    for _ in range(self.gradient_steps):
                        train_info = self._train_step()

                if train_info and self.num_timesteps % max(log_interval, 1) == 0:
                    ep_rew = (
                        float(np.mean([e["r"] for e in self._ep_info_buffer]))
                        if self._ep_info_buffer
                        else float("nan")
                    )
                    pbar.set_postfix(
                        crit=f"{train_info.get('critic_loss', 0):.2f}",
                        ep_rew=f"{ep_rew:.1f}" if ep_rew == ep_rew else "nan",
                        refresh=False,
                    )
                    if self._logger is not None:
                        metrics = {f"train/{k}": v for k, v in train_info.items()}
                        if self._ep_info_buffer:
                            metrics["rollout/ep_rew_mean"] = ep_rew
                            metrics["rollout/ep_len_mean"] = float(
                                np.mean([e["l"] for e in self._ep_info_buffer])
                            )
                        elapsed = time.time() - train_start
                        metrics["time/total_timesteps"] = float(self.num_timesteps)
                        metrics["time/time_elapsed"] = elapsed
                        metrics["time/fps"] = float(self.num_timesteps) / max(elapsed, 1e-6)
                        self._logger.record(metrics, self.num_timesteps)
                        self._logger.flush()
        finally:
            pbar.close()
            if self._logger is not None:
                self._logger.flush()
        return self
