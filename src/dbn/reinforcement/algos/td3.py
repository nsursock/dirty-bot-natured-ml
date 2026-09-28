"""TD3 (twin delayed DDPG) in MLX — continuous actions."""

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
    DeterministicActor,
    Logger,
    ReplayBuffer,
    TwinQ,
    action_scale_bias,
    as_numpy,
    soft_update,
    space_info,
    td3_actor_loss,
    td3_bellman_target,
    td3_critic_loss,
    td3_smooth_target_action,
)


class TD3:
    def __init__(
        self,
        policy: str,
        env,
        learning_rate: float = 1e-3,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 256,
        tau: float = 0.005,
        gamma: float = 0.99,
        train_freq: int = 1,
        gradient_steps: int = 1,
        policy_delay: int = 2,
        target_policy_noise: float = 0.2,
        target_noise_clip: float = 0.5,
        action_noise_std: float = 0.1,
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
        self.policy_delay = policy_delay
        self.target_policy_noise = target_policy_noise
        self.target_noise_clip = target_noise_clip
        self.action_noise_std = action_noise_std
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs or {}

        self.obs_dim, self.action_dim, continuous = space_info(env)
        if not continuous:
            raise ValueError("TD3 requires a continuous action space")
        self.action_scale, self.action_bias = action_scale_bias(env)
        self._act_low = as_numpy(self.action_bias - self.action_scale)
        self._act_high = as_numpy(self.action_bias + self.action_scale)

        net_arch = tuple(self.policy_kwargs.get("net_arch", (256, 256)))
        self.actor = DeterministicActor(
            self.obs_dim,
            self.action_dim,
            net_arch=net_arch,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        )
        self.actor_target = DeterministicActor(
            self.obs_dim,
            self.action_dim,
            net_arch=net_arch,
            action_scale=self.action_scale,
            action_bias=self.action_bias,
        )
        self.critic = TwinQ(self.obs_dim, self.action_dim, net_arch=net_arch)
        self.critic_target = TwinQ(self.obs_dim, self.action_dim, net_arch=net_arch)
        mx.eval(self.actor.parameters(), self.critic.parameters())
        self.actor_target.update(self.actor.parameters())
        self.critic_target.update(self.critic.parameters())
        mx.eval(self.actor_target.parameters(), self.critic_target.parameters())

        self.actor_opt = optim.Adam(learning_rate=learning_rate)
        self.critic_opt = optim.Adam(learning_rate=learning_rate)

        self.replay = ReplayBuffer(buffer_size, self.obs_dim, self.action_dim)
        self._logger: Logger | None = None
        self.num_timesteps = 0
        self._env_steps = 0
        self._n_updates = 0
        self._ep_info_buffer: deque = deque(maxlen=100)
        self._key = mx.random.key(seed if seed is not None else 0)
        self._compiled_critic: Callable | None = None
        self._compiled_actor: Callable | None = None
        self._build_compiled()

    def _next_key(self) -> mx.array:
        self._key, k = mx.random.split(self._key)
        return k

    def _build_compiled(self) -> None:
        actor, actor_target = self.actor, self.actor_target
        critic, critic_target = self.critic, self.critic_target
        actor_opt, critic_opt = self.actor_opt, self.critic_opt
        gamma, tau = self.gamma, self.tau
        noise_std, noise_clip = self.target_policy_noise, self.target_noise_clip

        critic_state: list[Any] = [critic, critic_target, actor_target, critic_opt.state]
        actor_state: list[Any] = [actor, critic, actor_target, critic_target, actor_opt.state]

        def critic_loss_fn(model, obs, actions, target_q):
            q1, q2 = model(obs, actions)
            return td3_critic_loss(q1, q2, target_q)

        def actor_loss_fn(model, obs):
            return td3_actor_loss(critic(obs, model(obs))[0])

        critic_vag = nn.value_and_grad(critic, critic_loss_fn)
        actor_vag = nn.value_and_grad(actor, actor_loss_fn)

        @partial(mx.compile, inputs=critic_state, outputs=critic_state)
        def critic_step(obs, actions, rewards, next_obs, dones, key):
            scale = mx.stop_gradient(actor_target.action_scale)
            bias = mx.stop_gradient(actor_target.action_bias)
            raw_noise = mx.random.normal(key=key, shape=actions.shape) * noise_std
            next_a = td3_smooth_target_action(
                actor_target(next_obs),
                raw_noise,
                noise_clip=noise_clip,
                action_scale=scale,
                action_bias=bias,
            )
            q1_t, q2_t = critic_target(next_obs, next_a)
            target_q = td3_bellman_target(rewards, dones, q1_t, q2_t, gamma)
            loss, grads = critic_vag(critic, obs, actions, mx.stop_gradient(target_q))
            critic_opt.update(critic, grads)

            # Diagnostics (no gradient contribution).
            q1_b, q2_b = critic(obs, actions)
            td_err1 = mx.abs(q1_b - mx.stop_gradient(target_q))
            td_err2 = mx.abs(q2_b - mx.stop_gradient(target_q))
            td_error = mx.concatenate([td_err1, td_err2])
            clipped_noise = mx.clip(raw_noise, -noise_clip, noise_clip) * scale
            clip_fraction = mx.mean(mx.abs(raw_noise) > noise_clip).astype(mx.float32)

            return loss, q1_b, q2_b, target_q, td_error, clipped_noise, clip_fraction

        @partial(mx.compile, inputs=actor_state, outputs=actor_state)
        def actor_step(obs):
            loss, grads = actor_vag(actor, obs)
            actor_opt.update(actor, grads)
            soft_update(actor_target, actor, tau)
            soft_update(critic_target, critic, tau)
            return loss

        self._compiled_critic = critic_step
        self._compiled_actor = actor_step

    def _obs_np(self, obs) -> np.ndarray:
        return as_numpy(obs).astype(np.float32).reshape(self.n_envs, self.obs_dim)

    def predict(self, obs, deterministic: bool = True):
        x = mx.array(np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim))
        actions = self.actor(x)
        if not deterministic:
            noise = mx.random.normal(key=self._next_key(), shape=actions.shape) * self.action_noise_std
            actions = mx.clip(
                actions + noise * self.action_scale,
                self.action_bias - self.action_scale,
                self.action_bias + self.action_scale,
            )
        mx.eval(actions)
        a = as_numpy(actions)
        return (a[0] if a.shape[0] == 1 else a), None

    def _train_step(self) -> dict[str, float]:
        b = self.replay.sample(self.batch_size)
        (
            crit_loss,
            q1_b,
            q2_b,
            target_q,
            td_error,
            applied_noise,
            clip_fraction,
        ) = self._compiled_critic(
            b["obs"], b["actions"], b["rewards"], b["next_obs"], b["dones"], self._next_key()
        )
        mx.eval(
            crit_loss,
            q1_b,
            q2_b,
            target_q,
            td_error,
            applied_noise,
            clip_fraction,
            self.critic.parameters(),
        )
        q_values = mx.concatenate([q1_b, q2_b])
        td_mean = mx.mean(td_error)
        info = {
            "critic_loss": float(crit_loss),
            "actor_loss": 0.0,
            "q_value_mean": float(mx.mean(q_values)),
            "q_value_std": float(mx.sqrt(mx.mean((q_values - mx.mean(q_values)) ** 2))),
            "target_q_mean": float(mx.mean(target_q)),
            "td_error_abs_mean": float(td_mean),
            "td_error_std": float(mx.sqrt(mx.mean((td_error - td_mean) ** 2))),
            "target_noise_std": float(mx.std(applied_noise.reshape(-1))),
            "target_noise_clip_fraction": float(clip_fraction),
            "actor_update_fraction": 0.0,
        }
        self._n_updates += 1
        if self._n_updates % self.policy_delay == 0:
            actor_loss = self._compiled_actor(b["obs"])
            mx.eval(actor_loss, self.actor.parameters(), self.actor_target.parameters(), self.critic_target.parameters())
            info["actor_loss"] = float(actor_loss)
            info["actor_update_fraction"] = 1.0
        return info

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 4,
        tb_log_name: str = "TD3",
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
            total=total_timesteps, desc="TD3", unit="step", disable=not progress_bar, dynamic_ncols=True
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
                    a_mx = self.actor(mx.array(obs))
                    noise = (
                        mx.random.normal(key=self._next_key(), shape=a_mx.shape)
                        * self.action_noise_std
                        * self.action_scale
                    )
                    a_mx = mx.clip(
                        a_mx + noise,
                        self.action_bias - self.action_scale,
                        self.action_bias + self.action_scale,
                    )
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
