"""PPO in MLX — concise API, vmapped + compiled update."""

from __future__ import annotations

from collections import deque
from functools import partial
from pathlib import Path
from typing import Any, Callable

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from tqdm import tqdm

from dbn.reinforcement.common import (
    Logger,
    MlpPolicy,
    RolloutBuffer,
    as_numpy,
    categorical_entropy,
    categorical_log_prob,
    categorical_sample,
    clip_grad_norm,
    explained_variance,
    gaussian_entropy,
    gaussian_log_prob,
    gaussian_sample,
    normalize_advantages,
    ppo_entropy_loss,
    ppo_policy_loss,
    ppo_ratio,
    ppo_value_loss,
    space_info,
    tree_flatten_dict,
)


class PPO:
    """Proximal Policy Optimization (clipped surrogate)."""

    def __init__(
        self,
        policy: str | type[MlpPolicy],
        env,
        learning_rate: float = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        clip_range_vf: float | None = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float | None = None,
        tensorboard_log: str | None = None,
        policy_kwargs: dict | None = None,
        verbose: int = 0,
        seed: int | None = None,
        _init_setup_model: bool = True,
    ):
        if seed is not None:
            mx.random.seed(seed)
            np.random.seed(seed)

        self.env = env
        self.n_envs = int(getattr(env, "num_envs", getattr(env, "n_envs", 1)))
        self.learning_rate = learning_rate
        self.n_steps = n_steps
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        self.tensorboard_log = tensorboard_log
        self.policy_kwargs = policy_kwargs or {}
        self.verbose = verbose
        self.seed = seed

        self.obs_dim, self.action_dim, self.continuous = space_info(env)
        self.policy_class = MlpPolicy if policy in ("MlpPolicy", MlpPolicy) else policy
        self.policy: MlpPolicy | None = None
        self.optimizer: optim.Optimizer | None = None
        self.buffer: RolloutBuffer | None = None
        self._logger: Logger | None = None
        self.num_timesteps = 0
        self._n_updates = 0
        self._ep_info_buffer: deque = deque(maxlen=100)
        self._key = mx.random.key(seed if seed is not None else 0)
        self._compiled_update: Callable | None = None

        if _init_setup_model:
            self._setup_model()

    def _next_key(self) -> mx.array:
        self._key, k = mx.random.split(self._key)
        return k

    def _setup_model(self) -> None:
        self.policy = self.policy_class(
            self.obs_dim,
            self.action_dim,
            continuous=self.continuous,
            **self.policy_kwargs,
        )
        mx.eval(self.policy.parameters())
        self.optimizer = optim.Adam(learning_rate=self.learning_rate)
        self.buffer = RolloutBuffer(
            self.n_steps,
            self.n_envs,
            (self.obs_dim,),
            self.action_dim,
            continuous=self.continuous,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
        )
        self._build_compiled_update()

    def _build_compiled_update(self) -> None:
        assert self.policy is not None and self.optimizer is not None
        policy = self.policy
        optimizer = self.optimizer
        clip_range = self.clip_range
        clip_range_vf = self.clip_range_vf
        ent_coef = self.ent_coef
        vf_coef = self.vf_coef
        max_grad_norm = self.max_grad_norm
        continuous = self.continuous
        normalize_advantage = self.normalize_advantage

        def loss_fn(model, obs, actions, old_logp, advantages, returns, old_values):
            if normalize_advantage:
                advantages = normalize_advantages(advantages)
            logits_or_mean, values = model(obs)
            if continuous:
                logp = gaussian_log_prob(logits_or_mean, model.log_std, actions)
                ent = mx.full(logp.shape, gaussian_entropy(model.log_std))
            else:
                logp = categorical_log_prob(logits_or_mean, actions)
                ent = categorical_entropy(logits_or_mean)
            ratio = ppo_ratio(logp, old_logp)
            policy_loss = ppo_policy_loss(logp, old_logp, advantages, clip_range)
            value_loss = ppo_value_loss(values, returns, old_values, clip_range_vf)
            entropy_loss = ppo_entropy_loss(ent)
            loss = policy_loss + vf_coef * value_loss + ent_coef * entropy_loss
            approx_kl = mx.mean(old_logp - logp)
            clip_frac = mx.mean((mx.abs(ratio - 1.0) > clip_range).astype(mx.float32))
            return loss, (policy_loss, value_loss, entropy_loss, approx_kl, clip_frac)

        loss_and_grad = nn.value_and_grad(policy, loss_fn)
        state: list[Any] = [policy, optimizer.state]

        @partial(mx.compile, inputs=state, outputs=state)
        def update_step(obs, actions, old_logp, advantages, returns, old_values):
            (loss, metrics), grads = loss_and_grad(
                policy, obs, actions, old_logp, advantages, returns, old_values
            )
            grads = clip_grad_norm(grads, max_grad_norm)
            optimizer.update(policy, grads)
            return loss, metrics

        self._compiled_update = update_step

    def _obs_to_mx(self, obs) -> mx.array:
        arr = as_numpy(obs).astype(np.float32).reshape(self.n_envs, self.obs_dim)
        return mx.array(arr)

    def _reset_env(self):
        out = self.env.reset()
        obs, info = out if isinstance(out, tuple) else (out, {})
        return self._obs_to_mx(obs), info

    def _step_env(self, actions: mx.array):
        a = as_numpy(actions)
        if not self.continuous:
            a = a.astype(np.int64)
            if self.n_envs == 1 and a.ndim == 1 and a.shape[0] == 1:
                a = int(a[0])
        elif self.n_envs == 1 and a.ndim == 2 and a.shape[0] == 1:
            a = a[0]
        out = self.env.step(a)
        if len(out) == 5:
            obs, rewards, terminations, truncations, infos = out
            dones = np.logical_or(terminations, truncations)
        else:
            obs, rewards, dones, infos = out
            terminations = dones
            truncations = np.zeros(self.n_envs, dtype=dones.dtype)
        rewards = np.asarray(rewards, dtype=np.float32).reshape(self.n_envs)
        dones = np.asarray(dones, dtype=np.float32).reshape(self.n_envs)
        terminations = np.asarray(terminations, dtype=np.float32).reshape(self.n_envs)
        truncations = np.asarray(truncations, dtype=np.float32).reshape(self.n_envs)
        if not isinstance(infos, (list, tuple)):
            infos = [infos]
        return (
            self._obs_to_mx(obs),
            mx.array(rewards),
            mx.array(terminations),
            mx.array(truncations),
            mx.array(dones),
            infos,
        )

    def predict(self, obs, deterministic: bool = False):
        assert self.policy is not None
        x = mx.array(np.asarray(obs, dtype=np.float32).reshape(-1, self.obs_dim))
        logits_or_mean, _ = self.policy(x)
        if self.continuous:
            actions = (
                logits_or_mean
                if deterministic
                else gaussian_sample(logits_or_mean, self.policy.log_std, self._next_key())
            )
        else:
            actions = (
                mx.argmax(logits_or_mean, axis=-1)
                if deterministic
                else categorical_sample(logits_or_mean, self._next_key())
            )
        mx.eval(actions)
        a = as_numpy(actions)
        return (a[0] if a.shape[0] == 1 else a), None

    def _collect_rollouts(self, obs: mx.array) -> mx.array:
        assert self.policy is not None and self.buffer is not None
        self.buffer.reset()
        for _ in range(self.n_steps):
            logits_or_mean, values = self.policy(obs)
            if self.continuous:
                actions = gaussian_sample(logits_or_mean, self.policy.log_std, self._next_key())
                logp = gaussian_log_prob(logits_or_mean, self.policy.log_std, actions)
            else:
                actions = categorical_sample(logits_or_mean, self._next_key())
                logp = categorical_log_prob(logits_or_mean, actions)
            mx.eval(actions, values, logp)
            next_obs, rewards, terminations, truncations, dones, infos = self._step_env(actions)
            terminal_values = self._make_terminal_values(truncations, infos)
            self.buffer.add(obs, actions, rewards, dones, terminations, truncations, terminal_values, values, logp)
            self.num_timesteps += self.n_envs
            for info in infos:
                if info and "episode" in info:
                    self._ep_info_buffer.append(info["episode"])
            obs = next_obs
        last_values = self.policy.value(obs)
        mx.eval(last_values)
        self.buffer.compute_returns_and_advantage(last_values)
        return obs

    def _make_terminal_values(self, truncations: mx.array, infos: list[dict]) -> mx.array:
        assert self.policy is not None
        terminal_values = mx.zeros((self.n_envs,), dtype=mx.float32)
        trunc_np = np.asarray(truncations).reshape(self.n_envs)
        if trunc_np.any():
            obs_list: list[np.ndarray] = []
            idx_list: list[int] = []
            for i in np.flatnonzero(trunc_np):
                term_obs = infos[i].get("terminal_observation") if i < len(infos) else None
                if term_obs is not None:
                    obs_list.append(np.asarray(term_obs, dtype=np.float32).reshape(self.obs_dim))
                    idx_list.append(i)
            if obs_list:
                batch = mx.array(np.stack(obs_list, axis=0))
                vals = self.policy.value(batch)
                mx.eval(vals)
                vals_np = np.asarray(vals).reshape(-1)
                terminal_values_np = np.asarray(terminal_values, dtype=np.float32)
                terminal_values_np[idx_list] = vals_np
                terminal_values = mx.array(terminal_values_np)
        return terminal_values

    def _train(self) -> dict[str, float]:
        assert self.buffer is not None and self._compiled_update is not None
        totals = {
            "policy_gradient_loss": 0.0,
            "value_loss": 0.0,
            "entropy_loss": 0.0,
            "approx_kl": 0.0,
            "clip_fraction": 0.0,
            "loss": 0.0,
        }
        n_updates = 0
        early_stop = False
        for _ in range(self.n_epochs):
            if early_stop:
                break
            for batch in self.buffer.get(self.batch_size):
                loss, metrics = self._compiled_update(
                    batch["obs"],
                    batch["actions"],
                    batch["old_log_prob"],
                    batch["advantages"],
                    batch["returns"],
                    batch["old_values"],
                )
                mx.eval(loss, *metrics, self.policy.parameters(), self.optimizer.state)
                pl, vl, el, kl, cf = metrics
                totals["loss"] += float(loss)
                totals["policy_gradient_loss"] += float(pl)
                totals["value_loss"] += float(vl)
                totals["entropy_loss"] += float(el)
                totals["approx_kl"] += float(kl)
                totals["clip_fraction"] += float(cf)
                n_updates += 1
                if self.target_kl is not None and float(kl) > 1.5 * self.target_kl:
                    early_stop = True
                    break
        self._n_updates += n_updates
        return {k: v / max(n_updates, 1) for k, v in totals.items()}

    def learn(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = True,
    ):
        if reset_num_timesteps:
            self.num_timesteps = 0
        if self.tensorboard_log is not None and self._logger is None:
            self._logger = Logger(Path(self.tensorboard_log) / f"{tb_log_name}_1")

        obs, _ = self._reset_env()
        iteration = 0
        pbar = tqdm(
            total=total_timesteps,
            desc="PPO",
            unit="step",
            disable=not progress_bar,
            dynamic_ncols=True,
        )
        try:
            while self.num_timesteps < total_timesteps:
                steps_before = self.num_timesteps
                obs = self._collect_rollouts(obs)
                train_info = self._train()
                iteration += 1
                assert self.buffer is not None
                ev = explained_variance(
                    self.buffer.values.reshape(-1), self.buffer.returns.reshape(-1)
                )
                ep_rew = (
                    float(np.mean([e["r"] for e in self._ep_info_buffer]))
                    if self._ep_info_buffer
                    else float("nan")
                )
                ep_len = (
                    float(np.mean([e["l"] for e in self._ep_info_buffer]))
                    if self._ep_info_buffer
                    else float("nan")
                )
                pbar.update(self.num_timesteps - steps_before)
                pbar.set_postfix(
                    loss=f"{train_info['loss']:.3f}",
                    kl=f"{train_info['approx_kl']:.3f}",
                    ep_rew=f"{ep_rew:.1f}" if ep_rew == ep_rew else "nan",
                    refresh=False,
                )

                metrics = {
                    **{f"train/{k}": v for k, v in train_info.items()},
                    "train/explained_variance": ev,
                    "train/n_updates": float(self._n_updates),
                    "train/learning_rate": float(self.learning_rate),
                }
                if self._ep_info_buffer:
                    metrics["rollout/ep_rew_mean"] = ep_rew
                    metrics["rollout/ep_len_mean"] = ep_len
                if self._logger is not None and iteration % log_interval == 0:
                    self._logger.record(metrics, self.num_timesteps)
                    self._logger.flush()
        finally:
            pbar.close()
            if self._logger is not None:
                self._logger.flush()
        return self

    def save(self, path: str) -> None:
        assert self.policy is not None
        path = path if path.endswith(".npz") else path + ".npz"
        weights = {k: np.array(v) for k, v in tree_flatten_dict(self.policy.parameters()).items()}
        net_arch = list(self.policy_kwargs.get("net_arch", (64, 64)))
        meta = {
            "_obs_dim": np.array(self.obs_dim),
            "_action_dim": np.array(self.action_dim),
            "_continuous": np.array(int(self.continuous)),
            "_learning_rate": np.array(self.learning_rate),
            "_n_steps": np.array(self.n_steps),
            "_batch_size": np.array(self.batch_size),
            "_n_epochs": np.array(self.n_epochs),
            "_gamma": np.array(self.gamma),
            "_gae_lambda": np.array(self.gae_lambda),
            "_clip_range": np.array(self.clip_range),
            "_ent_coef": np.array(self.ent_coef),
            "_vf_coef": np.array(self.vf_coef),
            "_seed": np.array(-1 if self.seed is None else self.seed),
            "_net_arch": np.array(net_arch, dtype=np.int64),
            "_log_std_init": np.array(float(self.policy_kwargs.get("log_std_init", 0.0))),
        }
        np.savez(path, **weights, **meta)

    @classmethod
    def load(cls, path: str, env, **kwargs):
        path = path if path.endswith(".npz") else path + ".npz"
        data = np.load(path, allow_pickle=False)
        files = set(data.files)

        def _meta(name: str, default):
            return data[name] if name in files else default

        seed_raw = int(_meta("_seed", -1))
        net_arch = tuple(int(x) for x in np.asarray(_meta("_net_arch", [64, 64])))
        policy_kwargs = dict(kwargs.pop("policy_kwargs", {}) or {})
        policy_kwargs.setdefault("net_arch", net_arch)
        if "_log_std_init" in files:
            policy_kwargs.setdefault("log_std_init", float(data["_log_std_init"]))
        model = cls(
            "MlpPolicy",
            env,
            learning_rate=float(_meta("_learning_rate", 3e-4)),
            n_steps=int(_meta("_n_steps", 2048)),
            batch_size=int(_meta("_batch_size", 64)),
            n_epochs=int(_meta("_n_epochs", 10)),
            gamma=float(_meta("_gamma", 0.99)),
            gae_lambda=float(_meta("_gae_lambda", 0.95)),
            clip_range=float(_meta("_clip_range", 0.2)),
            ent_coef=float(_meta("_ent_coef", 0.0)),
            vf_coef=float(_meta("_vf_coef", 0.5)),
            seed=None if seed_raw < 0 else seed_raw,
            policy_kwargs=policy_kwargs,
            _init_setup_model=True,
            **kwargs,
        )
        assert model.policy is not None
        flat = tree_flatten_dict(model.policy.parameters())
        loaded = {k: mx.array(data[k]) for k in flat if k in files}
        if len(loaded) != len(flat):
            missing = sorted(set(flat) - set(loaded))
            raise KeyError(f"checkpoint missing parameters: {missing}")
        params = model.policy.parameters()

        def _set(tree, prefix=""):
            if isinstance(tree, mx.array):
                return loaded.get(prefix, tree)
            if isinstance(tree, dict):
                return {kk: _set(vv, f"{prefix}.{kk}" if prefix else str(kk)) for kk, vv in tree.items()}
            if isinstance(tree, list):
                return [_set(vv, f"{prefix}.{i}" if prefix else str(i)) for i, vv in enumerate(tree)]
            if isinstance(tree, tuple):
                return tuple(
                    _set(vv, f"{prefix}.{i}" if prefix else str(i)) for i, vv in enumerate(tree)
                )
            return tree

        model.policy.update(_set(params))
        mx.eval(model.policy.parameters())
        return model
