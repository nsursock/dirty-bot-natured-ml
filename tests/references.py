"""NumPy reference implementations for RL math correctness tests."""

from __future__ import annotations

import math

import numpy as np


def reference_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    dones: np.ndarray,
    last_values: np.ndarray,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Classic GAE.

    Arrays shaped (T, N); last_values shaped (N,).
    Returns advantages, returns with shape (T, N).
    """
    T, N = rewards.shape
    advantages = np.zeros((T, N), dtype=np.float64)
    for n in range(N):
        gae = 0.0
        next_value = float(last_values[n])
        for t in range(T - 1, -1, -1):
            nonterminal = 1.0 - float(dones[t, n])
            delta = float(rewards[t, n]) + gamma * next_value * nonterminal - float(values[t, n])
            gae = delta + gamma * gae_lambda * nonterminal * gae
            advantages[t, n] = gae
            next_value = float(values[t, n])
    returns = advantages + values.astype(np.float64)
    return advantages.astype(np.float32), returns.astype(np.float32)


def reference_categorical_log_prob(logits: np.ndarray, actions: np.ndarray) -> np.ndarray:
    log_z = np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1, keepdims=True))
    log_probs = logits - logits.max(axis=-1, keepdims=True) - log_z
    return log_probs[np.arange(len(actions)), actions.astype(np.int64)].astype(np.float32)


def reference_categorical_entropy(logits: np.ndarray) -> np.ndarray:
    log_z = np.log(np.sum(np.exp(logits - logits.max(axis=-1, keepdims=True)), axis=-1, keepdims=True))
    log_p = logits - logits.max(axis=-1, keepdims=True) - log_z
    p = np.exp(log_p)
    return (-np.sum(p * log_p, axis=-1)).astype(np.float32)


def reference_gaussian_log_prob(
    mean: np.ndarray, log_std: np.ndarray, actions: np.ndarray
) -> np.ndarray:
    var = np.exp(2.0 * log_std)
    log_scale = log_std + 0.5 * math.log(2.0 * math.pi)
    return (
        -0.5 * np.sum(((actions - mean) ** 2) / var, axis=-1) - np.sum(log_scale, axis=-1)
    ).astype(np.float32)


def reference_gaussian_entropy(log_std: np.ndarray) -> float:
    return float(
        0.5 * math.log(2.0 * math.pi * math.e) * log_std.shape[-1] + np.sum(log_std)
    )


def reference_ppo_policy_loss(
    new_log_prob: np.ndarray,
    old_log_prob: np.ndarray,
    advantages: np.ndarray,
    clip_range: float,
) -> float:
    ratio = np.exp(new_log_prob - old_log_prob)
    pg1 = ratio * advantages
    pg2 = np.clip(ratio, 1.0 - clip_range, 1.0 + clip_range) * advantages
    return float(-np.mean(np.minimum(pg1, pg2)))


def reference_soft_update(target: np.ndarray, source: np.ndarray, tau: float) -> np.ndarray:
    return ((1.0 - tau) * target + tau * source).astype(np.float32)


def all_finite_tree(tree) -> bool:
    import mlx.core as mx

    if isinstance(tree, mx.array):
        arr = np.array(tree)
        return bool(np.isfinite(arr).all())
    if isinstance(tree, dict):
        return all(all_finite_tree(v) for v in tree.values())
    if isinstance(tree, (list, tuple)):
        return all(all_finite_tree(v) for v in tree)
    return True
