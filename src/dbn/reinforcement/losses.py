"""Pure RL loss / target primitives (testable, shared by algos)."""

from __future__ import annotations

import mlx.core as mx


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
