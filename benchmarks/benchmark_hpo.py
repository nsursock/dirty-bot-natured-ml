"""Context-aware HPO for PPO / SAC / TD3 — NOT part of pytest CI.

One Optuna study per algo. ``n_envs`` is sampled as a categorical
hyperparameter (4 .. 8192 by doubling), so the search can compare different
widths and find the configuration that minimizes wall-clock time-to-solve.

``MedianPruner(interval_steps=1)`` is required. Reports happen at chunk
boundaries, and those step counts are not multiples of 5000. Any other
interval never lands on a report, so pruning silently never runs.

A smoke run (``--trials 1``) checks the loop only. Pruning needs
``n_startup_trials`` completed trials first (default 3), so one trial
cannot be pruned.

The default grid stops starting new trials after 2 hours of wall clock.
A trial already in flight always finishes its current training run normally;
``--max-hours 0`` disables the cap. Existing studies resume by default, while
``--no-resume`` deletes the selected algorithms' study databases first.

Run:
  python benchmarks/benchmark_hpo.py --algo ppo --n-envs 4 8 --trials 1 --seed 0
  python benchmarks/benchmark_hpo.py --algo all --trials 8
"""

from __future__ import annotations

import argparse
import csv
import json
import signal
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Script directory is on sys.path when launched as `python benchmarks/benchmark_hpo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_solving import (  # noqa: E402
    N_ENVS,
    SOLVE_THRESHOLD,
    SOLVE_WINDOW,
    _ep_stats,
    chunk_steps_for,
    is_solved,
    step_budget,
)
from dbn.reinforcement.algos.ppo import PPO  # noqa: E402
from dbn.reinforcement.algos.sac import SAC  # noqa: E402
from dbn.reinforcement.algos.td3 import TD3  # noqa: E402
from dbn.reinforcement.envs import CartPole, Pendulum  # noqa: E402

try:
    import optuna
    from optuna.exceptions import TrialPruned
    from optuna.pruners import MedianPruner
except ImportError:  # pragma: no cover - import guard for missing extra
    optuna = None
    TrialPruned = Exception  # type: ignore[misc, assignment]
    MedianPruner = None  # type: ignore[misc, assignment]

_shutdown_requested = False


def _request_shutdown(signum, frame):
    global _shutdown_requested
    _shutdown_requested = True
    tqdm.write("\nshutdown requested: finishing current trial and saving state...")


signal.signal(signal.SIGINT, _request_shutdown)
signal.signal(signal.SIGTERM, _request_shutdown)

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
DEFAULT_MAX_HOURS = 2.0
# The scheduler stops starting work this long before the hard cap, so the
# chunk already in flight still finishes inside the cap.
WALL_MARGIN_S = 120.0
N_STARTUP_TRIALS = 3
# Env-step warmup. ``step`` passed to trial.report is num_timesteps.
WARMUP_STEPS = {"PPO": 40_000, "SAC": 40_000, "TD3": 60_000}
BASELINE_EPISODES = 20
N_EVAL_EPISODES = 10
ENVS = {"CartPole": CartPole, "Pendulum": Pendulum}

# SB3-style hyperparameter search ranges.
LR_LOW = 1e-5
LR_HIGH = 1e-2

PPO_N_STEPS_OPTIONS = (64, 128, 256, 512, 1024, 2048)
PPO_BATCH_SIZE_OPTIONS = (32, 64, 128, 256)
PPO_N_EPOCHS_OPTIONS = (4, 10, 20)
PPO_GAMMA_RANGE = (0.95, 0.999)
PPO_GAE_LAMBDA_RANGE = (0.8, 0.99)
PPO_CLIP_RANGE_RANGE = (0.05, 0.3)
PPO_ENT_COEF_RANGE = (1e-8, 1e-1)
PPO_VF_COEF_RANGE = (0.1, 1.0)
PPO_MAX_GRAD_NORM_RANGE = (0.1, 1.0)

OFFPOLICY_BUFFER_SIZE_OPTIONS = (100_000, 500_000, 1_000_000)
OFFPOLICY_LEARNING_STARTS_OPTIONS = (100, 1000, 5000, 10_000)
OFFPOLICY_BATCH_SIZE_OPTIONS = (128, 256, 512)
OFFPOLICY_TAU_RANGE = (0.001, 0.02)
OFFPOLICY_GAMMA_RANGE = (0.95, 0.999)
OFFPOLICY_TRAIN_FREQ_OPTIONS = (1, 2, 4, 8)
OFFPOLICY_GRADIENT_STEPS_OPTIONS = (1, 2, 4, 8)

TD3_POLICY_DELAY_OPTIONS = (1, 2, 3, 4)
TD3_TARGET_POLICY_NOISE_RANGE = (0.1, 0.3)
TD3_TARGET_NOISE_CLIP_RANGE = (0.3, 0.7)
TD3_ACTION_NOISE_STD_RANGE = (0.05, 0.2)

# Knobs recorded in CSV output (algos share gamma/batch_size keys).
HYPERPARAM_KNOBS = (
    "n_steps",
    "batch_size",
    "n_epochs",
    "gamma",
    "gae_lambda",
    "clip_range",
    "ent_coef",
    "vf_coef",
    "max_grad_norm",
    "normalize_advantage",
    "buffer_size",
    "learning_starts",
    "tau",
    "train_freq",
    "gradient_steps",
    "policy_delay",
    "target_policy_noise",
    "target_noise_clip",
    "action_noise_std",
)

CHUNK_COLUMNS = (
    "algo",
    "n_envs",
    "trial",
    "seed",
    "chunk_idx",
    "steps_at_chunk_start",
    "env_steps",
    "wall_clock_s",
    "ep_rew_mean",
    "eval_ep_rew_mean",
    "chunk_steps",
    "lr",
) + HYPERPARAM_KNOBS

TRIAL_COLUMNS = (
    "algo",
    "n_envs",
    "trial",
    "seed",
    "solved",
    "pruned",
    "time_capped",
    "elapsed_s",
    "steps_at_stop",
    "ep_rew_at_stop",
    "objective",
    "baseline",
    "threshold",
    "lr",
) + HYPERPARAM_KNOBS


def _blank(x) -> str | int | float:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return x


def _append_csv(path: Path, row: dict, columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        if new_file:
            writer.writeheader()
        writer.writerow({c: row.get(c, "") for c in columns})
        f.flush()


def _count_trial_states(db_path: Path, study_name: str) -> dict[str, int]:
    if optuna is None or not db_path.exists():
        return {}
    try:
        storage = "sqlite:///" + urllib.parse.quote(db_path.as_posix(), safe="/")
        study = optuna.load_study(study_name=study_name, storage=storage)
        return {
            state.name: sum(1 for t in study.trials if t.state == state)
            for state in optuna.trial.TrialState
        }
    except Exception:
        return {}


def _progress(ep_rew: float, baseline: float, threshold: float) -> float:
    """Clamped [0, 2]. The study minimizes, so callers report the negation."""
    if not np.isfinite(ep_rew) or threshold == baseline:
        return 0.0
    raw = (ep_rew - baseline) / (threshold - baseline)
    return float(np.clip(raw, 0.0, 2.0))


def _random_baseline(env_name: str, seed: int, cache_path: Path) -> float:
    """Mean return of a random policy. Cached by (env, seed), not by algo or width."""
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())
    env_cache = cache.setdefault(env_name, {})
    key = str(seed)
    if key in env_cache:
        return float(env_cache[key])

    env = ENVS[env_name](n_envs=1, seed=seed)
    env.reset(seed=seed)
    returns: list[float] = []
    while len(returns) < BASELINE_EPISODES:
        _obs, _rew, _term, _trunc, info = env.step(env.action_space.sample())
        if isinstance(info, dict) and "episode" in info:
            returns.append(float(info["episode"]["r"]))
    value = float(np.mean(returns))
    env_cache[key] = value
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2, sort_keys=True))
    return value


def _eval_mean_reward(model, env_name: str, seed: int, n_episodes: int) -> float:
    """Deterministic evaluation on a single fresh env (SB3 EvalCallback style)."""
    env = ENVS[env_name](n_envs=1, seed=seed)
    obs, _info = env.reset(seed=seed)
    returns: list[float] = []
    while len(returns) < n_episodes:
        action, _state = model.predict(obs, deterministic=True)
        obs, _reward, _terminated, _truncated, info = env.step(action)
        if info and "episode" in info:
            returns.append(float(info["episode"]["r"]))
            obs, _info = env.reset()
    return float(np.mean(returns))


def _make_model(algo: str, n_envs: int, seed: int, params: dict):
    # ``params`` includes sampled hyperparameters, including ``learning_rate``.
    # ``n_envs`` is handled by constructing the env; it is not passed to the model.
    model_kwargs = {k: v for k, v in params.items() if k != "n_envs"}
    if algo == "PPO":
        env = CartPole(n_envs=n_envs, seed=seed)
        return PPO(
            "MlpPolicy",
            env,
            seed=seed,
            policy_kwargs={"net_arch": (64, 64)},
            **model_kwargs,
        )
    env = Pendulum(n_envs=n_envs, seed=seed)
    return (SAC if algo == "SAC" else TD3)(
        "MlpPolicy",
        env,
        seed=seed,
        policy_kwargs={"net_arch": (256, 256)},
        **model_kwargs,
    )


def _sample(trial, algo: str, n_envs_choices: tuple[int, ...]) -> dict:
    params: dict[str, Any] = {
        "learning_rate": float(trial.suggest_float("learning_rate", LR_LOW, LR_HIGH, log=True)),
        "n_envs": int(trial.suggest_categorical("n_envs", list(n_envs_choices))),
    }
    if algo == "PPO":
        params.update(
            {
                "n_steps": int(trial.suggest_categorical("n_steps", list(PPO_N_STEPS_OPTIONS))),
                "batch_size": int(trial.suggest_categorical("batch_size", list(PPO_BATCH_SIZE_OPTIONS))),
                "n_epochs": int(trial.suggest_categorical("n_epochs", list(PPO_N_EPOCHS_OPTIONS))),
                "gamma": float(trial.suggest_float("gamma", *PPO_GAMMA_RANGE)),
                "gae_lambda": float(trial.suggest_float("gae_lambda", *PPO_GAE_LAMBDA_RANGE)),
                "clip_range": float(trial.suggest_float("clip_range", *PPO_CLIP_RANGE_RANGE)),
                "ent_coef": float(trial.suggest_float("ent_coef", *PPO_ENT_COEF_RANGE, log=True)),
                "vf_coef": float(trial.suggest_float("vf_coef", *PPO_VF_COEF_RANGE)),
                "max_grad_norm": float(trial.suggest_float("max_grad_norm", *PPO_MAX_GRAD_NORM_RANGE)),
                "normalize_advantage": bool(
                    trial.suggest_categorical("normalize_advantage", [True, False])
                ),
            }
        )
    else:
        params.update(
            {
                "buffer_size": int(trial.suggest_categorical("buffer_size", list(OFFPOLICY_BUFFER_SIZE_OPTIONS))),
                "learning_starts": int(trial.suggest_categorical("learning_starts", list(OFFPOLICY_LEARNING_STARTS_OPTIONS))),
                "batch_size": int(trial.suggest_categorical("batch_size", list(OFFPOLICY_BATCH_SIZE_OPTIONS))),
                "tau": float(trial.suggest_float("tau", *OFFPOLICY_TAU_RANGE, log=True)),
                "gamma": float(trial.suggest_float("gamma", *OFFPOLICY_GAMMA_RANGE)),
                "train_freq": int(trial.suggest_categorical("train_freq", list(OFFPOLICY_TRAIN_FREQ_OPTIONS))),
                "gradient_steps": int(trial.suggest_categorical("gradient_steps", list(OFFPOLICY_GRADIENT_STEPS_OPTIONS))),
            }
        )
        if algo == "TD3":
            params.update(
                {
                    "policy_delay": int(trial.suggest_categorical("policy_delay", list(TD3_POLICY_DELAY_OPTIONS))),
                    "target_policy_noise": float(trial.suggest_float("target_policy_noise", *TD3_TARGET_POLICY_NOISE_RANGE)),
                    "target_noise_clip": float(trial.suggest_float("target_noise_clip", *TD3_TARGET_NOISE_CLIP_RANGE)),
                    "action_noise_std": float(trial.suggest_float("action_noise_std", *TD3_ACTION_NOISE_STD_RANGE)),
                }
            )
    return params


def _knobs(params: dict) -> dict:
    knobs = {"lr": params["learning_rate"]}
    for k in HYPERPARAM_KNOBS:
        knobs[k] = _blank(params.get(k))
    return knobs


class _WallClock:
    """Splits a hard wall-clock cap across the trials still to run."""

    def __init__(self, max_hours: float, n_trials: int) -> None:
        if max_hours <= 0:
            self.hard: float | None = None
            self.schedule: float | None = None
        else:
            self.hard = time.perf_counter() + max_hours * 3600.0
            self.schedule = self.hard - WALL_MARGIN_S
        self.left = n_trials

    def expired(self) -> bool:
        return self.schedule is not None and time.perf_counter() >= self.schedule

    def pop(self) -> None:
        self.left = max(0, self.left - 1)


def _run_trial(
    trial,
    *,
    algo: str,
    env_name: str,
    n_envs_choices: tuple[int, ...],
    seed: int,
    baseline: float,
    threshold: float,
    chunk_csv: Path,
    trial_csv: Path,
) -> float:
    params = _sample(trial, algo, n_envs_choices)
    n_envs = int(params["n_envs"])
    model = _make_model(algo, n_envs, seed, params)
    budget = step_budget(n_envs)
    chunk = chunk_steps_for(n_envs, env_name)
    knobs = _knobs(params)
    t0 = time.perf_counter()
    first = True
    chunk_idx = 0
    solved = False
    pruned = False
    time_capped = False
    ep_rew = float("nan")
    elapsed = 0.0
    bar = tqdm(
        total=budget,
        desc=f"{algo} n={n_envs} trial {trial.number}",
        unit="step",
        position=1,
        leave=False,
        dynamic_ncols=True,
    )

    try:
        while model.num_timesteps < budget:
            steps_at_start = int(model.num_timesteps)
            target = min(budget, model.num_timesteps + chunk)
            model.learn(
                target,
                progress_bar=False,
                reset_num_timesteps=first,
            )
            first = False
            ep_rew, _n_eps = _ep_stats(model)
            # SB3-style early stopping: deterministic eval on a fresh env.
            eval_rew = _eval_mean_reward(model, env_name, seed, N_EVAL_EPISODES)
            elapsed = time.perf_counter() - t0
            bar.update(max(0, int(model.num_timesteps) - steps_at_start))
            bar.set_postfix(
                trial=trial.number,
                eval_rew=f"{eval_rew:.1f}" if np.isfinite(eval_rew) else "-",
            )
            progress = _progress(eval_rew, baseline, threshold)
            _append_csv(
                chunk_csv,
                {
                    "algo": algo,
                    "n_envs": n_envs,
                    "trial": trial.number,
                    "seed": seed,
                    "chunk_idx": chunk_idx,
                    "steps_at_chunk_start": steps_at_start,
                    "env_steps": int(model.num_timesteps),
                    "wall_clock_s": round(elapsed, 4),
                    "ep_rew_mean": _blank(ep_rew if np.isfinite(ep_rew) else None),
                    "eval_ep_rew_mean": _blank(eval_rew if np.isfinite(eval_rew) else None),
                    "chunk_steps": chunk,
                    **knobs,
                },
                CHUNK_COLUMNS,
            )
            # Lower is better: the study minimizes solve time, so progress is negated.
            trial.report(-progress, int(model.num_timesteps))
            if eval_rew >= threshold:
                solved = True
                ep_rew = eval_rew
                break
            if trial.should_prune():
                pruned = True
                raise TrialPruned()
            if _shutdown_requested:
                time_capped = True
                break
            chunk_idx += 1
    finally:
        bar.close()
        elapsed = time.perf_counter() - t0
        steps_at_stop = int(model.num_timesteps)
        objective: float
        if solved:
            objective = round(elapsed, 4)
        else:
            # Penalize unsolved/pruned/capped trials by extrapolating elapsed time
            # to the full budget. Capped trials get an extra-large penalty.
            per_step = elapsed / max(steps_at_stop, 1)
            objective = round(elapsed + max(0, budget - steps_at_stop) * per_step, 4)
            if time_capped:
                objective = 1e9
        _append_csv(
            trial_csv,
            {
                "algo": algo,
                "n_envs": n_envs,
                "trial": trial.number,
                "seed": seed,
                "solved": int(solved),
                "pruned": int(pruned),
                "time_capped": int(time_capped),
                "elapsed_s": round(elapsed, 4),
                "steps_at_stop": steps_at_stop,
                "ep_rew_at_stop": _blank(ep_rew if np.isfinite(ep_rew) else None),
                "objective": objective,
                "baseline": baseline,
                "threshold": threshold,
                **knobs,
            },
            TRIAL_COLUMNS,
        )
        trial.set_user_attr("solved", int(solved))
        trial.set_user_attr("time_capped", int(time_capped))
        trial.set_user_attr("ep_rew", round(ep_rew, 2) if np.isfinite(ep_rew) else None)

    if time_capped:
        raise TrialPruned()
    return float(objective)


def run_cell(
    algo: str,
    env_name: str,
    n_envs_choices: tuple[int, ...],
    *,
    n_trials: int,
    seed: int,
    chunk_csv: Path,
    trial_csv: Path,
    baseline_path: Path,
    clock: _WallClock,
    grid: tqdm,
) -> bool:
    if optuna is None or MedianPruner is None:
        raise ImportError(
            "Optuna is required for this benchmark. Install with: pip install -e '.[bench]'"
        )
    threshold = SOLVE_THRESHOLD[env_name]
    baseline = _random_baseline(env_name, seed, baseline_path)
    study_name = f"hpo_{algo}_sb3"
    db_path = OUT_DIR / f"{study_name}.db"
    states = _count_trial_states(db_path, study_name)
    n_completed = states.get("COMPLETE", 0)
    n_remaining = max(0, n_trials - n_completed)
    if n_remaining == 0:
        return False
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///" + urllib.parse.quote(db_path.as_posix(), safe="/")
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=WARMUP_STEPS[algo],
        # interval_steps=1: chunk-end step counts are not on a fixed grid.
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=study_name,
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )
    study.set_user_attr("threshold", threshold)
    study.set_user_attr("algo", algo)

    if n_trials <= N_STARTUP_TRIALS:
        tqdm.write(
            f"note: --trials {n_trials} does not exercise pruning "
            f"(n_startup_trials={N_STARTUP_TRIALS}). This checks the loop only."
        )
    tqdm.write(
        f"{algo} {env_name} trials={n_trials} seed={seed} "
        f"baseline={baseline:.2f} threshold={threshold} n_envs in {list(n_envs_choices)}"
    )

    def objective(trial) -> float:
        return _run_trial(
            trial,
            algo=algo,
            env_name=env_name,
            n_envs_choices=n_envs_choices,
            seed=seed,
            baseline=baseline,
            threshold=threshold,
            chunk_csv=chunk_csv,
            trial_csv=trial_csv,
        )

    for _ in range(n_remaining):
        if _shutdown_requested:
            return True
        if clock.expired():
            tqdm.write(f"wall clock: stopping before {algo}")
            return True
        grid.set_postfix_str(f"{algo}")
        n_before = len(study.trials)
        try:
            study.optimize(objective, n_trials=1)
        finally:
            clock.pop()
            if len(study.trials) > n_before:
                last = study.trials[-1]
                if last.user_attrs.get("time_capped"):
                    tag = "capped"
                elif last.state.name == "PRUNED":
                    tag = "pruned"
                elif last.user_attrs.get("solved"):
                    tag = "solved"
                else:
                    tag = "budget"
                rew = last.user_attrs.get("ep_rew", "-")
                grid.set_postfix_str(f"{algo} {tag} rew={rew}")
                grid.update(1)
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=["ppo", "sac", "td3", "all"], default="all")
    p.add_argument(
        "--n-envs",
        type=int,
        nargs="+",
        default=list(N_ENVS),
        help="candidate n_envs values to sample from (default: 4..8192 by doubling)",
    )
    p.add_argument("--trials", type=int, default=8)
    p.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="resume from existing Optuna sqlite DB (default: True); --no-resume deletes selected DBs",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-hours",
        type=float,
        default=DEFAULT_MAX_HOURS,
        help="stop the grid at this wall-clock cap (0 disables it)",
    )
    p.add_argument("--chunk-csv", type=Path, default=OUT_DIR / "hpo_sb3_chunks.csv")
    p.add_argument("--trial-csv", type=Path, default=OUT_DIR / "hpo_sb3_trials.csv")
    p.add_argument("--baseline-cache", type=Path, default=OUT_DIR / "hpo_baselines.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    algos = ["ppo", "sac", "td3"] if args.algo == "all" else [args.algo]
    names = {"ppo": ("PPO", "CartPole"), "sac": ("SAC", "Pendulum"), "td3": ("TD3", "Pendulum")}
    n_envs_choices = tuple(args.n_envs)

    if not args.resume:
        for key in algos:
            algo, _ = names[key]
            db_path = OUT_DIR / f"hpo_{algo}_sb3.db"
            if db_path.exists():
                db_path.unlink()

    plan = []
    total_remaining = 0
    for key in algos:
        algo, env_name = names[key]
        study_name = f"hpo_{algo}_sb3"
        db_path = OUT_DIR / f"{study_name}.db"
        states = _count_trial_states(db_path, study_name)
        n_completed = states.get("COMPLETE", 0)
        n_pruned = states.get("PRUNED", 0)
        n_fail = states.get("FAIL", 0)
        n_remaining = max(0, args.trials - n_completed)
        if n_completed or n_pruned or n_fail:
            tqdm.write(
                f"Study {study_name}: {n_completed} COMPLETE, {n_pruned} PRUNED, "
                f"{n_fail} FAIL; {n_remaining} remaining"
            )
        plan.append((key, algo, env_name, n_remaining))
        total_remaining += n_remaining

    if total_remaining == 0:
        print("All requested trials already completed.")
        return

    clock = _WallClock(args.max_hours, total_remaining)
    if args.max_hours > 0:
        tqdm.write(f"wall-clock cap: {args.max_hours:g}h")
    grid = tqdm(total=total_remaining, desc="HPO", unit="trial", position=0, dynamic_ncols=True)
    try:
        for _key, algo, env_name, n_remaining in plan:
            if n_remaining == 0:
                continue
            if _shutdown_requested:
                break
            if clock.expired():
                tqdm.write("wall clock: not starting further cells")
                break
            stopped = run_cell(
                algo,
                env_name,
                n_envs_choices,
                n_trials=args.trials,
                seed=args.seed,
                chunk_csv=args.chunk_csv,
                trial_csv=args.trial_csv,
                baseline_path=args.baseline_cache,
                clock=clock,
                grid=grid,
            )
            if _shutdown_requested or stopped:
                break
    finally:
        grid.close()


if __name__ == "__main__":
    main()
