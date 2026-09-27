"""Context-aware HPO for PPO / SAC / TD3 — NOT part of pytest CI.

One Optuna study per (algo, n_envs). ``n_envs`` is fixed by the outer loop
(8, 64, 1024), not sampled, so the pruner never compares different widths.

``MedianPruner(interval_steps=1)`` is required. Reports happen at chunk
boundaries, and those step counts are not multiples of 5000. Any other
interval never lands on a report, so pruning silently never runs.

A smoke run (``--trials 1``) checks the loop only. Pruning needs
``n_startup_trials`` completed trials first (default 3), so one trial
cannot be pruned.

The default grid stops at 2 hours of wall clock. Each trial gets an equal
share of the time still left, and fast trials donate the rest to later
ones. A trial stopped by that share is ``time_capped``: elapsed time is
the measured partial run, and the objective is left blank so a short cap
is not ranked as a fast solve. ``--max-hours 0`` disables the cap.

Run:
  python benchmarks/benchmark_hpo.py --algo ppo --n-envs 8 --trials 1 --seed 0
  python benchmarks/benchmark_hpo.py --algo all --trials 8
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
import urllib.parse
from pathlib import Path

import numpy as np
from tqdm import tqdm

# Script directory is on sys.path when launched as `python benchmarks/benchmark_hpo.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from benchmark_solving import (  # noqa: E402
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

OUT_DIR = Path(__file__).resolve().parent.parent / "outputs"
N_ENVS = (8, 64, 1024)
DEFAULT_MAX_HOURS = 2.0
# The scheduler stops starting work this long before the hard cap, so the
# chunk already in flight still finishes inside the cap.
WALL_MARGIN_S = 120.0
N_STARTUP_TRIALS = 3
# Env-step warmup. ``step`` passed to trial.report is num_timesteps.
WARMUP_STEPS = {"PPO": 40_000, "SAC": 40_000, "TD3": 60_000}
BASELINE_EPISODES = 20
GS_CAP = 32
PPO_N_STEPS = (8, 32, 128)
LR_LOW = 1e-4
LR_HIGH = 3e-3
LEARNING_STARTS = (1000, 5000, 10_000)
ENVS = {"CartPole": CartPole, "Pendulum": Pendulum}

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
    "chunk_steps",
    "lr",
    "learning_starts",
    "gradient_steps_effective",
    "update_ratio",
    "n_steps",
    "rollout",
)

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
    "lr",
    "learning_starts",
    "gradient_steps_effective",
    "update_ratio",
    "n_steps",
    "rollout",
    "baseline",
    "threshold",
)


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


def _make_model(algo: str, n_envs: int, seed: int, params: dict):
    if algo == "PPO":
        n_steps = int(params["n_steps"])
        env = CartPole(n_envs=n_envs, seed=seed)
        return PPO(
            "MlpPolicy",
            env,
            learning_rate=float(params["lr"]),
            n_steps=n_steps,
            batch_size=min(64, n_steps * n_envs),
            n_epochs=10,
            policy_kwargs={"net_arch": (64, 64)},
            seed=seed,
        )
    env = Pendulum(n_envs=n_envs, seed=seed)
    common = dict(
        learning_rate=float(params["lr"]),
        learning_starts=int(params["learning_starts"]),
        buffer_size=max(100_000, min(1_000_000, n_envs * 50)),
        batch_size=256,
        gradient_steps=int(params["gradient_steps"]),
        policy_kwargs={"net_arch": (256, 256)},
        seed=seed,
    )
    if algo == "SAC":
        return SAC("MlpPolicy", env, **common)
    return TD3("MlpPolicy", env, **common)


def _sample(trial, algo: str, n_envs: int) -> dict:
    lr = float(trial.suggest_float("learning_rate", LR_LOW, LR_HIGH, log=True))
    if algo == "PPO":
        n_steps = int(trial.suggest_categorical("n_steps", list(PPO_N_STEPS)))
        return {
            "lr": lr,
            "learning_starts": None,
            "gradient_steps": None,
            "update_ratio": None,
            "n_steps": n_steps,
            "rollout": n_steps * n_envs,
        }
    high = min(GS_CAP, n_envs)
    gradient_steps = int(trial.suggest_int("gradient_steps", 1, high, log=True))
    starts = int(trial.suggest_categorical("learning_starts", list(LEARNING_STARTS)))
    return {
        "lr": lr,
        "learning_starts": max(starts, n_envs),
        "gradient_steps": gradient_steps,
        "update_ratio": gradient_steps / n_envs,
        "n_steps": None,
        "rollout": None,
    }


def _knobs(params: dict) -> dict:
    return {
        "lr": params["lr"],
        "learning_starts": _blank(params["learning_starts"]),
        "gradient_steps_effective": _blank(params["gradient_steps"]),
        "update_ratio": _blank(params["update_ratio"]),
        "n_steps": _blank(params["n_steps"]),
        "rollout": _blank(params["rollout"]),
    }


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

    def trial_deadline(self) -> float:
        if self.schedule is None:
            return float("inf")
        now = time.perf_counter()
        share = max(0.0, self.schedule - now) / max(self.left, 1)
        return now + share

    def pop(self) -> None:
        self.left = max(0, self.left - 1)


def _run_trial(
    trial,
    *,
    algo: str,
    env_name: str,
    n_envs: int,
    seed: int,
    baseline: float,
    threshold: float,
    chunk_csv: Path,
    trial_csv: Path,
    trial_deadline: float,
) -> float:
    params = _sample(trial, algo, n_envs)
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
            if time.perf_counter() >= trial_deadline:
                time_capped = True
                break
            steps_at_start = int(model.num_timesteps)
            target = min(budget, model.num_timesteps + chunk)
            model.learn(
                target,
                progress_bar=False,
                reset_num_timesteps=first,
            )
            first = False
            ep_rew, _n_eps = _ep_stats(model)
            elapsed = time.perf_counter() - t0
            bar.update(max(0, int(model.num_timesteps) - steps_at_start))
            bar.set_postfix(rew=f"{ep_rew:.1f}" if np.isfinite(ep_rew) else "-")
            progress = _progress(ep_rew, baseline, threshold)
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
                    "chunk_steps": chunk,
                    **knobs,
                },
                CHUNK_COLUMNS,
            )
            # Lower is better: the study minimizes solve time, so progress is negated.
            trial.report(-progress, int(model.num_timesteps))
            if is_solved(model, threshold, window=SOLVE_WINDOW):
                solved = True
                break
            if trial.should_prune():
                pruned = True
                raise TrialPruned()
            chunk_idx += 1
    finally:
        bar.close()
        elapsed = time.perf_counter() - t0
        objective: float | str
        if pruned or time_capped:
            objective = ""
        else:
            objective = round(elapsed, 4)
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
                "steps_at_stop": int(model.num_timesteps),
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
    return elapsed


def run_cell(
    algo: str,
    env_name: str,
    n_envs: int,
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
    db_path = OUT_DIR / f"hpo_{algo}_{n_envs}.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    storage = "sqlite:///" + urllib.parse.quote(db_path.as_posix(), safe="/")
    pruner = MedianPruner(
        n_startup_trials=N_STARTUP_TRIALS,
        n_warmup_steps=WARMUP_STEPS[algo],
        # interval_steps=1: chunk-end step counts are not on a fixed grid.
        interval_steps=1,
    )
    study = optuna.create_study(
        study_name=f"hpo_{algo}_{n_envs}",
        storage=storage,
        load_if_exists=True,
        direction="minimize",
        pruner=pruner,
    )
    study.set_user_attr("n_envs", n_envs)
    study.set_user_attr("threshold", threshold)
    study.set_user_attr("algo", algo)

    if n_trials <= N_STARTUP_TRIALS:
        tqdm.write(
            f"note: --trials {n_trials} does not exercise pruning "
            f"(n_startup_trials={N_STARTUP_TRIALS}). This checks the loop only."
        )
    tqdm.write(
        f"{algo} {env_name} n_envs={n_envs} trials={n_trials} seed={seed} "
        f"baseline={baseline:.2f} threshold={threshold} budget={step_budget(n_envs)}"
    )

    deadline_s = {"t": float("inf")}

    def objective(trial) -> float:
        return _run_trial(
            trial,
            algo=algo,
            env_name=env_name,
            n_envs=n_envs,
            seed=seed,
            baseline=baseline,
            threshold=threshold,
            chunk_csv=chunk_csv,
            trial_csv=trial_csv,
            trial_deadline=deadline_s["t"],
        )

    for _ in range(n_trials):
        if clock.expired():
            tqdm.write(f"wall clock: stopping before {algo} n_envs={n_envs}")
            return True
        deadline_s["t"] = clock.trial_deadline()
        grid.set_postfix_str(f"{algo} n={n_envs}")
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
                grid.set_postfix_str(f"{algo} n={n_envs} {tag} rew={rew}")
                grid.update(1)
    return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=["ppo", "sac", "td3", "all"], default="all")
    p.add_argument("--n-envs", type=int, nargs="+", default=list(N_ENVS))
    p.add_argument("--trials", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-hours",
        type=float,
        default=DEFAULT_MAX_HOURS,
        help="stop the grid at this wall-clock cap (0 disables it)",
    )
    p.add_argument("--chunk-csv", type=Path, default=OUT_DIR / "hpo_chunks.csv")
    p.add_argument("--trial-csv", type=Path, default=OUT_DIR / "hpo_trials.csv")
    p.add_argument("--baseline-cache", type=Path, default=OUT_DIR / "hpo_baselines.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if optuna is not None:
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    algos = ["ppo", "sac", "td3"] if args.algo == "all" else [args.algo]
    names = {"ppo": ("PPO", "CartPole"), "sac": ("SAC", "Pendulum"), "td3": ("TD3", "Pendulum")}
    n_total = len(algos) * len(args.n_envs) * args.trials
    clock = _WallClock(args.max_hours, n_total)
    if args.max_hours > 0:
        tqdm.write(f"wall-clock cap: {args.max_hours:g}h")
    grid = tqdm(total=n_total, desc="HPO", unit="trial", position=0, dynamic_ncols=True)
    try:
        for key in algos:
            algo, env_name = names[key]
            for n in args.n_envs:
                if clock.expired():
                    tqdm.write("wall clock: not starting further cells")
                    return
                stopped = run_cell(
                    algo,
                    env_name,
                    n,
                    n_trials=args.trials,
                    seed=args.seed,
                    chunk_csv=args.chunk_csv,
                    trial_csv=args.trial_csv,
                    baseline_path=args.baseline_cache,
                    clock=clock,
                    grid=grid,
                )
                if stopped:
                    return
    finally:
        grid.close()


if __name__ == "__main__":
    main()
