"""n_envs solve-time sweep for PPO / SAC / TD3 — NOT part of pytest CI.

Gym / Gymnasium solve criteria (mean episode return over the last 100 episodes):
  CartPole-v1:  reward_threshold = 475.0   (official gym registry)
  Pendulum-v1:  reward_threshold = -200.0  (no registry threshold; community default)

Sweep: n_envs = 4, 8, 16, …, 8192. Env-step budget scales with n_envs
(base budget at n_envs=4, linear in n_envs thereafter).

Run:
  python benchmarks/benchmark_solving.py
  python benchmarks/benchmark_solving.py --algo ppo --seeds 1
  python benchmarks/benchmark_solving.py --base-budget 250000 --csv outputs/solving_results.csv
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import yaml

import numpy as np
from tabulate import tabulate
from tqdm import tqdm

from dbn.reinforcement.algos.ppo import PPO
from dbn.reinforcement.algos.sac import SAC
from dbn.reinforcement.algos.td3 import TD3
from dbn.reinforcement.common import evaluate_policy
from dbn.reinforcement.envs import CartPole, Pendulum

# 4 → 8192 by doubling
N_ENVS = tuple(4 * (2**i) for i in range(12))  # 4 .. 8192
REF_N_ENVS = 4
DEFAULT_SEEDS = 1
# Env-step budget at n_envs=REF_N_ENVS; scaled linearly for larger n.
DEFAULT_BASE_BUDGET = 250_000
# Floor on env-steps between solve checks (also scaled up with n_envs).
CHUNK_STEPS = 4096
SOLVE_WINDOW = 100  # gym: average over 100 consecutive trials

# Max episode length — used to size learn() chunks so episodes can finish
# (each learn() call resets the env, wiping in-progress episodes).
MAX_EP_STEPS = {
    "CartPole": 500,
    "Pendulum": 200,
}

SOLVE_THRESHOLD = {
    "CartPole": 475.0,  # CartPole-v1
    "Pendulum": -200.0,  # community; Pendulum-v1 has no gym reward_threshold
}

RAW_COLUMNS = (
    "algo",
    "env",
    "n_envs",
    "seed",
    "threshold",
    "budget",
    "solved",
    "steps_to_solve",
    "elapsed_s",
    "ep_rew_mean",
    "eval_mean",
    "eval_std",
    "n_episodes",
)

AGG_COLUMNS = (
    "algo",
    "env",
    "n_envs",
    "seeds",
    "threshold",
    "budget",
    "solve_rate",
    "steps_mean",
    "steps_std",
    "steps_median",
    "steps_q25",
    "steps_q75",
    "steps_min",
    "steps_max",
    "elapsed_s_mean",
    "ep_rew_mean",
    "eval_mean",
    "eval_std",
    "eval_median",
    "eval_q25",
    "eval_q75",
)


def step_budget(n_envs: int, base: int = DEFAULT_BASE_BUDGET, ref: int = REF_N_ENVS) -> int:
    """Env-step budget scaled linearly with n_envs (base at ``ref``)."""
    return max(base, int(base * n_envs / ref))


def chunk_steps_for(n_envs: int, env_name: str, base: int = CHUNK_STEPS) -> int:
    """Chunk large enough that ≥1 max-length episode can finish per env.

    ``learn()`` resets the env at the start of every call, so a tiny chunk at
    high n_envs never completes episodes and the solve buffer stays empty.
    """
    return max(base, n_envs * MAX_EP_STEPS[env_name])


# If still this far below threshold at halfway budget, abort early (hopeless).
HOPELESS_GAP = {
    "CartPole": 200.0,
    "Pendulum": 500.0,
}


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(np.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def _finite_float(x) -> float | None:
    if x is None or x == "":
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if np.isfinite(v) else None


def _ep_stats(model) -> tuple[float, int]:
    buf = model._ep_info_buffer
    if not buf:
        return float("nan"), 0
    return float(np.mean([e["r"] for e in buf])), len(buf)


def is_solved(model, threshold: float, window: int = SOLVE_WINDOW) -> bool:
    """Gym criterion: mean return over the last ``window`` episodes >= threshold."""
    ep_rew, n = _ep_stats(model)
    return n >= window and ep_rew >= threshold


def _make_ppo(n_envs: int, seed: int) -> PPO:
    # ~2048-step rollouts; floor n_steps so GAE horizon does not collapse to 1.
    n_steps = max(8, 2048 // n_envs) if n_envs < 256 else max(4, 8192 // n_envs)
    env = CartPole(n_envs=n_envs, seed=seed)
    return PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=64,
        n_epochs=10,
        policy_kwargs={"net_arch": (64, 64)},
        seed=seed,
    )


def _make_sac(n_envs: int, seed: int) -> SAC:
    env = Pendulum(n_envs=n_envs, seed=seed)
    return SAC(
        "MlpPolicy",
        env,
        learning_starts=min(10_000, max(n_envs * 2, 1000)),
        buffer_size=max(100_000, min(1_000_000, n_envs * 50)),
        batch_size=256,
        gradient_steps=1,
        policy_kwargs={"net_arch": (256, 256)},
        seed=seed,
    )


def _make_td3(n_envs: int, seed: int) -> TD3:
    env = Pendulum(n_envs=n_envs, seed=seed)
    # One update per transition (UTD=1). Cap so n_envs=8192 does not run
    # thousands of gradient steps on every vector step.
    return TD3(
        "MlpPolicy",
        env,
        learning_starts=10_000,
        buffer_size=max(100_000, min(1_000_000, n_envs * 50)),
        batch_size=256,
        gradient_steps=min(n_envs, 32),
        policy_kwargs={"net_arch": (256, 256)},
        seed=seed,
    )


def _make_from_yaml(path: Path):
    """Factory that loads a YAML config and builds the model per (n_envs, seed)."""
    raw = yaml.safe_load(path.read_text())
    env_name = raw["env"]
    algo = raw["algo"]
    base_kwargs = {k: v for k, v in raw.items() if k not in {"algo", "env", "n_envs", "seed"}}
    if "policy_kwargs" in base_kwargs and base_kwargs["policy_kwargs"] is not None:
        base_kwargs["policy_kwargs"] = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in base_kwargs["policy_kwargs"].items()
        }
    algo_cls = {"PPO": PPO, "SAC": SAC, "TD3": TD3}[algo]
    env_cls = {"CartPole": CartPole, "Pendulum": Pendulum}[env_name]

    def _make(n_envs: int, seed: int):
        env = env_cls(n_envs=n_envs, seed=seed)
        return algo_cls("MlpPolicy", env, seed=seed, verbose=0, **base_kwargs)

    return algo, env_name, _make


SWEEPS = {
    "ppo": ("PPO", "CartPole", _make_ppo),
    "sac": ("SAC", "Pendulum", _make_sac),
    "td3": ("TD3", "Pendulum", _make_td3),
}

ENV_CLS = {"CartPole": CartPole, "Pendulum": Pendulum}


def train_until_solved(
    model,
    *,
    threshold: float,
    max_steps: int,
    chunk_steps: int = CHUNK_STEPS,
    window: int = SOLVE_WINDOW,
    env_name: str = "CartPole",
    eval_env=None,
    eval_enabled: bool = True,
    n_eval_episodes: int = 10,
) -> dict:
    """Chunked learn(); stop at first gym-solve, hopeless mid-budget, or ``max_steps``."""
    t0 = time.perf_counter()
    first = True
    gap = HOPELESS_GAP[env_name]
    eval_mean = eval_std = float("nan")
    while model.num_timesteps < max_steps:
        target = min(max_steps, model.num_timesteps + chunk_steps)
        if first:
            model.learn(target, progress_bar=False, reset_num_timesteps=True)
            first = False
        else:
            model.learn(target, progress_bar=False, reset_num_timesteps=False)

        ep_rew, n_eps = _ep_stats(model)
        if eval_enabled and eval_env is not None:
            eval_mean, eval_std, _ = evaluate_policy(
                model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
            )
            solved = np.isfinite(eval_mean) and eval_mean >= threshold and n_eps >= window
        else:
            solved = is_solved(model, threshold, window=window)
        if solved:
            return {
                "solved": True,
                "steps_to_solve": int(model.num_timesteps),
                "elapsed_s": time.perf_counter() - t0,
                "ep_rew_mean": ep_rew,
                "eval_mean": eval_mean,
                "eval_std": eval_std,
                "n_episodes": n_eps,
            }
        # Bail at ≥50% budget if still far below threshold (scaled budgets get huge).
        if (
            model.num_timesteps >= max(max_steps // 2, chunk_steps)
            and n_eps >= window
            and np.isfinite(ep_rew)
            and ep_rew < threshold - gap
        ):
            break

    ep_rew, n_eps = _ep_stats(model)
    if eval_enabled and eval_env is not None and np.isnan(eval_mean):
        eval_mean, eval_std, _ = evaluate_policy(
            model, eval_env, n_eval_episodes=n_eval_episodes, deterministic=True
        )
    return {
        "solved": False,
        "steps_to_solve": int(model.num_timesteps),
        "elapsed_s": time.perf_counter() - t0,
        "ep_rew_mean": ep_rew,
        "eval_mean": eval_mean,
        "eval_std": eval_std,
        "n_episodes": n_eps,
    }


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in columns})


def _aggregate(raw_rows: list[dict], seeds: int) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for r in raw_rows:
        key = (r["algo"], r["env"], int(r["n_envs"]))
        groups.setdefault(key, []).append(r)

    def _q(xs, q):
        return float(np.percentile(xs, q)) if xs else ""

    agg: list[dict] = []
    for (algo, env, n), rows in sorted(groups.items(), key=lambda x: (x[0][0], x[0][2])):
        solved_rows = [r for r in rows if r["solved"]]
        steps = [float(r["steps_to_solve"]) for r in solved_rows]
        elapsed = [float(r["elapsed_s"]) for r in solved_rows]
        rews = [v for r in rows if (v := _finite_float(r.get("ep_rew_mean"))) is not None]
        evals = [v for r in rows if (v := _finite_float(r.get("eval_mean"))) is not None]
        agg.append(
            {
                "algo": algo,
                "env": env,
                "n_envs": n,
                "seeds": seeds,
                "threshold": rows[0]["threshold"],
                "budget": rows[0]["budget"],
                "solve_rate": round(len(solved_rows) / max(len(rows), 1), 3),
                "steps_mean": round(_mean(steps), 1) if steps else "",
                "steps_std": round(_std(steps), 1) if steps else "",
                "steps_median": round(_q(steps, 50), 1) if steps else "",
                "steps_q25": round(_q(steps, 25), 1) if steps else "",
                "steps_q75": round(_q(steps, 75), 1) if steps else "",
                "steps_min": int(min(steps)) if steps else "",
                "steps_max": int(max(steps)) if steps else "",
                "elapsed_s_mean": round(_mean(elapsed), 2) if elapsed else "",
                "ep_rew_mean": round(_mean(rews), 1) if rews else "",
                "eval_mean": round(_mean(evals), 1) if evals else "",
                "eval_std": round(_std(evals), 1) if evals else "",
                "eval_median": round(_q(evals, 50), 1) if evals else "",
                "eval_q25": round(_q(evals, 25), 1) if evals else "",
                "eval_q75": round(_q(evals, 75), 1) if evals else "",
            }
        )
    return agg


def _print_tables(agg_rows: list[dict]) -> None:
    if not agg_rows:
        return
    table = [
        [
            r["algo"],
            r["env"],
            r["n_envs"],
            r["budget"],
            r["solve_rate"],
            r["steps_mean"],
            r["steps_std"],
            r["elapsed_s_mean"],
            r["ep_rew_mean"],
            r["eval_mean"],
            r["eval_std"],
        ]
        for r in agg_rows
    ]
    print(
        tabulate(
            table,
            headers=[
                "algo",
                "env",
                "n_envs",
                "budget",
                "solve_rate",
                "steps_mean",
                "steps_std",
                "elapsed_s_mean",
                "ep_rew_mean",
                "eval_mean",
                "eval_std",
            ],
            tablefmt="github",
        )
    )


def _load_raw(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(newline="") as f:
        rows = []
        for r in csv.DictReader(f):
            r["n_envs"] = int(r["n_envs"])
            r["seed"] = int(r["seed"])
            r["threshold"] = float(r["threshold"])
            r["budget"] = int(r["budget"])
            r["solved"] = str(r["solved"]).lower() in ("1", "true", "True")
            r["steps_to_solve"] = int(float(r["steps_to_solve"]))
            r["elapsed_s"] = float(r["elapsed_s"])
            r["n_episodes"] = int(r["n_episodes"])
            rows.append(r)
        return rows


def run_sweep(
    algos: list[str],
    n_envs_list: list[int],
    *,
    sweeps: dict,
    seeds: int,
    base_budget: int,
    chunk_steps: int,
    csv_path: Path,
    raw_csv_path: Path | None = None,
    resume: bool = True,
    eval_enabled: bool = True,
    n_eval_episodes: int = 10,
) -> list[dict]:
    seed_list = list(range(seeds))
    if raw_csv_path is None:
        raw_csv_path = csv_path.with_name(csv_path.stem + "_raw.csv")

    raw_rows: list[dict] = _load_raw(raw_csv_path) if resume else []
    done = {(r["algo"], int(r["n_envs"]), int(r["seed"])) for r in raw_rows}
    if done:
        print(f"resume: skipping {len(done)} completed runs from {raw_csv_path}")

    total = len(algos) * len(n_envs_list) * seeds
    pbar = tqdm(total=total, desc="solving", unit="run", dynamic_ncols=True)
    try:
        for key in algos:
            algo, env_name, make_fn = sweeps[key]
            threshold = SOLVE_THRESHOLD[env_name]
            for n in n_envs_list:
                budget = step_budget(n, base=base_budget)
                chunk = chunk_steps_for(n, env_name, base=chunk_steps)
                for seed in seed_list:
                    if (algo, n, seed) in done:
                        pbar.update(1)
                        continue
                    model = make_fn(n, seed)
                    eval_env = ENV_CLS[env_name](n_envs=1, seed=seed + 100_000) if eval_enabled else None
                    result = train_until_solved(
                        model,
                        threshold=threshold,
                        max_steps=budget,
                        chunk_steps=chunk,
                        env_name=env_name,
                        eval_env=eval_env,
                        eval_enabled=eval_enabled,
                        n_eval_episodes=n_eval_episodes,
                    )
                    status = "SOLVED" if result["solved"] else "FAIL"
                    ep = result["ep_rew_mean"]
                    ev = result["eval_mean"]
                    ep_s = f"{ep:.1f}" if np.isfinite(ep) else "nan"
                    ev_s = f"{ev:.1f}" if np.isfinite(ev) else "nan"
                    pbar.write(
                        f"{algo} n_envs={n} seed={seed}: {status} "
                        f"steps={result['steps_to_solve']} elapsed_s={result['elapsed_s']:.1f} "
                        f"ep_rew={ep_s} eval_mean={ev_s}",
                    )
                    raw_rows.append(
                        {
                            "algo": algo,
                            "env": env_name,
                            "n_envs": n,
                            "seed": seed,
                            "threshold": threshold,
                            "budget": budget,
                            "solved": bool(result["solved"]),
                            "steps_to_solve": result["steps_to_solve"],
                            "elapsed_s": round(result["elapsed_s"], 3),
                            "ep_rew_mean": round(ep, 2) if np.isfinite(ep) else "",
                            "eval_mean": round(ev, 2) if np.isfinite(ev) else "",
                            "eval_std": round(result["eval_std"], 2) if np.isfinite(result["eval_std"]) else "",
                            "n_episodes": result["n_episodes"],
                        }
                    )
                    done.add((algo, n, seed))
                    _write_csv(raw_csv_path, raw_rows, RAW_COLUMNS)
                    pbar.set_postfix_str(f"{algo} n={n} {status}")
                    pbar.update(1)
    finally:
        pbar.close()

    agg_rows = _aggregate(raw_rows, seeds=seeds)
    _write_csv(csv_path, agg_rows, AGG_COLUMNS)
    print()
    _print_tables(agg_rows)
    print(f"\nwrote {csv_path}")
    print(f"wrote {raw_csv_path}")
    return agg_rows


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--algo", choices=["ppo", "sac", "td3", "all"], default="all")
    p.add_argument("--n-envs", type=int, nargs="+", default=None)
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS)
    p.add_argument(
        "--base-budget",
        type=int,
        default=DEFAULT_BASE_BUDGET,
        help=f"env-step budget at n_envs={REF_N_ENVS}; scaled linearly with n_envs",
    )
    p.add_argument(
        "--chunk-steps",
        type=int,
        default=CHUNK_STEPS,
        help="minimum env-steps between solve checks (also ≥ n_envs * max_ep_len)",
    )
    p.add_argument("--csv", type=Path, default=Path("outputs/solving_results.csv"))
    p.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="per-seed CSV (default: <csv>_raw.csv)",
    )
    p.add_argument(
        "--no-resume",
        action="store_true",
        help="ignore existing raw CSV and rerun from scratch",
    )
    p.add_argument(
        "--use-baseline-configs",
        action="store_true",
        help="load hyperparameters from configs/{algo}_baseline.yaml instead of defaults",
    )
    p.add_argument("--eval", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--n-eval-episodes", type=int, default=10)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    algos = list(SWEEPS) if args.algo == "all" else [args.algo]
    n_envs_list = list(args.n_envs) if args.n_envs is not None else list(N_ENVS)

    if args.use_baseline_configs:
        sweeps = {}
        for key in algos:
            path = Path("configs") / f"{key}_baseline.yaml"
            if not path.exists():
                raise FileNotFoundError(path)
            sweeps[key] = _make_from_yaml(path)
        print("using baseline YAML configs:", [str(sweeps[k][0]) for k in sweeps])
    else:
        sweeps = SWEEPS

    budgets = {n: step_budget(n, base=args.base_budget) for n in n_envs_list}
    print(
        f"solve window={SOLVE_WINDOW} episodes | thresholds={SOLVE_THRESHOLD}\n"
        f"n_envs={list(n_envs_list)} | base_budget={args.base_budget} "
        f"(at n_envs={REF_N_ENVS}) | budgets={budgets}"
    )
    run_sweep(
        algos,
        n_envs_list,
        sweeps=sweeps,
        seeds=args.seeds,
        base_budget=args.base_budget,
        chunk_steps=args.chunk_steps,
        csv_path=args.csv,
        raw_csv_path=args.raw_csv,
        resume=not args.no_resume,
        eval_enabled=args.eval,
        n_eval_episodes=args.n_eval_episodes,
    )


if __name__ == "__main__":
    main()
