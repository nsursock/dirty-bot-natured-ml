"""n_envs scaling sweep for PPO / SAC / TD3 — NOT part of pytest CI.

Design (pure scaling + stable timing):
  - each timed window lasts ~MIN_SECONDS (default 2s), not a fixed micro-step budget
  - chunk_steps=8192 per learn() call; loops until wall-clock floor is hit
  - --seeds N (default 5) → mean/std/min/max
  - absolute_speedup, parallel_efficiency, incremental_efficiency
  - train/env fps per env
  - system: CPU%, GPU% + GPU power (powermetrics), MLX mem, smctemp

Run:
  python benchmarks/benchmark_scaling.py
  python benchmarks/benchmark_scaling.py --algo ppo --n-envs 1024 2048 4096 8192 --seeds 5
  python benchmarks/benchmark_scaling.py --min-seconds 2 --csv outputs/scaling_results.csv
"""

from __future__ import annotations

import argparse
import cProfile
import csv
import pstats
import re
import subprocess
import threading
import time
from contextlib import contextmanager
from io import StringIO
from pathlib import Path

import mlx.core as mx
import numpy as np
from tabulate import tabulate

from dbn.reinforcement.algos.ppo import PPO
from dbn.reinforcement.algos.sac import SAC
from dbn.reinforcement.algos.td3 import TD3
from dbn.reinforcement.envs import CartPole, Pendulum

N_ENVS = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192)
# Chunk size per learn() call; wall-clock floor is MIN_SECONDS (not step count).
CHUNK_STEPS = 8192
MIN_SECONDS = 2.0
ROLLOUT_TARGET = 2048
DEFAULT_SEEDS = 5
ENV_BENCH_ITERS = 200  # floor; measure_env_fps also honors MIN_SECONDS
PROFILE_N_ENVS = (256, 1024, 8192)

# Per-run raw rows (one seed each)
RAW_COLUMNS = (
    "algo",
    "env",
    "n_envs",
    "seed",
    "steps",
    "elapsed_s",
    "train_fps",
    "env_fps",
    "train_fps_per_env",
    "env_fps_per_env",
    "cpu_util_pct",
    "gpu_util_pct",
    "gpu_power_w",
    "gpu_mem_mb",
    "cpu_temp_c",
    "gpu_temp_c",
)

# Aggregated rows (across seeds) + scaling derived from mean train_fps
AGG_COLUMNS = (
    "algo",
    "env",
    "n_envs",
    "seeds",
    "steps",
    "train_fps_mean",
    "train_fps_std",
    "train_fps_min",
    "train_fps_max",
    "env_fps_mean",
    "train_fps_per_env",
    "env_fps_per_env",
    "absolute_speedup",
    "parallel_efficiency",
    "incremental_efficiency",
    "cpu_util_pct",
    "gpu_util_pct",
    "gpu_power_w",
    "gpu_mem_mb",
    "cpu_temp_c",
    "gpu_temp_c",
)


def _fps(numer: float, elapsed: float) -> float:
    return numer / max(elapsed, 1e-9)


def _round_or_blank(x: float | None, nd: int = 1):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return ""
    return round(float(x), nd)


# ---------------------------------------------------------------------------
# system metrics
# ---------------------------------------------------------------------------

def _mlx_peak_mem_mb() -> float | None:
    try:
        return float(mx.get_peak_memory()) / (1024 ** 2)
    except Exception:
        return None


def _mlx_active_mem_mb() -> float | None:
    try:
        return float(mx.get_active_memory()) / (1024 ** 2)
    except Exception:
        return None


def _reset_mlx_peak() -> None:
    reset = getattr(mx, "reset_peak_memory", None)
    if callable(reset):
        reset()


def _parse_temp_c(text: str) -> float | None:
    digits = "".join(ch if (ch.isdigit() or ch == ".") else " " for ch in text)
    parts = [p for p in digits.split() if p]
    if not parts:
        return None
    val = float(parts[0])
    return val if val > 1.0 else None


def _smctemp(flag: str) -> float | None:
    try:
        out = subprocess.check_output(
            ["smctemp", flag, "-f"], text=True, stderr=subprocess.DEVNULL, timeout=2.0
        )
        return _parse_temp_c(out)
    except Exception:
        return None


def _read_cpu_temp_c() -> float | None:
    t = _smctemp("-c")
    if t is not None:
        return t
    try:
        out = subprocess.check_output(
            ["osx-cpu-temp"], text=True, stderr=subprocess.DEVNULL, timeout=1.0
        )
        return _parse_temp_c(out)
    except Exception:
        return None


def _read_gpu_temp_c() -> float | None:
    return _smctemp("-g")


def _powermetrics_gpu_sample(interval_ms: int = 200) -> tuple[float | None, float | None]:
    """
    Return (gpu_util_pct, gpu_power_w) from powermetrics.
    Tries `sudo -n` first (passwordless sudoers), then bare powermetrics.
    """
    cmds = [
        ["sudo", "-n", "powermetrics", "--samplers", "gpu_power", "-i", str(interval_ms), "-n", "1"],
        ["powermetrics", "--samplers", "gpu_power", "-i", str(interval_ms), "-n", "1"],
    ]
    out = None
    for cmd in cmds:
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=5.0)
            break
        except Exception:
            continue
    if not out:
        return None, None

    util = None
    power_w = None
    for line in out.splitlines():
        if "HW active residency" in line:
            m = re.search(r"([\d.]+)\s*%", line)
            if m:
                util = float(m.group(1))
        if line.strip().startswith("GPU Power:"):
            # e.g. "GPU Power: 89 mW" or "GPU Power: 1.2 W"
            m = re.search(r"([\d.]+)\s*(mW|W)", line, re.I)
            if m:
                val = float(m.group(1))
                power_w = val / 1000.0 if m.group(2).lower() == "mw" else val
    return util, power_w


class _CpuSampler:
    def __init__(self, interval: float = 0.05):
        self.interval = interval
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.samples: list[float] = []
        self._psutil = None
        self._proc = None
        try:
            import psutil

            self._psutil = psutil
        except ImportError:
            pass

    def start(self) -> None:
        if self._psutil is None:
            return
        self._psutil.cpu_percent(interval=None)
        self._proc = self._psutil.Process()
        self._proc.cpu_percent(interval=None)
        self._stop.clear()
        self.samples.clear()

        def _run():
            while not self._stop.wait(self.interval):
                self.samples.append(float(self._psutil.cpu_percent(interval=None)))

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> float | None:
        if self._psutil is None:
            return None
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        sys_pct = (
            float(sum(self.samples) / len(self.samples))
            if self.samples
            else float(self._psutil.cpu_percent(interval=0.05))
        )
        proc_pct = 0.0
        if self._proc is not None:
            n_cpu = max(int(self._psutil.cpu_count(logical=True) or 1), 1)
            proc_pct = float(self._proc.cpu_percent(interval=0.05)) / n_cpu
        return max(sys_pct, proc_pct)


class _GpuSampler:
    """Background powermetrics sampler for GPU util% and power (W)."""

    def __init__(self, interval_s: float = 0.4):
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.util_samples: list[float] = []
        self.power_samples: list[float] = []

    def start(self) -> None:
        self._stop.clear()
        self.util_samples.clear()
        self.power_samples.clear()

        def _run():
            while not self._stop.is_set():
                util, power = _powermetrics_gpu_sample(interval_ms=200)
                if util is not None:
                    self.util_samples.append(util)
                if power is not None:
                    self.power_samples.append(power)
                if self._stop.wait(self.interval_s):
                    break

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()

    def stop(self) -> tuple[float | None, float | None]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6.0)
        util = (
            float(sum(self.util_samples) / len(self.util_samples))
            if self.util_samples
            else None
        )
        power = (
            float(sum(self.power_samples) / len(self.power_samples))
            if self.power_samples
            else None
        )
        # one-shot fallback if background never got a sample
        if util is None and power is None:
            return _powermetrics_gpu_sample(interval_ms=200)
        return util, power


@contextmanager
def sample_system_metrics(*, sample_gpu: bool = True):
    cpu = _CpuSampler()
    gpu = _GpuSampler() if sample_gpu else None
    _reset_mlx_peak()
    cpu_t0 = _read_cpu_temp_c()
    gpu_t0 = _read_gpu_temp_c()
    cpu.start()
    if gpu is not None:
        gpu.start()
    metrics: dict[str, float | None] = {
        "cpu_util_pct": None,
        "gpu_util_pct": None,
        "gpu_power_w": None,
        "gpu_mem_mb": None,
        "cpu_temp_c": None,
        "gpu_temp_c": None,
    }
    try:
        yield metrics
    finally:
        metrics["cpu_util_pct"] = cpu.stop()
        if gpu is not None:
            metrics["gpu_util_pct"], metrics["gpu_power_w"] = gpu.stop()
        metrics["gpu_mem_mb"] = _mlx_peak_mem_mb() or _mlx_active_mem_mb()
        cpu_t1 = _read_cpu_temp_c()
        gpu_t1 = _read_gpu_temp_c()
        metrics["cpu_temp_c"] = cpu_t1 if cpu_t1 is not None else cpu_t0
        metrics["gpu_temp_c"] = gpu_t1 if gpu_t1 is not None else gpu_t0


# ---------------------------------------------------------------------------
# benches — wall-clock floored timing (stable FPS at high n_envs)
# ---------------------------------------------------------------------------

def _timed_learn(
    model,
    *,
    chunk_steps: int,
    min_seconds: float,
    sample_gpu: bool,
) -> tuple[float, int, float, dict]:
    """
    Run learn(chunk_steps) in a loop until wall time >= min_seconds.
    Returns (train_fps, total_env_steps, elapsed_s, sys_metrics).
    """
    total_steps = 0
    with sample_system_metrics(sample_gpu=sample_gpu) as sys_m:
        t0 = time.perf_counter()
        while True:
            model.learn(chunk_steps, progress_bar=False, reset_num_timesteps=True)
            total_steps += int(model.num_timesteps)
            if (time.perf_counter() - t0) >= min_seconds:
                break
        elapsed = time.perf_counter() - t0
    return _fps(total_steps, elapsed), total_steps, elapsed, dict(sys_m)


def measure_env_fps(
    env_name: str,
    n_envs: int,
    *,
    min_seconds: float = MIN_SECONDS,
    min_iters: int = ENV_BENCH_ITERS,
) -> float:
    if env_name == "CartPole":
        env = CartPole(n_envs=n_envs, seed=0)
        env.reset()
        actions = np.zeros(n_envs, dtype=np.int32) if n_envs > 1 else 0
    else:
        env = Pendulum(n_envs=n_envs, seed=0)
        env.reset()
        actions = (
            np.zeros((n_envs, 1), dtype=np.float32)
            if n_envs > 1
            else np.zeros(1, dtype=np.float32)
        )
    for _ in range(5):
        env.step(actions)

    iters = 0
    t0 = time.perf_counter()
    while True:
        env.step(actions)
        iters += 1
        if iters >= min_iters and (time.perf_counter() - t0) >= min_seconds:
            break
    return _fps(iters * n_envs, time.perf_counter() - t0)


def bench_ppo(
    n_envs: int,
    steps: int = CHUNK_STEPS,
    seed: int = 0,
    *,
    sample_gpu: bool = True,
    min_seconds: float = MIN_SECONDS,
) -> tuple[float, int, float, dict]:
    n_steps = max(1, ROLLOUT_TARGET // n_envs)
    batch = n_steps * n_envs
    env = CartPole(n_envs=n_envs, seed=seed)
    model = PPO(
        "MlpPolicy",
        env,
        n_steps=n_steps,
        batch_size=batch,
        n_epochs=1,
        policy_kwargs={"net_arch": (64, 64)},
        seed=seed,
    )
    model.learn(n_steps * n_envs, progress_bar=False, reset_num_timesteps=True)
    return _timed_learn(
        model, chunk_steps=steps, min_seconds=min_seconds, sample_gpu=sample_gpu
    )


def bench_sac(
    n_envs: int,
    steps: int = CHUNK_STEPS,
    seed: int = 0,
    *,
    sample_gpu: bool = True,
    min_seconds: float = MIN_SECONDS,
) -> tuple[float, int, float, dict]:
    env = Pendulum(n_envs=n_envs, seed=seed)
    model = SAC(
        "MlpPolicy",
        env,
        learning_starts=min(1000, max(n_envs, steps // 4)),
        buffer_size=100_000,
        batch_size=256,
        policy_kwargs={"net_arch": (256, 256)},
        seed=seed,
    )
    model.learn(n_envs, progress_bar=False, reset_num_timesteps=True)
    return _timed_learn(
        model, chunk_steps=steps, min_seconds=min_seconds, sample_gpu=sample_gpu
    )


def bench_td3(
    n_envs: int,
    steps: int = CHUNK_STEPS,
    seed: int = 0,
    *,
    sample_gpu: bool = True,
    min_seconds: float = MIN_SECONDS,
) -> tuple[float, int, float, dict]:
    env = Pendulum(n_envs=n_envs, seed=seed)
    model = TD3(
        "MlpPolicy",
        env,
        learning_starts=min(1000, max(n_envs, steps // 4)),
        buffer_size=100_000,
        batch_size=256,
        policy_kwargs={"net_arch": (256, 256)},
        seed=seed,
    )
    model.learn(n_envs, progress_bar=False, reset_num_timesteps=True)
    return _timed_learn(
        model, chunk_steps=steps, min_seconds=min_seconds, sample_gpu=sample_gpu
    )


SWEEPS = {
    "ppo": ("PPO", "CartPole", bench_ppo),
    "sac": ("SAC", "Pendulum", bench_sac),
    "td3": ("TD3", "Pendulum", bench_td3),
}


# ---------------------------------------------------------------------------
# sweep / aggregate
# ---------------------------------------------------------------------------

def _mean(xs: list[float]) -> float:
    return float(sum(xs) / max(len(xs), 1))


def _std(xs: list[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return float(np.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)))


def run_sweep(
    algos: list[str],
    n_envs_list: list[int],
    *,
    steps: int,
    seeds: int,
    csv_path: Path,
    raw_csv_path: Path | None = None,
    sample_gpu: bool = True,
    min_seconds: float = MIN_SECONDS,
) -> list[dict]:
    raw_rows: list[dict] = []
    seed_list = list(range(seeds))

    for key in algos:
        algo, env_name, fn = SWEEPS[key]
        for n in n_envs_list:
            env_fps = measure_env_fps(env_name, n, min_seconds=min_seconds)
            for seed in seed_list:
                print(
                    f"  {algo} n_envs={n} seed={seed} (min {min_seconds:.1f}s) ...",
                    flush=True,
                )
                train_fps, done_steps, elapsed, sys_m = fn(
                    n,
                    steps=steps,
                    seed=seed,
                    sample_gpu=sample_gpu,
                    min_seconds=min_seconds,
                )
                raw_rows.append(
                    {
                        "algo": algo,
                        "env": env_name,
                        "n_envs": n,
                        "seed": seed,
                        "steps": int(done_steps),
                        "elapsed_s": round(elapsed, 4),
                        "train_fps": round(train_fps, 1),
                        "env_fps": round(env_fps, 1),
                        "train_fps_per_env": round(train_fps / n, 1),
                        "env_fps_per_env": round(env_fps / n, 1),
                        "cpu_util_pct": _round_or_blank(sys_m.get("cpu_util_pct"), 1),
                        "gpu_util_pct": _round_or_blank(sys_m.get("gpu_util_pct"), 1),
                        "gpu_power_w": _round_or_blank(sys_m.get("gpu_power_w"), 3),
                        "gpu_mem_mb": _round_or_blank(sys_m.get("gpu_mem_mb"), 1),
                        "cpu_temp_c": _round_or_blank(sys_m.get("cpu_temp_c"), 1),
                        "gpu_temp_c": _round_or_blank(sys_m.get("gpu_temp_c"), 1),
                    }
                )

    if raw_csv_path is None:
        raw_csv_path = csv_path.with_name(csv_path.stem + "_raw.csv")
    _write_csv(raw_csv_path, raw_rows, RAW_COLUMNS)

    agg_rows = _aggregate(
        raw_rows, baseline_n=n_envs_list[0], seeds=seeds, chunk_steps=steps
    )
    _write_csv(csv_path, agg_rows, AGG_COLUMNS)
    _print_tables(agg_rows)
    print(f"\nwrote {csv_path}")
    print(f"wrote {raw_csv_path}")
    return agg_rows


def _aggregate(
    raw_rows: list[dict], *, baseline_n: int, seeds: int, chunk_steps: int
) -> list[dict]:
    # group by (algo, env, n_envs)
    groups: dict[tuple, list[dict]] = {}
    for r in raw_rows:
        key = (r["algo"], r["env"], int(r["n_envs"]))
        groups.setdefault(key, []).append(r)

    ordered_keys = sorted(groups.keys(), key=lambda k: (k[0], k[2]))
    baseline_fps: dict[str, float] = {}
    for (algo, _env, n), rows in groups.items():
        if n == baseline_n:
            baseline_fps[algo] = _mean([float(r["train_fps"]) for r in rows])

    prev_mean: dict[str, tuple[int, float]] = {}
    agg: list[dict] = []
    for algo, env, n in ordered_keys:
        rows = groups[(algo, env, n)]
        t_fps = [float(r["train_fps"]) for r in rows]
        e_fps = [float(r["env_fps"]) for r in rows]
        elaps = [float(r["elapsed_s"]) for r in rows]
        step_counts = [float(r["steps"]) for r in rows]
        mean_t = _mean(t_fps)
        mean_e = _mean(e_fps)
        base = baseline_fps.get(algo, mean_t)
        abs_speedup = mean_t / max(base, 1e-9)
        parallel_eff = abs_speedup / max(n / baseline_n, 1e-9)

        incr = ""
        if algo in prev_mean:
            prev_n, prev_fps = prev_mean[algo]
            if n == 2 * prev_n and prev_fps > 0:
                incr = round((mean_t / prev_fps) / 2.0, 3)
        prev_mean[algo] = (n, mean_t)

        def _avg_sys(col: str):
            vals = [float(r[col]) for r in rows if r[col] != "" and r[col] is not None]
            return _round_or_blank(_mean(vals) if vals else None, 3 if col == "gpu_power_w" else 1)

        agg.append(
            {
                "algo": algo,
                "env": env,
                "n_envs": n,
                "seeds": seeds,
                "steps": int(round(_mean(step_counts))),
                "train_fps_mean": round(mean_t, 1),
                "train_fps_std": round(_std(t_fps), 1),
                "train_fps_min": round(min(t_fps), 1),
                "train_fps_max": round(max(t_fps), 1),
                "env_fps_mean": round(mean_e, 1),
                "train_fps_per_env": round(mean_t / n, 1),
                "env_fps_per_env": round(mean_e / n, 1),
                "absolute_speedup": round(abs_speedup, 3),
                "parallel_efficiency": round(parallel_eff, 3),
                "incremental_efficiency": incr,
                "cpu_util_pct": _avg_sys("cpu_util_pct"),
                "gpu_util_pct": _avg_sys("gpu_util_pct"),
                "gpu_power_w": _avg_sys("gpu_power_w"),
                "gpu_mem_mb": _avg_sys("gpu_mem_mb"),
                "cpu_temp_c": _avg_sys("cpu_temp_c"),
                "gpu_temp_c": _avg_sys("gpu_temp_c"),
                # stash mean elapsed for table (not in AGG_COLUMNS → ignored by csv)
                "_elapsed_mean": round(_mean(elaps), 3),
                "_chunk_steps": chunk_steps,
            }
        )
    return agg


def _write_csv(path: Path, rows: list[dict], columns: tuple[str, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})


def _fmt(v, kind: str = "num") -> str:
    if v == "" or v is None:
        return ""
    if kind == "int":
        return f"{float(v):,.0f}"
    if kind == "fps_env":
        return f"{float(v):,.1f}"
    if kind == "ratio":
        return f"{float(v):.3f}"
    if kind == "pct":
        return f"{float(v):.1f}"
    if kind == "power":
        return f"{float(v):.3f}"
    return str(v)


def _print_tables(rows: list[dict]) -> None:
    primary = []
    for r in rows:
        primary.append(
            [
                r["algo"],
                r["n_envs"],
                _fmt(r["train_fps_mean"], "int"),
                _fmt(r["train_fps_std"], "int"),
                f"{_fmt(r['train_fps_min'], 'int')}–{_fmt(r['train_fps_max'], 'int')}",
                _fmt(r["train_fps_per_env"], "fps_env"),
                _fmt(r["env_fps_mean"], "int"),
                _fmt(r["env_fps_per_env"], "fps_env"),
                _fmt(r["absolute_speedup"], "ratio"),
                _fmt(r["parallel_efficiency"], "ratio"),
                _fmt(r["incremental_efficiency"], "ratio"),
            ]
        )
    print("\n=== throughput & scaling (mean±std over seeds, ≥min wall time) ===")
    print(
        tabulate(
            primary,
            headers=[
                "algo",
                "n_envs",
                "train_fps",
                "std",
                "min–max",
                "train/env",
                "env_fps",
                "env/env",
                "abs_speedup",
                "par_eff",
                "incr_eff",
            ],
            tablefmt="github",
            disable_numparse=True,
        )
    )

    timing = []
    for r in rows:
        timing.append(
            [
                r["algo"],
                r["n_envs"],
                r["steps"],
                f"{r.get('_elapsed_mean', '')}",
            ]
        )
    print("\n=== timing sanity (mean steps & elapsed per seed) ===")
    print(
        tabulate(
            timing,
            headers=["algo", "n_envs", "steps_mean", "elapsed_s"],
            tablefmt="github",
            disable_numparse=True,
        )
    )

    secondary = []
    for r in rows:
        secondary.append(
            [
                r["algo"],
                r["n_envs"],
                _fmt(r["cpu_util_pct"], "pct"),
                _fmt(r["gpu_util_pct"], "pct"),
                _fmt(r["gpu_power_w"], "power"),
                _fmt(r["gpu_mem_mb"], "fps_env"),
                _fmt(r["cpu_temp_c"], "pct"),
                _fmt(r["gpu_temp_c"], "pct"),
            ]
        )
    print("\n=== system (mean over seeds) ===")
    print(
        tabulate(
            secondary,
            headers=[
                "algo",
                "n_envs",
                "cpu%",
                "gpu%",
                "gpu_W",
                "gpu_mem_mb",
                "cpu_°C",
                "gpu_°C",
            ],
            tablefmt="github",
            disable_numparse=True,
        )
    )


def run_profile(
    algos: list[str],
    n_envs_list: list[int],
    *,
    out_dir: Path,
    steps: int,
    min_seconds: float,
    top: int = 40,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for key in algos:
        algo, env_name, fn = SWEEPS[key]
        for n in n_envs_list:
            tag = f"{key}_n{n}"
            print(f"\n=== cProfile {algo}/{env_name} n_envs={n} ===")
            pr = cProfile.Profile()
            pr.enable()
            fps, done, elapsed, _sys = fn(
                n, steps=steps, seed=0, sample_gpu=False, min_seconds=min_seconds
            )
            pr.disable()
            print(
                f"{algo}/{env_name} n_envs={n:4d}  train_fps={fps:10,.1f}  "
                f"train/env={fps / n:8,.1f}  steps={done:6d}  elapsed_s={elapsed:6.2f}"
            )
            text = StringIO()
            stats = pstats.Stats(pr, stream=text).sort_stats(pstats.SortKey.CUMULATIVE)
            stats.print_stats(top)
            text.write("\n--- by tottime ---\n")
            stats.sort_stats(pstats.SortKey.TIME)
            stats.print_stats(top)
            body = text.getvalue()
            print(body)
            pr.dump_stats(str(out_dir / f"{tag}.pstats"))
            (out_dir / f"{tag}.txt").write_text(body)
            print(f"wrote {out_dir / tag}.{{pstats,txt}}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", action="store_true")
    p.add_argument("--algo", choices=["ppo", "sac", "td3", "all"], default="all")
    p.add_argument("--n-envs", type=int, nargs="+", default=None)
    p.add_argument(
        "--steps",
        type=int,
        default=CHUNK_STEPS,
        help="env-steps per learn() chunk (default 8192); looped until --min-seconds",
    )
    p.add_argument(
        "--min-seconds",
        type=float,
        default=MIN_SECONDS,
        help="minimum wall-clock seconds per timed measurement (default 2.0)",
    )
    p.add_argument("--seeds", type=int, default=DEFAULT_SEEDS, help="seeds per (algo, n_envs)")
    p.add_argument("--csv", type=Path, default=Path("outputs/scaling_results.csv"))
    p.add_argument(
        "--raw-csv",
        type=Path,
        default=None,
        help="per-seed CSV (default: <csv>_raw.csv)",
    )
    p.add_argument(
        "--no-gpu-sample",
        action="store_true",
        help="skip powermetrics GPU util/power sampling",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/profiles"))
    p.add_argument("--top", type=int, default=40)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    algos = list(SWEEPS) if args.algo == "all" else [args.algo]
    if args.n_envs is not None:
        n_envs_list = list(args.n_envs)
    elif args.profile:
        n_envs_list = list(PROFILE_N_ENVS)
    else:
        n_envs_list = list(N_ENVS)

    if not args.no_gpu_sample and not args.profile:
        util, power = _powermetrics_gpu_sample()
        if util is None and power is None:
            print(
                "warning: powermetrics GPU sample failed "
                "(need passwordless sudo for powermetrics); gpu%%/W will be blank"
            )
        else:
            print(f"powermetrics ok: gpu_util≈{util}%  gpu_power≈{power} W")

    print(f"timing floor: {args.min_seconds:.1f}s per measurement (chunk={args.steps})")

    if args.profile:
        run_profile(
            algos,
            n_envs_list,
            out_dir=args.out_dir,
            steps=args.steps,
            min_seconds=args.min_seconds,
            top=args.top,
        )
    else:
        run_sweep(
            algos,
            n_envs_list,
            steps=args.steps,
            seeds=args.seeds,
            csv_path=args.csv,
            raw_csv_path=args.raw_csv,
            sample_gpu=not args.no_gpu_sample,
            min_seconds=args.min_seconds,
        )


if __name__ == "__main__":
    main()
