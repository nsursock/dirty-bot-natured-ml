"""Resample TensorBoard progress.csv files down to ~100 rows.

TD3 special case: policy-delayed actor updates mean `train/actor_loss` is
often zero on non-update steps. We first keep only rows where it is
non-zero, then uniformly resample to the target row count.

    python benchmarks/resample_tensorboard_csv.py --log-dir outputs/tensorboard
    python benchmarks/resample_tensorboard_csv.py --run TD3_n4_seed0_1 --target 100
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np


def _read_csv(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)
    return fieldnames, rows


def _float_or_nan(cell: str) -> float:
    try:
        return float(cell)
    except ValueError:
        return float("nan")


def _has_actor_delay_pattern(rows: list[dict[str, str]], col: str) -> bool:
    """Detect TD3-like delayed actor updates (some zero actor_loss rows)."""
    values = [_float_or_nan(row.get(col, "")) for row in rows]
    zeros = sum(1 for v in values if v == 0.0)
    non_zeros = sum(1 for v in values if v != 0.0 and v == v)
    return zeros > 0 and non_zeros > 0


def _filter_actor_rows(rows: list[dict[str, str]], col: str) -> list[dict[str, str]]:
    return [row for row in rows if _float_or_nan(row.get(col, "")) != 0.0]


def _resample_indices(n: int, target: int) -> list[int]:
    if n <= target:
        return list(range(n))
    return list(np.unique(np.linspace(0, n - 1, target).astype(int)))


def resample(csv_path: Path, target: int = 100, out_dir: Path | None = None) -> Path:
    fieldnames, rows = _read_csv(csv_path)
    if not rows:
        raise ValueError(f"{csv_path} is empty")

    actor_col = "train/actor_loss"
    if actor_col in fieldnames and _has_actor_delay_pattern(rows, actor_col):
        rows = _filter_actor_rows(rows, actor_col)

    indices = _resample_indices(len(rows), target)
    sampled = [rows[i] for i in indices]

    out_name = f"{csv_path.stem}_resampled_{len(sampled)}.csv"
    out = (out_dir or csv_path.parent) / out_name
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sampled)
    return out


def _find_csvs(log_dir: Path) -> dict[str, Path]:
    runs: dict[str, Path] = {}
    for csv_path in log_dir.rglob("progress.csv"):
        runs[csv_path.parent.name] = csv_path
    return runs


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, default=Path("outputs/tensorboard"))
    p.add_argument("--run", type=str, default=None, help="specific run directory name")
    p.add_argument("--target", type=int, default=100, help="target row count")
    p.add_argument("--out-dir", type=Path, default=None, help="output directory (default: same as source)")
    args = p.parse_args(argv)

    runs = _find_csvs(args.log_dir)
    if not runs:
        raise FileNotFoundError(f"no progress.csv files found under {args.log_dir}")

    selected = {args.run: runs[args.run]} if args.run else runs
    if args.run and args.run not in runs:
        raise ValueError(f"run {args.run!r} not found; have {list(runs)}")

    written: list[Path] = []
    for run_name, csv_path in sorted(selected.items()):
        out = resample(csv_path, target=args.target, out_dir=args.out_dir)
        written.append(out)
        print(f"{run_name}: {out}  ({len(_read_csv(out)[1])} rows)")

    return written


if __name__ == "__main__":
    main()
