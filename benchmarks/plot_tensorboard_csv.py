"""Plot TensorBoard progress.csv files in a themed 3x5 grid.

Matches the Dirty Bot Natured Mkt Data visualizer style: Plotly + Kaleido PNGs,
shared colour themes, JetBrains Mono typography.

    python benchmarks/plot_tensorboard_csv.py
    python benchmarks/plot_tensorboard_csv.py --log-dir outputs/tensorboard --run PPO_n4_seed0
    python benchmarks/plot_tensorboard_csv.py --log-dir outputs/tensorboard --theme all --out-dir outputs/viz
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

FONT_FAMILY = "JetBrains Mono"

# 3x5 grid fits the 15 standard metrics: 2 rollout + 3 time + 10 train.
N_ROWS = 3
N_COLS = 5
MAX_PANELS = N_ROWS * N_COLS


@dataclass(frozen=True)
class Theme:
    name: str
    title: str
    bg: str
    paper: str
    font: str
    muted: str
    grid: str
    raw: str
    smooth: str
    accent: str
    font_family: str = FONT_FAMILY


_THEMES: dict[str, Theme] = {
    "ghibli": Theme(
        name="ghibli",
        title="Ghibli Meadow",
        bg="#E8F0E4",
        paper="#F4F7F0",
        font="#2F4A3C",
        muted="#6B8F71",
        grid="#C5D5C0",
        raw="#A3C2A3",
        smooth="#2F4A3C",
        accent="#5B8C5A",
    ),
    "retrowave": Theme(
        name="retrowave",
        title="Retrowave Grid",
        bg="#12061F",
        paper="#1A0B2E",
        font="#F2E6FF",
        muted="#B388FF",
        grid="#3D1F5C",
        raw="#7B2CBF",
        smooth="#00F0FF",
        accent="#FF6EC7",
    ),
    "noir": Theme(
        name="noir",
        title="Noir Tape",
        bg="#0B0B0B",
        paper="#141414",
        font="#EDEDED",
        muted="#8A8A8A",
        grid="#2A2A2A",
        raw="#4A4A4A",
        smooth="#E8E8E8",
        accent="#B01030",
    ),
    "nord": Theme(
        name="nord",
        title="Nord Frost",
        bg="#2E3440",
        paper="#3B4252",
        font="#ECEFF4",
        muted="#D8DEE9",
        grid="#4C566A",
        raw="#5E81AC",
        smooth="#88C0D0",
        accent="#A3BE8C",
    ),
    "terminal": Theme(
        name="terminal",
        title="Phosphor Terminal",
        bg="#020B05",
        paper="#04140A",
        font="#33FF66",
        muted="#1FA644",
        grid="#0D2818",
        raw="#0D5C1F",
        smooth="#33FF66",
        accent="#FF3355",
    ),
}


def theme(name: str) -> Theme:
    try:
        return _THEMES[name]
    except KeyError as e:
        raise ValueError(f"unknown theme {name!r}; have {list(_THEMES)}") from e


def _metric_order(col: str) -> tuple[int, int, str]:
    """Sort rollout/ and time/ first, then train/ alphabetically."""
    priority: dict[str, list[str]] = {
        "rollout/": ["rollout/ep_rew_mean", "rollout/ep_len_mean"],
        "time/": ["time/fps", "time/total_timesteps", "time/time_elapsed"],
    }
    for prefix, order in priority.items():
        if col.startswith(prefix):
            try:
                idx = order.index(col)
            except ValueError:
                idx = 999
            return (list(priority).index(prefix), idx, col)
    return (2, 0, col)


def _panel_title(col: str) -> str:
    """Return a compact, readable subplot title."""
    return col.replace("_", " ").replace("/", " / ")


def _moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Rolling mean; window=1 returns the original array."""
    w = max(1, window)
    if w == 1 or len(values) == 0:
        return values
    # Convolution with uniform weights gives the rolling mean.
    kernel = np.ones(w) / w
    return np.convolve(values, kernel, mode="same")


def figure(data: dict[str, np.ndarray], theme_name: str, window: int = 1) -> go.Figure:
    """Build a 3x5 Plotly figure from a progress.csv column dict."""
    th = theme(theme_name)

    metric_cols = [c for c in data if c != "timesteps"]
    metric_cols = sorted(metric_cols, key=_metric_order)[:MAX_PANELS]
    n_panels = len(metric_cols)

    fig = make_subplots(
        rows=N_ROWS,
        cols=N_COLS,
        subplot_titles=[_panel_title(c) for c in metric_cols],
        vertical_spacing=0.06,
        horizontal_spacing=0.05,
    )

    x = data["timesteps"]

    for idx, col in enumerate(metric_cols):
        row = idx // N_COLS + 1
        col_idx = idx % N_COLS + 1
        y = np.asarray(data[col], dtype=np.float64)
        y = np.where(np.isfinite(y), y, np.nan)
        y_smooth = _moving_average(y, window)

        fig.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="lines",
                line=dict(color=th.raw, width=1),
                opacity=0.35,
                name="raw",
                showlegend=False,
            ),
            row=row,
            col=col_idx,
        )
        fig.add_trace(
            go.Scatter(
                x=x,
                y=y_smooth,
                mode="lines",
                line=dict(color=th.smooth, width=2),
                name=f"ma{window}" if window > 1 else "raw",
                showlegend=False,
            ),
            row=row,
            col=col_idx,
        )

    ma_tag = f"  ·  ma{window}" if window > 1 else ""
    fig.update_layout(
        title=dict(
            text=f"{th.title}  ·  RL training diagnostics{ma_tag}",
            font=dict(size=20, color=th.font, family=th.font_family),
            x=0.02,
            xanchor="left",
        ),
        paper_bgcolor=th.paper,
        plot_bgcolor=th.bg,
        font=dict(color=th.font, family=th.font_family, size=11),
        showlegend=False,
        margin=dict(l=56, r=24, t=72, b=40),
        hovermode="x unified",
    )

    for ann in fig.layout.annotations:
        ann.font = dict(size=12, color=th.font, family=th.font_family)

    axis = dict(
        showgrid=True,
        gridcolor=th.grid,
        zeroline=False,
        showline=True,
        linecolor=th.grid,
        tickfont=dict(color=th.muted, family=th.font_family, size=9),
        title_font=dict(color=th.muted, family=th.font_family, size=10),
    )
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    for c in range(1, N_COLS + 1):
        fig.update_xaxes(title_text="timesteps", row=N_ROWS, col=c)

    return fig


def _find_runs(log_dir: Path) -> dict[str, Path]:
    """Map run name -> progress.csv path."""
    runs: dict[str, Path] = {}
    for csv_path in log_dir.rglob("progress.csv"):
        run_dir = csv_path.parent
        # run name is the directory, e.g. PPO_n4_seed0_1
        runs[run_dir.name] = csv_path
    return runs


def _write_png(fig: go.Figure, path: Path, width: int, height: int, scale: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_image(str(path), width=width, height=height, scale=scale)
    return path


def main(argv: list[str] | None = None) -> list[Path]:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--log-dir", type=Path, default=Path("outputs/tensorboard"))
    p.add_argument("--run", type=str, default=None, help="specific run directory name")
    p.add_argument(
        "--theme",
        nargs="+",
        default=["retrowave"],
        help=f"one of {list(_THEMES)} or all",
    )
    p.add_argument("--out-dir", type=Path, default=Path("outputs/viz"))
    p.add_argument("--window", type=int, default=5, help="moving-average window")
    p.add_argument("--width", type=int, default=1600)
    p.add_argument("--height", type=int, default=1000)
    p.add_argument("--scale", type=int, default=2)
    args = p.parse_args(argv)

    theme_names = list(_THEMES) if "all" in args.theme else args.theme
    for name in theme_names:
        theme(name)

    runs = _find_runs(args.log_dir)
    if not runs:
        raise FileNotFoundError(f"no progress.csv files found under {args.log_dir}")

    if args.run:
        if args.run not in runs:
            raise ValueError(f"run {args.run!r} not found; have {list(runs)}")
        selected = {args.run: runs[args.run]}
    else:
        selected = runs

    written: list[Path] = []
    for run_name, csv_path in sorted(selected.items()):
        with csv_path.open(newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows or "timesteps" not in rows[0]:
            raise ValueError(f"{csv_path} missing 'timesteps' column")

        def _to_float(cell: str) -> float:
            return float(cell) if cell not in ("", "nan", "inf", "-inf") else np.nan

        data = {
            col: np.array([_to_float(row.get(col, "")) for row in rows], dtype=np.float64)
            for col in reader.fieldnames or []
        }

        for name in theme_names:
            fig = figure(data, name, window=args.window)
            safe_run = re.sub(r"[^\w\-]+", "_", run_name).strip("_")
            png_path = args.out_dir / f"{safe_run}_{name}.png"
            _write_png(fig, png_path, args.width, args.height, args.scale)
            written.append(png_path)
            print(png_path)

    return written


if __name__ == "__main__":
    main()
