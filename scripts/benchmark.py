#!/usr/bin/env python3
"""
IronStride — Benchmarking & Comparative Analytics.

Reads TensorBoard event files from PPO and SAC training runs and generates
publication-quality comparison plots for the portfolio README.

Usage:
    python scripts/benchmark.py
    python scripts/benchmark.py --log-dir logs --output-dir results
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def read_tensorboard_events(log_dir: str, tag: str = "rollout/ep_rew_mean"):
    """
    Read a scalar tag from TensorBoard event files.

    Returns (steps, values, wall_times) arrays.
    """
    try:
        from tensorboard.backend.event_processing.event_accumulator import (
            EventAccumulator,
        )
    except ImportError:
        print("ERROR: tensorboard is required. Install via: pip install tensorboard")
        sys.exit(1)

    # Find the event file directory (TensorBoard creates subdirs)
    log_path = Path(log_dir)
    event_dirs = []
    for p in log_path.rglob("events.out.tfevents.*"):
        event_dirs.append(str(p.parent))
    event_dirs = sorted(set(event_dirs))

    if not event_dirs:
        print(f"WARNING: No TensorBoard events found in {log_dir}")
        return np.array([]), np.array([]), np.array([])

    # Use the first event directory
    ea = EventAccumulator(event_dirs[0])
    ea.Reload()

    if tag not in ea.Tags().get("scalars", []):
        # Try to find a similar tag
        available = ea.Tags().get("scalars", [])
        print(f"WARNING: Tag '{tag}' not found. Available: {available}")
        return np.array([]), np.array([]), np.array([])

    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    values = np.array([e.value for e in events])
    wall_times = np.array([e.wall_time for e in events])

    return steps, values, wall_times


def smooth(values: np.ndarray, weight: float = 0.9) -> np.ndarray:
    """Exponential moving average smoothing."""
    smoothed = np.zeros_like(values)
    if len(values) == 0:
        return smoothed
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = weight * smoothed[i - 1] + (1 - weight) * values[i]
    return smoothed


def generate_comparison_plots(
    ppo_log_dir: str,
    sac_log_dir: str,
    output_dir: str,
) -> None:
    """Generate all comparison plots."""

    os.makedirs(output_dir, exist_ok=True)

    # ── Color scheme ────────────────────────────────────────────────
    PPO_COLOR = "#2196F3"   # Blue
    SAC_COLOR = "#FF5722"   # Deep Orange
    BG_COLOR = "#1a1a2e"
    GRID_COLOR = "#333355"
    TEXT_COLOR = "#e0e0e0"

    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": BG_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "text.color": TEXT_COLOR,
        "legend.facecolor": "#16213e",
        "legend.edgecolor": GRID_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.family": "sans-serif",
        "font.size": 12,
    })

    tags_to_plot = [
        ("rollout/ep_rew_mean", "Mean Episode Reward", "reward_comparison"),
        ("rollout/ep_len_mean", "Mean Episode Length", "episode_length_comparison"),
    ]

    for tag, ylabel, filename in tags_to_plot:
        ppo_steps, ppo_vals, ppo_wt = read_tensorboard_events(ppo_log_dir, tag)
        sac_steps, sac_vals, sac_wt = read_tensorboard_events(sac_log_dir, tag)

        if len(ppo_vals) == 0 and len(sac_vals) == 0:
            print(f"  Skipping {filename} — no data found")
            continue

        # ── Plot 1: Reward vs. Timesteps (Sample Efficiency) ────────
        fig, ax = plt.subplots(figsize=(12, 6))

        if len(ppo_vals) > 0:
            ax.plot(ppo_steps, ppo_vals, alpha=0.2, color=PPO_COLOR, linewidth=0.5)
            ax.plot(ppo_steps, smooth(ppo_vals), color=PPO_COLOR,
                    linewidth=2.5, label="PPO")
        if len(sac_vals) > 0:
            ax.plot(sac_steps, sac_vals, alpha=0.2, color=SAC_COLOR, linewidth=0.5)
            ax.plot(sac_steps, smooth(sac_vals), color=SAC_COLOR,
                    linewidth=2.5, label="SAC")

        ax.set_xlabel("Training Timesteps", fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_title(
            f"IronStride — {ylabel} (Sample Efficiency)",
            fontsize=16, fontweight="bold", pad=15,
        )
        ax.legend(fontsize=13, loc="lower right")
        ax.grid(True)
        ax.xaxis.set_major_formatter(
            FuncFormatter(lambda x, _: f"{x/1e6:.1f}M" if x >= 1e6 else f"{x/1e3:.0f}K")
        )

        filepath = os.path.join(output_dir, f"{filename}_vs_steps.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {filepath}")

    # ── Plot: Reward vs. Wall-Clock Time ────────────────────────────
    tag = "rollout/ep_rew_mean"
    ppo_steps, ppo_vals, ppo_wt = read_tensorboard_events(ppo_log_dir, tag)
    sac_steps, sac_vals, sac_wt = read_tensorboard_events(sac_log_dir, tag)

    if len(ppo_vals) > 0 or len(sac_vals) > 0:
        fig, ax = plt.subplots(figsize=(12, 6))

        if len(ppo_vals) > 0:
            ppo_time = (ppo_wt - ppo_wt[0]) / 60.0  # minutes
            ax.plot(ppo_time, ppo_vals, alpha=0.2, color=PPO_COLOR, linewidth=0.5)
            ax.plot(ppo_time, smooth(ppo_vals), color=PPO_COLOR,
                    linewidth=2.5, label="PPO")
        if len(sac_vals) > 0:
            sac_time = (sac_wt - sac_wt[0]) / 60.0  # minutes
            ax.plot(sac_time, sac_vals, alpha=0.2, color=SAC_COLOR, linewidth=0.5)
            ax.plot(sac_time, smooth(sac_vals), color=SAC_COLOR,
                    linewidth=2.5, label="SAC")

        ax.set_xlabel("Wall-Clock Time (minutes)", fontsize=14, fontweight="bold")
        ax.set_ylabel("Mean Episode Reward", fontsize=14, fontweight="bold")
        ax.set_title(
            "IronStride — Reward vs. Wall-Clock Time",
            fontsize=16, fontweight="bold", pad=15,
        )
        ax.legend(fontsize=13, loc="lower right")
        ax.grid(True)

        filepath = os.path.join(output_dir, "reward_vs_wallclock.png")
        fig.savefig(filepath, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✓ Saved: {filepath}")

    print("\n✓ Benchmarking plots generated successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="IronStride — Generate Benchmark Comparison Plots"
    )
    parser.add_argument(
        "--log-dir", type=str,
        default=str(PROJECT_ROOT / "logs"),
        help="Root TensorBoard log directory",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default=str(PROJECT_ROOT / "results"),
        help="Directory to save plots",
    )
    args = parser.parse_args()

    ppo_log = os.path.join(args.log_dir, "ppo")
    sac_log = os.path.join(args.log_dir, "sac")

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║       I R O N S T R I D E  —  Benchmark Analytics          ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  PPO logs: {ppo_log:<47}║")
    print(f"║  SAC logs: {sac_log:<47}║")
    print(f"║  Output:   {args.output_dir:<47}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    generate_comparison_plots(ppo_log, sac_log, args.output_dir)


if __name__ == "__main__":
    main()
