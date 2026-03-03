#!/usr/bin/env python3
"""
IronStride — Standalone PPO Training Script.

Usage:
    python scripts/train_ppo.py
    python scripts/train_ppo.py --total-timesteps 500000 --seed 42
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train_compare import load_config, train_ppo


def main():
    parser = argparse.ArgumentParser(description="IronStride — PPO Training")
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
    )
    parser.add_argument("--total-timesteps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--log-dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.total_timesteps is not None:
        cfg["training"]["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    seed = cfg["training"]["seed"]
    log_dir = str(PROJECT_ROOT / (args.log_dir or cfg["training"]["log_dir"]))
    model_dir = str(PROJECT_ROOT / cfg["training"]["model_dir"])

    train_ppo(cfg, log_dir, model_dir, seed)


if __name__ == "__main__":
    main()
