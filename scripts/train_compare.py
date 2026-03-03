#!/usr/bin/env python3
"""
IronStride — Comparative Training Pipeline (PPO vs SAC).

Sequentially trains both PPO and SAC under identical environmental seeds,
logging to TensorBoard for direct algorithmic comparison.

Usage:
    python scripts/train_compare.py
    python scripts/train_compare.py --total-timesteps 500000 --seed 123
    python scripts/train_compare.py --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import yaml
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

import ironstride  # registers the environment


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def make_env(seed: int, rank: int = 0):
    """Create a factory function for initialising environments."""
    def _init():
        env = gym.make("IronStrideEnv-v0")
        env.reset(seed=seed + rank)
        return env
    return _init


def train_ppo(cfg: dict, log_dir: str, model_dir: str, seed: int) -> float:
    """
    Train PPO and return wall-clock training time (seconds).
    """
    ppo_cfg = cfg["ppo"]
    train_cfg = cfg["training"]

    print("\n" + "=" * 70)
    print("  TRAINING PPO")
    print("=" * 70)

    # Vectorised environments for PPO (on-policy benefits from parallelism)
    n_envs = train_cfg.get("n_envs", 4)
    env = make_vec_env(
        "IronStrideEnv-v0",
        n_envs=n_envs,
        seed=seed,
    )

    # Evaluation environment (single)
    eval_env = make_vec_env("IronStrideEnv-v0", n_envs=1, seed=seed + 1000)

    # Callbacks
    ppo_log = os.path.join(log_dir, "ppo")
    ppo_model_dir = os.path.join(model_dir, "ppo_best")
    os.makedirs(ppo_log, exist_ok=True)
    os.makedirs(ppo_model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=ppo_model_dir,
        log_path=ppo_log,
        eval_freq=max(train_cfg["eval_freq"] // n_envs, 1),
        n_eval_episodes=train_cfg["eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    # Build policy kwargs
    policy_kwargs = {}
    if "policy_kwargs" in ppo_cfg and ppo_cfg["policy_kwargs"]:
        if "net_arch" in ppo_cfg["policy_kwargs"]:
            policy_kwargs["net_arch"] = ppo_cfg["policy_kwargs"]["net_arch"]

    # Instantiate PPO
    model = PPO(
        ppo_cfg["policy"],
        env,
        learning_rate=ppo_cfg["learning_rate"],
        n_steps=ppo_cfg["n_steps"],
        batch_size=ppo_cfg["batch_size"],
        n_epochs=ppo_cfg["n_epochs"],
        gamma=ppo_cfg["gamma"],
        gae_lambda=ppo_cfg["gae_lambda"],
        clip_range=ppo_cfg["clip_range"],
        ent_coef=ppo_cfg["ent_coef"],
        vf_coef=ppo_cfg["vf_coef"],
        max_grad_norm=ppo_cfg["max_grad_norm"],
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        tensorboard_log=ppo_log,
        seed=seed,
        verbose=1,
        device="auto",
    )

    # Train
    start = time.time()
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=eval_callback,
        progress_bar=True,
    )
    wall_time = time.time() - start

    # Save final model
    final_path = os.path.join(model_dir, "ppo_final")
    model.save(final_path)
    print(f"\n✓ PPO training complete in {wall_time:.1f}s")
    print(f"  Final model: {final_path}.zip")
    print(f"  Best model:  {ppo_model_dir}/best_model.zip")

    env.close()
    eval_env.close()
    return wall_time


def train_sac(cfg: dict, log_dir: str, model_dir: str, seed: int) -> float:
    """
    Train SAC and return wall-clock training time (seconds).
    """
    sac_cfg = cfg["sac"]
    train_cfg = cfg["training"]

    print("\n" + "=" * 70)
    print("  TRAINING SAC")
    print("=" * 70)

    # SAC is off-policy → single environment is typical
    env = make_vec_env("IronStrideEnv-v0", n_envs=1, seed=seed)
    eval_env = make_vec_env("IronStrideEnv-v0", n_envs=1, seed=seed + 1000)

    # Callbacks
    sac_log = os.path.join(log_dir, "sac")
    sac_model_dir = os.path.join(model_dir, "sac_best")
    os.makedirs(sac_log, exist_ok=True)
    os.makedirs(sac_model_dir, exist_ok=True)

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=sac_model_dir,
        log_path=sac_log,
        eval_freq=train_cfg["eval_freq"],
        n_eval_episodes=train_cfg["eval_episodes"],
        deterministic=True,
        verbose=1,
    )

    # Build policy kwargs
    policy_kwargs = {}
    if "policy_kwargs" in sac_cfg and sac_cfg["policy_kwargs"]:
        if "net_arch" in sac_cfg["policy_kwargs"]:
            policy_kwargs["net_arch"] = sac_cfg["policy_kwargs"]["net_arch"]

    # Instantiate SAC
    model = SAC(
        sac_cfg["policy"],
        env,
        learning_rate=sac_cfg["learning_rate"],
        buffer_size=sac_cfg["buffer_size"],
        learning_starts=sac_cfg["learning_starts"],
        batch_size=sac_cfg["batch_size"],
        tau=sac_cfg["tau"],
        gamma=sac_cfg["gamma"],
        ent_coef=sac_cfg["ent_coef"],
        target_entropy=sac_cfg["target_entropy"],
        policy_kwargs=policy_kwargs if policy_kwargs else None,
        tensorboard_log=sac_log,
        seed=seed,
        verbose=1,
        device="auto",
    )

    # Train
    start = time.time()
    model.learn(
        total_timesteps=train_cfg["total_timesteps"],
        callback=eval_callback,
        progress_bar=True,
    )
    wall_time = time.time() - start

    # Save final model
    final_path = os.path.join(model_dir, "sac_final")
    model.save(final_path)
    print(f"\n✓ SAC training complete in {wall_time:.1f}s")
    print(f"  Final model: {final_path}.zip")
    print(f"  Best model:  {sac_model_dir}/best_model.zip")

    env.close()
    eval_env.close()
    return wall_time


def main():
    parser = argparse.ArgumentParser(
        description="IronStride — PPO vs SAC Comparative Training"
    )
    parser.add_argument(
        "--config", type=str,
        default=str(PROJECT_ROOT / "configs" / "default.yaml"),
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=None,
        help="Override total training timesteps",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Override random seed",
    )
    parser.add_argument(
        "--log-dir", type=str, default=None,
        help="Override TensorBoard log directory",
    )
    parser.add_argument(
        "--algo", type=str, default="both", choices=["ppo", "sac", "both"],
        help="Which algorithm(s) to train",
    )
    args = parser.parse_args()

    # Load config
    cfg = load_config(args.config)

    # Apply overrides
    if args.total_timesteps is not None:
        cfg["training"]["total_timesteps"] = args.total_timesteps
    if args.seed is not None:
        cfg["training"]["seed"] = args.seed

    seed = cfg["training"]["seed"]
    log_dir = args.log_dir or cfg["training"]["log_dir"]
    model_dir = cfg["training"]["model_dir"]

    # Resolve relative paths from project root
    log_dir = str(PROJECT_ROOT / log_dir)
    model_dir = str(PROJECT_ROOT / model_dir)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          I R O N S T R I D E  —  Training Pipeline         ║")
    print("╠══════════════════════════════════════════════════════════════╣")
    print(f"║  Timesteps : {cfg['training']['total_timesteps']:>12,}                           ║")
    print(f"║  Seed      : {seed:>12}                           ║")
    print(f"║  Algorithm : {args.algo:>12}                           ║")
    print(f"║  Log dir   : {log_dir:<45}║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results = {}

    if args.algo in ("ppo", "both"):
        results["ppo_wall_time"] = train_ppo(cfg, log_dir, model_dir, seed)

    if args.algo in ("sac", "both"):
        results["sac_wall_time"] = train_sac(cfg, log_dir, model_dir, seed)

    # Summary
    print("\n" + "═" * 70)
    print("  TRAINING SUMMARY")
    print("═" * 70)
    for key, val in results.items():
        print(f"  {key}: {val:.1f}s ({val/60:.1f} min)")
    print("═" * 70)
    print("\nTo view results: tensorboard --logdir", log_dir)


if __name__ == "__main__":
    main()
