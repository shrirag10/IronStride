#!/usr/bin/env python3
"""
IronStride — Policy Evaluation & Video Recording.

Loads a trained PPO or SAC model, runs rollout episodes, and optionally
records video of the humanoid gait.

Usage:
    python scripts/evaluate.py --model-path models/ppo_best/best_model.zip --algo ppo
    python scripts/evaluate.py --model-path models/sac_best/best_model.zip --algo sac --record
    python scripts/evaluate.py --model-path models/ppo_final.zip --algo ppo --episodes 10
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
from stable_baselines3 import PPO, SAC

import ironstride  # registers the environment


ALGO_MAP = {"ppo": PPO, "sac": SAC}


def evaluate_policy(
    model_path: str,
    algo: str,
    n_episodes: int = 10,
    record: bool = False,
    video_dir: Optional[str] = None,
    deterministic: bool = True,
    seed: int = 42,
) -> dict:
    """
    Run rollout episodes and collect statistics.

    Returns dict with: mean_reward, std_reward, mean_length, std_length,
    success_rate, per_episode data.
    """
    AlgoClass = ALGO_MAP[algo]

    # Normalise model path: resolve relative paths & strip .zip
    # (SB3 auto-appends .zip internally)
    model_path = str(Path(model_path).resolve())
    if model_path.endswith(".zip"):
        model_path = model_path[:-4]

    model = AlgoClass.load(model_path)

    render_mode = "rgb_array" if record else None
    env = gym.make("IronStrideEnv-v0", render_mode=render_mode)

    if record:
        if video_dir is None:
            video_dir = str(PROJECT_ROOT / "videos")
        os.makedirs(video_dir, exist_ok=True)

        # Use imageio for recording
        import imageio

    all_rewards = []
    all_lengths = []
    all_max_heights = []
    all_max_velocities = []

    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        frames = []
        episode_reward = 0.0
        episode_length = 0
        max_height = 0.0
        max_velocity = 0.0

        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            episode_length += 1
            max_height = max(max_height, info.get("torso_height", 0.0))
            max_velocity = max(max_velocity, info.get("forward_velocity", 0.0))

            if record:
                frame = env.render()
                if frame is not None:
                    frames.append(frame)

            done = terminated or truncated

        all_rewards.append(episode_reward)
        all_lengths.append(episode_length)
        all_max_heights.append(max_height)
        all_max_velocities.append(max_velocity)

        # Save video for this episode
        if record and frames:
            video_path = os.path.join(video_dir, f"{algo}_episode_{ep:03d}.mp4")
            imageio.mimsave(video_path, frames, fps=50)
            print(f"  📹 Saved: {video_path}")

        print(
            f"  Episode {ep+1:3d}/{n_episodes} | "
            f"Reward: {episode_reward:8.1f} | "
            f"Length: {episode_length:5d} | "
            f"Max Height: {max_height:.3f}m | "
            f"Max Vel: {max_velocity:.3f} m/s"
        )

    env.close()

    results = {
        "algo": algo,
        "model_path": model_path,
        "n_episodes": n_episodes,
        "mean_reward": float(np.mean(all_rewards)),
        "std_reward": float(np.std(all_rewards)),
        "mean_length": float(np.mean(all_lengths)),
        "std_length": float(np.std(all_lengths)),
        "mean_max_height": float(np.mean(all_max_heights)),
        "mean_max_velocity": float(np.mean(all_max_velocities)),
        "per_episode_rewards": all_rewards,
        "per_episode_lengths": all_lengths,
    }

    return results


def print_results(results: dict) -> None:
    """Pretty-print evaluation results."""
    print("\n" + "═" * 60)
    print(f"  EVALUATION RESULTS — {results['algo'].upper()}")
    print("═" * 60)
    print(f"  Model:       {results['model_path']}")
    print(f"  Episodes:    {results['n_episodes']}")
    print(f"  Mean Reward: {results['mean_reward']:10.2f} ± {results['std_reward']:.2f}")
    print(f"  Mean Length: {results['mean_length']:10.1f} ± {results['std_length']:.1f}")
    print(f"  Max Height:  {results['mean_max_height']:10.3f} m")
    print(f"  Max Vel:     {results['mean_max_velocity']:10.3f} m/s")
    print("═" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="IronStride — Policy Evaluation"
    )
    parser.add_argument(
        "--model-path", type=str, required=True,
        help="Path to saved model (.zip)",
    )
    parser.add_argument(
        "--algo", type=str, required=True, choices=["ppo", "sac"],
        help="Algorithm used to train the model",
    )
    parser.add_argument(
        "--episodes", type=int, default=10,
        help="Number of evaluation episodes",
    )
    parser.add_argument(
        "--record", action="store_true",
        help="Record episodes as MP4 videos",
    )
    parser.add_argument(
        "--video-dir", type=str, default=None,
        help="Directory to save recorded videos",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for evaluation",
    )
    parser.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic (non-deterministic) policy",
    )
    args = parser.parse_args()

    results = evaluate_policy(
        model_path=args.model_path,
        algo=args.algo,
        n_episodes=args.episodes,
        record=args.record,
        video_dir=args.video_dir,
        deterministic=not args.stochastic,
        seed=args.seed,
    )
    print_results(results)


if __name__ == "__main__":
    main()
