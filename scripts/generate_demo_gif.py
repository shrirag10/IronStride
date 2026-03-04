#!/usr/bin/env python3
"""
Generate a GIF demo of the standing policy for README embedding.
Records frames from the best PPO model and converts to GIF.
"""
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import imageio
from stable_baselines3 import PPO
import ironstride

def main():
    model_path = PROJECT_ROOT / "models" / "ppo_best" / "best_model"
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)
    gif_path = output_dir / "standing_demo.gif"

    print("Loading model...", flush=True)
    model = PPO.load(str(model_path))

    env = gym.make("IronStrideEnv-v0", render_mode="rgb_array", width=480, height=360)
    obs, _ = env.reset(seed=42)

    frames = []
    max_steps = 500  # 10 seconds at 50Hz

    print(f"Recording {max_steps} steps...", flush=True)
    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        
        if i % 2 == 0:  # Every other frame to keep GIF small
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        
        if terminated:
            print(f"  Episode ended at step {i+1}", flush=True)
            break

    env.close()

    if len(frames) > 0:
        print(f"Writing GIF ({len(frames)} frames)...", flush=True)
        imageio.mimsave(str(gif_path), frames, fps=25, loop=0)
        size_mb = gif_path.stat().st_size / 1e6
        print(f"✓ Saved: {gif_path} ({size_mb:.1f} MB)", flush=True)
        
        # If GIF is too large (>10MB), reduce quality
        if size_mb > 10:
            print("GIF is large, creating optimized version...", flush=True)
            # Take every 3rd frame instead
            reduced_frames = frames[::3]
            imageio.mimsave(str(gif_path), reduced_frames, fps=17, loop=0)
            size_mb = gif_path.stat().st_size / 1e6
            print(f"✓ Optimized: {gif_path} ({size_mb:.1f} MB)", flush=True)
    else:
        print("ERROR: No frames captured!", flush=True)

if __name__ == "__main__":
    main()
