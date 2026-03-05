#!/usr/bin/env python3
"""
Generate side-by-side PPO vs SAC comparison GIF with labels.
Records both models and stitches frames horizontally with headers.
"""
import sys
from pathlib import Path
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import gymnasium as gym
import imageio
from PIL import Image, ImageDraw, ImageFont
from stable_baselines3 import PPO, SAC
import ironstride


def record_frames(model, algo_name, max_steps=500):
    """Record frames from a model rollout."""
    env = gym.make("IronStrideEnv-v0", render_mode="rgb_array", width=400, height=300)
    obs, _ = env.reset(seed=42)
    frames = []

    for i in range(max_steps):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        if i % 2 == 0:  # Every other frame
            frame = env.render()
            if frame is not None:
                frames.append(frame)
        if terminated:
            print(f"  {algo_name}: fell at step {i+1}", flush=True)
            break

    env.close()
    print(f"  {algo_name}: {len(frames)} frames recorded ({i+1} steps)", flush=True)
    return frames


def add_label(frame, label, color=(255, 255, 255), bg_color=None):
    """Add a text label at the top of a frame."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    # Create header bar
    bar_height = 35
    header = Image.new('RGB', (img.width, bar_height), bg_color or (30, 30, 50))
    header_draw = ImageDraw.Draw(header)

    # Try to load a nice font, fall back to default
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except:
        font = ImageFont.load_default()

    # Center the text
    bbox = header_draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    x = (img.width - text_w) // 2
    header_draw.text((x, 6), label, fill=color, font=font)

    # Combine header + frame
    combined = Image.new('RGB', (img.width, img.height + bar_height))
    combined.paste(header, (0, 0))
    combined.paste(img, (0, bar_height))
    return np.array(combined)


def main():
    output_dir = PROJECT_ROOT / "results"
    output_dir.mkdir(exist_ok=True)

    # Load models
    print("Loading PPO model...", flush=True)
    ppo_model = PPO.load(str(PROJECT_ROOT / "models" / "ppo_best" / "best_model"))

    print("Loading SAC model...", flush=True)
    sac_model = SAC.load(str(PROJECT_ROOT / "models" / "sac_best" / "best_model"))

    # Record frames
    print("Recording PPO...", flush=True)
    ppo_frames = record_frames(ppo_model, "PPO")

    print("Recording SAC...", flush=True)
    sac_frames = record_frames(sac_model, "SAC")

    # Match frame counts
    min_frames = min(len(ppo_frames), len(sac_frames))
    ppo_frames = ppo_frames[:min_frames]
    sac_frames = sac_frames[:min_frames]

    # Add labels and stitch side-by-side
    print(f"Stitching {min_frames} frames side-by-side...", flush=True)
    combined_frames = []
    for i, (pf, sf) in enumerate(zip(ppo_frames, sac_frames)):
        ppo_labeled = add_label(pf, "PPO — 2.5h Training | Reward: 9,735",
                               color=(77, 166, 255), bg_color=(20, 25, 45))
        sac_labeled = add_label(sf, "SAC — 17h Training | Reward: 8,479",
                               color=(255, 140, 66), bg_color=(20, 25, 45))

        # Add 4px divider
        divider = np.full((ppo_labeled.shape[0], 4, 3), (60, 60, 80), dtype=np.uint8)
        combined = np.concatenate([ppo_labeled, divider, sac_labeled], axis=1)
        combined_frames.append(combined)

    # Save GIF
    gif_path = output_dir / "ppo_vs_sac_comparison.gif"
    print(f"Writing GIF...", flush=True)
    imageio.mimsave(str(gif_path), combined_frames, fps=25, loop=0)
    size_mb = gif_path.stat().st_size / 1e6
    print(f"✓ Saved: {gif_path} ({size_mb:.1f} MB)", flush=True)

    # Optimize if too large
    if size_mb > 15:
        print("Optimizing (reducing frames)...", flush=True)
        reduced = combined_frames[::2]
        imageio.mimsave(str(gif_path), reduced, fps=13, loop=0)
        size_mb = gif_path.stat().st_size / 1e6
        print(f"✓ Optimized: {gif_path} ({size_mb:.1f} MB)", flush=True)

    # Also save individual SAC GIF
    sac_gif_path = output_dir / "sac_standing_demo.gif"
    sac_labeled_frames = [add_label(f, "SAC — Stable Standing", color=(255, 140, 66), bg_color=(20, 25, 45))
                          for f in sac_frames[::2]]
    imageio.mimsave(str(sac_gif_path), sac_labeled_frames, fps=13, loop=0)
    print(f"✓ Saved: {sac_gif_path}", flush=True)


if __name__ == "__main__":
    main()
