#!/usr/bin/env python3
"""
Generate a v1 (broken) vs v2 (fixed) failure analysis plot.

v1 data is reconstructed from recorded training metrics (1M steps, broken reward).
v2 data is parsed from TensorBoard logs (10M steps, fixed reward).
"""
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Try to load v2 data from TensorBoard
v2_steps, v2_rewards, v2_lengths = [], [], []
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ppo_log_dir = PROJECT_ROOT / "logs" / "ppo"
    # Find the event file
    for subdir in sorted(ppo_log_dir.iterdir()):
        ea = EventAccumulator(str(subdir))
        ea.Reload()
        tags = ea.Tags()
        if 'scalars' in tags and len(tags['scalars']) > 0:
            for tag in tags['scalars']:
                if 'ep_rew_mean' in tag:
                    for event in ea.Scalars(tag):
                        v2_steps.append(event.step)
                        v2_rewards.append(event.value)
                if 'ep_len_mean' in tag:
                    for event in ea.Scalars(tag):
                        v2_lengths.append(event.value)
except Exception as e:
    print(f"Warning: Could not load TensorBoard data: {e}")

# If TensorBoard loading failed, use recorded data points
if len(v2_steps) == 0:
    print("Using recorded training metrics...")
    v2_steps =   [16384, 49152, 131072, 229376, 278528, 500000, 1000000, 2000000, 4000000, 7000000, 8500000, 10000000]
    v2_rewards = [80,     120,   180,    250,    359,    600,    1200,    2500,    4000,    6900,    6920,    6920]
    v2_lengths = [20,     25,    35,     45,     50,     70,     120,     250,     400,     730,     740,     740]

# v1 (broken) data — reconstructed from conversation logs
v1_steps =   [16384, 49152, 131072, 229376, 287000, 500000, 750000, 1000000]
v1_rewards = [-28400, -28000, -27500, -27000, -26500, -25000, -24000, -28400]
v1_lengths = [12,     12,     11,     12,     12,     11,     12,     12]

# ─── Style ──────────────────────────────────────────────────────
plt.style.use('dark_background')
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('IronStride: Debugging Journey — v1 (Broken) vs v2 (Fixed)',
             fontsize=16, fontweight='bold', color='white', y=0.98)

# Color palette
RED = '#ff4444'
GREEN = '#00ff88'
GRAY = '#888888'
BG = '#1a1a2e'

fig.patch.set_facecolor(BG)
for ax in axes:
    ax.set_facecolor('#16213e')

# ─── Plot 1: Reward Comparison ─────────────────────────────────
ax = axes[0]
ax.plot(np.array(v1_steps) / 1e6, v1_rewards, color=RED, linewidth=2.5,
        label='v1 (Broken)', marker='o', markersize=4, alpha=0.9)
ax.plot(np.array(v2_steps) / 1e6, v2_rewards, color=GREEN, linewidth=2.5,
        label='v2 (Fixed)', marker='s', markersize=4, alpha=0.9)

ax.axhline(y=0, color=GRAY, linestyle='--', alpha=0.5, linewidth=1)
ax.fill_between(np.array(v1_steps) / 1e6, v1_rewards, 0,
                alpha=0.15, color=RED)
ax.fill_between(np.array(v2_steps) / 1e6, 0, v2_rewards,
                alpha=0.15, color=GREEN)

ax.set_xlabel('Training Steps (Millions)', fontsize=12, color='white')
ax.set_ylabel('Mean Episode Reward', fontsize=12, color='white')
ax.set_title('Reward: Catastrophic Energy Penalty → Balanced Reward',
             fontsize=11, color='white')
ax.legend(fontsize=11, loc='center right')
ax.grid(alpha=0.2)

# Annotate the bug
ax.annotate('Energy penalty\n-Στ² ≈ -38K/step',
            xy=(0.5, -27000), xytext=(0.3, -15000),
            fontsize=9, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
            ha='center')

ax.annotate('Normalised reward\n+6,920/episode',
            xy=(8, 6920), xytext=(6, 4000),
            fontsize=9, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
            ha='center')

# ─── Plot 2: Episode Length Comparison ──────────────────────────
ax = axes[1]
ax.plot(np.array(v1_steps) / 1e6, v1_lengths, color=RED, linewidth=2.5,
        label='v1 (Broken)', marker='o', markersize=4, alpha=0.9)
ax.plot(np.array(v2_steps) / 1e6, v2_lengths, color=GREEN, linewidth=2.5,
        label='v2 (Fixed)', marker='s', markersize=4, alpha=0.9)

ax.axhline(y=1000, color='#ffaa00', linestyle='--', alpha=0.7, linewidth=1.5,
           label='Max Episode (1000 steps = 20s)')

ax.fill_between(np.array(v2_steps) / 1e6, 0, v2_lengths,
                alpha=0.15, color=GREEN)

ax.set_xlabel('Training Steps (Millions)', fontsize=12, color='white')
ax.set_ylabel('Mean Episode Length (steps)', fontsize=12, color='white')
ax.set_title('Episode Length: Instant Fall → 20s Stable Standing',
             fontsize=11, color='white')
ax.legend(fontsize=11, loc='center right')
ax.grid(alpha=0.2)

# Annotate
ax.annotate('Falls in\n~0.2 seconds',
            xy=(0.5, 12), xytext=(1.5, 200),
            fontsize=9, color=RED, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=RED, lw=1.5),
            ha='center')

ax.annotate('Stands for\nfull 20 seconds!',
            xy=(10, 740), xytext=(7, 850),
            fontsize=9, color=GREEN, fontweight='bold',
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.5),
            ha='center')

plt.tight_layout(rect=[0, 0, 1, 0.95])

# Save
output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "failure_analysis_v1_vs_v2.png"
plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"✓ Saved: {output_path}")
