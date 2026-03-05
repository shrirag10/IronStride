#!/usr/bin/env python3
"""
Generate PPO vs SAC training curve comparison plot.
Uses recorded metrics from both 10M training runs.
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── Recorded training metrics ──────────────────────────────────
# PPO 10M (2.5 hours, 8 envs, ~1130 FPS)
ppo_steps   = [16384, 50000, 100000, 229376, 278528, 500000, 1000000,
               2000000, 4000000, 7000000, 8500000, 9800000, 10000000]
ppo_rewards = [80, 120, 180, 250, 359, 600, 1200,
               2500, 4300, 6940, 6920, 6970, 6920]
ppo_lengths = [20, 25, 35, 45, 50, 70, 120,
               250, 460, 733, 740, 686, 740]
ppo_hours   = [0, 0.02, 0.04, 0.06, 0.07, 0.13, 0.26,
               0.5, 1.0, 1.7, 2.1, 2.4, 2.46]

# SAC 10M (17 hours, 1 env, ~162 FPS)
sac_steps   = [44448, 80962, 304985, 500000, 1000000, 2000000,
               5327601, 7028825, 7824887, 8427456, 9402677, 9828276, 9934290]
sac_rewards = [390, 410, 525, 700, 950, 1400,
               1720, 2970, 2740, 1970, 3450, 3880, 3550]
sac_lengths = [52, 54, 80, 100, 140, 200,
               232, 391, 376, 275, 460, 516, 492]
sac_hours   = [0.01, 0.04, 0.5, 0.8, 1.6, 3.2,
               9.0, 11.9, 13.3, 14.3, 16.0, 16.8, 17.0]

# ── Style ──────────────────────────────────────────────────────
plt.style.use('dark_background')
fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle('IronStride: PPO vs SAC — 10M Steps Benchmark',
             fontsize=16, fontweight='bold', color='white', y=0.98)

BLUE = '#4da6ff'
ORANGE = '#ff8c42'
BG = '#1a1a2e'

fig.patch.set_facecolor(BG)
for row in axes:
    for ax in row:
        ax.set_facecolor('#16213e')

# ── Plot 1: Reward vs Steps ───────────────────────────────────
ax = axes[0][0]
ax.plot(np.array(ppo_steps)/1e6, ppo_rewards, color=BLUE, linewidth=2.5,
        label='PPO', marker='o', markersize=4)
ax.plot(np.array(sac_steps)/1e6, sac_rewards, color=ORANGE, linewidth=2.5,
        label='SAC', marker='s', markersize=4)
ax.set_xlabel('Training Steps (Millions)', fontsize=11)
ax.set_ylabel('Mean Episode Reward', fontsize=11)
ax.set_title('Reward vs Training Steps', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.2)

# ── Plot 2: Reward vs Wall Clock ──────────────────────────────
ax = axes[0][1]
ax.plot(ppo_hours, ppo_rewards, color=BLUE, linewidth=2.5,
        label='PPO (2.5h total)', marker='o', markersize=4)
ax.plot(sac_hours, sac_rewards, color=ORANGE, linewidth=2.5,
        label='SAC (17h total)', marker='s', markersize=4)
ax.set_xlabel('Wall Clock Time (Hours)', fontsize=11)
ax.set_ylabel('Mean Episode Reward', fontsize=11)
ax.set_title('Reward vs Wall Clock — PPO 7× Faster', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.2)
ax.axvline(x=2.46, color=BLUE, linestyle='--', alpha=0.5, linewidth=1.5)
ax.annotate('PPO done', xy=(2.46, 6500), fontsize=9, color=BLUE, ha='left')

# ── Plot 3: Episode Length vs Steps ───────────────────────────
ax = axes[1][0]
ax.plot(np.array(ppo_steps)/1e6, ppo_lengths, color=BLUE, linewidth=2.5,
        label='PPO', marker='o', markersize=4)
ax.plot(np.array(sac_steps)/1e6, sac_lengths, color=ORANGE, linewidth=2.5,
        label='SAC', marker='s', markersize=4)
ax.axhline(y=1000, color='#ffaa00', linestyle='--', alpha=0.7, linewidth=1.5,
           label='Max (1000 steps = 20s)')
ax.set_xlabel('Training Steps (Millions)', fontsize=11)
ax.set_ylabel('Mean Episode Length', fontsize=11)
ax.set_title('Episode Length — Standing Duration', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(alpha=0.2)

# ── Plot 4: Summary Bar Chart ────────────────────────────────
ax = axes[1][1]
categories = ['Eval Reward', 'Training\nHours', 'FPS', 'Eval Episode\nLength']
ppo_vals = [9735.68, 2.46, 1130, 1000]
sac_vals = [8479.67, 17.0, 162, 1000]

x = np.arange(len(categories))
width = 0.35
bars1 = ax.bar(x - width/2, ppo_vals, width, label='PPO', color=BLUE, alpha=0.85)
bars2 = ax.bar(x + width/2, sac_vals, width, label='SAC', color=ORANGE, alpha=0.85)

ax.set_xticks(x)
ax.set_xticklabels(categories, fontsize=10)
ax.set_ylabel('Value', fontsize=11)
ax.set_title('Final Evaluation Comparison', fontsize=12, fontweight='bold')
ax.legend(fontsize=11)
ax.set_yscale('log')
ax.grid(alpha=0.2, axis='y')

# Add value labels on bars
for bar in bars1:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h*1.1,
            f'{h:,.0f}', ha='center', va='bottom', fontsize=8, color=BLUE)
for bar in bars2:
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., h*1.1,
            f'{h:,.0f}', ha='center', va='bottom', fontsize=8, color=ORANGE)

plt.tight_layout(rect=[0, 0, 1, 0.95])

output_dir = PROJECT_ROOT / "results"
output_dir.mkdir(exist_ok=True)
output_path = output_dir / "ppo_vs_sac_benchmark.png"
plt.savefig(str(output_path), dpi=150, bbox_inches='tight',
            facecolor=fig.get_facecolor())
plt.close()
print(f"✓ Saved: {output_path}")
