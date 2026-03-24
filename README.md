# IronStride

Benchmarking PPO vs. SAC for humanoid locomotion on the Unitree H1 in MuJoCo. Both algorithms achieve stable standing (full 20-second episodes), but PPO converges 7x faster in wall-clock time.

## Results

| Metric              | PPO                     | SAC                    |
|---------------------|-------------------------|------------------------|
| Time to convergence | **2.5 hours**           | 17 hours               |
| Training FPS        | **~1,130** (8 envs)     | ~162 (1 env)           |
| Episode success     | 20s stable standing     | 20s stable standing    |

## Tech Stack

- **Simulation**: MuJoCo (via `mujoco_menagerie` Unitree H1 model)
- **RL**: Stable-Baselines3 (PPO, SAC)
- **Env API**: Gymnasium
- **Python**: 3.10+

## Architecture

```
ironstride/
└── envs/
    └── ironstride_env.py   # Gymnasium env wrapping H1 MuJoCo model
scripts/
├── train_ppo.py
├── train_sac.py
├── train_compare.py        # Side-by-side benchmarking run
├── evaluate.py
├── benchmark.py
└── failure_analysis.py
configs/
└── default.yaml
results/
├── ppo_vs_sac_benchmark.png
├── ppo_vs_sac_comparison.gif
└── standing_demo.gif
```

## Observation Space

Base velocities, joint positions and velocities, projected gravity vector, previous actions. No visual input — proprioceptive only.

## Reward Structure

Velocity tracking + postural alignment + energy penalty + movement symmetry + survival bonus. Domain randomization applies mid-episode perturbations and randomizes friction and mass.

## Setup

```bash
git clone https://github.com/shrirag10/IronStride.git
cd IronStride
pip install -e .
pip install -r requirements.txt
```

### Train

```bash
python scripts/train_ppo.py
python scripts/train_sac.py
```

### Benchmark both

```bash
python scripts/train_compare.py
```

### Evaluate

```bash
python scripts/evaluate.py --algo ppo --checkpoint <path>
```

## Why PPO Wins Here

PPO's on-policy parallelism (8 simultaneous envs) dominates for unimodal standing tasks. SAC's replay buffer efficiency matters more in multi-modal scenarios requiring behavioral diversity. For locomotion tasks with a clear reward basin, PPO's monotonic improvement guarantee gives it a clean edge.

## License

MIT
