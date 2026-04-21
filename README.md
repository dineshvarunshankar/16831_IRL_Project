# LocoMimic

RL algorithms (PPO, SAC, TDMPC2) with [Unitree RL mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) for G1 humanoid motion imitation.

## Setup

```bash

#create conda environment
conda create -n locomimic python=3.10
conda activate locomimic
# Initialize submodules
git submodule update --init --recursive

# Install Unitree RL mjlab
cd unitree_rl_mjlab
pip install -e .
cd ..

# Install dependencies (redundancies from submodule to be cleaned - TODO)
pip install -r requirements.txt
```

## Project Structure

```
.
├── unitree_rl_mjlab/          # Submodule: Unitree's framework
├── algorithms/                 # Custom RL algorithms (TODO: implement)
│   ├── base.py                # Common interface
│   ├── ppo/                   # PPO
│   ├── sac/                   # SAC
│   └── tdmpc2/                # TDMPC2
├── configs/                    # Algorithm configs
├── scripts/                    # Training & evaluation scripts
│   ├── train/                 # Training scripts (TODO: implement)
│   ├── test/                  # Testing scripts (TODO: implement)
│   └── train(old-reference)/  # Old training code (reference only)
├── utils/                      # Utility scripts (convert, play, visualize)
├── data/lafan1_retargeted/    # Motion data (CSV)
├── logs/                       # Training logs
├── models/                     # Saved checkpoints
└── results/                    # Evaluation results
```

## Usage

### Convert Motion Data

```bash
python utils/convert_motions.py
```

### Train with Unitree Baseline

```bash
cd unitree_rl_mjlab
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file ../data/motions_npz/walk1.npz \
  --env.scene.num-envs 4096
```

### Train Custom Algorithms (TODO)

```bash
python scripts/train_sac.py --motion-file data/motions_npz/walk1.npz
python scripts/train_ppo.py --motion-file data/motions_npz/walk1.npz
python scripts/train_tdmpc2.py --motion-file data/motions_npz/walk1.npz
```

## Implementation Guide

All algorithms implement `BaseAlgorithm` in `algorithms/base.py`:

```python
class BaseAlgorithm(ABC):
    def select_action(self, obs: torch.Tensor, deterministic: bool) -> torch.Tensor:
        """obs: [num_envs, obs_dim] -> actions: [num_envs, act_dim]"""
        pass
    
    def update(self, obs, actions, rewards, next_obs, dones) -> Dict[str, float]:
        """Update from batched transitions"""
        pass
    
    def save(self, path: str):
        pass
    
    def load(self, path: str):
        pass
```

Integration with Unitree RL mjlab:

```python
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg

env_cfg = load_env_cfg("Unitree-G1-Tracking-No-State-Estimation")
env_cfg.commands["motion"].motion_file = "path/to/motion.npz"

env = ManagerBasedRlEnv(cfg=env_cfg, device="cuda")
env = RslRlVecEnvWrapper(env, clip_actions=True)

obs = env.reset()  # [num_envs, obs_dim]
actions = agent.select_action(obs)
obs_next, rewards, dones, infos = env.step(actions)
```
