# LocoMimic

This repository contains tools for simulating, visualizing, and training retargeted human motion on the Unitree G1 humanoid robot using MuJoCo and Gymnasium. 

## Setup

1. **Conda Environment**:
```bash
conda create -n roblearn python=3.10
conda activate roblearn
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. External Dependencies
These are not committed to the repo — clone them separately:
```bash
git clone https://github.com/google-deepmind/mujoco_menagerie
git clone https://github.com/unitreerobotics/unitree_mujoco
```

## Usage

### Motion Visualization
```bash
python visualize.py
python visualize.py --csv data/lafan1_retargeted/g1/walk1_subject1.csv --speed 0.5
python visualize.py --list
```

### Training
```bash
# SAC
PYTHONPATH=. python train/train_sac.py

# PPO
PYTHONPATH=. python train/train_ppo.py

# Random baseline
PYTHONPATH=. python scripts/random_baseline.py
```

### Project Structure
```
env/
  motion_clip.py      — loads reference motion from CSV
  locomimic_env.py    — Gymnasium environment for motion imitation
train/
  agents/
    sac/              — SAC implementation
    ppo/              — PPO implementation
    base_agent.py     — common agent interface
  configs/            — YAML hyperparameter configs and config loader
  train_sac.py        — SAC training loop with wandb
  train_ppo.py        — PPO rollout-based training loop with wandb
scripts/
  random_baseline.py  — random agent evaluation
visualize.py          — interactive motion visualization
```
