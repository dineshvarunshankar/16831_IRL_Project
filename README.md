# LocoMimic

This repository contains tools for simulating, visualizing, and training retargeted human motion on the Unitree G1 humanoid robot using MuJoCo and Gymnasium. 

## Setup

1. **Conda Environment**:
```bash
conda create -n roblearn python=3.10
conda activate roblearn
```

### 2. Install PyTorch
Install PyTorch manually based on your platform before anything else:
```bash
# Linux with NVIDIA GPU (CUDA 12.x)
pip install torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Mac M-series
pip install torch==2.1.0

# CPU only
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. External Dependencies
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
PYTHONPATH=. python -m train.train

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
    sac/              — SAC implementation from scratch
  configs/            — YAML hyperparameter configs
  train.py            — main training loop with wandb logging
scripts/
  random_baseline.py  — random agent evaluation
visualize.py          — interactive motion visualization
```
