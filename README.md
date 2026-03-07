# LocoMimic

This repository contains tools for simulating, visualizing, and training retargeted human motion on the Unitree G1 humanoid robot using MuJoCo and Gymnasium. 

## Setup

1. **Conda Environment**:
```bash
conda create -n roblearn python=3.10
conda activate roblearn
```

2. **Install Dependencies**:
```bash
pip install -r requirements.txt
```
*(Requires `mujoco>=3.5.0` for full G1 robot compatibility).*

## Usage

### Motion Visualization
Visualize pre-retargeted motion data interactively:
```bash
# Auto-detect and run first available CSV
python visualize.py

# Run specific motion (e.g. at half speed)
python visualize.py --csv data/lafan1_retargeted/walk1_subject1.csv --speed 0.5

# List available motions
python visualize.py --list
```

### Reinforcement Learning Framework (`env/`)
The `env` directory contains a Gym environment for motion mimicry:
- **`motion_clip.py`**: Serves reference motion frames from datasets (e.g., LAFAN1).
- **`locomimic_env.py`**: A Gymnasium environment where the robot learns to match reference poses.

*(Note: Training scripts and baselines are located in `train/` and `scripts/`).*
