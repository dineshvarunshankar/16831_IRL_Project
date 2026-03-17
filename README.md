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

### Reinforcement Learning Framework (`env/`)
The `env` directory contains a Gym environment for motion mimicry:
- **`motion_clip.py`**: Serves reference motion frames from datasets (e.g., LAFAN1).
- **`locomimic_env.py`**: A Gymnasium environment where the robot learns to match reference poses.

*(Note: Training scripts and baselines are located in `train/` and `scripts/`).*
