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

# Linux with NVIDIA GPU (CUDA 13.x)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130


# Mac M-series
pip install torch==2.1.0

# CPU only
pip install torch==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. External Dependencies
This project uses external repositories as submodules. To initialize and download them, run:
```bash
git submodule update --init --recursive
```

### 4. Motion Data (LAFAN1)
Because motion data files are large, the `data/` folder is ignored by git. Download the retargeted LAFAN1 kinematic data and place it in the correct directory before visualizing or training:
1. Download the `lafan1_retargeted` dataset from the [HuggingFace](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset).
2. Extract it into the root of this project so the path looks like: `data/lafan1_retargeted/g1/walk1_subject1.csv`

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

# PPO (16-core async vectorized run)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python train/train_ppo.py \
  --run-name ppo_walk1_16env \
  --n-envs 16 \
  --vector-env-type async \
  --vector-env-context spawn \
  --total-steps 12000000

# Resume an interrupted PPO run
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python train/train_ppo.py \
  --run-name ppo_walk1_16env_resume \
  --n-envs 16 \
  --vector-env-type async \
  --vector-env-context spawn \
  --resume-checkpoint models/ppo/ppo_step_4000000.pt \
  --total-steps 12000000

# PPO long run (uses train/configs/ppo_config.yaml defaults)
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
PYTHONPATH=. python train/train_ppo.py \
  --run-name ppo_final_30m \
  --total-steps 30000000

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
