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

Run these commands from the repo root (`LocoMimic/`).

### Convert Motion Data

```bash
python utils/convert_motions.py
```

### Train with Unitree PPO Baseline

```bash
cd unitree_rl_mjlab
python scripts/train.py Unitree-G1-Tracking-No-State-Estimation \
  --motion-file ../data/motions_npz/walk1.npz \
  --env.scene.num-envs 4096
cd ..
```
### Evaluate PPO
```bash
python scripts/play.py Unitree-G1-Tracking --motion_file=/home/dtc/dinesh_varun/16831/project/16831_IRL_Project/unitree_rl_mjlab/src/assets/motions/g1/walk1_subject1.npz --checkpoint_file=/home/dtc/dinesh_varun/16831/project/16831_IRL_Project/logs/rsl_rl/g1_tracking/2026-04-23_16-27-13_PPO/model_20500.pt
```
### Train SAC (Repo Baseline)

```bash
python -m scripts.train.train_sac \
  --config agents/configs/sac_config.yaml \
  --task Unitree-G1-Tracking \
  --num_envs 4096
```

### Train TD-MPC2

```bash
python -m scripts.train.train_tdmpc2 \
  --config agents/configs/tdmpc2_config.yaml --name stable512_walk1
```


```bash
# Example override for motion file and env count
python -m scripts.train.train_tdmpc2 \
  --config agents/configs/tdmpc2_config.yaml \
  --motion-file data/motions_npz/walk1.npz \
  --num_envs 128 \
  --disable-wandb
```

```bash
# Resume from checkpoint
python -m scripts.train.train_tdmpc2 \
  --config agents/configs/tdmpc2_config.yaml \
  --load models/<run_name>/final.pt
```

### Evaluate TD-MPC2 Checkpoint

```bash
python -m scripts.eval.eval_tdmpc2 \
  --checkpoint models/run3/ckpt_step_7450000.pt \
  --config agents/configs/tdmpc2_config.yaml \
  --num-envs 32 \
  --episodes 20
```

### Save Video

```bash
python -m scripts.eval.eval_tdmpc2 \
  --checkpoint models/run3/ckpt_step_7450000.pt \
  --config agents/configs/tdmpc2_config.yaml \
  --num-envs 1 \
  --episodes 1 \
  --video
```

