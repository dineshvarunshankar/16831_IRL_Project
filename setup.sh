#!/usr/bin/env bash
# One-shot setup for LocoMimic on Linux + CUDA.
# Run from repo root:  bash setup.sh
set -euo pipefail

echo "==> [1/4] Initializing git submodules"
git submodule update --init --recursive

echo "==> [2/4] Installing unitree_rl_mjlab (pulls torch, numpy, tqdm, tyro, pyyaml)"
pip install -e ./unitree_rl_mjlab

echo "==> [3/4] Installing wandb"
pip install wandb

echo "==> [4/4] Verifying CUDA"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device 0: {torch.cuda.get_device_name(0)}")
PY

echo
echo "Setup complete."
echo "Next steps:"
echo "  1. wandb login"
echo "  2. Convert a motion CSV -> NPZ:"
echo "     python unitree_rl_mjlab/scripts/csv_to_npz.py \\"
echo "       --robot g1 --input-file data/lafan1_retargeted/g1/walk1_subject1.csv \\"
echo "       --output-name walk1_subject1.npz --device cuda:0"
echo "  3. Update motion_path in agents/configs/sac_config.yaml"
echo "  4. Launch training:"
echo "     python scripts/train/train_sac.py --num_envs 4096"
