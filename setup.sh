#!/usr/bin/env bash
# One-shot setup for LocoMimic on Linux + CUDA.
# Run from repo root:  bash setup.sh
#
# Submodules expected (see .gitmodules):
#   - unitree_rl_mjlab : mjlab fork providing the WBT env + PPO runner
#   - agents/tdmpc2    : TDMPC2 fork (vyvas33/tdmpc2) with our mjlab adapter
set -euo pipefail

echo "==> [1/5] Initializing git submodules (unitree_rl_mjlab, agents/tdmpc2)"
git submodule update --init --recursive

echo "==> [2/5] Installing unitree_rl_mjlab (pulls torch, numpy, tqdm, tyro, pyyaml)"
pip install -e ./unitree_rl_mjlab

echo "==> [3/5] Installing TDMPC2 runtime deps (state-obs only, no dm-control/kornia/mujoco)"
# We do NOT use TDMPC2's docker environment.yaml because it pins python=3.9 and
# torch nightly 2.6.0.dev — that would conflict with mjlab (needs python>=3.10 +
# its own torch). These are the minimum runtime deps for our mjlab adapter:
#   hydra-core/omegaconf : config system used by tdmpc2/train.py
#   gymnasium            : env interface (mjlab also uses it)
#   tensordict, torchrl  : episode-based replay buffer + SliceSampler
#   termcolor, pandas    : tdmpc2 logger
# (tqdm, wandb installed separately; mujoco/dm-control/kornia/imageio skipped —
# only needed for TDMPC2's bundled non-mjlab envs and rgb obs.)
pip install \
  "hydra-core==1.3.2" \
  "omegaconf>=2.3,<3" \
  "gymnasium>=0.29,<2" \
  "tensordict" \
  "torchrl" \
  "termcolor" \
  "pandas"

echo "==> [4/5] Installing wandb"
pip install wandb

echo "==> [5/5] Verifying CUDA + key imports"
python - <<'PY'
import torch
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device 0: {torch.cuda.get_device_name(0)}")
import hydra, omegaconf, gymnasium, tensordict, torchrl
print(f"hydra: {hydra.__version__}  omegaconf: {omegaconf.__version__}  "
      f"gymnasium: {gymnasium.__version__}  tensordict: {tensordict.__version__}  "
      f"torchrl: {torchrl.__version__}")
import mjlab
print(f"mjlab: {getattr(mjlab, '__version__', '?')}")
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
echo "     (and agents/tdmpc2/tdmpc2/config.yaml for TDMPC2)"
echo "  4. Launch training:"
echo "     python scripts/train/train_sac.py --num_envs 4096"
echo "     ./scripts/train/train_tdmpc2.sh wandb_entity=<your-entity>"
