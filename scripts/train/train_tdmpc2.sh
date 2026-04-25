#!/usr/bin/env bash
# Launcher for TDMPC2 on the mjlab whole-body-tracking env.
# Sets PYTHONPATH so the env adapter at agents/tdmpc2/tdmpc2/envs/mjlab.py can
# import `rl.sac_env_wrapper` (EmpiricalNormalization) and `src.tasks` (Unitree
# task registration), then invokes TDMPC2's native Hydra train.py.
#
# Usage:
#   ./scripts/train/train_tdmpc2.sh                                   # defaults from config.yaml
#   ./scripts/train/train_tdmpc2.sh steps=20000 enable_wandb=false    # override Hydra keys
#   ./scripts/train/train_tdmpc2.sh task=mjlab-Unitree-G1-Tracking-No-State-Estimation
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/unitree_rl_mjlab/src:${PYTHONPATH:-}"

cd "${REPO_ROOT}/agents/tdmpc2/tdmpc2"
exec python train.py "$@"
