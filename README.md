# LocoMimic

## Setup (Linux, CUDA)

```bash
# 1. Install Miniconda (skip if already installed)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
bash /tmp/miniconda.sh -b -p $HOME/miniconda3
source $HOME/miniconda3/etc/profile.d/conda.sh
conda init bash && exec bash   # or restart shell

# 2. Clone (vyvas-mjlab branch only, with submodules)
git clone --branch vyvas-mjlab --single-branch --recurse-submodules \
  https://github.com/dineshvarunshankar/LocoMimic.git
cd LocoMimic

# 3. Create env
conda create -n mjlab python=3.10 -y
conda activate mjlab

# 4. One-shot install
bash setup.sh
wandb login
```

`setup.sh` runs: submodule init → `pip install -e unitree_rl_mjlab` → `pip install wandb` → CUDA sanity check.

## Project Structure

```
.
├── unitree_rl_mjlab/          # Submodule: mjlab framework
├── agents/
│   ├── base.py               
│   ├── configs/
│   │   ├── sac_config.yaml
│   │   └── config_loader.py
│   └── sac/
│       ├── actor.py
│       ├── critic.py
│       ├── replay_buffer.py  
│       └── sac_agent.py
├── rl/
│   └── sac_env_wrapper.py     # vec-env wrapper for mjlab
├── scripts/train/
│   └── train_sac.py           # entry point
├── data/lafan1_retargeted/    # LAFAN1 motion CSVs
├── logs/                      # wandb + stdout logs
└── models/                    # saved checkpoints
```

## Motion data: CSV -> NPZ

mjlab's motion-tracking env consumes `.npz` motion files. LAFAN1 ships as CSV, so convert once before training.

```bash

python unitree_rl_mjlab/scripts/csv_to_npz.py \
  --robot g1 \
  --input-file data/lafan1_retargeted/g1/walk1_subject1.csv \
  --output-name walk1_subject1.npz \
  --device cuda:0
```

Notes:
- `input_fps=30` (LAFAN1) and `output_fps=50` (mjlab G1 policy rate) are correct defaults — don't change unless you know why.
- The output path is hardcoded to `./src/assets/motions/g1/`. Either pass a plain filename (as above) and update `motion_path` in the config to match, or edit `output_dir` inside `csv_to_npz.py`.
- After conversion, set `motion_path` in `agents/configs/sac_config.yaml` to point at the generated `.npz`.

## Train SAC

Run with `-m` from repo root so `sys.path` includes the project (needed for `rl` / `agents` imports).

### Smoke test (5 minutes)
```bash
python -m scripts.train.train_sac \
  --name smoketest \
  --num_envs 256 \
  --iter 500
```

Check: stdout prints log lines, wandb run opens, no NaNs in `loss/critic` or `q/max`, `buffer/size` grows, `perf/env_steps_per_sec` > 0.

### Full run (vanilla SAC)
```bash
python -m scripts.train.train_sac \
  --name sac_vanilla \
  --num_envs 4096
```

### Full run (FastSAC-style: LayerNorm + mean-of-Qs)
```bash
python -m scripts.train.train_sac \
  --name sac_fast \
  --num_envs 4096 \
  --fast-sac
```

### CLI flags
| Flag | Default | Description |
|---|---|---|
| `--name` | timestamp | run name (also wandb name) |
| `--config` | `agents/configs/sac_config.yaml` | config file |
| `--num_envs` | 4096 | parallel envs |
| `--task` | `Unitree-G1-Tracking` | mjlab task id |
| `--load` | None | checkpoint path to resume from |
| `--fast-sac` | off | enables `use_layer_norm=True` + `use_mean_q=True` |
| `--iter` | (config value) | override `num_learning_iterations` (for smoke tests) |

### Key config values (`agents/configs/sac_config.yaml`)
- `motion_path` — path to the `.npz` motion file (must exist before launch).
- `device: cuda:0` — set GPU.
- `num_learning_iterations: 25000` — each iter = one `env.step()` across all envs. Total env transitions = `num_learning_iterations × num_envs`.
- `buffer_size: 1024` — **per-env**; total capacity = `buffer_size × num_envs`.
- `gradient_steps: 8` — SGD updates per rollout tick.
- `learning_starts: 10` — iterations of random-action warmup.
- `gamma: 0.97`, `tau: 0.005`, `lr: 3e-4`, `weight_decay: 1e-3`.
- `alpha_init: 0.001`, `target_entropy_ratio: 0.5` (→ `target_entropy = -0.5 * act_dim`).
- `use_layer_norm`, `use_mean_q` — flipped on together by `--fast-sac`.

## Monitoring

Watch these wandb panels during training:
- `episode/avg_return` — primary success signal.
- `q/max`, `q/mean` — divergence canary; if `q/max` grows unboundedly, the critic is overestimating → likely loss of run.
- `loss/critic`, `loss/actor`, `loss/alpha` — should all trend downward or stabilize.
- `policy/entropy` vs `policy/target_entropy` — alpha autotune should close the gap over time.
- `alpha/value` — if stuck at `alpha_init` for thousands of iters, entropy pressure is not triggering; consider raising `alpha_init` or lowering `target_entropy_ratio`.
- `grad/critic_norm`, `grad/actor_norm` — spikes = instability.
- `perf/env_steps_per_sec` — throughput baseline; sudden drops suggest stalls.
- `buffer/size` — should saturate at `buffer_size × num_envs`.

## Checkpoints

Saved under `models/<run_name>/`:
- `ckpt_step_<N>.pt` at 25%, 50%, 75% of `num_learning_iterations` (configurable via `ckpt_fractions`).
- `final.pt` at end of run.

Resume with:
```bash
python -m scripts.train.train_sac --load models/<run_name>/ckpt_step_<N>.pt --name <new_name>
```

## Evaluate (macOS or Linux)

After a run, pull the checkpoint locally (see rsync section below), then:

```bash
# macOS (CPU, MuJoCo native viewer)
python -m scripts.eval.eval_sac \
  --ckpt models/<run_name>/final.pt \
  --device cpu

# Linux (GPU, headless or with viewer)
python -m scripts.eval.eval_sac \
  --ckpt models/<run_name>/final.pt \
  --device cuda:0
```

Flags:
| Flag | Default | Description |
|---|---|---|
| `--ckpt` | (required) | checkpoint `.pt` file (also loads matching `.rms.pt` sidecar if present) |
| `--config` | `agents/configs/sac_config.yaml` | same config used for training |
| `--task` | `Unitree-G1-Tracking` | mjlab task id |
| `--device` | `cpu` | `cpu` on Mac, `cuda:0` on Linux |
| `--num_envs` | 1 | single env for visualization |
| `--steps` | 2000 | total env steps to run |
| `--no-render` | off | disable viewer (for scripted return evaluation) |

macOS notes:
- Requires `pip install mujoco` (CPU build) — the CUDA-backed `mjlab` still runs on CPU via MuJoCo's native backend.
- If the viewer fails to open, try `export MUJOCO_GL=glfw` (or `egl` for offscreen).
- Expect low FPS — single-env CPU rollout is only for inspection, not training.

## Sync from EC2 (pull artifacts locally)

```bash
# Set once
EC2_IP=<public-ip>
KEY=~/.ssh/locomimic-key.pem
REMOTE=/home/ubuntu/LocoMimic

# Pull models, logs, wandb (non-destructive; won't delete local files)
rsync -avz --progress -e "ssh -i $KEY" ubuntu@$EC2_IP:$REMOTE/models/ ./models/
rsync -avz --progress -e "ssh -i $KEY" ubuntu@$EC2_IP:$REMOTE/logs/   ./logs/
rsync -avz --progress -e "ssh -i $KEY" ubuntu@$EC2_IP:$REMOTE/wandb/  ./wandb/
```

First connection to a new IP: add `-o StrictHostKeyChecking=accept-new` to the `-e` string.

## Troubleshooting

- **`FileNotFoundError: .../src/assets/motions/g1/...npz`** during conversion → the `--output-name` contained subdirs; pass just a filename.
- **`ModuleNotFoundError: tyro`** → `pip install tyro`.
- **CUDA not available** → check `torch.cuda.is_available()`; reinstall torch with matching CUDA version.
- **`src.tasks` import error** in `csv_to_npz.py` → the script expects a `src/` layout; add `src/` → `unitree_rl_mjlab/src/` symlink or run from the submodule root.
- **OOM at `num_envs=4096`** → drop `num_envs` or `critic_hidden_dim`/`actor_hidden_dim`.
