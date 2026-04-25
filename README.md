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

`setup.sh` runs: submodule init (`unitree_rl_mjlab`, `agents/tdmpc2`) → `pip install -e unitree_rl_mjlab` → TDMPC2 runtime deps (`hydra-core`, `omegaconf`, `gymnasium`, `tensordict`, `torchrl`, `termcolor`, `pandas`) → `pip install wandb` → CUDA + import sanity check.

## Project Structure

```
.
├── unitree_rl_mjlab/          # Submodule: mjlab framework
├── agents/
│   ├── base.py
│   ├── configs/
│   │   ├── sac_config.yaml
│   │   └── config_loader.py
│   ├── sac/
│   │   ├── actor.py
│   │   ├── critic.py
│   │   ├── replay_buffer.py
│   │   └── sac_agent.py
│   └── tdmpc2/                # Submodule: vyvas33/tdmpc2 fork (mjlab-integration branch)
│       └── tdmpc2/
│           ├── envs/mjlab.py  # single-env adapter for mjlab WBT
│           ├── config.yaml    # Hydra config (mjlab defaults)
│           └── train.py       # TDMPC2 entry point (Hydra)
├── rl/
│   └── sac_env_wrapper.py     # vec-env wrapper for mjlab (also exports EmpiricalNormalization, reused by TDMPC2 adapter)
├── scripts/train/
│   ├── train_sac.py           # SAC entry point
│   └── train_tdmpc2.sh        # TDMPC2 launcher (sets PYTHONPATH, runs Hydra train)
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

## Train TDMPC2

TDMPC2 is integrated as a submodule (`agents/tdmpc2`, fork: `vyvas33/tdmpc2` branch `mjlab-integration`). The single-env adapter at `agents/tdmpc2/tdmpc2/envs/mjlab.py` wraps mjlab's `ManagerBasedRlEnv` with `num_envs=1`, exposes the actor obs view (matches PPO/SAC for fair comparison), and reuses `EmpiricalNormalization` from `rl/sac_env_wrapper.py`.

Launch via the shell wrapper (handles `PYTHONPATH` so the adapter can import `rl.sac_env_wrapper` and `src.tasks`, then runs TDMPC2's native Hydra `train.py`):

### Smoke test (~10 min on A10G)
```bash
./scripts/train/train_tdmpc2.sh \
  steps=20000 \
  seed_steps=2000 \
  eval_freq=5000 \
  enable_wandb=false
```
Check: training loop runs past `seed_steps`, buffer fills, `agent.update()` executes, eval episodes complete, no shape errors.

### Full run
```bash
./scripts/train/train_tdmpc2.sh \
  exp_name=tdmpc2_g1_tracking \
  wandb_entity=<your-wandb-entity> \
  steps=1_000_000
```

### Hydra overrides (pass as `key=value` after the script name)
| Key | Default | Description |
|---|---|---|
| `task` | `mjlab-Unitree-G1-Tracking` | mjlab task id (prefix `mjlab-` is stripped before lookup) |
| `motion_path` | `data/lafan1_retargeted_npz/g1/walk1_subject1.npz` | path to NPZ motion (must exist) |
| `model_size` | `5` | TDMPC2 size preset: `1`, `5`, `19`, `48`, `317` (M params) |
| `episodic` | `true` | required — WBT terminates early on falls/anchor drift |
| `compile` | `false` | leave off until baseline trains; mjlab GPU sim + CUDA graphs is brittle |
| `save_video` | `false` | mjlab render API doesn't fit TDMPC2's video logger |
| `normalize_obs` | `true` | enable `EmpiricalNormalization` over actor obs |
| `mjlab_max_episode_steps` | `500` | cap; some env_cfgs set `episode_length_s=1e9` |
| `device` | `cuda` | torch device |
| `seed_steps` | (auto = `5 × episode_length`) | random-action warmup before training begins |
| `steps` | `10_000_000` | total env transitions |
| `eval_freq` | `50000` | env steps between eval episodes |

### Caveats
- **Single env only.** TDMPC2's native trainer assumes one env; we run mjlab with `num_envs=1`. Wall-clock is much slower than vectorized SAC/PPO at 4096 envs — expect overnight runs. TDMPC2's win shows up in **transitions-to-reward**, not wall-clock.
- **Logs land at** `agents/tdmpc2/tdmpc2/logs/mjlab-Unitree-G1-Tracking/<seed>/<exp_name>/` (Hydra default).
- **Comparing to SAC/PPO:** plot `episode_reward` vs. `env_step` (transitions), not vs. wall time.

### Updating the TDMPC2 submodule
If you change anything inside `agents/tdmpc2/`, commit + push from there first, then bump the parent's pin:
```bash
cd agents/tdmpc2 && git add -p && git commit -m "..." && git push
cd ../.. && git add agents/tdmpc2 && git commit -m "Bump tdmpc2 submodule" && git push
```

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
