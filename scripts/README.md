# Scripts

## eval.py

Evaluate a trained policy and print statistics. Optionally compare against a random baseline with plots.

```bash
# Basic evaluation
python -m scripts.eval models/tracking_v2/final.pt --algo sac

# PPO evaluation, deterministic actions
python -m scripts.eval models/ppo_v1/final.pt --algo ppo --deterministic

# Compare against random and save plots
python -m scripts.eval models/tracking_v2/final.pt --algo sac --compare-random --save-plots

# Custom episode count and plot name
python -m scripts.eval models/tracking_v2/final.pt --algo sac --episodes 50 --compare-random --save-plots --name sac_v2_vs_random
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `checkpoint` | (required) | Path to `.pt` checkpoint |
| `--algo` | `sac` | Algorithm: `sac` or `ppo` |
| `--config` | auto | Override config YAML path |
| `--episodes` | `100` | Number of eval episodes |
| `--deterministic` | off | Use mean action (no sampling) |
| `--render` | off | Render in MuJoCo viewer |
| `--compare-random` | off | Also evaluate random actions |
| `--save-plots` | off | Save plots to `results/plots/` (requires `--compare-random`) |
| `--name` | auto | Custom name for plot files |

---

## compare.py

Compare two models head-to-head. Use `random` as a model name for the random baseline.

```bash
# Default: sac_final vs random
python -m scripts.compare

# Compare two checkpoints
python -m scripts.compare --model_a models/tracking_v2/final.pt --model_b models/tracking_v2/ckpt_step_500000.pt

# Compare SAC vs random with rendering
python -m scripts.compare --model_a models/tracking_v2/final.pt --model_b random --episodes 50 --render
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `--model_a` | `models/sac_final.pt` | First model path (or `random`) |
| `--model_b` | `random` | Second model path (or `random`) |
| `--config` | `train/configs/sac_config.yaml` | Config YAML path |
| `--episodes` | `100` | Number of eval episodes per model |
| `--render` | off | Render in MuJoCo viewer |

---

## render_comparison.py

Render a side-by-side video of the trained policy vs the reference motion.

```bash
# SAC policy vs reference
python -m scripts.render_comparison models/tracking_v2/final.pt --algo sac

# PPO with 5 episodes
python -m scripts.render_comparison models/ppo_v1/final.pt --algo ppo --episodes 5

# Custom output path and resolution
python -m scripts.render_comparison models/tracking_v2/final.pt --algo sac --output results/videos/my_video.mp4 --width 800 --height 600
```

**Arguments:**
| Argument | Default | Description |
|---|---|---|
| `checkpoint` | (required) | Path to `.pt` checkpoint |
| `--algo` | `sac` | Algorithm: `sac` or `ppo` |
| `--config` | auto | Override config YAML path |
| `--episodes` | `1` | Number of episodes to render |
| `--output` | `results/videos/comparison_<algo>.mp4` | Output video path |
| `--width` | `640` | Frame width |
| `--height` | `480` | Frame height |

Videos are saved to `results/videos/` by default. Requires `opencv-python` (`pip install opencv-python`).
