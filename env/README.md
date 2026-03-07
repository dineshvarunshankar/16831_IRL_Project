# env/

This folder contains the core environment code for motion mimic.

---

## Files

### `motion_clip.py`
Loads a pre-retargeted G1 motion CSV and serves per-frame robot state.
The environment uses this as the reference the robot tries to imitate.

### `locomimic_env.py`
A Gymnasium environment where the robot learns to imitate a reference
motion by maximizing a reward that measures how closely its pose matches
the reference at each timestep. 

---

## How They Fit Together

```
CSV file
   ↓
MotionClip          — loads and serves reference motion frames
   ↓
MotionMimicEnv      — simulation + reward + RL interface
   ↓
PPO / SAC           — learns to maximize reward
```

---

## Data Format

Motion data comes from the LAFAN1 Retargeting Dataset:
- HuggingFace: `lvhaidong/LAFAN1_Retargeting_Dataset`
- Robot: Unitree G1
- Format: CSV, 36 columns, 30 FPS, no header row
- Columns: `root_xyz (3) | root_quat_xyzw (4) | joint_angles (29)`

---

## Robot Model

Unitree G1 from MuJoCo Menagerie:
- XML: `mujoco_menagerie/unitree_g1/scene.xml`
- Actuators: 29
- qpos dim: 36 (7 root + 29 joints)
- qvel dim: 35 (6 root + 29 joints)

---

## What Is Not Done Yet

- `step()` in `MotionMimicEnv` — needs to be implemented
- `_compute_reward()` — needs to be implemented
- PD gains may need tuning per joint group
- Observation space design may change based on results