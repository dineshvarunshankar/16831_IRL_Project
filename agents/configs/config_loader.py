from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import yaml


@dataclass
class SACConfig:
    motion_path: str
    actor_hidden_dim: list[int]
    critic_hidden_dim: list[int]
    lr: float
    weight_decay: float
    gamma: float
    tau: float
    batch_size: int
    buffer_size: int
    total_steps: int
    learning_starts: int
    gradient_steps: int
    log_freq: int
    save_freq: int
    device: str = "cpu"
    num_envs: int = 1


@dataclass
class TDMPC2Config:
    # Environment/task.
    motion_path: str
    task: str = "Unitree-G1-Tracking"
    num_envs: int = 512
    seed: int = 42
    device: str = "cuda"

    # Training.
    total_steps: int = 10_000_000
    step_unit: str = "env_frames"
    log_freq: int = 1000
    save_freq: int = 50_000
    eval_freq: int = 100_000
    eval_episodes: int = 5

    # Optimizer.
    lr: float = 3e-4
    encoder_lr: float = 1e-4
    grad_clip_norm: float = 20.0

    # Replay and update schedule.
    buffer_size: int = 1_000_000
    batch_size: int = 256
    seed_steps_floor: int = 1000
    seed_steps: int | None = None
    updates_per_collect: int = 16
    # Deprecated: kept only for backward compatibility.
    utd_start: int = 1
    utd_end: int = 8
    utd_warmup_steps: int = 100_000

    # RL.
    gamma: float = 0.99
    temporal_coef: float = 0.5
    tau: float = 0.01
    episodic: bool = True

    # Losses.
    consistency_coef: float = 20.0
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0
    entropy_coef: float = 1e-4

    # Architecture.
    encoder_dim: int = 256
    mlp_dim: int = 512
    latent_dim: int = 512
    simnorm_dim: int = 8
    simnorm_temp: float = 1.0
    num_q: int = 5
    q_dropout: float = 0.01
    num_bins: int = 101
    value_support: float = 10.0
    log_std_min: float = -10.0
    log_std_max: float = 2.0

    # Planning.
    horizon: int = 3
    mpc_iterations: int = 6
    mpc_action_dim_iters_threshold: int = 20
    mpc_action_dim_extra_iterations: int = 2
    mpc_samples: int = 512
    mpc_elites: int = 64
    mpc_policy_samples: int = 24
    mpc_min_std: float = 0.05
    mpc_max_std: float = 2.0
    mpc_temperature: float = 0.5

    # Hybrid acting.
    use_hybrid_acting: bool = True
    mpc_train_envs: int = 128
    mpc_eval: bool = True

    # Observation splicing.
    exogenous_terms: tuple[str, ...] = (
        "command",
        "motion_anchor_pos_b",
        "motion_anchor_ori_b",
    )

    # Logging.
    wandb_project: str = "locomimic"
    wandb_entity: str | None = None
    use_wandb: bool = True


def _auto_device(cfg: dict[str, Any]) -> dict[str, Any]:
    import torch

    out = dict(cfg)
    if "device" not in out:
        out["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    return out


def _load_yaml(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config file must contain a mapping: {path}")
    return cfg


def load_config(path: str) -> SACConfig:
    cfg = _auto_device(_load_yaml(path))
    return SACConfig(**cfg)


def load_tdmpc2_config(path: str) -> TDMPC2Config:
    raw = _load_yaml(path)

    # Backward compatibility with older nested templates.
    if "algorithm" in raw or "environment" in raw or "training" in raw:
        merged: dict[str, Any] = {}
        for key in ("algorithm", "environment", "training"):
            part = raw.get(key, {})
            if isinstance(part, dict):
                merged.update(part)
        raw = merged

    # Legacy aliases.
    if "motion_file" in raw and "motion_path" not in raw:
        raw["motion_path"] = raw["motion_file"]
    if "learning_rate" in raw and "lr" not in raw:
        raw["lr"] = raw["learning_rate"]
    if "num_samples" in raw and "mpc_samples" not in raw:
        raw["mpc_samples"] = raw["num_samples"]
    if "num_elites" in raw and "mpc_elites" not in raw:
        raw["mpc_elites"] = raw["num_elites"]
    if "num_iterations" in raw and "mpc_iterations" not in raw:
        raw["mpc_iterations"] = raw["num_iterations"]
    if "temperature" in raw and "mpc_temperature" not in raw:
        raw["mpc_temperature"] = raw["temperature"]
    if "q_target_ema" in raw and "tau" not in raw:
        raw["tau"] = 1.0 - float(raw["q_target_ema"])
    if "support_range" in raw and "value_support" not in raw:
        raw["value_support"] = raw["support_range"]

    cfg = _auto_device(raw)
    return TDMPC2Config(**cfg)
