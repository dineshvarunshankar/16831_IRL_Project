import yaml
from dataclasses import dataclass
import torch

@dataclass
class SACConfig:
    motion_path       : str
    actor_hidden_dim  : int
    critic_hidden_dim : int
    lr                : float
    gamma             : float
    tau               : float
    batch_size        : int
    buffer_size       : int
    total_steps       : int
    learning_starts   : int
    gradient_steps    : int
    log_freq          : int
    save_freq         : int
    device            : str = 'cpu'

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SACConfig(**cfg)

@dataclass
class PPOConfig:
    motion_path: str
    hidden_dims: list
    activation: str
    lr: float
    gamma: float
    gae_lambda: float
    clip_range: float
    n_epochs: int
    batch_size: int
    rollout_steps: int
    ent_coef: float
    vf_coef: float
    max_grad_norm: float
    total_steps: int
    n_envs: int
    log_freq: int
    save_freq: int
    vector_env_type: str = "async"  # async | sync
    vector_env_context: str = "spawn"  # spawn | fork | forkserver
    seed: int = 42
    torch_num_threads: int = 1
    torch_num_interop_threads: int = 1
    future_offsets: list | None = None
    smoothing_window: int = 5
    action_scale: float = 0.1
    init_log_std: float = -1.5
    mean_scale: float = 0.5
    obs_clip: float = 10.0
    episode_length: int = 1000
    reset_phase_start: float = 0.0
    reset_phase_end: float = 0.15
    reset_joint_noise: float = 0.0
    reset_vel_noise: float = 0.0
    initial_height_threshold: float = 0.45
    height_threshold: float = 0.25
    initial_ori_threshold: float = 1.5
    ori_threshold: float = 0.8
    min_root_height: float = 0.45
    curriculum_mode: str = "off"  # off | linear
    final_reset_phase_end: float = 1.0
    curriculum_update_freq: int = 100000
    contact_height_threshold: float = 0.06
    contact_vel_threshold: float = 0.35
    pose_reward_weight: float = 0.40
    vel_reward_weight: float = 0.10
    root_reward_weight: float = 0.20
    root_vel_reward_weight: float = 0.15
    eff_reward_weight: float = 0.10
    contact_reward_weight: float = 0.05
    pose_sigma: float = 0.35
    vel_sigma: float = 2.0
    root_sigma: float = 0.35
    root_vel_sigma: float = 1.0
    eff_sigma: float = 0.12
    action_rate_weight: float = 0.01
    joint_limit_weight: float = 2.0
    residual_reg_coef: float = 0.0
    target_kl: float = 0.0
    eval_deterministic_default: bool = True
    device: str = 'cpu'

def load_ppo_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if cfg.get('future_offsets') is None:
        cfg['future_offsets'] = [0, 1, 2, 4]
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return PPOConfig(**cfg)
