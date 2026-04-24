import yaml
from dataclasses import dataclass

@dataclass
class SACConfig:
    motion_path             : str
    device                  : str
    seed                    : int
    wandb_project           : str
    actor_hidden_dim        : list[int]
    critic_hidden_dim       : list[int]
    lr                      : float
    weight_decay            : float
    gamma                   : float
    tau                     : float
    batch_size              : int
    buffer_size             : int
    num_learning_iterations : int
    learning_starts         : int
    gradient_steps          : int
    alpha_init              : float
    target_entropy_ratio    : float
    log_std_min             : float
    log_std_max             : float
    log_freq                : int
    ep_stats_window         : int
    ckpt_fractions          : list[float]
    use_layer_norm          : bool
    use_mean_q              : bool
    num_envs                : int = 4096
    policy_frequency        : int = 1
    use_autotune            : bool = True
    alpha_lr                : float = 3e-5
    critic_grad_clip        : float = 1.0

@dataclass
class PPOConfig:
    motion_path       : str
    hidden_dim        : int
    lr                : float
    gamma             : float
    gae_lambda        : float
    clip_range        : float
    n_epochs          : int
    batch_size        : int
    rollout_steps     : int
    ent_coef          : float
    vf_coef           : float
    max_grad_norm     : float
    total_steps       : int
    log_freq          : int
    save_freq         : int
    n_envs            : int = 1
    action_scale      : float = 0.1
    reset_joint_noise : float = 0.0
    reset_vel_noise   : float = 0.0
    term_height_threshold: float = 0.4
    term_orientation_threshold: float = 1.2
    init_log_std      : float = -2.5
    lr_anneal         : bool = True
    seed              : int = 42
    target_kl         : float = 0.01
    vector_env_type   : str = "async"   # async | sync
    vector_env_context: str = "spawn"   # spawn | fork | forkserver
    torch_num_threads : int = 1
    device            : str = 'cpu'

def _auto_device(cfg):
    import torch
    if 'device' not in cfg:
        cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return cfg

def load_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return SACConfig(**_auto_device(cfg))

def load_ppo_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return PPOConfig(**_auto_device(cfg))
