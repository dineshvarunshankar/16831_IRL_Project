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
    motion_path   : str
    hidden_dims   : list
    activation    : str
    lr            : float
    gamma         : float
    gae_lambda    : float
    clip_range    : float
    n_epochs      : int
    batch_size    : int
    rollout_steps : int
    ent_coef      : float
    vf_coef       : float
    max_grad_norm : float
    total_steps   : int
    n_envs        : int
    log_freq      : int
    save_freq     : int
    device        : str = 'cpu'

def load_ppo_config(path):
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return PPOConfig(**cfg)