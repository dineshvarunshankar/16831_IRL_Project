import yaml
from dataclasses import dataclass

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
    import torch
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return SACConfig(**cfg)