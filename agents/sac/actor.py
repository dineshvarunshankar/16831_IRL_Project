import torch
import torch.nn as nn
from torch.distributions import Normal

LOG_STD_MAX = 0.0
LOG_STD_MIN = -20.0

class Actor(nn.Module):
    def __init__(self, actor_obs_dim, act_dim, hidden_dim:list[int]):
        super().__init__()

        layers = []
        in_dim = actor_obs_dim
        for out_dim in hidden_dim:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim

        self.net = nn.Sequential(*layers)

        self.mean_head = nn.Linear(hidden_dim[-1], act_dim)
        self.log_std_head = nn.Linear(hidden_dim[-1], act_dim)


    def forward(self, state):   
        x = self.net(state)
        mean = self.mean_head(x)
        log_std = self.log_std_head(x)

        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)

        return mean, log_std


    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)
        u = dist.rsample()

        action = torch.tanh(u) # to squash to (-1,1)

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob
