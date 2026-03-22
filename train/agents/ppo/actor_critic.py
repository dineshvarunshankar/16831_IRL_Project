"""
ActorCritic — Separate Actor and Critic Networks for PPO
Both use orthogonal initialization.
Actor output is squashed to [-1, 1] via tanh.
"""

import torch
import torch.nn as nn
import numpy as np


def orthogonal_init(module, gain=np.sqrt(2)):
    """Orthogonal weight initialization"""
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)

        # learnable log_std (state-independent)
        self.log_std = nn.Parameter(torch.zeros(act_dim))

        # orthogonal init
        self.net.apply(orthogonal_init)
        orthogonal_init(self.mean_head, gain=0.01)  # small init for action head

    def forward(self, state):
        """Returns action mean and std"""
        x = self.net(state)
        mean = self.mean_head(x)
        std = self.log_std.exp().expand_as(mean)
        return mean, std

    def get_distribution(self, state):
        """Returns a Normal distribution over actions"""
        mean, std = self.forward(state)
        return torch.distributions.Normal(mean, std)

    def log_prob(self, state, action):
        """
        Compute log probability of action under current policy.
        Accounts for tanh squashing: log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
        where a = tanh(u).
        """
        dist = self.get_distribution(state)

        # inverse tanh to get pre-squash action
        # atanh(a) = 0.5 * ln((1+a)/(1-a))
        action_clipped = action.clamp(-0.999, 0.999)
        u = torch.atanh(action_clipped)

        log_prob = dist.log_prob(u)
        # correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        return log_prob.sum(dim=-1, keepdim=True)

    def sample(self, state):
        """Sample action with tanh squashing, return action and log_prob"""
        dist = self.get_distribution(state)
        u = dist.rsample()
        action = torch.tanh(u)

        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

        # orthogonal init (output layer with gain=1.0)
        self.net.apply(orthogonal_init)
        orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, state):
        return self.net(state)
