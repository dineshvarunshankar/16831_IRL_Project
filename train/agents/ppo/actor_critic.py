"""
Actor-critic modules for PPO with squashed Gaussian actions.
"""

import numpy as np
import torch
import torch.nn as nn


def orthogonal_init(module, gain=np.sqrt(2)):
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        nn.init.zeros_(module.bias)


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        hidden_dims=[512, 256, 128],
        activation="elu",
        init_log_std=-1.5,
        mean_scale=0.5,
    ):
        super().__init__()
        act_layer = nn.ELU if activation.lower() == "elu" else nn.ReLU

        layers = []
        d = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(act_layer())
            d = h

        self.net = nn.Sequential(*layers)
        self.mean_head = nn.Linear(hidden_dims[-1], act_dim)
        self.log_std = nn.Parameter(torch.full((act_dim,), init_log_std))
        self.mean_scale = float(mean_scale)

        self.net.apply(orthogonal_init)
        orthogonal_init(self.mean_head, gain=0.01)

    def forward(self, state):
        x = self.net(state)
        # Bound deterministic residuals to reduce saturation and over-correction.
        mean = self.mean_scale * torch.tanh(self.mean_head(x))
        log_std = self.log_std.clamp(-5.0, 2.0).expand_as(mean)
        std = log_std.exp()
        return mean, std

    def get_distribution(self, state):
        mean, std = self.forward(state)
        return torch.distributions.Normal(mean, std)

    @staticmethod
    def _atanh(x):
        x = x.clamp(-0.999999, 0.999999)
        return 0.5 * (torch.log1p(x) - torch.log1p(-x))

    @staticmethod
    def _squash_log_prob(dist, pre_tanh_action, action):
        log_prob = dist.log_prob(pre_tanh_action).sum(dim=-1, keepdim=True)
        correction = torch.log(1.0 - action.pow(2) + 1e-6).sum(dim=-1, keepdim=True)
        return log_prob - correction

    def log_prob(self, state, action):
        dist = self.get_distribution(state)
        pre_tanh_action = self._atanh(action)
        return self._squash_log_prob(dist, pre_tanh_action, action)

    def sample(self, state):
        dist = self.get_distribution(state)
        pre_tanh_action = dist.rsample()
        action = torch.tanh(pre_tanh_action)
        log_prob = self._squash_log_prob(dist, pre_tanh_action, action)
        return action, log_prob


class Critic(nn.Module):
    def __init__(self, obs_dim, hidden_dims=[512, 256, 128], activation="elu"):
        super().__init__()
        act_layer = nn.ELU if activation.lower() == "elu" else nn.ReLU

        layers = []
        d = obs_dim
        for h in hidden_dims:
            layers.append(nn.Linear(d, h))
            layers.append(act_layer())
            d = h

        layers.append(nn.Linear(hidden_dims[-1], 1))
        self.net = nn.Sequential(*layers)

        self.net.apply(orthogonal_init)
        orthogonal_init(self.net[-1], gain=1.0)

    def forward(self, state):
        return self.net(state)
