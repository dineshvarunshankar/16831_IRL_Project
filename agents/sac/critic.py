import torch
import torch.nn as nn

class QNetwork(nn.Module):
    def __init__(self, critic_obs_dim, act_dim, hidden_dims: list[int]):
        super().__init__()
        
        layers = []
        in_dim = critic_obs_dim + act_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, 1))
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dims: list[int]):
        super().__init__()
        self.Q1 = QNetwork(obs_dim, act_dim, hidden_dims)
        self.Q2 = QNetwork(obs_dim, act_dim, hidden_dims)
    
    def forward(self, state, action):
        return self.Q1(state, action), self.Q2(state, action)
    
    def min_Q(self, state, action):
        return torch.min(self.Q1(state, action), self.Q2(state, action))