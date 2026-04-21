import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x)

class DoubleCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.Q1 = QNetwork(obs_dim, act_dim, hidden_dim)
        self.Q2 = QNetwork(obs_dim, act_dim, hidden_dim)
    
    def forward(self, state, action):
        return self.Q1(state, action), self.Q2(state, action)
    
    def min_Q(self, state, action):
        return torch.min(self.Q1(state, action), self.Q2(state, action))