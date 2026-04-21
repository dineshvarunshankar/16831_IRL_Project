import torch
import torch.nn.functional as F
import numpy as np

from agents.base import BaseAlgorithm
from agents.sac.actor import Actor
from agents.sac.critic import DoubleCritic
from agents.sac.replay_buffer import ReplayBuffer


class SACAgent(BaseAlgorithm):
    def __init__(self, actor_obs_dim, critic_obs_dim, act_dim, config):
        self.actor = Actor(actor_obs_dim, act_dim, config.actor_hidden_dim).to(config.device)
        self.critic = DoubleCritic(critic_obs_dim, act_dim, config.critic_hidden_dim).to(config.device) 

        # target critic, no gradients
        self.critic_target = DoubleCritic(critic_obs_dim, act_dim, config.critic_hidden_dim).to(config.device)
        self.critic_target.load_state_dict(self.critic.state_dict()) #identical networks 

        # opt
        self.actor_optimizer  = torch.optim.AdamW(
            self.actor.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )
        self.critic_optimizer = torch.optim.AdamW(
            self.critic.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
            betas=(0.9, 0.95),
            fused=True,
        )

        # entropy temperature autotuning
        self.target_entropy = -act_dim / 2  # FastSAC: -|A|/2 for tracking tasks
        self.log_alpha      = torch.tensor([np.log(0.001)], dtype=torch.float32, requires_grad=True, device=config.device)
        self.alpha_optimizer = torch.optim.AdamW([self.log_alpha], lr=config.lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True)

        # replay buffer
        self.buffer = ReplayBuffer(
            max_size       = config.buffer_size,
            num_envs       = config.num_envs,
            actor_obs_dim  = actor_obs_dim,
            critic_obs_dim = critic_obs_dim,
            act_dim        = act_dim,
            device         = config.device,
        )

        self.config = config
        self.device = config.device
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
    
    def select_action(self, state, deterministic=False):
        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
            else:
                action, _ = self.actor.sample(state)
        
        return action

    
    def collect(self, actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done):
        self.buffer.add(actor_obs, critic_obs, action, reward, next_actor_obs, next_critic_obs, done)
    
    def update(self):
        if len(self.buffer) < self.config.batch_size:
            return

        for _ in range(self.config.gradient_steps):
            actor_obs, critic_obs, a, r, next_actor_obs, next_critic_obs, done = self.buffer.sample(self.config.batch_size)
            self._update_critic(actor_obs, critic_obs, a, r, next_actor_obs, next_critic_obs, done)
            self._update_actor(actor_obs, critic_obs)
            self._update_alpha(actor_obs)
            self._soft_update_targets()
    
    def _update_critic(self, actor_obs, critic_obs, a, r, next_actor_obs, next_critic_obs, done):
        with torch.no_grad():
            a_next, log_prob = self.actor.sample(next_actor_obs)

            alpha = self.log_alpha.exp()

            y = r + self.config.gamma * (self.critic_target.min_Q(next_critic_obs, a_next) - alpha*log_prob) * (1 - done)

        q1, q2 = self.critic(critic_obs, a) #forward 
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def _update_actor(self, actor_obs, critic_obs):
        a, log_prob = self.actor.sample(actor_obs)
        alpha = self.log_alpha.exp().detach()

        actor_loss = (alpha * log_prob - self.critic.min_Q(critic_obs, a)).mean()

        self.actor_optimizer.zero_grad()  
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def _update_alpha(self, actor_obs):
        with torch.no_grad():
            _, log_prob = self.actor.sample(actor_obs)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    
    @torch.no_grad()
    def _soft_update_targets(self):
        tau = self.config.tau
        src = [p.data for p in self.critic.parameters()]
        tgt = [p.data for p in self.critic_target.parameters()]
        torch._foreach_mul_(tgt, 1.0 - tau)
        torch._foreach_add_(tgt, src, alpha=tau)
     
    def save(self, path):
        torch.save({
            'actor'          : self.actor.state_dict(),
            'critic'         : self.critic.state_dict(),
            'critic_target'  : self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha'      : self.log_alpha,
            'alpha_optimizer': self.alpha_optimizer.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])