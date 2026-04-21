import torch
import torch.nn.functional as F
import numpy as np
from train.agents.base_agent import BaseAgent
from train.agents.sac.actor import Actor
from train.agents.sac.critic import DoubleCritic
from train.replay_buffer import ReplayBuffer


class SACAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, config):
        self.actor = Actor(obs_dim, act_dim, config.actor_hidden_dim).to(config.device)
        self.critic = DoubleCritic(obs_dim, act_dim, config.critic_hidden_dim).to(config.device) 

        # target critic, no gradients
        self.critic_target = DoubleCritic(obs_dim, act_dim, config.critic_hidden_dim).to(config.device)
        self.critic_target.load_state_dict(self.critic.state_dict()) #identical networks 

        # opt
        self.actor_optimizer  = torch.optim.Adam(self.actor.parameters(),  lr=config.lr, weight_decay=0.001)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=config.lr, weight_decay=0.001)

        # entropy temperature autotuning
        self.target_entropy = -act_dim / 2  # FastSAC: -|A|/2 for tracking tasks
        self.log_alpha      = torch.tensor([np.log(0.001)], dtype=torch.float32, requires_grad=True, device=config.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.lr)

        # replay buffer
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, act_dim, config.device)

        self.config = config
        self.device = config.device
        self.obs_dim = obs_dim
        self.act_dim = act_dim
    
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        if deterministic:
            with torch.no_grad():
                mean, _ = self.actor.forward(state)
                action = torch.tanh(mean)
        else:
            with torch.no_grad():
                action, _ = self.actor.sample(state)
        
        return action.cpu().numpy().flatten()

    
    def collect(self, state, action, reward, next_state, done):
        self.buffer.add(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.buffer) < self.config.batch_size:
            return

        for _ in range(self.config.gradient_steps):
            s, a, r, s_next, done = self.buffer.sample(self.config.batch_size)

            s      = s.to(self.device)
            a      = a.to(self.device)
            r      = r.to(self.device)
            s_next = s_next.to(self.device)
            done   = done.to(self.device)

            self._update_critic(s, a, r, s_next, done)
            self._update_actor(s)
        
        # try doing this once per update instead of every gradient step
        self._update_alpha(s)
        self._soft_update_targets()


    
    def _update_critic(self, s, a, r, s_next, done):

        with torch.no_grad():
            a_next, log_prob = self.actor.sample(s_next)

            alpha = self.log_alpha.exp()

            y = r + self.config.gamma * (self.critic_target.min_Q(s_next, a_next) - alpha*log_prob) * (1 - done)

        q1, q2 = self.critic(s, a) #forward
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
    
    def _update_actor(self, s):
        a, log_prob = self.actor.sample(s)
        alpha = self.log_alpha.exp().detach()

        actor_loss = (alpha * log_prob - self.critic.min_Q(s, a)).mean()

        self.actor_optimizer.zero_grad()  
        actor_loss.backward()
        self.actor_optimizer.step()
    
    def _update_alpha(self, s):
        with torch.no_grad():
            a, log_prob = self.actor.sample(s)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
    
    def _soft_update_targets(self):
        tau = self.config.tau 

        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)
     
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
        try:
            checkpoint = torch.load(
                path, map_location=self.device, weights_only=False
            )
        except TypeError:
            # Backward compatibility for torch versions without `weights_only`.
            checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
        self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
