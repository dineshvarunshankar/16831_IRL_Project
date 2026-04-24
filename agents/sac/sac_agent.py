import torch
import torch.nn.functional as F
import numpy as np

from agents.base import BaseAlgorithm
from agents.sac.actor import Actor
from agents.sac.critic import DoubleCritic
from agents.sac.replay_buffer import ReplayBuffer


class SACAgent(BaseAlgorithm):
    def __init__(self, actor_obs_dim, critic_obs_dim, act_dim, config):
        self.actor = Actor(
            actor_obs_dim, act_dim, config.actor_hidden_dim,
            log_std_min=config.log_std_min, log_std_max=config.log_std_max,
        ).to(config.device)
        self.critic = DoubleCritic(critic_obs_dim, act_dim, config.critic_hidden_dim, use_layer_norm=config.use_layer_norm).to(config.device) 

        # target critic, no gradients
        self.critic_target = DoubleCritic(critic_obs_dim, act_dim, config.critic_hidden_dim, use_layer_norm=config.use_layer_norm).to(config.device)
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
        self.use_autotune = getattr(config, 'use_autotune', True)
        self.target_entropy = -act_dim * config.target_entropy_ratio
        self.log_alpha = torch.tensor(
            [np.log(config.alpha_init)],
            dtype=torch.float32,
            requires_grad=self.use_autotune,
            device=config.device,
        )
        if self.use_autotune:
            alpha_lr = getattr(config, 'alpha_lr', config.lr)
            self.alpha_optimizer = torch.optim.AdamW(
                [self.log_alpha], lr=alpha_lr, weight_decay=0.0, betas=(0.9, 0.95), fused=True
            )
        else:
            self.alpha_optimizer = None

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

        self._q_agg = self.critic.mean_Q if config.use_mean_q else self.critic.min_Q

        self._q_agg_target = self.critic_target.mean_Q if config.use_mean_q else self.critic_target.min_Q

        # metric accumulator: {name: (running_sum_tensor, count)}
        self._metric_sums: dict[str, torch.Tensor] = {}
        self._metric_count: int = 0

        # TD3-style delayed policy updates: actor + alpha update every N critic updates
        self.policy_frequency = getattr(config, 'policy_frequency', 1)
        self._update_counter = 0

    def _accum(self, m: dict[str, torch.Tensor]) -> None:
        for k, v in m.items():
            if k not in self._metric_sums:
                self._metric_sums[k] = v.detach().clone()
            else:
                self._metric_sums[k] += v.detach()
        self._metric_count += 1

    def pop_metrics(self) -> dict[str, float]:
        if self._metric_count == 0:
            return {}
        n = self._metric_count
        out = {k: (v / n).item() for k, v in self._metric_sums.items()}
        self._metric_sums.clear()
        self._metric_count = 0
        return out
    
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
            m_c = self._update_critic(actor_obs, critic_obs, a, r, next_actor_obs, next_critic_obs, done)

            # TD3-style delayed actor + alpha update
            self._update_counter += 1
            if self._update_counter % self.policy_frequency == 0:
                m_a = self._update_actor(actor_obs, critic_obs)
                if self.use_autotune:
                    m_alpha = self._update_alpha(actor_obs)
                    self._accum({**m_c, **m_a, **m_alpha})
                else:
                    self._accum({**m_c, **m_a, "alpha/value": self.log_alpha.exp().detach().squeeze()})
            else:
                self._accum(m_c)

            self._soft_update_targets()
    
    def _update_critic(self, actor_obs, critic_obs, a, r, next_actor_obs, next_critic_obs, done):
        with torch.no_grad():
            a_next, log_prob = self.actor.sample(next_actor_obs)

            alpha = self.log_alpha.exp()

            y = r + self.config.gamma * (self._q_agg_target(next_critic_obs, a_next) - alpha*log_prob) * (1 - done)

        q1, q2 = self.critic(critic_obs, a) #forward
        critic_loss = F.mse_loss(q1, y) + F.mse_loss(q2, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), float('inf'))
        self.critic_optimizer.step()

        with torch.no_grad():
            return {
                "loss/critic":    critic_loss.detach(),
                "q/mean":         0.5 * (q1.mean() + q2.mean()),
                "q/max":          torch.maximum(q1.max(), q2.max()),
                "q/min":          torch.minimum(q1.min(), q2.min()),
                "q/target_mean":  y.mean(),
                "grad/critic_norm": grad_norm.detach(),
            }
    
    def _update_actor(self, actor_obs, critic_obs):
        a, log_prob = self.actor.sample(actor_obs)
        alpha = self.log_alpha.exp().detach()

        actor_loss = (alpha * log_prob - self._q_agg(critic_obs, a)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), float('inf'))
        self.actor_optimizer.step()

        with torch.no_grad():
            _, log_std = self.actor.forward(actor_obs)
            return {
                "loss/actor":         actor_loss.detach(),
                "policy/entropy":     -log_prob.mean().detach(),
                "policy/log_prob":    log_prob.mean().detach(),
                "policy/action_std":  log_std.exp().mean().detach(),
                "grad/actor_norm":    grad_norm.detach(),
            }
    
    def _update_alpha(self, actor_obs):
        with torch.no_grad():
            _, log_prob = self.actor.sample(actor_obs)

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy)).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        with torch.no_grad():
            return {
                "loss/alpha":   alpha_loss.detach(),
                "alpha/value":  self.log_alpha.exp().detach().squeeze(),
                "alpha/log":    self.log_alpha.detach().squeeze(),
            }
    
    @torch.no_grad()
    def _soft_update_targets(self):
        tau = self.config.tau
        src = [p.data for p in self.critic.parameters()]
        tgt = [p.data for p in self.critic_target.parameters()]
        torch._foreach_mul_(tgt, 1.0 - tau)
        torch._foreach_add_(tgt, src, alpha=tau)
     
    def save(self, path):
        ckpt = {
            'actor'          : self.actor.state_dict(),
            'critic'         : self.critic.state_dict(),
            'critic_target'  : self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'log_alpha'      : self.log_alpha,
        }
        if self.alpha_optimizer is not None:
            ckpt['alpha_optimizer'] = self.alpha_optimizer.state_dict()
        torch.save(ckpt, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.log_alpha.data.copy_(checkpoint['log_alpha'].data)
        if self.alpha_optimizer is not None and 'alpha_optimizer' in checkpoint:
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])