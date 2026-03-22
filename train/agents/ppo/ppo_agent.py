"""
PPOAgent — Proximal Policy Optimization for Motion Imitation

On-policy RL agent with:
- GAE (Generalized Advantage Estimation) for low-variance advantage estimates
- Clipped surrogate objective for stable policy updates
- Running observation normalization for stable learning
- Advantage normalization per minibatch

Training loop:
    1. Collect rollout_steps transitions using current policy
    2. Compute GAE advantages and returns
    3. Split rollout into minibatches
    4. Run n_epochs of gradient updates on each minibatch
    5. Repeat
"""

import torch
import torch.nn.functional as F
import numpy as np
from train.agents.base_agent import BaseAgent
from train.agents.ppo.actor_critic import Actor, Critic


class RunningMeanStd:
    """
    Running mean and standard deviation for observation normalization.
    Uses Welford's online algorithm for numerical stability.
    """
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4  # small value to avoid division by zero

    def update(self, batch):
        batch = np.asarray(batch)
        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        # parallel variance computation
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RolloutBuffer:
    """
    Stores one rollout of transitions for PPO.
    Supports batched environments: states shape (steps_per_env, num_envs, obs_dim)
    """
    def __init__(self, steps_per_env, num_envs, obs_dim, act_dim):
        self.steps_per_env = steps_per_env
        self.num_envs = num_envs
        self.states = np.zeros((steps_per_env, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((steps_per_env, num_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.dones = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.values = np.zeros((steps_per_env, num_envs), dtype=np.float32)

        # computed after rollout
        self.advantages = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.returns = np.zeros((steps_per_env, num_envs), dtype=np.float32)

        self.ptr = 0

    def add(self, state, action, reward, done, log_prob, value):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.log_probs[self.ptr] = log_prob
        self.values[self.ptr] = value
        self.ptr += 1

    def is_full(self):
        return self.ptr >= self.steps_per_env

    def reset(self):
        self.ptr = 0

    def compute_gae(self, last_values, gamma, gae_lambda):
        """
        Compute Generalized Advantage Estimation for batched envs.
        """
        last_gae = np.zeros(self.num_envs, dtype=np.float32)
        for t in reversed(range(self.steps_per_env)):
            if t == self.steps_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            next_non_terminal = 1.0 - self.dones[t]
            delta = self.rewards[t] + gamma * next_values * next_non_terminal - self.values[t]
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[t] = last_gae

        self.returns = self.advantages + self.values

    def get_batches(self, batch_size, device):
        """
        Yield minibatches of shuffled rollout data flattened across envs and steps.
        """
        total_steps = self.steps_per_env * self.num_envs
        indices = np.arange(total_steps)
        np.random.shuffle(indices)
        
        flat_states = self.states.reshape(total_steps, -1)
        flat_actions = self.actions.reshape(total_steps, -1)
        flat_log_probs = self.log_probs.reshape(total_steps)
        flat_advantages = self.advantages.reshape(total_steps)
        flat_returns = self.returns.reshape(total_steps)

        for start in range(0, total_steps, batch_size):
            end = start + batch_size
            b = indices[start:end]

            yield (
                torch.FloatTensor(flat_states[b]).to(device),
                torch.FloatTensor(flat_actions[b]).to(device),
                torch.FloatTensor(flat_log_probs[b]).to(device),
                torch.FloatTensor(flat_advantages[b]).to(device),
                torch.FloatTensor(flat_returns[b]).to(device),
            )


class PPOAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, config):
        self.device = config.device
        self.n_envs = getattr(config, 'n_envs', 1)
        self.steps_per_env = config.rollout_steps // self.n_envs

        # networks (support variable hidden_dims and activation)
        hidden_dims = getattr(config, 'hidden_dims', [512, 256, 128])
        activation = getattr(config, 'activation', 'elu')
        
        self.actor = Actor(obs_dim, act_dim, hidden_dims, activation).to(self.device)
        self.critic = Critic(obs_dim, hidden_dims, activation).to(self.device)

        # single optimizer for both actor and critic
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': config.lr},
            {'params': self.critic.parameters(), 'lr': config.lr},
        ])

        # rollout buffer (supports num_envs)
        self.buffer = RolloutBuffer(self.steps_per_env, self.n_envs, obs_dim, act_dim)

        # observation normalization (supports batched obs)
        self.obs_normalizer = RunningMeanStd(shape=(obs_dim,))

        # config
        self.config = config
        self.obs_dim = obs_dim
        self.act_dim = act_dim

    def _normalize_obs(self, obs):
        """Normalize observation using running statistics"""
        return self.obs_normalizer.normalize(obs).astype(np.float32)

    def select_action(self, state, deterministic=False):
        """
        Select action from current policy for a batch of environments.
        Returns actions, log_probs, values.
        """
        # normalize batched observation
        state_norm = self._normalize_obs(state)
        state_tensor = torch.FloatTensor(state_norm).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
                log_prob = torch.zeros(self.n_envs)
            else:
                action, log_prob = self.actor.sample(state_tensor)

            value = self.critic(state_tensor)

        action = action.cpu().numpy()
        log_prob = log_prob.cpu().numpy().flatten()
        value = value.cpu().numpy().flatten()

        return action, log_prob, value

    def collect(self, state, action, reward, done, log_prob, value):
        """Add transition to rollout buffer and update observation normalizer"""
        # update normalizer with raw batched observations
        self.obs_normalizer.update(state)

        # store normalized observation in buffer
        state_norm = self._normalize_obs(state)
        self.buffer.add(state_norm, action, reward, done, log_prob, value)

    def update(self, next_obs):
        """
        Run PPO update on collected rollout data.
        next_obs is the batched observation after the last step of the rollout.
        """
        if not self.buffer.is_full():
            return {}

        # get value of the state *after* the rollout for GAE bootstrap
        next_state_norm = self._normalize_obs(next_obs)
        next_state_tensor = torch.FloatTensor(next_state_norm).to(self.device)
        with torch.no_grad():
            last_values = self.critic(next_state_tensor).cpu().numpy().flatten()

        # compute GAE advantages and returns
        self.buffer.compute_gae(last_values, self.config.gamma, self.config.gae_lambda)

        # track losses for logging
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        n_updates = 0

        # run multiple epochs over the rollout data
        for epoch in range(self.config.n_epochs):
            for states, actions, old_log_probs, advantages, returns in \
                    self.buffer.get_batches(self.config.batch_size, self.device):

                # normalize advantages (per minibatch)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # current policy evaluation
                new_log_probs = self.actor.log_prob(states, actions).squeeze(-1)
                values = self.critic(states).squeeze(-1)

                # policy loss (clipped surrogate objective)
                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.config.clip_range,
                                          1.0 + self.config.clip_range) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss (MSE)
                value_loss = F.mse_loss(values, returns)

                # entropy bonus (for exploration)
                dist = self.actor.get_distribution(states)
                entropy = dist.entropy().sum(dim=-1).mean()

                # combined loss
                loss = (policy_loss
                        + self.config.vf_coef * value_loss
                        - self.config.ent_coef * entropy)

                self.optimizer.zero_grad()
                loss.backward()

                # gradient clipping for stability
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)

                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        # reset buffer for next rollout
        self.buffer.reset()

        return {
            'policy_loss': total_policy_loss / max(n_updates, 1),
            'value_loss': total_value_loss / max(n_updates, 1),
            'entropy': total_entropy / max(n_updates, 1),
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'obs_mean': self.obs_normalizer.mean,
            'obs_var': self.obs_normalizer.var,
            'obs_count': self.obs_normalizer.count,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.obs_normalizer.mean = checkpoint['obs_mean']
        self.obs_normalizer.var = checkpoint['obs_var']
        self.obs_normalizer.count = checkpoint['obs_count']
