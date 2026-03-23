"""
PPO agent with observation normalization and squashed Gaussian actions.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from train.agents.base_agent import BaseAgent
from train.agents.ppo.actor_critic import Actor, Critic


class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, batch):
        batch = np.asarray(batch, dtype=np.float64)
        if batch.ndim == 1:
            batch = batch[None, :]

        batch_mean = batch.mean(axis=0)
        batch_var = batch.var(axis=0)
        batch_count = batch.shape[0]

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / total_count
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + (delta ** 2) * self.count * batch_count / total_count

        self.mean = new_mean
        self.var = m2 / total_count
        self.count = total_count

    def normalize(self, x):
        return (x - self.mean) / (np.sqrt(self.var) + 1e-8)


class RolloutBuffer:
    def __init__(self, steps_per_env, num_envs, obs_dim, act_dim):
        self.steps_per_env = steps_per_env
        self.num_envs = num_envs
        self.states = np.zeros((steps_per_env, num_envs, obs_dim), dtype=np.float32)
        self.actions = np.zeros((steps_per_env, num_envs, act_dim), dtype=np.float32)
        self.rewards = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.dones = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.log_probs = np.zeros((steps_per_env, num_envs), dtype=np.float32)
        self.values = np.zeros((steps_per_env, num_envs), dtype=np.float32)
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
            batch_idx = indices[start:end]
            yield (
                torch.as_tensor(flat_states[batch_idx], dtype=torch.float32, device=device),
                torch.as_tensor(flat_actions[batch_idx], dtype=torch.float32, device=device),
                torch.as_tensor(flat_log_probs[batch_idx], dtype=torch.float32, device=device),
                torch.as_tensor(flat_advantages[batch_idx], dtype=torch.float32, device=device),
                torch.as_tensor(flat_returns[batch_idx], dtype=torch.float32, device=device),
            )


class PPOAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, config):
        self.device = config.device
        self.n_envs = getattr(config, "n_envs", 1)
        self.steps_per_env = config.rollout_steps // self.n_envs
        self.obs_clip = float(getattr(config, "obs_clip", 10.0))

        hidden_dims = getattr(config, "hidden_dims", [512, 256, 128])
        activation = getattr(config, "activation", "elu")

        self.actor = Actor(
            obs_dim,
            act_dim,
            hidden_dims,
            activation,
            init_log_std=float(getattr(config, "init_log_std", -1.5)),
        ).to(self.device)
        self.critic = Critic(obs_dim, hidden_dims, activation).to(self.device)

        self.optimizer = torch.optim.Adam(
            [
                {"params": self.actor.parameters(), "lr": config.lr},
                {"params": self.critic.parameters(), "lr": config.lr},
            ]
        )

        self.buffer = RolloutBuffer(self.steps_per_env, self.n_envs, obs_dim, act_dim)
        self.obs_normalizer = RunningMeanStd(shape=(obs_dim,))
        self.config = config
        self._cached_state_norm = None

    def _ensure_batch(self, obs):
        obs = np.asarray(obs, dtype=np.float32)
        if obs.ndim == 1:
            return obs[None, :], True
        return obs, False

    def _normalize_obs(self, obs):
        obs_norm = self.obs_normalizer.normalize(obs)
        obs_norm = np.clip(obs_norm, -self.obs_clip, self.obs_clip)
        return obs_norm.astype(np.float32)

    def select_action(self, state, deterministic=False):
        state_batch, squeezed = self._ensure_batch(state)
        state_norm = self._normalize_obs(state_batch)
        state_tensor = torch.as_tensor(state_norm, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.actor(state_tensor)
                action = torch.tanh(mean)
                log_prob = self.actor.log_prob(state_tensor, action)
            else:
                action, log_prob = self.actor.sample(state_tensor)
            value = self.critic(state_tensor)

        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy().reshape(-1)
        value_np = value.cpu().numpy().reshape(-1)
        self._cached_state_norm = state_norm.copy()

        if squeezed:
            return action_np[0], float(log_prob_np[0]), float(value_np[0])
        return action_np, log_prob_np, value_np

    def collect(self, state, action, reward, done, log_prob, value):
        state_batch, _ = self._ensure_batch(state)
        action_batch, _ = self._ensure_batch(action)
        batch_size = state_batch.shape[0]
        reward_batch = np.asarray(reward, dtype=np.float32).reshape(batch_size)
        done_batch = np.asarray(done, dtype=np.float32).reshape(batch_size)
        log_prob_batch = np.asarray(log_prob, dtype=np.float32).reshape(batch_size)
        value_batch = np.asarray(value, dtype=np.float32).reshape(batch_size)

        if self._cached_state_norm is None:
            state_norm = self._normalize_obs(state_batch)
        else:
            state_norm = self._cached_state_norm

        self.buffer.add(
            state_norm,
            action_batch,
            reward_batch,
            done_batch,
            log_prob_batch,
            value_batch,
        )
        self.obs_normalizer.update(state_batch)
        self._cached_state_norm = None

    def update(self, next_obs):
        if not self.buffer.is_full():
            return {}

        next_obs_batch, _ = self._ensure_batch(next_obs)
        next_state_norm = self._normalize_obs(next_obs_batch)
        next_state_tensor = torch.as_tensor(
            next_state_norm, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            last_values = self.critic(next_state_tensor).cpu().numpy().reshape(-1)

        self.buffer.compute_gae(last_values, self.config.gamma, self.config.gae_lambda)

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        for _ in range(self.config.n_epochs):
            for states, actions, old_log_probs, advantages, returns in self.buffer.get_batches(
                self.config.batch_size, self.device
            ):
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                new_log_probs = self.actor.log_prob(states, actions).squeeze(-1)
                values = self.critic(states).squeeze(-1)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantages
                surr2 = (
                    torch.clamp(
                        ratio,
                        1.0 - self.config.clip_range,
                        1.0 + self.config.clip_range,
                    )
                    * advantages
                )
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, returns)
                entropy = self.actor.get_distribution(states).entropy().sum(dim=-1).mean()

                loss = (
                    policy_loss
                    + self.config.vf_coef * value_loss
                    - self.config.ent_coef * entropy
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.config.max_grad_norm,
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.item()
                n_updates += 1

        self.buffer.reset()
        return {
            "policy_loss": total_policy_loss / max(n_updates, 1),
            "value_loss": total_value_loss / max(n_updates, 1),
            "entropy": total_entropy / max(n_updates, 1),
        }

    def save(self, path):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "obs_mean": self.obs_normalizer.mean,
                "obs_var": self.obs_normalizer.var,
                "obs_count": self.obs_normalizer.count,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.obs_normalizer.mean = checkpoint["obs_mean"]
        self.obs_normalizer.var = checkpoint["obs_var"]
        self.obs_normalizer.count = checkpoint["obs_count"]
