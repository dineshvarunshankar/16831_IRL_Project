"""
PPO Training Script for Motion Imitation

On-policy training loop:
    1. Collect rollout_steps transitions
    2. PPO agent computes GAE and updates policy
    3. Log metrics to wandb
    4. Update curriculum termination thresholds
    5. Repeat until total_steps reached
"""

import numpy as np
import torch
import wandb
import os
import gymnasium as gym
from env.locomimic_env import LocoMimicEnv
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_ppo_config


def train(config_path='train/configs/ppo_config.yaml'):
    config = load_ppo_config(config_path)

    # init wandb
    wandb.init(
        project='locomimic',
        name='ppo_walk1',
        config=config.__dict__
    )

    # create parallel environments
    def make_env():
        return LocoMimicEnv(config.motion_path)
    
    n_envs = getattr(config, 'n_envs', 1)
    if n_envs > 1:
        env = gym.vector.AsyncVectorEnv([make_env for _ in range(n_envs)])
    else:
        env = gym.vector.SyncVectorEnv([make_env])

    # Agent initialized with batched dimensions
    agent = PPOAgent(obs_dim=139, act_dim=29, config=config)

    # create directories
    os.makedirs('models/ppo', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print(f'Device: {config.device}')
    print(f'Training for {config.total_steps} steps')
    print(f'Envs: {n_envs}')
    print(f'Steps per env per rollout: {agent.steps_per_env}')
    print(f'Batch size: {config.batch_size}')
    print(f'N epochs: {config.n_epochs}')

    obs, _ = env.reset()
    episode_returns = np.zeros(n_envs, dtype=np.float32)
    episode_steps = np.zeros(n_envs, dtype=np.int32)
    episode_num = 0
    global_step = 0

    while global_step < config.total_steps:

        # Collect rollout for n_envs simultaneously
        for rollout_step in range(agent.steps_per_env):
            # Select action
            action, log_prob, value = agent.select_action(obs)

            # Step environment
            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated | truncated

            # Store in agent (use the original obs)
            agent.collect(obs, action, reward, done, log_prob, value)

            # Update metrics per env
            obs = obs_next
            episode_returns += reward
            episode_steps += 1
            global_step += n_envs

            for i in range(n_envs):
                if done[i]:
                    wandb.log({
                        'episode_return': episode_returns[i],
                        'episode_steps': episode_steps[i],
                        'episode': episode_num,
                        'step': global_step,
                    })

                    if episode_num % 100 == 0:
                        print(f'Step {global_step:8d} | Episode {episode_num:5d} | '
                              f'Return {episode_returns[i]:8.4f} | Steps {episode_steps[i]:4d}')

                    # reset env-specific trackers (gym vector env auto-resets the physical env logic)
                    episode_returns[i] = 0.0
                    episode_steps[i] = 0
                    episode_num += 1

        # PPO Update (pass the current obs for GAE bootstrap)
        losses = agent.update(obs)

        if losses:
            wandb.log({
                'policy_loss': losses['policy_loss'],
                'value_loss': losses['value_loss'],
                'entropy': losses['entropy'],
                'step': global_step,
            })

        # Checkpoints
        if global_step % config.save_freq < (agent.steps_per_env * n_envs):
            agent.save(f'models/ppo/ppo_step_{global_step}.pt')
            print(f'Saved checkpoint at step {global_step}')

    agent.save('models/ppo/ppo_final.pt')
    wandb.finish()
    print('Training complete.')

if __name__ == '__main__':
    train()
