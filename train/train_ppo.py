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

    # create env and agent
    env = LocoMimicEnv(config.motion_path)
    agent = PPOAgent(obs_dim=139, act_dim=29, config=config)

    # create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)

    print(f'Device: {config.device}')
    print(f'Training for {config.total_steps} steps')
    print(f'Rollout steps: {config.rollout_steps}')
    print(f'Batch size: {config.batch_size}')
    print(f'N epochs: {config.n_epochs}')

    obs, _ = env.reset()
    episode_return = 0.0
    episode_steps = 0
    episode_num = 0
    global_step = 0

    while global_step < config.total_steps:

        # Collect rollout
        for rollout_step in range(config.rollout_steps):
            action, log_prob, value = agent.select_action(obs)

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.collect(obs, action, reward, done, log_prob, value)

            obs = obs_next
            episode_return += reward
            episode_steps += 1
            global_step += 1

            if done:
                wandb.log({
                    'episode_return': episode_return,
                    'episode_steps': episode_steps,
                    'episode': episode_num,
                    'step': global_step,
                })

                if episode_num % 100 == 0:
                    print(f'Step {global_step:8d} | Episode {episode_num:5d} | '
                          f'Return {episode_return:8.4f} | Steps {episode_steps:4d}')

                obs, _ = env.reset()
                episode_return = 0.0
                episode_steps = 0
                episode_num += 1

        # PPO Update
        losses = agent.update()

        if losses:
            wandb.log({
                'policy_loss': losses['policy_loss'],
                'value_loss': losses['value_loss'],
                'entropy': losses['entropy'],
                'step': global_step,
            })

        # Checkpoints
        if global_step % config.save_freq < config.rollout_steps:
            agent.save(f'models/ppo/ppo_step_{global_step}.pt')
            print(f'Saved checkpoint at step {global_step}')

        # Curriculum Update
        if global_step % 100000 < config.rollout_steps:
            progress = global_step / config.total_steps
            env.update_curriculum(progress)
            print(f'Curriculum updated: height_thresh={env.height_threshold:.3f}, '
                  f'ori_thresh={env.ori_threshold:.3f}')

    agent.save('models/ppo/ppo_final.pt')
    wandb.finish()
    print('Training complete.')


if __name__ == '__main__':
    train()
