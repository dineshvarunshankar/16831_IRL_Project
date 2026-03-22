import argparse
import numpy as np
import torch
import wandb
import os
from datetime import datetime
from env.locomimic_env import LocoMimicEnv
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_ppo_config


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--config', type=str, default='train/configs/ppo_config.yaml')
    parser.add_argument('--load', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    config = load_ppo_config(args.config)

    # experiment name and per-run directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.name or f'ppo_{timestamp}'
    run_model_dir = f'models/{run_name}'
    run_log_dir = f'logs/{run_name}'
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    wandb.init(
        project='locomimic',
        name=run_name,
        config=config.__dict__
    )

    # parallel environments
    envs = [LocoMimicEnv(config.motion_path) for _ in range(config.n_envs)]
    agent = PPOAgent(obs_dim=139, act_dim=29, config=config)

    if args.load:
        agent.load(args.load)
        print(f'Loaded checkpoint: {args.load}')

    print(f'Run      : {run_name}')
    print(f'Device   : {config.device}')
    print(f'Steps    : {config.total_steps}')
    print(f'N envs   : {config.n_envs}')
    print(f'Rollout  : {config.rollout_steps}')

    # per-env state
    obs_list = []
    for env in envs:
        o, _ = env.reset()
        obs_list.append(o)

    ep_returns = [0.0] * config.n_envs
    ep_steps = [0] * config.n_envs
    episode_num = 0
    global_step = 0

    # checkpoint at 25%, 50%, 75%, and final
    ckpt_steps = {
        int(config.total_steps * 0.25),
        int(config.total_steps * 0.50),
        int(config.total_steps * 0.75),
    }

    recent_returns = []
    recent_lengths = []

    while global_step < config.total_steps:

        # collect rollout across all envs
        for rollout_step in range(config.rollout_steps):
            for i, env in enumerate(envs):
                action, log_prob, value = agent.select_action(obs_list[i])

                obs_next, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated

                agent.collect(obs_list[i], action, reward, done, log_prob, value)

                obs_list[i] = obs_next
                ep_returns[i] += reward
                ep_steps[i] += 1

                if done:
                    recent_returns.append(ep_returns[i])
                    recent_lengths.append(ep_steps[i])

                    log_data = {
                        'episode/return': ep_returns[i],
                        'episode/steps': ep_steps[i],
                        'episode/num': episode_num,
                        'episode/terminated': int(terminated),
                        'env_step': global_step,
                    }

                    if len(recent_returns) >= 100:
                        log_data['episode/avg_return_100'] = np.mean(recent_returns[-100:])
                        log_data['episode/avg_steps_100'] = np.mean(recent_lengths[-100:])

                    wandb.log(log_data, step=global_step)

                    if episode_num % 500 == 0:
                        avg_r = np.mean(recent_returns[-100:]) if len(recent_returns) >= 100 else np.mean(recent_returns)
                        avg_l = np.mean(recent_lengths[-100:]) if len(recent_lengths) >= 100 else np.mean(recent_lengths)
                        print(f'Step {global_step:8d} | Ep {episode_num:5d} | '
                              f'Return {ep_returns[i]:8.2f} | Steps {ep_steps[i]:4d} | '
                              f'Avg100 R={avg_r:7.2f} L={avg_l:5.1f}')

                    obs_list[i], _ = env.reset()
                    ep_returns[i] = 0.0
                    ep_steps[i] = 0
                    episode_num += 1

            global_step += 1

        # PPO update
        losses = agent.update()

        if losses:
            wandb.log({
                'agent/policy_loss': losses['policy_loss'],
                'agent/value_loss': losses['value_loss'],
                'agent/entropy': losses['entropy'],
            }, step=global_step)

        # checkpoints
        if any(global_step >= s and global_step - config.rollout_steps < s for s in ckpt_steps):
            path = f'{run_model_dir}/ckpt_step_{global_step}.pt'
            agent.save(path)
            print(f'Checkpoint saved: {path}')

    # final save
    final_path = f'{run_model_dir}/final.pt'
    agent.save(final_path)
    print(f'Final model saved: {final_path}')

    for env in envs:
        env.close()
    wandb.finish()
    print('Training complete.')


if __name__ == '__main__':
    train()
