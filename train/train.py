import argparse
import numpy as np
import torch
import wandb
import os
from datetime import datetime
from env.locomimic_env import LocoMimicEnv
from train.agents.sac.sac_agent import SACAgent
from train.configs.config_loader import load_config

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default=None, help='Experiment name')
    parser.add_argument('--config', type=str, default='train/configs/sac_config.yaml')
    args = parser.parse_args()

    config = load_config(args.config)

    # experiment name and per-run directories
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = args.name or f'sac_{timestamp}'
    run_model_dir = f'models/{run_name}'
    run_log_dir = f'logs/{run_name}'
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    wandb.init(
        project='locomimic',
        name=run_name,
        config=config.__dict__
    )

    env = LocoMimicEnv(config.motion_path)
    agent = SACAgent(obs_dim=139, act_dim=29, config=config)

    print(f'Run    : {run_name}')
    print(f'Device : {config.device}')
    print(f'Steps  : {config.total_steps}')

    obs, _ = env.reset()
    episode_return = 0.0
    episode_steps = 0
    episode_num = 0

    # checkpoint at 25%, 50%, 75%, and final
    ckpt_steps = {
        int(config.total_steps * 0.25),
        int(config.total_steps * 0.50),
        int(config.total_steps * 0.75),
    }

    # rolling averages for logging
    recent_returns = []
    recent_lengths = []

    for step in range(config.total_steps):

        if step < config.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.collect(obs, action, reward, obs_next, done)

        obs = obs_next
        episode_return += reward
        episode_steps += 1

        if step >= config.learning_starts:
            agent.update()

        if done:
            recent_returns.append(episode_return)
            recent_lengths.append(episode_steps)

            log_data = {
                'episode/return': episode_return,
                'episode/steps': episode_steps,
                'episode/num': episode_num,
                'episode/terminated': int(terminated),
                'env_step': step,
            }

            # rolling averages (last 100 episodes)
            if len(recent_returns) >= 100:
                log_data['episode/avg_return_100'] = np.mean(recent_returns[-100:])
                log_data['episode/avg_steps_100'] = np.mean(recent_lengths[-100:])

            # agent stats (after learning starts)
            if step >= config.learning_starts:
                alpha = agent.log_alpha.exp().item()
                log_data['agent/alpha'] = alpha
                log_data['agent/log_alpha'] = agent.log_alpha.item()

            wandb.log(log_data, step=step)

            if episode_num % 500 == 0:
                avg_r = np.mean(recent_returns[-100:]) if len(recent_returns) >= 100 else np.mean(recent_returns)
                avg_l = np.mean(recent_lengths[-100:]) if len(recent_lengths) >= 100 else np.mean(recent_lengths)
                print(f'Step {step:7d} | Ep {episode_num:5d} | '
                      f'Return {episode_return:8.2f} | Steps {episode_steps:4d} | '
                      f'Avg100 R={avg_r:7.2f} L={avg_l:5.1f} | '
                      f'α={agent.log_alpha.exp().item():.4f}')

            obs, _ = env.reset()
            episode_return = 0.0
            episode_steps = 0
            episode_num += 1

        if step in ckpt_steps:
            path = f'{run_model_dir}/ckpt_step_{step}.pt'
            agent.save(path)
            print(f'Checkpoint saved: {path}')

    # final save
    final_path = f'{run_model_dir}/final.pt'
    agent.save(final_path)
    print(f'Final model saved: {final_path}')

    wandb.finish()
    print('Training complete.')

if __name__ == '__main__':
    train()
