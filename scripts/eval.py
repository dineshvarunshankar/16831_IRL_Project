import argparse
import numpy as np
from env.locomimic_env import LocoMimicEnv
from train.agents.sac.sac_agent import SACAgent
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_config, load_ppo_config

parser = argparse.ArgumentParser()
parser.add_argument('checkpoint', type=str, help='Path to trained policy checkpoint (.pt)')
parser.add_argument('--algo', type=str, choices=['sac', 'ppo'], default='ppo')
parser.add_argument('--config', type=str, default=None, help='Override default config path')
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--render', action='store_true')
parser.add_argument('--deterministic', action='store_true', help='Use mean action (no sampling)')
args = parser.parse_args()

if args.algo == 'ppo':
    config_path = args.config if args.config else 'train/configs/ppo_config.yaml'
    config = load_ppo_config(config_path)
    agent = PPOAgent(obs_dim=139, act_dim=29, config=config)
else:
    config_path = args.config if args.config else 'train/configs/sac_config.yaml'
    config = load_config(config_path)
    agent = SACAgent(obs_dim=139, act_dim=29, config=config)

env = LocoMimicEnv(config.motion_path, render_mode='human' if args.render else None)
agent.load(args.checkpoint)

episode_returns = []
episode_lengths = []

for ep in range(args.episodes):
    obs, _ = env.reset()
    done = False
    episode_return = 0.0
    steps = 0

    while not done:
        action = agent.select_action(obs, deterministic=args.deterministic)
        if args.algo == 'ppo':
            action = action[0]  # PPO returns (action, log_prob, value)
            
        obs, reward, terminated, truncated, _ = env.step(action)

        if args.render:
            env.render()

        episode_return += reward
        steps += 1
        done = terminated or truncated

    episode_returns.append(episode_return)
    episode_lengths.append(steps)
    print(f'Episode {ep+1:3d} | Return: {episode_return:8.2f} | Steps: {steps:4d}')

returns = np.array(episode_returns)
lengths = np.array(episode_lengths)
print()
print(f'Eval over {args.episodes} episodes:')
print(f'  Return  — Mean: {returns.mean():.2f}  Std: {returns.std():.2f}  Min: {returns.min():.2f}  Max: {returns.max():.2f}')
print(f'  Length  — Mean: {lengths.mean():.1f}  Std: {lengths.std():.1f}  Min: {lengths.min()}  Max: {lengths.max()}')

env.close()
