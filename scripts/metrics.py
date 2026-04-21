"""
Print eval metrics for a trained policy.

Usage:
    python -m scripts.metrics models/sac_v3/final.pt --algo sac
    python -m scripts.metrics models/ppo_v1/final.pt --algo ppo --episodes 100
"""

import argparse
import numpy as np
from env.locomimic_env import LocoMimicEnv
from train.agents.sac.sac_agent import SACAgent
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_config, load_ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--algo', type=str, choices=['sac', 'ppo'], default='sac')
    parser.add_argument('--episodes', type=int, default=100)
    args = parser.parse_args()

    if args.algo == 'ppo':
        config = load_ppo_config('train/configs/ppo_config.yaml')
        agent = PPOAgent(obs_dim=139, act_dim=29, config=config)
    else:
        config = load_config('train/configs/sac_config.yaml')
        agent = SACAgent(obs_dim=139, act_dim=29, config=config)

    env = LocoMimicEnv(config.motion_path)
    agent.load(args.checkpoint)

    returns, lengths = [], []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done, ep_ret, steps = False, 0.0, 0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            if args.algo == 'ppo':
                action = action[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            steps += 1
            done = terminated or truncated
        returns.append(ep_ret)
        lengths.append(steps)

    env.close()

    returns = np.array(returns)
    lengths = np.array(lengths)
    survival = (lengths >= 1000).sum() / len(lengths) * 100

    print(f'\n=== {args.algo.upper()} Metrics ({args.episodes} episodes) ===')
    print(f'Checkpoint: {args.checkpoint}')
    print(f'')
    print(f'Return   — Mean: {returns.mean():.1f}  Std: {returns.std():.1f}  Min: {returns.min():.1f}  Max: {returns.max():.1f}')
    print(f'Steps    — Mean: {lengths.mean():.1f}  Std: {lengths.std():.1f}  Min: {lengths.min()}  Max: {lengths.max()}')
    print(f'Survival — {survival:.0f}% reached 1000 steps ({(lengths >= 1000).sum()}/{len(lengths)})')
    print(f'Median steps: {np.median(lengths):.0f}')
    print(f'')

    # Distribution of episode lengths
    buckets = [(0, 50), (50, 100), (100, 250), (250, 500), (500, 1000)]
    print('Episode length distribution:')
    for lo, hi in buckets:
        count = ((lengths >= lo) & (lengths < hi)).sum()
        pct = count / len(lengths) * 100
        bar = '#' * int(pct / 2)
        print(f'  {lo:4d}-{hi:4d}: {count:3d} ({pct:5.1f}%) {bar}')
    count = (lengths >= 1000).sum()
    pct = count / len(lengths) * 100
    bar = '#' * int(pct / 2)
    print(f'  1000+   : {count:3d} ({pct:5.1f}%) {bar}')


if __name__ == '__main__':
    main()
