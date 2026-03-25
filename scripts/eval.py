import argparse
import os
import numpy as np
from env.locomimic_env import LocoMimicEnv
from train.agents.sac.sac_agent import SACAgent
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_config, load_ppo_config


def run_policy(env, agent, algo, episodes, deterministic=False, render=False):
    """Run a trained policy and return per-episode returns and lengths."""
    ep_returns = []
    ep_lengths = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done:
            action = agent.select_action(obs, deterministic=deterministic)
            if algo == 'ppo':
                action = action[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            ep_return += reward
            steps += 1
            done = terminated or truncated
        ep_returns.append(ep_return)
        ep_lengths.append(steps)
        print(f'  [{algo}] Episode {ep+1:3d} | Return: {ep_return:8.2f} | Steps: {steps:4d}')
    return np.array(ep_returns), np.array(ep_lengths)


def run_random(env, episodes):
    """Run random actions and return per-episode returns and lengths."""
    ep_returns = []
    ep_lengths = []
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            steps += 1
            done = terminated or truncated
        ep_returns.append(ep_return)
        ep_lengths.append(steps)
        print(f'  [random] Episode {ep+1:3d} | Return: {ep_return:8.2f} | Steps: {steps:4d}')
    return np.array(ep_returns), np.array(ep_lengths)


def print_stats(label, returns, lengths):
    print(f'\n{label} over {len(returns)} episodes:')
    print(f'  Return  — Mean: {returns.mean():.2f}  Std: {returns.std():.2f}  Min: {returns.min():.2f}  Max: {returns.max():.2f}')
    print(f'  Length  — Mean: {lengths.mean():.1f}  Std: {lengths.std():.1f}  Min: {lengths.min()}  Max: {lengths.max()}')


def save_plots(results, name):
    """Save comparison bar plots and per-episode curves."""
    import matplotlib.pyplot as plt

    os.makedirs('results/plots', exist_ok=True)
    labels = list(results.keys())
    colors = {'sac': '#2196F3', 'ppo': '#4CAF50', 'random': '#9E9E9E'}

    # --- Bar chart: mean return and length ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    mean_returns = [results[l]['returns'].mean() for l in labels]
    std_returns = [results[l]['returns'].std() for l in labels]
    bar_colors = [colors.get(l, '#FF9800') for l in labels]

    axes[0].bar(labels, mean_returns, yerr=std_returns, color=bar_colors, capsize=5)
    axes[0].set_ylabel('Return')
    axes[0].set_title('Mean Episode Return')

    mean_lengths = [results[l]['lengths'].mean() for l in labels]
    std_lengths = [results[l]['lengths'].std() for l in labels]
    axes[1].bar(labels, mean_lengths, yerr=std_lengths, color=bar_colors, capsize=5)
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Mean Episode Length')

    plt.tight_layout()
    bar_path = f'results/plots/{name}_bar.png'
    fig.savefig(bar_path, dpi=150)
    plt.close(fig)
    print(f'Saved {bar_path}')

    # --- Per-episode curves ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for l in labels:
        c = colors.get(l, '#FF9800')
        episodes = np.arange(1, len(results[l]['returns']) + 1)
        axes[0].plot(episodes, results[l]['returns'], label=l, color=c, alpha=0.8)
        axes[1].plot(episodes, results[l]['lengths'], label=l, color=c, alpha=0.8)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Return')
    axes[0].set_title('Per-Episode Return')
    axes[0].legend()

    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Per-Episode Length')
    axes[1].legend()

    plt.tight_layout()
    curve_path = f'results/plots/{name}_curves.png'
    fig.savefig(curve_path, dpi=150)
    plt.close(fig)
    print(f'Saved {curve_path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str, help='Path to trained policy checkpoint (.pt)')
    parser.add_argument('--algo', type=str, choices=['sac', 'ppo'], default='sac')
    parser.add_argument('--config', type=str, default=None, help='Override default config path')
    parser.add_argument('--episodes', type=int, default=100)
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--deterministic', action='store_true', help='Use mean action (no sampling)')
    parser.add_argument('--save-plots', action='store_true', help='Save comparison plots (only with --compare-random)')
    parser.add_argument('--compare-random', action='store_true', help='Also evaluate random actions')
    parser.add_argument('--name', type=str, default=None, help='Name for saved plots')
    args = parser.parse_args()

    if args.algo == 'ppo':
        config_path = args.config if args.config else 'train/configs/ppo_config.yaml'
        config = load_ppo_config(config_path)
    else:
        config_path = args.config if args.config else 'train/configs/sac_config.yaml'
        config = load_config(config_path)

    env = LocoMimicEnv(
        config.motion_path,
        config=config,
        render_mode='human' if args.render else None,
    )
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if args.algo == 'ppo':
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)
    else:
        agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)
    agent.load(args.checkpoint)

    print(f'Evaluating {args.algo}...')
    returns, lengths = run_policy(env, agent, args.algo, args.episodes,
                                  deterministic=args.deterministic, render=args.render)
    print_stats(args.algo, returns, lengths)

    results = {args.algo: {'returns': returns, 'lengths': lengths}}

    if args.compare_random:
        print(f'\nEvaluating random...')
        rand_returns, rand_lengths = run_random(env, args.episodes)
        print_stats('random', rand_returns, rand_lengths)
        results['random'] = {'returns': rand_returns, 'lengths': rand_lengths}

    if args.save_plots and args.compare_random:
        plot_name = args.name or f'eval_{args.algo}_vs_random'
        save_plots(results, plot_name)

    env.close()


if __name__ == '__main__':
    main()
