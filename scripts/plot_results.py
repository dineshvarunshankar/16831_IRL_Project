"""
Plot learning curves from training logs and eval performance comparison.

Usage:
    # Learning curves from logs
    python -m scripts.plot_results --logs logs/sac_v3/run.log logs/ppo_v1/run.log

    # Performance comparison (50 eps each of SAC, PPO, random)
    python -m scripts.plot_results --eval --sac-ckpt models/sac_v3/final.pt --ppo-ckpt models/ppo_v1/final.pt

    # Both
    python -m scripts.plot_results --logs logs/sac_v3/run.log logs/ppo_v1/run.log \
        --eval --sac-ckpt models/sac_v3/final.pt --ppo-ckpt models/ppo_v1/final.pt

    # Custom name
    python -m scripts.plot_results --logs logs/sac_v3/run.log --name my_experiment
"""

import argparse
import os
import re
import numpy as np
import matplotlib.pyplot as plt


def parse_log(log_path):
    """Parse a training log and extract (step, avg_return, avg_steps) tuples."""
    steps, returns, lengths = [], [], []
    with open(log_path) as f:
        for line in f:
            m = re.search(
                r'Step\s+(\d+)\s+\|.*Avg100 R=\s*([\d.]+)\s+L=\s*([\d.]+)',
                line
            )
            if m:
                steps.append(int(m.group(1)))
                returns.append(float(m.group(2)))
                lengths.append(float(m.group(3)))
    return np.array(steps), np.array(returns), np.array(lengths)


def infer_label(log_path):
    """Infer a label from the log path."""
    name = os.path.basename(os.path.dirname(log_path))
    if name in ('', '.', 'logs'):
        name = os.path.splitext(os.path.basename(log_path))[0]
    return name


def plot_learning_curves(log_paths, name):
    """Plot return and episode length vs training steps for multiple runs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ['#2196F3', '#4CAF50', '#FF9800', '#E91E63', '#9C27B0']

    for i, path in enumerate(log_paths):
        label = infer_label(path)
        steps, returns, lengths = parse_log(path)
        if len(steps) == 0:
            print(f'Warning: no data parsed from {path}')
            continue
        c = colors[i % len(colors)]
        axes[0].plot(steps, returns, label=label, color=c, alpha=0.85, linewidth=1.5)
        axes[1].plot(steps, lengths, label=label, color=c, alpha=0.85, linewidth=1.5)

    axes[0].set_xlabel('Environment Steps')
    axes[0].set_ylabel('Avg Return (100 ep)')
    axes[0].set_title('Learning Curve — Return')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel('Environment Steps')
    axes[1].set_ylabel('Avg Episode Length (100 ep)')
    axes[1].set_title('Learning Curve — Episode Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    out = f'results/plots/{name}_learning_curves.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out}')


def run_eval(env, agent, algo, episodes):
    """Run a policy for N episodes, return per-episode returns and lengths."""
    ep_returns, ep_lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ep_ret, steps = False, 0.0, 0
        while not done:
            action = agent.select_action(obs, deterministic=True)
            if algo == 'ppo':
                action = action[0]
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            steps += 1
            done = terminated or truncated
        ep_returns.append(ep_ret)
        ep_lengths.append(steps)
        if (ep + 1) % 10 == 0:
            print(f'  [{algo}] {ep+1}/{episodes} done')
    return np.array(ep_returns), np.array(ep_lengths)


def run_random_eval(env, episodes):
    ep_returns, ep_lengths = [], []
    for ep in range(episodes):
        obs, _ = env.reset()
        done, ep_ret, steps = False, 0.0, 0
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, _ = env.step(action)
            ep_ret += reward
            steps += 1
            done = terminated or truncated
        ep_returns.append(ep_ret)
        ep_lengths.append(steps)
        if (ep + 1) % 10 == 0:
            print(f'  [random] {ep+1}/{episodes} done')
    return np.array(ep_returns), np.array(ep_lengths)


def plot_performance(results, name):
    """Bar chart comparing eval performance of SAC, PPO, and random."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    colors = {'sac': '#2196F3', 'ppo': '#4CAF50', 'random': '#9E9E9E'}

    labels = list(results.keys())
    bar_colors = [colors.get(l, '#FF9800') for l in labels]

    mean_ret = [results[l]['returns'].mean() for l in labels]
    std_ret = [results[l]['returns'].std() for l in labels]
    axes[0].bar(labels, mean_ret, yerr=std_ret, color=bar_colors, capsize=5, edgecolor='black', linewidth=0.5)
    axes[0].set_ylabel('Return')
    axes[0].set_title('Eval Return (50 episodes)')
    axes[0].grid(True, axis='y', alpha=0.3)

    mean_len = [results[l]['lengths'].mean() for l in labels]
    std_len = [results[l]['lengths'].std() for l in labels]
    axes[1].bar(labels, mean_len, yerr=std_len, color=bar_colors, capsize=5, edgecolor='black', linewidth=0.5)
    axes[1].set_ylabel('Steps')
    axes[1].set_title('Eval Episode Length (50 episodes)')
    axes[1].grid(True, axis='y', alpha=0.3)

    # Add value labels on bars
    for ax, means in [(axes[0], mean_ret), (axes[1], mean_len)]:
        for j, v in enumerate(means):
            ax.text(j, v + ax.get_ylim()[1] * 0.02, f'{v:.1f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    os.makedirs('results/plots', exist_ok=True)
    out = f'results/plots/{name}_performance.png'
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f'Saved {out}')

    # Print stats
    for l in labels:
        r = results[l]['returns']
        s = results[l]['lengths']
        print(f'  {l:8s} — Return: {r.mean():.1f} +/- {r.std():.1f}  |  Steps: {s.mean():.1f} +/- {s.std():.1f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logs', nargs='+', help='Log files to plot learning curves from')
    parser.add_argument('--eval', action='store_true', help='Run eval performance comparison')
    parser.add_argument('--sac-ckpt', type=str, help='SAC checkpoint for eval')
    parser.add_argument('--ppo-ckpt', type=str, help='PPO checkpoint for eval')
    parser.add_argument('--episodes', type=int, default=50, help='Eval episodes per algo')
    parser.add_argument('--name', type=str, default='experiment', help='Name prefix for saved plots')
    args = parser.parse_args()

    if args.logs:
        plot_learning_curves(args.logs, args.name)

    if args.eval:
        from env.locomimic_env import LocoMimicEnv
        from train.agents.sac.sac_agent import SACAgent
        from train.agents.ppo.ppo_agent import PPOAgent
        from train.configs.config_loader import load_config, load_ppo_config

        sac_config = load_config('train/configs/sac_config.yaml')
        env = LocoMimicEnv(sac_config.motion_path)
        results = {}

        if args.sac_ckpt:
            print(f'Evaluating SAC: {args.sac_ckpt}')
            agent = SACAgent(obs_dim=139, act_dim=29, config=sac_config)
            agent.load(args.sac_ckpt)
            ret, leng = run_eval(env, agent, 'sac', args.episodes)
            results['sac'] = {'returns': ret, 'lengths': leng}

        if args.ppo_ckpt:
            print(f'Evaluating PPO: {args.ppo_ckpt}')
            ppo_config = load_ppo_config('train/configs/ppo_config.yaml')
            agent = PPOAgent(obs_dim=139, act_dim=29, config=ppo_config)
            agent.load(args.ppo_ckpt)
            ret, leng = run_eval(env, agent, 'ppo', args.episodes)
            results['ppo'] = {'returns': ret, 'lengths': leng}

        print('Evaluating random...')
        ret, leng = run_random_eval(env, args.episodes)
        results['random'] = {'returns': ret, 'lengths': leng}

        plot_performance(results, args.name)
        env.close()

    if not args.logs and not args.eval:
        parser.print_help()


if __name__ == '__main__':
    main()
