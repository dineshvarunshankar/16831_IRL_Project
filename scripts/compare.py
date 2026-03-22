"""
Compare two policies side by side.

Usage:
    python -m scripts.compare                                          # sac_final vs random
    python -m scripts.compare --model_a models/sac_step_500000.pt --model_b models/sac_step_100000.pt
    python -m scripts.compare --model_a models/sac_final.pt --model_b random --episodes 50 --render
"""

import argparse
import numpy as np
from env.locomimic_env import LocoMimicEnv
from train.agents.sac.sac_agent import SACAgent
from train.configs.config_loader import load_config


def rollout(env, get_action, episodes, render=False):
    returns = []
    lengths = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0
        while not done:
            action = get_action(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            if render:
                env.render()
            ep_return += reward
            steps += 1
            done = terminated or truncated
        returns.append(ep_return)
        lengths.append(steps)
    return np.array(returns), np.array(lengths)


def load_policy(checkpoint, config, env):
    if checkpoint == "random":
        return lambda obs: env.action_space.sample(), "Random"
    agent = SACAgent(obs_dim=139, act_dim=29, config=config)
    agent.load(checkpoint)
    return lambda obs: agent.select_action(obs, deterministic=True), checkpoint


def print_stats(name, returns, lengths):
    print(f"  {name}")
    print(f"    Return — Mean: {returns.mean():8.2f}  Std: {returns.std():7.2f}  Min: {returns.min():8.2f}  Max: {returns.max():8.2f}")
    print(f"    Length — Mean: {lengths.mean():8.1f}  Std: {lengths.std():7.1f}  Min: {lengths.min():8d}  Max: {lengths.max():8d}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_a", type=str, default="models/sac_final.pt")
    parser.add_argument("--model_b", type=str, default="random")
    parser.add_argument("--config", type=str, default="train/configs/sac_config.yaml")
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    env = LocoMimicEnv(config.motion_path, render_mode="human" if args.render else None)

    policy_a, name_a = load_policy(args.model_a, config, env)
    policy_b, name_b = load_policy(args.model_b, config, env)

    print(f"\nEvaluating {args.episodes} episodes each...\n")

    returns_a, lengths_a = rollout(env, policy_a, args.episodes, args.render)
    returns_b, lengths_b = rollout(env, policy_b, args.episodes, args.render)

    print_stats(name_a, returns_a, lengths_a)
    print()
    print_stats(name_b, returns_b, lengths_b)

    # comparison
    diff = returns_a.mean() - returns_b.mean()
    print(f"\n  Delta (A - B): {diff:+.2f} mean return")
    print(f"  A survives {lengths_a.mean() / max(lengths_b.mean(), 1):.1f}x longer on average")

    env.close()


if __name__ == "__main__":
    main()
