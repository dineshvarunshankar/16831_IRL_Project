"""
Sanity check: train SAC on standard Gymnasium envs to verify
the training pipeline works independently of LocoMimicEnv.

Usage:
    python -m tests.test_sac_gym                        # Pendulum (quick)
    python -m tests.test_sac_gym --env HalfCheetah-v5   # harder, needs mujoco
    python -m tests.test_sac_gym --env Humanoid-v5 --steps 500000
"""

import argparse
import numpy as np
import gymnasium as gym
from train.agents.sac.sac_agent import SACAgent
from train.configs.config_loader import load_config, SACConfig

def make_config(obs_dim, act_dim, args):
    return SACConfig(
        motion_path="",
        actor_hidden_dim=256,
        critic_hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        tau=0.005,
        batch_size=256,
        buffer_size=100000,
        total_steps=args.steps,
        learning_starts=1000,
        gradient_steps=args.gradient_steps,
        log_freq=1000,
        save_freq=50000,
        device='cpu',
    )

def train(args):
    env = gym.make(args.env)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    act_limit = env.action_space.high[0]

    config = make_config(obs_dim, act_dim, args)
    agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)

    print(f'Env: {args.env} | obs: {obs_dim} | act: {act_dim} | steps: {args.steps} | grad_steps: {args.gradient_steps}')

    obs, _ = env.reset()
    episode_return = 0.0
    episode_steps = 0
    episode_num = 0
    recent_returns = []

    for step in range(args.steps):
        if step < config.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        # scale action to env range (SAC outputs [-1, 1])
        env_action = action * act_limit

        obs_next, reward, terminated, truncated, _ = env.step(env_action)
        done = terminated or truncated

        agent.collect(obs, action, reward, obs_next, done)
        obs = obs_next
        episode_return += reward
        episode_steps += 1

        if step >= config.learning_starts:
            agent.update()

        if done:
            recent_returns.append(episode_return)
            if episode_num % 20 == 0:
                avg = np.mean(recent_returns[-20:])
                print(f'Step {step:7d} | Ep {episode_num:4d} | Return {episode_return:8.2f} | Avg20 {avg:8.2f} | Steps {episode_steps}')
            obs, _ = env.reset()
            episode_return = 0.0
            episode_steps = 0
            episode_num += 1

    env.close()
    returns = np.array(recent_returns)
    print(f'\nDone. {episode_num} episodes.')
    print(f'Last 20 avg: {returns[-20:].mean():.2f}')
    print(f'Best 20 avg: {max(np.convolve(returns, np.ones(20)/20, mode="valid")):.2f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='Pendulum-v1')
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--gradient_steps', type=int, default=8)
    args = parser.parse_args()
    train(args)
