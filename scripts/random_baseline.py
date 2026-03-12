import numpy as np
from env.locomimic_env import LocoMimicEnv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Render the environment')
parser.add_argument('--episodes', type=int, default=100)
args = parser.parse_args()

N_EPISODES = args.episodes

MOTION_PATH = './data/lafan1_retargeted/g1/walk1_subject1.csv'
N_EPISODES  = args.episodes

env = LocoMimicEnv(MOTION_PATH, render_mode='human' if args.render else None)

episode_returns = []

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    episode_return = 0.0

    while not done:
        
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if args.render:
            env.render()

        episode_return += reward
        done = terminated or truncated  

    episode_returns.append(episode_return)
    print(f'Episode {ep+1:3d} | return: {episode_return:.4f}')

episode_returns = np.array(episode_returns)
print()
print(f'Random baseline over {N_EPISODES} episodes:')
print(f'  Mean   : {episode_returns.mean():.4f}')
print(f'  Std    : {episode_returns.std():.4f}')
print(f'  Min    : {episode_returns.min():.4f}')
print(f'  Max    : {episode_returns.max():.4f}')

np.save('./logs/random_baseline.npy', episode_returns)
print()
print('Saved to logs/random_baseline.npy')

env.close()