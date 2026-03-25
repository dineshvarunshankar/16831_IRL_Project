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
parser.add_argument('--deterministic', action='store_true', help='Force deterministic (mean) actions')
parser.add_argument('--stochastic', action='store_true', help='Force stochastic action sampling')
parser.add_argument(
    '--reset-phase-start',
    type=float,
    default=None,
    help='Override reset phase start in [0, 1].',
)
parser.add_argument(
    '--reset-phase-end',
    type=float,
    default=None,
    help='Override reset phase end in [0, 1].',
)
args = parser.parse_args()

if args.algo == 'ppo':
    config_path = args.config if args.config else 'train/configs/ppo_config.yaml'
    config = load_ppo_config(config_path)
else:
    config_path = args.config if args.config else 'train/configs/sac_config.yaml'
    config = load_config(config_path)

if args.algo == 'ppo':
    if args.reset_phase_start is not None:
        config.reset_phase_start = float(args.reset_phase_start)
    if args.reset_phase_end is not None:
        config.reset_phase_end = float(args.reset_phase_end)

env = LocoMimicEnv(
    config.motion_path,
    config=config if args.algo == 'ppo' else None,
    render_mode='human' if args.render else None,
)
if args.algo == 'ppo':
    config.n_envs = 1
    agent = PPOAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        config=config,
    )
else:
    agent = SACAgent(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        config=config,
    )
agent.load(args.checkpoint)

deterministic_default = bool(getattr(config, "eval_deterministic_default", True))
if args.deterministic and args.stochastic:
    raise ValueError("Use only one of --deterministic or --stochastic")
if args.deterministic:
    deterministic_eval = True
elif args.stochastic:
    deterministic_eval = False
else:
    deterministic_eval = deterministic_default

if args.algo == 'ppo':
    print(
        f"Eval reset phase range: [{config.reset_phase_start:.3f}, {config.reset_phase_end:.3f}]"
    )
print(f"Deterministic eval: {deterministic_eval}")

episode_returns = []
episode_lengths = []

for ep in range(args.episodes):
    obs, _ = env.reset()
    done = False
    episode_return = 0.0
    steps = 0

    while not done:
        if args.algo == 'ppo':
            action, _, _ = agent.select_action(obs, deterministic=deterministic_eval)
        else:
            action = agent.select_action(obs, deterministic=deterministic_eval)
            
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
