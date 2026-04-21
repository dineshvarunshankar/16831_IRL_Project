"""
Evaluate and visualize trained policies with Unitree RL mjlab.

This script loads a trained policy and evaluates it in the environment,
optionally rendering the results.

Usage:
    # Evaluate SAC policy
    python scripts/play.py --checkpoint models/sac/final.pt --algo sac
    
    # Evaluate with rendering
    python scripts/play.py --checkpoint models/ppo/final.pt --algo ppo --render
    
    # Evaluate specific motion
    python scripts/play.py --checkpoint models/sac/final.pt --algo sac \
        --motion-file data/motions_npz/walk1.npz
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np

# Import from Unitree RL mjlab
try:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
except ImportError:
    print("Error: Cannot import from mjlab!")
    print("Make sure unitree_rl_mjlab is installed:")
    print("  cd unitree_rl_mjlab && pip install -e .")
    sys.exit(1)


def load_algorithm(algo: str, checkpoint_path: str, obs_dim: int, act_dim: int):
    """
    Load trained algorithm from checkpoint.
    
    Args:
        algo: Algorithm name (ppo, sac, tdmpc2)
        checkpoint_path: Path to checkpoint file
        obs_dim: Observation dimension
        act_dim: Action dimension
    
    Returns:
        Loaded algorithm instance
    """
    if algo == "ppo":
        from algorithms.ppo import PPOAgent
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim)
    elif algo == "sac":
        from algorithms.sac import SACAgent
        agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim)
    elif algo == "tdmpc2":
        from algorithms.tdmpc2 import TDMPC2Agent
        agent = TDMPC2Agent(obs_dim=obs_dim, act_dim=act_dim)
    else:
        raise ValueError(f"Unknown algorithm: {algo}")
    
    agent.load(checkpoint_path)
    return agent


def evaluate(
    agent,
    env,
    num_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False
):
    """
    Evaluate agent in environment.
    
    Args:
        agent: Trained algorithm
        env: Environment
        num_episodes: Number of episodes to evaluate
        deterministic: Use deterministic actions
        render: Render episodes
    
    Returns:
        Dictionary of evaluation metrics
    """
    episode_returns = []
    episode_lengths = []
    
    for episode in range(num_episodes):
        obs = env.reset()
        episode_return = 0.0
        episode_length = 0
        done = False
        
        while not done:
            # Select action
            with torch.no_grad():
                actions = agent.select_action(obs, deterministic=deterministic)
            
            # Step environment
            obs, rewards, dones, infos = env.step(actions)
            
            # Accumulate metrics (use first env)
            episode_return += rewards[0].item()
            episode_length += 1
            done = dones[0].item()
            
            if render:
                env.render()
        
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        
        print(f"Episode {episode + 1}/{num_episodes}: "
              f"Return = {episode_return:.2f}, Length = {episode_length}")
    
    # Compute statistics
    metrics = {
        "mean_return": np.mean(episode_returns),
        "std_return": np.std(episode_returns),
        "mean_length": np.mean(episode_lengths),
        "std_length": np.std(episode_lengths),
        "min_return": np.min(episode_returns),
        "max_return": np.max(episode_returns),
    }
    
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained policy with Unitree RL mjlab"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to checkpoint file"
    )
    parser.add_argument(
        "--algo",
        type=str,
        required=True,
        choices=["ppo", "sac", "tdmpc2"],
        help="Algorithm type"
    )
    parser.add_argument(
        "--task-id",
        type=str,
        default="Unitree-G1-Tracking-No-State-Estimation",
        help="Task ID (default: Unitree-G1-Tracking-No-State-Estimation)"
    )
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Path to NPZ motion file"
    )
    parser.add_argument(
        "--num-episodes",
        type=int,
        default=10,
        help="Number of evaluation episodes (default: 10)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions"
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render episodes"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device (cuda or cpu)"
    )
    args = parser.parse_args()
    
    # Check checkpoint exists
    if not Path(args.checkpoint).exists():
        print(f"Error: Checkpoint not found: {args.checkpoint}")
        sys.exit(1)
    
    # Load environment config
    print(f"Loading environment: {args.task_id}")
    env_cfg = load_env_cfg(args.task_id)
    
    # Set motion file if provided
    if args.motion_file:
        if not Path(args.motion_file).exists():
            print(f"Error: Motion file not found: {args.motion_file}")
            sys.exit(1)
        env_cfg.commands["motion"].motion_file = args.motion_file
        print(f"Using motion file: {args.motion_file}")
    
    # Create environment (single env for evaluation)
    env = ManagerBasedRlEnv(
        cfg=env_cfg,
        device=args.device,
        render_mode="rgb_array" if args.render else None
    )
    env = RslRlVecEnvWrapper(env, clip_actions=True)
    
    # Get dimensions
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {act_dim}")
    
    # Load algorithm
    print(f"Loading {args.algo.upper()} checkpoint: {args.checkpoint}")
    agent = load_algorithm(args.algo, args.checkpoint, obs_dim, act_dim)
    
    # Evaluate
    print(f"\nEvaluating for {args.num_episodes} episodes...")
    print("=" * 60)
    
    metrics = evaluate(
        agent=agent,
        env=env,
        num_episodes=args.num_episodes,
        deterministic=args.deterministic,
        render=args.render
    )
    
    # Print results
    print("=" * 60)
    print("\nEvaluation Results:")
    print(f"  Mean Return: {metrics['mean_return']:.2f} ± {metrics['std_return']:.2f}")
    print(f"  Mean Length: {metrics['mean_length']:.1f} ± {metrics['std_length']:.1f}")
    print(f"  Min Return:  {metrics['min_return']:.2f}")
    print(f"  Max Return:  {metrics['max_return']:.2f}")
    
    env.close()


if __name__ == "__main__":
    main()
