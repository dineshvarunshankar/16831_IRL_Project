"""
PPO training script for motion imitation.
"""

import argparse
import os
import random
import re

import gymnasium as gym
import numpy as np
import torch
import wandb

from env.locomimic_env import LocoMimicEnv
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_ppo_config


def _override_config(config, args):
    if args.total_steps is not None:
        config.total_steps = int(args.total_steps)
    if args.n_envs is not None:
        config.n_envs = int(args.n_envs)
    if args.rollout_steps is not None:
        config.rollout_steps = int(args.rollout_steps)
    if args.batch_size is not None:
        config.batch_size = int(args.batch_size)
    if args.save_freq is not None:
        config.save_freq = int(args.save_freq)
    if args.seed is not None:
        config.seed = int(args.seed)
    if args.vector_env_type is not None:
        config.vector_env_type = str(args.vector_env_type)
    if args.vector_env_context is not None:
        config.vector_env_context = str(args.vector_env_context)
    if args.torch_num_threads is not None:
        config.torch_num_threads = int(args.torch_num_threads)
    if args.torch_num_interop_threads is not None:
        config.torch_num_interop_threads = int(args.torch_num_interop_threads)


def _infer_step_from_checkpoint(path: str) -> int:
    filename = os.path.basename(path)
    match = re.search(r"ppo_step_(\d+)\.pt$", filename)
    if match is None:
        return 0
    return int(match.group(1))


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_torch_threads(num_threads: int, num_interop_threads: int):
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(max(1, int(num_interop_threads)))
    except RuntimeError:
        # Can only be set once per process. Ignore if already initialized.
        pass


def _make_env_factory(config, base_seed: int, env_rank: int):
    def _thunk():
        env = LocoMimicEnv(config.motion_path, config=config)
        env.reset(seed=base_seed + env_rank)
        return env

    return _thunk


def train(config_path="train/configs/ppo_config.yaml", args=None):
    config = load_ppo_config(config_path)
    if args is not None:
        _override_config(config, args)

    curriculum_mode = str(getattr(config, "curriculum_mode", "off")).lower()
    curriculum_update_freq = int(getattr(config, "curriculum_update_freq", 100000))
    run_name = args.run_name if args and args.run_name else "ppo_walk1"
    seed = int(getattr(config, "seed", 42))
    resume_checkpoint = args.resume_checkpoint if args else None
    if args and args.start_step is not None:
        start_step = int(args.start_step)
    elif resume_checkpoint:
        start_step = _infer_step_from_checkpoint(resume_checkpoint)
    else:
        start_step = 0

    _set_seeds(seed)
    _set_torch_threads(
        int(getattr(config, "torch_num_threads", 1)),
        int(getattr(config, "torch_num_interop_threads", 1)),
    )

    n_envs = int(getattr(config, "n_envs", 1))
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")
    if int(config.rollout_steps) % n_envs != 0:
        raise ValueError(
            f"rollout_steps ({config.rollout_steps}) must be divisible by n_envs ({n_envs})"
        )

    vector_env_type = str(getattr(config, "vector_env_type", "async")).lower()
    vector_env_context = str(getattr(config, "vector_env_context", "spawn")).lower()
    if vector_env_type not in {"async", "sync"}:
        raise ValueError("vector_env_type must be 'async' or 'sync'")
    valid_contexts = {"spawn", "fork", "forkserver"}
    if vector_env_context not in valid_contexts:
        raise ValueError(f"vector_env_context must be one of {sorted(valid_contexts)}")

    wandb.init(project="locomimic", name=run_name, config=config.__dict__)

    env_fns = [_make_env_factory(config, seed, i) for i in range(n_envs)]
    if n_envs == 1 or vector_env_type == "sync":
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        try:
            env = gym.vector.AsyncVectorEnv(env_fns, context=vector_env_context)
        except TypeError:
            env = gym.vector.AsyncVectorEnv(env_fns)

    try:
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)
        if resume_checkpoint:
            agent.load(resume_checkpoint)
            print(f"Loaded checkpoint: {resume_checkpoint}")
            print(f"Resume step: {start_step}")

        os.makedirs("models/ppo", exist_ok=True)
        os.makedirs("logs", exist_ok=True)

        print(f"Device: {config.device}")
        print(f"Training for {config.total_steps} steps")
        print(f"Run name: {run_name}")
        print(f"Seed: {seed}")
        print(f"Envs: {n_envs}")
        print(f"Vector env: {vector_env_type} ({vector_env_context})")
        print(
            f"Torch threads: intra-op={torch.get_num_threads()}, "
            f"interop={getattr(config, 'torch_num_interop_threads', 1)}"
        )
        print(f"Obs dim: {obs_dim}")
        print(f"Act dim: {act_dim}")
        print(f"Steps per env per rollout: {agent.steps_per_env}")
        print(f"Batch size: {config.batch_size}")
        print(f"N epochs: {config.n_epochs}")

        obs, _ = env.reset(seed=seed)
        episode_returns = np.zeros(n_envs, dtype=np.float32)
        episode_steps = np.zeros(n_envs, dtype=np.int32)
        episode_num = 0
        global_step = start_step

        while global_step < config.total_steps:
            for _ in range(agent.steps_per_env):
                action, log_prob, value = agent.select_action(obs)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                done = terminated | truncated
                agent.collect(obs, action, reward, done, log_prob, value)

                obs = obs_next
                episode_returns += reward
                episode_steps += 1
                global_step += n_envs

                for i in range(n_envs):
                    if done[i]:
                        wandb.log(
                            {
                                "episode_return": episode_returns[i],
                                "episode_steps": episode_steps[i],
                                "episode": episode_num,
                                "step": global_step,
                            }
                        )
                        if episode_num % 100 == 0:
                            print(
                                f"Step {global_step:8d} | Episode {episode_num:5d} | "
                                f"Return {episode_returns[i]:8.4f} | Steps {episode_steps[i]:4d}"
                            )
                        episode_returns[i] = 0.0
                        episode_steps[i] = 0
                        episode_num += 1

            losses = agent.update(obs)
            if losses:
                wandb.log(
                    {
                        "policy_loss": losses["policy_loss"],
                        "value_loss": losses["value_loss"],
                        "entropy": losses["entropy"],
                        "residual_reg": losses.get("residual_reg", 0.0),
                        "approx_kl": losses.get("approx_kl", 0.0),
                        "early_stopped": losses.get("early_stopped", 0.0),
                        "step": global_step,
                    }
                )

            should_update_curriculum = (
                curriculum_mode == "linear"
                and global_step % curriculum_update_freq < (agent.steps_per_env * n_envs)
            )
            if should_update_curriculum:
                progress = min(global_step / config.total_steps, 1.0)
                env.call("update_curriculum", progress)
                print(f"Curriculum updated at step {global_step}: progress={progress:.3f}")

            if global_step % config.save_freq < (agent.steps_per_env * n_envs):
                agent.save(f"models/ppo/ppo_step_{global_step}.pt")
                print(f"Saved checkpoint at step {global_step}")

        agent.save("models/ppo/ppo_final.pt")
        print("Training complete.")
    finally:
        env.close()
        wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/ppo_config.yaml",
        help="Path to PPO config yaml.",
    )
    parser.add_argument("--run-name", type=str, default="ppo_walk1")
    parser.add_argument("--total-steps", type=int, default=None)
    parser.add_argument("--n-envs", type=int, default=None)
    parser.add_argument("--rollout-steps", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--save-freq", type=int, default=None)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument(
        "--start-step",
        type=int,
        default=None,
        help="Optional manual override for global step when resuming.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--vector-env-type",
        type=str,
        choices=["async", "sync"],
        default=None,
        help="Use async (multiprocess) or sync vector env.",
    )
    parser.add_argument(
        "--vector-env-context",
        type=str,
        choices=["spawn", "fork", "forkserver"],
        default=None,
        help="Multiprocessing context for AsyncVectorEnv.",
    )
    parser.add_argument("--torch-num-threads", type=int, default=None)
    parser.add_argument("--torch-num-interop-threads", type=int, default=None)
    cli_args = parser.parse_args()
    train(config_path=cli_args.config, args=cli_args)
