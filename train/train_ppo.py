import argparse
import os
import random
from datetime import datetime

import gymnasium as gym
import numpy as np
import torch
import wandb

from env.locomimic_env import LocoMimicEnv
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_ppo_config


def _set_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _set_torch_threads(num_threads: int):
    torch.set_num_threads(max(1, int(num_threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _make_env_factory(motion_path: str, config, base_seed: int, env_rank: int):
    def _thunk():
        env = LocoMimicEnv(motion_path, config=config)
        env.reset(seed=base_seed + env_rank)
        return env

    return _thunk


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None, help="Experiment name")
    parser.add_argument(
        "--config", type=str, default="train/configs/ppo_config.yaml"
    )
    parser.add_argument(
        "--load", type=str, default=None, help="Path to checkpoint to resume from"
    )
    args = parser.parse_args()

    config = load_ppo_config(args.config)
    _set_seeds(int(config.seed))
    _set_torch_threads(int(config.torch_num_threads))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"ppo_{timestamp}"
    run_model_dir = f"models/{run_name}"
    run_log_dir = f"logs/{run_name}"
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    wandb.init(project="locomimic", name=run_name, config=config.__dict__)

    env_fns = [
        _make_env_factory(config.motion_path, config, int(config.seed), i)
        for i in range(int(config.n_envs))
    ]
    vector_type = str(getattr(config, "vector_env_type", "async")).lower()
    if int(config.n_envs) == 1 or vector_type == "sync":
        env = gym.vector.SyncVectorEnv(env_fns)
    else:
        env = gym.vector.AsyncVectorEnv(
            env_fns, context=str(getattr(config, "vector_env_context", "spawn")).lower()
        )

    try:
        log_freq = max(1, int(config.log_freq))
        obs_dim = env.single_observation_space.shape[0]
        act_dim = env.single_action_space.shape[0]
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)

        if args.load:
            agent.load(args.load)
            print(f"Loaded checkpoint: {args.load}")

        print(f"Run      : {run_name}")
        print(f"Device   : {config.device}")
        print(f"Steps    : {config.total_steps}")
        print(f"N envs   : {config.n_envs}")
        print(f"Vector   : {vector_type}")
        print(f"Obs dim  : {obs_dim}")
        print(f"Act dim  : {act_dim}")
        print(f"Rollout  : total={config.rollout_steps}, per_env={agent.steps_per_env}")
        print(f"Batch    : {config.batch_size}")
        print(f"Epochs   : {config.n_epochs}")

        obs, _ = env.reset(seed=int(config.seed))
        n_envs = int(config.n_envs)
        episode_return = np.zeros(n_envs, dtype=np.float32)
        episode_steps = np.zeros(n_envs, dtype=np.int32)
        episode_num = 0
        global_step = 0

        ckpt_steps = {
            int(config.total_steps * 0.25),
            int(config.total_steps * 0.50),
            int(config.total_steps * 0.75),
        }

        recent_returns = []
        recent_lengths = []

        while global_step < int(config.total_steps):
            for _ in range(agent.steps_per_env):
                action, log_prob, value = agent.select_action(obs)
                obs_next, reward, terminated, truncated, _ = env.step(action)
                done = np.logical_or(terminated, truncated)

                agent.collect(obs, action, reward, done, log_prob, value)

                obs = obs_next
                episode_return += reward
                episode_steps += 1
                global_step += n_envs

                for i in range(n_envs):
                    if done[i]:
                        recent_returns.append(float(episode_return[i]))
                        recent_lengths.append(int(episode_steps[i]))
                        log_data = {
                            "episode/return": float(episode_return[i]),
                            "episode/steps": int(episode_steps[i]),
                            "episode/num": episode_num,
                            "episode/terminated": int(terminated[i]),
                            "env_step": global_step,
                        }
                        if len(recent_returns) >= 100:
                            log_data["episode/avg_return_100"] = float(
                                np.mean(recent_returns[-100:])
                            )
                            log_data["episode/avg_steps_100"] = float(
                                np.mean(recent_lengths[-100:])
                            )
                        if global_step % log_freq < n_envs:
                            wandb.log(log_data, step=global_step)

                        if episode_num % 200 == 0:
                            avg_r = (
                                np.mean(recent_returns[-100:])
                                if len(recent_returns) >= 100
                                else np.mean(recent_returns)
                            )
                            avg_l = (
                                np.mean(recent_lengths[-100:])
                                if len(recent_lengths) >= 100
                                else np.mean(recent_lengths)
                            )
                            print(
                                f"Step {global_step:9d} | Ep {episode_num:6d} | "
                                f"Return {episode_return[i]:8.2f} | Steps {episode_steps[i]:4d} | "
                                f"Avg100 R={avg_r:7.2f} L={avg_l:5.1f}"
                            )

                        episode_return[i] = 0.0
                        episode_steps[i] = 0
                        episode_num += 1

            if bool(getattr(config, "lr_anneal", False)):
                progress = min(max(global_step / float(config.total_steps), 0.0), 1.0)
                current_lr = max(agent.base_lr * (1.0 - progress), 1e-8)
                agent.set_learning_rate(current_lr)
            else:
                current_lr = float(agent.optimizer.param_groups[0]["lr"])

            losses = agent.update(obs)
            if losses and (global_step % log_freq < n_envs):
                wandb.log(
                    {
                        "agent/policy_loss": losses["policy_loss"],
                        "agent/value_loss": losses["value_loss"],
                        "agent/entropy": losses["entropy"],
                        "agent/approx_kl": losses["approx_kl"],
                        "agent/clipfrac": losses["clipfrac"],
                        "agent/explained_variance": losses["explained_variance"],
                        "agent/early_stopped": losses["early_stopped"],
                        "agent/lr": current_lr,
                    },
                    step=global_step,
                )

            if any(
                global_step >= s and global_step - int(config.rollout_steps) < s
                for s in ckpt_steps
            ):
                path = f"{run_model_dir}/ckpt_step_{global_step}.pt"
                agent.save(path)
                print(f"Checkpoint saved: {path}")

            if global_step % int(config.save_freq) < int(config.rollout_steps):
                path = f"{run_model_dir}/ckpt_step_{global_step}.pt"
                agent.save(path)
                print(f"Checkpoint saved: {path}")

        final_path = f"{run_model_dir}/final.pt"
        agent.save(final_path)
        print(f"Final model saved: {final_path}")
        print("Training complete.")
    finally:
        env.close()
        wandb.finish()


if __name__ == "__main__":
    train()
