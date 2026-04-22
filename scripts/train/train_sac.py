import argparse
import time
import torch
import wandb
import os
from collections import deque
from datetime import datetime

import mjlab.tasks

from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from rl.sac_env_wrapper import SACVecEnvWrapper
from agents.sac.sac_agent import SACAgent
from agents.configs.config_loader import load_config


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name',   type=str, default=None)
    parser.add_argument('--config', type=str, default='agents/configs/sac_config.yaml')
    parser.add_argument('--load',   type=str, default=None)
    parser.add_argument('--num_envs', type=int, default=4096)
    parser.add_argument('--task',   type=str, default='Unitree-G1-Tracking')
    parser.add_argument('--fast-sac', action='store_true')
    parser.add_argument("--iter", type=int,
    help="override config.num_learning_iterations; used for smoke tests"
    )
    args = parser.parse_args()

    config = load_config(args.config)
    config.num_envs = args.num_envs

    if args.iter is not None:
        config.num_learning_iterations = args.iter

    if args.fast_sac:
        config.use_layer_norm = True
        config.use_mean_q = True

    # --- EXPERIMENT SETUP ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name      = args.name or f'sac_{timestamp}'
    run_model_dir = f'models/{run_name}'
    run_log_dir   = f'logs/{run_name}'
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir,   exist_ok=True)

    wandb.init(
        project=config.wandb_project,
        name=run_name,
        config=config.__dict__
    )

    # --- BUILD ENV ---
    configure_torch_backends()

    env_cfg = load_env_cfg(args.task)

    motion_cmd = env_cfg.commands['motion']
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = config.motion_path

    env_cfg.auto_reset     = False
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed           = config.seed

    env = ManagerBasedRlEnv(cfg=env_cfg, device=config.device)
    env = SACVecEnvWrapper(env)

    print(f'actor_obs_dim : {env.actor_obs_dim}')
    print(f'critic_obs_dim: {env.critic_obs_dim}')
    print(f'act_dim       : {env.act_dim}')

    # --- INIT AGENT ---
    agent = SACAgent(
        actor_obs_dim  = env.actor_obs_dim,
        critic_obs_dim = env.critic_obs_dim,
        act_dim        = env.act_dim,
        config         = config,
    )

    if args.load:
        agent.load(args.load)
        print(f'Loaded checkpoint: {args.load}')

    print(f'Run    : {run_name}')
    print(f'Device : {config.device}')
    print(f'Envs   : {args.num_envs}')
    print(f'Iters  : {config.num_learning_iterations}')

    # --- INIT STATE ---
    actor_obs, critic_obs, _ = env.reset()

    episode_returns = torch.zeros(args.num_envs, device=config.device)
    episode_steps   = torch.zeros(args.num_envs, device=config.device)
    episode_num     = 0

    recent_returns = deque(maxlen=config.ep_stats_window)
    recent_lengths = deque(maxlen=config.ep_stats_window)

    # rolling per-step reward stats since last log
    reward_sum   = torch.zeros((), device=config.device)
    reward_sqsum = torch.zeros((), device=config.device)
    reward_count = 0
    last_log_time = time.time()

    ckpt_steps = {int(config.num_learning_iterations * f) for f in config.ckpt_fractions}

    # --- TRAINING LOOP ---
    for step in range(config.num_learning_iterations):

        # select action
        if step < config.learning_starts:
            action = (torch.rand(args.num_envs, env.act_dim, device=config.device) * 2 - 1)
        else:
            action = agent.select_action(actor_obs)

        # step env
        next_actor_obs, next_critic_obs, reward, terminated, time_outs, _ = env.step(action)

        # store — terminated only as done, not time_outs
        agent.collect(
            actor_obs       = actor_obs,
            critic_obs      = critic_obs,
            action          = action,
            reward          = reward,
            next_actor_obs  = next_actor_obs,
            next_critic_obs = next_critic_obs,
            done            = terminated,
        )

        # track per-env episode stats
        episode_returns += reward
        episode_steps   += 1

        # rolling reward stats (on-device, no sync)
        reward_sum   += reward.sum()
        reward_sqsum += (reward * reward).sum()
        reward_count += reward.numel()

        # handle done envs
        done_any = terminated | time_outs
        if done_any.any():
            finished = done_any.nonzero(as_tuple=False).squeeze(-1)

            returns_cpu = episode_returns[finished].cpu().tolist()
            lengths_cpu = episode_steps[finished].cpu().tolist()
            recent_returns.extend(returns_cpu)
            recent_lengths.extend(lengths_cpu)
            episode_num += len(returns_cpu)

            episode_returns[finished] = 0.0
            episode_steps[finished]   = 0.0

            # manual reset of done envs
            reset_actor, reset_critic, _ = env.reset(env_ids=finished)
            next_actor_obs[finished]  = reset_actor[finished]
            next_critic_obs[finished] = reset_critic[finished]

        # update agent
        if step >= config.learning_starts:
            agent.update()

        # advance obs
        actor_obs  = next_actor_obs
        critic_obs = next_critic_obs

        # logging
        if step % config.log_freq == 0 and step > 0:
            now = time.time()
            dt = max(now - last_log_time, 1e-6)
            last_log_time = now

            log_data = {
                'env_step':            step,
                'env/transitions':     step * args.num_envs,
                'buffer/size':         len(agent.buffer),
                'perf/iters_per_sec':  config.log_freq / dt,
                'perf/env_steps_per_sec': config.log_freq * args.num_envs / dt,
            }

            # reward stats (one sync for the whole block)
            if reward_count > 0:
                r_mean = (reward_sum / reward_count).item()
                r_var  = (reward_sqsum / reward_count).item() - r_mean * r_mean
                log_data['env/reward_mean'] = r_mean
                log_data['env/reward_std']  = max(r_var, 0.0) ** 0.5
                reward_sum.zero_(); reward_sqsum.zero_(); reward_count = 0

            # episode stats
            if len(recent_returns) > 0:
                log_data['episode/avg_return'] = sum(recent_returns) / len(recent_returns)
                log_data['episode/avg_steps']  = sum(recent_lengths) / len(recent_lengths)
                log_data['episode/num']        = episode_num

            # agent training metrics
            log_data.update(agent.pop_metrics())
            log_data['policy/target_entropy'] = agent.target_entropy

            wandb.log(log_data, step=step)

            avg_r = log_data.get('episode/avg_return', float('nan'))
            avg_l = log_data.get('episode/avg_steps',  float('nan'))
            qf    = log_data.get('loss/critic', float('nan'))
            al    = log_data.get('alpha/value', float('nan'))
            fps   = log_data['perf/env_steps_per_sec']
            print(f'Step {step:7d} | Ep {episode_num:5d} | '
                  f'R={avg_r:7.2f} L={avg_l:5.1f} | '
                  f'qf={qf:.3f} α={al:.4f} | '
                  f'{fps:.0f} env-steps/s')

        # checkpointing
        if step in ckpt_steps:
            path = f'{run_model_dir}/ckpt_step_{step}.pt'
            agent.save(path)
            print(f'Checkpoint saved: {path}')

    # final save
    final_path = f'{run_model_dir}/final.pt'
    agent.save(final_path)
    print(f'Final model saved: {final_path}')

    env.close()
    wandb.finish()
    print('Training complete.')


if __name__ == '__main__':
    train()