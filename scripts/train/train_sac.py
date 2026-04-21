import argparse
import torch
import wandb
import os
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
    args = parser.parse_args()

    config = load_config(args.config)
    config.num_envs = args.num_envs
    config.device   = 'cuda:0'

    # --- EXPERIMENT SETUP ---
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name      = args.name or f'sac_{timestamp}'
    run_model_dir = f'models/{run_name}'
    run_log_dir   = f'logs/{run_name}'
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir,   exist_ok=True)

    wandb.init(
        project='locomimic',
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
    env_cfg.seed           = 42

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
    print(f'Steps  : {config.total_steps}')

    # --- INIT STATE ---
    actor_obs, critic_obs, _ = env.reset()

    episode_returns = torch.zeros(args.num_envs, device=config.device)
    episode_steps   = torch.zeros(args.num_envs, device=config.device)
    episode_num     = 0

    recent_returns = []
    recent_lengths = []

    ckpt_steps = {
        int(config.total_steps * 0.25),
        int(config.total_steps * 0.50),
        int(config.total_steps * 0.75),
    }

    # --- TRAINING LOOP ---
    for step in range(config.total_steps):

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

        # handle done envs
        done_any = terminated | time_outs
        if done_any.any():
            finished = done_any.nonzero(as_tuple=False).squeeze(-1)

            for i in finished:
                recent_returns.append(episode_returns[i].item())
                recent_lengths.append(episode_steps[i].item())
                episode_num += 1

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
        if step % config.log_freq == 0 and len(recent_returns) > 0:
            avg_r = sum(recent_returns[-100:]) / min(len(recent_returns), 100)
            avg_l = sum(recent_lengths[-100:]) / min(len(recent_lengths), 100)

            log_data = {
                'episode/avg_return': avg_r,
                'episode/avg_steps':  avg_l,
                'episode/num':        episode_num,
                'env_step':           step,
            }

            if step >= config.learning_starts:
                log_data['agent/alpha']     = agent.log_alpha.exp().item()
                log_data['agent/log_alpha'] = agent.log_alpha.item()

            wandb.log(log_data, step=step)

            print(f'Step {step:7d} | Ep {episode_num:5d} | '
                  f'Avg100 R={avg_r:7.2f} L={avg_l:5.1f} | '
                  f'α={agent.log_alpha.exp().item():.4f}')

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