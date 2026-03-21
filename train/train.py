import numpy as np
import torch
import wandb
import os
from env.locomimic_env import LocoMimicEnv
from train.agents.sac.sac_agent import SACAgent
from train.configs.config_loader import load_config

def train(config_path='train/configs/sac_config.yaml'):
    config = load_config(config_path)

    # init wandb
    wandb.init(
        project='locomimic',
        name='sac_walk1',
        config=config.__dict__
    )

    # create env and agent
    env   = LocoMimicEnv(config.motion_path)
    agent = SACAgent(obs_dim=139, act_dim=29, config=config)

    # create directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs',   exist_ok=True)

    print(f'Device : {config.device}')
    print(f'Training for {config.total_steps} steps')

    obs, _ = env.reset()
    episode_return = 0.0
    episode_steps = 0
    episode_num = 0

    for step in range(config.total_steps):

        if step < config.learning_starts:
            action = env.action_space.sample()
        else:
            action = agent.select_action(obs)

        obs_next, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        agent.collect(obs, action, reward, obs_next, done)

        obs = obs_next 
        episode_return += reward
        episode_steps += 1

        if step >= config.learning_starts:
            agent.update()

        if done:
            wandb.log({
                'episode_return' : episode_return,
                'episode_steps'  : episode_steps,
                'episode'        : episode_num,
                'step'           : step,
            })
            
            if episode_num % 1000 == 0:  # print every 1000 episodes
                print(f'Step {step:7d} | Episode {episode_num:4d} | '
                f'Return {episode_return:8.2f} | Steps {episode_steps:4d}')
            
            obs, _ = env.reset()
            episode_return = 0.0
            episode_steps = 0
            episode_num += 1
            
        if step % config.save_freq == 0 and step > 0:
            agent.save(f'models/sac_step_{step}.pt')
            print(f'Saved checkpoint at step {step}')

        # update curriculum termination thresholds every 100K steps
        if step % 100000 == 0 and step > 0:
            progress = step / config.total_steps
            env.update_curriculum(progress)
            print(f'Curriculum updated: height_thresh={env.height_threshold:.3f}, '
                  f'ori_thresh={env.ori_threshold:.3f}')

    agent.save('models/sac_final.pt')
    wandb.finish()
    print('Training complete.')   
          
if __name__ == '__main__':
    train()
    