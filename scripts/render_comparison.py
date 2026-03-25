"""
Render policy vs reference motion side by side as a video.

Usage:
    python -m scripts.render_comparison models/tracking_v2/final.pt --algo sac
    python -m scripts.render_comparison models/ppo_v1/final.pt --algo ppo --episodes 3
    python -m scripts.render_comparison models/tracking_v2/final.pt --algo sac --output my_comparison.mp4
"""

import argparse
import os
import numpy as np
import mujoco
import cv2
from env.locomimic_env import LocoMimicEnv
from train.agents.sac.sac_agent import SACAgent
from train.agents.ppo.ppo_agent import PPOAgent
from train.configs.config_loader import load_config, load_ppo_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoint', type=str)
    parser.add_argument('--algo', type=str, choices=['sac', 'ppo'], default='sac')
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--episodes', type=int, default=1)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    if args.algo == 'ppo':
        config_path = args.config or 'train/configs/ppo_config.yaml'
        config = load_ppo_config(config_path)
    else:
        config_path = args.config or 'train/configs/sac_config.yaml'
        config = load_config(config_path)

    env = LocoMimicEnv(config.motion_path, config=config)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if args.algo == 'ppo':
        agent = PPOAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)
    else:
        agent = SACAgent(obs_dim=obs_dim, act_dim=act_dim, config=config)

    agent.load(args.checkpoint)

    # two renderers: one for policy, one for reference
    policy_renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    ref_renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)

    # cameras for each view
    policy_cam = mujoco.MjvCamera()
    policy_cam.distance = 3.0
    policy_cam.azimuth = 90
    policy_cam.elevation = -20

    ref_cam = mujoco.MjvCamera()
    ref_cam.distance = 3.0
    ref_cam.azimuth = 90
    ref_cam.elevation = -20

    os.makedirs('results/videos', exist_ok=True)
    output_path = args.output or f'results/videos/comparison_{args.algo}.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, env.fps, (args.width * 2, args.height))

    total_frames = 0

    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        ep_return = 0.0
        steps = 0

        while not done:
            # get action
            if args.algo == 'ppo':
                action = agent.select_action(obs, deterministic=True)[0]
            else:
                action = agent.select_action(obs, deterministic=True)

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            ep_return += reward
            steps += 1

            # render policy state
            policy_cam.lookat[:] = env.data.qpos[0:3].copy()
            policy_renderer.update_scene(env.data, camera=policy_cam)
            policy_frame = policy_renderer.render()

            # render reference state
            env.ref_data.qpos[:] = env.motion.get_qpos(env.phase)
            env.ref_data.qvel[:] = env.motion.get_qvel(env.phase)
            mujoco.mj_forward(env.model, env.ref_data)

            ref_cam.lookat[:] = env.ref_data.qpos[0:3].copy()
            ref_renderer.update_scene(env.ref_data, camera=ref_cam)
            ref_frame = ref_renderer.render()

            # add labels
            policy_labeled = policy_frame.copy()
            ref_labeled = ref_frame.copy()
            cv2.putText(policy_labeled, 'Policy', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(ref_labeled, 'Reference', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            # stitch side by side
            combined = np.concatenate([policy_labeled, ref_labeled], axis=1)
            video.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            total_frames += 1

        print(f'Episode {ep+1} | Return: {ep_return:.2f} | Steps: {steps}')

    video.release()
    policy_renderer.close()
    ref_renderer.close()
    env.close()
    print(f'\nSaved {total_frames} frames to {output_path}')


if __name__ == '__main__':
    main()
