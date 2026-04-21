"""
Render episodes where the policy fails (terminated early).
Saves a video of each failure for debugging.

Usage:
    python -m scripts.render_failures models/sac_v3/final.pt --algo sac
    python -m scripts.render_failures models/sac_v3/final.pt --algo sac --max-failures 3 --threshold 50
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
    parser.add_argument('--max-failures', type=int, default=5, help='Max failure videos to save')
    parser.add_argument('--threshold', type=int, default=100, help='Episodes shorter than this are failures')
    parser.add_argument('--max-episodes', type=int, default=200, help='Max episodes to search for failures')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    args = parser.parse_args()

    if args.algo == 'ppo':
        config = load_ppo_config('train/configs/ppo_config.yaml')
        agent = PPOAgent(obs_dim=139, act_dim=29, config=config)
    else:
        config = load_config('train/configs/sac_config.yaml')
        agent = SACAgent(obs_dim=139, act_dim=29, config=config)

    env = LocoMimicEnv(config.motion_path)
    agent.load(args.checkpoint)

    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)
    ref_renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)

    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -20

    ref_cam = mujoco.MjvCamera()
    ref_cam.distance = 3.0
    ref_cam.azimuth = 90
    ref_cam.elevation = -20

    os.makedirs('results/videos/failures', exist_ok=True)
    failures_found = 0

    print(f'Searching for failures (episodes < {args.threshold} steps)...\n')

    for ep in range(args.max_episodes):
        if failures_found >= args.max_failures:
            break

        obs, _ = env.reset()
        done = False
        frames = []
        steps = 0
        ep_return = 0.0

        while not done:
            action = agent.select_action(obs, deterministic=True)
            if args.algo == 'ppo':
                action = action[0]

            obs, reward, terminated, truncated, _ = env.step(action)
            ep_return += reward
            steps += 1
            done = terminated or truncated

            # Render policy
            cam.lookat[:] = env.data.qpos[0:3].copy()
            renderer.update_scene(env.data, camera=cam)
            policy_frame = renderer.render()

            # Render reference
            env.ref_data.qpos[:] = env.motion.get_qpos(env.phase)
            env.ref_data.qvel[:] = env.motion.get_qvel(env.phase)
            mujoco.mj_forward(env.model, env.ref_data)
            ref_cam.lookat[:] = env.ref_data.qpos[0:3].copy()
            ref_renderer.update_scene(env.ref_data, camera=ref_cam)
            ref_frame = ref_renderer.render()

            # Labels
            policy_labeled = policy_frame.copy()
            ref_labeled = ref_frame.copy()
            cv2.putText(policy_labeled, f'Policy (step {steps})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(ref_labeled, 'Reference', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            combined = np.concatenate([policy_labeled, ref_labeled], axis=1)
            frames.append(combined)

        if steps < args.threshold:
            failures_found += 1
            out_path = f'results/videos/failures/fail_{failures_found}_{steps}steps.mp4'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video = cv2.VideoWriter(out_path, fourcc, env.fps, (args.width * 2, args.height))
            for frame in frames:
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            video.release()
            print(f'  Failure {failures_found}: ep {ep}, {steps} steps, return {ep_return:.1f} -> {out_path}')
        else:
            if ep % 20 == 0:
                print(f'  ep {ep}: {steps} steps (ok)')

    renderer.close()
    ref_renderer.close()
    env.close()

    print(f'\nFound {failures_found} failures in {min(ep + 1, args.max_episodes)} episodes.')


if __name__ == '__main__':
    main()
