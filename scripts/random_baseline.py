import os
import numpy as np
import mujoco
from env.locomimic_env import LocoMimicEnv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--render', action='store_true', help='Render the environment')
parser.add_argument('--video', action='store_true', help='Save side-by-side video (random vs reference)')
parser.add_argument('--episodes', type=int, default=100)
parser.add_argument('--output', type=str, default=None)
args = parser.parse_args()

MOTION_PATH = './data/lafan1_retargeted/g1/walk1_subject1.csv'
N_EPISODES  = args.episodes

env = LocoMimicEnv(MOTION_PATH, render_mode='human' if args.render else None)

video = None
if args.video:
    import cv2
    os.makedirs('results/videos', exist_ok=True)
    output_path = args.output or 'results/videos/random_baseline.mp4'
    renderer = mujoco.Renderer(env.model, height=480, width=640)
    ref_renderer = mujoco.Renderer(env.model, height=480, width=640)
    cam = mujoco.MjvCamera()
    cam.distance = 3.0
    cam.azimuth = 90
    cam.elevation = -20
    ref_cam = mujoco.MjvCamera()
    ref_cam.distance = 3.0
    ref_cam.azimuth = 90
    ref_cam.elevation = -20
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, env.fps, (1280, 480))

episode_returns = []
total_frames = 0

for ep in range(N_EPISODES):
    obs, _ = env.reset()
    done = False
    episode_return = 0.0
    steps = 0

    while not done:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, _ = env.step(action)

        if args.render:
            env.render()

        episode_return += reward
        steps += 1
        done = terminated or truncated

        if video is not None:
            cam.lookat[:] = env.data.qpos[0:3].copy()
            renderer.update_scene(env.data, camera=cam)
            policy_frame = renderer.render()

            env.ref_data.qpos[:] = env.motion.get_qpos(env.phase)
            env.ref_data.qvel[:] = env.motion.get_qvel(env.phase)
            mujoco.mj_forward(env.model, env.ref_data)
            ref_cam.lookat[:] = env.ref_data.qpos[0:3].copy()
            ref_renderer.update_scene(env.ref_data, camera=ref_cam)
            ref_frame = ref_renderer.render()

            policy_labeled = policy_frame.copy()
            ref_labeled = ref_frame.copy()
            cv2.putText(policy_labeled, f'Random (ep {ep+1}, step {steps})', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(ref_labeled, 'Reference', (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            combined = np.concatenate([policy_labeled, ref_labeled], axis=1)
            video.write(cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
            total_frames += 1

    episode_returns.append(episode_return)
    print(f'Episode {ep+1:3d} | Return: {episode_return:.4f} | Steps: {steps}')

episode_returns = np.array(episode_returns)
print()
print(f'Random baseline over {N_EPISODES} episodes:')
print(f'  Mean   : {episode_returns.mean():.4f}')
print(f'  Std    : {episode_returns.std():.4f}')
print(f'  Min    : {episode_returns.min():.4f}')
print(f'  Max    : {episode_returns.max():.4f}')

np.save('./logs/random_baseline.npy', episode_returns)
print('Saved to logs/random_baseline.npy')

if video is not None:
    video.release()
    renderer.close()
    ref_renderer.close()
    print(f'Saved {total_frames} frames to {output_path}')

env.close()