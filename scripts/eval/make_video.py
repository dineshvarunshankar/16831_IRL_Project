"""Side-by-side reference vs. policy videos (mp4 + gif).

Reference motion is rendered on the LEFT, policy rollout on the RIGHT.
No labels/overlays. One concatenated clip across all episodes.

usage:
  mjpython -m scripts.eval.make_video --algo random
  mjpython -m scripts.eval.make_video --algo sac      --ckpt path/to/ckpt.pt
  mjpython -m scripts.eval.make_video --algo fast_sac --ckpt path/to/ckpt.pt
  mjpython -m scripts.eval.make_video --algo ppo      --ckpt path/to/ckpt.pt   # stub
  mjpython -m scripts.eval.make_video --algo tdmpc    --ckpt path/to/ckpt.pt   # stub
"""

import argparse
import copy
import os
import time
from pathlib import Path

import imageio
import mujoco
import numpy as np
import torch
from PIL import Image

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401  registers Unitree-G1-Tracking

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends


# ----- per-robot offscreen renderer ---------------------------------------- #

class RobotRenderer:
    """Renders a single qpos pose of a copy of the compiled mj_model."""

    def __init__(self, model: mujoco.MjModel, width: int, height: int,
                 track_body_id: int, distance: float, elevation: float,
                 azimuth: float):
        self._model = copy.deepcopy(model)
        self._model.vis.global_.offwidth  = width
        self._model.vis.global_.offheight = height
        self._model.light_castshadow[:] = False
        self._model.mat_reflectance[:] = 0.0
        self._data = mujoco.MjData(self._model)
        self._renderer = mujoco.Renderer(self._model, height=height, width=width)

        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self._model, cam)
        cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
        cam.trackbodyid = track_body_id
        cam.fixedcamid = -1
        cam.distance = distance
        cam.elevation = elevation
        cam.azimuth = azimuth
        self._cam = cam

    def render(self, qpos: np.ndarray) -> np.ndarray:
        self._data.qpos[:] = qpos
        self._data.qvel[:] = 0.0
        mujoco.mj_forward(self._model, self._data)
        self._renderer.update_scene(self._data, camera=self._cam)
        return self._renderer.render()

    def close(self):
        self._renderer.close()


# ----- policy dispatch ----------------------------------------------------- #

def _make_random_policy(env):
    inner = env.unwrapped
    act_dim = inner.single_action_space.shape[0]
    device = inner.device

    def policy(_obs):
        return torch.empty(inner.num_envs, act_dim, device=device).uniform_(-1.0, 1.0)
    return policy


def _make_sac_policy(env, args, fast_sac: bool):
    from agents.sac.sac_agent import SACAgent
    from agents.configs.config_loader import load_config
    from rl.sac_env_wrapper import EmpiricalNormalization

    cfg = load_config(args.config or 'agents/configs/sac_config.yaml')
    cfg.device   = args.device
    cfg.num_envs = 1
    if fast_sac:
        cfg.use_layer_norm = True
        cfg.use_mean_q     = True

    # auto-detect distributional critic from the checkpoint: q_support holds
    # the per-atom values, so its shape gives num_atoms / v_min / v_max
    ckpt = torch.load(args.ckpt, map_location=args.device)
    critic_sd = ckpt.get('critic', {})
    if 'q_support' in critic_sd:
        q_support = critic_sd['q_support']
        cfg.use_distributional = True
        cfg.num_atoms = int(q_support.numel())
        cfg.v_min = float(q_support.min().item())
        cfg.v_max = float(q_support.max().item())
        print(f'[INFO] distributional critic: atoms={cfg.num_atoms} '
              f'v_min={cfg.v_min:.2f} v_max={cfg.v_max:.2f}')

    inner = env.unwrapped
    actor_obs_dim  = inner.single_observation_space.spaces['actor'].shape[0]
    critic_obs_dim = inner.single_observation_space.spaces['critic'].shape[0]
    act_dim        = inner.single_action_space.shape[0]

    agent = SACAgent(actor_obs_dim, critic_obs_dim, act_dim, cfg)
    agent.load(args.ckpt)

    rms = EmpiricalNormalization(actor_obs_dim, torch.device(args.device))
    rms_path = args.ckpt.replace('.pt', '.rms.pt')
    if os.path.exists(rms_path):
        sd = torch.load(rms_path, map_location=args.device)
        rms.load_state_dict(sd['actor_rms'])
        print(f'[INFO] loaded obs normalizer: {rms_path}')
    else:
        print(f'[WARN] no .rms.pt sidecar at {rms_path}; obs will be unnormalized')

    @torch.no_grad()
    def policy(obs_dict):
        return agent.select_action(rms(obs_dict['actor']), deterministic=True)
    return policy


def _make_ppo_policy(env, args):
    raise NotImplementedError(
        'ppo is not yet wired into make_video.py — load via mjlab.rl.runner '
        '(rsl_rl) and return a callable mapping obs_dict -> action tensor.'
    )


def _make_tdmpc_policy(env, args):
    raise NotImplementedError(
        'tdmpc is not yet wired into make_video.py — load via '
        'agents.tdmpc2.tdmpc2.tdmpc2.TDMPC2 and adapt obs preprocessing to '
        'match its env adapter.'
    )


def make_policy(algo: str, env, args):
    if algo == 'random':   return _make_random_policy(env)
    if algo == 'sac':      return _make_sac_policy(env, args, fast_sac=False)
    if algo == 'fast_sac': return _make_sac_policy(env, args, fast_sac=True)
    if algo == 'ppo':      return _make_ppo_policy(env, args)
    if algo == 'tdmpc':    return _make_tdmpc_policy(env, args)
    raise ValueError(f'unknown algo {algo!r}')


# ----- env setup ----------------------------------------------------------- #

def build_env(args):
    env_cfg = load_env_cfg(args.task, play=True)
    motion_cmd_cfg = env_cfg.commands['motion']
    assert isinstance(motion_cmd_cfg, MotionCommandCfg)
    if args.motion_path:
        motion_cmd_cfg.motion_file = args.motion_path
    motion_cmd_cfg.sampling_mode = 'uniform'
    motion_cmd_cfg.pose_range = {}
    motion_cmd_cfg.velocity_range = {}
    motion_cmd_cfg.joint_position_range = (0.0, 0.0)

    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.terminations = {}  # play through the full --steps every episode

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)
    return RslRlVecEnvWrapper(env), motion_cmd_cfg


# ----- main ---------------------------------------------------------------- #

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', required=True,
                   choices=['random', 'sac', 'fast_sac', 'ppo', 'tdmpc'])
    p.add_argument('--ckpt', type=str, default=None)
    p.add_argument('--config', type=str, default=None)
    p.add_argument('--motion-path', type=str,
                   default='data/lafan1_retargeted_npz/g1/walk1_subject1.npz',
                   help='reference motion .npz to play back')
    p.add_argument('--task', type=str, default='Unitree-G1-Tracking')
    p.add_argument('--device', type=str, default='cpu')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--steps', type=int, default=250,
                   help='steps per episode (50hz -> 500 = 10s)')
    p.add_argument('--episodes', type=int, default=3,
                   help='number of episodes; init frame sampled uniformly each time')
    p.add_argument('--init-frames', type=str, default='1000,4000,600',
                   help='comma-separated motion frame indices, one per episode; '
                        'pass empty string to fall back to random sampling')
    p.add_argument('--width',  type=int, default=480)
    p.add_argument('--height', type=int, default=360)
    p.add_argument('--gif-fps', type=int, default=25)
    p.add_argument('--gif-width', type=int, default=480,
                   help='downscale gif to this combined (side-by-side) width')
    p.add_argument('--cam-distance',  type=float, default=2.5)
    p.add_argument('--cam-elevation', type=float, default=-15.0)
    p.add_argument('--cam-azimuth',   type=float, default=120.0)
    p.add_argument('--out-dir', type=str, default='outputs/videos')
    args = p.parse_args()

    if args.algo not in ('random',) and args.ckpt is None:
        p.error(f'--ckpt is required for --algo {args.algo}')

    init_frames: list[int] | None = None
    if args.init_frames:
        init_frames = [int(x) for x in args.init_frames.split(',') if x.strip()]
        if len(init_frames) != args.episodes:
            p.error(f'--init-frames has {len(init_frames)} entries but '
                    f'--episodes is {args.episodes}')

    configure_torch_backends()
    env, _ = build_env(args)
    inner = env.unwrapped
    motion_cmd = inner.command_manager.get_term('motion')

    # if --init-frames was supplied, override the motion command's sampling
    # hooks so each reset places time_steps at our pre-set frame index
    next_frame_idx = [0]
    if init_frames is not None:
        total = int(motion_cmd.motion.time_step_total)
        for f in init_frames:
            if not (0 <= f < total):
                p.error(f'--init-frames contains {f} but motion has {total} frames')

        def _fixed_sampling(env_ids, _mcmd=motion_cmd):
            _mcmd.time_steps[env_ids] = init_frames[next_frame_idx[0]]

        motion_cmd._uniform_sampling = _fixed_sampling
        motion_cmd._adaptive_sampling = _fixed_sampling
        print(f'[INFO] init-frames override: {init_frames}')

    # camera tracks the robot's anchor body
    robot = inner.scene[motion_cmd.cfg.entity_name]
    anchor_body_ids, _ = robot.find_bodies(motion_cmd.cfg.anchor_body_name)
    track_body_id = robot.indexing.bodies[anchor_body_ids[0]].id

    # qpos addressing for synthesizing the reference pose
    free_q_adr  = robot.indexing.free_joint_q_adr.cpu().numpy()
    joint_q_adr = robot.indexing.joint_q_adr.cpu().numpy()

    common = dict(
        model=inner.sim.mj_model,
        width=args.width, height=args.height,
        track_body_id=track_body_id,
        distance=args.cam_distance,
        elevation=args.cam_elevation,
        azimuth=args.cam_azimuth,
    )
    ref_renderer = RobotRenderer(**common)
    pol_renderer = RobotRenderer(**common)

    policy = make_policy(args.algo, env, args)

    # ---- rollout ---------------------------------------------------------- #
    nq = inner.sim.mj_model.nq
    frames: list[np.ndarray] = []
    fps = float(1.0 / inner.step_dt)
    print(f'[INFO] step_dt={inner.step_dt:.4f}s  fps={fps:.1f}  '
          f'total={args.episodes * args.steps} steps '
          f'({args.episodes * args.steps / fps:.1f}s)')

    for ep in range(args.episodes):
        next_frame_idx[0] = ep
        obs_dict, _ = env.reset()
        print(f'[INFO] episode {ep+1}/{args.episodes} '
              f'init_frame={int(motion_cmd.time_steps[0].item())}')

        for _ in range(args.steps):
            # actual robot pose
            actual_qpos = inner.sim.data.qpos[0].cpu().numpy()

            # reference pose (env 0)
            ref_qpos = np.zeros(nq)
            ref_qpos[free_q_adr[0:3]] = motion_cmd.body_pos_w[0, 0].cpu().numpy() \
                                        - inner.scene.env_origins[0].cpu().numpy()
            ref_qpos[free_q_adr[3:7]] = motion_cmd.body_quat_w[0, 0].cpu().numpy()
            ref_qpos[joint_q_adr]     = motion_cmd.joint_pos[0].cpu().numpy()

            ref_img = ref_renderer.render(ref_qpos)
            pol_img = pol_renderer.render(actual_qpos)
            frames.append(np.concatenate([ref_img, pol_img], axis=1))

            with torch.no_grad():
                action = policy(obs_dict)
            obs_dict, _, _, _ = env.step(action)

    env.close()
    ref_renderer.close()
    pol_renderer.close()

    # ---- write outputs ---------------------------------------------------- #
    stamp = time.strftime('%Y%m%d-%H%M%S')
    out_dir = Path(args.out_dir) / f'{args.algo}_{stamp}'
    out_dir.mkdir(parents=True, exist_ok=True)

    mp4_path = out_dir / f'{args.algo}.mp4'
    gif_path = out_dir / f'{args.algo}.gif'

    print(f'[INFO] writing {mp4_path}  ({len(frames)} frames @ {fps:.1f} fps)')
    imageio.mimwrite(mp4_path, frames, fps=fps, codec='libx264', quality=8,
                     macro_block_size=1)

    # gif: downscale and lower fps
    target_w = args.gif_width
    h, w = frames[0].shape[:2]
    if target_w < w:
        scale = target_w / w
        new_size = (target_w, int(h * scale))
        gif_frames = [
            np.array(Image.fromarray(f).resize(new_size, Image.BILINEAR))
            for f in frames
        ]
    else:
        gif_frames = frames

    stride = max(1, int(round(fps / args.gif_fps)))
    gif_frames = gif_frames[::stride]
    print(f'[INFO] writing {gif_path}  ({len(gif_frames)} frames @ {args.gif_fps} fps, '
          f'{gif_frames[0].shape[1]}x{gif_frames[0].shape[0]})')
    imageio.mimwrite(gif_path, gif_frames, fps=args.gif_fps, loop=0)

    print(f'[DONE] {out_dir}')


if __name__ == '__main__':
    main()
