"""Evaluate a trained SAC checkpoint with mjlab's native MuJoCo viewer.

macOS:  mjpython -m scripts.eval.eval_sac --ckpt <path> [--fast-sac]
Linux:  python    -m scripts.eval.eval_sac --ckpt <path> [--fast-sac]
"""

import argparse
import os

import torch

import mjlab.tasks  # noqa: F401
import src.tasks    # noqa: F401  registers Unitree-G1-Tracking

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer

from rl.sac_env_wrapper import EmpiricalNormalization
from agents.sac.sac_agent import SACAgent
from agents.configs.config_loader import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt',     type=str, required=True)
    parser.add_argument('--config',   type=str, default='agents/configs/sac_config.yaml')
    parser.add_argument('--task',     type=str, default='Unitree-G1-Tracking')
    parser.add_argument('--device',   type=str, default='cpu')
    parser.add_argument('--num_envs', type=int, default=1)
    parser.add_argument('--fast-sac', action='store_true',
                        help='must match training flag for the checkpoint')
    parser.add_argument('--keep-terminations', action='store_true',
                        help='keep termination conditions (default: disabled for eval)')
    parser.add_argument('--sampling-mode', type=str, default='start',
                        choices=['start', 'uniform', 'adaptive'],
                        help='motion frame init mode (start=frame 0, uniform=random, adaptive=curriculum)')
    parser.add_argument('--init-frame', type=int, default=None,
                        help='fixed motion frame index to start from (overrides --sampling-mode)')
    args = parser.parse_args()

    config = load_config(args.config)
    config.device   = args.device
    config.num_envs = args.num_envs
    if args.fast_sac:
        config.use_layer_norm = True
        config.use_mean_q     = True

    configure_torch_backends()

    env_cfg = load_env_cfg(args.task, play=True)
    motion_cmd = env_cfg.commands['motion']
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = config.motion_path
    # eval-time motion-frame init mode
    motion_cmd.sampling_mode = args.sampling_mode
    # also disable RSI pose/velocity randomization so init is deterministic
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}
    motion_cmd.joint_position_range = (0.0, 0.0)

    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed           = config.seed
    if not args.keep_terminations:
        env_cfg.terminations = {}
        print('[INFO] terminations disabled (pass --keep-terminations to enable)')

    env = ManagerBasedRlEnv(cfg=env_cfg, device=args.device)

    # Optional: pin starting motion frame. We monkey-patch the motion command's
    # sampling hook so every subsequent reset places time_steps at --init-frame
    # (rather than 0 / random / curriculum). The command's own _resample_command
    # reads time_steps AFTER sampling to teleport the robot to that frame, so
    # this also controls the initial physical pose.
    if args.init_frame is not None:
        mcmd = env.command_manager.get_term('motion')
        total = int(mcmd.motion.time_step_total)
        frame = max(0, min(args.init_frame, total - 1))

        def _fixed_sampling(env_ids, _mcmd=mcmd, _frame=frame):
            _mcmd.time_steps[env_ids] = _frame

        mcmd._uniform_sampling = _fixed_sampling
        mcmd._adaptive_sampling = _fixed_sampling
        # also handle the "start" branch
        original_resample = mcmd._resample_command
        def _resample_fixed(env_ids, _mcmd=mcmd, _frame=frame, _orig=original_resample):
            # call original so robot state gets written, but force frame first
            _mcmd.cfg.sampling_mode = 'uniform'  # ensures _uniform_sampling is called (our fixed hook)
            _orig(env_ids)
        mcmd._resample_command = _resample_fixed
        print(f'[INFO] init-frame = {frame} / {total}')

    env = RslRlVecEnvWrapper(env)

    # dims for agent (use unwrapped to access mjlab obs space)
    inner = env.unwrapped
    actor_obs_dim  = inner.single_observation_space.spaces['actor'].shape[0]
    critic_obs_dim = inner.single_observation_space.spaces['critic'].shape[0]
    act_dim        = inner.single_action_space.shape[0]

    agent = SACAgent(
        actor_obs_dim  = actor_obs_dim,
        critic_obs_dim = critic_obs_dim,
        act_dim        = act_dim,
        config         = config,
    )
    agent.load(args.ckpt)

    # load obs normalizer (if present)
    actor_rms = EmpiricalNormalization(actor_obs_dim, torch.device(args.device))
    rms_path = args.ckpt.replace('.pt', '.rms.pt')
    if os.path.exists(rms_path):
        sd = torch.load(rms_path, map_location=args.device)
        actor_rms.load_state_dict(sd['actor_rms'])
        print(f'Loaded obs normalizer: {rms_path}')
    else:
        print(f'WARNING: no .rms.pt sidecar at {rms_path} — obs will be unnormalized (policy will misbehave)')

    @torch.no_grad()
    def policy(obs_dict):
        actor_obs = obs_dict['actor']
        actor_obs = actor_rms(actor_obs)
        return agent.select_action(actor_obs, deterministic=True)

    NativeMujocoViewer(env, policy).run()
    env.close()


if __name__ == '__main__':
    main()
