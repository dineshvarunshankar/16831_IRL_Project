"""
DeepMimic-style locomotion imitation environment for Unitree G1.

The environment tracks a processed reference motion using:
- heading-local observations
- residual target-angle control around the reference motion
- imitation rewards on joint, root, end-effector, and contact features
- curriculum-friendly reset and termination settings
"""

from __future__ import annotations

import os
import time

import gymnasium as gym
import mujoco
import mujoco.viewer
import numpy as np

from env.motion_clip import (
    DEFAULT_MODEL_XML,
    LEFT_FOOT_SITE,
    RIGHT_FOOT_SITE,
    LEFT_HAND_BODY,
    RIGHT_HAND_BODY,
    ROOT_BODY,
    MotionClip,
)


class LocoMimicEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, motion_clip_path, config=None, render_mode=None):
        self.config = config
        model_xml = os.path.join(os.path.dirname(__file__), "..", DEFAULT_MODEL_XML)
        model_xml = os.path.normpath(model_xml)
        self.model = mujoco.MjModel.from_xml_path(model_xml)
        self.data = mujoco.MjData(self.model)
        self.ref_data = mujoco.MjData(self.model)

        self.fps = 30
        self.dt = 1.0 / self.fps
        self.n_substeps = int(round(self.dt / self.model.opt.timestep))
        self.max_steps = int(getattr(config, "episode_length", 1000))

        self.action_scale = float(getattr(config, "action_scale", 0.1))
        self.reset_joint_noise = float(getattr(config, "reset_joint_noise", 0.0))
        self.reset_vel_noise = float(getattr(config, "reset_vel_noise", 0.0))
        self.contact_height_threshold = float(
            getattr(config, "contact_height_threshold", 0.06)
        )
        self.contact_vel_threshold = float(
            getattr(config, "contact_vel_threshold", 0.35)
        )

        self.future_offsets = list(getattr(config, "future_offsets", [0, 1, 2, 4]))
        if 0 not in self.future_offsets:
            self.future_offsets = [0] + self.future_offsets
        self.future_offsets = sorted(set(self.future_offsets))

        self.reset_phase_start = float(getattr(config, "reset_phase_start", 0.0))
        self.reset_phase_end = float(getattr(config, "reset_phase_end", 0.15))
        self.initial_reset_phase_end = self.reset_phase_end

        self.final_height_threshold = float(getattr(config, "height_threshold", 0.25))
        self.initial_height_threshold = float(
            getattr(config, "initial_height_threshold", 0.45)
        )
        self.height_threshold = self.initial_height_threshold

        self.final_ori_threshold = float(getattr(config, "ori_threshold", 0.8))
        self.initial_ori_threshold = float(
            getattr(config, "initial_ori_threshold", 1.5)
        )
        self.ori_threshold = self.initial_ori_threshold
        self.min_root_height = float(getattr(config, "min_root_height", 0.45))

        self.pose_reward_weight = float(getattr(config, "pose_reward_weight", 0.40))
        self.vel_reward_weight = float(getattr(config, "vel_reward_weight", 0.10))
        self.root_reward_weight = float(getattr(config, "root_reward_weight", 0.20))
        self.root_vel_reward_weight = float(
            getattr(config, "root_vel_reward_weight", 0.15)
        )
        self.eff_reward_weight = float(getattr(config, "eff_reward_weight", 0.10))
        self.contact_reward_weight = float(
            getattr(config, "contact_reward_weight", 0.05)
        )

        self.pose_sigma = float(getattr(config, "pose_sigma", 0.35))
        self.vel_sigma = float(getattr(config, "vel_sigma", 2.0))
        self.root_sigma = float(getattr(config, "root_sigma", 0.35))
        self.root_vel_sigma = float(getattr(config, "root_vel_sigma", 1.0))
        self.eff_sigma = float(getattr(config, "eff_sigma", 0.12))

        self.action_rate_weight = float(getattr(config, "action_rate_weight", 0.01))
        self.joint_limit_weight = float(getattr(config, "joint_limit_weight", 2.0))

        self.motion = MotionClip(
            motion_clip_path,
            model_xml=DEFAULT_MODEL_XML,
            fps=self.fps,
            smoothing_window=int(getattr(config, "smoothing_window", 5)),
            contact_height_threshold=self.contact_height_threshold,
            contact_vel_threshold=self.contact_vel_threshold,
        )

        self.root_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, ROOT_BODY
        )
        self.hand_body_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_HAND_BODY),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_HAND_BODY),
            ],
            dtype=np.int32,
        )
        self.foot_site_ids = np.array(
            [
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, LEFT_FOOT_SITE),
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, RIGHT_FOOT_SITE),
            ],
            dtype=np.int32,
        )

        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.model.nu,),
            dtype=np.float32,
        )

        self.base_obs_dim = (
            self.model.nu
            + self.model.nu
            + 1
            + 6
            + 3
            + 3
            + 12
            + 2
            + self.model.nu
            + 1
        )
        self.ref_obs_dim = self.model.nu + self.model.nu + 1 + 6 + 3 + 3 + 12 + 2
        self.obs_dim = self.base_obs_dim + len(self.future_offsets) * self.ref_obs_dim
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.obs_dim,),
            dtype=np.float32,
        )

        self.phase = 0
        self.n_steps = 0
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.prev_foot_positions = np.zeros((2, 3), dtype=np.float32)
        self.current_features = None
        self.reference_features = None

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None

    def update_curriculum(self, progress: float):
        progress = float(np.clip(progress, 0.0, 1.0))
        self.height_threshold = (
            (1.0 - progress) * self.initial_height_threshold
            + progress * self.final_height_threshold
        )
        self.ori_threshold = (
            (1.0 - progress) * self.initial_ori_threshold
            + progress * self.final_ori_threshold
        )
        self.reset_phase_end = (
            (1.0 - progress) * self.initial_reset_phase_end + progress * 1.0
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.phase = self._sample_reset_phase()
        self.data.qpos[:] = self.motion.get_qpos(self.phase)
        self.data.qvel[:] = self.motion.get_qvel(self.phase)

        if self.reset_joint_noise > 0.0:
            self.data.qpos[7:] += self.np_random.normal(
                0.0, self.reset_joint_noise, self.model.nu
            )
        if self.reset_vel_noise > 0.0:
            self.data.qvel[6:] += self.np_random.normal(
                0.0, self.reset_vel_noise, self.model.nu
            )

        mujoco.mj_forward(self.model, self.data)

        self.n_steps = 0
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.prev_foot_positions = self.data.site_xpos[self.foot_site_ids].copy()
        self._update_tracking_cache()

        return self._get_obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32)
        action = np.clip(action, -1.0, 1.0)
        self._apply_reference_targets(action)

        for _ in range(self.n_substeps):
            mujoco.mj_step(self.model, self.data)

        self.n_steps += 1
        self.phase = (self.phase + 1) % len(self.motion)
        self._update_tracking_cache()

        obs = self._get_obs()
        reward = self._compute_reward(action)
        terminated = self._is_terminated()
        truncated = self.n_steps >= self.max_steps
        self.last_action = action.copy()

        return obs, reward, terminated, truncated, {}

    def _sample_reset_phase(self) -> int:
        phase_start = int(self.reset_phase_start * len(self.motion))
        phase_end = max(phase_start + 1, int(self.reset_phase_end * len(self.motion)))
        return int(self.np_random.integers(phase_start, phase_end))

    def _apply_reference_targets(self, action: np.ndarray):
        ref_joint_pos = self.motion.get_qpos(self.phase)[7:]
        target_pos = ref_joint_pos + action * self.action_scale
        self.data.ctrl[:] = target_pos

    def _update_tracking_cache(self):
        self.reference_features = self.motion.get_features(self.phase)
        self.current_features = self._extract_current_features()

    def _extract_current_features(self):
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return {
                "joint_pos": np.zeros(self.model.nu, dtype=np.float32),
                "joint_vel": np.zeros(self.model.nu, dtype=np.float32),
                "root_height": np.zeros(1, dtype=np.float32),
                "root_rot_6d": np.zeros(6, dtype=np.float32),
                "root_lin_vel_local": np.zeros(3, dtype=np.float32),
                "root_ang_vel_local": np.zeros(3, dtype=np.float32),
                "effector_pos_local": np.zeros(12, dtype=np.float32),
                "foot_contacts": np.zeros(2, dtype=np.float32),
                "root_rot_world": np.eye(3, dtype=np.float32),
            }

        root_rot_world = self.data.xmat[self.root_body_id].reshape(3, 3)
        heading_rot = self._heading_frame(root_rot_world)
        heading_inv = heading_rot.T
        rel_root_rot = heading_inv @ root_rot_world

        root_pos_world = self.data.xpos[self.root_body_id]
        foot_pos_world = self.data.site_xpos[self.foot_site_ids].copy()
        hand_pos_world = self.data.xpos[self.hand_body_ids].copy()
        effectors_world = np.concatenate([foot_pos_world, hand_pos_world], axis=0)
        effector_pos_local = (
            heading_inv @ (effectors_world - root_pos_world).T
        ).T.reshape(-1)

        foot_vel_world = (foot_pos_world - self.prev_foot_positions) / self.dt
        foot_speed = np.linalg.norm(foot_vel_world, axis=1)
        foot_contacts = (
            (foot_pos_world[:, 2] < self.contact_height_threshold)
            & (foot_speed < self.contact_vel_threshold)
        ).astype(np.float32)
        self.prev_foot_positions = foot_pos_world

        return {
            "joint_pos": self.data.qpos[7:].copy().astype(np.float32),
            "joint_vel": self.data.qvel[6:].copy().astype(np.float32),
            "root_height": self.data.qpos[2:3].copy().astype(np.float32),
            "root_rot_6d": self._rotmat_to_6d(rel_root_rot),
            "root_lin_vel_local": (heading_inv @ self.data.qvel[0:3]).astype(np.float32),
            "root_ang_vel_local": (heading_inv @ self.data.qvel[3:6]).astype(np.float32),
            "effector_pos_local": effector_pos_local.astype(np.float32),
            "foot_contacts": foot_contacts.astype(np.float32),
            "root_rot_world": root_rot_world.astype(np.float32),
        }

    def _reference_to_dict(self, ref_frame):
        return {
            "joint_pos": ref_frame.joint_pos,
            "joint_vel": ref_frame.joint_vel,
            "root_height": ref_frame.root_height,
            "root_rot_6d": ref_frame.root_rot_6d,
            "root_lin_vel_local": ref_frame.root_lin_vel_local,
            "root_ang_vel_local": ref_frame.root_ang_vel_local,
            "effector_pos_local": ref_frame.effector_pos_local,
            "foot_contacts": ref_frame.foot_contacts,
        }

    def _get_obs(self):
        if self.current_features is None:
            return np.zeros(self.obs_dim, dtype=np.float32)

        current = self.current_features
        phase_norm = np.array([self.phase / len(self.motion)], dtype=np.float32)

        obs_parts = [
            current["joint_pos"],
            current["joint_vel"],
            current["root_height"],
            current["root_rot_6d"],
            current["root_lin_vel_local"],
            current["root_ang_vel_local"],
            current["effector_pos_local"],
            current["foot_contacts"],
            self.last_action.astype(np.float32),
            phase_norm,
        ]

        for offset in self.future_offsets:
            ref = self._reference_to_dict(self.motion.get_features(self.phase + offset))
            obs_parts.extend(
                [
                    ref["joint_pos"],
                    ref["joint_vel"],
                    ref["root_height"],
                    ref["root_rot_6d"],
                    ref["root_lin_vel_local"],
                    ref["root_ang_vel_local"],
                    ref["effector_pos_local"],
                    ref["foot_contacts"],
                ]
            )

        return np.concatenate(obs_parts).astype(np.float32)

    def _compute_reward(self, action: np.ndarray) -> float:
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return -100.0

        current = self.current_features
        ref = self._reference_to_dict(self.reference_features)

        pose_err = np.mean((current["joint_pos"] - ref["joint_pos"]) ** 2)
        vel_err = np.mean((current["joint_vel"] - ref["joint_vel"]) ** 2)
        root_height_err = np.mean((current["root_height"] - ref["root_height"]) ** 2)
        root_rot_err = np.mean((current["root_rot_6d"] - ref["root_rot_6d"]) ** 2)
        root_lin_vel_err = np.mean(
            (current["root_lin_vel_local"] - ref["root_lin_vel_local"]) ** 2
        )
        root_ang_vel_err = np.mean(
            (current["root_ang_vel_local"] - ref["root_ang_vel_local"]) ** 2
        )
        eff_err = np.mean((current["effector_pos_local"] - ref["effector_pos_local"]) ** 2)
        contact_reward = np.mean(
            1.0 - np.abs(current["foot_contacts"] - ref["foot_contacts"])
        )

        r_pose = np.exp(-pose_err / (self.pose_sigma ** 2))
        r_vel = np.exp(-vel_err / (self.vel_sigma ** 2))
        r_root = np.exp(-(root_height_err + root_rot_err) / (self.root_sigma ** 2))
        r_root_vel = np.exp(
            -(root_lin_vel_err + root_ang_vel_err) / (self.root_vel_sigma ** 2)
        )
        r_eff = np.exp(-eff_err / (self.eff_sigma ** 2))

        tracking_reward = (
            self.pose_reward_weight * r_pose
            + self.vel_reward_weight * r_vel
            + self.root_reward_weight * r_root
            + self.root_vel_reward_weight * r_root_vel
            + self.eff_reward_weight * r_eff
            + self.contact_reward_weight * contact_reward
        )

        action_rate_penalty = self.action_rate_weight * np.mean(
            (action - self.last_action) ** 2
        )
        joint_limit_penalty = self.joint_limit_weight * self._joint_limit_violation()

        return float(tracking_reward - action_rate_penalty - joint_limit_penalty)

    def _joint_limit_violation(self) -> float:
        joint_pos = self.data.qpos[7:]
        lower = self.model.jnt_range[1:, 0]
        upper = self.model.jnt_range[1:, 1]
        below = np.clip(lower - joint_pos, 0.0, None)
        above = np.clip(joint_pos - upper, 0.0, None)
        return float(np.mean(below + above))

    def _is_terminated(self) -> bool:
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True

        ref_qpos = self.motion.get_qpos(self.phase)
        height_err = abs(self.data.qpos[2] - ref_qpos[2])
        if self.data.qpos[2] < self.min_root_height or height_err > self.height_threshold:
            return True

        ref_rot = self._quat_to_rotmat(ref_qpos[3:7])
        root_rot = self.current_features["root_rot_world"]
        rel_rot = ref_rot @ root_rot.T
        cos_angle = np.clip((np.trace(rel_rot) - 1.0) / 2.0, -1.0, 1.0)
        ori_err = np.arccos(cos_angle)
        if ori_err > self.ori_threshold:
            return True

        return False

    @staticmethod
    def _quat_to_rotmat(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = quat_wxyz
        return np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ],
            dtype=np.float32,
        )

    @staticmethod
    def _heading_frame(root_rot_world: np.ndarray) -> np.ndarray:
        forward = root_rot_world[:, 0].copy()
        forward[2] = 0.0
        norm = np.linalg.norm(forward)
        if norm < 1e-6:
            forward = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        else:
            forward = forward / norm
        up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        left = np.cross(up, forward)
        left /= np.clip(np.linalg.norm(left), 1e-6, None)
        return np.column_stack((forward, left, up)).astype(np.float32)

    @staticmethod
    def _rotmat_to_6d(rot: np.ndarray) -> np.ndarray:
        return np.concatenate([rot[:, 0], rot[:, 1]]).astype(np.float32)

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.cam.lookat[:] = self.data.qpos[0:3]
            self.viewer.cam.distance = 3.0
            self.viewer.cam.azimuth = 90
            self.viewer.cam.elevation = -20
            self.viewer.sync()
            time.sleep(self.dt)
        elif self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="side")
            return self.renderer.render()
        return None
