"""
Processed reference motion utilities for G1 motion imitation.

The raw LAFAN-retargeted CSV contains root pose and joint angles only. This
module turns that into richer, model-aware features inspired by DeepMimic and
BeyondMimic:

- smoothed qpos / qvel trajectories
- heading-local root features
- heading-local end-effector positions
- contact heuristics for both feet

The environment can then compare the simulated robot against a processed
reference frame rather than only the raw generalized coordinates.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import mujoco
import numpy as np


DEFAULT_MODEL_XML = "mujoco_menagerie/unitree_g1/scene.xml"
LEFT_FOOT_SITE = "left_foot"
RIGHT_FOOT_SITE = "right_foot"
LEFT_HAND_BODY = "left_wrist_yaw_link"
RIGHT_HAND_BODY = "right_wrist_yaw_link"
ROOT_BODY = "pelvis"


@dataclass(frozen=True)
class MotionFeatureFrame:
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    root_height: np.ndarray
    root_rot_6d: np.ndarray
    root_lin_vel_local: np.ndarray
    root_ang_vel_local: np.ndarray
    effector_pos_local: np.ndarray
    foot_contacts: np.ndarray


class MotionClip:
    def __init__(
        self,
        csv_path: str,
        model_xml: str = DEFAULT_MODEL_XML,
        fps: int = 30,
        smoothing_window: int = 5,
        contact_height_threshold: float = 0.06,
        contact_vel_threshold: float = 0.35,
    ):
        raw = np.loadtxt(csv_path, delimiter=",")
        self.fps = fps
        self.dt = 1.0 / fps
        self.n_frames = raw.shape[0]

        root_pos = raw[:, 0:3]
        root_quat_xyzw = raw[:, 3:7]
        joint_pos = raw[:, 7:36]

        root_quat = np.concatenate(
            [root_quat_xyzw[:, 3:4], root_quat_xyzw[:, 0:3]],
            axis=1,
        )
        root_quat = self._ensure_quat_continuity(root_quat)

        self.root_pos = self._smooth(root_pos, smoothing_window)
        self.root_quat = self._normalize_quaternions(
            self._smooth(root_quat, smoothing_window)
        )
        self.root_quat = self._ensure_quat_continuity(self.root_quat)
        self.joint_pos = self._smooth(joint_pos, smoothing_window)

        self.root_vel = self._central_difference(self.root_pos, self.dt)
        self.joint_vel = self._central_difference(self.joint_pos, self.dt)
        self.root_angvel = self._quat_angular_velocity(self.root_quat, self.dt)

        self.qpos = np.zeros((self.n_frames, 36), dtype=np.float32)
        self.qvel = np.zeros((self.n_frames, 35), dtype=np.float32)
        self.qpos[:, 0:3] = self.root_pos
        self.qpos[:, 3:7] = self.root_quat
        self.qpos[:, 7:36] = self.joint_pos
        self.qvel[:, 0:3] = self.root_vel
        self.qvel[:, 3:6] = self.root_angvel
        self.qvel[:, 6:35] = self.joint_vel

        model_path = model_xml
        if not os.path.isabs(model_path):
            model_path = os.path.join(os.path.dirname(__file__), "..", model_xml)
            model_path = os.path.normpath(model_path)
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

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

        self.root_height = self.qpos[:, 2:3].copy()
        self.root_rot_6d = np.zeros((self.n_frames, 6), dtype=np.float32)
        self.root_lin_vel_local = np.zeros((self.n_frames, 3), dtype=np.float32)
        self.root_ang_vel_local = np.zeros((self.n_frames, 3), dtype=np.float32)
        self.effector_pos_local = np.zeros((self.n_frames, 4, 3), dtype=np.float32)
        self.foot_site_world = np.zeros((self.n_frames, 2, 3), dtype=np.float32)

        for frame in range(self.n_frames):
            self.data.qpos[:] = self.qpos[frame]
            self.data.qvel[:] = self.qvel[frame]
            mujoco.mj_forward(self.model, self.data)

            root_rot_world = self.data.xmat[self.root_body_id].reshape(3, 3)
            heading_rot = self._heading_frame(root_rot_world)
            heading_inv = heading_rot.T
            rel_root_rot = heading_inv @ root_rot_world

            self.root_rot_6d[frame] = self._rotmat_to_6d(rel_root_rot)
            self.root_lin_vel_local[frame] = heading_inv @ self.qvel[frame, 0:3]
            self.root_ang_vel_local[frame] = heading_inv @ self.qvel[frame, 3:6]

            root_pos_world = self.data.xpos[self.root_body_id]
            foot_pos_world = self.data.site_xpos[self.foot_site_ids].copy()
            hand_pos_world = self.data.xpos[self.hand_body_ids].copy()
            effectors_world = np.concatenate([foot_pos_world, hand_pos_world], axis=0)

            self.effector_pos_local[frame] = (
                heading_inv @ (effectors_world - root_pos_world).T
            ).T
            self.foot_site_world[frame] = foot_pos_world

        foot_vel_world = self._central_difference(self.foot_site_world, self.dt)
        foot_speed = np.linalg.norm(foot_vel_world, axis=-1)
        foot_height = self.foot_site_world[:, :, 2]
        self.foot_contacts = (
            (foot_height < contact_height_threshold)
            & (foot_speed < contact_vel_threshold)
        ).astype(np.float32)

    @staticmethod
    def _smooth(sequence: np.ndarray, window: int) -> np.ndarray:
        if window <= 1:
            return sequence.astype(np.float32)
        if window % 2 == 0:
            window += 1
        pad = window // 2
        padded = np.pad(sequence, ((pad, pad), (0, 0)), mode="edge")
        kernel = np.ones(window, dtype=np.float32) / window
        smoothed = np.zeros_like(sequence, dtype=np.float32)
        for dim in range(sequence.shape[1]):
            smoothed[:, dim] = np.convolve(padded[:, dim], kernel, mode="valid")
        return smoothed

    @staticmethod
    def _central_difference(sequence: np.ndarray, dt: float) -> np.ndarray:
        derivative = np.zeros_like(sequence, dtype=np.float32)
        derivative[1:-1] = (sequence[2:] - sequence[:-2]) / (2.0 * dt)
        derivative[0] = (sequence[1] - sequence[0]) / dt
        derivative[-1] = (sequence[-1] - sequence[-2]) / dt
        return derivative

    @staticmethod
    def _ensure_quat_continuity(quat_seq: np.ndarray) -> np.ndarray:
        quat_seq = quat_seq.copy()
        for idx in range(1, len(quat_seq)):
            if np.dot(quat_seq[idx - 1], quat_seq[idx]) < 0.0:
                quat_seq[idx] *= -1.0
        return quat_seq

    @staticmethod
    def _normalize_quaternions(quat_seq: np.ndarray) -> np.ndarray:
        quat_seq = quat_seq.copy()
        norms = np.linalg.norm(quat_seq, axis=1, keepdims=True)
        quat_seq /= np.clip(norms, 1e-8, None)
        return quat_seq

    @staticmethod
    def _quat_conjugate(q: np.ndarray) -> np.ndarray:
        return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)

    @staticmethod
    def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=np.float32,
        )

    @classmethod
    def _quat_angular_velocity(cls, quat_seq: np.ndarray, dt: float) -> np.ndarray:
        angvel = np.zeros((len(quat_seq), 3), dtype=np.float32)
        for idx in range(1, len(quat_seq)):
            q_prev = quat_seq[idx - 1]
            q_curr = quat_seq[idx]
            if np.dot(q_prev, q_curr) < 0.0:
                q_curr = -q_curr
            q_delta = cls._quat_multiply(cls._quat_conjugate(q_prev), q_curr)
            angvel[idx] = 2.0 * q_delta[1:4] / dt
        angvel[0] = angvel[1]
        return angvel

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

    def get_qpos(self, frame_idx: int) -> np.ndarray:
        return self.qpos[frame_idx % self.n_frames].copy()

    def get_qvel(self, frame_idx: int) -> np.ndarray:
        return self.qvel[frame_idx % self.n_frames].copy()

    def get_features(self, frame_idx: int) -> MotionFeatureFrame:
        idx = frame_idx % self.n_frames
        return MotionFeatureFrame(
            joint_pos=self.joint_pos[idx].copy(),
            joint_vel=self.joint_vel[idx].copy(),
            root_height=self.root_height[idx].copy(),
            root_rot_6d=self.root_rot_6d[idx].copy(),
            root_lin_vel_local=self.root_lin_vel_local[idx].copy(),
            root_ang_vel_local=self.root_ang_vel_local[idx].copy(),
            effector_pos_local=self.effector_pos_local[idx].reshape(-1).copy(),
            foot_contacts=self.foot_contacts[idx].copy(),
        )

    def __len__(self) -> int:
        return self.n_frames
