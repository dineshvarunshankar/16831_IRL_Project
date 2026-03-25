"""
DeepMimic-style locomotion imitation environment for Unitree G1.

The environment tracks a processed reference motion using:
- heading-local observations
- residual target-angle control around the reference motion
- imitation rewards on joint, root, end-effector, and contact features
- curriculum-friendly reset and termination settings
- optional adaptive reset-phase sampling on failure-prone motion bins
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
        self.final_reset_phase_end = float(
            getattr(config, "final_reset_phase_end", self.reset_phase_end)
        )
        self.curriculum_mode = str(getattr(config, "curriculum_mode", "off")).lower()
        self.competence_ema_alpha = float(
            getattr(config, "competence_ema_alpha", 0.98)
        )
        self.competence_low = float(getattr(config, "competence_low", 0.10))
        self.competence_high = float(getattr(config, "competence_high", 0.80))
        self.episode_len_ema = 0.0
        self.episode_len_ema_initialized = False

        self.adaptive_reset_enabled = bool(
            getattr(config, "adaptive_reset_enabled", False)
        )
        self.adaptive_reset_bins = int(getattr(config, "adaptive_reset_bins", 80))
        self.adaptive_reset_warmup_episodes = int(
            getattr(config, "adaptive_reset_warmup_episodes", 500)
        )
        self.adaptive_reset_uniform_mix = float(
            getattr(config, "adaptive_reset_uniform_mix", 0.30)
        )
        self.adaptive_reset_smoothing = int(
            getattr(config, "adaptive_reset_smoothing", 9)
        )
        self.adaptive_reset_power = float(getattr(config, "adaptive_reset_power", 1.5))
        self.adaptive_reset_min_visits = float(
            getattr(config, "adaptive_reset_min_visits", 5.0)
        )

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
        self.adaptive_reset_bins = max(4, min(self.adaptive_reset_bins, len(self.motion)))
        self.adaptive_fail_counts = np.zeros(self.adaptive_reset_bins, dtype=np.float32)
        self.adaptive_visit_counts = np.zeros(self.adaptive_reset_bins, dtype=np.float32)
        self.completed_episodes = 0

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
        self.reset_phase = 0
        self.n_steps = 0
        self.last_action = np.zeros(self.model.nu, dtype=np.float32)
        self.prev_foot_positions = np.zeros((2, 3), dtype=np.float32)
        self.current_features = None
        self.reference_features = None

        self.render_mode = render_mode
        self.viewer = None
        self.renderer = None
        self._diag_reason_keys = ("height", "orientation", "nan", "timeout", "other")
        self._diag_reward_keys = (
            "tracking_reward",
            "r_pose",
            "r_vel",
            "r_root",
            "r_root_vel",
            "r_eff",
            "contact_reward",
            "action_rate_penalty",
            "joint_limit_penalty",
            "total_reward",
        )
        self._reset_diagnostics()

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
        reset_phase_end = (
            (1.0 - progress) * self.initial_reset_phase_end
            + progress * self.final_reset_phase_end
        )
        min_span = 1.0 / max(len(self.motion), 1)
        self.reset_phase_end = float(
            np.clip(reset_phase_end, self.reset_phase_start + min_span, 1.0)
        )

    def _reset_diagnostics(self):
        self.diag_episode_count = 0
        self.diag_episode_len_sum = 0.0
        self.diag_terminated_count = 0
        self.diag_truncated_count = 0
        self.diag_reason_counts = {k: 0 for k in self._diag_reason_keys}
        self.diag_reward_step_count = 0
        self.diag_reward_sums = {k: 0.0 for k in self._diag_reward_keys}
        self.diag_height_err_count = 0
        self.diag_ori_err_count = 0
        self.diag_height_err_sum = 0.0
        self.diag_ori_err_sum = 0.0
        self.diag_terminal_bin_counts = np.zeros(self.adaptive_reset_bins, dtype=np.float32)

    def get_and_reset_diagnostics(self):
        payload = {
            "episode_count": int(self.diag_episode_count),
            "episode_len_sum": float(self.diag_episode_len_sum),
            "terminated_count": int(self.diag_terminated_count),
            "truncated_count": int(self.diag_truncated_count),
            "reason_counts": dict(self.diag_reason_counts),
            "reward_step_count": int(self.diag_reward_step_count),
            "reward_sums": dict(self.diag_reward_sums),
            "height_err_count": int(self.diag_height_err_count),
            "ori_err_count": int(self.diag_ori_err_count),
            "height_err_sum": float(self.diag_height_err_sum),
            "ori_err_sum": float(self.diag_ori_err_sum),
            "terminal_bin_counts": self.diag_terminal_bin_counts.astype(np.float32).tolist(),
            "reset_phase_end": float(self.reset_phase_end),
        }
        self._reset_diagnostics()
        return payload

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.phase = self._sample_reset_phase()
        self.reset_phase = self.phase
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
        reward, reward_terms = self._compute_reward(action)
        self.diag_reward_step_count += 1
        for key, value in reward_terms.items():
            self.diag_reward_sums[key] += float(value)

        terminated, term_reason, term_metrics = self._check_termination()
        truncated = self.n_steps >= self.max_steps
        if terminated or truncated:
            if terminated:
                end_reason = term_reason
            elif truncated:
                end_reason = "timeout"
            else:
                end_reason = "other"
            self._on_episode_end(
                terminated=terminated,
                truncated=truncated,
                reason=end_reason,
                term_metrics=term_metrics,
            )
        self.last_action = action.copy()

        return obs, reward, terminated, truncated, {}

    def _sample_reset_phase(self) -> int:
        phase_start, phase_end = self._phase_bounds()
        if not self.adaptive_reset_enabled:
            return int(self.np_random.integers(phase_start, phase_end))

        should_use_uniform = (
            self.completed_episodes < self.adaptive_reset_warmup_episodes
            or self.adaptive_visit_counts.sum() < self.adaptive_reset_min_visits
        )
        if should_use_uniform:
            return int(self.np_random.integers(phase_start, phase_end))

        bin_probs = self._adaptive_bin_probabilities(phase_start, phase_end)
        chosen_bin = int(self.np_random.choice(self.adaptive_reset_bins, p=bin_probs))
        return self._sample_phase_from_bin(chosen_bin, phase_start, phase_end)

    def _phase_bounds(self) -> tuple[int, int]:
        phase_start = int(self.reset_phase_start * len(self.motion))
        phase_end = max(phase_start + 1, int(self.reset_phase_end * len(self.motion)))
        phase_end = min(phase_end, len(self.motion))
        return phase_start, phase_end

    def _phase_to_bin(self, phase_idx: int) -> int:
        idx = int(np.clip(phase_idx, 0, len(self.motion) - 1))
        ratio = idx / max(len(self.motion), 1)
        return int(np.clip(ratio * self.adaptive_reset_bins, 0, self.adaptive_reset_bins - 1))

    def _sample_phase_from_bin(self, bin_idx: int, phase_start: int, phase_end: int) -> int:
        bin_start = int(bin_idx * len(self.motion) / self.adaptive_reset_bins)
        bin_end = int((bin_idx + 1) * len(self.motion) / self.adaptive_reset_bins)
        sample_start = max(phase_start, bin_start)
        sample_end = min(phase_end, max(bin_end, sample_start + 1))
        if sample_end <= sample_start:
            return int(self.np_random.integers(phase_start, phase_end))
        return int(self.np_random.integers(sample_start, sample_end))

    def _adaptive_bin_probabilities(self, phase_start: int, phase_end: int) -> np.ndarray:
        failure_rate = (self.adaptive_fail_counts + 1e-4) / (
            self.adaptive_visit_counts + 1e-4
        )
        failure_rate = np.power(failure_rate, self.adaptive_reset_power)

        if self.adaptive_reset_smoothing > 1:
            window = int(self.adaptive_reset_smoothing)
            if window % 2 == 0:
                window += 1
            kernel = np.ones(window, dtype=np.float32) / window
            failure_rate = np.convolve(failure_rate, kernel, mode="same")

        allowed = np.zeros(self.adaptive_reset_bins, dtype=np.float32)
        start_bin = self._phase_to_bin(phase_start)
        end_bin = self._phase_to_bin(max(phase_start, phase_end - 1)) + 1
        allowed[start_bin:end_bin] = 1.0

        weighted = failure_rate * allowed
        weighted_sum = weighted.sum()
        if weighted_sum <= 1e-8:
            weighted = allowed
            weighted_sum = weighted.sum()

        adaptive = weighted / max(weighted_sum, 1e-8)
        uniform = allowed / max(allowed.sum(), 1e-8)
        mix = float(np.clip(self.adaptive_reset_uniform_mix, 0.0, 1.0))
        probs = mix * uniform + (1.0 - mix) * adaptive
        probs /= max(probs.sum(), 1e-8)
        return probs

    def _on_episode_end(self, terminated: bool, truncated: bool, reason: str, term_metrics: dict):
        self.completed_episodes += 1
        terminal_bin = self._phase_to_bin(self.phase)
        self.adaptive_visit_counts[terminal_bin] += 1.0
        if terminated:
            self.adaptive_fail_counts[terminal_bin] += 1.0

        self.diag_episode_count += 1
        self.diag_episode_len_sum += float(self.n_steps)
        self.diag_terminal_bin_counts[terminal_bin] += 1.0
        if terminated:
            self.diag_terminated_count += 1
        if truncated:
            self.diag_truncated_count += 1
        if reason not in self.diag_reason_counts:
            reason = "other"
        self.diag_reason_counts[reason] += 1

        height_err = term_metrics.get("height_err", np.nan)
        ori_err = term_metrics.get("ori_err", np.nan)
        if np.isfinite(height_err):
            self.diag_height_err_sum += float(height_err)
            self.diag_height_err_count += 1
        if np.isfinite(ori_err):
            self.diag_ori_err_sum += float(ori_err)
            self.diag_ori_err_count += 1

        # Competence curriculum grows reset coverage based on achieved episode length.
        if self.curriculum_mode == "competence":
            ratio = float(np.clip(self.n_steps / max(self.max_steps, 1), 0.0, 1.0))
            if not self.episode_len_ema_initialized:
                self.episode_len_ema = ratio
                self.episode_len_ema_initialized = True
            else:
                alpha = float(np.clip(self.competence_ema_alpha, 0.0, 0.9999))
                self.episode_len_ema = alpha * self.episode_len_ema + (1.0 - alpha) * ratio

            denom = max(self.competence_high - self.competence_low, 1e-6)
            progress = (self.episode_len_ema - self.competence_low) / denom
            self.update_curriculum(progress)

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

    def _compute_reward(self, action: np.ndarray) -> tuple[float, dict]:
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            terms = {k: 0.0 for k in self._diag_reward_keys}
            terms["total_reward"] = -100.0
            return -100.0, terms

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

        total_reward = float(tracking_reward - action_rate_penalty - joint_limit_penalty)
        terms = {
            "tracking_reward": float(tracking_reward),
            "r_pose": float(r_pose),
            "r_vel": float(r_vel),
            "r_root": float(r_root),
            "r_root_vel": float(r_root_vel),
            "r_eff": float(r_eff),
            "contact_reward": float(contact_reward),
            "action_rate_penalty": float(action_rate_penalty),
            "joint_limit_penalty": float(joint_limit_penalty),
            "total_reward": total_reward,
        }
        return total_reward, terms

    def _joint_limit_violation(self) -> float:
        joint_pos = self.data.qpos[7:]
        lower = self.model.jnt_range[1:, 0]
        upper = self.model.jnt_range[1:, 1]
        below = np.clip(lower - joint_pos, 0.0, None)
        above = np.clip(joint_pos - upper, 0.0, None)
        return float(np.mean(below + above))

    def _check_termination(self) -> tuple[bool, str, dict]:
        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return True, "nan", {"height_err": np.nan, "ori_err": np.nan}

        ref_qpos = self.motion.get_qpos(self.phase)
        height_err = abs(self.data.qpos[2] - ref_qpos[2])
        if self.data.qpos[2] < self.min_root_height or height_err > self.height_threshold:
            return True, "height", {"height_err": float(height_err), "ori_err": np.nan}

        ref_rot = self._quat_to_rotmat(ref_qpos[3:7])
        root_rot = self.current_features["root_rot_world"]
        rel_rot = ref_rot @ root_rot.T
        cos_angle = np.clip((np.trace(rel_rot) - 1.0) / 2.0, -1.0, 1.0)
        ori_err = np.arccos(cos_angle)
        if ori_err > self.ori_threshold:
            return True, "orientation", {"height_err": float(height_err), "ori_err": float(ori_err)}

        return False, "none", {"height_err": float(height_err), "ori_err": float(ori_err)}

    def _is_terminated(self) -> bool:
        terminated, _, _ = self._check_termination()
        return terminated

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
