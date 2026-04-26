from __future__ import annotations

from collections.abc import Mapping

import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.utils.lab_api.math import matrix_from_quat, subtract_frame_transforms


class TDMPC2VecEnvWrapper:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        exogenous_terms: tuple[str, ...],
        clip_actions: float | None = None,
    ):
        self.env = env
        self.num_envs = int(env.num_envs)
        self.device = torch.device(env.device)
        self.clip_actions = clip_actions
        self.exogenous_terms = tuple(exogenous_terms)

        self.actor_obs_key = "actor"
        self.critic_obs_key = "critic"
        self.act_dim = int(env.single_action_space.shape[0])
        self.motion_command = None
        try:
            self.motion_command = self.env.command_manager.get_term("motion")
        except Exception:
            self.motion_command = None

        self._last_endog = None
        self._last_exog = None
        self._last_actor_terms: dict[str, torch.Tensor] | None = None
        self._active_exogenous_terms: tuple[str, ...] = tuple()

        endog_obs, exog_obs, _ = self.reset()
        self.endog_obs_dim = int(endog_obs.shape[-1])
        self.exog_obs_dim = int(exog_obs.shape[-1])

    @property
    def actor_term_names(self) -> tuple[str, ...]:
        if self._last_actor_terms is None:
            return tuple()
        return tuple(self._last_actor_terms.keys())

    @property
    def actor_term_shapes(self) -> dict[str, tuple[int, ...]]:
        if self._last_actor_terms is None:
            return {}
        return {name: tuple(value.shape[1:]) for name, value in self._last_actor_terms.items()}

    @property
    def active_exogenous_terms(self) -> tuple[str, ...]:
        return self._active_exogenous_terms

    def _flatten(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)

    def _split_actor(self, actor_obs) -> tuple[torch.Tensor, torch.Tensor]:
        if isinstance(actor_obs, Mapping):
            exog_parts = []
            endog_parts = []
            term_map: dict[str, torch.Tensor] = {}
            active_exog_terms: list[str] = []
            for key, value in actor_obs.items():
                flat = self._flatten(value)
                term_map[key] = flat
                if key in self.exogenous_terms:
                    exog_parts.append(flat)
                    active_exog_terms.append(key)
                else:
                    endog_parts.append(flat)
            self._last_actor_terms = term_map
            self._active_exogenous_terms = tuple(active_exog_terms)

            if len(exog_parts) == 0:
                exog = torch.zeros((self.num_envs, 0), device=self.device)
            else:
                exog = torch.cat(exog_parts, dim=-1)
            endog = torch.cat(endog_parts, dim=-1) if len(endog_parts) > 0 else exog
            return endog, exog

        if isinstance(actor_obs, torch.Tensor):
            flat = self._flatten(actor_obs)
            exog = torch.zeros((flat.shape[0], 0), device=flat.device, dtype=flat.dtype)
            return flat, exog

        raise TypeError(f"Unsupported actor observation type: {type(actor_obs)}")

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, reward, terminated, truncated, extras = self.env.step(actions)
        actor_obs = obs_dict[self.actor_obs_key]
        endog, exog = self._split_actor(actor_obs)
        self._last_endog = endog
        self._last_exog = exog
        return endog, exog, reward, terminated, truncated, extras

    def reset(self, env_ids: torch.Tensor | None = None):
        obs_dict, extras = self.env.reset(env_ids=env_ids)
        actor_obs = obs_dict[self.actor_obs_key]
        endog, exog = self._split_actor(actor_obs)
        self._last_endog = endog
        self._last_exog = exog
        return endog, exog, extras

    def _repeat_current_exog(self, horizon: int) -> torch.Tensor:
        if self._last_exog is None:
            return torch.zeros(
                (self.num_envs, horizon + 1, 0),
                device=self.device,
                dtype=torch.float32,
            )
        return self._last_exog.unsqueeze(1).expand(-1, horizon + 1, -1).contiguous()

    @torch.no_grad()
    def get_exog_plan_sequence(self, horizon: int) -> torch.Tensor:
        if horizon < 0:
            raise ValueError(f"horizon must be >= 0, got {horizon}")
        if self.exog_obs_dim == 0:
            return torch.zeros(
                (self.num_envs, horizon + 1, 0),
                device=self.device,
                dtype=torch.float32,
            )
        if self.motion_command is None:
            return self._repeat_current_exog(horizon)

        cmd = self.motion_command
        motion = cmd.motion
        t0 = cmd.time_steps
        offs = torch.arange(horizon + 1, device=self.device).unsqueeze(0)
        idx = (t0.unsqueeze(1) + offs).clamp(max=motion.time_step_total - 1)

        parts = []
        term_names = self._active_exogenous_terms
        if len(term_names) == 0:
            term_names = self.exogenous_terms

        for name in term_names:
            if name == "command":
                joint_pos = motion.joint_pos[idx]
                joint_vel = motion.joint_vel[idx]
                parts.append(torch.cat([joint_pos, joint_vel], dim=-1))
                continue

            if name in {"motion_anchor_pos_b", "motion_anchor_ori_b"}:
                anchor_idx = cmd.motion_anchor_body_index
                anchor_pos_w = motion.body_pos_w[idx, anchor_idx] + self.env.scene.env_origins.unsqueeze(1)
                anchor_quat_w = motion.body_quat_w[idx, anchor_idx]

                robot_pos = cmd.robot_anchor_pos_w.unsqueeze(1).expand(-1, horizon + 1, -1)
                robot_quat = cmd.robot_anchor_quat_w.unsqueeze(1).expand(-1, horizon + 1, -1)
                pos_b, ori_b = subtract_frame_transforms(
                    robot_pos,
                    robot_quat,
                    anchor_pos_w,
                    anchor_quat_w,
                )
                if name == "motion_anchor_pos_b":
                    parts.append(pos_b.reshape(self.num_envs, horizon + 1, -1))
                else:
                    mat = matrix_from_quat(ori_b)
                    parts.append(mat[..., :2].reshape(self.num_envs, horizon + 1, -1))
                continue

            if self._last_actor_terms is not None and name in self._last_actor_terms:
                cur = self._last_actor_terms[name]
                parts.append(cur.unsqueeze(1).expand(-1, horizon + 1, -1))
            else:
                dim = 0
                if self._last_actor_terms is not None:
                    maybe = self._last_actor_terms.get(name)
                    if maybe is not None:
                        dim = int(maybe.shape[-1])
                parts.append(
                    torch.zeros(
                        (self.num_envs, horizon + 1, dim),
                        device=self.device,
                        dtype=torch.float32,
                    )
                )

        if len(parts) == 0:
            return self._repeat_current_exog(horizon)
        planned_exog = torch.cat(parts, dim=-1)
        if planned_exog.shape[-1] != self.exog_obs_dim:
            # Fallback to exact observed exogenous layout to avoid planner/model shape mismatch.
            return self._repeat_current_exog(horizon)
        return planned_exog

    def close(self):
        self.env.close()
