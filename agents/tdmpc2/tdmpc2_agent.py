from __future__ import annotations

from copy import deepcopy
from typing import Dict

import torch
import torch.nn.functional as F
from torch.distributions import Normal

from agents.base import BaseAlgorithm
from agents.tdmpc2.model import WorldModel, WorldModelSpec
from agents.tdmpc2.planner import MPPIPlanner
from agents.tdmpc2.replay_buffer import SequenceReplayBuffer
from agents.tdmpc2.scale import RunningScale
from agents.tdmpc2.utils import (
    build_support,
    logits_to_scalar,
    soft_cross_entropy,
    symlog,
    two_hot_from_scalar,
)


class TDMPC2Agent(BaseAlgorithm):
    def __init__(self, endog_obs_dim: int, exog_obs_dim: int, act_dim: int, config):
        self.config = config
        self.device = torch.device(config.device)
        self.endog_obs_dim = int(endog_obs_dim)
        self.exog_obs_dim = int(exog_obs_dim)
        self.act_dim = int(act_dim)

        spec = WorldModelSpec(
            endog_dim=self.endog_obs_dim,
            exog_dim=self.exog_obs_dim,
            act_dim=self.act_dim,
            encoder_dim=config.encoder_dim,
            mlp_dim=config.mlp_dim,
            latent_dim=config.latent_dim,
            simnorm_dim=config.simnorm_dim,
            simnorm_temp=config.simnorm_temp,
            num_bins=config.num_bins,
            num_q=config.num_q,
            q_dropout=config.q_dropout,
            log_std_min=config.log_std_min,
            log_std_max=config.log_std_max,
            episodic=config.episodic,
        )

        self.model = WorldModel(spec).to(self.device)
        self.target_q_heads = deepcopy(self.model.q_heads).to(self.device)
        for p in self.target_q_heads.parameters():
            p.requires_grad_(False)

        self.support = build_support(config.num_bins, config.value_support, self.device)
        self.planner = MPPIPlanner(config, self.support)

        self.replay = SequenceReplayBuffer(
            num_envs=config.num_envs,
            capacity=config.buffer_size,
            endog_dim=self.endog_obs_dim,
            exog_dim=self.exog_obs_dim,
            act_dim=self.act_dim,
            device=self.device,
        )

        model_dynamics_params = (
            list(self.model.dynamics.parameters())
            + list(self.model.reward_head.parameters())
            + list(self.model.q_heads.parameters())
        )
        if self.model.termination_head is not None:
            model_dynamics_params += list(self.model.termination_head.parameters())

        model_params = [
            {
                "params": list(self.model.encoder.parameters()),
                "lr": config.encoder_lr,
            },
            {
                "params": model_dynamics_params,
                "lr": config.lr,
            },
        ]
        self.model_optimizer = torch.optim.Adam(model_params, lr=config.lr)
        self.policy_optimizer = torch.optim.Adam(
            self.model.pi_head.parameters(),
            lr=config.lr,
            eps=1e-5,
        )

        self.current_utd = int(config.utd_start)
        self.global_updates = 0

        self.plan_mean = torch.zeros(
            config.num_envs,
            config.horizon,
            self.act_dim,
            device=self.device,
        )
        self.q_scale = RunningScale(tau=config.tau, device=self.device)
        self.last_metrics: Dict[str, float] = {}

    def set_utd(self, utd: int) -> None:
        self.current_utd = max(1, int(utd))

    @torch.no_grad()
    def reset_planner(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.plan_mean[env_ids] = 0.0

    def _sample_policy(
        self, z: torch.Tensor, exog_obs: torch.Tensor, deterministic: bool
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.model.policy_dist_params(z, exog_obs)
        if deterministic:
            u = mean
        else:
            std = log_std.exp()
            u = mean + std * torch.randn_like(mean)
        action = torch.tanh(u)

        std = log_std.exp()
        dist = Normal(mean, std)
        log_prob = dist.log_prob(u) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

    def _sample_two_q_values(self, q_logits: torch.Tensor) -> torch.Tensor:
        idx = torch.randperm(q_logits.shape[0], device=q_logits.device)[:2]
        return logits_to_scalar(q_logits[idx], self.support)

    @torch.no_grad()
    def _target_q_scalar(
        self,
        z_next: torch.Tensor,
        exog_next: torch.Tensor,
    ) -> torch.Tensor:
        a_next, _ = self._sample_policy(z_next, exog_next, deterministic=False)
        x = torch.cat([z_next, a_next, exog_next], dim=-1)
        logits = torch.stack([head(x) for head in self.target_q_heads], dim=0)
        q1, q2 = self._sample_two_q_values(logits)
        return torch.minimum(q1, q2).unsqueeze(-1)

    def _model_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        endog_seq = batch["endog_seq"]
        exog_seq = batch["exog_seq"]
        actions = batch["actions"]
        rewards = batch["rewards"]
        terminated = batch["terminated"]
        time_out = batch["time_out"]
        done_any = torch.clamp(terminated + time_out, 0.0, 1.0)

        horizon = actions.shape[1]
        z = self.model.encode(endog_seq[:, 0], exog_seq[:, 0])
        with torch.no_grad():
            z_target_all = self.model.encode(
                endog_seq[:, 1:].reshape(-1, self.endog_obs_dim),
                exog_seq[:, 1:].reshape(-1, self.exog_obs_dim),
            ).reshape(endog_seq.shape[0], horizon, -1)

        cons_loss = torch.zeros((), device=self.device)
        rew_loss = torch.zeros((), device=self.device)
        val_loss = torch.zeros((), device=self.device)
        term_loss = torch.zeros((), device=self.device)
        weight_sum = torch.zeros((), device=self.device)
        alive = torch.ones((endog_seq.shape[0], 1), device=self.device)

        for t in range(horizon):
            w = (self.config.temporal_coef ** t) * alive
            weight_sum = weight_sum + w.sum()

            z_next_pred = self.model.next_latent(z, actions[:, t], exog_seq[:, t + 1])
            z_target = z_target_all[:, t]
            cons_step = ((z_next_pred - z_target) ** 2).mean(dim=-1, keepdim=True)
            cons_loss = cons_loss + (w * cons_step).sum()

            r_logits = self.model.reward_logits(z, actions[:, t], exog_seq[:, t])
            r_target = two_hot_from_scalar(symlog(rewards[:, t, 0]), self.support)
            r_ce = soft_cross_entropy(r_logits, r_target).unsqueeze(-1)
            rew_loss = rew_loss + (w * r_ce).sum()

            with torch.no_grad():
                q_next = self._target_q_scalar(z_target, exog_seq[:, t + 1])
                td_target = rewards[:, t] + self.config.gamma * (1.0 - terminated[:, t]) * q_next
                td_target_probs = two_hot_from_scalar(symlog(td_target[:, 0]), self.support)

            q_logits = self.model.q_logits(z, actions[:, t], exog_seq[:, t])
            q_ces = []
            for qi in range(q_logits.shape[0]):
                q_ces.append(soft_cross_entropy(q_logits[qi], td_target_probs).unsqueeze(-1))
            q_ce = torch.stack(q_ces, dim=0).mean(dim=0)
            val_loss = val_loss + (w * q_ce).sum()

            if self.model.termination_head is not None:
                term_logits = self.model.termination_logits(z_next_pred, exog_seq[:, t + 1]).squeeze(-1)
                term_target = terminated[:, t, 0]
                term_bce = F.binary_cross_entropy_with_logits(
                    term_logits, term_target, reduction="none"
                ).unsqueeze(-1)
                term_loss = term_loss + (w * term_bce).sum()

            z = z_next_pred
            alive = alive * (1.0 - done_any[:, t])

        denom = torch.clamp(weight_sum, min=1.0)
        cons_loss = cons_loss / denom
        rew_loss = rew_loss / denom
        val_loss = val_loss / denom
        term_loss = term_loss / denom

        total = (
            self.config.consistency_coef * cons_loss
            + self.config.reward_coef * rew_loss
            + self.config.value_coef * val_loss
            + self.config.termination_coef * term_loss
        )
        metrics = {
            "loss/consistency": float(cons_loss.item()),
            "loss/reward_ce": float(rew_loss.item()),
            "loss/value_ce": float(val_loss.item()),
            "loss/termination_bce": float(term_loss.item()),
            "loss/model_total": float(total.item()),
        }
        return total, metrics

    def _policy_loss(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, dict[str, float]]:
        endog_seq = batch["endog_seq"]
        exog_seq = batch["exog_seq"]
        horizon = endog_seq.shape[1] - 1

        with torch.no_grad():
            z = self.model.encode(endog_seq[:, 0], exog_seq[:, 0])

        actor_terms = []
        q_values = []
        for t in range(horizon):
            a_pi, log_prob = self._sample_policy(z, exog_seq[:, t], deterministic=False)
            q_logits = self.model.q_logits(z, a_pi, exog_seq[:, t])
            q_pair = self._sample_two_q_values(q_logits)
            q_avg = q_pair.mean(dim=0)
            q_values.append(q_avg.detach())

            scaled_entropy = -log_prob.squeeze(-1) / float(self.act_dim)
            actor_terms.append((q_avg, scaled_entropy))

            with torch.no_grad():
                z = self.model.next_latent(z, a_pi, exog_seq[:, t + 1])

        if len(q_values) == 0:
            zero = torch.zeros((), device=self.device)
            return zero, {"loss/policy": 0.0, "agent/q_scale": float(self.q_scale.value.item())}

        q_stack = torch.stack(q_values, dim=0)
        self.q_scale.update(q_stack[0])

        loss = torch.zeros((), device=self.device)
        for t, (q_avg, entropy_scaled) in enumerate(actor_terms):
            q_norm = self.q_scale(q_avg)
            term = -(q_norm + self.config.entropy_coef * entropy_scaled).mean()
            loss = loss + (self.config.temporal_coef ** t) * term
        loss = loss / max(horizon, 1)
        metrics = {
            "loss/policy": float(loss.item()),
            "agent/q_scale": float(self.q_scale.value.item()),
        }
        return loss, metrics

    @torch.no_grad()
    def _ema_update_q_target(self) -> None:
        tau = float(self.config.tau)
        for src, tgt in zip(self.model.q_heads.parameters(), self.target_q_heads.parameters()):
            tgt.data.lerp_(src.data, tau)

    def act(
        self,
        endog_obs: torch.Tensor,
        exog_obs: torch.Tensor,
        exog_plan_seq: torch.Tensor | None = None,
        deterministic: bool = False,
        planner_env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with torch.no_grad():
            z = self.model.encode(endog_obs, exog_obs)
            action, _ = self._sample_policy(z, exog_obs, deterministic=deterministic)

            if planner_env_ids is not None and planner_env_ids.numel() > 0:
                z_sub = z[planner_env_ids]
                if exog_plan_seq is None:
                    exog_plan_seq = exog_obs.unsqueeze(1).expand(-1, self.config.horizon + 1, -1)
                ex_seq_sub = exog_plan_seq[planner_env_ids]
                prev_mean = self.plan_mean[planner_env_ids]
                mpc_action, next_mean = self.planner.plan(
                    model=self.model,
                    z=z_sub,
                    exog_seq=ex_seq_sub,
                    prev_mean=prev_mean,
                    deterministic=deterministic,
                )
                action[planner_env_ids] = mpc_action
                self.plan_mean[planner_env_ids] = next_mean
        return action

    def select_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if isinstance(obs, tuple) and len(obs) == 2:
            endog_obs, exog_obs = obs
            return self.act(endog_obs, exog_obs, deterministic=deterministic)
        raise ValueError(
            "TDMPC2Agent.select_action expects obs=(endog_obs, exog_obs) tuple for this implementation."
        )

    def store_transition(self, **kwargs) -> None:
        self.replay.add(
            endog_obs=kwargs["endog_obs"],
            exog_obs=kwargs["exog_obs"],
            action=kwargs["action"],
            reward=kwargs["reward"],
            next_endog_obs=kwargs["next_endog_obs"],
            next_exog_obs=kwargs["next_exog_obs"],
            terminated=kwargs["terminated"],
            time_out=kwargs["time_out"],
        )

    # Compatibility alias.
    def collect(self, **kwargs) -> None:
        self.store_transition(**kwargs)

    def update_from_replay(self) -> Dict[str, float]:
        if len(self.replay) < self.config.batch_size:
            return {}

        agg: Dict[str, float] = {}
        for _ in range(self.current_utd):
            batch = self.replay.sample_sequence(
                batch_size=self.config.batch_size,
                horizon=self.config.horizon,
            )

            model_loss, model_metrics = self._model_loss(batch)
            self.model_optimizer.zero_grad(set_to_none=True)
            model_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
            self.model_optimizer.step()

            policy_loss, policy_metrics = self._policy_loss(batch)
            self.policy_optimizer.zero_grad(set_to_none=True)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.pi_head.parameters(), self.config.grad_clip_norm
            )
            self.policy_optimizer.step()

            self._ema_update_q_target()
            self.global_updates += 1

            merged = {**model_metrics, **policy_metrics}
            for k, v in merged.items():
                agg[k] = agg.get(k, 0.0) + float(v)

        for k in list(agg.keys()):
            agg[k] /= float(self.current_utd)
        self.last_metrics = agg
        return agg

    # Compatibility with existing interface.
    def update(self, *args, **kwargs) -> Dict[str, float]:
        return self.update_from_replay()

    def get_metrics(self) -> Dict[str, float]:
        return dict(self.last_metrics)

    def save(self, path: str) -> None:
        torch.save(
            {
                "model": self.model.state_dict(),
                "target_q_heads": self.target_q_heads.state_dict(),
                "model_optimizer": self.model_optimizer.state_dict(),
                "policy_optimizer": self.policy_optimizer.state_dict(),
                "q_scale": self.q_scale.state_dict(),
                "config": self.config.__dict__,
                "global_updates": self.global_updates,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.target_q_heads.load_state_dict(ckpt["target_q_heads"])
        self.model_optimizer.load_state_dict(ckpt["model_optimizer"])
        self.policy_optimizer.load_state_dict(ckpt["policy_optimizer"])
        if "q_scale" in ckpt:
            self.q_scale.load_state_dict(ckpt["q_scale"])
        self.global_updates = int(ckpt.get("global_updates", 0))
