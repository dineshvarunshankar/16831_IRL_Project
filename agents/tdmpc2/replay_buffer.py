from __future__ import annotations

import torch


class SequenceReplayBuffer:
    """
    Per-environment circular replay with vectorized uniform slice sampling.
    """

    def __init__(
        self,
        num_envs: int,
        capacity: int,
        endog_dim: int,
        exog_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        self.num_envs = int(num_envs)
        self.capacity = int(capacity)
        self.endog_dim = int(endog_dim)
        self.exog_dim = int(exog_dim)
        self.act_dim = int(act_dim)
        self.device = device

        self.per_env_capacity = max(self.capacity // max(self.num_envs, 1), 16)
        n = self.num_envs
        t = self.per_env_capacity

        cpu = torch.device("cpu")
        self.endog_obs = torch.zeros((n, t, endog_dim), dtype=torch.float32, device=cpu)
        self.exog_obs = torch.zeros((n, t, exog_dim), dtype=torch.float32, device=cpu)
        self.actions = torch.zeros((n, t, act_dim), dtype=torch.float32, device=cpu)
        self.rewards = torch.zeros((n, t, 1), dtype=torch.float32, device=cpu)
        self.next_endog_obs = torch.zeros((n, t, endog_dim), dtype=torch.float32, device=cpu)
        self.next_exog_obs = torch.zeros((n, t, exog_dim), dtype=torch.float32, device=cpu)
        self.terminated = torch.zeros((n, t, 1), dtype=torch.float32, device=cpu)
        self.time_out = torch.zeros((n, t, 1), dtype=torch.float32, device=cpu)
        self.step_id = torch.full((n, t), -1, dtype=torch.long, device=cpu)
        self.episode_id = torch.full((n, t), -1, dtype=torch.long, device=cpu)
        self.valid = torch.zeros((n, t), dtype=torch.bool, device=cpu)
        self.env_episode_id = torch.zeros((n,), dtype=torch.long, device=cpu)

        self.ptr = 0
        self.step = 0
        self.size = 0

    def __len__(self) -> int:
        return int(self.size)

    @torch.no_grad()
    def add(
        self,
        endog_obs: torch.Tensor,
        exog_obs: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_endog_obs: torch.Tensor,
        next_exog_obs: torch.Tensor,
        terminated: torch.Tensor,
        time_out: torch.Tensor,
    ) -> None:
        idx = self.ptr

        self.endog_obs[:, idx].copy_(endog_obs.detach().to("cpu"))
        self.exog_obs[:, idx].copy_(exog_obs.detach().to("cpu"))
        self.actions[:, idx].copy_(action.detach().to("cpu"))
        reward_cpu = reward.detach().to("cpu").reshape(self.num_envs, -1)
        self.rewards[:, idx].copy_(reward_cpu[:, :1])
        self.next_endog_obs[:, idx].copy_(next_endog_obs.detach().to("cpu"))
        self.next_exog_obs[:, idx].copy_(next_exog_obs.detach().to("cpu"))
        terminated_cpu = terminated.detach().to("cpu").reshape(self.num_envs, -1).float()
        timeout_cpu = time_out.detach().to("cpu").reshape(self.num_envs, -1).float()
        self.terminated[:, idx].copy_(terminated_cpu[:, :1])
        self.time_out[:, idx].copy_(timeout_cpu[:, :1])

        self.step_id[:, idx] = self.step
        self.episode_id[:, idx] = self.env_episode_id
        self.valid[:, idx] = True

        done = (terminated_cpu[:, :1] + timeout_cpu[:, :1]).squeeze(-1) > 0.0
        self.env_episode_id[done] += 1

        self.ptr = (self.ptr + 1) % self.per_env_capacity
        self.step += 1
        self.size = min(self.size + self.num_envs, self.capacity)

    def _sequence_ok(self, env_id: int, start: int, horizon: int) -> bool:
        idx = (start + torch.arange(horizon, device=self.valid.device)) % self.per_env_capacity
        if not bool(self.valid[env_id, idx].all()):
            return False
        ids = self.step_id[env_id, idx]
        eps = self.episode_id[env_id, idx]
        return bool(torch.all(ids[1:] == ids[:-1] + 1) and torch.all(eps == eps[0]))

    @torch.no_grad()
    def _valid_start_mask(self, horizon: int) -> torch.Tensor:
        starts = torch.arange(self.per_env_capacity, device=self.valid.device)
        offs = torch.arange(horizon, device=self.valid.device)
        idx = (starts[:, None] + offs[None, :]) % self.per_env_capacity  # [T, H]

        valid = self.valid[:, idx].all(dim=-1)
        step_ids = self.step_id[:, idx]
        episode_ids = self.episode_id[:, idx]

        if horizon > 1:
            contiguous = (step_ids[:, :, 1:] == (step_ids[:, :, :-1] + 1)).all(dim=-1)
            same_episode = (episode_ids[:, :, 1:] == episode_ids[:, :, :-1]).all(dim=-1)
            valid = valid & contiguous & same_episode
        return valid

    @torch.no_grad()
    def sample_sequence(self, batch_size: int, horizon: int) -> dict[str, torch.Tensor]:
        if self.size < (horizon + 1) * self.num_envs:
            raise RuntimeError("Replay buffer does not have enough data for sequence sampling")
        valid_mask = self._valid_start_mask(horizon)
        flat_valid = valid_mask.reshape(-1)
        valid_flat_idx = torch.nonzero(flat_valid, as_tuple=False).squeeze(-1)
        if valid_flat_idx.numel() < batch_size:
            raise RuntimeError(
                f"Could not sample {batch_size} valid sequences (got {int(valid_flat_idx.numel())})."
            )

        pick = torch.randint(0, valid_flat_idx.numel(), (batch_size,), device=self.valid.device)
        picked = valid_flat_idx[pick]
        env_idx = picked // self.per_env_capacity
        start_idx = picked % self.per_env_capacity

        offs = torch.arange(horizon, dtype=torch.long, device=self.valid.device)
        trans_idx = (start_idx[:, None] + offs[None, :]) % self.per_env_capacity

        env_expand = env_idx[:, None]
        endog0 = self.endog_obs[env_idx, trans_idx[:, 0]]
        exog0 = self.exog_obs[env_idx, trans_idx[:, 0]]
        next_endog = self.next_endog_obs[env_expand, trans_idx]
        next_exog = self.next_exog_obs[env_expand, trans_idx]
        endog_seq = torch.cat([endog0.unsqueeze(1), next_endog], dim=1)
        exog_seq = torch.cat([exog0.unsqueeze(1), next_exog], dim=1)

        out = {
            "endog_seq": endog_seq.to(self.device, non_blocking=True),
            "exog_seq": exog_seq.to(self.device, non_blocking=True),
            "actions": self.actions[env_expand, trans_idx].to(self.device, non_blocking=True),
            "rewards": self.rewards[env_expand, trans_idx].to(self.device, non_blocking=True),
            "terminated": self.terminated[env_expand, trans_idx].to(self.device, non_blocking=True),
            "time_out": self.time_out[env_expand, trans_idx].to(self.device, non_blocking=True),
        }
        return out
