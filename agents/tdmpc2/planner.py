from __future__ import annotations

import torch

from agents.tdmpc2.utils import logits_to_scalar


class MPPIPlanner:
    def __init__(self, cfg, support: torch.Tensor):
        self.cfg = cfg
        self.support = support

    @torch.no_grad()
    def plan(
        self,
        model,
        z: torch.Tensor,
        exog_seq: torch.Tensor,
        prev_mean: torch.Tensor | None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Plan first action for each env using MPPI in latent space.
        """
        device = z.device
        bsz, _ = z.shape
        h = self.cfg.horizon
        a_dim = model.spec.act_dim

        if prev_mean is None or prev_mean.shape[:2] != (bsz, h):
            mean = torch.zeros(bsz, h, a_dim, device=device)
        else:
            mean = prev_mean.clone()

        std = torch.full_like(mean, self.cfg.mpc_max_std)
        std = std.clamp(min=self.cfg.mpc_min_std, max=self.cfg.mpc_max_std)

        s = self.cfg.mpc_samples
        pi_n = min(self.cfg.mpc_policy_samples, s)
        n_iters = int(self.cfg.mpc_iterations)
        if a_dim >= int(self.cfg.mpc_action_dim_iters_threshold):
            n_iters += int(self.cfg.mpc_action_dim_extra_iterations)

        elite_actions = None
        elite_weights = None
        for _ in range(n_iters):
            noise = torch.randn(bsz, s, h, a_dim, device=device)
            actions = (mean[:, None] + std[:, None] * noise).clamp(-1.0, 1.0)

            if pi_n > 0:
                z_roll = z.unsqueeze(1).expand(-1, pi_n, -1).reshape(-1, z.shape[-1])
                for t in range(h):
                    ex_t = (
                        exog_seq[:, t]
                        .unsqueeze(1)
                        .expand(-1, pi_n, -1)
                        .reshape(-1, exog_seq.shape[-1])
                    )
                    ex_tp1 = (
                        exog_seq[:, t + 1]
                        .unsqueeze(1)
                        .expand(-1, pi_n, -1)
                        .reshape(-1, exog_seq.shape[-1])
                    )
                    m, ls = model.policy_dist_params(z_roll, ex_t)
                    if deterministic:
                        a = torch.tanh(m)
                    else:
                        a = torch.tanh(m + ls.exp() * torch.randn_like(m))
                    actions[:, :pi_n, t] = a.view(bsz, pi_n, a_dim)
                    z_roll = model.next_latent(z_roll, a, ex_tp1)

            returns = torch.zeros(bsz, s, device=device)
            gamma_pow = 1.0
            z_roll = z.unsqueeze(1).expand(-1, s, -1).reshape(-1, z.shape[-1])

            for t in range(h):
                a_t = actions[:, :, t].reshape(-1, a_dim)
                ex_t = (
                    exog_seq[:, t]
                    .unsqueeze(1)
                    .expand(-1, s, -1)
                    .reshape(-1, exog_seq.shape[-1])
                )
                ex_tp1 = (
                    exog_seq[:, t + 1]
                    .unsqueeze(1)
                    .expand(-1, s, -1)
                    .reshape(-1, exog_seq.shape[-1])
                )
                r_logits = model.reward_logits(z_roll, a_t, ex_t)
                r = logits_to_scalar(r_logits, self.support).view(bsz, s)
                returns += gamma_pow * r
                z_roll = model.next_latent(z_roll, a_t, ex_tp1)
                gamma_pow *= self.cfg.gamma

            ex_h = (
                exog_seq[:, h]
                .unsqueeze(1)
                .expand(-1, s, -1)
                .reshape(-1, exog_seq.shape[-1])
            )
            m_pi, _ = model.policy_dist_params(z_roll, ex_h)
            a_pi = torch.tanh(m_pi)
            q_logits = model.q_logits(z_roll, a_pi, ex_h)
            q_idx = torch.randperm(q_logits.shape[0], device=q_logits.device)[:2]
            q_pair = logits_to_scalar(q_logits[q_idx], self.support)
            q_avg = q_pair.mean(dim=0).view(bsz, s)
            returns += gamma_pow * q_avg

            elite_vals, elite_idx = torch.topk(
                returns, k=self.cfg.mpc_elites, dim=1, largest=True
            )
            elite_actions = actions.gather(
                dim=1,
                index=elite_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, h, a_dim),
            )

            scaled = self.cfg.mpc_temperature * (
                elite_vals - elite_vals.max(dim=1, keepdim=True).values
            )
            elite_weights = torch.softmax(scaled, dim=1)
            w = elite_weights.unsqueeze(-1).unsqueeze(-1)
            mean = (w * elite_actions).sum(dim=1)
            var = (w * (elite_actions - mean.unsqueeze(1)) ** 2).sum(dim=1)
            std = torch.sqrt(var + 1e-6).clamp(self.cfg.mpc_min_std, self.cfg.mpc_max_std)

        if elite_actions is None or elite_weights is None:
            raise RuntimeError("MPPI planner failed to produce elite trajectories")

        if deterministic:
            pick = elite_weights.argmax(dim=1)
        else:
            u = torch.rand_like(elite_weights).clamp_(1e-6, 1.0 - 1e-6)
            gumbel = -torch.log(-torch.log(u))
            pick = torch.argmax(torch.log(elite_weights + 1e-8) + gumbel, dim=1)
        first_action = elite_actions[torch.arange(bsz, device=device), pick, 0]

        next_mean = torch.zeros_like(mean)
        next_mean[:, :-1] = mean[:, 1:]
        return first_action, next_mean
