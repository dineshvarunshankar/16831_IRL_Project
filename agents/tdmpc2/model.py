from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


class SimNorm(nn.Module):
    """
    Project latent vector into groups of simplices using temperature-controlled softmax.
    """

    def __init__(self, group_dim: int, temperature: float = 1.0):

        """
        group dim - the size of each group 
        temperature - modulates the sparsity of the representation.
                    high the temp -> distribution becomes smoother/more uniform.
                    low the temp -> distribution becomes sharper favoring largest number.
        """
        super().__init__()
        self.group_dim = group_dim
        self.temperature = temperature

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        *lead, dim = x.shape
        if dim % self.group_dim != 0:
            raise ValueError(
                f"SimNorm expected last dim divisible by {self.group_dim}, got {dim}"
            )
        groups = dim // self.group_dim
        y = x.view(*lead, groups, self.group_dim) / max(self.temperature, 1e-6)
        y = torch.softmax(y, dim=-1)
        return y.view(*lead, dim)


class NormedLinear(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        activation: str = "mish",
        dropout: float = 0.0,
    ):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim)]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        if activation == "mish":
            layers.append(nn.Mish())
        elif activation == "relu":
            layers.append(nn.ReLU())
        else:
            raise ValueError(f"Unsupported activation: {activation}")
        #unpacked the list in sequential 
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _weight_init(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.Embedding):
        nn.init.uniform_(module.weight, -0.02, 0.02)


def _zero_parameters(params: list[torch.Tensor]) -> None:
    for p in params:
        p.data.fill_(0.0)


@dataclass
class WorldModelSpec:
    endog_dim: int
    exog_dim: int
    act_dim: int
    encoder_dim: int
    mlp_dim: int
    latent_dim: int
    simnorm_dim: int
    simnorm_temp: float
    num_bins: int
    num_q: int
    q_dropout: float
    log_std_min: float
    log_std_max: float
    episodic: bool


class WorldModel(nn.Module):
    def __init__(self, spec: WorldModelSpec):
        super().__init__()
        self.spec = spec
        s = spec
        # Encoder: maps observations to latent representations.
        self.encoder = nn.Sequential(
            NormedLinear(s.endog_dim + s.exog_dim, s.encoder_dim, activation="mish"),
            nn.Linear(s.encoder_dim, s.latent_dim),
            nn.LayerNorm(s.latent_dim),
            SimNorm(s.simnorm_dim, s.simnorm_temp),
        )
        # Dynamics model: models forward dynamics.
        self.dynamics = nn.Sequential(
            NormedLinear(s.latent_dim + s.act_dim + s.exog_dim, s.mlp_dim, activation="mish"),
            NormedLinear(s.mlp_dim, s.mlp_dim, activation="mish"),
            nn.Linear(s.mlp_dim, s.latent_dim),
            nn.LayerNorm(s.latent_dim),
            SimNorm(s.simnorm_dim, s.simnorm_temp),
        )
        # Reward head: predicts reward distribution.
        self.reward_head = nn.Sequential(
            NormedLinear(s.latent_dim + s.act_dim + s.exog_dim, s.mlp_dim, activation="mish"),
            NormedLinear(s.mlp_dim, s.mlp_dim, activation="mish"),
            nn.Linear(s.mlp_dim, s.num_bins),
        )
        # Termination head for episodic tasks.
        self.termination_head = (
            nn.Sequential(
                NormedLinear(s.latent_dim + s.exog_dim, s.mlp_dim, activation="mish"),
                NormedLinear(s.mlp_dim, s.mlp_dim, activation="mish"),
                nn.Linear(s.mlp_dim, 1),
            )
            if s.episodic
            else None
        )
        # Policy prior.
        self.pi_head = nn.Sequential(
            NormedLinear(s.latent_dim + s.exog_dim, s.mlp_dim, activation="mish"),
            NormedLinear(s.mlp_dim, s.mlp_dim, activation="mish"),
            nn.Linear(s.mlp_dim, 2 * s.act_dim),
        )
        # Q heads: distributional critics.
        self.q_heads = nn.ModuleList(
            [
                nn.Sequential(
                    NormedLinear(
                        s.latent_dim + s.act_dim + s.exog_dim,
                        s.mlp_dim,
                        activation="mish",
                        dropout=s.q_dropout,
                    ),
                    NormedLinear(s.mlp_dim, s.mlp_dim, activation="mish"),
                    nn.Linear(s.mlp_dim, s.num_bins),
                )
                for _ in range(s.num_q)
            ]
        )

        self.apply(_weight_init)
        zero_params = [self.reward_head[-1].weight]
        zero_params.extend(head[-1].weight for head in self.q_heads)
        _zero_parameters(zero_params)

        self.log_std_min = s.log_std_min
        self.log_std_max = s.log_std_max

    def encode(self, endog_obs: torch.Tensor, exog_obs: torch.Tensor) -> torch.Tensor:
        x = torch.cat([endog_obs, exog_obs], dim=-1)
        return self.encoder(x)

    def next_latent(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        next_exog_obs: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z, action, next_exog_obs], dim=-1)
        return self.dynamics(x)

    def reward_logits(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        exog_obs: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z, action, exog_obs], dim=-1)
        return self.reward_head(x)

    def termination_logits(
        self,
        z_next: torch.Tensor,
        next_exog_obs: torch.Tensor,
    ) -> torch.Tensor:
        if self.termination_head is None:
            raise RuntimeError("termination_logits called while episodic=False")
        x = torch.cat([z_next, next_exog_obs], dim=-1)
        return self.termination_head(x)

    def q_logits(
        self,
        z: torch.Tensor,
        action: torch.Tensor,
        exog_obs: torch.Tensor,
    ) -> torch.Tensor:
        x = torch.cat([z, action, exog_obs], dim=-1)
        out = [head(x) for head in self.q_heads]
        return torch.stack(out, dim=0)  # [Nq, B, bins]

    def policy_dist_params(
        self,
        z: torch.Tensor,
        exog_obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.cat([z, exog_obs], dim=-1)
        raw = self.pi_head(x)
        mean, log_std = torch.chunk(raw, 2, dim=-1)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
