from __future__ import annotations

import torch
import torch.nn as nn


class RunningScale(nn.Module):
    """
    Running trimmed scale estimator (divide-only normalization).
    """

    def __init__(self, tau: float, device: torch.device):
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("value", torch.ones(1, dtype=torch.float32, device=device))
        self.register_buffer(
            "percentiles",
            torch.tensor([0.05, 0.95], dtype=torch.float32, device=device),
        )

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1)
        if flat.numel() == 0:
            return
        p = torch.quantile(flat, self.percentiles)
        scale = torch.clamp(p[1] - p[0], min=1.0)
        self.value.lerp_(scale, self.tau)

    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self.update(x)
        return x / self.value
