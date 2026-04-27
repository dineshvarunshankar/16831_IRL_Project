from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(torch.abs(x))


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(torch.abs(x)))


def soft_cross_entropy(logits: torch.Tensor, target_probs: torch.Tensor) -> torch.Tensor:
    log_probs = F.log_softmax(logits, dim=-1)
    return -(target_probs * log_probs).sum(dim=-1)

#categorical regression -predicting a probability distribution over discrete bins
# NN predicts the prob that reward lands in each bucket.

def build_support(
    num_bins: int, 
    value_support: float, #max and min limit of the bins
    #ex: if 20 -> -20 to + 20 which can predict values from -500,000,000 to +500,000,000 using symlog, which split among num of bins
    device: torch.device) -> torch.Tensor:
    return torch.linspace(-value_support, value_support, num_bins, device=device)

#scalar -> prob dist.  
def two_hot_from_scalar(
    values: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    """
    Build two-hot labels over fixed support for transformed scalar targets.
    """
    num_bins = support.shape[0]
    v = values.clamp(support[0], support[-1])
    # to find the index() of the bin to which the value belongs - often its a non integer value
    scale = (num_bins - 1) / (support[-1] - support[0])
    b = (v - support[0]) * scale

    # floor and ceiling of the index
    lo = torch.floor(b).long()
    hi = torch.ceil(b).long()

    #clamping the index to be within the range of the number of bins
    hi = torch.clamp(hi, 0, num_bins - 1)
    lo = torch.clamp(lo, 0, num_bins - 1)

    # to find the how much weight/probability to put in each bins.
    whi = (b - lo.float()).clamp(0.0, 1.0)
    wlo = 1.0 - whi

    out_shape = (*values.shape, num_bins)
    out = torch.zeros(out_shape, device=values.device, dtype=values.dtype)
    #(last dim, which index, what weight/value)
    out.scatter_add_(-1, lo.unsqueeze(-1), wlo.unsqueeze(-1))
    out.scatter_add_(-1, hi.unsqueeze(-1), whi.unsqueeze(-1))
    return out


def logits_to_scalar(
    logits: torch.Tensor,
    support: torch.Tensor,
) -> torch.Tensor:
    probs = F.softmax(logits, dim=-1)
    symlog_value = (probs * support).sum(dim=-1)
    return symexp(symlog_value)


class RunningObsNormalizer(nn.Module):
    """Running mean/std observation normalization.During training the running statistics are
    updated with every new batch of observations; at inference time
    (``update=False``) the frozen statistics are used.
    """

    def __init__(self, dim: int, clip: float = 5.0, eps: float = 1e-5):
        super().__init__()
        self.clip = clip
        self.eps = eps
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def _update_stats(self, x: torch.Tensor) -> None:
        """Welford online update across batch dimension."""
        flat = x.detach().reshape(-1, x.shape[-1])
        batch_mean = flat.mean(dim=0)
        batch_var = flat.var(dim=0, unbiased=False)
        batch_count = flat.shape[0]

        old_count = self.count.item()
        new_count = old_count + batch_count
        if new_count == 0:
            return

        delta = batch_mean - self.mean
        new_mean = self.mean + delta * (batch_count / new_count)
        m_a = self.var * old_count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta.pow(2) * old_count * batch_count / new_count
        new_var = m2 / new_count

        self.mean.copy_(new_mean)
        self.var.copy_(new_var)
        self.count.fill_(new_count)

    def forward(self, x: torch.Tensor, update: bool = False) -> torch.Tensor:
        if update:
            self._update_stats(x)
        std = torch.sqrt(self.var + self.eps)
        normed = (x - self.mean) / std
        return normed.clamp(-self.clip, self.clip)

