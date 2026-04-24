import torch
import torch.nn as nn
import torch.nn.functional as F


class DistributionalQNetwork(nn.Module):
    def __init__(
        self,
        critic_obs_dim: int,
        act_dim: int,
        hidden_dims: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        layers = []
        in_dim = critic_obs_dim + act_dim
        for out_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, out_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(out_dim))
            layers.append(nn.ReLU())
            in_dim = out_dim
        layers.append(nn.Linear(in_dim, num_atoms))
        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, action], dim=-1))

    def projection(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        rewards: torch.Tensor,          # [batch] — may include entropy-adjusted reward
        bootstrap: torch.Tensor,        # [batch] — (1 - done)
        discount: torch.Tensor | float, # scalar (gamma) for 1-step
        q_support: torch.Tensor,        # [num_atoms]
    ) -> torch.Tensor:
        """Shift this network's predicted next-state distribution by the Bellman
        update and project it back onto the fixed atom support."""
        device = q_support.device
        delta_z = (self.v_max - self.v_min) / (self.num_atoms - 1)
        batch_size = rewards.shape[0]

        if not torch.is_tensor(discount):
            discount_t = torch.as_tensor(discount, device=device, dtype=q_support.dtype)
        else:
            discount_t = discount

        # Shift atoms: target_z[i,j] = r_i + bootstrap_i * discount * z_j
        target_z = rewards.unsqueeze(1) + bootstrap.unsqueeze(1) * discount_t * q_support
        target_z = target_z.clamp(self.v_min, self.v_max)

        # Fractional bin index for each shifted atom
        b = (target_z - self.v_min) / delta_z
        lower = torch.floor(b).long()
        upper = torch.ceil(b).long()

        # If lower == upper (exactly on an atom), push them apart so we don't
        # double-count. Mirror HoloSoma's handling (special-case boundary too).
        is_integer = upper == lower
        lower_mask = torch.logical_and(lower > 0, is_integer)
        upper_mask = torch.logical_and(lower == 0, is_integer)
        lower = torch.where(lower_mask, lower - 1, lower)
        upper = torch.where(upper_mask, upper + 1, upper)

        next_dist = F.softmax(self(state, action), dim=1)
        proj_dist = torch.zeros_like(next_dist)

        offset = (
            torch.linspace(0, (batch_size - 1) * self.num_atoms, batch_size, device=device)
            .unsqueeze(1)
            .expand(batch_size, self.num_atoms)
            .long()
        )

        lower_indices = (lower + offset).view(-1)
        upper_indices = (upper + offset).view(-1)
        max_index = proj_dist.numel() - 1
        lower_indices = torch.clamp(lower_indices, 0, max_index)
        upper_indices = torch.clamp(upper_indices, 0, max_index)

        proj_dist.view(-1).index_add_(0, lower_indices, (next_dist * (upper.float() - b)).view(-1))
        proj_dist.view(-1).index_add_(0, upper_indices, (next_dist * (b - lower.float())).view(-1))
        return proj_dist


class DistributionalDoubleCritic(nn.Module):
    """Two independent distributional Q-networks, matching the DoubleCritic
    interface so the SAC agent can swap between them via a config flag."""

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_dims: list[int],
        num_atoms: int,
        v_min: float,
        v_max: float,
        use_layer_norm: bool = False,
    ):
        super().__init__()
        self.num_atoms = num_atoms
        self.v_min = v_min
        self.v_max = v_max

        self.Q1 = DistributionalQNetwork(
            obs_dim, act_dim, hidden_dims, num_atoms, v_min, v_max, use_layer_norm
        )
        self.Q2 = DistributionalQNetwork(
            obs_dim, act_dim, hidden_dims, num_atoms, v_min, v_max, use_layer_norm
        )

        support = torch.linspace(v_min, v_max, num_atoms)
        self.register_buffer("q_support", support)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns raw logits (pre-softmax) for each critic."""
        return self.Q1(state, action), self.Q2(state, action)

    def get_value(self, probs: torch.Tensor) -> torch.Tensor:
        """Scalar expected value from a distribution: sum(p_i * z_i)."""
        return (probs * self.q_support).sum(dim=-1, keepdim=True)

    def expected_Q(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Both expected values as scalars [batch, 1]."""
        l1, l2 = self(state, action)
        p1 = F.softmax(l1, dim=-1)
        p2 = F.softmax(l2, dim=-1)
        return self.get_value(p1), self.get_value(p2)

    def min_Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.expected_Q(state, action)
        return torch.min(q1, q2)

    def mean_Q(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q1, q2 = self.expected_Q(state, action)
        return 0.5 * (q1 + q2)

    def project_targets(
        self,
        next_state: torch.Tensor,
        next_action: torch.Tensor,
        rewards: torch.Tensor,
        bootstrap: torch.Tensor,
        discount: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Projected target distributions for both critics. Caller supplies the
        entropy-adjusted reward (r - alpha * log_prob_next scaled appropriately)."""
        p1 = self.Q1.projection(next_state, next_action, rewards, bootstrap, discount, self.q_support)
        p2 = self.Q2.projection(next_state, next_action, rewards, bootstrap, discount, self.q_support)
        return p1, p2
