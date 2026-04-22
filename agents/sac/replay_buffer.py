import torch


class ReplayBuffer:
    """Per-env replay buffer.

    Stores `buffer_size` transitions *per env*, so total capacity is
    `buffer_size * num_envs`. Each env has its own ring; a single scalar
    pointer advances once per `add()` call. Sampling draws uniformly over
    all (step, env) pairs currently populated.
    """

    def __init__(
        self,
        max_size: int,        # per-env capacity
        num_envs: int,
        actor_obs_dim: int,
        critic_obs_dim: int,
        act_dim: int,
        device: torch.device,
    ):
        self.max_size = max_size
        self.num_envs = num_envs
        self.actor_obs_dim = actor_obs_dim
        self.critic_obs_dim = critic_obs_dim
        self.act_dim = act_dim
        self.device = device

        shape = (max_size, num_envs)
        self.actor_obs       = torch.zeros((*shape, actor_obs_dim),  dtype=torch.float32, device=device)
        self.critic_obs      = torch.zeros((*shape, critic_obs_dim), dtype=torch.float32, device=device)
        self.actions         = torch.zeros((*shape, act_dim),        dtype=torch.float32, device=device)
        self.rewards         = torch.zeros((*shape, 1),              dtype=torch.float32, device=device)
        self.next_actor_obs  = torch.zeros((*shape, actor_obs_dim),  dtype=torch.float32, device=device)
        self.next_critic_obs = torch.zeros((*shape, critic_obs_dim), dtype=torch.float32, device=device)
        self.dones           = torch.zeros((*shape, 1),              dtype=torch.float32, device=device)

        self.ptr  = 0  # next row to write (per env)
        self.size = 0  # number of rows populated per env

    def add(
        self,
        actor_obs: torch.Tensor,        # (num_envs, actor_obs_dim)
        critic_obs: torch.Tensor,       # (num_envs, critic_obs_dim)
        action: torch.Tensor,           # (num_envs, act_dim)
        reward: torch.Tensor,           # (num_envs,) or (num_envs, 1)
        next_actor_obs: torch.Tensor,
        next_critic_obs: torch.Tensor,
        done: torch.Tensor,             # (num_envs,) bool/float
    ):
        if reward.dim() == 1:
            reward = reward.unsqueeze(-1)
        if done.dim() == 1:
            done = done.unsqueeze(-1)

        p = self.ptr
        self.actor_obs[p]       = actor_obs
        self.critic_obs[p]      = critic_obs
        self.actions[p]         = action
        self.rewards[p]         = reward
        self.next_actor_obs[p]  = next_actor_obs
        self.next_critic_obs[p] = next_critic_obs
        self.dones[p]           = done.float()

        self.ptr  = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size: int):
        step_idx = torch.randint(0, self.size,      (batch_size,), device=self.device)
        env_idx  = torch.randint(0, self.num_envs,  (batch_size,), device=self.device)

        return (
            self.actor_obs[step_idx, env_idx],
            self.critic_obs[step_idx, env_idx],
            self.actions[step_idx, env_idx],
            self.rewards[step_idx, env_idx],
            self.next_actor_obs[step_idx, env_idx],
            self.next_critic_obs[step_idx, env_idx],
            self.dones[step_idx, env_idx],
        )

    def __len__(self):
        return self.size * self.num_envs
