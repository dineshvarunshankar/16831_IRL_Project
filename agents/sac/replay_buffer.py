from torch import device
import torch 
import numpy as np

# need to make this work with mjlab, using torch tensors instead of numpy arrays

class ReplayBuffer:
    def __init__(
        self, 
        max_size: int,
        num_envs: int, 
        actor_obs_dim: int, 
        critic_obs_dim: int, 
        act_dim: int, 
        device: torch.device
    ):
        self.max_size = max_size # max number of transitions to store
        self.num_envs = num_envs # number of environments
        self.actor_obs_dim = actor_obs_dim # dimension of observation space for actor
        self.critic_obs_dim = critic_obs_dim # dimension of observation space for critic
        self.act_dim = act_dim # dimension of action space
        self.device = device # device to store the data

        self.actor_obs = torch.zeros((max_size, actor_obs_dim), dtype=torch.float32, device=device)
        self.critic_obs = torch.zeros((max_size, critic_obs_dim), dtype=torch.float32, device=device)
        self.actions = torch.zeros((max_size, act_dim), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((max_size, 1), dtype=torch.float32, device=device)
        self.next_actor_obs = torch.zeros((max_size, actor_obs_dim), dtype=torch.float32, device=device)
        self.next_critic_obs = torch.zeros((max_size, critic_obs_dim), dtype=torch.float32, device=device)
        self.dones = torch.zeros((max_size, 1), dtype=torch.float32, device=device)

        self.ptr = 0 # pointer to the next available slot
        self.size = 0 # number of transitions currently stored

    def add(
        self,
        actor_obs: torch.Tensor, # shape: (num_envs, actor_obs_dim)
        critic_obs: torch.Tensor, # shape: (num_envs, critic_obs_dim)
        action: torch.Tensor, # shape: (num_envs, act_dim)
        reward: torch.Tensor, # shape: (num_envs,)
        next_actor_obs: torch.Tensor, # shape: (num_envs, actor_obs_dim)
        next_critic_obs: torch.Tensor, # shape: (num_envs, critic_obs_dim)
        done: torch.Tensor, # shape: (num_envs,)
    ):
        num_envs = self.num_envs

        idx = torch.arange(self.ptr, self.ptr + num_envs, device=self.device) % self.max_size

        self.actor_obs[idx] = actor_obs
        self.critic_obs[idx] = critic_obs
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_actor_obs[idx] = next_actor_obs
        self.next_critic_obs[idx] = next_critic_obs
        self.dones[idx] = done

        self.ptr = (self.ptr + num_envs) % self.max_size # advance pointer and wrap around max_size 
        self.size = min(self.size + num_envs, self.max_size)

    def sample(self, batch_size):

        indices = torch.randint(0, self.size, size=(batch_size, ), device=self.device) # random indices of size 'batch size'

        return (
            self.actor_obs[indices],
            self.critic_obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_actor_obs[indices],
            self.next_critic_obs[indices],
            self.dones[indices]
        )

    def __len__(self):
        return self.size
        