import torch 
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device='cpu'):
        self.capacity = capacity # max number of transitions to store
        self.obs_dim = obs_dim # dimension of observation space
        self.act_dim = act_dim # dimension of action space
        self.device = device # device to store the data

        self.states = np.zeros((capacity, obs_dim), dtype=np.float32)    
        self.actions = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        self.ptr = 0 # pointer to the next available slot
        self.size = 0 # number of transitions currently stored

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.capacity # advance pointer and wrap around capacity 
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):

        indices = np.random.randint(0, self.size, size=batch_size) # random indices of size 'batch size'

        return (
            torch.FloatTensor(self.states[indices]).to(self.device),
            torch.FloatTensor(self.actions[indices]).to(self.device),
            torch.FloatTensor(self.rewards[indices]).to(self.device),
            torch.FloatTensor(self.next_states[indices]).to(self.device),
            torch.FloatTensor(self.dones[indices]).to(self.device)
        )

    def __len__(self):
        return self.size
        