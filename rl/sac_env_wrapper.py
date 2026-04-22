import torch
from mjlab.envs import ManagerBasedRlEnv

class SACVecEnvWrapper:
    def __init__(self, env: ManagerBasedRlEnv, clip_actions: float | None = None):
    
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.env.num_envs
        self.device = torch.device(self.env.device)

        self.actor_obs_key = "actor"
        self.critic_obs_key = "critic"

        self.clip_actions = clip_actions

        self.actor_obs_dim  = env.single_observation_space.spaces["actor"].shape[0]
        self.critic_obs_dim = env.single_observation_space.spaces["critic"].shape[0]
        self.act_dim        = env.single_action_space.shape[0]
        self.env.reset()    # call once at init
    

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        actor_obs = obs_dict[self.actor_obs_key]
        critic_obs = obs_dict[self.critic_obs_key]

        return actor_obs, critic_obs, rew, terminated, truncated, extras

    def reset(self, env_ids: torch.Tensor | None = None):
        obs_dict, extras = self.env.reset(env_ids=env_ids)

        actor_obs = obs_dict[self.actor_obs_key]
        critic_obs = obs_dict[self.critic_obs_key]

        return actor_obs, critic_obs, extras
        
    def close(self):
        self.env.close()


    
  