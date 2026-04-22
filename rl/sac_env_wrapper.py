import torch
from mjlab.envs import ManagerBasedRlEnv


class EmpiricalNormalization:
    """Running mean/std over a per-feature vector. Welford-style online update.

    Mirrors rsl_rl's EmpiricalNormalization (used by mjlab's PPO runner) and
    FastSAC's obs norm: mean/var accumulated over all envs' observations,
    used to z-score obs before feeding to actor/critic.
    """

    def __init__(self, shape: int, device: torch.device, eps: float = 1e-5):
        self.mean  = torch.zeros(shape, device=device)
        self.var   = torch.ones(shape,  device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean = x.mean(dim=0)
        b_var  = x.var(dim=0, unbiased=False)
        b_n    = x.shape[0]

        tot   = self.count + b_n
        delta = b_mean - self.mean

        new_mean = self.mean + delta * (b_n / tot)
        new_var  = (self.var * self.count + b_var * b_n + (delta ** 2) * self.count * b_n / tot) / tot

        self.mean  = new_mean
        self.var   = new_var
        self.count = tot

    @torch.no_grad()
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / (self.var.sqrt() + 1e-8)

    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, sd: dict) -> None:
        self.mean  = sd["mean"].to(self.mean.device)
        self.var   = sd["var"].to(self.var.device)
        self.count = sd["count"].to(self.count.device)


class SACVecEnvWrapper:
    def __init__(
        self,
        env: ManagerBasedRlEnv,
        clip_actions: float | None = None,
        normalize_obs: bool = True,
    ):
        self.env = env
        self.clip_actions = clip_actions
        self.num_envs = self.env.num_envs
        self.device = torch.device(self.env.device)

        self.actor_obs_key  = "actor"
        self.critic_obs_key = "critic"

        self.actor_obs_dim  = env.single_observation_space.spaces["actor"].shape[0]
        self.critic_obs_dim = env.single_observation_space.spaces["critic"].shape[0]
        self.act_dim        = env.single_action_space.shape[0]

        self.normalize_obs = normalize_obs
        if self.normalize_obs:
            self.actor_rms  = EmpiricalNormalization(self.actor_obs_dim,  self.device)
            self.critic_rms = EmpiricalNormalization(self.critic_obs_dim, self.device)
        else:
            self.actor_rms = self.critic_rms = None

    def _norm(self, actor_obs: torch.Tensor, critic_obs: torch.Tensor, training: bool):
        if not self.normalize_obs:
            return actor_obs, critic_obs
        if training:
            self.actor_rms.update(actor_obs)
            self.critic_rms.update(critic_obs)
        return self.actor_rms(actor_obs), self.critic_rms(critic_obs)

    def step(self, actions: torch.Tensor):
        if self.clip_actions is not None:
            actions = torch.clamp(actions, -self.clip_actions, self.clip_actions)

        obs_dict, rew, terminated, truncated, extras = self.env.step(actions)

        actor_obs  = obs_dict[self.actor_obs_key]
        critic_obs = obs_dict[self.critic_obs_key]
        actor_obs, critic_obs = self._norm(actor_obs, critic_obs, training=True)

        return actor_obs, critic_obs, rew, terminated, truncated, extras

    def reset(self, env_ids: torch.Tensor | None = None):
        obs_dict, extras = self.env.reset(env_ids=env_ids)

        actor_obs  = obs_dict[self.actor_obs_key]
        critic_obs = obs_dict[self.critic_obs_key]
        # do NOT update stats on reset-only obs (biased toward init poses)
        actor_obs, critic_obs = self._norm(actor_obs, critic_obs, training=False)

        return actor_obs, critic_obs, extras

    def state_dict(self) -> dict:
        if not self.normalize_obs:
            return {}
        return {
            "actor_rms":  self.actor_rms.state_dict(),
            "critic_rms": self.critic_rms.state_dict(),
        }

    def load_state_dict(self, sd: dict) -> None:
        if not self.normalize_obs or not sd:
            return
        self.actor_rms.load_state_dict(sd["actor_rms"])
        self.critic_rms.load_state_dict(sd["critic_rms"])

    def close(self):
        self.env.close()
