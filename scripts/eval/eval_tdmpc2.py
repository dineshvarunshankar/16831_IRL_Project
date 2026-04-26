import argparse

import torch

import mjlab.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from agents.configs.config_loader import load_tdmpc2_config
from agents.tdmpc2.tdmpc2_agent import TDMPC2Agent
from rl.tdmpc2_env_wrapper import TDMPC2VecEnvWrapper


def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--config", type=str, default="agents/configs/tdmpc2_config.yaml")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--motion-file", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--episodes", type=int, default=10)
    args = parser.parse_args()

    cfg = load_tdmpc2_config(args.config)
    if args.task is not None:
        cfg.task = args.task
    if args.motion_file is not None:
        cfg.motion_path = args.motion_file
    cfg.num_envs = max(1, args.num_envs)
    if "cuda" in cfg.device and not torch.cuda.is_available():
        cfg.device = "cpu"

    configure_torch_backends()
    env_cfg = load_env_cfg(cfg.task, play=True)
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = cfg.motion_path
    env_cfg.auto_reset = False
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.observations["actor"].concatenate_terms = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device)
    env = TDMPC2VecEnvWrapper(env, exogenous_terms=tuple(cfg.exogenous_terms), clip_actions=None)

    agent = TDMPC2Agent(
        endog_obs_dim=env.endog_obs_dim,
        exog_obs_dim=env.exog_obs_dim,
        act_dim=env.act_dim,
        config=cfg,
    )
    agent.load(args.checkpoint)

    endog_obs, exog_obs, _ = env.reset()
    episode_returns = torch.zeros(cfg.num_envs, device=cfg.device)
    completed: list[float] = []

    while len(completed) < args.episodes:
        planner_env_ids = None
        if cfg.mpc_eval:
            planner_env_ids = torch.arange(cfg.num_envs, device=cfg.device)
        exog_plan_seq = env.get_exog_plan_sequence(cfg.horizon)

        action = agent.act(
            endog_obs=endog_obs,
            exog_obs=exog_obs,
            exog_plan_seq=exog_plan_seq,
            deterministic=True,
            planner_env_ids=planner_env_ids,
        )
        next_endog, next_exog, reward, terminated, time_out, _ = env.step(action)
        episode_returns += reward

        done_any = terminated | time_out
        if done_any.any():
            finished = done_any.nonzero(as_tuple=False).squeeze(-1)
            for idx in finished:
                completed.append(float(episode_returns[idx].item()))
                if len(completed) >= args.episodes:
                    break
            episode_returns[finished] = 0.0
            reset_endog, reset_exog, _ = env.reset(env_ids=finished)
            next_endog = next_endog.clone()
            next_exog = next_exog.clone()
            next_endog[finished] = reset_endog[finished]
            next_exog[finished] = reset_exog[finished]
            agent.reset_planner(finished)

        endog_obs = next_endog
        exog_obs = next_exog

    env.close()
    mean_return = sum(completed) / max(len(completed), 1)
    print(f"Evaluated {len(completed)} episodes | mean return: {mean_return:.3f}")


if __name__ == "__main__":
    evaluate()
