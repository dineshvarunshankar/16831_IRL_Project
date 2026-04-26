import argparse
import os
from datetime import datetime

import torch
import wandb

import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.utils.torch import configure_torch_backends

from agents.configs.config_loader import load_tdmpc2_config
from agents.tdmpc2.tdmpc2_agent import TDMPC2Agent
from rl.tdmpc2_env_wrapper import TDMPC2VecEnvWrapper


def _pick_device(config_device: str) -> str:
    if "cuda" in config_device and torch.cuda.is_available():
        return config_device
    return "cpu"


def _episode_length_steps(env_cfg, env: ManagerBasedRlEnv) -> int:
    try:
        return max(1, int(round(float(env_cfg.episode_length_s) / float(env.step_dt))))
    except Exception:
        return 1000


def _mean_metrics(metrics_list: list[dict[str, float]]) -> dict[str, float]:
    if len(metrics_list) == 0:
        return {}
    agg: dict[str, float] = {}
    for metrics in metrics_list:
        for k, v in metrics.items():
            agg[k] = agg.get(k, 0.0) + float(v)
    inv = 1.0 / float(len(metrics_list))
    for k in list(agg.keys()):
        agg[k] *= inv
    return agg


def train():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--config", type=str, default="agents/configs/tdmpc2_config.yaml")
    parser.add_argument("--load", type=str, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--motion-file", type=str, default=None)
    parser.add_argument("--disable-wandb", action="store_true")
    args = parser.parse_args()

    config = load_tdmpc2_config(args.config)
    if args.num_envs is not None:
        config.num_envs = args.num_envs
    if args.task is not None:
        config.task = args.task
    if args.motion_file is not None:
        config.motion_path = args.motion_file
    config.device = _pick_device(config.device)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = args.name or f"tdmpc2_{timestamp}"
    run_model_dir = f"models/{run_name}"
    run_log_dir = f"logs/{run_name}"
    os.makedirs(run_model_dir, exist_ok=True)
    os.makedirs(run_log_dir, exist_ok=True)

    use_wandb = bool(config.use_wandb and (not args.disable_wandb))
    if use_wandb:
        wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=config.__dict__,
        )

    configure_torch_backends()
    env_cfg = load_env_cfg(config.task)

    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    motion_cmd.motion_file = config.motion_path

    env_cfg.auto_reset = False
    env_cfg.scene.num_envs = config.num_envs
    env_cfg.seed = config.seed
    # Keep actor terms as mjlab source of truth but return per-term tensors for splicing.
    env_cfg.observations["actor"].concatenate_terms = False

    env = ManagerBasedRlEnv(cfg=env_cfg, device=config.device)
    env = TDMPC2VecEnvWrapper(
        env=env,
        exogenous_terms=tuple(config.exogenous_terms),
        clip_actions=None,
    )

    print(f"endog_obs_dim: {env.endog_obs_dim}")
    print(f"exog_obs_dim : {env.exog_obs_dim}")
    print(f"act_dim      : {env.act_dim}")

    agent = TDMPC2Agent(
        endog_obs_dim=env.endog_obs_dim,
        exog_obs_dim=env.exog_obs_dim,
        act_dim=env.act_dim,
        config=config,
    )
    if args.load:
        agent.load(args.load)
        print(f"Loaded checkpoint: {args.load}")

    episode_steps = _episode_length_steps(env_cfg, env.env)
    seed_steps_env = (
        int(config.seed_steps)
        if config.seed_steps is not None
        else max(5 * episode_steps, int(config.seed_steps_floor))
    )
    seed_collect_iters = max(
        (seed_steps_env + config.num_envs - 1) // config.num_envs,
        config.horizon + 1,
    )
    print(f"Run       : {run_name}")
    print(f"Device    : {config.device}")
    print(f"Num envs  : {config.num_envs}")
    print(f"Total step: {config.total_steps} ({config.step_unit})")
    print(f"Seed step : {seed_steps_env} env-frames")
    print(f"Seed iters: {seed_collect_iters}")
    print(f"Upd/iter  : {config.updates_per_collect}")

    endog_obs, exog_obs, _ = env.reset()

    episode_returns = torch.zeros(config.num_envs, device=config.device)
    episode_lengths = torch.zeros(config.num_envs, device=config.device)
    episode_num = 0
    recent_returns: list[float] = []
    recent_lengths: list[float] = []
    global_env_step = 0
    collect_iter = 0
    next_log_step = config.log_freq
    next_save_step = config.save_freq

    while global_env_step < config.total_steps:
        if collect_iter < seed_collect_iters:
            action = 2.0 * torch.rand(config.num_envs, env.act_dim, device=config.device) - 1.0
        else:
            planner_env_ids = None
            if config.use_hybrid_acting and config.mpc_train_envs > 0:
                n_plan = min(config.num_envs, config.mpc_train_envs)
                planner_env_ids = torch.randperm(config.num_envs, device=config.device)[:n_plan]
            exog_plan_seq = env.get_exog_plan_sequence(config.horizon)
            action = agent.act(
                endog_obs=endog_obs,
                exog_obs=exog_obs,
                exog_plan_seq=exog_plan_seq,
                deterministic=False,
                planner_env_ids=planner_env_ids,
            )

        next_endog_raw, next_exog_raw, reward, terminated, time_out, _ = env.step(action)

        # Store raw env transition before reset patching.
        agent.store_transition(
            endog_obs=endog_obs,
            exog_obs=exog_obs,
            action=action,
            reward=reward,
            next_endog_obs=next_endog_raw,
            next_exog_obs=next_exog_raw,
            terminated=terminated,
            time_out=time_out,
        )

        episode_returns += reward
        episode_lengths += 1.0

        done_any = terminated | time_out
        next_endog = next_endog_raw
        next_exog = next_exog_raw
        if done_any.any():
            finished = done_any.nonzero(as_tuple=False).squeeze(-1)
            for i in finished:
                recent_returns.append(float(episode_returns[i].item()))
                recent_lengths.append(float(episode_lengths[i].item()))
                episode_num += 1
            episode_returns[finished] = 0.0
            episode_lengths[finished] = 0.0

            reset_endog, reset_exog, _ = env.reset(env_ids=finished)
            next_endog = next_endog.clone()
            next_exog = next_exog.clone()
            next_endog[finished] = reset_endog[finished]
            next_exog[finished] = reset_exog[finished]
            agent.reset_planner(finished)

        metrics = {}
        if collect_iter >= seed_collect_iters:
            agent.set_utd(1)
            update_metrics: list[dict[str, float]] = []
            for _ in range(int(config.updates_per_collect)):
                m = agent.update_from_replay()
                if len(m) > 0:
                    update_metrics.append(m)
            metrics = _mean_metrics(update_metrics)
            metrics["agent/updates_per_collect"] = float(config.updates_per_collect)

        endog_obs = next_endog
        exog_obs = next_exog
        collect_iter += 1
        global_env_step += config.num_envs

        if global_env_step >= next_log_step and len(recent_returns) > 0:
            next_log_step += config.log_freq
            avg_r = sum(recent_returns[-100:]) / min(100, len(recent_returns))
            avg_l = sum(recent_lengths[-100:]) / min(100, len(recent_lengths))
            n_term = float(terminated.float().sum().item())
            n_timeout = float(time_out.float().sum().item())

            log_data = {
                "episode/avg_return": avg_r,
                "episode/avg_steps": avg_l,
                "episode/num": episode_num,
                "episode/terminated_count": n_term,
                "episode/timeout_count": n_timeout,
                "collect_iter": collect_iter,
                "env_step": global_env_step,
            }
            log_data.update(metrics)

            if use_wandb:
                wandb.log(log_data, step=global_env_step)

            print(
                f"Step {global_env_step:10d} | Ep {episode_num:6d} | "
                f"Avg100 R={avg_r:8.2f} L={avg_l:6.1f} | "
                f"term={n_term:.0f} timeout={n_timeout:.0f}"
            )

        if global_env_step >= next_save_step:
            ckpt_path = f"{run_model_dir}/ckpt_step_{next_save_step}.pt"
            agent.save(ckpt_path)
            print(f"Checkpoint saved: {ckpt_path}")
            next_save_step += config.save_freq

    final_path = f"{run_model_dir}/final.pt"
    agent.save(final_path)
    print(f"Final model saved: {final_path}")

    env.close()
    if use_wandb:
        wandb.finish()
    print("Training complete.")


if __name__ == "__main__":
    train()
