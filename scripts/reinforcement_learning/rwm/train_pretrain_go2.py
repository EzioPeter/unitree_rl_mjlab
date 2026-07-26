"""Train a Go2 RWM dynamics ensemble from live mjlab PPO rollouts."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", default="Unitree-Go2-Flat-RWM-Pretrain-Ens")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--num_steps_per_env", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--run_name", default="")
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default=None)
    parser.add_argument("--system_dynamics_mini_batch_size", type=int, default=None)
    parser.add_argument("--system_dynamics_updates_per_iter", type=int, default=None)
    parser.add_argument("--system_dynamics_min_transitions", type=int, default=None)
    parser.add_argument("--system_dynamics_replay_capacity", type=int, default=None)
    parser.add_argument("--history_horizon", type=int, default=None)
    parser.add_argument("--forecast_horizon", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    os.environ.setdefault("MUJOCO_GL", "egl")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(args.task)
    agent_cfg = load_rl_cfg(args.task)
    if args.seed is not None:
        env_cfg.seed = args.seed
        agent_cfg.seed = args.seed
    if args.num_envs is not None:
        env_cfg.scene.num_envs = args.num_envs
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    if args.run_name:
        agent_cfg.run_name = args.run_name
    if args.logger is not None:
        agent_cfg.logger = args.logger

    print(f"[Go2-RWM] Stage 1 pretrain: task={args.task}, device={device}, num_envs={env_cfg.scene.num_envs}")
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    vec_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    train_cfg = asdict(agent_cfg)
    rwm_overrides = {}
    if args.system_dynamics_mini_batch_size is not None:
        rwm_overrides["batch_size"] = args.system_dynamics_mini_batch_size
    if args.system_dynamics_updates_per_iter is not None:
        rwm_overrides["updates_per_iter"] = args.system_dynamics_updates_per_iter
    if args.system_dynamics_min_transitions is not None:
        rwm_overrides["min_transitions"] = args.system_dynamics_min_transitions
    if args.system_dynamics_replay_capacity is not None:
        rwm_overrides["capacity"] = args.system_dynamics_replay_capacity
    if args.history_horizon is not None:
        rwm_overrides["history_horizon"] = args.history_horizon
    if args.forecast_horizon is not None:
        rwm_overrides["forecast_horizon"] = args.forecast_horizon
    train_cfg["rwm"] = rwm_overrides

    runner_cls = load_runner_cls(args.task)
    if runner_cls is None:
        raise RuntimeError(f"Task {args.task} did not register a custom Go2 RWM runner.")

    log_root = REPO_ROOT / "logs" / "rsl_rl" / agent_cfg.experiment_name
    run_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if agent_cfg.run_name:
        run_dir += f"_{agent_cfg.run_name}"
    log_dir = log_root / run_dir
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Go2-RWM] Logging to {log_dir}")

    runner = runner_cls(vec_env, train_cfg, str(log_dir), device)
    runner.add_git_repo_to_log(__file__)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)
    vec_env.close()


if __name__ == "__main__":
    main()
