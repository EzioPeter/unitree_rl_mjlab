"""Train a Go2 policy purely in learned RWM imagination."""

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
    parser.add_argument("--task", default="go2_flat")
    parser.add_argument("--model_resume_path", required=True)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--run_num", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--num_steps_per_env", type=int, default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--logger", choices=("wandb", "tensorboard"), default=None)
    parser.add_argument("--uncertainty_penalty_weight", type=float, default=None)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    from scripts.reinforcement_learning.rwm.configs.go2_flat_cfg import (
        Go2FlatRWMConfig,
        unitree_go2_rwm_model_based_runner_cfg,
    )
    from scripts.reinforcement_learning.rwm.dynamics import SequenceReplayBuffer, load_dynamics_checkpoint
    from scripts.reinforcement_learning.rwm.envs.go2_flat import Go2FlatRWMImaginationEnv
    from scripts.reinforcement_learning.rwm.policy_training import RWMPolicyRunner

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    model_path = Path(args.model_resume_path).expanduser()
    if not model_path.is_absolute():
        model_path = (REPO_ROOT / model_path).resolve()
    dataset_path = Path(args.dataset_path).expanduser() if args.dataset_path else model_path.parent / "dataset.pt"
    if not dataset_path.is_absolute():
        dataset_path = (REPO_ROOT / dataset_path).resolve()

    dynamics, _checkpoint = load_dynamics_checkpoint(model_path, device=device)
    dataset = SequenceReplayBuffer.load(dataset_path, device=device)

    cfg = Go2FlatRWMConfig()
    if args.num_envs is not None:
        cfg.imagination.num_envs = args.num_envs
    if args.uncertainty_penalty_weight is not None:
        cfg.imagination.uncertainty_penalty_weight = args.uncertainty_penalty_weight

    env = Go2FlatRWMImaginationEnv(dynamics=dynamics, dataset=dataset, cfg=cfg.imagination, device=device)
    agent_cfg = unitree_go2_rwm_model_based_runner_cfg()
    if args.seed is not None:
        agent_cfg.seed = args.seed
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations
    if args.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args.num_steps_per_env
    if args.save_interval is not None:
        agent_cfg.save_interval = args.save_interval
    if args.logger is not None:
        agent_cfg.logger = args.logger

    train_cfg = asdict(agent_cfg)
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.run_num is not None:
        run_name += f"_{args.run_num}"
    log_dir = REPO_ROOT / "logs" / "model_based" / args.task / run_name
    log_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Go2-RWM] Stage 2 imagination training: model={model_path}")
    print(f"[Go2-RWM] Dataset: {dataset_path}")
    print(f"[Go2-RWM] Logging to {log_dir}")

    runner = RWMPolicyRunner(env, train_cfg, str(log_dir), device)
    runner.add_git_repo_to_log(__file__)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=False)
    env.close()


if __name__ == "__main__":
    main()
