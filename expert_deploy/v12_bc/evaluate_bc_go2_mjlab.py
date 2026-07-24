"""Evaluate the transition-wise 45D BC baseline in MJLab."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expert_deploy.v12_bc.play_bc_go2_mjlab import (  # noqa: E402
    _BCPolicy,
    _configure_dataset_random_commands,
    _load_actor,
)
from flash_rl.envs.mjlab import configure_mjlab_randomization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=64)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--enable_randomization", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = (REPO_ROOT / args.checkpoint).resolve()
    config_path = (REPO_ROOT / args.config).resolve()
    dataset_path = (REPO_ROOT / args.dataset).resolve()
    output_path = (REPO_ROOT / args.output).resolve()

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(
        "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
        play=True,
    )
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    _configure_dataset_random_commands(
        env_cfg,
        dataset_path,
        forward_only=args.forward_only,
    )
    if not args.enable_randomization:
        configure_mjlab_randomization(
            env_cfg,
            use_domain_randomization=False,
            use_push_randomization=False,
            use_observation_noise=False,
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
    policy = _BCPolicy(_load_actor(checkpoint_path, config_path, device))
    env.reset()
    observations = env.observation_manager.compute()

    linear_tracking_squared_error = 0.0
    yaw_tracking_squared_error = 0.0
    reward_sum = 0.0
    action_squared_sum = 0.0
    action_abs_max = 0.0
    terminated_count = 0
    timeout_count = 0
    sample_count = 0
    for _step in range(args.num_steps):
        actions = policy(observations)
        robot = env.scene["robot"]
        true_velocity = robot.data.root_link_lin_vel_b
        true_angular_velocity = robot.data.root_link_ang_vel_b
        command = env.command_manager.get_command("twist")
        linear_tracking_squared_error += (
            true_velocity[:, :2] - command[:, :2]
        ).square().sum().item()
        yaw_tracking_squared_error += (
            true_angular_velocity[:, 2] - command[:, 2]
        ).square().sum().item()
        action_squared_sum += actions.square().sum().item()
        action_abs_max = max(action_abs_max, actions.abs().max().item())
        observations, rewards, terminated, truncated, _extras = env.step(actions)
        reward_sum += rewards.sum().item()
        terminated_count += int(terminated.sum().item())
        timeout_count += int(truncated.sum().item())
        sample_count += args.num_envs

    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "transitions": sample_count,
        "forward_only": args.forward_only,
        "randomization": args.enable_randomization,
        "closed_loop": {
            "mean_reward_per_step": reward_sum / sample_count,
            "linear_velocity_tracking_rmse": (
                linear_tracking_squared_error / (sample_count * 2)
            )
            ** 0.5,
            "yaw_velocity_tracking_rmse": (yaw_tracking_squared_error / sample_count) ** 0.5,
            "action_rms": (action_squared_sum / (sample_count * 12)) ** 0.5,
            "action_abs_max": action_abs_max,
            "terminations": terminated_count,
            "timeouts": timeout_count,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    env.close()


if __name__ == "__main__":
    main()
