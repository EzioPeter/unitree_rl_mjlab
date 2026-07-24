"""Evaluate CENet velocity estimation and closed-loop BC behavior in MJLab."""

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

from expert_deploy.v12_bc.play_bc_go2_mjlab import _configure_dataset_random_commands  # noqa: E402
from expert_deploy.v12_bc.play_cenet_bc_go2_mjlab import (  # noqa: E402
    OnlineCENetPolicy,
    load_policy,
)
from flash_rl.envs.mjlab import configure_mjlab_randomization  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
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
    online_policy = OnlineCENetPolicy(load_policy(checkpoint_path, device))
    env.reset()
    observations = env.observation_manager.compute()
    online_policy.reset(observations["actor"])

    velocity_squared_error = torch.zeros(3, device=device, dtype=torch.float64)
    velocity_absolute_error = torch.zeros(3, device=device, dtype=torch.float64)
    linear_tracking_squared_error = 0.0
    yaw_tracking_squared_error = 0.0
    reward_sum = 0.0
    action_squared_sum = 0.0
    action_abs_max = 0.0
    terminated_count = 0
    timeout_count = 0
    sample_count = 0
    for _step in range(args.num_steps):
        actor_observation = observations["actor"]
        actions, estimated_velocity = online_policy.act(actor_observation)
        robot = env.scene["robot"]
        true_velocity = robot.data.root_link_lin_vel_b
        true_angular_velocity = robot.data.root_link_ang_vel_b
        command = env.command_manager.get_command("twist")
        velocity_error = estimated_velocity - true_velocity
        velocity_squared_error += velocity_error.square().sum(dim=0).double()
        velocity_absolute_error += velocity_error.abs().sum(dim=0).double()
        linear_tracking_squared_error += (
            true_velocity[:, :2] - command[:, :2]
        ).square().sum().item()
        yaw_tracking_squared_error += (
            true_angular_velocity[:, 2] - command[:, 2]
        ).square().sum().item()
        action_squared_sum += actions.square().sum().item()
        action_abs_max = max(action_abs_max, actions.abs().max().item())
        step_result = env.step(actions)
        observations, rewards, terminated, truncated, _extras = step_result
        reward_sum += rewards.sum().item()
        terminated_count += int(terminated.sum().item())
        timeout_count += int(truncated.sum().item())
        done_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        if done_ids.numel() > 0:
            online_policy.reset_indices(done_ids, observations["actor"])
        sample_count += args.num_envs

    velocity_mse_per_axis = velocity_squared_error / sample_count
    velocity_mae_per_axis = velocity_absolute_error / sample_count
    report = {
        "checkpoint": str(checkpoint_path),
        "dataset": str(dataset_path),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_steps": args.num_steps,
        "transitions": sample_count,
        "forward_only": args.forward_only,
        "randomization": args.enable_randomization,
        "velocity_estimation": {
            "mse": float(velocity_mse_per_axis.mean()),
            "rmse": float(torch.sqrt(velocity_mse_per_axis.mean())),
            "mae": float(velocity_mae_per_axis.mean()),
            "mse_per_axis": velocity_mse_per_axis.tolist(),
            "rmse_per_axis": torch.sqrt(velocity_mse_per_axis).tolist(),
            "mae_per_axis": velocity_mae_per_axis.tolist(),
        },
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
