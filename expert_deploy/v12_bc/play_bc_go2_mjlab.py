"""Play a deterministic V12 Go2 behavior-cloning policy in MJLab."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.agents.flashSAC.network import FlashSACActor
from flash_rl.envs.mjlab import configure_mjlab_randomization


class _MjlabViewerEnv:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.num_envs = env.num_envs

    @property
    def device(self) -> Any:
        return self._env.device

    @property
    def cfg(self) -> Any:
        return self._env.cfg

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env

    def get_observations(self) -> dict[str, torch.Tensor]:
        return self._env.observation_manager.compute()

    def step(self, actions: torch.Tensor) -> Any:
        return self._env.step(actions)

    def reset(self, **kwargs: Any) -> Any:
        return self._env.reset(**kwargs)

    def close(self) -> None:
        self._env.close()


class _BCPolicy:
    def __init__(self, actor: FlashSACActor) -> None:
        self._actor = actor

    @torch.no_grad()
    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        mean, _std = self._actor.get_mean_and_std(
            observations["actor"],
            training=False,
        )
        return torch.tanh(mean)


def _load_actor(checkpoint_path: Path, config_path: Path, device: torch.device) -> FlashSACActor:
    config = json.loads(config_path.read_text(encoding="utf-8"))
    actor = FlashSACActor(
        num_blocks=int(config["num_blocks"]),
        input_dim=45,
        hidden_dim=int(config["hidden_dim"]),
        action_dim=12,
    ).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    actor.load_state_dict(checkpoint["network_state_dict"])
    actor.eval()
    return actor


def _configure_dataset_random_commands(
    env_cfg: Any,
    dataset_path: Path,
    *,
    forward_only: bool,
) -> None:
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    metadata = dataset.get("metadata", {})
    command_ranges = metadata.get("command_ranges", {})
    command_weights = metadata.get("command_mode_weights", {})
    command_interval = metadata.get("command_resample_interval", [150, 400])
    dt = float(metadata.get("dt", 0.02))

    twist = env_cfg.commands["twist"]
    x_range = command_ranges.get("x_range", [-0.5, 0.5])
    y_abs = command_ranges.get("y_abs_range", [0.05, 0.2])
    yaw_abs = command_ranges.get("yaw_abs_range", [0.08, 0.4])
    twist.ranges.lin_vel_x = (float(x_range[0]), float(x_range[1]))
    twist.ranges.lin_vel_y = (-float(y_abs[1]), float(y_abs[1]))
    twist.ranges.ang_vel_z = (-float(yaw_abs[1]), float(yaw_abs[1]))
    twist.resampling_time_range = (
        float(command_interval[0]) * dt,
        float(command_interval[1]) * dt,
    )

    # Collapse the dataset's eight command modes into the four modes exposed by
    # ModeBalancedVelocityCommandCfg.
    twist.xy_command_prob = float(
        command_weights.get("pure_x", 0.25)
        + command_weights.get("pure_y", 0.10)
        + command_weights.get("xy", 0.14)
    )
    twist.yaw_command_prob = float(command_weights.get("pure_yaw", 0.08))
    twist.mixed_command_prob = float(
        command_weights.get("x_yaw", 0.17)
        + command_weights.get("y_yaw", 0.05)
        + command_weights.get("xy_yaw", 0.13)
    )
    twist.stand_command_prob = float(command_weights.get("stand", 0.08))
    twist.min_lin_speed = min(float(command_ranges.get("x_abs_range", [0.08])[0]), float(y_abs[0]))
    twist.min_yaw_speed = float(yaw_abs[0])
    if forward_only:
        x_abs = command_ranges.get("x_abs_range", [0.08, 0.5])
        twist.ranges.lin_vel_x = (float(x_abs[0]), float(x_abs[1]))
        twist.ranges.lin_vel_y = (0.0, 0.0)
        twist.ranges.ang_vel_z = (0.0, 0.0)
        twist.xy_command_prob = 1.0
        twist.yaw_command_prob = 0.0
        twist.mixed_command_prob = 0.0
        twist.stand_command_prob = 0.0
        twist.min_lin_speed = float(x_abs[0])
    del dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="runs/v12_bc/g0/actor.pt")
    parser.add_argument("--config", default="runs/v12_bc/g0/bc_config.json")
    parser.add_argument("--dataset", default="datasets/g0/selected_50k/dataset.pt")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_rate", type=float, default=50.0)
    parser.add_argument(
        "--enable_randomization",
        action="store_true",
        help="Keep task domain randomization, pushes, and observation noise enabled.",
    )
    parser.add_argument(
        "--forward_only",
        action="store_true",
        help="Sample only positive forward velocity; force lateral and yaw commands to zero.",
    )
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

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.viewer import NativeMujocoViewer

    env_cfg = load_env_cfg(
        "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
        play=True,
    )
    env_cfg.scene.num_envs = 1
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
    actor = _load_actor(checkpoint_path, config_path, device)
    policy = _BCPolicy(actor)
    viewer_env = _MjlabViewerEnv(env)

    env.reset()
    print(f"[BC Play] checkpoint={checkpoint_path}")
    if args.forward_only:
        print("[BC Play] forward-only commands: vx=[0.08,0.5], vy=0, yaw=0")
    else:
        print("[BC Play] random commands: vx=[-0.5,0.5], vy=[-0.2,0.2], yaw=[-0.4,0.4]")
    print(
        "[BC Play] randomization="
        + ("enabled" if args.enable_randomization else "disabled (nominal G0)")
    )
    NativeMujocoViewer(viewer_env, policy, frame_rate=args.frame_rate).run()
    env.close()


if __name__ == "__main__":
    main()
