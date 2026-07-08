"""Evaluate a Go2 FlashSAC-RWM policy in the real mjlab simulator."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_dataset.broken_go2 import (
    apply_go2_broken_pd_joints,
    apply_go2_pd_joint_strength_scales,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    apply_policy_observation_mask_np,
    configure_low_thread_env,
    expand_policy_actions_np,
    get_world_model_broken_joint_names,
    get_world_model_action_masks,
    get_world_model_policy_observation_mask,
    load_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    scalarize,
    select_device,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--task", default="Unitree-Go2-Flat-RWM-Pretrain-Ens")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--fixed_command", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=None)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--output_json", default=None)
    parser.add_argument("--broken_joint_names", nargs="*", default=None)
    parser.add_argument(
        "--joint_strength_scales",
        nargs="*",
        default=(),
        metavar="JOINT=SCALE",
        help="Per-joint actuator strength scales for mjlab eval, e.g. RR_calf_joint=0.5.",
    )
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _disable_randomization(env_cfg: Any) -> None:
    """Turn the RWM task into a clean deterministic-ish evaluation config."""
    env_cfg.events.pop("push_robot", None)
    for event_name in ("foot_friction", "encoder_bias", "base_com"):
        env_cfg.events.pop(event_name, None)
    for group_name in ("actor", "critic"):
        obs_group = env_cfg.observations.get(group_name)
        if obs_group is not None:
            obs_group.enable_corruption = False
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = {}
    if hasattr(env_cfg, "curriculums"):
        env_cfg.curriculums = {}


def _actor_obs(obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
    return obs_dict["actor"].detach().cpu().numpy().astype(np.float32)


def _parse_joint_strength_scales(spec: list[str] | tuple[str, ...]) -> dict[str, float]:
    scales: dict[str, float] = {}
    for item in spec:
        raw = str(item).strip()
        if not raw:
            continue
        if "=" in raw:
            name, value = raw.split("=", maxsplit=1)
        elif ":" in raw:
            name, value = raw.split(":", maxsplit=1)
        else:
            raise ValueError(f"Invalid joint strength scale {raw!r}; expected JOINT=SCALE.")
        name = name.strip()
        if not name:
            raise ValueError(f"Invalid joint strength scale {raw!r}; joint name is empty.")
        scale = float(value)
        if scale < 0.0 or scale > 1.0:
            raise ValueError(f"Joint strength scale for {name!r} must be in [0, 1], got {scale}.")
        scales[name] = scale
    return scales


def _configure_fixed_command_range(env_cfg: Any, fixed_command: tuple[float, float, float] | None) -> None:
    if fixed_command is None or "twist" not in env_cfg.commands:
        return
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.ranges.lin_vel_x = (fixed_command[0], fixed_command[0])
    twist_cmd.ranges.lin_vel_y = (fixed_command[1], fixed_command[1])
    twist_cmd.ranges.ang_vel_z = (fixed_command[2], fixed_command[2])
    if hasattr(twist_cmd.ranges, "heading"):
        twist_cmd.ranges.heading = None
    if hasattr(twist_cmd, "heading_command"):
        twist_cmd.heading_command = False
    if hasattr(twist_cmd, "rel_standing_envs"):
        twist_cmd.rel_standing_envs = 0.0
    if hasattr(twist_cmd, "rel_heading_envs"):
        twist_cmd.rel_heading_envs = 0.0
    if hasattr(twist_cmd, "init_velocity_prob"):
        twist_cmd.init_velocity_prob = 0.0


def _force_fixed_command(env: Any, fixed_command: tuple[float, float, float] | None) -> None:
    if fixed_command is None:
        return
    try:
        command = env.unwrapped.command_manager.get_term("twist")
    except Exception:
        return
    value = torch.tensor(fixed_command, dtype=torch.float32, device=env.unwrapped.device)
    if hasattr(command, "vel_command_b"):
        command.vel_command_b[:, :] = value
    if hasattr(command, "is_standing_env"):
        command.is_standing_env[:] = False
    if hasattr(command, "is_heading_env"):
        command.is_heading_env[:] = False


def _apply_command_to_obs(
    observations: np.ndarray,
    fixed_command: tuple[float, float, float] | None,
) -> np.ndarray:
    if fixed_command is None:
        return observations
    observations = observations.copy()
    observations[:, 9:12] = np.asarray(fixed_command, dtype=np.float32)
    return observations


def _mean(values: list[float], fallback: float = 0.0) -> float:
    return float(np.mean(values)) if values else fallback


def _summarize_metric(metrics: dict[str, list[float]], key: str) -> float | None:
    values = metrics.get(key)
    if not values:
        return None
    return float(np.mean(values))


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse_args()

    checkpoint_path = resolve_repo_path(args.checkpoint_path)
    config_path = Path(args.config_path).expanduser() if args.config_path else checkpoint_path / "rwm_flashsac_config.yaml"
    if not config_path.is_absolute():
        config_path = resolve_repo_path(config_path)
    cfg = load_config(config_path, overrides=args.overrides)

    device = select_device(args.device or cfg.agent.device_type)
    set_seed(args.seed)
    broken_joint_names = get_world_model_broken_joint_names(cfg, args.broken_joint_names)
    joint_strength_scales = _parse_joint_strength_scales(args.joint_strength_scales)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    if args.clean:
        _disable_randomization(env_cfg)
    if joint_strength_scales:
        apply_go2_pd_joint_strength_scales(env_cfg, joint_strength_scales)
    if broken_joint_names:
        apply_go2_broken_pd_joints(env_cfg, broken_joint_names)
    fixed_command = tuple(args.fixed_command) if args.fixed_command is not None else None
    _configure_fixed_command_range(env_cfg, fixed_command)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _force_fixed_command(env, fixed_command)
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    full_action_dim = int(env.single_action_space.shape[0])
    if actor_dim != 48:
        raise RuntimeError(f"Expected 48-dim RWM actor observation, got {actor_dim}.")

    policy_action_mask_indices, world_model_action_mask_indices = get_world_model_action_masks(cfg, full_action_dim)
    policy_observation_mask_indices = get_world_model_policy_observation_mask(cfg, actor_dim)
    policy_action_dim = full_action_dim - len(policy_action_mask_indices)
    policy_observation_dim = actor_dim - len(policy_observation_mask_indices)

    obs_space, action_space = make_vector_spaces(
        args.num_envs,
        obs_dim=policy_observation_dim,
        action_dim=policy_action_dim,
    )
    agent_cfg = make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_agent(obs_space, action_space, agent_cfg)
    agent.load(str(checkpoint_path))

    obs_dict, _ = env.reset()
    _force_fixed_command(env, fixed_command)
    observations = _apply_command_to_obs(_actor_obs(obs_dict), fixed_command)
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    terminated_count = 0
    timeout_count = 0
    logged_metrics: dict[str, list[float]] = defaultdict(list)
    base_lin_vel_samples: list[np.ndarray] = []
    base_ang_vel_samples: list[np.ndarray] = []
    command_samples: list[np.ndarray] = []
    action_abs_samples: list[float] = []

    print(f"[Go2-FlashSAC-RWM-Eval] checkpoint={checkpoint_path}")
    print(f"[Go2-FlashSAC-RWM-Eval] task={args.task}, clean={args.clean}, device={device}, num_envs={args.num_envs}")
    print(f"[Go2-FlashSAC-RWM-Eval] broken_joint_names={list(broken_joint_names)}")
    print(f"[Go2-FlashSAC-RWM-Eval] joint_strength_scales={joint_strength_scales}")
    print(
        "[Go2-FlashSAC-RWM-Eval] "
        f"full_action_dim={full_action_dim}, policy_action_dim={policy_action_dim}, "
        f"policy_action_mask_indices={list(policy_action_mask_indices)}, "
        f"world_model_action_mask_indices={list(world_model_action_mask_indices)}, "
        f"policy_observation_mask_indices={list(policy_observation_mask_indices)}"
    )
    if fixed_command is not None:
        print(f"[Go2-FlashSAC-RWM-Eval] fixed_command={fixed_command}")

    with torch.no_grad():
        for _step in range(args.steps):
            _force_fixed_command(env, fixed_command)
            observations = _apply_command_to_obs(observations, fixed_command)
            policy_observations = apply_policy_observation_mask_np(
                observations,
                policy_observation_mask_indices,
            )
            actions_np = agent.sample_actions(
                interaction_step=0,
                prev_transition={"next_observation": policy_observations},
                training=False,
            )
            actions_np = expand_policy_actions_np(
                actions_np,
                action_dim=full_action_dim,
                policy_action_mask_indices=policy_action_mask_indices,
                world_model_action_mask_indices=world_model_action_mask_indices,
            )
            action_abs_samples.append(float(np.abs(actions_np).mean()))
            base_lin_vel_samples.append(observations[:, 0:3].copy())
            base_ang_vel_samples.append(observations[:, 3:6].copy())
            command_samples.append(observations[:, 9:12].copy())
            actions_t = torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)
            obs_dict, rewards, terminateds, truncateds, extras = env.step(actions_t)
            rewards_np = rewards.detach().cpu().numpy().astype(np.float64)
            term_np = terminateds.detach().cpu().numpy().astype(bool)
            trunc_np = truncateds.detach().cpu().numpy().astype(bool)
            done_np = np.logical_or(term_np, trunc_np)

            ep_returns += rewards_np
            ep_lengths += 1

            if done_np.any():
                completed_returns.extend(ep_returns[done_np].tolist())
                completed_lengths.extend(ep_lengths[done_np].tolist())
                terminated_count += int(term_np.sum())
                timeout_count += int(trunc_np.sum())
                ep_returns[done_np] = 0.0
                ep_lengths[done_np] = 0

            for key, value in (extras.get("log") or {}).items():
                logged_metrics[key].append(float(scalarize(value)))

            _force_fixed_command(env, fixed_command)
            observations = _apply_command_to_obs(_actor_obs(obs_dict), fixed_command)

    env.close()

    mean_return = _mean(completed_returns, fallback=float(ep_returns.mean()))
    std_return = float(np.std(completed_returns)) if completed_returns else float(np.std(ep_returns))
    mean_episode_length = _mean(completed_lengths, fallback=float(ep_lengths.mean()))
    base_lin_vel = np.concatenate(base_lin_vel_samples, axis=0) if base_lin_vel_samples else np.zeros((1, 3))
    base_ang_vel = np.concatenate(base_ang_vel_samples, axis=0) if base_ang_vel_samples else np.zeros((1, 3))
    commands = np.concatenate(command_samples, axis=0) if command_samples else np.zeros((1, 3))
    vel_error_xy = np.linalg.norm(base_lin_vel[:, 0:2] - commands[:, 0:2], axis=1)
    yaw_error = np.abs(base_ang_vel[:, 2] - commands[:, 2])
    summary = {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_episode_length": mean_episode_length,
        "completed_episodes": len(completed_returns),
        "terminated_count": terminated_count,
        "timeout_count": timeout_count,
        "non_timeout_termination_count": terminated_count,
        "command_x": float(commands[:, 0].mean()),
        "command_y": float(commands[:, 1].mean()),
        "command_yaw": float(commands[:, 2].mean()),
        "base_lin_vel_x": float(base_lin_vel[:, 0].mean()),
        "base_lin_vel_y": float(base_lin_vel[:, 1].mean()),
        "base_speed_xy": float(np.linalg.norm(base_lin_vel[:, 0:2], axis=1).mean()),
        "base_yaw_vel": float(base_ang_vel[:, 2].mean()),
        "error_vel_xy": float(vel_error_xy.mean()),
        "error_vel_yaw": float(yaw_error.mean()),
        "action_abs_mean": _mean(action_abs_samples),
    }

    print("[Go2-FlashSAC-RWM-Eval] Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    key_metrics = (
        "Episode_Reward/track_linear_velocity",
        "Episode_Reward/track_angular_velocity",
        "Episode_Reward/body_orientation_l2",
        "Episode_Reward/pose",
        "Episode_Reward/action_rate_l2",
        "Episode_Reward/foot_gait",
        "Episode_Termination/fell_over",
        "Episode_Termination/illegal_contact",
        "Metrics/twist/error_vel_xy",
        "Metrics/twist/error_vel_yaw",
    )
    for key in key_metrics:
        value = _summarize_metric(logged_metrics, key)
        if value is not None:
            print(f"  {key}: {value:.6f}")
            summary[key] = value

    if args.output_json:
        output_path = Path(args.output_json)
        if not output_path.is_absolute():
            output_path = resolve_repo_path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[Go2-FlashSAC-RWM-Eval] wrote {output_path}")


if __name__ == "__main__":
    main()
