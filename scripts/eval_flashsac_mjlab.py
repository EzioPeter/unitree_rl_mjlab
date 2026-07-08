"""Evaluate a trained FlashSAC checkpoint in mjlab."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.agents import create_agent
from flash_rl.envs.mjlab import (
    configure_mjlab_randomization,
    expand_masked_actions_t,
    normalize_action_mask_indices,
)
from scripts.flashsac_mjlab_overrides import apply_mjlab_env_overrides


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
FLASHSAC_CONFIG_FILENAME = "flashsac_config.yaml"


def _compose_config(config_path: str, config_name: str, overrides: list[str]):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    GlobalHydra.instance().clear()
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = (REPO_ROOT / config_dir).resolve()
    hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _load_saved_config(config_file: Path, overrides: list[str]):
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_file).resolve()
    cfg = OmegaConf.load(config_file)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _find_checkpoint_config(checkpoint_path: Path) -> Path | None:
    candidates = (
        checkpoint_path / FLASHSAC_CONFIG_FILENAME,
        checkpoint_path.parent / FLASHSAC_CONFIG_FILENAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_eval_config(args: argparse.Namespace):
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if args.config_file is not None:
        return _load_saved_config(Path(args.config_file).expanduser(), args.overrides), checkpoint_path
    checkpoint_config = _find_checkpoint_config(checkpoint_path)
    if checkpoint_config is not None:
        print(f"[INFO] Loading FlashSAC config: {checkpoint_config}")
        return _load_saved_config(checkpoint_config, args.overrides), checkpoint_path
    return _compose_config(args.config_path, args.config_name, args.overrides), checkpoint_path


def _configure_clean_eval(env_cfg: Any) -> None:
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=False,
        use_push_randomization=False,
        use_observation_noise=False,
    )
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = {}
    if hasattr(env_cfg, "curriculums"):
        env_cfg.curriculums = {}


def _configure_fixed_command_range(
    env_cfg: Any,
    fixed_command: tuple[float, float, float] | None,
) -> None:
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
        command = env.command_manager.get_term("twist")
    except Exception:
        return
    value = torch.tensor(fixed_command, dtype=torch.float32, device=env.device)
    if hasattr(command, "vel_command_b"):
        command.vel_command_b[:, :] = value
    if hasattr(command, "is_standing_env"):
        command.is_standing_env[:] = False
    if hasattr(command, "is_heading_env"):
        command.is_heading_env[:] = False


def _flatten_obs(
    obs_dict: dict[str, torch.Tensor],
    *,
    has_critic_obs: bool,
    use_critic_observation_as_full_observation: bool,
) -> np.ndarray:
    if has_critic_obs and use_critic_observation_as_full_observation:
        flat = obs_dict["critic"]
    elif has_critic_obs:
        flat = torch.cat([obs_dict["actor"], obs_dict["critic"]], dim=-1)
    else:
        flat = obs_dict["actor"]
    return flat.detach().cpu().numpy().astype(np.float32)


def _scalarize(value: Any) -> float | int | Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.float().mean().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.astype(np.float32).mean())
    if isinstance(value, (float, int)):
        return value
    return value


def _mean(values: list[float], fallback: float = 0.0) -> float:
    return float(np.mean(values)) if values else fallback


def _masked_mean(values: np.ndarray, mask: np.ndarray) -> float:
    if not np.any(mask):
        return float("nan")
    return float(values[mask].mean())


def _get_command(env: Any, command_name: str = "twist") -> torch.Tensor:
    command = env.command_manager.get_command(command_name)
    if command is None:
        return torch.zeros((env.num_envs, 3), dtype=torch.float32, device=env.device)
    return command


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config_name", type=str, default="flashsac_g1_velocity")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--fixed_command", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=None)
    parser.add_argument("--broken_joint_names", nargs="*", default=None)
    parser.add_argument("--output_json", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse_args()
    cfg, checkpoint_path = _load_eval_config(args)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = args.device or cfg.env.get("device") or ("cuda:0" if torch.cuda.is_available() else "cpu")
    fixed_command = tuple(args.fixed_command) if args.fixed_command is not None else None

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(cfg.env.env_name, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    apply_mjlab_env_overrides(env_cfg, cfg)
    if args.clean:
        _configure_clean_eval(env_cfg)
    _configure_fixed_command_range(env_cfg, fixed_command)

    joint_strength_scales = {
        str(name): float(scale)
        for name, scale in (cfg.env.get("joint_strength_scales", {}) or {}).items()
    }
    broken_joint_names = tuple(args.broken_joint_names or cfg.env.get("broken_joint_names", []) or ())
    for joint_name in broken_joint_names:
        joint_strength_scales[str(joint_name)] = 0.0
    if joint_strength_scales:
        from scripts.reinforcement_learning.rwm_dataset.broken_go2 import apply_go2_pd_joint_strength_scales

        apply_go2_pd_joint_strength_scales(env_cfg, joint_strength_scales)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _force_fixed_command(env, fixed_command)

    obs_groups = list(env.single_observation_space.spaces.keys())
    has_critic_obs = "actor" in obs_groups and "critic" in obs_groups
    use_critic_observation_as_full_observation = bool(
        cfg.env.get("use_critic_observation_as_full_observation", False)
    ) and has_critic_obs
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    if use_critic_observation_as_full_observation:
        flat_dim = int(env.single_observation_space.spaces["critic"].shape[0])
    elif has_critic_obs:
        flat_dim = actor_dim + int(env.single_observation_space.spaces["critic"].shape[0])
    else:
        flat_dim = actor_dim

    full_action_dim = int(env.single_action_space.shape[0])
    action_mask_indices = normalize_action_mask_indices(cfg.env.get("action_mask_indices", []), full_action_dim)
    policy_action_dim = full_action_dim - len(action_mask_indices)

    obs_space = batch_space(gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32), args.num_envs)
    act_space = batch_space(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(policy_action_dim,), dtype=np.float32),
        args.num_envs,
    )
    env_info: dict[str, Any] = {}
    if has_critic_obs:
        env_info["actor_observation_size"] = (actor_dim,)
    agent = create_agent(observation_space=obs_space, action_space=act_space, env_info=env_info, cfg=cfg.agent)
    agent.load(str(checkpoint_path))

    obs_dict, _ = env.reset()
    _force_fixed_command(env, fixed_command)
    observations = _flatten_obs(
        obs_dict,
        has_critic_obs=has_critic_obs,
        use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
    )

    robot = env.scene["robot"]
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    terminated_count = 0
    timeout_count = 0
    logged_metrics: dict[str, list[float]] = defaultdict(list)
    command_samples: list[np.ndarray] = []
    base_lin_vel_samples: list[np.ndarray] = []
    base_ang_vel_samples: list[np.ndarray] = []
    action_abs_samples: list[float] = []
    reward_samples: list[float] = []

    print(f"[FlashSAC-Eval] checkpoint={checkpoint_path}")
    print(f"[FlashSAC-Eval] task={cfg.env.env_name}, clean={args.clean}, device={device}, num_envs={args.num_envs}")
    print(
        "[FlashSAC-Eval] "
        f"actor_dim={actor_dim}, flat_dim={flat_dim}, full_action_dim={full_action_dim}, "
        f"policy_action_dim={policy_action_dim}, action_mask_indices={list(action_mask_indices)}, "
        f"extra_broken_joint_names={list(broken_joint_names)}"
    )
    if fixed_command is not None:
        print(f"[FlashSAC-Eval] fixed_command={fixed_command}")

    with torch.no_grad():
        for _step in range(args.steps):
            _force_fixed_command(env, fixed_command)
            if fixed_command is not None:
                obs_dict = env.observation_manager.compute()
                observations = _flatten_obs(
                    obs_dict,
                    has_critic_obs=has_critic_obs,
                    use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
                )

            actions_np = agent.sample_actions(
                interaction_step=0,
                prev_transition={"next_observation": observations},
                training=False,
            )
            actions_t = torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)
            full_actions_t = expand_masked_actions_t(
                actions_t,
                full_action_dim=full_action_dim,
                action_mask_indices=action_mask_indices,
            )

            action_abs_samples.append(float(torch.abs(full_actions_t).mean().item()))
            obs_dict, rewards, terminateds, truncateds, extras = env.step(full_actions_t)
            _force_fixed_command(env, fixed_command)

            rewards_np = rewards.detach().cpu().numpy().astype(np.float64)
            term_np = terminateds.detach().cpu().numpy().astype(bool)
            trunc_np = truncateds.detach().cpu().numpy().astype(bool)
            done_np = np.logical_or(term_np, trunc_np)
            ep_returns += rewards_np
            ep_lengths += 1
            reward_samples.append(float(rewards_np.mean()))

            command_samples.append(_get_command(env).detach().cpu().numpy().astype(np.float32))
            base_lin_vel_samples.append(robot.data.root_link_lin_vel_b.detach().cpu().numpy().astype(np.float32))
            base_ang_vel_samples.append(robot.data.root_link_ang_vel_b.detach().cpu().numpy().astype(np.float32))

            if done_np.any():
                completed_returns.extend(ep_returns[done_np].tolist())
                completed_lengths.extend(ep_lengths[done_np].tolist())
                terminated_count += int(term_np.sum())
                timeout_count += int(trunc_np.sum())
                ep_returns[done_np] = 0.0
                ep_lengths[done_np] = 0

            for key, value in (extras.get("log") or {}).items():
                scalar = _scalarize(value)
                if isinstance(scalar, (float, int)):
                    logged_metrics[key].append(float(scalar))

            observations = _flatten_obs(
                obs_dict,
                has_critic_obs=has_critic_obs,
                use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
            )

    env.close()

    commands = np.concatenate(command_samples, axis=0) if command_samples else np.zeros((1, 3), dtype=np.float32)
    base_lin_vel = (
        np.concatenate(base_lin_vel_samples, axis=0) if base_lin_vel_samples else np.zeros((1, 3), dtype=np.float32)
    )
    base_ang_vel = (
        np.concatenate(base_ang_vel_samples, axis=0) if base_ang_vel_samples else np.zeros((1, 3), dtype=np.float32)
    )
    lin_error = base_lin_vel[:, :2] - commands[:, :2]
    yaw_error = base_ang_vel[:, 2] - commands[:, 2]
    abs_x_error = np.abs(lin_error[:, 0])
    abs_y_error = np.abs(lin_error[:, 1])
    abs_xy_error = np.linalg.norm(lin_error[:, :2], axis=1)
    abs_yaw_error = np.abs(yaw_error)

    summary = {
        "mean_return": _mean(completed_returns, fallback=float(ep_returns.mean())),
        "std_return": float(np.std(completed_returns)) if completed_returns else float(np.std(ep_returns)),
        "mean_episode_length": _mean(completed_lengths, fallback=float(ep_lengths.mean())),
        "completed_episodes": len(completed_returns),
        "terminated_count": terminated_count,
        "timeout_count": timeout_count,
        "non_timeout_termination_count": terminated_count,
        "mean_step_reward": _mean(reward_samples),
        "command_x": float(commands[:, 0].mean()),
        "command_y": float(commands[:, 1].mean()),
        "command_yaw": float(commands[:, 2].mean()),
        "command_speed_xy": float(np.linalg.norm(commands[:, :2], axis=1).mean()),
        "base_lin_vel_x": float(base_lin_vel[:, 0].mean()),
        "base_lin_vel_y": float(base_lin_vel[:, 1].mean()),
        "base_speed_xy": float(np.linalg.norm(base_lin_vel[:, :2], axis=1).mean()),
        "base_yaw_vel": float(base_ang_vel[:, 2].mean()),
        "error_vel_x_abs": float(abs_x_error.mean()),
        "error_vel_y_abs": float(abs_y_error.mean()),
        "error_vel_xy": float(abs_xy_error.mean()),
        "error_vel_yaw": float(abs_yaw_error.mean()),
        "error_vel_x_abs_pos_cmd": _masked_mean(abs_x_error, commands[:, 0] > 0.05),
        "error_vel_x_abs_neg_cmd": _masked_mean(abs_x_error, commands[:, 0] < -0.05),
        "error_vel_y_abs_pos_cmd": _masked_mean(abs_y_error, commands[:, 1] > 0.05),
        "error_vel_y_abs_neg_cmd": _masked_mean(abs_y_error, commands[:, 1] < -0.05),
        "error_vel_yaw_abs_pos_cmd": _masked_mean(abs_yaw_error, commands[:, 2] > 0.05),
        "error_vel_yaw_abs_neg_cmd": _masked_mean(abs_yaw_error, commands[:, 2] < -0.05),
        "action_abs_mean": _mean(action_abs_samples),
    }

    for key in (
        "Reward/track_linear_velocity",
        "Episode_Reward/track_linear_velocity",
        "Episode_Reward/track_angular_velocity",
        "Episode_Termination/fell_over",
        "Episode_Termination/illegal_contact",
        "Metrics/lin_vel_xy_error_mean",
        "Metrics/twist/error_vel_xy",
        "Metrics/twist/error_vel_yaw",
    ):
        values = logged_metrics.get(key)
        if values:
            summary[key] = float(np.mean(values))

    print("[FlashSAC-Eval] Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    if args.output_json:
        output_path = Path(args.output_json).expanduser()
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"[FlashSAC-Eval] wrote {output_path}")


if __name__ == "__main__":
    main()
