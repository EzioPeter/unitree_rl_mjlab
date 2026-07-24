"""Collect an expert-only Go2 dataset with broad random velocity-command coverage."""

from __future__ import annotations

import argparse
import inspect
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.agents import create_agent
from flash_rl.envs.mjlab import configure_mjlab_randomization, mjlab_randomization_manifest
from flash_rl.envs.mjlab_dr import friend_dr_runtime_snapshot
from scripts.reinforcement_learning.rwm_dataset.dataset import (
    COLLECTOR_ID_TO_NAME,
    COLLECTOR_NAME_TO_ID,
    Go2MixedDatasetBuilder,
    merge_dataset_dicts,
    parse_collector_mix,
    sample_collector_ids,
    save_dataset_dict,
    stack_time_key,
)
try:
    from scripts.reinforcement_learning.rwm_dataset.broken_go2 import apply_go2_pd_joint_strength_scales
except ImportError:
    from src.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg

    def apply_go2_pd_joint_strength_scales(
        env_cfg: Any,
        joint_strength_scales: dict[str, float],
    ) -> dict[str, float]:
        scales = {str(name): float(scale) for name, scale in joint_strength_scales.items() if str(name)}
        if not scales:
            return {}
        if "robot" not in env_cfg.scene.entities:
            raise KeyError("Expected env_cfg.scene.entities to contain a 'robot' entity.")
        env_cfg.scene.entities["robot"] = get_go2_robot_cfg(joint_strength_scales=scales)
        return scales
from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.agent_proprioceptive import (
    create_go2_flashsac_proprioceptive_agent,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    load_config as load_flashsac_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    select_device,
    set_seed,
)
from scripts.reinforcement_learning.rwm_flashsac.world_model_env_proprioceptive import (
    CRITIC_OBS_DIM_FULL_RWM,
    PROPRIOCEPTIVE_ACTOR_OBS_DIM,
    proprioceptive_obs_t,
)
from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
    SNAPSHOT_VERSION,
    canonical_snapshot_from_rwm_state,
    capture_go2_simulator_snapshot,
    repeat_snapshot_rows,
    restore_go2_simulator_snapshot,
    step_without_automatic_reset,
    synchronize_branch_domain_parameters,
)
from src.tasks.rwm_velocity.mdp.extractors import Go2RWMExtractor, make_go2_policy_obs


DEFAULT_EXPERT_POLICY = "logs/model_based/go2_flat_flashsac_rwm_sacwm/2026-06-05_17-27-59/step48828"


@dataclass(frozen=True)
class CommandMode:
    name: str
    axes: tuple[str, ...]


@dataclass(frozen=True)
class LoadedPolicy:
    agent: Any
    observation_kind: str
    config_path: Path


COMMAND_MODES: tuple[CommandMode, ...] = (
    CommandMode("stand", ()),
    CommandMode("pure_x", ("x",)),
    CommandMode("pure_y", ("y",)),
    CommandMode("pure_yaw", ("yaw",)),
    CommandMode("xy", ("x", "y")),
    CommandMode("x_yaw", ("x", "yaw")),
    CommandMode("y_yaw", ("y", "yaw")),
    CommandMode("xy_yaw", ("x", "y", "yaw")),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--task", default="Unitree-Go2-Flat-RWM-Pretrain-Ens")
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--num_transitions", type=int, default=1_000_000)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--save_path", default="logs/rwm_datasets/go2_flat_expert_command_coverage_1m/dataset.pt")
    parser.add_argument("--expert_policy_path", default=DEFAULT_EXPERT_POLICY)
    parser.add_argument("--medium_policy_path", default=None)
    parser.add_argument("--collector_mix", default="expert:1.0")
    parser.add_argument("--action_noise_std", type=float, default=0.12)
    parser.add_argument("--medium_action_noise_std", type=float, default=0.25)
    parser.add_argument("--failure_action_noise_std", type=float, default=0.45)
    parser.add_argument("--chunk_size", type=int, default=200_000)
    parser.add_argument("--command_modes", default="pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw")
    parser.add_argument("--command_mode_weights", default=None)
    parser.add_argument("--command_resample_interval_min", type=int, default=150)
    parser.add_argument("--command_resample_interval_max", type=int, default=400)
    parser.add_argument("--x_range", type=float, nargs=2, default=(0.2, 1.2), metavar=("MIN", "MAX"))
    parser.add_argument("--signed_x", action="store_true")
    parser.add_argument("--x_abs_range", type=float, nargs=2, default=(0.2, 1.2), metavar=("MIN", "MAX"))
    parser.add_argument("--y_abs_range", type=float, nargs=2, default=(0.2, 0.7), metavar=("MIN", "MAX"))
    parser.add_argument("--yaw_abs_range", type=float, nargs=2, default=(0.2, 0.9), metavar=("MIN", "MAX"))
    parser.add_argument(
        "--dataset_obs_kind",
        choices=("auto", "full_rwm", "proprioceptive"),
        default="auto",
        help=(
            "Observation stored in dataset['observations']. "
            "auto stores proprioceptive 45-dim obs for 45-dim live actor tasks, otherwise full 48-dim RWM obs."
        ),
    )
    parser.add_argument("--broken_joint_names", nargs="*", default=())
    parser.add_argument(
        "--joint_strength_scales",
        nargs="*",
        default=(),
        metavar="JOINT=SCALE",
        help="Per-joint actuator strength scales, e.g. RR_calf_joint=0.5. broken_joint_names override to 0.0.",
    )
    parser.add_argument("--use_domain_randomization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_push_randomization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_observation_noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument(
        "--randomization_preset",
        choices=(
            "default",
            "old",
            "friend_flat",
            "calibrated_default",
            "calibrated_friend_flat",
        ),
        default="default",
    )
    parser.add_argument(
        "--randomization_components",
        default=None,
        help="Comma-separated friend_flat components for diagnostics; omitted means all.",
    )
    parser.add_argument(
        "--randomization_scale",
        type=float,
        default=1.0,
        help="Scale friend_flat ranges around their nominal centers; must be in [0, 1].",
    )
    parser.add_argument(
        "--payload_mass_range_kg",
        type=float,
        nargs=2,
        default=(0.0, 0.0),
        metavar=("MIN", "MAX"),
        help="Additional payload attached to the base link for the target gap.",
    )
    parser.add_argument(
        "--payload_position_body_m",
        type=float,
        nargs=3,
        default=(0.0, 0.0, 0.10),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--payload_box_size_m",
        type=float,
        nargs=3,
        default=(0.20, 0.12, 0.05),
        metavar=("X", "Y", "Z"),
    )
    parser.add_argument(
        "--rr_calf_strength_range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        metavar=("MIN", "MAX"),
        help="Target RR calf motor-strength gap, separate from background DR.",
    )
    parser.add_argument(
        "--fixed_collector_assignment",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep each vector-env column on one collector so pure contiguous subsets can be extracted.",
    )
    parser.add_argument(
        "--collector_assignment_seed",
        type=int,
        default=42,
        help="Seed for fixed collector-to-env assignment; keep equal across independently collected shards.",
    )
    parser.add_argument("--env_action_noise_std", type=float, default=0.0)
    parser.add_argument("--env_action_bias_std", type=float, default=0.0)
    parser.add_argument(
        "--env_action_scale_range",
        type=float,
        nargs=2,
        default=(1.0, 1.0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument(
        "--env_action_delay_steps",
        type=int,
        nargs=2,
        default=(0, 0),
        metavar=("MIN", "MAX"),
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--trace_reset_dataset",
        default=None,
        help="Optional offline dataset whose 45D states initialize fixed-length imperfect-simulator rollouts.",
    )
    parser.add_argument("--trace_rollout_length", type=int, default=20)
    parser.add_argument("--trace_trajectories_per_state", type=int, default=4)
    parser.add_argument(
        "--trace_reset_mode",
        choices=("exact_snapshot", "canonical_real_projection"),
        default="exact_snapshot",
        help=(
            "exact_snapshot requires simulator snapshot fields in trace_reset_dataset. "
            "canonical_real_projection is the explicit approximate mode for real logs."
        ),
    )
    parser.add_argument(
        "--save_trace_snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Save reset-relevant MJLab snapshots with ordinary simulator datasets.",
    )
    parser.add_argument(
        "--trace_identity_tolerance",
        type=float,
        default=1.0e-4,
        help="Maximum repeated one-step state error allowed after restoring the same snapshot/action.",
    )
    parser.add_argument(
        "--trace_action_temperature",
        type=float,
        default=1.0,
        help=(
            "Policy sampling temperature used only for imperfect-simulator TRACE candidate rollouts. "
            "Ordinary dataset collection remains deterministic."
        ),
    )
    return parser.parse_args()


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


def _fixed_collector_ids(
    collector_mix: dict[str, float],
    num_envs: int,
    device: torch.device | str,
    seed: int,
) -> torch.Tensor:
    """Build an exact, reproducible collector allocation shared by all shards."""
    names = [name for name, weight in collector_mix.items() if float(weight) > 0.0]
    raw = [float(collector_mix[name]) * num_envs for name in names]
    counts = [int(value) for value in raw]
    remaining = num_envs - sum(counts)
    fractional_order = sorted(
        range(len(names)),
        key=lambda index: raw[index] - counts[index],
        reverse=True,
    )
    for index in fractional_order[:remaining]:
        counts[index] += 1
    ids = torch.cat(
        [
            torch.full((count,), COLLECTOR_NAME_TO_ID[name], dtype=torch.long)
            for name, count in zip(names, counts, strict=True)
        ]
    )
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return ids[torch.randperm(num_envs, generator=generator)].to(device=device)


def _command_start_for_obs(obs_dim: int, actor_dim: int) -> int:
    # RWM observations include base_lin_vel before the command.  The
    # proprioceptive broken expert omits base_lin_vel from the actor prefix.
    if obs_dim == 45 or actor_dim == 45:
        return 6
    return 9


def _overwrite_command(obs: torch.Tensor, command: torch.Tensor | None, *, actor_dim: int) -> torch.Tensor:
    if command is not None:
        start = _command_start_for_obs(int(obs.shape[-1]), actor_dim)
        obs[:, start : start + 3] = command.to(device=obs.device, dtype=obs.dtype)
    return obs


def _obs_group_np(
    obs_dict: dict[str, torch.Tensor],
    group_name: str,
    command: torch.Tensor | None,
    *,
    actor_dim: int,
) -> np.ndarray:
    obs = obs_dict[group_name].detach().clone()
    obs = _overwrite_command(obs, command, actor_dim=actor_dim)
    return obs.cpu().numpy().astype(np.float32)


def _policy_obs_np(
    obs_dict: dict[str, torch.Tensor],
    command: torch.Tensor | None,
    *,
    kind: str,
    actor_dim: int,
) -> np.ndarray:
    if kind == "rwm":
        raise ValueError("RWM policy observations are built from extractor state, not obs_dict.")
    if kind == "actor":
        return _obs_group_np(obs_dict, "actor", command, actor_dim=actor_dim)
    if kind == "critic":
        return _obs_group_np(obs_dict, "critic", command, actor_dim=actor_dim)
    if kind == "actor_with_critic_tail":
        actor = obs_dict["actor"].detach().clone()
        critic = obs_dict["critic"].detach().clone()
        actor = _overwrite_command(actor, command, actor_dim=actor_dim)
        critic = _overwrite_command(critic, command, actor_dim=actor_dim)
        return torch.cat([actor, critic[:, actor_dim:]], dim=-1).cpu().numpy().astype(np.float32)
    if kind == "actor_critic":
        actor = obs_dict["actor"].detach().clone()
        critic = obs_dict["critic"].detach().clone()
        actor = _overwrite_command(actor, command, actor_dim=actor_dim)
        critic = _overwrite_command(critic, command, actor_dim=actor_dim)
        return torch.cat([actor, critic], dim=-1).cpu().numpy().astype(np.float32)
    raise ValueError(f"Unknown policy observation kind: {kind!r}")


def _resolve_dataset_obs_kind(kind: str, *, actor_dim: int) -> str:
    if kind != "auto":
        return kind
    return "proprioceptive" if int(actor_dim) == 45 else "full_rwm"


def _dataset_obs_t(full_rwm_obs: torch.Tensor, *, kind: str) -> torch.Tensor:
    if kind == "full_rwm":
        return full_rwm_obs
    if kind == "proprioceptive":
        return proprioceptive_obs_t(full_rwm_obs)
    raise ValueError(f"Unknown dataset_obs_kind: {kind!r}")


def _parse_modes(spec: str) -> list[CommandMode]:
    requested = [item.strip() for item in spec.split(",") if item.strip()]
    by_name = {mode.name: mode for mode in COMMAND_MODES}
    unknown = [name for name in requested if name not in by_name]
    if unknown:
        raise ValueError(f"Unknown command modes: {unknown}. Allowed: {sorted(by_name)}")
    if not requested:
        raise ValueError("At least one command mode is required.")
    return [by_name[name] for name in requested]


def _parse_mode_weights(spec: str | None, modes: list[CommandMode], device: torch.device | str) -> torch.Tensor:
    if spec is None:
        return torch.full((len(modes),), 1.0 / len(modes), device=device)
    raw: dict[str, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        name, value = item.split(":", maxsplit=1)
        raw[name.strip()] = float(value)
    weights = torch.tensor([raw.get(mode.name, 0.0) for mode in modes], dtype=torch.float32, device=device)
    if float(weights.sum()) <= 0.0:
        raise ValueError("command_mode_weights must sum to a positive value.")
    return weights / weights.sum()


def _sample_signed_abs(
    n: int,
    abs_range: tuple[float, float],
    device: torch.device | str,
) -> torch.Tensor:
    lo, hi = abs_range
    mag = lo + torch.rand(n, device=device) * (hi - lo)
    sign = torch.where(torch.rand(n, device=device) < 0.5, -1.0, 1.0)
    return mag * sign


def _sample_commands_for_modes(
    mode_ids: torch.Tensor,
    modes: list[CommandMode],
    *,
    x_range: tuple[float, float],
    signed_x: bool,
    x_abs_range: tuple[float, float],
    y_abs_range: tuple[float, float],
    yaw_abs_range: tuple[float, float],
) -> torch.Tensor:
    device = mode_ids.device
    commands = torch.zeros(mode_ids.numel(), 3, device=device)
    for mode_idx, mode in enumerate(modes):
        env_ids = (mode_ids == mode_idx).nonzero(as_tuple=False).flatten()
        if env_ids.numel() == 0:
            continue
        n = int(env_ids.numel())
        if "x" in mode.axes:
            if signed_x:
                commands[env_ids, 0] = _sample_signed_abs(n, x_abs_range, device)
            else:
                commands[env_ids, 0] = x_range[0] + torch.rand(n, device=device) * (x_range[1] - x_range[0])
        if "y" in mode.axes:
            commands[env_ids, 1] = _sample_signed_abs(n, y_abs_range, device)
        if "yaw" in mode.axes:
            commands[env_ids, 2] = _sample_signed_abs(n, yaw_abs_range, device)
    return commands


def _sample_mode_ids(weights: torch.Tensor, num_envs: int) -> torch.Tensor:
    return torch.multinomial(weights, num_envs, replacement=True)


def _force_commands(env: Any, commands: torch.Tensor) -> None:
    try:
        command_term = env.command_manager.get_term("twist")
    except Exception:
        command_term = None
    if command_term is None:
        return
    if hasattr(command_term, "vel_command_b"):
        command_term.vel_command_b[:, :] = commands
    if hasattr(command_term, "is_standing_env"):
        command_term.is_standing_env[:] = False
    if hasattr(command_term, "is_heading_env"):
        command_term.is_heading_env[:] = False


def _configure_command_cfg(env_cfg: Any, args: argparse.Namespace) -> None:
    if "twist" not in env_cfg.commands:
        return
    twist_cmd = env_cfg.commands["twist"]
    twist_cmd.ranges.lin_vel_x = tuple(args.x_range)
    twist_cmd.ranges.lin_vel_y = (-float(args.y_abs_range[1]), float(args.y_abs_range[1]))
    twist_cmd.ranges.ang_vel_z = (-float(args.yaw_abs_range[1]), float(args.yaw_abs_range[1]))
    if hasattr(twist_cmd.ranges, "heading"):
        twist_cmd.ranges.heading = None
    if hasattr(twist_cmd, "heading_command"):
        twist_cmd.heading_command = False
    if hasattr(twist_cmd, "rel_heading_envs"):
        twist_cmd.rel_heading_envs = 0.0
    if hasattr(twist_cmd, "rel_standing_envs"):
        twist_cmd.rel_standing_envs = 0.0
    if hasattr(twist_cmd, "init_velocity_prob"):
        twist_cmd.init_velocity_prob = 0.0


def _load_expert_agent(
    checkpoint_path: str,
    *,
    num_envs: int,
    actor_dim: int,
    critic_dim: int | None,
    action_dim: int,
    device: str,
) -> LoadedPolicy:
    ckpt_dir = resolve_repo_path(checkpoint_path)
    rwm_cfg_path = ckpt_dir / "rwm_flashsac_config.yaml"
    flashsac_cfg_path = ckpt_dir / "flashsac_config.yaml"

    if rwm_cfg_path.exists():
        cfg = load_flashsac_config(rwm_cfg_path)
        cfg.agent.device_type = device
        cfg.agent.buffer_device_type = "cpu"
        cfg.agent.buffer_max_length = 1024
        cfg.agent.buffer_min_length = 1
        cfg.agent.sample_batch_size = 32
        cfg.agent.load_optimizer = False
        cfg.agent.load_reward_normalizer = False
        if device.startswith("cpu"):
            cfg.agent.use_amp = False
        obs_space, action_space = make_vector_spaces(num_envs, obs_dim=48, action_dim=action_dim)
        create_agent_fn = (
            create_go2_flashsac_proprioceptive_agent
            if bool(cfg.agent.get("asymmetric_observation", False))
            else create_go2_flashsac_agent
        )
        agent = create_agent_fn(obs_space, action_space, make_flashsac_config(cfg, device=device))
        agent.load(str(ckpt_dir))
        return LoadedPolicy(agent=agent, observation_kind="rwm", config_path=rwm_cfg_path)

    if not flashsac_cfg_path.exists():
        raise FileNotFoundError(
            f"FlashSAC checkpoint config not found. Expected either {rwm_cfg_path} or {flashsac_cfg_path}."
        )

    cfg = load_flashsac_config(flashsac_cfg_path)
    cfg.agent.device_type = device
    cfg.agent.buffer_device_type = "cpu"
    cfg.agent.buffer_max_length = 1024
    cfg.agent.buffer_min_length = 1
    cfg.agent.sample_batch_size = 32
    cfg.agent.load_optimizer = False
    cfg.agent.load_reward_normalizer = False
    if device.startswith("cpu"):
        cfg.agent.use_amp = False

    has_critic = critic_dim is not None
    use_critic_full = bool(cfg.env.get("use_critic_observation_as_full_observation", False))
    if use_critic_full:
        if not has_critic:
            raise RuntimeError(
                f"{flashsac_cfg_path} expects critic observations as the full observation, "
                "but the collection task has no critic observation group."
            )
        # Preserve the checkpoint's full critic-sized input while ensuring the
        # actor prefix comes from the (possibly corrupted) actor observation.
        observation_kind = "actor_with_critic_tail"
        obs_dim = int(critic_dim)
    elif has_critic:
        observation_kind = "actor_critic"
        obs_dim = int(actor_dim) + int(critic_dim)
    else:
        observation_kind = "actor"
        obs_dim = int(actor_dim)

    env_info: dict[str, Any] = {}
    if bool(cfg.agent.get("asymmetric_observation", False)):
        env_info["actor_observation_size"] = (int(actor_dim),)

    obs_space, action_space = make_vector_spaces(num_envs, obs_dim=obs_dim, action_dim=action_dim)
    agent = create_agent(obs_space, action_space, env_info, cfg.agent)
    agent.load(str(ckpt_dir))
    return LoadedPolicy(agent=agent, observation_kind=observation_kind, config_path=flashsac_cfg_path)


def _maybe_load_agent(
    checkpoint_path: str | None,
    *,
    num_envs: int,
    actor_dim: int,
    critic_dim: int | None,
    action_dim: int,
    device: str,
) -> LoadedPolicy | None:
    if checkpoint_path is None or str(checkpoint_path).strip().lower() in {"", "none", "null"}:
        return None
    return _load_expert_agent(
        str(checkpoint_path),
        num_envs=num_envs,
        actor_dim=actor_dim,
        critic_dim=critic_dim,
        action_dim=action_dim,
        device=device,
    )


def _policy_actions(
    policy: LoadedPolicy | None,
    observations_by_kind: dict[str, np.ndarray],
    *,
    action_temperature: float | None = None,
) -> torch.Tensor | None:
    if policy is None:
        return None
    observations = observations_by_kind[policy.observation_kind]
    sample_kwargs = {
        "interaction_step": 0,
        "prev_transition": {"next_observation": observations},
        "training": False,
    }
    # Ordinary expert collection uses the stock deterministic FlashSAC API,
    # which has no action_temperature keyword.  Temperature is a TRACE-only
    # extension and must not leak into the baseline collection path.
    if action_temperature is not None:
        sample_kwargs["action_temperature"] = float(action_temperature)
    actions_np = policy.agent.sample_actions(**sample_kwargs).astype(np.float32)
    return torch.from_numpy(actions_np)


def _mixed_actions(
    *,
    collector_ids: torch.Tensor,
    observations_by_kind: dict[str, np.ndarray],
    expert_agent: LoadedPolicy | None,
    medium_agent: LoadedPolicy | None,
    action_dim: int,
    device: torch.device | str,
    action_noise_std: float,
    medium_action_noise_std: float,
    failure_action_noise_std: float,
    action_temperature: float | None = None,
) -> torch.Tensor:
    actions = torch.empty(int(collector_ids.numel()), action_dim, device=device).uniform_(-1.0, 1.0)

    expert_t = _policy_actions(
        expert_agent,
        observations_by_kind,
        action_temperature=action_temperature,
    )
    if expert_t is not None:
        expert_t = expert_t.to(device=device, dtype=torch.float32)
    medium_t = _policy_actions(
        medium_agent,
        observations_by_kind,
        action_temperature=action_temperature,
    )
    if medium_t is not None:
        medium_t = medium_t.to(device=device, dtype=torch.float32)
    if medium_t is None:
        medium_t = expert_t

    expert_mask = collector_ids == COLLECTOR_NAME_TO_ID["expert"]
    if expert_t is not None and expert_mask.any():
        actions[expert_mask] = expert_t[expert_mask]

    noisy_mask = collector_ids == COLLECTOR_NAME_TO_ID["noisy_expert"]
    if expert_t is not None and noisy_mask.any():
        actions[noisy_mask] = expert_t[noisy_mask] + action_noise_std * torch.randn_like(actions[noisy_mask])

    medium_mask = collector_ids == COLLECTOR_NAME_TO_ID["medium"]
    if medium_t is not None and medium_mask.any():
        actions[medium_mask] = medium_t[medium_mask] + medium_action_noise_std * torch.randn_like(actions[medium_mask])

    failure_mask = collector_ids == COLLECTOR_NAME_TO_ID["failure_border"]
    if expert_t is not None and failure_mask.any():
        actions[failure_mask] = expert_t[failure_mask] + failure_action_noise_std * torch.randn_like(actions[failure_mask])

    return actions.clamp(-1.0, 1.0)


def _validate_action_interface_args(args: argparse.Namespace) -> tuple[int, int, float, float]:
    delay_min = int(args.env_action_delay_steps[0])
    delay_max = int(args.env_action_delay_steps[1])
    if delay_min < 0 or delay_max < delay_min:
        raise ValueError(f"Invalid env_action_delay_steps: {args.env_action_delay_steps}")
    scale_min = float(args.env_action_scale_range[0])
    scale_max = float(args.env_action_scale_range[1])
    if scale_min <= 0.0 or scale_max < scale_min:
        raise ValueError(f"Invalid env_action_scale_range: {args.env_action_scale_range}")
    return delay_min, delay_max, scale_min, scale_max


def _sample_env_action_scale(
    num_envs: int,
    action_dim: int,
    *,
    scale_min: float,
    scale_max: float,
    device: torch.device | str,
) -> torch.Tensor:
    if scale_min == 1.0 and scale_max == 1.0:
        return torch.ones(num_envs, action_dim, device=device)
    return torch.empty(num_envs, action_dim, device=device).uniform_(scale_min, scale_max)


def _sample_env_action_bias(
    num_envs: int,
    action_dim: int,
    *,
    bias_std: float,
    device: torch.device | str,
) -> torch.Tensor:
    if bias_std <= 0.0:
        return torch.zeros(num_envs, action_dim, device=device)
    return torch.randn(num_envs, action_dim, device=device) * float(bias_std)


def _apply_env_step_action_interface(
    actions: torch.Tensor,
    *,
    args: argparse.Namespace,
    action_delay_history: torch.Tensor,
    action_scale: torch.Tensor,
    action_bias: torch.Tensor,
    delay_min: int,
    delay_max: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_t = actions * action_scale + action_bias
    if float(args.env_action_noise_std) > 0.0:
        action_t = action_t + float(args.env_action_noise_std) * torch.randn_like(action_t)
    action_t = action_t.clamp(-1.0, 1.0)
    if delay_max <= 0:
        delays = torch.zeros(actions.shape[0], device=actions.device, dtype=torch.long)
        return action_t, delays

    action_delay_history[:-1].copy_(action_delay_history[1:].clone())
    action_delay_history[-1].copy_(action_t)
    delays = torch.randint(delay_min, delay_max + 1, (actions.shape[0],), device=actions.device)
    env_ids = torch.arange(actions.shape[0], device=actions.device)
    return action_delay_history[delay_max - delays, env_ids].clone(), delays


def _startup_domain_table(env: Any) -> dict[str, Any]:
    """Capture startup-sampled physical parameters for every vector environment."""

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    model = base_env.sim.model
    collision_local_ids = [
        idx for idx, name in enumerate(robot.geom_names) if str(name).endswith("_collision")
    ]
    collision_global_ids = robot.indexing.geom_ids[
        torch.as_tensor(collision_local_ids, device=base_env.device, dtype=torch.long)
    ].long()
    body_global_ids = robot.indexing.body_ids.long()
    base_body_id = body_global_ids[0]
    link_body_ids = body_global_ids[1:]

    default_mass = base_env.sim.get_default_field("body_mass")
    default_inertia = base_env.sim.get_default_field("body_inertia")
    default_ipos = base_env.sim.get_default_field("body_ipos")
    realized_mass = model.body_mass[:, body_global_ids]
    realized_inertia = model.body_inertia[:, body_global_ids]

    return {
        "startup_domain_id": torch.arange(base_env.num_envs, dtype=torch.long),
        "environment_local_id": torch.arange(base_env.num_envs, dtype=torch.long),
        "collision_geom_names": [robot.geom_names[idx] for idx in collision_local_ids],
        "body_names": list(robot.body_names),
        "collision_friction": model.geom_friction[:, collision_global_ids, 0].detach().cpu(),
        "base_mass_delta": (model.body_mass[:, base_body_id] - default_mass[base_body_id]).detach().cpu(),
        "base_mass": model.body_mass[:, base_body_id].detach().cpu(),
        "base_inertia_scale": (
            model.body_inertia[:, base_body_id] / default_inertia[base_body_id].clamp_min(1e-12)
        ).detach().cpu(),
        "link_mass_scale": (
            model.body_mass[:, link_body_ids] / default_mass[link_body_ids].clamp_min(1e-12)
        ).detach().cpu(),
        "link_inertia_scale": (
            realized_inertia[:, 1:] / default_inertia[link_body_ids].clamp_min(1e-12)
        ).detach().cpu(),
        "base_com_offset": (
            model.body_ipos[:, base_body_id] - default_ipos[base_body_id]
        ).detach().cpu(),
        "realized_body_mass": realized_mass.detach().cpu(),
    }


def _episode_domain_rows(
    env: Any,
    *,
    env_ids: torch.Tensor,
    episode_ids: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Capture reset-time parameters once per episode instead of per transition."""

    base_env = env.unwrapped
    env_ids = env_ids.to(base_env.device, dtype=torch.long)
    runtime = friend_dr_runtime_snapshot(base_env, env_ids)
    robot = base_env.scene["robot"]
    num_rows = len(env_ids)
    action_dim = int(robot.num_actuators)

    def runtime_or(key: str, width: int, value: float) -> torch.Tensor:
        tensor = runtime.get(key)
        if tensor is None:
            return torch.full((num_rows, width), value, device=base_env.device)
        return tensor

    root_velocity = getattr(robot.data, "root_link_vel_w", None)
    if root_velocity is None:
        root_velocity_rows = torch.zeros(num_rows, 6, device=base_env.device)
    else:
        root_velocity_rows = root_velocity[env_ids, :6]
    joint_position_rows = robot.data.joint_pos[env_ids]
    joint_velocity_rows = robot.data.joint_vel[env_ids]
    default_joint_position = robot.data.default_joint_pos[env_ids]
    derived_joint_scale = joint_position_rows / torch.where(
        default_joint_position.abs() > 1e-8,
        default_joint_position,
        torch.ones_like(default_joint_position),
    )
    joint_reset_scale = runtime.get("joint_reset_scale", derived_joint_scale)

    return {
        "episode_id": episode_ids.detach().cpu().long(),
        "startup_domain_id": env_ids.detach().cpu().long(),
        "kp_scale": runtime_or("kp_scale", action_dim, 1.0).detach().cpu().float(),
        "kd_scale": runtime_or("kd_scale", action_dim, 1.0).detach().cpu().float(),
        "motor_strength": runtime_or("motor_strength", action_dim, 1.0).detach().cpu().float(),
        "motor_zero_offset": runtime_or("motor_zero_offset", action_dim, 0.0).detach().cpu().float(),
        "joint_reset_scale": joint_reset_scale.detach().cpu().float(),
        "reset_joint_position": joint_position_rows.detach().cpu().float(),
        "reset_joint_velocity": joint_velocity_rows.detach().cpu().float(),
        "reset_base_velocity": root_velocity_rows.detach().cpu().float(),
    }


def _append_table_chunks(
    chunks: dict[str, list[torch.Tensor]],
    rows: dict[str, torch.Tensor],
) -> None:
    for key, value in rows.items():
        chunks.setdefault(key, []).append(value)


def _finish_table_chunks(chunks: dict[str, list[torch.Tensor]]) -> dict[str, torch.Tensor]:
    return {key: torch.cat(values, dim=0) for key, values in chunks.items()}


def _new_builder(args: argparse.Namespace, dims: Any, metadata: dict[str, Any]) -> Go2MixedDatasetBuilder:
    return Go2MixedDatasetBuilder(
        state_dim=int(dims.state_dim),
        action_dim=int(dims.action_dim),
        contact_dim=int(dims.contact_dim),
        termination_dim=int(dims.termination_dim),
        num_envs=int(args.num_envs),
        capacity=int(args.num_transitions),
        metadata=metadata,
    )


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse_args()
    if args.trace_reset_dataset and (
        not math.isfinite(float(args.trace_action_temperature)) or float(args.trace_action_temperature) <= 0.0
    ):
        raise ValueError(
            "TRACE imperfect-simulator action temperature must be finite and greater than zero, "
            f"got {args.trace_action_temperature}."
        )
    device = select_device(args.device)
    set_seed(int(args.seed))
    random.seed(int(args.seed))

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(str(args.task))
    env_cfg.scene.num_envs = int(args.num_envs)
    env_cfg.seed = int(args.seed)
    if hasattr(env_cfg, "auto_reset"):
        env_cfg.auto_reset = True
    requested_strength_scales = _parse_joint_strength_scales(args.joint_strength_scales)
    broken_joint_names = tuple(dict.fromkeys(str(name) for name in args.broken_joint_names if str(name)))
    joint_strength_scales = dict(requested_strength_scales)
    for joint_name in broken_joint_names:
        joint_strength_scales[joint_name] = 0.0
    applied_strength_scales = apply_go2_pd_joint_strength_scales(env_cfg, joint_strength_scales)
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=bool(args.use_domain_randomization),
        use_push_randomization=bool(args.use_push_randomization),
        use_observation_noise=bool(args.use_observation_noise),
        randomization_preset=str(args.randomization_preset),
        randomization_components=args.randomization_components,
        randomization_scale=float(args.randomization_scale),
        payload_mass_range_kg=tuple(float(x) for x in args.payload_mass_range_kg),
        payload_position_body_m=tuple(float(x) for x in args.payload_position_body_m),
        payload_box_size_m=tuple(float(x) for x in args.payload_box_size_m),
        rr_calf_strength_range=tuple(float(x) for x in args.rr_calf_strength_range),
    )
    _configure_command_cfg(env_cfg, args)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    extractor = Go2RWMExtractor(env.unwrapped)
    dims = extractor.dims
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    critic_space = env.single_observation_space.spaces.get("critic")
    critic_dim = None if critic_space is None else int(critic_space.shape[0])
    action_dim = int(env.single_action_space.shape[0])
    delay_min, delay_max, action_scale_min, action_scale_max = _validate_action_interface_args(args)
    dataset_obs_kind = _resolve_dataset_obs_kind(str(args.dataset_obs_kind), actor_dim=actor_dim)
    dataset_obs_dim = PROPRIOCEPTIVE_ACTOR_OBS_DIM if dataset_obs_kind == "proprioceptive" else int(dims.policy_obs_dim)
    trace_source_states: torch.Tensor | None = None
    trace_source_next_states: torch.Tensor | None = None
    trace_start_state_ids: torch.Tensor | None = None
    trace_source_commands: torch.Tensor | None = None
    trace_source_previous_actions: torch.Tensor | None = None
    trace_source_actions: torch.Tensor | None = None
    trace_source_snapshot: dict[str, Any] | None = None
    if args.trace_reset_dataset:
        if int(args.trace_rollout_length) < 1 or int(args.trace_trajectories_per_state) < 1:
            raise ValueError("TRACE rollout length and trajectories per state must be positive.")
        trace_source_path = resolve_repo_path(str(args.trace_reset_dataset))
        trace_source_dataset = torch.load(trace_source_path, map_location="cpu", weights_only=False)
        source_states = stack_time_key(trace_source_dataset, "states").reshape(-1, int(dims.state_dim)).float()
        source_next_states = stack_time_key(trace_source_dataset, "next_states").reshape(
            -1, int(dims.state_dim)
        ).float()
        if source_states.shape[-1] != 45:
            raise ValueError(f"TRACE reset dataset must contain 45D states, got {source_states.shape[-1]}.")
        if len(source_next_states) != len(source_states):
            raise ValueError("TRACE source states and next_states must have identical flattened lengths.")
        num_unique = int(np.ceil(int(args.num_envs) / int(args.trace_trajectories_per_state)))
        generator = torch.Generator(device="cpu").manual_seed(int(args.seed) + 1701)
        if len(source_states) >= num_unique:
            selected_ids = torch.randperm(len(source_states), generator=generator)[:num_unique]
        else:
            selected_ids = torch.randint(len(source_states), (num_unique,), generator=generator)
        trace_start_state_ids = selected_ids.repeat_interleave(int(args.trace_trajectories_per_state))[
            : int(args.num_envs)
        ]
        trace_source_states = source_states[trace_start_state_ids].to(env.device)
        trace_source_next_states = source_next_states[trace_start_state_ids].to(env.device)
        source_commands = stack_time_key(trace_source_dataset, "commands").reshape(-1, 3).float()
        source_previous_actions = stack_time_key(trace_source_dataset, "prev_actions").reshape(
            -1, action_dim
        ).float()
        source_actions = stack_time_key(trace_source_dataset, "actions").reshape(-1, action_dim).float()
        if (
            len(source_commands) != len(source_states)
            or len(source_previous_actions) != len(source_states)
            or len(source_actions) != len(source_states)
        ):
            raise ValueError(
                "TRACE source states/next_states/commands/prev_actions/actions must have identical flattened lengths."
            )
        trace_source_commands = source_commands[trace_start_state_ids].to(env.device)
        trace_source_previous_actions = source_previous_actions[trace_start_state_ids].to(env.device)
        trace_source_actions = source_actions[trace_start_state_ids].to(env.device)
        if args.trace_reset_mode == "exact_snapshot":
            snapshot_keys = {
                "root_state_local": "sim_root_states_local",
                "joint_position": "sim_joint_positions",
                "joint_velocity": "sim_joint_velocities",
                "action": "sim_action_histories",
                "prev_action": "sim_prev_action_histories",
                "prev_prev_action": "sim_prev_prev_action_histories",
                "command": "sim_snapshot_commands",
            }
            missing = [dataset_key for dataset_key in snapshot_keys.values() if trace_source_dataset.get(dataset_key) is None]
            if missing:
                raise ValueError(
                    "TRACE exact_snapshot mode requires simulator snapshots; missing "
                    f"{missing}. Regenerate the simulator source with --save_trace_snapshots."
                )
            leader_snapshot: dict[str, Any] = {"snapshot_version": SNAPSHOT_VERSION}
            for snapshot_key, dataset_key in snapshot_keys.items():
                values = stack_time_key(trace_source_dataset, dataset_key).reshape(
                    len(source_states), -1
                )
                leader_snapshot[snapshot_key] = values[selected_ids]
            trace_source_snapshot = {
                key: value.to(env.device) if isinstance(value, torch.Tensor) else value
                for key, value in repeat_snapshot_rows(
                    leader_snapshot,
                    int(args.trace_trajectories_per_state),
                ).items()
            }
            # Keep source command/action columns authoritative even if an old
            # snapshot was captured with inconsistent auxiliary fields.
            trace_source_snapshot["command"] = trace_source_commands
            trace_source_snapshot["action"] = trace_source_previous_actions

    expert_agent = _maybe_load_agent(
        None if args.expert_policy_path is None else str(args.expert_policy_path),
        num_envs=int(args.num_envs),
        actor_dim=actor_dim,
        critic_dim=critic_dim,
        action_dim=action_dim,
        device=device,
    )
    medium_agent = _maybe_load_agent(
        None if args.medium_policy_path is None else str(args.medium_policy_path),
        num_envs=int(args.num_envs),
        actor_dim=actor_dim,
        critic_dim=critic_dim,
        action_dim=action_dim,
        device=device,
    )

    modes = _parse_modes(str(args.command_modes))
    mode_weights = _parse_mode_weights(args.command_mode_weights, modes, env.device)
    collector_mix = parse_collector_mix(str(args.collector_mix))
    if expert_agent is None:
        non_random = {name: weight for name, weight in collector_mix.items() if name != "random" and weight > 0.0}
        if non_random:
            raise ValueError(f"collector_mix uses policy-based collectors {non_random}, but expert_policy_path is empty.")
    x_range = (float(args.x_range[0]), float(args.x_range[1]))
    x_abs_range = (float(args.x_abs_range[0]), float(args.x_abs_range[1]))
    y_abs_range = (float(args.y_abs_range[0]), float(args.y_abs_range[1]))
    yaw_abs_range = (float(args.yaw_abs_range[0]), float(args.yaw_abs_range[1]))
    save_path = resolve_repo_path(str(args.save_path))
    part_dir = save_path.parent / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    metadata = {
        "task": str(args.task),
        "robot": "Unitree-Go2",
        "seed": int(args.seed),
        "dt": float(getattr(env_cfg, "step_dt", 0.02)),
        "obs_dim": int(dataset_obs_dim),
        "dataset_obs_kind": dataset_obs_kind,
        "full_rwm_obs_dim": CRITIC_OBS_DIM_FULL_RWM,
        "action_dim": action_dim,
        "state_dim": int(dims.state_dim),
        "contact_dim": int(dims.contact_dim),
        "termination_dim": int(dims.termination_dim),
        "live_actor_obs_dim": actor_dim,
        "live_critic_obs_dim": critic_dim,
        "actuator_names": list(env.scene["robot"].actuator_names),
        "joint_names": list(env.scene["robot"].joint_names),
        "expert_policy_observation_kind": None if expert_agent is None else expert_agent.observation_kind,
        "medium_policy_observation_kind": None if medium_agent is None else medium_agent.observation_kind,
        "num_envs": int(args.num_envs),
        "shard_id": int(args.shard_id),
        "requested_num_transitions": int(args.num_transitions),
        "collector_mix": dict(collector_mix),
        "collector_id_to_name": dict(COLLECTOR_ID_TO_NAME),
        "expert_policy_path": str(args.expert_policy_path),
        "medium_policy_path": None if args.medium_policy_path is None else str(args.medium_policy_path),
        "action_noise_std": float(args.action_noise_std),
        "medium_action_noise_std": float(args.medium_action_noise_std),
        "failure_action_noise_std": float(args.failure_action_noise_std),
        "command_modes": [mode.name for mode in modes],
        "command_mode_weights": {mode.name: float(mode_weights[i].detach().cpu()) for i, mode in enumerate(modes)},
        "command_ranges": {
            "x_range": list(x_range),
            "signed_x": bool(args.signed_x),
            "x_abs_range": list(x_abs_range),
            "y_abs_range": list(y_abs_range),
            "yaw_abs_range": list(yaw_abs_range),
        },
        "command_resample_interval": [
            int(args.command_resample_interval_min),
            int(args.command_resample_interval_max),
        ],
        "broken_pd_joint_names": list(broken_joint_names),
        "joint_strength_scales": dict(applied_strength_scales),
        "target_gap": {
            "payload_mass_range_kg": [float(x) for x in args.payload_mass_range_kg],
            "payload_position_body_m": [float(x) for x in args.payload_position_body_m],
            "payload_box_size_m": [float(x) for x in args.payload_box_size_m],
            "rr_calf_strength_range": [float(x) for x in args.rr_calf_strength_range],
        },
        "fixed_collector_assignment": bool(args.fixed_collector_assignment),
        "collector_assignment_seed": int(args.collector_assignment_seed),
        "env_randomization": {
            "use_domain_randomization": bool(args.use_domain_randomization),
            "use_push_randomization": bool(args.use_push_randomization),
            "use_observation_noise": bool(args.use_observation_noise),
            "randomization_preset": str(args.randomization_preset),
            "randomization_components": args.randomization_components,
            "randomization_scale": float(args.randomization_scale),
            "manifest": mjlab_randomization_manifest(
                str(args.randomization_preset),
                args.randomization_components,
                float(args.randomization_scale),
            ),
        },
        "env_step_action_interface": {
            "env_action_noise_std": float(args.env_action_noise_std),
            "env_action_bias_std": float(args.env_action_bias_std),
            "env_action_scale_range": [float(action_scale_min), float(action_scale_max)],
            "env_action_delay_steps": [int(delay_min), int(delay_max)],
            "dataset_action": "post_interface_env_step_action",
        },
        "trace_candidates": {
            "enabled": trace_source_states is not None,
            "reset_dataset": None if args.trace_reset_dataset is None else str(args.trace_reset_dataset),
            "rollout_length": int(args.trace_rollout_length),
            "trajectories_per_state": int(args.trace_trajectories_per_state),
            "action_temperature": (
                float(args.trace_action_temperature) if trace_source_states is not None else None
            ),
            "reset_mode": str(args.trace_reset_mode) if trace_source_states is not None else None,
            "reset_is_exact": bool(
                trace_source_states is not None and args.trace_reset_mode == "exact_snapshot"
            ),
            "snapshot_version": SNAPSHOT_VERSION if trace_source_states is not None else None,
            "controlled_branch_domain": bool(trace_source_states is not None),
            "command_from_source_transition": bool(trace_source_states is not None),
            "stop_on_done": bool(trace_source_states is not None),
            "unrecoverable_fields": (
                []
                if trace_source_states is not None and args.trace_reset_mode == "exact_snapshot"
                else ["root_height", "root_yaw", "simulator_hidden_state"]
            ),
        },
        "save_trace_snapshots": bool(args.save_trace_snapshots),
    }

    builder = _new_builder(args, dims, metadata)
    part_paths: list[Path] = []

    def flush_part(force: bool = False) -> None:
        nonlocal builder
        if builder.num_time_steps == 0:
            return
        if not force and (int(args.chunk_size) <= 0 or builder.num_transitions < int(args.chunk_size)):
            return
        part_path = part_dir / f"dataset_part_{len(part_paths):03d}.pt"
        builder.save(part_path)
        part_paths.append(part_path)
        print(f"[Go2-ExpertCoverageDataset] saved part {part_path} ({builder.num_transitions} transitions)")
        builder = _new_builder(args, dims, metadata)

    obs_dict, _ = env.reset()
    mode_ids = _sample_mode_ids(mode_weights, int(args.num_envs))
    commands = _sample_commands_for_modes(
        mode_ids,
        modes,
        x_range=x_range,
        signed_x=bool(args.signed_x),
        x_abs_range=x_abs_range,
        y_abs_range=y_abs_range,
        yaw_abs_range=yaw_abs_range,
    )
    intervals = torch.randint(
        int(args.command_resample_interval_min),
        int(args.command_resample_interval_max) + 1,
        (int(args.num_envs),),
        device=env.device,
    )
    command_ages = torch.zeros(int(args.num_envs), dtype=torch.long, device=env.device)
    if trace_source_commands is not None:
        commands = trace_source_commands.clone()
    _force_commands(env.unwrapped, commands)

    trace_reset_error = torch.full(
        (int(args.num_envs),),
        float("nan"),
        device=env.device,
    )

    realized_trace_snapshot: dict[str, Any] | None = None

    def reset_trace_candidates() -> None:
        nonlocal obs_dict, trace_reset_error, realized_trace_snapshot
        if trace_source_states is None:
            return
        all_ids = torch.arange(int(args.num_envs), device=env.device)
        synchronize_branch_domain_parameters(
            env.unwrapped,
            int(args.trace_trajectories_per_state),
        )
        if args.trace_reset_mode == "exact_snapshot":
            delay_state = friend_dr_runtime_snapshot(env.unwrapped).get("actuator_delay_substeps")
            if delay_max > 0 or (delay_state is not None and bool((delay_state != 0).any())):
                raise RuntimeError(
                    "exact_snapshot currently requires zero action/actuator delay; "
                    "delay history is not present in the source snapshot."
                )
            if trace_source_snapshot is None:
                raise RuntimeError("exact_snapshot mode was selected without a loaded snapshot.")
            realized_trace_snapshot = trace_source_snapshot
        else:
            branch_count = int(args.trace_trajectories_per_state)
            leader_ids = torch.arange(0, int(args.num_envs), branch_count, device=env.device)
            leader_snapshot = canonical_snapshot_from_rwm_state(
                env.unwrapped,
                trace_source_states[leader_ids],
                trace_source_commands[leader_ids],
                trace_source_previous_actions[leader_ids],
                leader_ids,
            )
            realized_trace_snapshot = repeat_snapshot_rows(leader_snapshot, branch_count)
        restore_go2_simulator_snapshot(env.unwrapped, realized_trace_snapshot, all_ids)
        reconstructed = extractor.extract_state()
        compared_width = 45 if args.trace_reset_mode == "exact_snapshot" else 33
        trace_reset_error = torch.linalg.norm(
            reconstructed[:, :compared_width] - trace_source_states[:, :compared_width],
            dim=-1,
        )
        obs_dict = env.unwrapped.observation_manager.compute()

    reset_trace_candidates()

    last_actions = (
        trace_source_previous_actions.clone()
        if trace_source_previous_actions is not None
        else torch.zeros(int(args.num_envs), action_dim, device=env.device)
    )
    action_delay_history = torch.zeros(delay_max + 1, int(args.num_envs), action_dim, device=env.device)
    if trace_source_previous_actions is not None:
        action_delay_history[:] = trace_source_previous_actions.unsqueeze(0)
    trace_active = torch.ones(int(args.num_envs), dtype=torch.bool, device=env.device)

    if trace_source_states is not None:
        if args.trace_identity_tolerance <= 0.0:
            raise ValueError("trace_identity_tolerance must be positive.")
        if (
            realized_trace_snapshot is None
            or trace_source_actions is None
            or trace_source_next_states is None
        ):
            raise RuntimeError("TRACE identity check requires realized snapshots and source actions.")
        identity_ids = torch.arange(int(args.num_envs), device=env.device)
        restore_go2_simulator_snapshot(env.unwrapped, realized_trace_snapshot, identity_ids)
        step_without_automatic_reset(env.unwrapped, trace_source_actions)
        identity_next_a = extractor.extract_state().clone()
        identity_term_a = extractor.extract_termination().clone()
        restore_go2_simulator_snapshot(env.unwrapped, realized_trace_snapshot, identity_ids)
        step_without_automatic_reset(env.unwrapped, trace_source_actions)
        identity_next_b = extractor.extract_state().clone()
        identity_term_b = extractor.extract_termination().clone()
        identity_error = torch.max(torch.abs(identity_next_a - identity_next_b), dim=-1).values
        identity_max_error = float(identity_error.max())
        if identity_max_error > float(args.trace_identity_tolerance):
            raise RuntimeError(
                "TRACE repeated one-step identity failed: "
                f"max_state_error={identity_max_error:.6g} > tolerance={args.trace_identity_tolerance:.6g}."
            )
        if not torch.equal(identity_term_a.bool(), identity_term_b.bool()):
            raise RuntimeError("TRACE repeated one-step identity produced inconsistent terminations.")
        source_next_error = torch.max(
            torch.abs(identity_next_a - trace_source_next_states), dim=-1
        ).values
        source_next_max_error = float(source_next_error.max())
        source_match_available = args.trace_reset_mode == "exact_snapshot"
        if source_match_available and source_next_max_error > float(args.trace_identity_tolerance):
            raise RuntimeError(
                "TRACE exact reset failed the dataset transition identity check: "
                f"sim(snapshot, source_action) differs from dataset next_state by "
                f"{source_next_max_error:.6g} > tolerance={args.trace_identity_tolerance:.6g}."
            )
        metadata["trace_candidates"]["one_step_identity"] = {
            "kind": "repeated_snapshot_same_action",
            "max_state_abs_error": identity_max_error,
            "tolerance": float(args.trace_identity_tolerance),
            "passed": True,
        }
        metadata["trace_candidates"]["source_transition_identity"] = {
            "kind": "sim_snapshot_source_action_vs_dataset_next_state",
            "available": bool(source_match_available),
            "max_state_abs_error": source_next_max_error if source_match_available else None,
            "tolerance": float(args.trace_identity_tolerance) if source_match_available else None,
            "passed": True if source_match_available else None,
            "unavailable_reason": (
                None
                if source_match_available
                else "real logs do not contain an exact simulator snapshot"
            ),
        }
        reset_trace_candidates()
        last_actions.copy_(trace_source_previous_actions)
        action_delay_history.copy_(
            trace_source_previous_actions.unsqueeze(0).expand_as(action_delay_history)
        )
    env_action_scale = _sample_env_action_scale(
        int(args.num_envs),
        action_dim,
        scale_min=action_scale_min,
        scale_max=action_scale_max,
        device=env.device,
    )
    env_action_bias = _sample_env_action_bias(
        int(args.num_envs),
        action_dim,
        bias_std=float(args.env_action_bias_std),
        device=env.device,
    )
    episode_ids = torch.arange(int(args.num_envs), device=env.device, dtype=torch.long)
    next_episode_id = int(args.num_envs)
    startup_domain_table = _startup_domain_table(env)
    episode_domain_chunks: dict[str, list[torch.Tensor]] = {}
    all_env_ids = torch.arange(int(args.num_envs), device=env.device, dtype=torch.long)
    _append_table_chunks(
        episode_domain_chunks,
        _episode_domain_rows(env, env_ids=all_env_ids, episode_ids=episode_ids),
    )
    timesteps = torch.zeros(int(args.num_envs), device=env.device, dtype=torch.long)
    if args.fixed_collector_assignment:
        collector_ids = _fixed_collector_ids(
            collector_mix,
            int(args.num_envs),
            env.device,
            int(args.collector_assignment_seed),
        )
    else:
        collector_ids = sample_collector_ids(collector_mix, int(args.num_envs), env.device)
    collector_counts = torch.zeros(len(COLLECTOR_ID_TO_NAME), dtype=torch.long)
    mode_counts = torch.zeros(len(modes), dtype=torch.long)
    returns = torch.zeros(int(args.num_envs), device=env.device)
    lengths = torch.zeros(int(args.num_envs), device=env.device, dtype=torch.long)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    termination_count = 0
    timeout_count = 0

    total_steps = int(np.ceil(int(args.num_transitions) / int(args.num_envs)))
    start_time = time.perf_counter()
    print(f"[Go2-ExpertCoverageDataset] task={args.task}")
    print(f"[Go2-ExpertCoverageDataset] expert_policy={args.expert_policy_path}")
    print(f"[Go2-ExpertCoverageDataset] medium_policy={args.medium_policy_path}")
    print(f"[Go2-ExpertCoverageDataset] save_path={save_path}")
    if broken_joint_names:
        print(f"[Go2-ExpertCoverageDataset] broken_pd_joint_names={broken_joint_names}")
    if applied_strength_scales:
        print(f"[Go2-ExpertCoverageDataset] joint_strength_scales={applied_strength_scales}")
    print(
        "[Go2-ExpertCoverageDataset] dataset_observation="
        f"{dataset_obs_kind} ({dataset_obs_dim} dims); state_dim={int(dims.state_dim)}"
    )
    if expert_agent is not None:
        print(
            "[Go2-ExpertCoverageDataset] expert_policy_observation="
            f"{expert_agent.observation_kind} ({expert_agent.config_path})"
        )
    if medium_agent is not None:
        print(
            "[Go2-ExpertCoverageDataset] medium_policy_observation="
            f"{medium_agent.observation_kind} ({medium_agent.config_path})"
        )
    print(
        f"[Go2-ExpertCoverageDataset] mix={collector_mix}, modes={metadata['command_mode_weights']}, "
        f"num_envs={args.num_envs}, target_transitions={args.num_transitions}"
    )
    print(
        "[Go2-ExpertCoverageDataset] target_gap="
        f"{metadata['target_gap']}, fixed_collector_assignment={args.fixed_collector_assignment}"
    )

    for collection_step in tqdm.trange(total_steps, smoothing=0.1, mininterval=0.5):
        _force_commands(env.unwrapped, commands)
        transition_valid_mask = trace_active.clone()
        transition_episode_ids = episode_ids.clone()
        transition_timesteps = timesteps.clone()
        transition_collector_ids = collector_ids.clone()
        state = extractor.extract_state()
        command = commands.clone()
        prev_action = last_actions.clone()
        transition_simulator_snapshot = (
            capture_go2_simulator_snapshot(
                env.unwrapped,
                torch.arange(int(args.num_envs), device=env.device),
                command=command,
            )
            if args.save_trace_snapshots
            else None
        )
        full_obs_t = make_go2_policy_obs(state, command, prev_action)
        obs_t = _dataset_obs_t(full_obs_t, kind=dataset_obs_kind)
        observations_by_kind = {
            "rwm": full_obs_t.detach().cpu().numpy().astype(np.float32),
            "actor": _policy_obs_np(obs_dict, command, kind="actor", actor_dim=actor_dim),
        }
        if critic_dim is not None:
            observations_by_kind["critic"] = _policy_obs_np(obs_dict, command, kind="critic", actor_dim=actor_dim)
            observations_by_kind["actor_with_critic_tail"] = _policy_obs_np(
                obs_dict,
                command,
                kind="actor_with_critic_tail",
                actor_dim=actor_dim,
            )
            observations_by_kind["actor_critic"] = _policy_obs_np(
                obs_dict,
                command,
                kind="actor_critic",
                actor_dim=actor_dim,
            )

        action_t = _mixed_actions(
            collector_ids=collector_ids,
            observations_by_kind=observations_by_kind,
            expert_agent=expert_agent,
            medium_agent=medium_agent,
            action_dim=action_dim,
            device=env.device,
            action_noise_std=float(args.action_noise_std),
            medium_action_noise_std=float(args.medium_action_noise_std),
            failure_action_noise_std=float(args.failure_action_noise_std),
            action_temperature=(
                float(args.trace_action_temperature) if trace_source_states is not None else None
            ),
        )
        if trace_source_states is not None:
            action_t[~trace_active] = 0.0
        env_action_t, env_action_delay_steps = _apply_env_step_action_interface(
            action_t,
            args=args,
            action_delay_history=action_delay_history,
            action_scale=env_action_scale,
            action_bias=env_action_bias,
            delay_min=delay_min,
            delay_max=delay_max,
        )

        if trace_source_states is not None:
            obs_dict, rewards, terminateds, truncateds, _extras = step_without_automatic_reset(
                env.unwrapped,
                env_action_t,
            )
        else:
            obs_dict, rewards, terminateds, truncateds, _extras = env.step(env_action_t)
        actuator_delay_substeps = friend_dr_runtime_snapshot(env.unwrapped).get(
            "actuator_delay_substeps",
            torch.zeros(int(args.num_envs), device=env.device, dtype=torch.long),
        )
        next_state = extractor.extract_state()
        contact = extractor.extract_contact()
        termination = extractor.extract_termination()
        done = terminateds | truncateds
        if trace_source_states is not None:
            done = done & trace_active

        counted_collector_ids = collector_ids[transition_valid_mask] if trace_source_states is not None else collector_ids
        counted_mode_ids = mode_ids[transition_valid_mask] if trace_source_states is not None else mode_ids
        collector_counts += torch.bincount(
            counted_collector_ids.detach().cpu(),
            minlength=len(COLLECTOR_ID_TO_NAME),
        )
        mode_counts += torch.bincount(counted_mode_ids.detach().cpu(), minlength=len(modes))

        returns += rewards
        lengths += 1
        done_ids = done.nonzero(as_tuple=False).flatten()
        resample_ids = (
            torch.empty(0, device=env.device, dtype=torch.long)
            if trace_source_states is not None
            else (command_ages + 1 >= intervals).nonzero(as_tuple=False).flatten()
        )
        if done_ids.numel() > 0:
            completed_returns.extend(returns[done_ids].detach().cpu().tolist())
            completed_lengths.extend(lengths[done_ids].detach().cpu().float().tolist())
            termination_count += int((terminateds & transition_valid_mask).sum().item())
            timeout_count += int((truncateds & transition_valid_mask).sum().item())
            returns[done_ids] = 0.0
            lengths[done_ids] = 0
            if trace_source_states is not None:
                trace_active[done_ids] = False
            else:
                new_ids = torch.arange(
                    next_episode_id,
                    next_episode_id + int(done_ids.numel()),
                    device=env.device,
                    dtype=torch.long,
                )
                next_episode_id += int(done_ids.numel())
                episode_ids[done_ids] = new_ids
                _append_table_chunks(
                    episode_domain_chunks,
                    _episode_domain_rows(env, env_ids=done_ids, episode_ids=new_ids),
                )
                timesteps[done_ids] = 0
                resample_ids = torch.unique(torch.cat([resample_ids, done_ids]))
            action_delay_history[:, done_ids] = 0.0
            env_action_scale[done_ids] = _sample_env_action_scale(
                int(done_ids.numel()),
                action_dim,
                scale_min=action_scale_min,
                scale_max=action_scale_max,
                device=env.device,
            )
            env_action_bias[done_ids] = _sample_env_action_bias(
                int(done_ids.numel()),
                action_dim,
                bias_std=float(args.env_action_bias_std),
                device=env.device,
            )

        if resample_ids.numel() > 0:
            mode_ids[resample_ids] = _sample_mode_ids(mode_weights, int(resample_ids.numel()))
            commands[resample_ids] = _sample_commands_for_modes(
                mode_ids[resample_ids],
                modes,
                x_range=x_range,
                signed_x=bool(args.signed_x),
                x_abs_range=x_abs_range,
                y_abs_range=y_abs_range,
                yaw_abs_range=yaw_abs_range,
            )
            intervals[resample_ids] = torch.randint(
                int(args.command_resample_interval_min),
                int(args.command_resample_interval_max) + 1,
                (int(resample_ids.numel()),),
                device=env.device,
            )
            command_ages[resample_ids] = 0
            if not args.fixed_collector_assignment:
                collector_ids[resample_ids] = sample_collector_ids(
                    collector_mix,
                    int(resample_ids.numel()),
                    env.device,
                )

        not_done = ~done
        timesteps[not_done] += 1
        command_ages[not_done] += 1
        next_prev_action = env_action_t.clone()
        next_prev_action[done_ids] = 0.0
        last_actions = next_prev_action
        _force_commands(env.unwrapped, commands)
        next_full_obs_t = make_go2_policy_obs(next_state, commands, next_prev_action)
        next_obs_t = _dataset_obs_t(next_full_obs_t, kind=dataset_obs_kind)

        add_kwargs = dict(
            obs=obs_t,
            next_obs=next_obs_t,
            state=state,
            action=env_action_t,
            raw_action=action_t,
            env_action_delay_step=env_action_delay_steps,
            actuator_delay_substep=actuator_delay_substeps,
            noisy_actor_observation=torch.from_numpy(observations_by_kind["actor"]),
            next_state=next_state,
            contact=contact,
            termination=termination,
            command=command,
            reward=rewards,
            done=done,
            timeout=truncateds,
            prev_action=prev_action,
            episode_id=transition_episode_ids,
            timestep=transition_timesteps,
            collector_type=transition_collector_ids,
        )
        # TRACE diagnostics and simulator snapshots belong to the extended
        # builder interface.  Do not pass them to the stock baseline builder
        # during ordinary expert collection.
        if trace_source_states is not None or args.save_trace_snapshots:
            extended = {
                "trace_reset_reconstruction_error": trace_reset_error,
                "trace_valid_mask": transition_valid_mask,
                "simulator_snapshot": transition_simulator_snapshot,
            }
            unsupported = set(extended).difference(
                inspect.signature(builder.add).parameters
            )
            if unsupported:
                raise RuntimeError(
                    "TRACE/snapshot collection requires an extended dataset "
                    f"builder; unsupported fields: {sorted(unsupported)}"
                )
            add_kwargs.update(extended)
        builder.add(**add_kwargs)
        flush_part(force=False)

        # Keep terminated inactive worlds numerically safe without admitting
        # post-reset transitions into the proposal trajectory.
        if trace_source_states is not None and done_ids.numel() > 0:
            if realized_trace_snapshot is None:
                raise RuntimeError("TRACE realized snapshot is unavailable after reset.")
            done_snapshot = {
                "snapshot_version": SNAPSHOT_VERSION,
                **{
                    key: value[done_ids]
                    for key, value in realized_trace_snapshot.items()
                    if isinstance(value, torch.Tensor)
                },
            }
            restore_go2_simulator_snapshot(env.unwrapped, done_snapshot, done_ids)

        trace_boundary = (
            trace_source_states is not None
            and (collection_step + 1) % int(args.trace_rollout_length) == 0
            and (collection_step + 1) < total_steps
        )
        if trace_boundary:
            reset_trace_candidates()
            all_env_ids = torch.arange(int(args.num_envs), device=env.device)
            new_ids = torch.arange(
                next_episode_id,
                next_episode_id + int(args.num_envs),
                device=env.device,
                dtype=torch.long,
            )
            next_episode_id += int(args.num_envs)
            episode_ids = new_ids
            timesteps.zero_()
            returns.zero_()
            lengths.zero_()
            last_actions.copy_(trace_source_previous_actions)
            action_delay_history.copy_(
                trace_source_previous_actions.unsqueeze(0).expand_as(action_delay_history)
            )
            trace_active.fill_(True)
            command_ages.zero_()
            commands = trace_source_commands.clone()
            _force_commands(env.unwrapped, commands)
            _append_table_chunks(
                episode_domain_chunks,
                _episode_domain_rows(env, env_ids=all_env_ids, episode_ids=new_ids),
            )

    flush_part(force=True)
    env.close()

    if len(part_paths) == 1:
        dataset = torch.load(part_paths[0], map_location="cpu", weights_only=False)
    else:
        dataset = merge_dataset_dicts(
            [torch.load(path, map_location="cpu", weights_only=False) for path in part_paths]
        )
    valid_transition_count = (
        int(stack_time_key(dataset, "trace_valid_masks").bool().sum())
        if dataset.get("trace_valid_masks") is not None
        else len(dataset["states"]) * int(dataset["num_envs"])
    )
    dataset["metadata"].update(
        {
            "actual_num_transitions": valid_transition_count,
            "rectangular_storage_rows": len(dataset["states"]) * int(dataset["num_envs"]),
            "collector_counts": {
                COLLECTOR_ID_TO_NAME[idx]: int(count)
                for idx, count in enumerate(collector_counts.tolist())
                if idx in COLLECTOR_ID_TO_NAME
            },
            "command_mode_counts": {
                modes[idx].name: int(count)
                for idx, count in enumerate(mode_counts.tolist())
            },
            "mean_reward": float(np.mean(completed_returns)) if completed_returns else float(returns.mean().item()),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float(lengths.float().mean().item()),
            "completed_episode_count": int(len(completed_lengths)),
            "termination_count": int(termination_count),
            "timeout_count": int(timeout_count),
            "collection_seconds": float(time.perf_counter() - start_time),
        }
    )
    if trace_start_state_ids is not None:
        dataset["trace_start_state_ids"] = trace_start_state_ids.cpu().long()
    dataset["startup_domain_table"] = startup_domain_table
    dataset["episode_domain_table"] = _finish_table_chunks(episode_domain_chunks)
    dataset["metadata"]["dr_provenance_schema"] = {
        "version": 1,
        "startup_domain_rows": int(startup_domain_table["startup_domain_id"].numel()),
        "episode_domain_rows": int(dataset["episode_domain_table"]["episode_id"].numel()),
        "transition_fields": [
            "raw_actions",
            "actions",
            "env_action_delay_steps",
            "actuator_delay_substeps",
            "observations",
            "next_observations",
            "noisy_actor_observations",
        ],
        "actions_semantics": "post_collector_and_env_interface_command_sent_to_mjlab",
        "actuator_delay_semantics": "physics_substeps_using_the_previous_policy_target",
    }
    save_dataset_dict(dataset, save_path)
    file_mb = save_path.stat().st_size / (1024 * 1024)
    print("[Go2-ExpertCoverageDataset] complete")
    print(f"  dataset: {save_path}")
    print(f"  transitions: {dataset['metadata']['actual_num_transitions']}")
    print(f"  file_size_mb: {file_mb:.2f}")
    print(f"  collector_counts: {dataset['metadata']['collector_counts']}")
    print(f"  command_mode_counts: {dataset['metadata']['command_mode_counts']}")
    print(f"  mean_reward: {dataset['metadata']['mean_reward']:.4f}")
    print(f"  mean_episode_length: {dataset['metadata']['mean_episode_length']:.2f}")
    print(f"  termination_count: {termination_count}, timeout_count: {timeout_count}")


if __name__ == "__main__":
    main()
