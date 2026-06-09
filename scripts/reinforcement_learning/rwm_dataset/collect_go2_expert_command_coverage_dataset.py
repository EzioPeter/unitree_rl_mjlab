"""Collect an expert-only Go2 dataset with broad random velocity-command coverage."""

from __future__ import annotations

import argparse
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

from flash_rl.envs.mjlab import configure_mjlab_randomization
from scripts.reinforcement_learning.rwm_dataset.dataset import (
    COLLECTOR_ID_TO_NAME,
    COLLECTOR_NAME_TO_ID,
    Go2MixedDatasetBuilder,
    merge_dataset_dicts,
    parse_collector_mix,
    sample_collector_ids,
    save_dataset_dict,
)
from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    load_config as load_flashsac_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    select_device,
    set_seed,
)
from src.tasks.rwm_velocity.mdp.extractors import Go2RWMExtractor


DEFAULT_EXPERT_POLICY = "logs/model_based/go2_flat_flashsac_rwm_sacwm/2026-06-05_17-27-59/step48828"


@dataclass(frozen=True)
class CommandMode:
    name: str
    axes: tuple[str, ...]


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
    parser.add_argument("--use_domain_randomization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_push_randomization", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--use_observation_noise", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def _actor_obs_np(obs_dict: dict[str, torch.Tensor], command: torch.Tensor | None = None) -> np.ndarray:
    obs = obs_dict["actor"].detach().clone()
    if command is not None:
        obs[:, 9:12] = command.to(device=obs.device, dtype=obs.dtype)
    return obs.cpu().numpy().astype(np.float32)


def _compute_actor_obs_np(env: Any, command: torch.Tensor) -> np.ndarray:
    obs_dict = env.observation_manager.compute()
    return _actor_obs_np(obs_dict, command)


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
    action_dim: int,
    device: str,
) -> Any:
    ckpt_dir = resolve_repo_path(checkpoint_path)
    cfg_path = ckpt_dir / "rwm_flashsac_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"FlashSAC checkpoint config not found: {cfg_path}")
    cfg = load_flashsac_config(cfg_path)
    cfg.agent.device_type = device
    cfg.agent.buffer_device_type = "cpu"
    cfg.agent.buffer_max_length = 1024
    cfg.agent.buffer_min_length = 1
    cfg.agent.sample_batch_size = 32
    cfg.agent.load_optimizer = False
    cfg.agent.load_reward_normalizer = False
    if device.startswith("cpu"):
        cfg.agent.use_amp = False
    obs_space, action_space = make_vector_spaces(num_envs, obs_dim=actor_dim, action_dim=action_dim)
    agent = create_go2_flashsac_agent(obs_space, action_space, make_flashsac_config(cfg, device=device))
    agent.load(str(ckpt_dir))
    return agent


def _maybe_load_agent(
    checkpoint_path: str | None,
    *,
    num_envs: int,
    actor_dim: int,
    action_dim: int,
    device: str,
) -> Any | None:
    if checkpoint_path is None or str(checkpoint_path).strip().lower() in {"", "none", "null"}:
        return None
    return _load_expert_agent(
        str(checkpoint_path),
        num_envs=num_envs,
        actor_dim=actor_dim,
        action_dim=action_dim,
        device=device,
    )


def _policy_actions(agent: Any | None, observations: np.ndarray) -> torch.Tensor | None:
    if agent is None:
        return None
    actions_np = agent.sample_actions(
        interaction_step=0,
        prev_transition={"next_observation": observations},
        training=False,
    ).astype(np.float32)
    return torch.from_numpy(actions_np)


def _mixed_actions(
    *,
    collector_ids: torch.Tensor,
    observations: np.ndarray,
    expert_agent: Any | None,
    medium_agent: Any | None,
    action_dim: int,
    device: torch.device | str,
    action_noise_std: float,
    medium_action_noise_std: float,
    failure_action_noise_std: float,
) -> torch.Tensor:
    actions = torch.empty(int(collector_ids.numel()), action_dim, device=device).uniform_(-1.0, 1.0)

    expert_t = _policy_actions(expert_agent, observations)
    if expert_t is not None:
        expert_t = expert_t.to(device=device, dtype=torch.float32)
    medium_t = _policy_actions(medium_agent, observations)
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
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=bool(args.use_domain_randomization),
        use_push_randomization=bool(args.use_push_randomization),
        use_observation_noise=bool(args.use_observation_noise),
    )
    _configure_command_cfg(env_cfg, args)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    extractor = Go2RWMExtractor(env.unwrapped)
    dims = extractor.dims
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    action_dim = int(env.single_action_space.shape[0])
    if actor_dim != int(dims.policy_obs_dim):
        raise RuntimeError(f"Expected {dims.policy_obs_dim}-dim RWM obs, got {actor_dim}.")

    expert_agent = _maybe_load_agent(
        None if args.expert_policy_path is None else str(args.expert_policy_path),
        num_envs=int(args.num_envs),
        actor_dim=actor_dim,
        action_dim=action_dim,
        device=device,
    )
    medium_agent = _maybe_load_agent(
        None if args.medium_policy_path is None else str(args.medium_policy_path),
        num_envs=int(args.num_envs),
        actor_dim=actor_dim,
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
        "dt": float(getattr(env_cfg, "step_dt", 0.02)),
        "obs_dim": actor_dim,
        "action_dim": action_dim,
        "state_dim": int(dims.state_dim),
        "contact_dim": int(dims.contact_dim),
        "termination_dim": int(dims.termination_dim),
        "num_envs": int(args.num_envs),
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
        "env_randomization": {
            "use_domain_randomization": bool(args.use_domain_randomization),
            "use_push_randomization": bool(args.use_push_randomization),
            "use_observation_noise": bool(args.use_observation_noise),
        },
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
    _force_commands(env.unwrapped, commands)
    observations = _actor_obs_np(obs_dict, commands)

    last_actions = torch.zeros(int(args.num_envs), action_dim, device=env.device)
    episode_ids = torch.arange(int(args.num_envs), device=env.device, dtype=torch.long)
    next_episode_id = int(args.num_envs)
    timesteps = torch.zeros(int(args.num_envs), device=env.device, dtype=torch.long)
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
    print(
        f"[Go2-ExpertCoverageDataset] mix={collector_mix}, modes={metadata['command_mode_weights']}, "
        f"num_envs={args.num_envs}, target_transitions={args.num_transitions}"
    )

    for _ in tqdm.trange(total_steps, smoothing=0.1, mininterval=0.5):
        _force_commands(env.unwrapped, commands)
        observations[:, 9:12] = commands.detach().cpu().numpy().astype(np.float32)
        state = extractor.extract_state()
        command = commands.clone()
        prev_action = last_actions.clone()

        action_t = _mixed_actions(
            collector_ids=collector_ids,
            observations=observations,
            expert_agent=expert_agent,
            medium_agent=medium_agent,
            action_dim=action_dim,
            device=env.device,
            action_noise_std=float(args.action_noise_std),
            medium_action_noise_std=float(args.medium_action_noise_std),
            failure_action_noise_std=float(args.failure_action_noise_std),
        )

        obs_dict, rewards, terminateds, truncateds, _extras = env.step(action_t)
        next_state = extractor.extract_state()
        contact = extractor.extract_contact()
        termination = extractor.extract_termination()
        done = terminateds | truncateds

        collector_counts += torch.bincount(
            collector_ids.detach().cpu(),
            minlength=len(COLLECTOR_ID_TO_NAME),
        )
        mode_counts += torch.bincount(mode_ids.detach().cpu(), minlength=len(modes))

        returns += rewards
        lengths += 1
        done_ids = done.nonzero(as_tuple=False).flatten()
        resample_ids = (command_ages + 1 >= intervals).nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            completed_returns.extend(returns[done_ids].detach().cpu().tolist())
            completed_lengths.extend(lengths[done_ids].detach().cpu().float().tolist())
            termination_count += int(terminateds.sum().item())
            timeout_count += int(truncateds.sum().item())
            returns[done_ids] = 0.0
            lengths[done_ids] = 0
            new_ids = torch.arange(
                next_episode_id,
                next_episode_id + int(done_ids.numel()),
                device=env.device,
                dtype=torch.long,
            )
            next_episode_id += int(done_ids.numel())
            episode_ids[done_ids] = new_ids
            timesteps[done_ids] = 0
            resample_ids = torch.unique(torch.cat([resample_ids, done_ids]))

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
            collector_ids[resample_ids] = sample_collector_ids(collector_mix, int(resample_ids.numel()), env.device)

        not_done = ~done
        timesteps[not_done] += 1
        command_ages[not_done] += 1
        last_actions = action_t
        _force_commands(env.unwrapped, commands)
        next_obs_np = _actor_obs_np(obs_dict, commands)
        next_obs_t = torch.from_numpy(next_obs_np).to(env.device)
        obs_t = torch.from_numpy(observations).to(env.device)

        builder.add(
            obs=obs_t,
            next_obs=next_obs_t,
            state=state,
            action=action_t,
            next_state=next_state,
            contact=contact,
            termination=termination,
            command=command,
            reward=rewards,
            done=done,
            timeout=truncateds,
            prev_action=prev_action,
            episode_id=episode_ids,
            timestep=timesteps,
            collector_type=collector_ids,
        )

        observations = next_obs_np
        flush_part(force=False)

    flush_part(force=True)
    env.close()

    if len(part_paths) == 1:
        dataset = torch.load(part_paths[0], map_location="cpu", weights_only=False)
    else:
        dataset = merge_dataset_dicts(
            [torch.load(path, map_location="cpu", weights_only=False) for path in part_paths]
        )
    dataset["metadata"].update(
        {
            "actual_num_transitions": len(dataset["states"]) * int(dataset["num_envs"]),
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
            "termination_count": int(termination_count),
            "timeout_count": int(timeout_count),
            "collection_seconds": float(time.perf_counter() - start_time),
        }
    )
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
