"""Visualize a Go2 FlashSAC-RWM policy in the real mjlab simulator."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

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
    select_device,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--task", default="Unitree-Go2-Flat-RWM-Pretrain-Ens")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--viewer", choices=("auto", "native", "viser"), default="auto")
    parser.add_argument("--frame_rate", type=float, default=60.0)
    parser.add_argument("--fixed_command", type=float, nargs=3, metavar=("VX", "VY", "YAW"), default=None)
    parser.add_argument(
        "--command_sequence",
        nargs="+",
        default=None,
        help='Optional command schedule, e.g. "0.3,0,0" "0.5,0,0" "0.5,0,0.3".',
    )
    parser.add_argument("--command_switch_steps", type=int, default=500)
    parser.add_argument("--random_command", action="store_true")
    parser.add_argument("--random_lin_vel_x", type=float, nargs=2, default=(0.0, 1.0), metavar=("MIN", "MAX"))
    parser.add_argument("--random_lin_vel_y", type=float, nargs=2, default=(-0.3, 0.3), metavar=("MIN", "MAX"))
    parser.add_argument("--random_ang_vel_z", type=float, nargs=2, default=(-0.5, 0.5), metavar=("MIN", "MAX"))
    parser.add_argument("--random_stand_prob", type=float, default=0.0)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_terminations", action="store_true")
    parser.add_argument("--broken_joint_names", nargs="*", default=None)
    parser.add_argument(
        "--joint_strength_scales",
        nargs="*",
        default=(),
        metavar="JOINT=SCALE",
        help="Per-joint actuator strength scales for mjlab play, e.g. RR_calf_joint=0.5.",
    )
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _disable_randomization(env_cfg: Any) -> None:
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


def _parse_command_sequence(values: list[str] | None) -> list[tuple[float, float, float]]:
    if not values:
        return []
    commands: list[tuple[float, float, float]] = []
    for value in values:
        parts = value.replace(";", ",").split(",")
        if len(parts) != 3:
            raise ValueError(f"Expected command triplet 'vx,vy,yaw', got: {value}")
        commands.append((float(parts[0]), float(parts[1]), float(parts[2])))
    return commands


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


def _configure_command_ranges(
    env_cfg: Any,
    fixed_command: tuple[float, float, float] | None,
    command_sequence: list[tuple[float, float, float]],
    random_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] | None = None,
) -> None:
    if "twist" not in env_cfg.commands:
        return
    twist_cmd = env_cfg.commands["twist"]
    commands = command_sequence or ([fixed_command] if fixed_command is not None else [])
    if random_ranges is not None:
        twist_cmd.ranges.lin_vel_x = random_ranges[0]
        twist_cmd.ranges.lin_vel_y = random_ranges[1]
        twist_cmd.ranges.ang_vel_z = random_ranges[2]
    elif commands:
        # Viser command sliders require a positive symmetric max even when the
        # actual command is fixed to zero on an axis.
        max_x = max(0.1, max(abs(cmd[0]) for cmd in commands))
        max_y = max(0.1, max(abs(cmd[1]) for cmd in commands))
        max_yaw = max(0.1, max(abs(cmd[2]) for cmd in commands))
        twist_cmd.ranges.lin_vel_x = (-max_x, max_x)
        twist_cmd.ranges.lin_vel_y = (-max_y, max_y)
        twist_cmd.ranges.ang_vel_z = (-max_yaw, max_yaw)
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


def _resolve_viewer(viewer: Literal["auto", "native", "viser"]) -> Literal["native", "viser"]:
    if viewer != "auto":
        return viewer
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    return "native" if has_display else "viser"


class FlashSACPolicyAdapter:
    def __init__(
        self,
        agent: Any,
        device: str,
        fixed_command: tuple[float, float, float] | None,
        command_sequence: list[tuple[float, float, float]],
        command_switch_steps: int,
        random_command: bool,
        random_ranges: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
        random_stand_prob: float,
        env: Any,
        full_action_dim: int,
        policy_action_mask_indices: tuple[int, ...],
        world_model_action_mask_indices: tuple[int, ...],
        policy_observation_mask_indices: tuple[int, ...],
    ):
        self.agent = agent
        self.device = device
        self.fixed_command = fixed_command
        self.command_sequence = command_sequence
        self.command_switch_steps = max(1, int(command_switch_steps))
        self.random_command = random_command
        self.random_ranges = random_ranges
        self.random_stand_prob = float(random_stand_prob)
        self.env = env
        self.full_action_dim = int(full_action_dim)
        self.policy_action_mask_indices = policy_action_mask_indices
        self.world_model_action_mask_indices = world_model_action_mask_indices
        self.policy_observation_mask_indices = policy_observation_mask_indices
        self.step_count = 0
        self.sampled_command: tuple[float, float, float] | None = None

    def _current_command(self) -> tuple[float, float, float] | None:
        if self.random_command:
            if self.sampled_command is None or self.step_count % self.command_switch_steps == 0:
                if random.random() < self.random_stand_prob:
                    self.sampled_command = (0.0, 0.0, 0.0)
                else:
                    self.sampled_command = (
                        random.uniform(*self.random_ranges[0]),
                        random.uniform(*self.random_ranges[1]),
                        random.uniform(*self.random_ranges[2]),
                    )
                print(f"[Go2-FlashSAC-RWM-Play] sampled_command={self.sampled_command}")
            return self.sampled_command
        if self.command_sequence:
            index = (self.step_count // self.command_switch_steps) % len(self.command_sequence)
            return self.command_sequence[index]
        return self.fixed_command

    def __call__(self, obs: Any) -> torch.Tensor:
        command = self._current_command()
        _force_fixed_command(self.env, command)
        actor_obs = obs["actor"]
        if command is not None:
            actor_obs = actor_obs.clone()
            actor_obs[:, 9:12] = torch.tensor(command, dtype=actor_obs.dtype, device=actor_obs.device)
        obs_np = actor_obs.detach().cpu().numpy().astype(np.float32)
        obs_np = apply_policy_observation_mask_np(obs_np, self.policy_observation_mask_indices)
        actions_np = self.agent.sample_actions(
            interaction_step=0,
            prev_transition={"next_observation": obs_np},
            training=False,
        )
        actions_np = expand_policy_actions_np(
            actions_np,
            action_dim=self.full_action_dim,
            policy_action_mask_indices=self.policy_action_mask_indices,
            world_model_action_mask_indices=self.world_model_action_mask_indices,
        )
        self.step_count += 1
        return torch.from_numpy(actions_np).to(device=self.device, dtype=torch.float32)

    def reset(self) -> None:
        self.step_count = 0
        self.sampled_command = None
        _force_fixed_command(self.env, self._current_command())


def main() -> None:
    configure_low_thread_env()
    args = _parse_args()
    resolved_viewer = _resolve_viewer(args.viewer)
    if resolved_viewer == "native":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    checkpoint_path = resolve_repo_path(args.checkpoint_path)
    config_path = Path(args.config_path).expanduser() if args.config_path else checkpoint_path / "rwm_flashsac_config.yaml"
    if not config_path.is_absolute():
        config_path = resolve_repo_path(config_path)
    cfg = load_config(config_path, overrides=args.overrides)

    device = select_device(args.device or cfg.agent.device_type)
    set_seed(args.seed)
    fixed_command = tuple(args.fixed_command) if args.fixed_command is not None else None
    command_sequence = _parse_command_sequence(args.command_sequence)
    random_ranges = (
        tuple(args.random_lin_vel_x),
        tuple(args.random_lin_vel_y),
        tuple(args.random_ang_vel_z),
    )
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
    if args.no_terminations:
        env_cfg.terminations = {}
    if args.clean:
        _disable_randomization(env_cfg)
    if joint_strength_scales:
        apply_go2_pd_joint_strength_scales(env_cfg, joint_strength_scales)
    if broken_joint_names:
        apply_go2_broken_pd_joints(env_cfg, broken_joint_names)
    _configure_command_ranges(
        env_cfg,
        fixed_command,
        command_sequence,
        random_ranges=random_ranges if args.random_command else None,
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    _force_fixed_command(env, fixed_command)
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    full_action_dim = int(env.single_action_space.shape[0])
    if actor_dim != 48:
        env.close()
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

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    _force_fixed_command(wrapped_env, fixed_command)
    policy = FlashSACPolicyAdapter(
        agent=agent,
        device=device,
        fixed_command=fixed_command,
        command_sequence=command_sequence,
        command_switch_steps=args.command_switch_steps,
        random_command=bool(args.random_command),
        random_ranges=random_ranges,
        random_stand_prob=args.random_stand_prob,
        env=wrapped_env,
        full_action_dim=full_action_dim,
        policy_action_mask_indices=policy_action_mask_indices,
        world_model_action_mask_indices=world_model_action_mask_indices,
        policy_observation_mask_indices=policy_observation_mask_indices,
    )

    print(f"[Go2-FlashSAC-RWM-Play] checkpoint={checkpoint_path}")
    print(f"[Go2-FlashSAC-RWM-Play] viewer={resolved_viewer}, device={device}, num_envs={args.num_envs}")
    print(f"[Go2-FlashSAC-RWM-Play] broken_joint_names={list(broken_joint_names)}")
    print(f"[Go2-FlashSAC-RWM-Play] joint_strength_scales={joint_strength_scales}")
    print(
        "[Go2-FlashSAC-RWM-Play] "
        f"full_action_dim={full_action_dim}, policy_action_dim={policy_action_dim}, "
        f"policy_action_mask_indices={list(policy_action_mask_indices)}, "
        f"world_model_action_mask_indices={list(world_model_action_mask_indices)}, "
        f"policy_observation_mask_indices={list(policy_observation_mask_indices)}"
    )
    if fixed_command is not None:
        print(f"[Go2-FlashSAC-RWM-Play] fixed_command={fixed_command}")
    if command_sequence:
        print(
            "[Go2-FlashSAC-RWM-Play] command_sequence="
            f"{command_sequence}, switch_steps={args.command_switch_steps}"
        )
    if args.random_command:
        print(
            "[Go2-FlashSAC-RWM-Play] random_command=True, "
            f"switch_steps={args.command_switch_steps}, ranges={random_ranges}, "
            f"stand_prob={args.random_stand_prob}"
        )

    if resolved_viewer == "native":
        NativeMujocoViewer(wrapped_env, policy, frame_rate=args.frame_rate).run()
    else:
        ViserPlayViewer(wrapped_env, policy, frame_rate=args.frame_rate).run()
    wrapped_env.close()


if __name__ == "__main__":
    main()
