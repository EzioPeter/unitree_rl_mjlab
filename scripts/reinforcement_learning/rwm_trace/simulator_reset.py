"""Controlled Go2 simulator reset primitives for TRACE.

There are deliberately two reset modes:

``exact_snapshot``
    Restores a simulator snapshot captured from MJLab.  This is the only mode
    allowed to claim an exact simulator reset.

``canonical_real_projection``
    Projects a 45-D real-robot RWM state into a deterministic MJLab state.  It
    is useful for real-data TRACE proposals, but is never reported as exact
    because the real log does not contain root height/yaw or simulator hidden
    state.

All proposal branches must be created from one captured snapshot.  Resetting
each vector environment independently is not a controlled counterfactual when
domain randomization is enabled.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch


SNAPSHOT_VERSION = "go2_trace_snapshot_v1"
SNAPSHOT_REQUIRED_KEYS = (
    "root_state_local",
    "joint_position",
    "joint_velocity",
    "action",
    "prev_action",
    "prev_prev_action",
    "command",
)
CONTROLLED_MODEL_FIELDS = (
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "geom_friction",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
)


def roll_pitch_from_projected_gravity(projected_gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = projected_gravity / torch.clamp(torch.linalg.norm(projected_gravity, dim=-1, keepdim=True), min=1.0e-8)
    pitch = torch.asin(torch.clamp(gravity[..., 0], -1.0, 1.0))
    roll = torch.atan2(-gravity[..., 1], -gravity[..., 2])
    return roll, pitch


def quaternion_wxyz_from_roll_pitch(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    return torch.stack([cr * cp, sr * cp, cr * sp, -sr * sp], dim=-1)


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    first_cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (scalar * first_cross + torch.cross(xyz, first_cross, dim=-1))


def _force_commands(env: Any, command: torch.Tensor, env_ids: torch.Tensor) -> None:
    term = env.command_manager.get_term("twist")
    command = command.to(device=env.device, dtype=torch.float32)
    if hasattr(term, "vel_command_b"):
        term.vel_command_b[env_ids] = command
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[env_ids] = torch.linalg.norm(command, dim=-1) < 1.0e-8
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[env_ids] = False


def capture_go2_simulator_snapshot(
    env: Any,
    env_ids: torch.Tensor,
    *,
    command: torch.Tensor | None = None,
) -> dict[str, torch.Tensor | str]:
    """Capture the reset-relevant MJLab state for selected environments."""

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    robot = env.scene["robot"]
    root_state = torch.cat([robot.data.root_link_pose_w, robot.data.root_link_vel_w], dim=-1)[env_ids].clone()
    root_state[:, :3] -= env.scene.env_origins[env_ids]
    if command is None:
        command = env.command_manager.get_command("twist")[env_ids]
    manager = env.action_manager
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "root_state_local": root_state.detach().clone(),
        "joint_position": robot.data.joint_pos[env_ids].detach().clone(),
        "joint_velocity": robot.data.joint_vel[env_ids].detach().clone(),
        "action": manager.action[env_ids].detach().clone(),
        "prev_action": manager.prev_action[env_ids].detach().clone(),
        "prev_prev_action": manager.prev_prev_action[env_ids].detach().clone(),
        "command": command.to(device=env.device, dtype=torch.float32).detach().clone(),
    }


def validate_snapshot(snapshot: Mapping[str, Any], batch_size: int | None = None) -> None:
    version = snapshot.get("snapshot_version", SNAPSHOT_VERSION)
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported TRACE snapshot version: {version!r}.")
    missing = [key for key in SNAPSHOT_REQUIRED_KEYS if snapshot.get(key) is None]
    if missing:
        raise ValueError(f"TRACE exact snapshot is missing required keys: {missing}.")
    if batch_size is not None:
        bad = {
            key: tuple(torch.as_tensor(snapshot[key]).shape)
            for key in SNAPSHOT_REQUIRED_KEYS
            if int(torch.as_tensor(snapshot[key]).shape[0]) != int(batch_size)
        }
        if bad:
            raise ValueError(f"TRACE snapshot batch mismatch, expected {batch_size}: {bad}.")


def repeat_snapshot_rows(snapshot: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat every source row contiguously for controlled proposal branches."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    validate_snapshot(snapshot)
    result: dict[str, Any] = {"snapshot_version": SNAPSHOT_VERSION}
    for key in SNAPSHOT_REQUIRED_KEYS:
        result[key] = torch.as_tensor(snapshot[key]).repeat_interleave(int(repeats), dim=0)
    return result


def synchronize_branch_domain_parameters(env: Any, branches_per_state: int) -> None:
    """Make DR parameters identical inside every contiguous branch group."""

    if branches_per_state < 1 or env.num_envs % branches_per_state:
        raise ValueError(
            f"num_envs={env.num_envs} must be divisible by branches_per_state={branches_per_state}."
        )
    leaders = torch.arange(0, env.num_envs, branches_per_state, device=env.device)
    source = leaders.repeat_interleave(branches_per_state)
    for field_name in CONTROLLED_MODEL_FIELDS:
        value = getattr(env.sim.model, field_name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == env.num_envs:
            value.copy_(value[source].clone())
    runtime = getattr(env, "_friend_dr_runtime", None)
    if isinstance(runtime, dict):
        for value in runtime.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == env.num_envs:
                value.copy_(value[source].clone())


def step_without_automatic_reset(env: Any, action: torch.Tensor):
    """MJLab policy step that preserves the physical terminal next state.

    ``ManagerBasedRlEnv.step`` resets terminated worlds before returning its
    observation.  TRACE needs the terminal transition itself and must stop the
    proposal at that boundary, so candidate collection uses this equivalent
    step with only the reset block omitted.
    """

    env.action_manager.process_action(action.to(env.device))
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
    env.metrics_manager.compute()
    env.sim.forward()
    env.command_manager.compute(dt=env.step_dt)
    if "step" in env.event_manager.available_modes:
        env.event_manager.apply(mode="step", dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return (
        env.obs_buf,
        env.reward_buf,
        env.reset_terminated,
        env.reset_time_outs,
        env.extras,
    )


def restore_go2_simulator_snapshot(
    env: Any,
    snapshot: Mapping[str, Any],
    env_ids: torch.Tensor,
) -> dict[str, Any]:
    """Restore a complete TRACE snapshot, including policy history and command."""

    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    validate_snapshot(snapshot, len(env_ids))
    robot = env.scene["robot"]
    root_state = torch.as_tensor(snapshot["root_state_local"], device=env.device, dtype=torch.float32).clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    joint_position = torch.as_tensor(snapshot["joint_position"], device=env.device, dtype=torch.float32)
    joint_velocity = torch.as_tensor(snapshot["joint_velocity"], device=env.device, dtype=torch.float32)
    action = torch.as_tensor(snapshot["action"], device=env.device, dtype=torch.float32)
    prev_action = torch.as_tensor(snapshot["prev_action"], device=env.device, dtype=torch.float32)
    prev_prev_action = torch.as_tensor(snapshot["prev_prev_action"], device=env.device, dtype=torch.float32)
    command = torch.as_tensor(snapshot["command"], device=env.device, dtype=torch.float32)

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)

    # Process the held action so action terms reconstruct their position targets.
    manager = env.action_manager
    all_actions = manager.action.clone()
    all_actions[env_ids] = action
    manager.process_action(all_actions)
    manager._action[env_ids] = action
    manager._prev_action[env_ids] = prev_action
    manager._prev_prev_action[env_ids] = prev_prev_action
    _force_commands(env, command, env_ids)

    env.episode_length_buf[env_ids] = 0
    env.scene.write_data_to_sim()
    env.sim.forward()
    env.sim.sense()
    return {
        "exact_reset": True,
        "reset_kind": "exact_snapshot",
        "snapshot_version": SNAPSHOT_VERSION,
    }


def canonical_snapshot_from_rwm_state(
    env: Any,
    states: torch.Tensor,
    commands: torch.Tensor,
    previous_actions: torch.Tensor,
    env_ids: torch.Tensor,
) -> dict[str, Any]:
    """Create a deterministic, explicitly approximate snapshot from real data."""

    if states.ndim != 2 or states.shape[-1] != 45:
        raise ValueError(f"Expected [batch, 45] RWM states, got {tuple(states.shape)}.")
    if len(states) != len(env_ids):
        raise ValueError("states and env_ids must have the same batch length.")
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    states = states.to(device=env.device, dtype=torch.float32)
    commands = commands.to(device=env.device, dtype=torch.float32)
    previous_actions = previous_actions.to(device=env.device, dtype=torch.float32)
    robot = env.scene["robot"]
    roll, pitch = roll_pitch_from_projected_gravity(states[:, 6:9])
    quaternion = quaternion_wxyz_from_roll_pitch(roll, pitch)
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :2] = 0.0
    root_state[:, 3:7] = quaternion
    root_state[:, 7:10] = _quat_apply_wxyz(quaternion, states[:, 0:3])
    root_state[:, 10:13] = _quat_apply_wxyz(quaternion, states[:, 3:6])
    snapshot = {
        "snapshot_version": SNAPSHOT_VERSION,
        "root_state_local": root_state,
        "joint_position": robot.data.default_joint_pos[env_ids] + states[:, 9:21],
        "joint_velocity": states[:, 21:33],
        "action": previous_actions,
        "prev_action": previous_actions,
        "prev_prev_action": previous_actions,
        "command": commands,
    }
    restore_go2_simulator_snapshot(env, snapshot, env_ids)
    # Re-capture the realized simulator state; all branches are cloned from this
    # one canonical state rather than independently reconstructed.
    return capture_go2_simulator_snapshot(env, env_ids, command=commands)


def reset_go2_from_rwm_state(
    env: Any,
    states: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    robot_name: str = "robot",
) -> dict[str, Any]:
    """Deprecated compatibility wrapper; never claims an exact reset."""

    del robot_name
    zeros_command = torch.zeros(len(states), 3, device=env.device)
    zeros_action = torch.zeros(len(states), env.action_manager.total_action_dim, device=env.device)
    snapshot = canonical_snapshot_from_rwm_state(env, states, zeros_command, zeros_action, env_ids)
    restore_go2_simulator_snapshot(env, snapshot, env_ids)
    return {
        "exact_reset": False,
        "reset_kind": "canonical_real_projection",
        "fixed_fields": ["root_x", "root_y", "root_z", "root_yaw"],
        "unrecoverable_fields": ["actuator_force", "simulator_hidden_state"],
        "recoverable_state_slice": [0, 33],
    }
