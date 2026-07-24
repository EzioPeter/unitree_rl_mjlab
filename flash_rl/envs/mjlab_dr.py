from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from mjlab.actuator import (
    ActuatorCmd,
    BuiltinPositionActuator,
    DelayedActuator,
    DelayedActuatorCfg,
)
from mjlab.envs.mdp.dr._core import _get_entity_indices
from mjlab.managers.event_manager import RecomputeLevel, requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import sample_uniform


def _runtime_store(env: Any) -> dict[str, torch.Tensor]:
    store = getattr(env, "_friend_dr_runtime", None)
    if store is None:
        store = {}
        setattr(env, "_friend_dr_runtime", store)
    return store


def _runtime_matrix(
    env: Any,
    key: str,
    width: int,
    *,
    fill_value: float,
) -> torch.Tensor:
    store = _runtime_store(env)
    value = store.get(key)
    expected_shape = (int(env.num_envs), int(width))
    if value is None or tuple(value.shape) != expected_shape:
        value = torch.full(expected_shape, fill_value, device=env.device, dtype=torch.float32)
        store[key] = value
    return value


@requires_model_fields(
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    recompute=RecomputeLevel.set_const,
)
def friend_go2_base_mass(
    env: Any,
    env_ids: torch.Tensor | None,
    *,
    asset_cfg: SceneEntityCfg,
    added_mass_range: tuple[float, float] = (-1.0, 1.0),
) -> None:
    """Add base mass and scale its inertia as Isaac Gym recomputation does."""

    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)
    body_ids = _get_entity_indices(asset.indexing, asset_cfg, "body", False)
    default_mass = env.sim.get_default_field("body_mass")[body_ids]
    default_inertia = env.sim.get_default_field("body_inertia")[body_ids]
    shape = (len(env_ids), len(body_ids))
    mass_delta = torch.empty(shape, device=env.device).uniform_(*added_mass_range)
    realized_mass = default_mass.unsqueeze(0) + mass_delta
    if torch.any(realized_mass <= 0.0):
        raise ValueError("friend base-mass range produced a non-positive body mass")
    mass_scale = realized_mass / default_mass.unsqueeze(0)

    env_grid = env_ids[:, None]
    body_grid = body_ids[None, :]
    env.sim.model.body_mass[env_grid, body_grid] = realized_mass
    env.sim.model.body_inertia[env_grid, body_grid, :] = (
        default_inertia.unsqueeze(0) * mass_scale.unsqueeze(-1)
    )

    store = _runtime_store(env)
    table = store.get("base_mass_delta")
    if table is None or tuple(table.shape) != (int(env.num_envs), len(body_ids)):
        table = torch.zeros(int(env.num_envs), len(body_ids), device=env.device)
        store["base_mass_delta"] = table
    table[env_ids] = mass_delta


@requires_model_fields(
    "body_mass",
    "body_inertia",
    recompute=RecomputeLevel.set_const,
)
def add_go2_base_payload_mass(
    env: Any,
    env_ids: torch.Tensor | None,
    *,
    asset_cfg: SceneEntityCfg,
    payload_mass_range_kg: tuple[float, float],
    payload_position_body_m: tuple[float, float, float] = (0.0, 0.0, 0.10),
    payload_box_size_m: tuple[float, float, float] = (0.20, 0.12, 0.05),
) -> None:
    """Attach a virtual box payload at a fixed location in the base frame."""

    low, high = map(float, payload_mass_range_kg)
    if low < 0.0 or high < low:
        raise ValueError(f"Invalid payload mass range: {(low, high)}")
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)
    body_ids = _get_entity_indices(asset.indexing, asset_cfg, "body", False)
    if len(body_ids) != 1:
        raise ValueError("Payload randomization must target exactly one base body")

    payload = torch.empty(len(env_ids), 1, device=env.device).uniform_(low, high)
    env_grid = env_ids[:, None]
    body_grid = body_ids[None, :]

    # These are the already-randomized base properties because this event is
    # inserted after the regular mass and COM startup events.
    base_mass = env.sim.model.body_mass[env_grid, body_grid].clone()
    base_com = env.sim.model.body_ipos[env_grid, body_grid, :].clone()
    base_inertia = env.sim.model.body_inertia[env_grid, body_grid, :].clone()
    base_iquat = env.sim.model.body_iquat[env_grid, body_grid, :].clone()
    payload_position = torch.tensor(
        payload_position_body_m, device=env.device, dtype=base_com.dtype
    ).reshape(1, 1, 3)
    payload_size = torch.tensor(
        payload_box_size_m, device=env.device, dtype=base_com.dtype
    ).reshape(1, 1, 3)
    if torch.any(payload_size <= 0.0):
        raise ValueError(f"payload_box_size_m must be positive: {payload_box_size_m}")

    total_mass = base_mass + payload
    combined_com = (
        base_mass.unsqueeze(-1) * base_com
        + payload.unsqueeze(-1) * payload_position
    ) / total_mass.unsqueeze(-1)

    # MuJoCo stores diagonal inertia in an inertial frame oriented by the
    # wxyz body_iquat. Keep that frame and project the combined inertia onto it.
    w, x, y, z = base_iquat.unbind(dim=-1)
    rotation = torch.stack(
        (
            1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w),
            2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w),
            2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y),
        ),
        dim=-1,
    ).reshape(*base_iquat.shape[:-1], 3, 3)
    base_inertia_body = rotation @ torch.diag_embed(base_inertia) @ rotation.transpose(-1, -2)

    def parallel_axis(mass: torch.Tensor, displacement: torch.Tensor) -> torch.Tensor:
        squared_norm = torch.sum(displacement * displacement, dim=-1)
        identity = torch.eye(3, device=env.device, dtype=displacement.dtype)
        return mass.unsqueeze(-1).unsqueeze(-1) * (
            squared_norm.unsqueeze(-1).unsqueeze(-1) * identity
            - displacement.unsqueeze(-1) * displacement.unsqueeze(-2)
        )

    base_shift = base_com - combined_com
    payload_shift = payload_position - combined_com
    sx, sy, sz = payload_size.unbind(dim=-1)
    payload_center_inertia = torch.diag_embed(
        payload.unsqueeze(-1)
        * torch.stack((sy * sy + sz * sz, sx * sx + sz * sz, sx * sx + sy * sy), dim=-1)
        / 12.0
    )
    combined_inertia_body = (
        base_inertia_body
        + parallel_axis(base_mass, base_shift)
        + payload_center_inertia
        + parallel_axis(payload, payload_shift)
    )
    combined_inertia_principal_frame = (
        rotation.transpose(-1, -2) @ combined_inertia_body @ rotation
    )
    realized_inertia = torch.diagonal(combined_inertia_principal_frame, dim1=-2, dim2=-1)

    env.sim.model.body_mass[env_grid, body_grid] = total_mass
    env.sim.model.body_ipos[env_grid, body_grid, :] = combined_com
    env.sim.model.body_inertia[env_grid, body_grid, :] = realized_inertia
    _runtime_matrix(env, "payload_mass_kg", 1, fill_value=0.0)[env_ids] = payload
    _runtime_matrix(env, "payload_position_body_m", 3, fill_value=0.0)[env_ids] = (
        payload_position.reshape(1, 3)
    )


def reset_joints_by_scale(
    env: Any,
    env_ids: torch.Tensor | None,
    *,
    scale_range: tuple[float, float],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset joints by multiplying the default pose, matching the friend code."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)
    asset = env.scene[asset_cfg.name]
    default_joint_pos = asset.data.default_joint_pos
    default_joint_vel = asset.data.default_joint_vel
    soft_joint_pos_limits = asset.data.soft_joint_pos_limits
    assert default_joint_pos is not None
    assert default_joint_vel is not None
    assert soft_joint_pos_limits is not None

    joint_pos = default_joint_pos[env_ids][:, asset_cfg.joint_ids].clone()
    scales = sample_uniform(*scale_range, joint_pos.shape, env.device)
    joint_pos.mul_(scales)
    limits = soft_joint_pos_limits[env_ids][:, asset_cfg.joint_ids]
    joint_pos.clamp_(limits[..., 0], limits[..., 1])

    joint_vel = default_joint_vel[env_ids][:, asset_cfg.joint_ids].clone()
    joint_vel += sample_uniform(*velocity_range, joint_vel.shape, env.device)
    joint_ids = asset_cfg.joint_ids
    if isinstance(joint_ids, list):
        joint_ids = torch.tensor(joint_ids, device=env.device)
    asset.write_joint_state_to_sim(
        joint_pos.view(len(env_ids), -1),
        joint_vel.view(len(env_ids), -1),
        env_ids=env_ids,
        joint_ids=joint_ids,
    )

    table = _runtime_matrix(env, "joint_reset_scale", int(asset.num_joints), fill_value=1.0)
    if isinstance(asset_cfg.joint_ids, slice):
        table[env_ids] = scales
    else:
        joint_ids_t = torch.as_tensor(asset_cfg.joint_ids, device=env.device, dtype=torch.long)
        table[env_ids[:, None], joint_ids_t[None, :]] = scales


@dataclass
class _SharedDelayState:
    lags: torch.Tensor
    last_applied_lags: torch.Tensor
    physics_step: int = 0
    leader: Any = None


class FriendDelayedActuator(DelayedActuator):
    """Delayed actuator whose lag is shared by all Go2 actuator groups."""

    cfg: "FriendDelayedActuatorCfg"

    def initialize(self, mj_model, model, data, device: str) -> None:
        super().initialize(mj_model, model, data, device)
        state = getattr(self.entity, "_friend_shared_delay_state", None)
        if state is None or len(state.lags) != int(data.nworld):
            state = _SharedDelayState(
                lags=torch.zeros(int(data.nworld), dtype=torch.long, device=device),
                last_applied_lags=torch.zeros(int(data.nworld), dtype=torch.long, device=device),
            )
            setattr(self.entity, "_friend_shared_delay_state", state)
        if state.leader is None:
            state.leader = self
        self._friend_shared_delay_state = state

    @property
    def current_lags(self) -> torch.Tensor:
        return self._friend_shared_delay_state.lags

    def compute(self, cmd: ActuatorCmd) -> torch.Tensor:
        state = self._friend_shared_delay_state
        if state.leader is self:
            if state.physics_step % int(self.cfg.policy_decimation) == 0:
                state.lags.random_(int(self.cfg.delay_min_lag), int(self.cfg.delay_max_lag) + 1)
                state.last_applied_lags.copy_(state.lags)
            state.physics_step += 1

        position_target = cmd.position_target
        velocity_target = cmd.velocity_target
        effort_target = cmd.effort_target
        for target, value in (
            ("position", cmd.position_target),
            ("velocity", cmd.velocity_target),
            ("effort", cmd.effort_target),
        ):
            buffer = self._delay_buffers.get(target)
            if buffer is None:
                continue
            buffer.append(value)
            buffer.set_lags(state.lags)
            delayed = buffer.compute()
            if target == "position":
                position_target = delayed
            elif target == "velocity":
                velocity_target = delayed
            else:
                effort_target = delayed

        return self.base_actuator.compute(
            ActuatorCmd(
                position_target=position_target,
                velocity_target=velocity_target,
                effort_target=effort_target,
                pos=cmd.pos,
                vel=cmd.vel,
            )
        )

    def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        state = getattr(self, "_friend_shared_delay_state", None)
        if state is None:
            return
        idx = slice(None) if env_ids is None else env_ids
        state.lags[idx] = 0
        if env_ids is None:
            state.last_applied_lags.zero_()
            state.physics_step = 0


@dataclass(kw_only=True)
class FriendDelayedActuatorCfg(DelayedActuatorCfg):
    """Build a delay wrapper synchronized across actuator groups."""

    policy_decimation: int = 4

    def build(self, entity, target_ids: list[int], target_names: list[str]) -> FriendDelayedActuator:
        base_actuator = self.base_cfg.build(entity, target_ids, target_names)
        return FriendDelayedActuator(self, base_actuator)


def friend_dr_runtime_snapshot(
    env: Any,
    env_ids: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """Copy episode-level friend DR values currently active in the environment."""

    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)
    result = {
        key: value[env_ids].detach().clone()
        for key, value in _runtime_store(env).items()
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == int(env.num_envs)
    }

    robot = env.scene["robot"]
    delay_state = getattr(robot, "_friend_shared_delay_state", None)
    if delay_state is None:
        result["actuator_delay_substeps"] = torch.zeros(
            len(env_ids), device=env.device, dtype=torch.long
        )
    else:
        result["actuator_delay_substeps"] = delay_state.last_applied_lags[env_ids].detach().clone()
    return result


@requires_model_fields(
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
)
def friend_go2_actuator_parameters(
    env: Any,
    env_ids: torch.Tensor | None,
    *,
    asset_cfg: SceneEntityCfg,
    stiffness_scale_range: tuple[float, float] = (0.9, 1.1),
    damping_scale_range: tuple[float, float] = (0.9, 1.1),
    motor_strength_range: tuple[float, float] = (0.8, 1.2),
    motor_zero_offset_range: tuple[float, float] = (-0.035, 0.035),
) -> None:
    """Apply the friend's independent PD-gain and motor-strength DR.

    Scaling gains and force limits by the same positive motor-strength sample is
    equivalent to scaling the clipped PD torque, while still using MJLab's
    per-world model fields.
    """
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)

    actuator_ids = asset_cfg.actuator_ids
    if isinstance(actuator_ids, list):
        actuators = [asset.actuators[idx] for idx in actuator_ids]
    elif isinstance(actuator_ids, slice):
        actuators = list(asset.actuators[actuator_ids])
    else:
        actuators = [asset.actuators[actuator_ids]]

    default_gain = env.sim.get_default_field("actuator_gainprm")
    default_bias = env.sim.get_default_field("actuator_biasprm")
    default_force_range = env.sim.get_default_field("actuator_forcerange")

    for actuator in actuators:
        base_actuator = actuator.base_actuator if isinstance(actuator, DelayedActuator) else actuator
        if not isinstance(base_actuator, BuiltinPositionActuator):
            raise TypeError(
                "friend_go2_actuator_parameters requires BuiltinPositionActuator "
                f"(optionally delayed), got {type(base_actuator).__name__}"
            )

        ctrl_ids = torch.as_tensor(base_actuator.global_ctrl_ids, device=env.device, dtype=torch.long)
        shape = (len(env_ids), len(ctrl_ids))
        kp_scale = torch.empty(shape, device=env.device).uniform_(*stiffness_scale_range)
        kd_scale = torch.empty(shape, device=env.device).uniform_(*damping_scale_range)
        motor_strength = torch.empty(shape, device=env.device).uniform_(*motor_strength_range)
        motor_zero_offset = torch.empty(shape, device=env.device).uniform_(*motor_zero_offset_range)
        effective_kp = default_gain[ctrl_ids, 0] * kp_scale * motor_strength

        env_grid = env_ids[:, None]
        ctrl_grid = ctrl_ids[None, :]
        env.sim.model.actuator_gainprm[env_grid, ctrl_grid, 0] = effective_kp
        env.sim.model.actuator_biasprm[env_grid, ctrl_grid, 0] = (
            default_bias[ctrl_ids, 0] * motor_strength + effective_kp * motor_zero_offset
        )
        env.sim.model.actuator_biasprm[env_grid, ctrl_grid, 1] = (
            default_bias[ctrl_ids, 1] * kp_scale * motor_strength
        )
        env.sim.model.actuator_biasprm[env_grid, ctrl_grid, 2] = (
            default_bias[ctrl_ids, 2] * kd_scale * motor_strength
        )
        env.sim.model.actuator_forcerange[env_grid, ctrl_grid, :] = (
            default_force_range[ctrl_ids, :] * motor_strength.unsqueeze(-1)
        )

        width = int(default_gain.shape[0])
        _runtime_matrix(env, "kp_scale", width, fill_value=1.0)[env_grid, ctrl_grid] = kp_scale
        _runtime_matrix(env, "kd_scale", width, fill_value=1.0)[env_grid, ctrl_grid] = kd_scale
        _runtime_matrix(env, "motor_strength", width, fill_value=1.0)[env_grid, ctrl_grid] = motor_strength
        _runtime_matrix(env, "motor_zero_offset", width, fill_value=0.0)[env_grid, ctrl_grid] = motor_zero_offset


@requires_model_fields(
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
)
def randomize_go2_joint_motor_strength(
    env: Any,
    env_ids: torch.Tensor | None,
    *,
    asset_cfg: SceneEntityCfg,
    joint_names: tuple[str, ...],
    strength_range: tuple[float, float],
) -> None:
    """Apply an extra per-world motor-strength multiplier to named joints."""

    low, high = map(float, strength_range)
    if low <= 0.0 or high < low or high > 1.0:
        raise ValueError(f"Invalid joint motor-strength range: {(low, high)}")
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(env.device, dtype=torch.long)

    requested = set(map(str, joint_names))
    ctrl_by_name: dict[str, int] = {}
    for actuator in asset.actuators:
        base = actuator.base_actuator if isinstance(actuator, DelayedActuator) else actuator
        if not isinstance(base, BuiltinPositionActuator):
            continue
        for name, ctrl_id in zip(base.target_names, base.global_ctrl_ids, strict=True):
            if name in requested:
                ctrl_by_name[name] = int(ctrl_id)
    missing = sorted(requested - set(ctrl_by_name))
    if missing:
        raise ValueError(f"Could not resolve actuator control IDs for joints: {missing}")
    ctrl_ids = torch.tensor(
        [ctrl_by_name[name] for name in joint_names], device=env.device, dtype=torch.long
    )

    default_gain = env.sim.get_default_field("actuator_gainprm")
    default_bias = env.sim.get_default_field("actuator_biasprm")
    default_force = env.sim.get_default_field("actuator_forcerange")
    runtime = _runtime_store(env)
    width = int(default_gain.shape[0])
    kp_scale = runtime.get("kp_scale", torch.ones(env.num_envs, width, device=env.device))
    kd_scale = runtime.get("kd_scale", torch.ones(env.num_envs, width, device=env.device))
    motor = runtime.get("motor_strength", torch.ones(env.num_envs, width, device=env.device))
    zero = runtime.get("motor_zero_offset", torch.zeros(env.num_envs, width, device=env.device))
    gap = torch.empty(len(env_ids), len(ctrl_ids), device=env.device).uniform_(low, high)
    env_grid = env_ids[:, None]
    ctrl_grid = ctrl_ids[None, :]
    combined_motor = motor[env_grid, ctrl_grid] * gap
    effective_kp = default_gain[ctrl_ids, 0].unsqueeze(0) * kp_scale[env_grid, ctrl_grid] * combined_motor
    env.sim.model.actuator_gainprm[env_grid, ctrl_grid, 0] = effective_kp
    env.sim.model.actuator_biasprm[env_grid, ctrl_grid, 0] = (
        default_bias[ctrl_ids, 0].unsqueeze(0) * combined_motor
        + effective_kp * zero[env_grid, ctrl_grid]
    )
    env.sim.model.actuator_biasprm[env_grid, ctrl_grid, 1] = (
        default_bias[ctrl_ids, 1].unsqueeze(0) * kp_scale[env_grid, ctrl_grid] * combined_motor
    )
    env.sim.model.actuator_biasprm[env_grid, ctrl_grid, 2] = (
        default_bias[ctrl_ids, 2].unsqueeze(0) * kd_scale[env_grid, ctrl_grid] * combined_motor
    )
    env.sim.model.actuator_forcerange[env_grid, ctrl_grid, :] = (
        default_force[ctrl_ids, :].unsqueeze(0) * combined_motor.unsqueeze(-1)
    )
    table = _runtime_matrix(env, "gap_joint_motor_strength", width, fill_value=1.0)
    table[env_grid, ctrl_grid] = gap


__all__ = [
    "FriendDelayedActuator",
    "FriendDelayedActuatorCfg",
    "add_go2_base_payload_mass",
    "friend_go2_actuator_parameters",
    "friend_go2_base_mass",
    "friend_dr_runtime_snapshot",
    "randomize_go2_joint_motor_strength",
    "reset_joints_by_scale",
]
