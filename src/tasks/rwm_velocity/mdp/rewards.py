"""Tensor-only Go2 velocity rewards for imagination rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .extractors import split_go2_state


@dataclass
class Go2RWMRewardWeights:
    track_linear_velocity: float = 5.0
    track_angular_velocity: float = 3.0
    body_orientation_l2: float = -0.75
    body_ang_vel: float = -0.03
    dof_torques_l2: float = -2.5e-5
    dof_acc_l2: float = -2.5e-7
    action_rate_l2: float = -0.05
    stand_action_rate_l2: float = -0.05
    foot_gait: float = 0.4
    stand_still: float = -1.0
    uncertainty: float = -1.0
    command_response: float = 2.0
    yaw_command_response: float = 1.0
    wrong_direction: float = -2.0
    response_shortfall: float = -6.0
    response_floor: float = 0.30
    active_command_bias: float = -0.20
    active_survival: float = 0.0
    action_saturation: float = 0.0
    action_saturation_threshold: float = 0.85
    transport_linear: float = 2.0
    transport_yaw: float = 1.0
    gait_initiation: float = 0.15
    gait_transport: float = 0.35
    # Kept as a diagnostic weight.  A persistent all-feet penalty makes
    # stationary foot tapping preferable to quiet standing, so V1.2 does not
    # optimize it directly.
    stuck_all_feet: float = 0.0


@dataclass
class Go2RWMRewardState:
    last_joint_vel: torch.Tensor
    last_action: torch.Tensor
    last_foot_pos_b: torch.Tensor
    last_foot_contact: torch.Tensor
    contact_transition_ema: torch.Tensor
    diagonal_foot_velocity_abs_ema: torch.Tensor
    diagonal_foot_velocity_sq_ema: torch.Tensor
    base_lin_vel_xy_ema: torch.Tensor
    base_yaw_vel_ema: torch.Tensor
    step_dt: float
    gait_period: float = 0.6
    gait_offsets: torch.Tensor | None = None
    reward_version: str = "v1"
    command_active_threshold: float = 0.02
    command_active_threshold_x: float = 0.03
    command_active_threshold_y: float = 0.02
    command_active_threshold_yaw: float = 0.03
    motion_gate_low: float = 0.05
    motion_gate_high: float = 0.30
    response_command_scale_floor: float = 0.05
    response_clip_min: float = -2.0
    response_clip_max: float = 2.0
    response_shortfall_max: float = 1.0
    action_cost_gate_floor: float = 0.10
    model_support_uncertainty_low: float = 0.05
    model_support_uncertainty_high: float = 0.10
    transport_ema_time_constant: float = 0.30
    weights: Go2RWMRewardWeights = field(default_factory=Go2RWMRewardWeights)

    @classmethod
    def create(
        cls,
        num_envs: int,
        action_dim: int,
        device: torch.device | str,
        step_dt: float,
        reward_version: str = "v1",
    ) -> "Go2RWMRewardState":
        return cls(
            last_joint_vel=torch.zeros(num_envs, 12, device=device),
            last_action=torch.zeros(num_envs, action_dim, device=device),
            last_foot_pos_b=torch.zeros(num_envs, 4, 3, device=device),
            last_foot_contact=torch.zeros(num_envs, 4, dtype=torch.bool, device=device),
            contact_transition_ema=torch.zeros(num_envs, device=device),
            diagonal_foot_velocity_abs_ema=torch.zeros(num_envs, device=device),
            diagonal_foot_velocity_sq_ema=torch.zeros(num_envs, device=device),
            base_lin_vel_xy_ema=torch.zeros(num_envs, 2, device=device),
            base_yaw_vel_ema=torch.zeros(num_envs, device=device),
            step_dt=step_dt,
            gait_offsets=torch.tensor([0.0, 0.5, 0.5, 0.0], device=device),
            reward_version=reward_version,
        )


def _go2_foot_positions_b(joint_pos_rel: torch.Tensor) -> torch.Tensor:
    """Return Go2 foot positions in the base frame from policy-ordered joint positions.

    The 12 joint features are ordered FL, FR, RL, RR and are relative to the
    deployment default pose.  This deliberately uses only geometry and state
    features already available to the frozen RWM; it does not add privileged
    observations to the actor or dynamics model.
    """

    q = joint_pos_rel.reshape(-1, 4, 3)
    default = q.new_tensor([0.0, 0.8, -1.5]).view(1, 1, 3)
    q = q + default
    hip, thigh, calf = q.unbind(dim=-1)

    side = q.new_tensor([1.0, -1.0, 1.0, -1.0]).view(1, 4)
    front = q.new_tensor([1.0, 1.0, -1.0, -1.0]).view(1, 4)
    hip_x = 0.1934 * front
    hip_y = 0.0465 * side
    abad_link = 0.0955 * side
    leg_length = 0.213

    sin_hip, cos_hip = torch.sin(hip), torch.cos(hip)
    sin_thigh, cos_thigh = torch.sin(thigh), torch.cos(thigh)
    sin_calf, cos_calf = torch.sin(thigh + calf), torch.cos(thigh + calf)

    x = hip_x - leg_length * (sin_thigh + sin_calf)
    link_z_before_abad = -leg_length * (cos_thigh + cos_calf)
    y = hip_y + abad_link * cos_hip - sin_hip * link_z_before_abad
    z = abad_link * sin_hip + cos_hip * link_z_before_abad
    return torch.stack((x, y, z), dim=-1)


def compute_go2_imagination_reward(
    state: torch.Tensor,
    action: torch.Tensor,
    command: torch.Tensor,
    foot_contact: torch.Tensor,
    episode_length: torch.Tensor,
    reward_state: Go2RWMRewardState,
    epistemic_uncertainty: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    parts = split_go2_state(state)
    base_lin_vel = parts["base_lin_vel"]
    base_ang_vel = parts["base_ang_vel"]
    projected_gravity = parts["projected_gravity"]
    joint_pos = parts["joint_pos"]
    joint_vel = parts["joint_vel"]
    actuator_force = parts["actuator_force"]

    lin_error = torch.sum(torch.square(command[:, :2] - base_lin_vel[:, :2]), dim=-1)
    lin_error = lin_error + 2.0 * torch.square(base_lin_vel[:, 2])
    track_linear_velocity = torch.exp(-lin_error / 0.25)

    ang_error = torch.square(command[:, 2] - base_ang_vel[:, 2])
    ang_error = ang_error + 0.05 * torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
    track_angular_velocity = torch.exp(-ang_error / 0.5)

    body_orientation_l2 = torch.sum(torch.square(projected_gravity[:, :2]), dim=-1)
    body_ang_vel = torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
    dof_torques_l2 = torch.sum(torch.square(actuator_force), dim=-1)
    joint_acc = (joint_vel - reward_state.last_joint_vel) / reward_state.step_dt
    dof_acc_l2 = torch.sum(torch.square(joint_acc), dim=-1)
    action_rate_l2 = torch.sum(torch.square(action - reward_state.last_action), dim=-1)
    saturation_threshold = float(reward_state.weights.action_saturation_threshold)
    saturation_width = max(1.0 - saturation_threshold, 1.0e-6)
    normalized_saturation_excess = torch.relu(torch.abs(action) - saturation_threshold) / saturation_width
    action_saturation = torch.mean(torch.square(normalized_saturation_excess), dim=-1)

    linear_command_norm = torch.linalg.norm(command[:, :2], dim=-1)
    yaw_command_abs = torch.abs(command[:, 2])
    if reward_state.reward_version == "v1_2_transport_curriculum":
        linear_thresholds = command.new_tensor(
            [
                reward_state.command_active_threshold_x,
                reward_state.command_active_threshold_y,
            ]
        )
        linear_axis_active = torch.abs(command[:, :2]) > linear_thresholds
        yaw_active = yaw_command_abs > reward_state.command_active_threshold_yaw
    else:
        linear_axis_active = torch.abs(command[:, :2]) > reward_state.command_active_threshold
        yaw_active = yaw_command_abs > reward_state.command_active_threshold
    linear_active = linear_axis_active.any(dim=-1)
    hierarchical_active = linear_active | yaw_active
    legacy_active = (linear_command_norm + yaw_command_abs) > 0.1
    hierarchical_versions = {
        "v1_2_transport_curriculum",
        "v2_hierarchical",
        "v2_1_bounded",
        "v2_2_model_guarded",
        "v2_3_split_action_rate",
        "v2_3_1_locomotion_guarded",
        "v2_3_2_analytic_gait_guarded",
        "v2_3_3_model_support_guarded",
    }
    active_bool = hierarchical_active if reward_state.reward_version in hierarchical_versions else legacy_active
    active = active_bool.float()
    phase = ((episode_length.float() * reward_state.step_dt) / reward_state.gait_period).unsqueeze(-1)
    offsets = reward_state.gait_offsets
    assert offsets is not None
    stance = ((phase + offsets.view(1, -1)) % 1.0) < 0.56
    foot_gait = (stance == foot_contact.bool()).float().mean(dim=-1) * active
    stand_still = torch.sum(torch.square(joint_pos), dim=-1) * (1.0 - active)

    uncertainty = epistemic_uncertainty.reshape(-1)
    model_support_guarded = reward_state.reward_version == "v2_3_3_model_support_guarded"
    support_width = max(
        float(reward_state.model_support_uncertainty_high - reward_state.model_support_uncertainty_low),
        1.0e-6,
    )
    support_x = torch.clamp(
        (float(reward_state.model_support_uncertainty_high) - uncertainty) / support_width,
        0.0,
        1.0,
    )
    model_support_gate = support_x * support_x * (3.0 - 2.0 * support_x)
    uncertainty_excess = torch.relu(
        (uncertainty - float(reward_state.model_support_uncertainty_low)) / support_width
    )
    uncertainty_barrier = torch.square(uncertainty_excess)
    uncertainty_cost = uncertainty_barrier if model_support_guarded else uncertainty

    # A learned dynamics model can hallucinate base velocity from body or joint
    # oscillation.  Validate that velocity against an independent stance-foot
    # kinematic estimate and require an actual swing/contact cycle before
    # granting positive active-command rewards.
    foot_contact_bool = foot_contact.bool()
    foot_pos_b = _go2_foot_positions_b(joint_pos)
    episode_start = episode_length == 0
    previous_foot_pos_b = torch.where(
        episode_start[:, None, None], foot_pos_b, reward_state.last_foot_pos_b
    )
    previous_foot_contact = torch.where(
        episode_start[:, None], foot_contact_bool, reward_state.last_foot_contact
    )
    foot_vel_b = (foot_pos_b - previous_foot_pos_b) / reward_state.step_dt
    contact_transition = (foot_contact_bool != previous_foot_contact).float().mean(dim=-1)
    contact_transition = torch.where(episode_start, torch.zeros_like(contact_transition), contact_transition)
    transition_alpha = min(max(reward_state.step_dt / 0.20, 0.0), 1.0)
    previous_transition_ema = torch.where(
        episode_start, torch.zeros_like(reward_state.contact_transition_ema), reward_state.contact_transition_ema
    )
    contact_transition_ema = (
        (1.0 - transition_alpha) * previous_transition_ema + transition_alpha * contact_transition
    )

    analytic_gait_guarded = reward_state.reward_version in {
        "v2_3_2_analytic_gait_guarded",
        "v2_3_3_model_support_guarded",
    }

    # Contact logits are predictions of the same frozen RWM that predicts body
    # velocity.  Using them as an independent locomotion check lets the actor
    # exploit two mutually consistent model errors.  For V2.3.2, select the two
    # geometrically lowest feet as stance candidates and measure sustained
    # diagonal vertical cycling directly from joint kinematics instead.
    if analytic_gait_guarded:
        stance_indices = torch.topk(-foot_pos_b[..., 2], k=2, dim=-1).indices
        stance_mask = torch.zeros_like(foot_pos_b[..., 2])
        stance_mask.scatter_(1, stance_indices, 1.0)
    else:
        stance_mask = foot_contact_bool.float()
    stance_count = stance_mask.sum(dim=-1).clamp_min(1.0)
    rotational_foot_vel_b = torch.cross(base_ang_vel[:, None, :].expand_as(foot_pos_b), foot_pos_b, dim=-1)
    implied_base_vel_per_foot = -(rotational_foot_vel_b + foot_vel_b)
    kinematic_base_vel = (
        implied_base_vel_per_foot * stance_mask[..., None]
    ).sum(dim=1) / stance_count[:, None]
    stance_residual = base_lin_vel[:, None, :] + rotational_foot_vel_b + foot_vel_b
    stance_residual_xy = torch.sqrt(
        (
            torch.sum(torch.square(stance_residual[..., :2]), dim=-1) * stance_mask
        ).sum(dim=-1)
        / stance_count
        + 1.0e-8
    )
    kinematic_consistency_gate = torch.exp(-torch.square(stance_residual_xy / 0.25))

    swing_fraction = (~foot_contact_bool).float().mean(dim=-1)
    has_stance = (stance_mask.sum(dim=-1) >= 1.0).float()
    has_swing = (swing_fraction >= 0.25).float()
    transition_x = torch.clamp(contact_transition_ema / 0.04, 0.0, 1.0)
    transition_gate = transition_x * transition_x * (3.0 - 2.0 * transition_x)
    gait_cycle_gate = has_stance * has_swing * transition_gate

    diagonal_foot_velocity = 0.5 * (
        foot_vel_b[:, 0, 2] + foot_vel_b[:, 3, 2]
        - foot_vel_b[:, 1, 2] - foot_vel_b[:, 2, 2]
    )
    gait_alpha = min(max(reward_state.step_dt / 0.20, 0.0), 1.0)
    previous_diag_abs = torch.where(
        episode_start,
        torch.zeros_like(reward_state.diagonal_foot_velocity_abs_ema),
        reward_state.diagonal_foot_velocity_abs_ema,
    )
    previous_diag_sq = torch.where(
        episode_start,
        torch.zeros_like(reward_state.diagonal_foot_velocity_sq_ema),
        reward_state.diagonal_foot_velocity_sq_ema,
    )
    diagonal_velocity_abs_ema = (
        (1.0 - gait_alpha) * previous_diag_abs
        + gait_alpha * torch.abs(diagonal_foot_velocity)
    )
    diagonal_velocity_sq_ema = (
        (1.0 - gait_alpha) * previous_diag_sq
        + gait_alpha * torch.square(diagonal_foot_velocity)
    )
    diagonal_activity_x = torch.clamp(
        (diagonal_velocity_abs_ema - 0.025) / 0.055, 0.0, 1.0
    )
    diagonal_activity_gate = diagonal_activity_x * diagonal_activity_x * (3.0 - 2.0 * diagonal_activity_x)
    diagonal_sustained_ratio = diagonal_velocity_abs_ema / torch.sqrt(
        diagonal_velocity_sq_ema + 1.0e-8
    )
    diagonal_sustained_x = torch.clamp(
        (diagonal_sustained_ratio - 0.45) / 0.30, 0.0, 1.0
    )
    diagonal_sustained_gate = (
        diagonal_sustained_x * diagonal_sustained_x * (3.0 - 2.0 * diagonal_sustained_x)
    )
    analytic_gait_gate = diagonal_activity_gate * diagonal_sustained_gate
    if analytic_gait_guarded:
        gait_cycle_gate = analytic_gait_gate

    command_direction_xy = command[:, :2] / linear_command_norm.clamp_min(1.0e-6)[:, None]
    kinematic_signed_speed = torch.sum(kinematic_base_vel[:, :2] * command_direction_xy, dim=-1)
    kinematic_response = kinematic_signed_speed / linear_command_norm.clamp_min(
        float(reward_state.response_command_scale_floor)
    )
    kinematic_gate_denominator = max(
        float(reward_state.motion_gate_high - reward_state.motion_gate_low), 1.0e-6
    )
    kinematic_x = torch.clamp(
        (kinematic_response - float(reward_state.motion_gate_low)) / kinematic_gate_denominator,
        0.0,
        1.0,
    )
    kinematic_progress_gate = kinematic_x * kinematic_x * (3.0 - 2.0 * kinematic_x)
    kinematic_progress_gate = torch.where(
        linear_active, kinematic_progress_gate, torch.ones_like(kinematic_progress_gate)
    )
    locomotion_validity_gate = (
        gait_cycle_gate * kinematic_consistency_gate * kinematic_progress_gate * active
    )

    linear_static_track = torch.exp(-torch.sum(torch.square(command[:, :2]), dim=-1) / 0.25)
    yaw_static_track = torch.exp(-torch.square(command[:, 2]) / 0.5)
    linear_tracking_advantage = (track_linear_velocity - linear_static_track) * linear_active.float()
    yaw_tracking_advantage = (track_angular_velocity - yaw_static_track) * yaw_active.float()

    # Dense command progress relative to the stationary solution.  Unlike the
    # later locomotion gates, this signal is available from the first non-zero
    # velocity: stationary is exactly zero, exact tracking is one, and moving
    # away from the command is negative.  The lower denominator bound only
    # prevents tiny commands from amplifying model noise.
    progress_scale_floor_sq = float(reward_state.response_command_scale_floor) ** 2
    linear_command_sq = torch.sum(torch.square(command[:, :2]), dim=-1)
    linear_error_sq = torch.sum(
        torch.square(command[:, :2] - base_lin_vel[:, :2]), dim=-1
    )
    linear_dense_progress = torch.clamp(
        (linear_command_sq - linear_error_sq)
        / linear_command_sq.clamp_min(progress_scale_floor_sq),
        min=-1.0,
        max=1.0,
    )
    yaw_command_sq = torch.square(command[:, 2])
    yaw_error_sq = torch.square(command[:, 2] - base_ang_vel[:, 2])
    yaw_dense_progress = torch.clamp(
        (yaw_command_sq - yaw_error_sq)
        / yaw_command_sq.clamp_min(progress_scale_floor_sq),
        min=-1.0,
        max=1.0,
    )
    linear_zero_velocity_quality = (
        torch.exp(
            -(
                torch.sum(torch.square(base_lin_vel[:, :2]), dim=-1)
                + 2.0 * torch.square(base_lin_vel[:, 2])
            )
            / 0.25
        )
        - 1.0
    )
    yaw_zero_velocity_quality = (
        torch.exp(
            -(
                torch.square(base_ang_vel[:, 2])
                + 0.05 * torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
            )
            / 0.5
        )
        - 1.0
    )

    # A short-horizon transport signal distinguishes persistent commanded
    # motion from alternating body oscillation.  This is additive shaping:
    # it never gates away the ordinary dense tracking gradient.
    transport_alpha = min(
        max(reward_state.step_dt / max(reward_state.transport_ema_time_constant, reward_state.step_dt), 0.0),
        1.0,
    )
    previous_lin_vel_ema = torch.where(
        episode_start[:, None],
        base_lin_vel[:, :2],
        reward_state.base_lin_vel_xy_ema,
    )
    previous_yaw_vel_ema = torch.where(
        episode_start,
        base_ang_vel[:, 2],
        reward_state.base_yaw_vel_ema,
    )
    base_lin_vel_xy_ema = (
        (1.0 - transport_alpha) * previous_lin_vel_ema
        + transport_alpha * base_lin_vel[:, :2]
    )
    base_yaw_vel_ema = (
        (1.0 - transport_alpha) * previous_yaw_vel_ema
        + transport_alpha * base_ang_vel[:, 2]
    )
    track_linear_velocity_ema = torch.exp(
        -torch.sum(torch.square(command[:, :2] - base_lin_vel_xy_ema), dim=-1) / 0.25
    )
    track_angular_velocity_ema = torch.exp(
        -(
            torch.square(command[:, 2] - base_yaw_vel_ema)
            + 0.05 * torch.sum(torch.square(base_ang_vel[:, :2]), dim=-1)
        )
        / 0.5
    )
    linear_transport = torch.clamp(
        torch.sum(base_lin_vel_xy_ema * command[:, :2], dim=-1)
        / linear_command_sq.clamp_min(progress_scale_floor_sq),
        min=-1.0,
        max=1.0,
    ) * linear_active.float()
    yaw_transport = torch.clamp(
        base_yaw_vel_ema * command[:, 2]
        / yaw_command_sq.clamp_min(progress_scale_floor_sq),
        min=-1.0,
        max=1.0,
    ) * yaw_active.float()
    active_component_count = linear_active.float() + yaw_active.float()
    positive_transport = (
        torch.clamp(linear_transport, 0.0, 1.0)
        + torch.clamp(yaw_transport, 0.0, 1.0)
    ) / active_component_count.clamp_min(1.0)
    all_feet_contact = foot_contact_bool.all(dim=-1).float()
    gait_initiation_window = torch.clamp(
        1.0 - episode_length.float() * reward_state.step_dt / 0.30,
        min=0.0,
        max=1.0,
    )
    # The 32-step imagination horizon resets every 0.64 s.  A time-window-only
    # initiation bonus would therefore be available during almost half of
    # every synthetic episode and could teach perpetual in-place foot tapping.
    # Require commanded-direction transport even during initiation: lifting
    # feet is useful only when it contributes to actual locomotion.
    gait_initiation_quality = (
        gait_cycle_gate * positive_transport * gait_initiation_window * active
    )
    gait_transport_quality = gait_cycle_gate * positive_transport * active
    stuck_all_feet = all_feet_contact * (1.0 - positive_transport) * active

    linear_axis_count = linear_axis_active.float().sum(dim=-1).clamp_min(1.0)
    if reward_state.reward_version in {
        "v2_1_bounded", "v2_2_model_guarded", "v2_3_split_action_rate",
        "v2_3_2_analytic_gait_guarded",
        "v2_3_3_model_support_guarded",
    }:
        linear_command_scale = torch.abs(command[:, :2]).clamp_min(
            float(reward_state.response_command_scale_floor)
        )
        linear_axis_response = (base_lin_vel[:, :2] * torch.sign(command[:, :2])) / linear_command_scale
        linear_axis_response = torch.clamp(
            linear_axis_response,
            min=float(reward_state.response_clip_min),
            max=float(reward_state.response_clip_max),
        )
    else:
        linear_axis_response = (base_lin_vel[:, :2] * command[:, :2]) / torch.square(
            command[:, :2]
        ).clamp_min(1e-6)
    linear_response = (
        torch.where(linear_axis_active, linear_axis_response, torch.zeros_like(linear_axis_response)).sum(dim=-1)
        / linear_axis_count
    )
    linear_response_min = torch.where(
        linear_axis_active,
        linear_axis_response,
        torch.full_like(linear_axis_response, float("inf")),
    ).amin(dim=-1)
    linear_response_min = torch.where(linear_active, linear_response_min, torch.ones_like(linear_response_min))
    if reward_state.reward_version in {
        "v2_1_bounded", "v2_2_model_guarded", "v2_3_split_action_rate",
        "v2_3_2_analytic_gait_guarded",
        "v2_3_3_model_support_guarded",
    }:
        yaw_response = (base_ang_vel[:, 2] * torch.sign(command[:, 2])) / yaw_command_abs.clamp_min(
            float(reward_state.response_command_scale_floor)
        )
        yaw_response = torch.clamp(
            yaw_response,
            min=float(reward_state.response_clip_min),
            max=float(reward_state.response_clip_max),
        )
    else:
        yaw_response = (base_ang_vel[:, 2] * command[:, 2]) / torch.square(yaw_command_abs).clamp_min(1e-6)
    inactive_response = torch.full_like(linear_response, float("inf"))
    component_response = torch.minimum(
        torch.where(linear_active, linear_response_min, inactive_response),
        torch.where(yaw_active, yaw_response, inactive_response),
    )
    component_response = torch.where(hierarchical_active, component_response, torch.ones_like(component_response))
    linear_response_quality = (
        torch.where(
            linear_axis_active,
            torch.clamp(linear_axis_response, min=0.0, max=1.0),
            torch.zeros_like(linear_axis_response),
        ).sum(dim=-1)
        / linear_axis_count
    ) * linear_active.float()
    yaw_response_quality = torch.clamp(yaw_response, min=0.0, max=1.0) * yaw_active.float()
    response_floor = float(reward_state.weights.response_floor)
    linear_response_shortfall = (
        torch.where(
            linear_axis_active,
            torch.relu(torch.full_like(linear_axis_response, response_floor) - linear_axis_response),
            torch.zeros_like(linear_axis_response),
        ).sum(dim=-1)
        / linear_axis_count
    ) * linear_active.float()
    yaw_response_shortfall = (
        torch.relu(torch.full_like(yaw_response, response_floor) - yaw_response)
        * yaw_active.float()
    )
    response_shortfall = linear_response_shortfall + yaw_response_shortfall
    if reward_state.reward_version in {
        "v2_1_bounded", "v2_2_model_guarded", "v2_3_split_action_rate",
        "v2_3_2_analytic_gait_guarded",
        "v2_3_3_model_support_guarded",
    }:
        response_shortfall = torch.clamp(
            response_shortfall,
            min=0.0,
            max=float(reward_state.response_shortfall_max),
        )
    linear_wrong_direction = (
        torch.where(
            linear_axis_active,
            torch.clamp(-linear_axis_response, min=0.0, max=1.0),
            torch.zeros_like(linear_axis_response),
        ).sum(dim=-1)
        / linear_axis_count
    ) * linear_active.float()
    wrong_direction = linear_wrong_direction + (
        torch.clamp(-yaw_response, min=0.0, max=1.0) * yaw_active.float()
    )
    gate_denominator = max(float(reward_state.motion_gate_high - reward_state.motion_gate_low), 1e-6)
    gate_x = torch.clamp((component_response - float(reward_state.motion_gate_low)) / gate_denominator, 0.0, 1.0)
    motion_gate = (gate_x * gate_x * (3.0 - 2.0 * gate_x)) * active

    terms = {
        "track_linear_velocity": track_linear_velocity,
        "track_angular_velocity": track_angular_velocity,
        "body_orientation_l2": body_orientation_l2,
        "body_ang_vel": body_ang_vel,
        "dof_torques_l2": dof_torques_l2,
        "dof_acc_l2": dof_acc_l2,
        "action_rate_l2": action_rate_l2,
        "action_saturation": action_saturation,
        "foot_gait": foot_gait,
        "stand_still": stand_still,
        "uncertainty": uncertainty,
        "model_support_gate": model_support_gate,
        "uncertainty_barrier": uncertainty_barrier,
        "linear_tracking_advantage": linear_tracking_advantage,
        "yaw_tracking_advantage": yaw_tracking_advantage,
        "linear_dense_progress": linear_dense_progress,
        "yaw_dense_progress": yaw_dense_progress,
        "linear_zero_velocity_quality": linear_zero_velocity_quality,
        "yaw_zero_velocity_quality": yaw_zero_velocity_quality,
        "track_linear_velocity_ema": track_linear_velocity_ema,
        "track_angular_velocity_ema": track_angular_velocity_ema,
        "linear_transport": linear_transport,
        "yaw_transport": yaw_transport,
        "positive_transport": positive_transport,
        "gait_initiation_quality": gait_initiation_quality,
        "gait_initiation_window": gait_initiation_window,
        "gait_transport_quality": gait_transport_quality,
        "stuck_all_feet": stuck_all_feet,
        "linear_command_response": linear_response_quality,
        "linear_component_response_min": linear_response_min,
        "yaw_command_response": yaw_response_quality,
        "command_response": linear_response_quality + yaw_response_quality,
        "linear_response_shortfall": linear_response_shortfall,
        "yaw_response_shortfall": yaw_response_shortfall,
        "response_shortfall": response_shortfall,
        "wrong_direction": wrong_direction,
        "motion_gate": motion_gate,
        "contact_transition_ema": contact_transition_ema,
        "swing_fraction": swing_fraction,
        "gait_cycle_gate": gait_cycle_gate,
        "diagonal_foot_velocity": diagonal_foot_velocity,
        "diagonal_velocity_abs_ema": diagonal_velocity_abs_ema,
        "diagonal_sustained_ratio": diagonal_sustained_ratio,
        "analytic_gait_gate": analytic_gait_gate,
        "stance_kinematic_residual": stance_residual_xy,
        "kinematic_base_speed_x": kinematic_base_vel[:, 0],
        "kinematic_base_speed_y": kinematic_base_vel[:, 1],
        "kinematic_response": kinematic_response,
        "kinematic_consistency_gate": kinematic_consistency_gate,
        "locomotion_validity_gate": locomotion_validity_gate,
    }

    weights = reward_state.weights
    common_quality = (
        weights.track_linear_velocity * track_linear_velocity
        + weights.track_angular_velocity * track_angular_velocity
        + weights.body_orientation_l2 * body_orientation_l2
        + weights.body_ang_vel * body_ang_vel
        + weights.dof_torques_l2 * dof_torques_l2
        + weights.dof_acc_l2 * dof_acc_l2
        + weights.action_rate_l2 * action_rate_l2
        + weights.action_saturation * action_saturation
        + weights.foot_gait * foot_gait
        + weights.stand_still * stand_still
        + weights.uncertainty * uncertainty_cost
    )
    if reward_state.reward_version == "v1":
        reward = common_quality * reward_state.step_dt
    elif reward_state.reward_version == "v1_2_transport_curriculum":
        # Stand and locomotion are separate tasks.  Stand keeps the original
        # zero-command objective and a stronger action-rate cost.  Active
        # commands retain dense tracking, add persistent net transport, and
        # receive only a bounded additive gait-cycle bonus.  No term is
        # multiplied by a learned motion/gait/support gate.
        regularization_quality = (
            weights.body_orientation_l2 * body_orientation_l2
            + weights.body_ang_vel * body_ang_vel
            + weights.dof_torques_l2 * dof_torques_l2
            + weights.dof_acc_l2 * dof_acc_l2
            + weights.action_rate_l2 * action_rate_l2
            + weights.action_saturation * action_saturation
            + weights.uncertainty * uncertainty_cost
        )
        stand_reward = common_quality + (
            weights.stand_action_rate_l2 - weights.action_rate_l2
        ) * action_rate_l2
        active_reward = (
            0.25 * weights.track_linear_velocity * track_linear_velocity
            + 0.75 * weights.track_linear_velocity * track_linear_velocity_ema
            + 0.25 * weights.track_angular_velocity * track_angular_velocity
            + 0.75 * weights.track_angular_velocity * track_angular_velocity_ema
            + weights.transport_linear * linear_transport
            + weights.transport_yaw * yaw_transport
            + weights.wrong_direction * wrong_direction
            + weights.gait_initiation * gait_initiation_quality
            + weights.gait_transport * gait_transport_quality
            + weights.stuck_all_feet * stuck_all_feet
            + regularization_quality
        )
        reward = torch.where(active_bool, active_reward, stand_reward) * reward_state.step_dt
    elif reward_state.reward_version == "v1_1_dense_progress":
        # Preserve the original dense stand objective.  For active commands,
        # replace only the two broad Gaussian task rewards with normalized
        # error reduction relative to standing still.  All original additive
        # safety/effort regularizers remain, while phase gait, response,
        # shortfall, wrong-direction, and multiplicative gates are absent.
        regularization_quality = (
            common_quality
            - weights.track_linear_velocity * track_linear_velocity
            - weights.track_angular_velocity * track_angular_velocity
            - weights.foot_gait * foot_gait
        )
        dense_linear_quality = torch.where(
            linear_active,
            linear_dense_progress,
            linear_zero_velocity_quality,
        )
        dense_yaw_quality = torch.where(
            yaw_active,
            yaw_dense_progress,
            yaw_zero_velocity_quality,
        )
        active_reward = (
            weights.track_linear_velocity * dense_linear_quality
            + weights.track_angular_velocity * dense_yaw_quality
            + regularization_quality
        )
        reward = torch.where(
            hierarchical_active,
            active_reward,
            common_quality,
        ) * reward_state.step_dt
    elif reward_state.reward_version in hierarchical_versions:
        stand_reward = common_quality
        if reward_state.reward_version == "v2_3_split_action_rate":
            # V2.2 deliberately relaxed action-rate cost so active commands
            # could initiate locomotion, but that same relaxation made zero-
            # command behavior jittery.  V2.3 restores the stronger cost only
            # for the stand branch and leaves active-command shaping unchanged.
            stand_reward = stand_reward + (
                weights.stand_action_rate_l2 - weights.action_rate_l2
            ) * action_rate_l2
        active_axis_count = linear_active.float() + yaw_active.float()
        locomotion_guarded = reward_state.reward_version in {
            "v2_3_1_locomotion_guarded",
            "v2_3_2_analytic_gait_guarded",
            "v2_3_3_model_support_guarded",
        }
        positive_motion_gate = locomotion_validity_gate if locomotion_guarded else torch.ones_like(motion_gate)
        if model_support_guarded:
            positive_motion_gate = positive_motion_gate * model_support_gate
        effective_motion_gate = motion_gate * positive_motion_gate
        guarded_response_shortfall = response_shortfall
        if locomotion_guarded:
            guarded_response_shortfall = guarded_response_shortfall + (
                float(weights.response_floor) * active_axis_count * (1.0 - positive_motion_gate)
            )
        gated_effort_scale = 0.10 + 0.90 * effective_motion_gate
        action_cost_gate_scale = (
            reward_state.action_cost_gate_floor
            + (1.0 - reward_state.action_cost_gate_floor) * effective_motion_gate
        )
        active_reward = (
            weights.active_command_bias * active_axis_count
            + (
                weights.active_survival * active_axis_count
                if reward_state.reward_version in {
                    "v2_1_bounded",
                    "v2_2_model_guarded",
                    "v2_3_split_action_rate",
                }
                else 0.0
            )
            + positive_motion_gate * weights.command_response * linear_response_quality
            + positive_motion_gate * weights.yaw_command_response * yaw_response_quality
            + weights.response_shortfall * guarded_response_shortfall
            + weights.wrong_direction * wrong_direction
            + positive_motion_gate * weights.track_linear_velocity * linear_tracking_advantage
            + positive_motion_gate * weights.track_angular_velocity * yaw_tracking_advantage
            # Model-trust and safety costs must remain active before locomotion is established.
            + weights.body_orientation_l2 * body_orientation_l2
            + weights.body_ang_vel * body_ang_vel
            + weights.uncertainty * uncertainty_cost
            + gated_effort_scale
            * (
                weights.dof_torques_l2 * dof_torques_l2
                + weights.dof_acc_l2 * dof_acc_l2
                + weights.foot_gait * foot_gait * (gait_cycle_gate if locomotion_guarded else 1.0)
            )
            + action_cost_gate_scale
            * weights.action_rate_l2
            * action_rate_l2
            # Saturated actions are never discounted by the motion gate.
            + weights.action_saturation * action_saturation
            + (
                weights.foot_gait * analytic_gait_gate
                if analytic_gait_guarded
                else 0.0
            )
        )
        reward = torch.where(active_bool, active_reward, stand_reward) * reward_state.step_dt
    else:
        raise ValueError(f"Unknown Go2 reward version: {reward_state.reward_version!r}")

    reward_state.last_joint_vel = joint_vel.detach()
    reward_state.last_action = action.detach()
    reward_state.last_foot_pos_b = foot_pos_b.detach()
    reward_state.last_foot_contact = foot_contact_bool.detach()
    reward_state.contact_transition_ema = contact_transition_ema.detach()
    reward_state.diagonal_foot_velocity_abs_ema = diagonal_velocity_abs_ema.detach()
    reward_state.diagonal_foot_velocity_sq_ema = diagonal_velocity_sq_ema.detach()
    reward_state.base_lin_vel_xy_ema = base_lin_vel_xy_ema.detach()
    reward_state.base_yaw_vel_ema = base_yaw_vel_ema.detach()
    return reward, terms


def bad_orientation_from_state(state: torch.Tensor, limit_angle: float = math.radians(70.0)) -> torch.Tensor:
    projected_gravity = split_go2_state(state)["projected_gravity"]
    return torch.linalg.norm(projected_gravity[:, :2], dim=-1) > math.sin(limit_angle)
