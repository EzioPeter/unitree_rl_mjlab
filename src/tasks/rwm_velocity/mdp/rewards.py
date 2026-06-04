"""Tensor-only Go2 velocity rewards for imagination rollouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch

from .extractors import split_go2_state


@dataclass
class Go2RWMRewardWeights:
    track_linear_velocity: float = 1.0
    track_angular_velocity: float = 1.0
    body_orientation_l2: float = -1.0
    body_ang_vel: float = -0.05
    dof_torques_l2: float = -2.5e-5
    dof_acc_l2: float = -2.5e-7
    action_rate_l2: float = -0.05
    foot_gait: float = 0.5
    stand_still: float = -1.0
    uncertainty: float = -1.0


@dataclass
class Go2RWMRewardState:
    last_joint_vel: torch.Tensor
    last_action: torch.Tensor
    step_dt: float
    gait_period: float = 0.6
    gait_offsets: torch.Tensor | None = None
    weights: Go2RWMRewardWeights = field(default_factory=Go2RWMRewardWeights)

    @classmethod
    def create(
        cls,
        num_envs: int,
        action_dim: int,
        device: torch.device | str,
        step_dt: float,
    ) -> "Go2RWMRewardState":
        return cls(
            last_joint_vel=torch.zeros(num_envs, 12, device=device),
            last_action=torch.zeros(num_envs, action_dim, device=device),
            step_dt=step_dt,
            gait_offsets=torch.tensor([0.0, 0.5, 0.5, 0.0], device=device),
        )


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

    command_norm = torch.linalg.norm(command[:, :2], dim=-1) + torch.abs(command[:, 2])
    active = (command_norm > 0.1).float()
    phase = ((episode_length.float() * reward_state.step_dt) / reward_state.gait_period).unsqueeze(-1)
    offsets = reward_state.gait_offsets
    assert offsets is not None
    stance = ((phase + offsets.view(1, -1)) % 1.0) < 0.56
    foot_gait = (stance == foot_contact.bool()).float().mean(dim=-1) * active
    stand_still = torch.sum(torch.square(joint_pos), dim=-1) * (1.0 - active)

    uncertainty = epistemic_uncertainty.reshape(-1)

    terms = {
        "track_linear_velocity": track_linear_velocity,
        "track_angular_velocity": track_angular_velocity,
        "body_orientation_l2": body_orientation_l2,
        "body_ang_vel": body_ang_vel,
        "dof_torques_l2": dof_torques_l2,
        "dof_acc_l2": dof_acc_l2,
        "action_rate_l2": action_rate_l2,
        "foot_gait": foot_gait,
        "stand_still": stand_still,
        "uncertainty": uncertainty,
    }

    weights = reward_state.weights
    reward = (
        weights.track_linear_velocity * track_linear_velocity
        + weights.track_angular_velocity * track_angular_velocity
        + weights.body_orientation_l2 * body_orientation_l2
        + weights.body_ang_vel * body_ang_vel
        + weights.dof_torques_l2 * dof_torques_l2
        + weights.dof_acc_l2 * dof_acc_l2
        + weights.action_rate_l2 * action_rate_l2
        + weights.foot_gait * foot_gait
        + weights.stand_still * stand_still
        + weights.uncertainty * uncertainty
    ) * reward_state.step_dt

    reward_state.last_joint_vel = joint_vel.detach()
    reward_state.last_action = action.detach()
    return reward, terms


def bad_orientation_from_state(state: torch.Tensor, limit_angle: float = math.radians(70.0)) -> torch.Tensor:
    projected_gravity = split_go2_state(state)["projected_gravity"]
    return torch.linalg.norm(projected_gravity[:, :2], dim=-1) > math.sin(limit_angle)
