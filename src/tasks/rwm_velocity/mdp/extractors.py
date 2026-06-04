"""Go2 state/action/contact extraction for RWM training and imagination."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class Go2RWMDimensions:
    state_dim: int = 45
    action_dim: int = 12
    contact_dim: int = 4
    termination_dim: int = 1
    policy_obs_dim: int = 48


class Go2RWMExtractor:
    """Extract the RWM state from a live mjlab Go2 environment.

    State layout:
    [base_lin_vel_b(3), base_ang_vel_b(3), projected_gravity_b(3),
     joint_pos_rel(12), joint_vel(12), actuator_force(12)].
    """

    def __init__(
        self,
        env: Any,
        robot_name: str = "robot",
        command_name: str = "twist",
        foot_contact_sensor: str = "feet_ground_contact",
    ) -> None:
        self.env = env
        self.robot_name = robot_name
        self.command_name = command_name
        self.foot_contact_sensor = foot_contact_sensor

    @property
    def robot(self) -> Any:
        return self.env.scene[self.robot_name]

    @property
    def dims(self) -> Go2RWMDimensions:
        action_dim = int(self.env.action_manager.total_action_dim)
        state_dim = int(self.extract_state().shape[-1])
        contact_dim = int(self.extract_contact().shape[-1])
        return Go2RWMDimensions(
            state_dim=state_dim,
            action_dim=action_dim,
            contact_dim=contact_dim,
            termination_dim=1,
            policy_obs_dim=48,
        )

    def extract_state(self) -> torch.Tensor:
        data = self.robot.data
        joint_pos_rel = data.joint_pos - data.default_joint_pos
        return torch.cat(
            [
                data.root_link_lin_vel_b,
                data.root_link_ang_vel_b,
                data.projected_gravity_b,
                joint_pos_rel,
                data.joint_vel,
                data.actuator_force,
            ],
            dim=-1,
        ).float()

    def extract_action(self, actions: torch.Tensor) -> torch.Tensor:
        return actions.detach().float()

    def extract_contact(self) -> torch.Tensor:
        sensor = self.env.scene[self.foot_contact_sensor]
        found = sensor.data.found
        if found is None:
            return torch.zeros(self.env.num_envs, 4, device=self.env.device)
        return (found > 0).float()

    def extract_termination(self) -> torch.Tensor:
        terminated = getattr(self.env, "reset_terminated", None)
        truncated = getattr(self.env, "reset_time_outs", None)
        if terminated is None:
            terminated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        if truncated is None:
            truncated = torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        return (terminated | truncated).float().unsqueeze(-1)

    def extract_command(self) -> torch.Tensor:
        command = self.env.command_manager.get_command(self.command_name)
        if command is None:
            return torch.zeros(self.env.num_envs, 3, device=self.env.device)
        return command.float()


def split_go2_state(state: torch.Tensor) -> dict[str, torch.Tensor]:
    return {
        "base_lin_vel": state[..., 0:3],
        "base_ang_vel": state[..., 3:6],
        "projected_gravity": state[..., 6:9],
        "joint_pos": state[..., 9:21],
        "joint_vel": state[..., 21:33],
        "actuator_force": state[..., 33:45],
    }


def make_go2_policy_obs(state: torch.Tensor, command: torch.Tensor, last_action: torch.Tensor) -> torch.Tensor:
    parts = split_go2_state(state)
    return torch.cat(
        [
            parts["base_lin_vel"],
            parts["base_ang_vel"],
            parts["projected_gravity"],
            command,
            parts["joint_pos"],
            parts["joint_vel"],
            last_action,
        ],
        dim=-1,
    )
