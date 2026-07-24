"""Command generators used by the RWM Go2 expert tasks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from mjlab.tasks.velocity.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv


class ModeBalancedVelocityCommand(UniformVelocityCommand):
    """Velocity command sampler with explicit xy/yaw/mixed/stand modes."""

    cfg: "ModeBalancedVelocityCommandCfg"

    def _sample_scalar_without_deadzone(
        self,
        count: int,
        value_range: tuple[float, float],
        min_abs: float,
    ) -> torch.Tensor:
        low, high = value_range
        values = torch.empty(count, device=self.device).uniform_(low, high)
        if min_abs <= 0.0 or count == 0:
            return values

        deadzone = torch.abs(values) < min_abs
        if not deadzone.any():
            return values

        positive_available = high >= min_abs
        negative_available = low <= -min_abs
        dead_count = int(deadzone.sum().item())
        if positive_available and negative_available:
            use_positive = torch.rand(dead_count, device=self.device) < 0.5
            max_abs = torch.where(
                use_positive,
                torch.full((dead_count,), high, device=self.device),
                torch.full((dead_count,), -low, device=self.device),
            )
            magnitude = torch.empty(dead_count, device=self.device).uniform_(0.0, 1.0)
            magnitude = min_abs + magnitude * (max_abs - min_abs)
            values[deadzone] = torch.where(use_positive, magnitude, -magnitude)
        elif positive_available:
            values[deadzone] = torch.empty(dead_count, device=self.device).uniform_(min_abs, high)
        elif negative_available:
            values[deadzone] = -torch.empty(dead_count, device=self.device).uniform_(min_abs, -low)
        return values

    def _sample_linear_xy(self, count: int) -> torch.Tensor:
        xy = torch.empty(count, 2, device=self.device)
        for _ in range(5):
            xy[:, 0].uniform_(*self.cfg.ranges.lin_vel_x)
            xy[:, 1].uniform_(*self.cfg.ranges.lin_vel_y)
            too_small = torch.norm(xy, dim=1) < self.cfg.min_lin_speed
            if not too_small.any():
                return xy
            resample_count = int(too_small.sum().item())
            xy[too_small, 0] = torch.empty(resample_count, device=self.device).uniform_(
                *self.cfg.ranges.lin_vel_x
            )
            xy[too_small, 1] = torch.empty(resample_count, device=self.device).uniform_(
                *self.cfg.ranges.lin_vel_y
            )
        return xy

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        count = len(env_ids)
        if count == 0:
            return

        probs = torch.tensor(
            [
                self.cfg.xy_command_prob,
                self.cfg.yaw_command_prob,
                self.cfg.mixed_command_prob,
                self.cfg.stand_command_prob,
            ],
            device=self.device,
            dtype=torch.float32,
        )
        probs = probs / torch.clamp(torch.sum(probs), min=1.0e-6)
        cumulative = torch.cumsum(probs, dim=0)
        mode = torch.rand(count, device=self.device)
        xy_mask = mode < cumulative[0]
        yaw_mask = (mode >= cumulative[0]) & (mode < cumulative[1])
        mixed_mask = (mode >= cumulative[1]) & (mode < cumulative[2])
        stand_mask = mode >= cumulative[2]

        self.vel_command_b[env_ids, :] = 0.0
        if xy_mask.any():
            xy_env_ids = env_ids[xy_mask]
            self.vel_command_b[xy_env_ids, :2] = self._sample_linear_xy(len(xy_env_ids))
        if yaw_mask.any():
            yaw_env_ids = env_ids[yaw_mask]
            self.vel_command_b[yaw_env_ids, 2] = self._sample_scalar_without_deadzone(
                len(yaw_env_ids),
                self.cfg.ranges.ang_vel_z,
                self.cfg.min_yaw_speed,
            )
        if mixed_mask.any():
            mixed_env_ids = env_ids[mixed_mask]
            self.vel_command_b[mixed_env_ids, :2] = self._sample_linear_xy(len(mixed_env_ids))
            self.vel_command_b[mixed_env_ids, 2] = self._sample_scalar_without_deadzone(
                len(mixed_env_ids),
                self.cfg.ranges.ang_vel_z,
                self.cfg.min_yaw_speed,
            )

        self.is_heading_env[env_ids] = False
        self.is_standing_env[env_ids] = stand_mask

        if self.cfg.init_velocity_prob <= 0.0:
            return
        r = torch.empty(count, device=self.device)
        init_vel_mask = r.uniform_(0.0, 1.0) < self.cfg.init_velocity_prob
        init_vel_env_ids = env_ids[init_vel_mask]
        if len(init_vel_env_ids) == 0:
            return
        root_pos = self.robot.data.root_link_pos_w[init_vel_env_ids]
        root_quat = self.robot.data.root_link_quat_w[init_vel_env_ids]
        lin_vel_b = self.robot.data.root_link_lin_vel_b[init_vel_env_ids]
        lin_vel_b[:, :2] = self.vel_command_b[init_vel_env_ids, :2]
        root_lin_vel_w = quat_apply(root_quat, lin_vel_b)
        root_ang_vel_b = self.robot.data.root_link_ang_vel_b[init_vel_env_ids]
        root_ang_vel_b[:, 2] = self.vel_command_b[init_vel_env_ids, 2]
        root_state = torch.cat([root_pos, root_quat, root_lin_vel_w, root_ang_vel_b], dim=-1)
        self.robot.write_root_state_to_sim(root_state, init_vel_env_ids)


@dataclass(kw_only=True)
class ModeBalancedVelocityCommandCfg(UniformVelocityCommandCfg):
    xy_command_prob: float = 0.4
    yaw_command_prob: float = 0.3
    mixed_command_prob: float = 0.2
    stand_command_prob: float = 0.1
    min_lin_speed: float = 0.05
    min_yaw_speed: float = 0.05

    def build(self, env: "ManagerBasedRlEnv") -> ModeBalancedVelocityCommand:
        return ModeBalancedVelocityCommand(self, env)
