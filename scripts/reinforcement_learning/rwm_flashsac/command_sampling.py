"""Command samplers shared by Go2 frozen-RWM training and tests."""

from __future__ import annotations

from typing import Protocol

import torch


class ModeBalancedCommandConfig(Protocol):
    lin_vel_x_min: float
    lin_vel_x_max: float
    lin_vel_y_min: float
    lin_vel_y_max: float
    ang_vel_z_min: float
    ang_vel_z_max: float
    command_min_abs_x: float
    command_min_abs_y: float
    command_min_abs_yaw: float
    command_mode_weight_stand: float
    command_mode_weight_pure_x: float
    command_mode_weight_pure_y: float
    command_mode_weight_pure_yaw: float
    command_mode_weight_xy: float
    command_mode_weight_x_yaw: float
    command_mode_weight_y_yaw: float
    command_mode_weight_xy_yaw: float


COMMAND_MODE_ACTIVE_AXES = (
    (False, False, False),
    (True, False, False),
    (False, True, False),
    (False, False, True),
    (True, True, False),
    (True, False, True),
    (False, True, True),
    (True, True, True),
)


def sample_mode_balanced_commands(
    count: int,
    cfg: ModeBalancedCommandConfig,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample the eight deployment command modes with dataset-aligned weights."""

    weights = torch.tensor(
        [
            cfg.command_mode_weight_stand,
            cfg.command_mode_weight_pure_x,
            cfg.command_mode_weight_pure_y,
            cfg.command_mode_weight_pure_yaw,
            cfg.command_mode_weight_xy,
            cfg.command_mode_weight_x_yaw,
            cfg.command_mode_weight_y_yaw,
            cfg.command_mode_weight_xy_yaw,
        ],
        dtype=torch.float32,
        device=device,
    )
    if bool((weights < 0.0).any()) or float(weights.sum()) <= 0.0:
        raise ValueError("Mode-balanced command weights must be non-negative with a positive sum.")
    mode_ids = torch.multinomial(weights / weights.sum(), int(count), replacement=True)
    commands = torch.zeros(int(count), 3, dtype=torch.float32, device=device)
    active_table = torch.tensor(COMMAND_MODE_ACTIVE_AXES, dtype=torch.bool, device=device)
    active = active_table[mode_ids]
    axis_ranges = (
        (float(cfg.lin_vel_x_min), float(cfg.lin_vel_x_max), float(cfg.command_min_abs_x)),
        (float(cfg.lin_vel_y_min), float(cfg.lin_vel_y_max), float(cfg.command_min_abs_y)),
        (float(cfg.ang_vel_z_min), float(cfg.ang_vel_z_max), float(cfg.command_min_abs_yaw)),
    )
    for axis, (minimum, maximum, minimum_abs) in enumerate(axis_ranges):
        axis_active = active[:, axis]
        axis_count = int(axis_active.sum())
        if axis_count == 0:
            continue
        negative_capacity = max(0.0, -minimum)
        positive_capacity = max(0.0, maximum)
        if negative_capacity < minimum_abs and positive_capacity < minimum_abs:
            raise ValueError(
                f"Command axis {axis} has no signed range above minimum magnitude {minimum_abs}."
            )
        choose_positive = torch.rand(axis_count, device=device) < 0.5
        if negative_capacity < minimum_abs:
            choose_positive[:] = True
        elif positive_capacity < minimum_abs:
            choose_positive[:] = False
        maximum_abs = torch.where(
            choose_positive,
            torch.full((axis_count,), positive_capacity, device=device),
            torch.full((axis_count,), negative_capacity, device=device),
        )
        magnitude = minimum_abs + torch.rand(axis_count, device=device) * (maximum_abs - minimum_abs)
        commands[axis_active, axis] = torch.where(choose_positive, magnitude, -magnitude)
    return commands, mode_ids
