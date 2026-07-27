"""Shared lateral reward shaping for RWM and simulator transitions."""

from __future__ import annotations

import math

import numpy as np
import torch


def shifted_lateral_quality_delta_torch(
    *,
    velocity_y: torch.Tensor,
    command_y: torch.Tensor,
    active_threshold_y: float,
    command_scale_floor: float,
    signed_exp_weight: float,
    signed_exp_clip: float,
    shifted_tanh_weight: float,
    shifted_tanh_gain: float,
) -> torch.Tensor:
    active_y = torch.abs(command_y) > float(active_threshold_y)
    directed_velocity = velocity_y * torch.sign(command_y)
    half_command = (0.5 * torch.abs(command_y)).clamp_min(
        float(command_scale_floor)
    )
    shifted_response = (directed_velocity - half_command) / half_command
    signed_exp = (
        torch.sign(shifted_response)
        * torch.expm1(
            torch.clamp(
                torch.abs(shifted_response),
                max=float(signed_exp_clip),
            )
        )
        / math.expm1(1.0)
    )
    shifted_tanh = torch.tanh(float(shifted_tanh_gain) * shifted_response)
    return (
        float(signed_exp_weight) * signed_exp
        + float(shifted_tanh_weight) * shifted_tanh
    ) * active_y.float()


def shifted_lateral_quality_delta_numpy(
    *,
    velocity_y: np.ndarray,
    command_y: np.ndarray,
    active_threshold_y: float,
    command_scale_floor: float,
    signed_exp_weight: float,
    signed_exp_clip: float,
    shifted_tanh_weight: float,
    shifted_tanh_gain: float,
) -> np.ndarray:
    active_y = np.abs(command_y) > float(active_threshold_y)
    directed_velocity = velocity_y * np.sign(command_y)
    half_command = np.maximum(
        0.5 * np.abs(command_y),
        float(command_scale_floor),
    )
    shifted_response = (directed_velocity - half_command) / half_command
    signed_exp = (
        np.sign(shifted_response)
        * np.expm1(
            np.minimum(np.abs(shifted_response), float(signed_exp_clip))
        )
        / math.expm1(1.0)
    )
    shifted_tanh = np.tanh(
        float(shifted_tanh_gain) * shifted_response
    )
    return (
        float(signed_exp_weight) * signed_exp
        + float(shifted_tanh_weight) * shifted_tanh
    ) * active_y.astype(np.float32)
