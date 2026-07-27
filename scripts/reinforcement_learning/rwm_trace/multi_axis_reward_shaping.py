"""Source-aligned, bounded command-tracking reward shaping."""

from __future__ import annotations

import numpy as np
import torch


SUPPORTED_MODES = {
    "soft_gaussian_advantage",
    "corrected_half_tanh",
}


def _validate_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in SUPPORTED_MODES:
        raise ValueError(f"Unsupported aligned multi-axis reward mode: {mode!r}")
    return normalized


def aligned_multi_axis_quality_delta_torch(
    *,
    velocity: torch.Tensor,
    command: torch.Tensor,
    active_thresholds: tuple[float, float, float],
    axis_weights: tuple[float, float, float],
    command_scale_floor: float,
    tracking_stds: tuple[float, float, float],
    mode: str,
    weight: float,
    tanh_gain: float,
    overspeed_weight: float,
) -> torch.Tensor:
    normalized_mode = _validate_mode(mode)
    thresholds = command.new_tensor(active_thresholds)
    axis_weight_tensor = command.new_tensor(axis_weights)
    active = torch.abs(command) > thresholds
    if normalized_mode == "soft_gaussian_advantage":
        stds = command.new_tensor(tracking_stds)
        tracking = torch.exp(-torch.square((velocity - command) / stds))
        stationary = torch.exp(-torch.square(command / stds))
        per_axis = tracking - stationary
    else:
        response = (
            velocity
            * torch.sign(command)
            / torch.abs(command).clamp_min(float(command_scale_floor))
        )
        shifted_response = 2.0 * response - 1.0
        per_axis = torch.tanh(float(tanh_gain) * shifted_response)
        per_axis = per_axis - float(overspeed_weight) * torch.square(
            torch.relu(response - 1.0)
        )
        per_axis = torch.clamp(per_axis, min=-1.0, max=1.0)
    active_float = active.float()
    active_count = active_float.sum(dim=-1).clamp_min(1.0)
    return (
        float(weight)
        * (per_axis * active_float * axis_weight_tensor).sum(dim=-1)
        / active_count
    )


def aligned_multi_axis_quality_delta_numpy(
    *,
    velocity: np.ndarray,
    command: np.ndarray,
    active_thresholds: tuple[float, float, float],
    axis_weights: tuple[float, float, float],
    command_scale_floor: float,
    tracking_stds: tuple[float, float, float],
    mode: str,
    weight: float,
    tanh_gain: float,
    overspeed_weight: float,
) -> np.ndarray:
    normalized_mode = _validate_mode(mode)
    thresholds = np.asarray(active_thresholds, dtype=np.float32)
    axis_weight_array = np.asarray(axis_weights, dtype=np.float32)
    active = np.abs(command) > thresholds
    if normalized_mode == "soft_gaussian_advantage":
        stds = np.asarray(tracking_stds, dtype=np.float32)
        tracking = np.exp(-np.square((velocity - command) / stds))
        stationary = np.exp(-np.square(command / stds))
        per_axis = tracking - stationary
    else:
        response = (
            velocity
            * np.sign(command)
            / np.maximum(np.abs(command), float(command_scale_floor))
        )
        shifted_response = 2.0 * response - 1.0
        per_axis = np.tanh(float(tanh_gain) * shifted_response)
        per_axis = per_axis - float(overspeed_weight) * np.square(
            np.maximum(response - 1.0, 0.0)
        )
        per_axis = np.clip(per_axis, -1.0, 1.0)
    active_float = active.astype(np.float32)
    active_count = np.maximum(active_float.sum(axis=-1), 1.0)
    return (
        float(weight)
        * (per_axis * active_float * axis_weight_array).sum(axis=-1)
        / active_count
    ).astype(np.float32, copy=False)
