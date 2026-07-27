from __future__ import annotations

import numpy as np
import torch

from scripts.reinforcement_learning.rwm_trace.lateral_reward_shaping import (
    shifted_lateral_quality_delta_numpy,
    shifted_lateral_quality_delta_torch,
)
from scripts.reinforcement_learning.rwm_trace.multi_axis_reward_shaping import (
    aligned_multi_axis_quality_delta_numpy,
    aligned_multi_axis_quality_delta_torch,
)


def test_numpy_and_torch_shifted_lateral_rewards_match() -> None:
    velocity = np.array([-0.3, -0.1, 0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    command = np.array([-0.2, -0.2, 0.0, 0.2, 0.2, 0.2], dtype=np.float32)
    parameters = {
        "active_threshold_y": 0.02,
        "command_scale_floor": 0.05,
        "signed_exp_weight": 2.0,
        "signed_exp_clip": 2.0,
        "shifted_tanh_weight": 4.0,
        "shifted_tanh_gain": 2.0,
    }
    numpy_delta = shifted_lateral_quality_delta_numpy(
        velocity_y=velocity,
        command_y=command,
        **parameters,
    )
    torch_delta = shifted_lateral_quality_delta_torch(
        velocity_y=torch.from_numpy(velocity),
        command_y=torch.from_numpy(command),
        **parameters,
    )
    np.testing.assert_allclose(
        numpy_delta,
        torch_delta.numpy(),
        rtol=1.0e-6,
        atol=1.0e-6,
    )


def test_half_command_is_the_shared_zero_point() -> None:
    velocity = np.array([0.1, -0.1], dtype=np.float32)
    command = np.array([0.2, -0.2], dtype=np.float32)
    delta = shifted_lateral_quality_delta_numpy(
        velocity_y=velocity,
        command_y=command,
        active_threshold_y=0.02,
        command_scale_floor=0.05,
        signed_exp_weight=2.0,
        signed_exp_clip=2.0,
        shifted_tanh_weight=4.0,
        shifted_tanh_gain=2.0,
    )
    np.testing.assert_allclose(delta, np.zeros_like(delta), atol=1.0e-7)


def test_multi_axis_numpy_and_torch_match() -> None:
    velocity = np.array(
        [[0.5, 0.1, 0.0], [0.0, -0.3, 0.4]],
        dtype=np.float32,
    )
    command = np.array(
        [[0.5, 0.2, 0.0], [0.0, -0.2, 0.4]],
        dtype=np.float32,
    )
    common = {
        "active_thresholds": (0.03, 0.02, 0.03),
        "axis_weights": (1.0, 1.5, 1.0),
        "command_scale_floor": 0.05,
        "tracking_stds": (0.25, 0.10, 0.20),
        "weight": 1.0,
        "tanh_gain": 2.0,
        "overspeed_weight": 4.0,
    }
    for mode in ("soft_gaussian_advantage", "corrected_half_tanh"):
        numpy_delta = aligned_multi_axis_quality_delta_numpy(
            velocity=velocity,
            command=command,
            mode=mode,
            **common,
        )
        torch_delta = aligned_multi_axis_quality_delta_torch(
            velocity=torch.from_numpy(velocity),
            command=torch.from_numpy(command),
            mode=mode,
            **common,
        )
        np.testing.assert_allclose(
            numpy_delta,
            torch_delta.numpy(),
            rtol=1.0e-6,
            atol=1.0e-6,
        )


def test_corrected_tanh_peaks_near_full_command() -> None:
    command = np.array([[0.0, 0.2, 0.0]] * 4, dtype=np.float32)
    velocity = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.1, 0.0],
            [0.0, 0.2, 0.0],
            [0.0, 0.4, 0.0],
        ],
        dtype=np.float32,
    )
    delta = aligned_multi_axis_quality_delta_numpy(
        velocity=velocity,
        command=command,
        active_thresholds=(0.03, 0.02, 0.03),
        axis_weights=(1.0, 1.5, 1.0),
        command_scale_floor=0.05,
        tracking_stds=(0.25, 0.10, 0.20),
        mode="corrected_half_tanh",
        weight=1.0,
        tanh_gain=2.0,
        overspeed_weight=4.0,
    )
    assert delta[0] < 0.0
    assert abs(float(delta[1])) < 1.0e-6
    assert delta[2] > 0.9
    assert delta[3] < 0.0
