"""Helpers for collecting Go2 datasets with disabled joint PD gains."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.assets.robots.unitree_go2.go2_constants import get_go2_robot_cfg


GO2_ACTION_JOINT_NAMES: tuple[str, ...] = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)


def go2_joint_names_to_action_indices(joint_names: Sequence[str]) -> tuple[int, ...]:
    """Map Go2 joint names to mjlab action indices."""

    name_to_index = {name: idx for idx, name in enumerate(GO2_ACTION_JOINT_NAMES)}
    broken = tuple(dict.fromkeys(str(name) for name in joint_names if name))
    unknown = sorted(set(broken) - set(name_to_index))
    if unknown:
        raise ValueError(
            f"Unknown Go2 joint names for action masking: {unknown}. "
            f"Known joints: {list(GO2_ACTION_JOINT_NAMES)}"
        )
    return tuple(name_to_index[name] for name in broken)


def apply_go2_broken_pd_joints(env_cfg: Any, joint_names: Sequence[str]) -> tuple[str, ...]:
    """Replace the Go2 robot cfg with one whose selected joints have kp=kd=0."""

    broken = tuple(dict.fromkeys(str(name) for name in joint_names if name))
    if not broken:
        return ()
    if "robot" not in env_cfg.scene.entities:
        raise KeyError("Expected env_cfg.scene.entities to contain a 'robot' entity.")
    env_cfg.scene.entities["robot"] = get_go2_robot_cfg(broken_pd_joint_names=broken)
    return broken


def apply_go2_pd_joint_strength_scales(
    env_cfg: Any,
    joint_strength_scales: Mapping[str, float],
) -> dict[str, float]:
    """Replace the Go2 robot cfg with selected joint PD gains scaled by alpha."""

    scales = {
        str(name): float(scale)
        for name, scale in joint_strength_scales.items()
        if str(name)
    }
    if not scales:
        return {}
    if "robot" not in env_cfg.scene.entities:
        raise KeyError("Expected env_cfg.scene.entities to contain a 'robot' entity.")
    env_cfg.scene.entities["robot"] = get_go2_robot_cfg(joint_strength_scales=scales)
    return scales
