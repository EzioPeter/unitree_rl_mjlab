"""Go2 configuration aligned with the local unitree_mujoco simulator.

The visual meshes, inertials, collision shapes, joint names, and site names
come from the existing mjlab Go2 asset so the task observation/action
interfaces remain unchanged.  Physical parameters are changed to match:

  /home/xjy/Go2/unitree_mujoco/unitree_robots/go2/go2.xml

The local simulator applies position targets through the DDS bridge with
per-joint PD gains.  MuJoCo built-in position actuators are equivalent for the
zero feed-forward torque and zero target velocity used by this policy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg

from .go2_constants import FIXSTAND_INIT_STATE, get_spec as get_base_spec


_FOOT_REGEX = "^[FR][LR]_foot_collision$"

_JOINT_GROUPS = (
    (
        ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"),
        30.0,
        1.5,
        23.7,
    ),
    (
        ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"),
        40.0,
        2.0,
        23.7,
    ),
    (
        ("FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"),
        50.0,
        2.5,
        45.43,
    ),
)


UNITREE_MUJOCO_COLLISION = CollisionCfg(
    geom_names_expr=(".*_collision",),
    condim={_FOOT_REGEX: 6, ".*_collision": 1},
    priority={_FOOT_REGEX: 1, ".*_collision": 0},
    friction={
        _FOOT_REGEX: (0.8, 0.02, 0.01),
        ".*_collision": (0.4, 0.005, 0.0001),
    },
    contype=1,
    conaffinity=1,
)


def get_unitree_mujoco_spec():
    """Return the standard named Go2 spec with local passive joint/contact values."""

    spec = get_base_spec()
    for joint_names, _, _, _ in _JOINT_GROUPS:
        for joint_name in joint_names:
            joint = spec.joint(joint_name)
            joint.damping[0] = 0.1
            joint.armature = 0.01
            joint.frictionloss = 0.2
    for geom in spec.geoms:
        if geom.name.endswith("_collision"):
            geom.margin = 0.001
    return spec


def _joint_names_expr(joint_names: Sequence[str]) -> tuple[str, ...]:
    return ("(" + "|".join(re.escape(name) for name in joint_names) + ")",)


def make_unitree_mujoco_articulation(
    broken_pd_joint_names: Sequence[str] = (),
    joint_strength_scales: Mapping[str, float] | None = None,
) -> EntityArticulationInfoCfg:
    """Build local-PD actuators while retaining the five-condition scale semantics."""

    strength_scales = {
        str(name): float(scale)
        for name, scale in (joint_strength_scales or {}).items()
        if str(name)
    }
    for name in broken_pd_joint_names:
        if name:
            strength_scales[str(name)] = 0.0

    known_joints = {name for group, _, _, _ in _JOINT_GROUPS for name in group}
    unknown = sorted(set(strength_scales) - known_joints)
    if unknown:
        raise ValueError(
            f"Unknown Go2 joint strength scale joints={unknown}; "
            f"known joints={sorted(known_joints)}"
        )
    invalid = {
        name: scale
        for name, scale in strength_scales.items()
        if scale < 0.0 or scale > 1.0
    }
    if invalid:
        raise ValueError(f"Go2 joint strength scales must be in [0, 1], got {invalid}")

    actuators: list[BuiltinPositionActuatorCfg] = []
    for joint_names, stiffness, damping, effort_limit in _JOINT_GROUPS:
        for joint_name in joint_names:
            scale = strength_scales.get(joint_name, 1.0)
            actuators.append(
                BuiltinPositionActuatorCfg(
                    target_names_expr=_joint_names_expr((joint_name,)),
                    stiffness=stiffness * scale,
                    damping=damping * scale,
                    effort_limit=(
                        None if scale == 0.0 else effort_limit * scale
                    ),
                    armature=0.01,
                    frictionloss=0.2,
                )
            )

    return EntityArticulationInfoCfg(
        actuators=tuple(actuators),
        soft_joint_pos_limit_factor=0.9,
    )


def get_go2_unitree_mujoco_robot_cfg(
    broken_pd_joint_names: Sequence[str] = (),
    joint_strength_scales: Mapping[str, float] | None = None,
) -> EntityCfg:
    """Return the Go2 entity used by the unitree_mujoco-parity task."""

    return EntityCfg(
        init_state=FIXSTAND_INIT_STATE,
        collisions=(UNITREE_MUJOCO_COLLISION,),
        spec_fn=get_unitree_mujoco_spec,
        articulation=make_unitree_mujoco_articulation(
            broken_pd_joint_names=broken_pd_joint_names,
            joint_strength_scales=joint_strength_scales,
        ),
    )
