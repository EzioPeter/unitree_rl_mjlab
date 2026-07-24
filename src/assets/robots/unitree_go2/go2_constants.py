"""Unitree Go2 constants."""

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import mujoco

from src import SRC_PATH
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.actuator import ElectricActuator, reflected_inertia
from mjlab.utils.os import update_assets
from mjlab.utils.spec_config import CollisionCfg

##
# MJCF and assets.
##

GO2_XML: Path = (
  SRC_PATH / "assets" / "robots" / "unitree_go2" / "xmls" / "go2.xml"
)
assert GO2_XML.exists()


def get_assets(meshdir: str) -> dict[str, bytes]:
  assets: dict[str, bytes] = {}
  update_assets(assets, GO2_XML.parent / "assets", meshdir)
  return assets


def get_spec() -> mujoco.MjSpec:
  spec = mujoco.MjSpec.from_file(str(GO2_XML))
  spec.assets = get_assets(spec.meshdir)
  return spec


##
# Actuator config.
##

GO2_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*hip_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*thigh_.*",
  ),
  stiffness=20.0,
  damping=1.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(
    ".*calf_.*",
  ),
  stiffness=40.0,
  damping=2.0,
  effort_limit=45,
  armature=0.02,
)

# Dedicated controller used by the normal sim-to-real baseline. These gains
# sit between the old soft locomotion controller and the much stiffer
# robot-side FixStand controller; deployment blends between them on entry.
GO2_FIXSTAND_ACTUATOR_HIP = BuiltinPositionActuatorCfg(
  target_names_expr=(".*hip_.*",),
  stiffness=30.0,
  damping=1.5,
  effort_limit=23.5,
  armature=0.01,
)
GO2_FIXSTAND_ACTUATOR_THIGH = BuiltinPositionActuatorCfg(
  target_names_expr=(".*thigh_.*",),
  stiffness=40.0,
  damping=2.0,
  effort_limit=23.5,
  armature=0.01,
)
GO2_FIXSTAND_ACTUATOR_CALF = BuiltinPositionActuatorCfg(
  target_names_expr=(".*calf_.*",),
  stiffness=50.0,
  damping=2.5,
  effort_limit=45.0,
  armature=0.02,
)

_GO2_ACTUATOR_GROUPS = (
  (
    ("FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint"),
    GO2_ACTUATOR_HIP.stiffness,
    GO2_ACTUATOR_HIP.damping,
    GO2_ACTUATOR_HIP.effort_limit,
    GO2_ACTUATOR_HIP.armature,
  ),
  (
    ("FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint"),
    GO2_ACTUATOR_THIGH.stiffness,
    GO2_ACTUATOR_THIGH.damping,
    GO2_ACTUATOR_THIGH.effort_limit,
    GO2_ACTUATOR_THIGH.armature,
  ),
  (
    ("FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint"),
    GO2_ACTUATOR_CALF.stiffness,
    GO2_ACTUATOR_CALF.damping,
    GO2_ACTUATOR_CALF.effort_limit,
    GO2_ACTUATOR_CALF.armature,
  ),
)


def _joint_names_expr(joint_names: Sequence[str]) -> tuple[str, ...]:
  return ("(" + "|".join(re.escape(name) for name in joint_names) + ")",)


def make_go2_articulation(
  broken_pd_joint_names: Sequence[str] = (),
  joint_strength_scales: Mapping[str, float] | None = None,
) -> EntityArticulationInfoCfg:
  """Create Go2 articulation info, optionally weakening selected PD joints."""

  broken = tuple(dict.fromkeys(str(name) for name in broken_pd_joint_names if name))
  strength_scales = {
    str(name): float(scale)
    for name, scale in (joint_strength_scales or {}).items()
    if str(name)
  }
  for name in broken:
    strength_scales[name] = 0.0
  if not strength_scales:
    return GO2_ARTICULATION

  known_joints = {name for group in _GO2_ACTUATOR_GROUPS for name in group[0]}
  unknown = sorted(set(strength_scales) - known_joints)
  if unknown:
    raise ValueError(
      f"Unknown Go2 joint_strength_scales joints={unknown}. "
      f"Known joints: {sorted(known_joints)}"
    )
  bad_scales = {
    name: scale for name, scale in strength_scales.items() if scale < 0.0 or scale > 1.0
  }
  if bad_scales:
    raise ValueError(f"Go2 joint strength scales must be in [0, 1], got {bad_scales}.")

  actuators: list[BuiltinPositionActuatorCfg] = []
  for joint_names, stiffness, damping, effort_limit, armature in _GO2_ACTUATOR_GROUPS:
    healthy_joints = tuple(
      name for name in joint_names if strength_scales.get(name, 1.0) == 1.0
    )
    if healthy_joints:
      actuators.append(
        BuiltinPositionActuatorCfg(
          target_names_expr=_joint_names_expr(healthy_joints),
          stiffness=stiffness,
          damping=damping,
          effort_limit=effort_limit,
          armature=armature,
        )
      )
    for joint_name in joint_names:
      if joint_name in strength_scales and strength_scales[joint_name] != 1.0:
        scale = strength_scales[joint_name]
        actuators.append(
          BuiltinPositionActuatorCfg(
            target_names_expr=(re.escape(joint_name),),
            stiffness=stiffness * scale,
            damping=damping * scale,
            effort_limit=None if effort_limit is None or scale == 0.0 else effort_limit * scale,
            armature=armature,
          )
        )

  return EntityArticulationInfoCfg(
    actuators=tuple(actuators),
    soft_joint_pos_limit_factor=GO2_ARTICULATION.soft_joint_pos_limit_factor,
  )

##
# Keyframes.
##


INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.32),
  joint_pos={
    ".*thigh_joint": 0.9,
    ".*calf_joint": -1.8,
    ".*R_hip_joint": 0.1,
    ".*L_hip_joint": -0.1,
  },
  joint_vel={".*": 0.0},
)

FIXSTAND_INIT_STATE = EntityCfg.InitialStateCfg(
  pos=(0.0, 0.0, 0.36),
  joint_pos={
    ".*hip_joint": 0.0,
    ".*thigh_joint": 0.8,
    ".*calf_joint": -1.5,
  },
  joint_vel={".*": 0.0},
)

##
# Collision config.
##

_foot_regex = "^[FR][LR]_foot_collision$"

# This disables all collisions except the feet.
# Furthermore, feet self collisions are disabled.
FEET_ONLY_COLLISION = CollisionCfg(
  geom_names_expr=(_foot_regex,),
  contype=0,
  conaffinity=1,
  condim=3,
  priority=1,
  friction=(0.6,),
  solimp=(0.9, 0.95, 0.023),
)

# This enables all collisions, excluding self collisions.
# Foot collisions are given custom condim, friction and solimp.
FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.6,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

FIXSTAND_FULL_COLLISION = CollisionCfg(
  geom_names_expr=(".*_collision",),
  condim={_foot_regex: 3, ".*_collision": 1},
  priority={_foot_regex: 1},
  friction={_foot_regex: (0.8,)},
  solimp={_foot_regex: (0.9, 0.95, 0.023)},
  contype=1,
  conaffinity=0,
)

##
# Final config.
##

GO2_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_ACTUATOR_HIP,
    GO2_ACTUATOR_THIGH,
    GO2_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)

GO2_FIXSTAND_ARTICULATION = EntityArticulationInfoCfg(
  actuators=(
    GO2_FIXSTAND_ACTUATOR_HIP,
    GO2_FIXSTAND_ACTUATOR_THIGH,
    GO2_FIXSTAND_ACTUATOR_CALF,
  ),
  soft_joint_pos_limit_factor=0.9,
)


def get_go2_robot_cfg(
  broken_pd_joint_names: Sequence[str] = (),
  joint_strength_scales: Mapping[str, float] | None = None,
) -> EntityCfg:
  """Get a fresh Go2 robot configuration instance.

  Returns a new EntityCfg instance each time to avoid mutation issues when
  the config is shared across multiple places.
  """
  return EntityCfg(
    init_state=INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=make_go2_articulation(
      broken_pd_joint_names=broken_pd_joint_names,
      joint_strength_scales=joint_strength_scales,
    ),
  )


def get_go2_fixstand_robot_cfg() -> EntityCfg:
  """Return the healthy Go2 with FixStand-aligned pose and controller gains."""

  return EntityCfg(
    init_state=FIXSTAND_INIT_STATE,
    collisions=(FIXSTAND_FULL_COLLISION,),
    spec_fn=get_spec,
    articulation=GO2_FIXSTAND_ARTICULATION,
  )

if __name__ == "__main__":
  import mujoco.viewer as viewer

  from mjlab.entity.entity import Entity

  robot = Entity(get_go2_robot_cfg())

  viewer.launch(robot.spec.compile())
