"""Unitree Go2 RWM-friendly velocity environment configs.

The stock Go2 task is not modified.  This module constructs a fresh config from
the existing Go2 flat task and then narrows the observation surface to the
state fields that the imagination environment can reconstruct.
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from src.tasks.velocity import mdp as velocity_mdp
from src.assets.robots import get_go2_robot_cfg
from src.assets.robots.unitree_go2.go2_constants import get_go2_fixstand_robot_cfg
from src.tasks.rwm_velocity.mdp.commands import ModeBalancedVelocityCommandCfg


NORMAL_FIXSTAND_COMMAND_RANGES = {
    "lin_vel_x": (-0.8, 0.8),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.6, 0.6),
}


def _make_rwm_observation_groups(cfg: ManagerBasedRlEnvCfg) -> None:
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms

    # Keep only terms that can be reconstructed from the learned state and the
    # sampled command/action during imagination.
    rwm_terms = {
        "base_lin_vel": deepcopy(critic_terms["base_lin_vel"]),
        "base_ang_vel": deepcopy(actor_terms["base_ang_vel"]),
        "projected_gravity": deepcopy(actor_terms["projected_gravity"]),
        "command": deepcopy(actor_terms["command"]),
        "joint_pos": deepcopy(actor_terms["joint_pos"]),
        "joint_vel": deepcopy(actor_terms["joint_vel"]),
        "actions": deepcopy(actor_terms["actions"]),
    }

    actor_group = ObservationGroupCfg(
        terms=deepcopy(rwm_terms),
        concatenate_terms=True,
        enable_corruption=cfg.observations["actor"].enable_corruption,
        history_length=1,
    )
    critic_group = ObservationGroupCfg(
        terms=deepcopy(rwm_terms),
        concatenate_terms=True,
        enable_corruption=False,
        history_length=1,
    )
    cfg.observations = {"actor": actor_group, "critic": critic_group}


def _make_proprioceptive_expert_observation_groups(cfg: ManagerBasedRlEnvCfg) -> None:
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms

    actor_rwm_terms = {
        "base_ang_vel": deepcopy(actor_terms["base_ang_vel"]),
        "projected_gravity": deepcopy(actor_terms["projected_gravity"]),
        "command": deepcopy(actor_terms["command"]),
        "joint_pos": deepcopy(actor_terms["joint_pos"]),
        "joint_vel": deepcopy(actor_terms["joint_vel"]),
        "actions": deepcopy(actor_terms["actions"]),
    }
    critic_rwm_terms = {
        **deepcopy(actor_rwm_terms),
        "base_lin_vel": deepcopy(critic_terms["base_lin_vel"]),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_rwm_terms,
            concatenate_terms=True,
            enable_corruption=cfg.observations["actor"].enable_corruption,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_rwm_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }


def _configure_broken_rr_calf_expert_rewards(cfg: ManagerBasedRlEnvCfg) -> None:
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    cfg.commands["twist"] = ModeBalancedVelocityCommandCfg(
        entity_name=twist_cmd.entity_name,
        resampling_time_range=twist_cmd.resampling_time_range,
        heading_command=False,
        heading_control_stiffness=twist_cmd.heading_control_stiffness,
        rel_heading_envs=0.0,
        rel_standing_envs=0.0,
        init_velocity_prob=0.0,
        debug_vis=twist_cmd.debug_vis,
        ranges=ModeBalancedVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.3, 0.3),
            lin_vel_y=(-0.15, 0.15),
            ang_vel_z=(-0.3, 0.3),
            heading=None,
        ),
        xy_command_prob=0.45,
        yaw_command_prob=0.2,
        mixed_command_prob=0.3,
        stand_command_prob=0.05,
        min_lin_speed=0.05,
        min_yaw_speed=0.05,
    )

    cfg.rewards["track_linear_velocity"].weight = 5.0
    cfg.rewards["track_linear_velocity"].params["std"] = 0.3
    cfg.rewards["track_angular_velocity"].weight = 3.0
    cfg.rewards["track_angular_velocity"].params["std"] = 0.3
    cfg.rewards["yaw_drift_when_no_yaw"] = RewardTermCfg(
        func=velocity_mdp.yaw_drift_when_no_yaw,
        weight=-0.8,
        params={
            "command_name": "twist",
            "command_threshold": 0.05,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    cfg.rewards["pose"].weight = 0.15
    cfg.rewards["body_orientation_l2"].weight = -0.75
    cfg.rewards["body_ang_vel"].weight = -0.03
    cfg.rewards["angular_momentum"].weight = -0.015
    cfg.rewards["action_rate_l2"].weight = -0.02
    cfg.rewards["foot_gait"].weight = 0.4
    cfg.rewards["foot_clearance"].weight = -0.25
    cfg.rewards["foot_slip"].weight = -0.15
    cfg.rewards["stand_still"].weight = -0.1


def _configure_normal_fixstand_task(cfg: ManagerBasedRlEnvCfg) -> None:
    """Align the healthy normal task with robot FixStand and wider commands."""

    cfg.scene.entities["robot"] = get_go2_fixstand_robot_cfg()
    twist_cmd = cfg.commands["twist"]
    assert isinstance(twist_cmd, UniformVelocityCommandCfg)
    cfg.commands["twist"] = ModeBalancedVelocityCommandCfg(
        entity_name=twist_cmd.entity_name,
        resampling_time_range=twist_cmd.resampling_time_range,
        heading_command=False,
        heading_control_stiffness=twist_cmd.heading_control_stiffness,
        rel_heading_envs=0.0,
        rel_standing_envs=0.0,
        init_velocity_prob=0.0,
        debug_vis=twist_cmd.debug_vis,
        ranges=ModeBalancedVelocityCommandCfg.Ranges(
            lin_vel_x=NORMAL_FIXSTAND_COMMAND_RANGES["lin_vel_x"],
            lin_vel_y=NORMAL_FIXSTAND_COMMAND_RANGES["lin_vel_y"],
            ang_vel_z=NORMAL_FIXSTAND_COMMAND_RANGES["ang_vel_z"],
            heading=None,
        ),
        xy_command_prob=0.45,
        yaw_command_prob=0.20,
        mixed_command_prob=0.30,
        stand_command_prob=0.05,
        min_lin_speed=0.10,
        min_yaw_speed=0.10,
    )
    # The command support is intentionally broad from the first step.  Keeping
    # the stock narrow-to-wide curriculum would reintroduce the small-gait
    # behavior this normal baseline is intended to test.
    cfg.curriculum.pop("command_vel", None)
    cfg.rewards["base_height_l2"] = RewardTermCfg(
        func=velocity_mdp.base_height_l2,
        weight=-10.0,
        params={
            "target_height": 0.32,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


def unitree_go2_flat_rwm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create a Go2 flat velocity task with RWM-reconstructable observations."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 4096
    _make_rwm_observation_groups(cfg)
    return cfg


def unitree_go2_flat_proprioceptive_expert_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create the normal-strength deployable Go2 expert task.

    The actor sees 45-dimensional hardware proprioception. The critic appends
    privileged base linear velocity for a 48-dimensional asymmetric input.
    """

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 1024
    _make_proprioceptive_expert_observation_groups(cfg)
    return cfg


def unitree_go2_flat_normal_fixstand_rwm_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Healthy normal-strength RWM task aligned with robot FixStand."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 4096
    _configure_normal_fixstand_task(cfg)
    _make_rwm_observation_groups(cfg)
    return cfg


def unitree_go2_flat_normal_fixstand_proprioceptive_expert_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """45D/48D healthy expert with FixStand-aligned pose and firm gains."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 1024
    _configure_normal_fixstand_task(cfg)
    _make_proprioceptive_expert_observation_groups(cfg)
    return cfg


def unitree_go2_flat_broken_rr_calf_proprioceptive_expert_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create a Go2 flat FlashSAC expert task targeting RR calf failure.

    The actor observation is 45-dim proprioception without base linear velocity.
    The critic observation is 48-dim: the same actor prefix plus base_lin_vel.
    FlashSAC uses cfg.env.use_critic_observation_as_full_observation=true so the
    actor sees the 45-dim prefix and the critic sees all 48 dims.  Training can
    override the RR calf strength for actuator-failure curriculum stages.
    """

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 1024
    cfg.scene.entities["robot"] = get_go2_robot_cfg(
        joint_strength_scales={"RR_calf_joint": 0.0}
    )
    _make_proprioceptive_expert_observation_groups(cfg)
    _configure_broken_rr_calf_expert_rewards(cfg)
    return cfg
