"""Unitree Go2 deployable FlashSAC expert environment."""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg

from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg
from src.tasks.velocity import mdp as velocity_mdp
from src.assets.robots.unitree_go2.go2_constants import get_go2_fixstand_robot_cfg
from src.tasks.rwm_velocity.mdp.commands import ModeBalancedVelocityCommandCfg


NORMAL_FIXSTAND_COMMAND_RANGES = {
    "lin_vel_x": (-0.8, 0.8),
    "lin_vel_y": (-0.3, 0.3),
    "ang_vel_z": (-0.6, 0.6),
}


def _make_proprioceptive_expert_observation_groups(cfg: ManagerBasedRlEnvCfg) -> None:
    actor_terms = cfg.observations["actor"].terms
    critic_terms = cfg.observations["critic"].terms

    actor_expert_terms = {
        "base_ang_vel": deepcopy(actor_terms["base_ang_vel"]),
        "projected_gravity": deepcopy(actor_terms["projected_gravity"]),
        "command": deepcopy(actor_terms["command"]),
        "joint_pos": deepcopy(actor_terms["joint_pos"]),
        "joint_vel": deepcopy(actor_terms["joint_vel"]),
        "actions": deepcopy(actor_terms["actions"]),
    }
    critic_expert_terms = {
        **deepcopy(actor_expert_terms),
        "base_lin_vel": deepcopy(critic_terms["base_lin_vel"]),
    }

    cfg.observations = {
        "actor": ObservationGroupCfg(
            terms=actor_expert_terms,
            concatenate_terms=True,
            enable_corruption=cfg.observations["actor"].enable_corruption,
            history_length=1,
        ),
        "critic": ObservationGroupCfg(
            terms=critic_expert_terms,
            concatenate_terms=True,
            enable_corruption=False,
            history_length=1,
        ),
    }


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


def unitree_go2_flat_normal_fixstand_proprioceptive_expert_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """45D/48D healthy expert with FixStand-aligned pose and firm gains."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 1024
    _configure_normal_fixstand_task(cfg)
    _make_proprioceptive_expert_observation_groups(cfg)
    return cfg
