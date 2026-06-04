"""Unitree Go2 RWM-friendly velocity environment configs.

The stock Go2 task is not modified.  This module constructs a fresh config from
the existing Go2 flat task and then narrows the observation surface to the
state fields that the imagination environment can reconstruct.
"""

from __future__ import annotations

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg

from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg


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


def unitree_go2_flat_rwm_env_cfg(play: bool = False) -> ManagerBasedRlEnvCfg:
    """Create a Go2 flat velocity task with RWM-reconstructable observations."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 4096
    _make_rwm_observation_groups(cfg)
    return cfg
