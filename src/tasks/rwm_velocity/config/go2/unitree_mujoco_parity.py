"""RWM collection task matching the local unitree_mujoco Go2 physics."""

from __future__ import annotations

from mjlab.envs import ManagerBasedRlEnvCfg

from src.assets.robots.unitree_go2.unitree_mujoco_parity import (
    get_go2_unitree_mujoco_robot_cfg,
)
from src.tasks.velocity.config.go2.env_cfgs import unitree_go2_flat_env_cfg

from .env_cfgs import (
    _configure_normal_fixstand_task,
    _make_proprioceptive_expert_observation_groups,
)


def unitree_go2_flat_unitree_mujoco_proprioceptive_expert_env_cfg(
    play: bool = False,
) -> ManagerBasedRlEnvCfg:
    """45D/48D expert task with local unitree_mujoco physics parameters."""

    cfg = unitree_go2_flat_env_cfg(play=play)
    cfg.scene.num_envs = 1 if play else 1024
    _configure_normal_fixstand_task(cfg)
    cfg.scene.entities["robot"] = get_go2_unitree_mujoco_robot_cfg()

    # Local unitree_mujoco uses MuJoCo defaults except for elliptic contacts
    # and impratio=100. Its DDS controller updates targets every 0.02 seconds.
    cfg.sim.mujoco.timestep = 0.002
    cfg.sim.mujoco.integrator = "euler"
    cfg.sim.mujoco.cone = "elliptic"
    cfg.sim.mujoco.impratio = 100.0
    cfg.sim.mujoco.solver = "newton"
    cfg.sim.mujoco.iterations = 100
    cfg.sim.mujoco.ls_iterations = 50
    cfg.decimation = 10

    _make_proprioceptive_expert_observation_groups(cfg)
    return cfg
