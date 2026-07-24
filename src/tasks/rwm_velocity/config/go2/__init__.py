"""Unitree Go2 deployable-expert task registration."""

from mjlab.tasks.registry import register_mjlab_task

from .env_cfgs import unitree_go2_flat_normal_fixstand_proprioceptive_expert_env_cfg
from .rl_cfg import unitree_go2_rwm_pretrain_runner_cfg

register_mjlab_task(
    task_id="Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
    env_cfg=unitree_go2_flat_normal_fixstand_proprioceptive_expert_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_normal_fixstand_proprioceptive_expert_env_cfg(play=True),
    rl_cfg=unitree_go2_rwm_pretrain_runner_cfg(),
    runner_cls=None,
)
