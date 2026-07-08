"""Unitree Go2 RWM task registrations."""

from mjlab.tasks.registry import register_mjlab_task

from scripts.reinforcement_learning.rwm.pretrain_runner import Go2RWMPretrainRunner

from .env_cfgs import unitree_go2_flat_rwm_env_cfg
from .env_cfgs import unitree_go2_flat_broken_rr_calf_proprioceptive_expert_env_cfg
from .rl_cfg import unitree_go2_rwm_pretrain_runner_cfg


register_mjlab_task(
    task_id="Unitree-Go2-Flat-RWM-Pretrain-Ens",
    env_cfg=unitree_go2_flat_rwm_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_rwm_env_cfg(play=True),
    rl_cfg=unitree_go2_rwm_pretrain_runner_cfg(),
    runner_cls=Go2RWMPretrainRunner,
)

register_mjlab_task(
    task_id="Unitree-Go2-Flat-RWM-Imagination",
    env_cfg=unitree_go2_flat_rwm_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_rwm_env_cfg(play=True),
    rl_cfg=unitree_go2_rwm_pretrain_runner_cfg(),
    runner_cls=None,
)

register_mjlab_task(
    task_id="Unitree-Go2-Flat-Broken-RR-Calf-Proprioceptive-Expert",
    env_cfg=unitree_go2_flat_broken_rr_calf_proprioceptive_expert_env_cfg(play=False),
    play_env_cfg=unitree_go2_flat_broken_rr_calf_proprioceptive_expert_env_cfg(play=True),
    rl_cfg=unitree_go2_rwm_pretrain_runner_cfg(),
    runner_cls=None,
)
