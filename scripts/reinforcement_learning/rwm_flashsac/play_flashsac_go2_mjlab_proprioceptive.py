"""Visualize a proprioceptive Go2 FlashSAC-RWM policy in mjlab.

This entrypoint is the place for proprioceptive-specific observation handling.
It supports both the regular 48-dim RWM task and the 45-dim expert task whose
actor observation omits base linear velocity.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac import play_flashsac_go2_mjlab as base_play
from scripts.reinforcement_learning.rwm_flashsac.agent_proprioceptive import (
    create_go2_flashsac_proprioceptive_agent,
)


def _command_start_for_obs_dim(obs_dim: int) -> int:
    if obs_dim == 45:
        return 6
    if obs_dim == 48:
        return 9
    raise ValueError(f"Expected 45-dim proprioceptive or 48-dim RWM observation, got {obs_dim}.")


def _overwrite_command_t(
    observations: torch.Tensor,
    command: tuple[float, float, float] | None,
) -> torch.Tensor:
    if command is None:
        return observations
    observations = observations.clone()
    start = _command_start_for_obs_dim(int(observations.shape[-1]))
    observations[:, start : start + 3] = torch.tensor(
        command,
        dtype=observations.dtype,
        device=observations.device,
    )
    return observations


def _policy_observation_t(obs: dict[str, torch.Tensor], command: tuple[float, float, float] | None) -> torch.Tensor:
    actor_obs = _overwrite_command_t(obs["actor"], command)
    actor_dim = int(actor_obs.shape[-1])
    if actor_dim == 48:
        return actor_obs
    if actor_dim != 45:
        raise ValueError(f"Expected 45-dim proprioceptive or 48-dim RWM actor observation, got {actor_dim}.")

    critic_obs = obs.get("critic")
    if critic_obs is not None and int(critic_obs.shape[-1]) >= 48:
        # Expert-task critic order is actor-prefix + base_lin_vel, while the
        # RWM policy checkpoint expects base_lin_vel + actor-prefix.
        base_lin_vel = critic_obs[:, -3:].to(device=actor_obs.device, dtype=actor_obs.dtype)
    else:
        # Proprioceptive actor inference discards the first three dims, so a
        # zero base velocity fallback is sufficient if no critic group exists.
        base_lin_vel = torch.zeros(actor_obs.shape[0], 3, device=actor_obs.device, dtype=actor_obs.dtype)
    return torch.cat([base_lin_vel, actor_obs], dim=-1)


class ProprioceptiveFlashSACPolicyAdapter(base_play.FlashSACPolicyAdapter):
    def __call__(self, obs: Any) -> torch.Tensor:
        command = self._current_command()
        base_play._force_fixed_command(self.env, command)
        policy_obs = _policy_observation_t(obs, command)
        obs_np = policy_obs.detach().cpu().numpy().astype(np.float32)
        obs_np = base_play.apply_policy_observation_mask_np(obs_np, self.policy_observation_mask_indices)
        actions_np = self.agent.sample_actions(
            interaction_step=0,
            prev_transition={"next_observation": obs_np},
            training=False,
        )
        actions_np = base_play.expand_policy_actions_np(
            actions_np,
            action_dim=self.full_action_dim,
            policy_action_mask_indices=self.policy_action_mask_indices,
            world_model_action_mask_indices=self.world_model_action_mask_indices,
        )
        self.step_count += 1
        return torch.from_numpy(actions_np).to(device=self.device, dtype=torch.float32)


def main() -> None:
    base_play.configure_low_thread_env()
    args = base_play._parse_args()
    resolved_viewer = base_play._resolve_viewer(args.viewer)
    if resolved_viewer == "native":
        os.environ.setdefault("MUJOCO_GL", "glfw")
    else:
        os.environ.setdefault("MUJOCO_GL", "egl")

    checkpoint_path = base_play.resolve_repo_path(args.checkpoint_path)
    config_path = Path(args.config_path).expanduser() if args.config_path else checkpoint_path / "rwm_flashsac_config.yaml"
    if not config_path.is_absolute():
        config_path = base_play.resolve_repo_path(config_path)
    cfg = base_play.load_config(config_path, overrides=args.overrides)

    device = base_play.select_device(args.device or cfg.agent.device_type)
    base_play.set_seed(args.seed)
    fixed_command = tuple(args.fixed_command) if args.fixed_command is not None else None
    command_sequence = base_play._parse_command_sequence(args.command_sequence)
    random_ranges = (
        tuple(args.random_lin_vel_x),
        tuple(args.random_lin_vel_y),
        tuple(args.random_ang_vel_z),
    )
    broken_joint_names = base_play.get_world_model_broken_joint_names(cfg, args.broken_joint_names)
    joint_strength_scales = base_play._parse_joint_strength_scales(args.joint_strength_scales)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.rl import RslRlVecEnvWrapper
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    if args.no_terminations:
        env_cfg.terminations = {}
    if args.clean:
        base_play._disable_randomization(env_cfg)
    if joint_strength_scales:
        base_play.apply_go2_pd_joint_strength_scales(env_cfg, joint_strength_scales)
    if broken_joint_names:
        base_play.apply_go2_broken_pd_joints(env_cfg, broken_joint_names)
    base_play._configure_command_ranges(
        env_cfg,
        fixed_command,
        command_sequence,
        random_ranges=random_ranges if args.random_command else None,
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    base_play._force_fixed_command(env, fixed_command)
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    critic_space = env.single_observation_space.spaces.get("critic")
    critic_dim = None if critic_space is None else int(critic_space.shape[0])
    full_action_dim = int(env.single_action_space.shape[0])
    if actor_dim not in (45, 48):
        env.close()
        raise RuntimeError(f"Expected 45-dim proprioceptive or 48-dim RWM actor observation, got {actor_dim}.")
    policy_base_observation_dim = 48

    policy_action_mask_indices, world_model_action_mask_indices = base_play.get_world_model_action_masks(
        cfg,
        full_action_dim,
    )
    policy_observation_mask_indices = base_play.get_world_model_policy_observation_mask(
        cfg,
        policy_base_observation_dim,
    )
    policy_action_dim = full_action_dim - len(policy_action_mask_indices)
    policy_observation_dim = policy_base_observation_dim - len(policy_observation_mask_indices)

    obs_space, action_space = base_play.make_vector_spaces(
        args.num_envs,
        obs_dim=policy_observation_dim,
        action_dim=policy_action_dim,
    )
    agent_cfg = base_play.make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_proprioceptive_agent(obs_space, action_space, agent_cfg)
    agent.load(str(checkpoint_path))

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=1.0)
    base_play._force_fixed_command(wrapped_env, fixed_command)
    policy = ProprioceptiveFlashSACPolicyAdapter(
        agent=agent,
        device=device,
        fixed_command=fixed_command,
        command_sequence=command_sequence,
        command_switch_steps=args.command_switch_steps,
        random_command=bool(args.random_command),
        random_ranges=random_ranges,
        random_stand_prob=args.random_stand_prob,
        env=wrapped_env,
        full_action_dim=full_action_dim,
        policy_action_mask_indices=policy_action_mask_indices,
        world_model_action_mask_indices=world_model_action_mask_indices,
        policy_observation_mask_indices=policy_observation_mask_indices,
    )

    print(f"[Go2-FlashSAC-RWM-Proprio-Play] checkpoint={checkpoint_path}")
    print(f"[Go2-FlashSAC-RWM-Proprio-Play] viewer={resolved_viewer}, device={device}, num_envs={args.num_envs}")
    print(f"[Go2-FlashSAC-RWM-Proprio-Play] task={args.task}, env_actor_dim={actor_dim}, env_critic_dim={critic_dim}")
    print(f"[Go2-FlashSAC-RWM-Proprio-Play] broken_joint_names={list(broken_joint_names)}")
    print(f"[Go2-FlashSAC-RWM-Proprio-Play] joint_strength_scales={joint_strength_scales}")
    print(
        "[Go2-FlashSAC-RWM-Proprio-Play] "
        f"full_action_dim={full_action_dim}, policy_action_dim={policy_action_dim}, "
        f"policy_action_mask_indices={list(policy_action_mask_indices)}, "
        f"world_model_action_mask_indices={list(world_model_action_mask_indices)}, "
        f"policy_observation_mask_indices={list(policy_observation_mask_indices)}"
    )
    if fixed_command is not None:
        print(f"[Go2-FlashSAC-RWM-Proprio-Play] fixed_command={fixed_command}")
    if command_sequence:
        print(
            "[Go2-FlashSAC-RWM-Proprio-Play] command_sequence="
            f"{command_sequence}, switch_steps={args.command_switch_steps}"
        )
    if args.random_command:
        print(
            "[Go2-FlashSAC-RWM-Proprio-Play] random_command=True, "
            f"switch_steps={args.command_switch_steps}, ranges={random_ranges}, "
            f"stand_prob={args.random_stand_prob}"
        )

    if resolved_viewer == "native":
        NativeMujocoViewer(wrapped_env, policy, frame_rate=args.frame_rate).run()
    else:
        ViserPlayViewer(wrapped_env, policy, frame_rate=args.frame_rate).run()
    wrapped_env.close()


if __name__ == "__main__":
    main()
