from __future__ import annotations

import math
from collections import deque
from dataclasses import replace
from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray


_FRIEND_FLAT_DR_COMPONENTS = frozenset(
    {"friction", "mass_com", "motor", "delay", "observation", "push", "initial_state"}
)
_CALIBRATED_FRICTION_CENTER = 0.8
_CALIBRATED_FRICTION_FULL_HALF_WIDTH = 0.2


def normalize_friend_flat_dr_components(components: Any = None) -> frozenset[str]:
    """Normalize optional friend-flat DR component selection.

    Omitting the selection preserves the original friend-flat preset and enables
    every component. Component selection is intended for diagnostic ablations.
    """
    if components is None:
        return _FRIEND_FLAT_DR_COMPONENTS
    if isinstance(components, str):
        values = components.replace(",", " ").split()
    else:
        values = [str(value) for value in components]
    normalized = frozenset(value.strip().lower() for value in values if value.strip())
    if not normalized or normalized == {"all"}:
        return _FRIEND_FLAT_DR_COMPONENTS
    unknown = sorted(normalized - _FRIEND_FLAT_DR_COMPONENTS)
    if unknown:
        raise ValueError(
            f"Unknown friend-flat DR components: {unknown}; "
            f"allowed={sorted(_FRIEND_FLAT_DR_COMPONENTS)}"
        )
    return normalized


def normalize_friend_flat_dr_scale(scale: Any = 1.0) -> float:
    value = float(scale)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError(f"friend_flat randomization_scale must be in [0, 1], got {scale!r}.")
    return value


def calibrated_flat_friction_range(scale: Any = 1.0) -> tuple[float, float]:
    """Return the real-floor-calibrated friction range centered at 0.8."""
    value = normalize_friend_flat_dr_scale(scale)
    half_width = _CALIBRATED_FRICTION_FULL_HALF_WIDTH * value
    return (
        _CALIBRATED_FRICTION_CENTER - half_width,
        _CALIBRATED_FRICTION_CENTER + half_width,
    )


def _scaled_range(low: float, high: float, center: float, scale: float) -> tuple[float, float]:
    return (center + scale * (low - center), center + scale * (high - center))


def normalize_action_mask_indices(indices: Any, action_dim: int) -> tuple[int, ...]:
    if indices is None:
        return ()
    if isinstance(indices, (int, np.integer)):
        values = (int(indices),)
    else:
        values = tuple(int(idx) for idx in indices)
    normalized = tuple(dict.fromkeys(values))
    bad = [idx for idx in normalized if idx < 0 or idx >= action_dim]
    if bad:
        raise ValueError(f"Action mask indices out of range for action_dim={action_dim}: {bad}")
    return normalized


def expand_masked_actions_t(
    actions: torch.Tensor,
    *,
    full_action_dim: int,
    action_mask_indices: tuple[int, ...],
) -> torch.Tensor:
    if not action_mask_indices:
        if actions.shape[-1] != full_action_dim:
            raise ValueError(f"Action dim mismatch: expected {full_action_dim}, got {actions.shape[-1]}.")
        return actions

    expected_dim = full_action_dim - len(action_mask_indices)
    if actions.shape[-1] != expected_dim:
        raise ValueError(f"Masked action dim mismatch: expected {expected_dim}, got {actions.shape[-1]}.")
    kept_indices = [idx for idx in range(full_action_dim) if idx not in set(action_mask_indices)]
    full_actions = torch.zeros((*actions.shape[:-1], full_action_dim), device=actions.device, dtype=actions.dtype)
    full_actions[..., kept_indices] = actions
    return full_actions


def configure_mjlab_randomization(
    env_cfg: Any,
    *,
    use_domain_randomization: bool = True,
    use_push_randomization: bool = True,
    use_observation_noise: bool = True,
    randomization_preset: str = "default",
    randomization_components: Any = None,
    randomization_scale: float = 1.0,
    payload_mass_range_kg: Any = None,
    payload_position_body_m: Any = None,
    payload_box_size_m: Any = None,
    rr_calf_strength_range: Any = None,
) -> None:
    """Apply randomization switches and the flat-ground friend DR preset."""
    preset = str(randomization_preset or "default").lower()
    if not use_domain_randomization:
        for event_name in (
            "foot_friction",
            "encoder_bias",
            "base_com",
            "friend_base_mass",
            "friend_link_mass",
            "friend_actuator_parameters",
        ):
            env_cfg.events.pop(event_name, None)

    if not use_push_randomization:
        env_cfg.events.pop("push_robot", None)

    if not use_observation_noise:
        for group_name in ("actor", "critic"):
            obs_group = env_cfg.observations.get(group_name)
            if obs_group is not None:
                obs_group.enable_corruption = False

    def add_gap_events() -> None:
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg
        from flash_rl.envs.mjlab_dr import (
            add_go2_base_payload_mass,
            randomize_go2_joint_motor_strength,
        )

        payload = tuple(float(x) for x in (payload_mass_range_kg or (0.0, 0.0)))
        if len(payload) != 2:
            raise ValueError("payload_mass_range_kg must contain exactly two values")
        if payload != (0.0, 0.0):
            payload_position = tuple(
                float(x) for x in (payload_position_body_m or (0.0, 0.0, 0.10))
            )
            payload_size = tuple(
                float(x) for x in (payload_box_size_m or (0.20, 0.12, 0.05))
            )
            if len(payload_position) != 3 or len(payload_size) != 3:
                raise ValueError("payload position and box size must contain three values")
            env_cfg.events["gap_base_payload_mass"] = EventTermCfg(
                mode="startup",
                func=add_go2_base_payload_mass,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(r"^base_link$",)),
                    "payload_mass_range_kg": payload,
                    "payload_position_body_m": payload_position,
                    "payload_box_size_m": payload_size,
                },
            )
        strength = tuple(float(x) for x in (rr_calf_strength_range or (1.0, 1.0)))
        if len(strength) != 2:
            raise ValueError("rr_calf_strength_range must contain exactly two values")
        if strength != (1.0, 1.0):
            env_cfg.events["gap_rr_calf_strength"] = EventTermCfg(
                mode="reset",
                func=randomize_go2_joint_motor_strength,
                params={
                    "asset_cfg": SceneEntityCfg("robot"),
                    "joint_names": ("RR_calf_joint",),
                    "strength_range": strength,
                },
            )

    if preset in {"", "default", "old", "mjlab_default", "none"}:
        add_gap_events()
        return
    if preset == "calibrated_default":
        if use_domain_randomization:
            from mjlab.managers.scene_entity_config import SceneEntityCfg

            foot_friction = env_cfg.events.get("foot_friction")
            if foot_friction is not None:
                foot_friction.params["ranges"] = calibrated_flat_friction_range(
                    randomization_scale
                )
                foot_friction.params["shared_random"] = True
                foot_friction.params["asset_cfg"] = SceneEntityCfg(
                    "robot", geom_names=(r".*_collision",)
                )
        add_gap_events()
        return
    if preset not in {"friend_flat", "calibrated_friend_flat"}:
        raise ValueError(f"Unknown mjlab randomization preset: {randomization_preset!r}")

    components = normalize_friend_flat_dr_components(randomization_components)
    scale = normalize_friend_flat_dr_scale(randomization_scale)
    calibrated_friction = preset == "calibrated_friend_flat"

    # Component selection is authoritative. This matters for combined
    # ablations where the caller enables the preset but intentionally omits
    # push or actor-observation corruption.
    if "push" not in components:
        env_cfg.events.pop("push_robot", None)
    if "observation" not in components:
        for group_name in ("actor", "critic"):
            obs_group = env_cfg.observations.get(group_name)
            if obs_group is not None:
                obs_group.enable_corruption = False

    # The base task already defines lightweight DR events. Remove components
    # that are outside this diagnostic variant before applying friend ranges.
    if use_domain_randomization:
        if "friction" not in components:
            env_cfg.events.pop("foot_friction", None)
        if "mass_com" not in components:
            for event_name in ("base_com", "friend_base_mass", "friend_link_mass"):
                env_cfg.events.pop(event_name, None)
        if "motor" not in components:
            for event_name in ("encoder_bias", "friend_actuator_parameters"):
                env_cfg.events.pop(event_name, None)

    if use_domain_randomization:
        from mjlab.actuator import DelayedActuatorCfg
        from mjlab.envs.mdp import dr
        from mjlab.managers.event_manager import EventTermCfg
        from mjlab.managers.scene_entity_config import SceneEntityCfg

        from flash_rl.envs.mjlab_dr import (
            FriendDelayedActuatorCfg,
            friend_go2_actuator_parameters,
            friend_go2_base_mass,
            reset_joints_by_scale,
        )

        foot_friction = env_cfg.events.get("foot_friction")
        if "friction" in components and foot_friction is not None:
            foot_friction.params["ranges"] = (
                calibrated_flat_friction_range(scale)
                if calibrated_friction
                else _scaled_range(0.0, 2.0, 1.0, scale)
            )
            foot_friction.params["shared_random"] = True
            foot_friction.params["asset_cfg"] = SceneEntityCfg(
                "robot", geom_names=(r".*_collision",)
            )

        # The friend zero offset is a PD target calibration error, not actor
        # encoder noise. It is applied by friend_go2_actuator_parameters.
        env_cfg.events.pop("encoder_bias", None)

        base_com = env_cfg.events.get("base_com")
        if "mass_com" in components and base_com is not None:
            base_com.params["ranges"] = {
                0: _scaled_range(-0.03, 0.03, 0.0, scale),
                1: _scaled_range(-0.03, 0.03, 0.0, scale),
                2: _scaled_range(-0.03, 0.03, 0.0, scale),
            }

        if "mass_com" in components:
            env_cfg.events["friend_base_mass"] = EventTermCfg(
                mode="startup",
                func=friend_go2_base_mass,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(r"^base_link$",)),
                    "added_mass_range": _scaled_range(-1.0, 1.0, 0.0, scale),
                },
            )
            env_cfg.events["friend_link_mass"] = EventTermCfg(
                mode="startup",
                func=dr.pseudo_inertia,
                params={
                    "asset_cfg": SceneEntityCfg("robot", body_names=(r"^(?!base_link$).+",)),
                    # pseudo_inertia scales mass and inertia by exp(2 * alpha).
                    "alpha_range": (
                        0.5 * math.log(1.0 + scale * (0.9 - 1.0)),
                        0.5 * math.log(1.0 + scale * (1.1 - 1.0)),
                    ),
                },
            )
        if "motor" in components:
            env_cfg.events["friend_actuator_parameters"] = EventTermCfg(
                mode="reset",
                func=friend_go2_actuator_parameters,
                params={
                    # MJLab actuator DR indexes actuator groups, not MuJoCo ctrl IDs.
                    # The default slice selects every configured group on the robot.
                    "asset_cfg": SceneEntityCfg("robot"),
                    "stiffness_scale_range": _scaled_range(0.9, 1.1, 1.0, scale),
                    "damping_scale_range": _scaled_range(0.9, 1.1, 1.0, scale),
                    "motor_strength_range": _scaled_range(0.8, 1.2, 1.0, scale),
                    "motor_zero_offset_range": _scaled_range(-0.035, 0.035, 0.0, scale),
                },
            )

        if "delay" in components:
            robot_cfg = env_cfg.scene.entities.get("robot")
            if robot_cfg is None or robot_cfg.articulation is None:
                raise ValueError("friend_flat DR requires an articulated scene entity named 'robot'.")
            delayed_actuators = []
            policy_decimation = int(getattr(env_cfg, "decimation", 4))
            delay_max_lag = round(policy_decimation * scale)
            for actuator_cfg in robot_cfg.articulation.actuators:
                if isinstance(actuator_cfg, DelayedActuatorCfg):
                    base_cfg = actuator_cfg.base_cfg
                else:
                    base_cfg = actuator_cfg
                delayed_actuators.append(
                    FriendDelayedActuatorCfg(
                        base_cfg=base_cfg,
                        delay_target="position",
                        delay_min_lag=0,
                        delay_max_lag=delay_max_lag,
                        # The custom wrapper sets a synchronized lag explicitly.
                        delay_hold_prob=1.0,
                        delay_update_period=0,
                        delay_per_env_phase=False,
                        policy_decimation=policy_decimation,
                    )
                )
            env_cfg.scene.entities["robot"] = replace(
                robot_cfg,
                articulation=replace(robot_cfg.articulation, actuators=tuple(delayed_actuators)),
            )

    if use_observation_noise and "observation" in components:
        actor_group = env_cfg.observations.get("actor")
        if actor_group is not None:
            friend_noise_ranges = {
                "base_ang_vel": (-0.2, 0.2),
                "projected_gravity": (-0.05, 0.05),
                "joint_pos": (-0.01, 0.01),
                "joint_vel": (-1.5, 1.5),
            }
            for term_name, (n_min, n_max) in friend_noise_ranges.items():
                term = actor_group.terms.get(term_name)
                if term is not None and term.noise is not None:
                    term.noise.n_min = scale * n_min
                    term.noise.n_max = scale * n_max

    if use_domain_randomization and "initial_state" in components:
        reset_joints = env_cfg.events.get("reset_robot_joints")
        if reset_joints is not None:
            asset_cfg = reset_joints.params.get("asset_cfg", SceneEntityCfg("robot"))
            env_cfg.events["reset_robot_joints"] = EventTermCfg(
                mode="reset",
                func=reset_joints_by_scale,
                params={
                    "scale_range": _scaled_range(0.5, 1.5, 1.0, scale),
                    "velocity_range": (0.0, 0.0),
                    "asset_cfg": asset_cfg,
                },
            )
        reset_base = env_cfg.events.get("reset_base")
        if reset_base is not None:
            half_range = 0.5 * scale
            reset_base.params["velocity_range"] = {
                key: (-half_range, half_range)
                for key in ("x", "y", "z", "roll", "pitch", "yaw")
            }

    push_robot = env_cfg.events.get("push_robot")
    if use_push_randomization and "push" in components and push_robot is not None:
        push_robot.interval_range_s = (4.0, 4.0)
        push_robot.params["velocity_range"] = {
            "x": _scaled_range(-0.4, 0.4, 0.0, scale),
            "y": _scaled_range(-0.4, 0.4, 0.0, scale),
            "z": (0.0, 0.0),
            "roll": _scaled_range(-0.6, 0.6, 0.0, scale),
            "pitch": _scaled_range(-0.6, 0.6, 0.0, scale),
            "yaw": _scaled_range(-0.6, 0.6, 0.0, scale),
        }
    add_gap_events()


def mjlab_randomization_manifest(
    randomization_preset: str,
    randomization_components: Any = None,
    randomization_scale: float = 1.0,
) -> dict[str, Any]:
    preset = str(randomization_preset or "default").lower()
    if preset == "calibrated_default":
        scale = normalize_friend_flat_dr_scale(randomization_scale)
        return {
            "preset": preset,
            "source": "mjlab_task_default_with_real_floor_friction_calibration",
            "scale": scale,
            "implemented": {
                "foot_friction": list(calibrated_flat_friction_range(scale)),
                "friction_center": _CALIBRATED_FRICTION_CENTER,
                "friction_scope": "all_robot_collision_geoms_shared_per_environment",
            },
        }
    if preset not in {"friend_flat", "calibrated_friend_flat"}:
        return {"preset": preset, "source": "mjlab_task_default"}
    components = normalize_friend_flat_dr_components(randomization_components)
    scale = normalize_friend_flat_dr_scale(randomization_scale)
    calibrated_friction = preset == "calibrated_friend_flat"
    return {
        "preset": preset,
        "active_components": sorted(components),
        "scale": scale,
        "source": (
            "wty-yy/go2_rl_gym@vanilla_train_with_real_floor_friction_calibration"
            if calibrated_friction
            else "wty-yy/go2_rl_gym@vanilla_train"
        ),
        "implemented": {
            "foot_friction": list(
                calibrated_flat_friction_range(scale)
                if calibrated_friction
                else _scaled_range(0.0, 2.0, 1.0, scale)
            ),
            "friction_center": (
                _CALIBRATED_FRICTION_CENTER if calibrated_friction else 1.0
            ),
            "friction_scope": "all_robot_collision_geoms_shared_per_environment",
            "base_mass_add_kg": list(_scaled_range(-1.0, 1.0, 0.0, scale)),
            "base_mass_inertia_mapping": "mass_and_inertia_scaled_consistently",
            "non_base_link_mass_scale": [1.0 + scale * (0.9 - 1.0), 1.0 + scale * (1.1 - 1.0)],
            "non_base_link_inertia_mapping": "physics_consistent_pseudo_inertia",
            "base_com_offset_m": list(_scaled_range(-0.03, 0.03, 0.0, scale)),
            "pd_gain_scale": list(_scaled_range(0.9, 1.1, 1.0, scale)),
            "motor_zero_offset_rad": list(_scaled_range(-0.035, 0.035, 0.0, scale)),
            "motor_zero_offset_semantics": "pd_target_offset",
            "motor_strength_scale": list(_scaled_range(0.8, 1.2, 1.0, scale)),
            "push_interval_s": 4.0,
            "push_linear_velocity_xy": list(_scaled_range(-0.4, 0.4, 0.0, scale)),
            "push_angular_velocity": list(_scaled_range(-0.6, 0.6, 0.0, scale)),
            "actuator_delay_physics_steps": [0, round(4 * scale)],
            "actuator_delay_correlation": "one_lag_per_environment_shared_across_joints",
            "observation_noise_uniform": {
                "base_ang_vel": list(_scaled_range(-0.2, 0.2, 0.0, scale)),
                "projected_gravity": list(_scaled_range(-0.05, 0.05, 0.0, scale)),
                "joint_pos": list(_scaled_range(-0.01, 0.01, 0.0, scale)),
                "joint_vel": list(_scaled_range(-1.5, 1.5, 0.0, scale)),
            },
            "joint_reset_scale": list(_scaled_range(0.5, 1.5, 1.0, scale)),
            "root_velocity_reset": list(_scaled_range(-0.5, 0.5, 0.0, scale)),
        },
        "not_mapped": {
            "restitution": "MuJoCo has no direct coefficient-of-restitution field; no solref proxy is used.",
        },
    }


class MjlabVectorEnv(VectorEnv[F32NDArray, F32NDArray, F32NDArray]):
    """Gymnasium VectorEnv adapter around mjlab's ManagerBasedRlEnv.

    The underlying environment is created and stepped with the same
    ManagerBasedRlEnv path used by the PPO runner in scripts/train.py. mjlab
    performs same-step autoreset internally for done envs; this adapter does
    not reset done envs a second time.

    Observations are flattened from mjlab's dict format:
    - If both "actor" and "critic" groups exist: observations are stored as
      [actor | critic]. env_info["actor_observation_size"] is set so FlashSAC's
      agent slices obs[:actor_dim] for the actor and uses the full vector for the
      critic. This keeps noisy actor observations distinct from clean critic
      observations.
    - Otherwise: the single group is used as-is.

    Actions are passed through unchanged (mjlab action terms handle scaling internally).
    """

    def __init__(
        self,
        task_id: str,
        num_envs: int,
        seed: int,
        device: str = "cuda:0",
        to_numpy: bool = True,
        use_domain_randomization: bool = True,
        use_push_randomization: bool = True,
        use_observation_noise: bool = True,
        action_mask_indices: Any = None,
        use_critic_observation_as_full_observation: bool = False,
        randomization_preset: str = "default",
        randomization_components: Any = None,
        randomization_scale: float = 1.0,
    ) -> None:
        import mjlab.tasks  # noqa: F401  # populates the built-in task registry
        import src.tasks  # noqa: F401  # populates this repository's Unitree task registry
        from mjlab.envs import ManagerBasedRlEnv
        from mjlab.tasks.registry import load_env_cfg

        env_cfg = load_env_cfg(task_id)
        env_cfg.scene.num_envs = num_envs
        env_cfg.seed = seed
        configure_mjlab_randomization(
            env_cfg,
            use_domain_randomization=use_domain_randomization,
            use_push_randomization=use_push_randomization,
            use_observation_noise=use_observation_noise,
            randomization_preset=randomization_preset,
            randomization_components=randomization_components,
            randomization_scale=randomization_scale,
        )

        env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
        self._init_from_env(
            env,
            to_numpy=to_numpy,
            action_mask_indices=action_mask_indices,
            use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
        )

    def _flatten_obs(self, obs_dict: dict[str, Any]) -> F32NDArray:
        if self._has_critic_obs and self._use_critic_observation_as_full_observation:
            flat = obs_dict["critic"]
        elif self._has_critic_obs:
            flat = torch.cat([obs_dict["actor"], obs_dict["critic"]], dim=-1)
        else:
            flat = obs_dict["actor"]
        return flat.cpu().numpy().astype(np.float32)

    @staticmethod
    def _scalarize_log_value(value: Any) -> float | int | Any:
        if isinstance(value, torch.Tensor):
            if value.numel() == 0:
                return 0.0
            return float(value.float().mean().item())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return 0.0
            return float(value.astype(np.float32).mean())
        if isinstance(value, (float, int)):
            return value
        return value

    def _extract_episode_info(self, extras: dict[str, Any]) -> dict[str, Any]:
        raw_log = extras.get("log") or {}
        episode_info = {k: self._scalarize_log_value(v) for k, v in raw_log.items()}

        if len(self._reward_buffer) > 0:
            episode_info["Train/mean_reward"] = float(np.mean(self._reward_buffer))
            episode_info["Train/mean_episode_length"] = float(np.mean(self._length_buffer))
            # Keep the original FlashSAC tag names as aliases for older runs/tools.
            episode_info["episode_rewards"] = episode_info["Train/mean_reward"]
            episode_info["episode_length"] = episode_info["Train/mean_episode_length"]

        return episode_info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[F32NDArray, dict[str, Any]]:
        obs_dict, _ = self._env.reset()
        self._ep_returns[:] = 0.0
        self._ep_lengths[:] = 0
        self._reward_buffer.clear()
        self._length_buffer.clear()
        env_info: dict[str, Any] = {}
        if self._has_critic_obs:
            env_info["actor_observation_size"] = (self._actor_obs_dim,)
        return self._flatten_obs(obs_dict), env_info

    def step(
        self,
        actions: Union[F32NDArray, torch.Tensor],
    ) -> tuple[F32NDArray, F32NDArray, NDArray, NDArray, dict[str, Any]]:
        if isinstance(actions, np.ndarray):
            actions_t = torch.from_numpy(actions).float().to(self._device)
        else:
            actions_t = actions.to(self._device)
        actions_t = expand_masked_actions_t(
            actions_t,
            full_action_dim=self._full_action_dim,
            action_mask_indices=self._action_mask_indices,
        )

        obs_dict, rewards, terminateds, truncateds, extras = self._env.step(actions_t)

        rewards_np = rewards.cpu().numpy().astype(np.float32)
        self._ep_returns += rewards_np
        self._ep_lengths += 1

        # PPO's ManagerBasedRlEnv path already performs same-step autoreset for
        # done envs before returning observations. Do not reset here again.
        next_obs = self._flatten_obs(obs_dict)
        dones = terminateds | truncateds
        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)

        done_ids_np = done_ids.cpu().numpy()
        if len(done_ids_np) > 0:
            self._reward_buffer.extend(self._ep_returns[done_ids_np].tolist())
            self._length_buffer.extend(self._ep_lengths[done_ids_np].tolist())
            self._ep_returns[done_ids_np] = 0.0
            self._ep_lengths[done_ids_np] = 0

        episode_info = self._extract_episode_info(extras)
        infos: dict[str, Any] = {
            # mjlab's PPO path returns post-autoreset obs for done envs. The
            # adapter exposes that same observation to the FlashSAC loop.
            "final_obs": next_obs.copy(),
        }
        if episode_info:
            infos["episode_info"] = episode_info

        return (
            next_obs,
            rewards_np,
            terminateds.cpu().numpy(),
            truncateds.cpu().numpy(),
            infos,
        )

    def close(self, **kwargs: Any) -> None:
        if hasattr(self, "_env"):
            self._env.close()

    @classmethod
    def from_env(
        cls,
        env: Any,
        to_numpy: bool = True,
        action_mask_indices: Any = None,
        use_critic_observation_as_full_observation: bool = False,
    ) -> "MjlabVectorEnv":
        """Wrap an already-created ManagerBasedRlEnv."""
        instance = cls.__new__(cls)
        instance._init_from_env(
            env,
            to_numpy=to_numpy,
            action_mask_indices=action_mask_indices,
            use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
        )
        return instance

    def _init_from_env(
        self,
        env: Any,
        to_numpy: bool = True,
        action_mask_indices: Any = None,
        use_critic_observation_as_full_observation: bool = False,
    ) -> None:
        self._env = env
        self._device = str(env.device)
        self._to_numpy = to_numpy
        self.num_envs = env.num_envs

        obs_groups = list(env.single_observation_space.spaces.keys())
        self._has_critic_obs = "actor" in obs_groups and "critic" in obs_groups
        self._use_critic_observation_as_full_observation = (
            bool(use_critic_observation_as_full_observation) and self._has_critic_obs
        )
        self._actor_obs_dim = int(env.single_observation_space.spaces["actor"].shape[0])
        if self._has_critic_obs and self._use_critic_observation_as_full_observation:
            flat_dim = int(env.single_observation_space.spaces["critic"].shape[0])
            if self._actor_obs_dim > flat_dim:
                raise ValueError(
                    "actor observation dim cannot exceed critic observation dim when "
                    "use_critic_observation_as_full_observation=true."
                )
        elif self._has_critic_obs:
            flat_dim = self._actor_obs_dim + int(env.single_observation_space.spaces["critic"].shape[0])
        else:
            flat_dim = self._actor_obs_dim

        self._full_action_dim = int(env.single_action_space.shape[0])
        self._action_mask_indices = normalize_action_mask_indices(action_mask_indices, self._full_action_dim)
        action_dim = self._full_action_dim - len(self._action_mask_indices)
        if action_dim <= 0:
            raise ValueError("action_mask_indices cannot mask every action dimension.")

        self.single_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, env.num_envs)
        self.single_action_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32)
        self.action_space = batch_space(self.single_action_space, env.num_envs)

        self.obs_size = (flat_dim,)
        self.action_size = (action_dim,)
        self._ep_returns = np.zeros(env.num_envs, dtype=np.float32)
        self._ep_lengths = np.zeros(env.num_envs, dtype=np.int32)
        self._reward_buffer = deque(maxlen=100)
        self._length_buffer = deque(maxlen=100)


def make_mjlab_env(
    task_id: str,
    num_envs: int,
    seed: int,
    device: str = "cuda:0",
    use_domain_randomization: bool = True,
    use_push_randomization: bool = True,
    use_observation_noise: bool = True,
    action_mask_indices: Any = None,
    use_critic_observation_as_full_observation: bool = False,
    randomization_preset: str = "default",
    randomization_components: Any = None,
    randomization_scale: float = 1.0,
) -> MjlabVectorEnv:
    env = MjlabVectorEnv(
        task_id=task_id,
        num_envs=num_envs,
        seed=seed,
        device=device,
        use_domain_randomization=use_domain_randomization,
        use_push_randomization=use_push_randomization,
        use_observation_noise=use_observation_noise,
        action_mask_indices=action_mask_indices,
        use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
        randomization_preset=randomization_preset,
        randomization_components=randomization_components,
        randomization_scale=randomization_scale,
    )
    return env
