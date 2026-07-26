"""Gymnasium VectorEnv wrapper around the learned Go2 RWM dynamics."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from scripts.reinforcement_learning.rwm.dynamics import SequenceReplayBuffer, SystemDynamicsEnsemble
from scripts.reinforcement_learning.rwm_flashsac.command_sampling import sample_mode_balanced_commands
from scripts.reinforcement_learning.rwm_dataset.action_mask import normalize_action_mask_indices
from scripts.reinforcement_learning.rwm_dataset.joint_feature_mask import mask_tensor_features_t
from src.tasks.rwm_velocity.mdp.extractors import make_go2_policy_obs
from src.tasks.rwm_velocity.mdp.rewards import (
    Go2RWMRewardState,
    bad_orientation_from_state,
    compute_go2_imagination_reward,
)


_INTERFACE_OBS_NOISE_PROFILES = {
    "none",
    "friend",
    "legacy_friend",
    "deployment_small",
}


def interface_observation_noise_half_range(
    profile: str,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Return unscaled uniform half-ranges in the 48D RWM observation order."""

    normalized = str(profile or "none").lower()
    if normalized not in _INTERFACE_OBS_NOISE_PROFILES:
        raise ValueError(f"Unknown interface observation noise profile: {profile!r}")

    half_range = torch.zeros(48, device=device)
    if normalized == "friend":
        # Exact sensor-noise ranges from wty-yy/go2_rl_gym@vanilla_train.
        half_range[3:6] = 0.2
        half_range[6:9] = 0.05
        half_range[12:24] = 0.01
        half_range[24:36] = 1.5
    elif normalized in {"legacy_friend", "deployment_small"}:
        # legacy_friend preserves the previously shipped RWM-interface mapping.
        # deployment_small intentionally reuses those small base ranges as a
        # separate, explicit profile for the first normal-Go2 factorial screen.
        half_range[3:6] = 0.05
        half_range[6:9] = 0.05
        half_range[12:24] = 0.01
        half_range[24:36] = 0.075
    return half_range


@dataclass
class FlashSACWorldModelEnvConfig:
    num_envs: int = 4096
    max_episode_length: int = 1000
    step_dt: float = 0.02
    command_resample_interval_min: int = 150
    command_resample_interval_max: int = 400
    lin_vel_x_min: float = -1.0
    lin_vel_x_max: float = 2.0
    lin_vel_y_min: float = -1.0
    lin_vel_y_max: float = 1.0
    ang_vel_z_min: float = -1.0
    ang_vel_z_max: float = 1.0
    rel_standing_envs: float = 0.05
    command_sampling_mode: str = "independent_uniform"
    command_min_abs_x: float = 0.05
    command_min_abs_y: float = 0.03
    command_min_abs_yaw: float = 0.05
    command_mode_weight_stand: float = 0.08
    command_mode_weight_pure_x: float = 0.25
    command_mode_weight_pure_y: float = 0.10
    command_mode_weight_pure_yaw: float = 0.08
    command_mode_weight_xy: float = 0.14
    command_mode_weight_x_yaw: float = 0.17
    command_mode_weight_y_yaw: float = 0.05
    command_mode_weight_xy_yaw: float = 0.13
    uncertainty_penalty_weight: float = -1.0
    reward_version: str = "v1"
    reward_track_linear_velocity_weight: float = 5.0
    reward_track_angular_velocity_weight: float = 3.0
    reward_body_orientation_l2_weight: float = -0.75
    reward_body_ang_vel_weight: float = -0.03
    reward_foot_gait_weight: float = 0.4
    reward_stand_still_weight: float = -1.0
    reward_command_response_weight: float = 2.0
    reward_yaw_command_response_weight: float = 1.0
    reward_wrong_direction_weight: float = -2.0
    reward_response_shortfall_weight: float = -6.0
    reward_response_floor: float = 0.30
    reward_active_command_bias: float = -0.20
    reward_active_survival: float = 0.0
    reward_action_rate_l2: float = -0.05
    reward_stand_action_rate_l2: float = -0.05
    reward_action_saturation: float = 0.0
    reward_action_saturation_threshold: float = 0.9
    reward_dof_acc_l2: float = -2.5e-7
    reward_dof_torques_l2: float = -2.5e-5
    reward_command_active_threshold: float = 0.02
    reward_command_active_threshold_x: float = 0.03
    reward_command_active_threshold_y: float = 0.02
    reward_command_active_threshold_yaw: float = 0.03
    reward_motion_gate_low: float = 0.05
    reward_motion_gate_high: float = 0.30
    reward_response_command_scale_floor: float = 0.05
    reward_response_clip_min: float = -2.0
    reward_response_clip_max: float = 2.0
    reward_response_shortfall_max: float = 1.0
    reward_action_cost_gate_floor: float = 0.10
    reward_model_support_uncertainty_low: float = 0.05
    reward_model_support_uncertainty_high: float = 0.10
    reward_termination_penalty: float = 0.0
    policy_action_mask_indices: tuple[int, ...] = ()
    world_model_action_mask_indices: tuple[int, ...] = ()
    policy_observation_mask_indices: tuple[int, ...] = ()
    broken_joint_names: tuple[str, ...] = ()
    interface_action_noise_std: float = 0.0
    interface_action_bias_std: float = 0.0
    interface_action_scale_min: float = 1.0
    interface_action_scale_max: float = 1.0
    interface_action_delay_steps_min: int = 0
    interface_action_delay_steps_max: int = 0
    interface_obs_noise_profile: str = "none"
    interface_obs_noise_scale: float = 1.0
    interface_obs_joint_pos_bias_min: float = 0.0
    interface_obs_joint_pos_bias_max: float = 0.0
    # Legacy isotropic knobs. Formal DR runs leave these at zero.
    interface_obs_noise_std: float = 0.0
    interface_obs_bias_std: float = 0.0


def configure_go2_reward_state(
    cfg: FlashSACWorldModelEnvConfig,
    *,
    num_envs: int,
    action_dim: int,
    device: torch.device | str,
) -> Go2RWMRewardState:
    """Create one reward state shared by imagination and real replay."""

    state = Go2RWMRewardState.create(
        num_envs=num_envs,
        action_dim=action_dim,
        device=device,
        step_dt=cfg.step_dt,
        reward_version=cfg.reward_version,
    )
    state.weights.track_linear_velocity = cfg.reward_track_linear_velocity_weight
    state.weights.track_angular_velocity = cfg.reward_track_angular_velocity_weight
    state.weights.body_orientation_l2 = cfg.reward_body_orientation_l2_weight
    state.weights.body_ang_vel = cfg.reward_body_ang_vel_weight
    state.weights.foot_gait = cfg.reward_foot_gait_weight
    state.weights.stand_still = cfg.reward_stand_still_weight
    state.weights.uncertainty = cfg.uncertainty_penalty_weight
    state.weights.command_response = cfg.reward_command_response_weight
    state.weights.yaw_command_response = cfg.reward_yaw_command_response_weight
    state.weights.wrong_direction = cfg.reward_wrong_direction_weight
    state.weights.response_shortfall = cfg.reward_response_shortfall_weight
    state.weights.response_floor = cfg.reward_response_floor
    state.weights.active_command_bias = cfg.reward_active_command_bias
    state.weights.active_survival = cfg.reward_active_survival
    state.weights.action_rate_l2 = cfg.reward_action_rate_l2
    state.weights.stand_action_rate_l2 = cfg.reward_stand_action_rate_l2
    state.weights.action_saturation = cfg.reward_action_saturation
    state.weights.action_saturation_threshold = cfg.reward_action_saturation_threshold
    state.weights.dof_acc_l2 = cfg.reward_dof_acc_l2
    state.weights.dof_torques_l2 = cfg.reward_dof_torques_l2
    state.command_active_threshold = cfg.reward_command_active_threshold
    state.command_active_threshold_x = cfg.reward_command_active_threshold_x
    state.command_active_threshold_y = cfg.reward_command_active_threshold_y
    state.command_active_threshold_yaw = cfg.reward_command_active_threshold_yaw
    state.motion_gate_low = cfg.reward_motion_gate_low
    state.motion_gate_high = cfg.reward_motion_gate_high
    state.response_command_scale_floor = cfg.reward_response_command_scale_floor
    state.response_clip_min = cfg.reward_response_clip_min
    state.response_clip_max = cfg.reward_response_clip_max
    state.response_shortfall_max = cfg.reward_response_shortfall_max
    state.action_cost_gate_floor = cfg.reward_action_cost_gate_floor
    state.model_support_uncertainty_low = cfg.reward_model_support_uncertainty_low
    state.model_support_uncertainty_high = cfg.reward_model_support_uncertainty_high
    return state


class Go2RWMFlashSACWorldModelEnv(VectorEnv):
    """VectorEnv that produces synthetic Go2 transitions from a frozen RWM."""

    metadata: dict[str, Any] = {}

    def __init__(
        self,
        dynamics: SystemDynamicsEnsemble,
        dataset: SequenceReplayBuffer,
        cfg: FlashSACWorldModelEnvConfig,
        device: torch.device | str,
    ) -> None:
        self.system_dynamics = dynamics.to(device).eval()
        self.dataset = dataset
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self._device = torch.device(device)
        self._full_obs_dim = 48
        self._policy_observation_mask_indices = normalize_action_mask_indices(
            cfg.policy_observation_mask_indices,
            action_dim=self._full_obs_dim,
        )
        self._obs_dim = self._full_obs_dim - len(self._policy_observation_mask_indices)
        self._full_action_dim = int(getattr(dynamics.cfg, "full_action_dim", dynamics.cfg.action_dim))
        self._model_action_dim = int(dynamics.cfg.action_dim)
        self._policy_action_mask_indices = normalize_action_mask_indices(
            cfg.policy_action_mask_indices,
            action_dim=self._full_action_dim,
        )
        self._world_model_action_mask_indices = normalize_action_mask_indices(
            cfg.world_model_action_mask_indices,
            action_dim=self._full_action_dim,
        )
        if len(self._policy_action_mask_indices) >= self._full_action_dim:
            raise ValueError("policy_action_mask_indices cannot mask every action dimension.")
        self._policy_action_dim = self._full_action_dim - len(self._policy_action_mask_indices)
        self._kept_action_indices = tuple(
            idx for idx in range(self._full_action_dim) if idx not in set(self._policy_action_mask_indices)
        )
        self._kept_action_indices_t = torch.tensor(self._kept_action_indices, dtype=torch.long, device=self._device)
        self._full_action_zero_indices = tuple(
            sorted(set(self._policy_action_mask_indices) | set(self._world_model_action_mask_indices))
        )
        self._full_action_zero_indices_t = torch.tensor(
            self._full_action_zero_indices,
            dtype=torch.long,
            device=self._device,
        )
        self._interface_action_delay_min = int(cfg.interface_action_delay_steps_min)
        self._interface_action_delay_max = int(cfg.interface_action_delay_steps_max)
        if self._interface_action_delay_min < 0 or self._interface_action_delay_max < self._interface_action_delay_min:
            raise ValueError(
                "Invalid interface action delay range: "
                f"{self._interface_action_delay_min}, {self._interface_action_delay_max}"
            )
        if float(cfg.interface_action_scale_min) <= 0.0 or float(cfg.interface_action_scale_max) < float(
            cfg.interface_action_scale_min
        ):
            raise ValueError(
                "Invalid interface action scale range: "
                f"{cfg.interface_action_scale_min}, {cfg.interface_action_scale_max}"
            )
        self._interface_obs_noise_profile = str(cfg.interface_obs_noise_profile or "none").lower()
        if self._interface_obs_noise_profile not in _INTERFACE_OBS_NOISE_PROFILES:
            raise ValueError(f"Unknown interface observation noise profile: {cfg.interface_obs_noise_profile!r}")
        if float(cfg.interface_obs_noise_scale) < 0.0:
            raise ValueError("interface_obs_noise_scale must be non-negative.")
        if float(cfg.interface_obs_joint_pos_bias_max) < float(cfg.interface_obs_joint_pos_bias_min):
            raise ValueError("Invalid interface joint-position bias range.")
        self.single_observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self._obs_dim,),
            dtype=np.float32,
        )
        self.observation_space = batch_space(self.single_observation_space, self.num_envs)
        self.single_action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._policy_action_dim,),
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self.model_ids = torch.randint(0, dynamics.ensemble_size, (self.num_envs,), device=self._device)
        self.command_intervals = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self.command = torch.zeros(self.num_envs, 3, device=self._device)
        self.state_history: torch.Tensor
        self.action_history: torch.Tensor
        self.reward_state = configure_go2_reward_state(
            cfg,
            num_envs=self.num_envs,
            action_dim=self._full_action_dim,
            device=self._device,
        )
        self._ep_returns = torch.zeros(self.num_envs, device=self._device)
        self._ep_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self._reward_buffer: deque[float] = deque(maxlen=100)
        self._length_buffer: deque[float] = deque(maxlen=100)
        self._latest_log: dict[str, float] = {}
        self._interface_action_history: torch.Tensor
        self._interface_action_scale: torch.Tensor
        self._interface_action_bias: torch.Tensor
        self._interface_obs_bias: torch.Tensor
        self._interface_obs_noise_half_range = self._make_interface_obs_noise_half_range()
        self._reset_all()

    @property
    def full_action_dim(self) -> int:
        return self._full_action_dim

    @property
    def policy_action_dim(self) -> int:
        return self._policy_action_dim

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        if self.cfg.command_sampling_mode == "mode_balanced":
            commands, _mode_ids = sample_mode_balanced_commands(n, self.cfg, self._device)
            self.command[env_ids] = commands
            self.command_intervals[env_ids] = torch.randint(
                self.cfg.command_resample_interval_min,
                self.cfg.command_resample_interval_max + 1,
                (n,),
                device=self._device,
            )
            return
        if self.cfg.command_sampling_mode != "independent_uniform":
            raise ValueError(f"Unknown command_sampling_mode={self.cfg.command_sampling_mode!r}")
        r = torch.rand(n, 3, device=self._device)
        self.command[env_ids, 0] = self.cfg.lin_vel_x_min + r[:, 0] * (
            self.cfg.lin_vel_x_max - self.cfg.lin_vel_x_min
        )
        self.command[env_ids, 1] = self.cfg.lin_vel_y_min + r[:, 1] * (
            self.cfg.lin_vel_y_max - self.cfg.lin_vel_y_min
        )
        self.command[env_ids, 2] = self.cfg.ang_vel_z_min + r[:, 2] * (
            self.cfg.ang_vel_z_max - self.cfg.ang_vel_z_min
        )
        standing = torch.rand(n, device=self._device) < self.cfg.rel_standing_envs
        self.command[env_ids[standing]] = 0.0
        self.command_intervals[env_ids] = torch.randint(
            self.cfg.command_resample_interval_min,
            self.cfg.command_resample_interval_max + 1,
            (n,),
            device=self._device,
        )

    def _apply_full_action_zero_mask(self, actions: torch.Tensor) -> torch.Tensor:
        if self._full_action_zero_indices_t.numel() > 0:
            actions[..., self._full_action_zero_indices_t] = 0.0
        return actions

    def _sample_interface_action_scale(self, count: int) -> torch.Tensor:
        lo = float(self.cfg.interface_action_scale_min)
        hi = float(self.cfg.interface_action_scale_max)
        if lo == 1.0 and hi == 1.0:
            return torch.ones(count, self._full_action_dim, device=self._device)
        return torch.empty(count, self._full_action_dim, device=self._device).uniform_(lo, hi)

    def _sample_interface_action_bias(self, count: int) -> torch.Tensor:
        std = float(self.cfg.interface_action_bias_std)
        if std <= 0.0:
            return torch.zeros(count, self._full_action_dim, device=self._device)
        return torch.randn(count, self._full_action_dim, device=self._device) * std

    def _make_interface_obs_noise_half_range(self) -> torch.Tensor:
        # Full RWM observation layout:
        # [base_lin_vel, base_ang_vel, gravity, command, joint_pos,
        #  joint_vel, previous_action]. Commands, previous actions, and the
        # privileged base linear velocity remain clean.
        full_range = interface_observation_noise_half_range(
            self._interface_obs_noise_profile,
            device=self._device,
        )
        return mask_tensor_features_t(
            full_range.unsqueeze(0),
            self._policy_observation_mask_indices,
        ).squeeze(0)

    def _sample_interface_obs_bias(self, count: int) -> torch.Tensor:
        full_bias = torch.zeros(count, self._full_obs_dim, device=self._device)
        if self._interface_obs_noise_profile in {"friend", "legacy_friend"}:
            lo = float(self.cfg.interface_obs_joint_pos_bias_min)
            hi = float(self.cfg.interface_obs_joint_pos_bias_max)
            if lo != 0.0 or hi != 0.0:
                full_bias[:, 12:24] = torch.empty(count, 12, device=self._device).uniform_(lo, hi)
        bias = mask_tensor_features_t(full_bias, self._policy_observation_mask_indices)
        legacy_std = float(self.cfg.interface_obs_bias_std)
        if legacy_std > 0.0:
            bias = bias + torch.randn_like(bias) * legacy_std
        return bias

    def _reset_interface_randomization_all(self) -> None:
        history_len = self._interface_action_delay_max + 1
        latest_action = self.action_history[:, -1]
        self._interface_action_history = latest_action.unsqueeze(0).repeat(history_len, 1, 1).clone()
        self._interface_action_scale = self._sample_interface_action_scale(self.num_envs)
        self._interface_action_bias = self._sample_interface_action_bias(self.num_envs)
        self._interface_obs_bias = self._sample_interface_obs_bias(self.num_envs)

    def _reset_interface_randomization_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._interface_action_history[:, env_ids] = self.action_history[env_ids, -1].unsqueeze(0).repeat(
            self._interface_action_delay_max + 1,
            1,
            1,
        )
        self._interface_action_scale[env_ids] = self._sample_interface_action_scale(len(env_ids))
        self._interface_action_bias[env_ids] = self._sample_interface_action_bias(len(env_ids))
        self._interface_obs_bias[env_ids] = self._sample_interface_obs_bias(len(env_ids))

    def _apply_interface_action_dr(self, actions: torch.Tensor) -> torch.Tensor:
        actual = actions * self._interface_action_scale + self._interface_action_bias
        if float(self.cfg.interface_action_noise_std) > 0.0:
            actual = actual + float(self.cfg.interface_action_noise_std) * torch.randn_like(actual)
        actual = torch.clamp(actual, -1.0, 1.0)
        if self._interface_action_delay_max <= 0:
            return self._apply_full_action_zero_mask(actual)

        self._interface_action_history[:-1].copy_(self._interface_action_history[1:].clone())
        self._interface_action_history[-1].copy_(actual)
        delays = torch.randint(
            self._interface_action_delay_min,
            self._interface_action_delay_max + 1,
            (actions.shape[0],),
            device=self._device,
        )
        env_ids = torch.arange(actions.shape[0], device=self._device)
        delayed = self._interface_action_history[self._interface_action_delay_max - delays, env_ids].clone()
        return self._apply_full_action_zero_mask(delayed)

    def _apply_interface_obs_dr(self, observations: torch.Tensor) -> torch.Tensor:
        obs = observations + self._interface_obs_bias
        if self._interface_obs_noise_profile != "none" and float(self.cfg.interface_obs_noise_scale) > 0.0:
            noise = torch.empty_like(obs).uniform_(-1.0, 1.0)
            obs = obs + noise * self._interface_obs_noise_half_range * float(self.cfg.interface_obs_noise_scale)
        if float(self.cfg.interface_obs_noise_std) > 0.0:
            obs = obs + float(self.cfg.interface_obs_noise_std) * torch.randn_like(obs)
        return obs

    def _expand_policy_actions(self, actions: torch.Tensor) -> torch.Tensor:
        if actions.shape[-1] != self._policy_action_dim:
            raise ValueError(
                f"Action dim mismatch: expected policy action dim {self._policy_action_dim}, "
                f"got {int(actions.shape[-1])}."
            )
        actions = torch.clamp(actions, -1.0, 1.0)
        if not self._policy_action_mask_indices:
            full_actions = actions.clone()
        else:
            full_actions = torch.zeros(
                *actions.shape[:-1],
                self._full_action_dim,
                dtype=actions.dtype,
                device=actions.device,
            )
            full_actions[..., self._kept_action_indices_t] = actions
        return self._apply_full_action_zero_mask(full_actions)

    def _reset_histories(self, env_ids: torch.Tensor) -> None:
        states, actions = self.dataset.sample_initial_history(
            batch_size=len(env_ids),
            history_horizon=self.system_dynamics.cfg.history_horizon,
            device=self._device,
        )
        self.state_history[env_ids] = states
        self.action_history[env_ids] = self._apply_full_action_zero_mask(actions)

    def _reset_all(self) -> None:
        self.state_history, self.action_history = self.dataset.sample_initial_history(
            batch_size=self.num_envs,
            history_horizon=self.system_dynamics.cfg.history_horizon,
            device=self._device,
        )
        self.action_history = self._apply_full_action_zero_mask(self.action_history)
        env_ids = torch.arange(self.num_envs, device=self._device)
        self.episode_length_buf.zero_()
        self.model_ids = torch.randint(0, self.system_dynamics.ensemble_size, (self.num_envs,), device=self._device)
        self._sample_commands(env_ids)
        self.reward_state.last_joint_vel = self.state_history[:, -1, 21:33].clone()
        self.reward_state.last_action = self.action_history[:, -1].clone()
        self.reward_state.base_lin_vel_xy_ema = self.state_history[:, :, 0:2].mean(dim=1).clone()
        self.reward_state.base_yaw_vel_ema = self.state_history[:, :, 5].mean(dim=1).clone()
        self._reset_interface_randomization_all()
        self._ep_returns.zero_()
        self._ep_lengths.zero_()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._reset_histories(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.model_ids[env_ids] = torch.randint(
            0,
            self.system_dynamics.ensemble_size,
            (len(env_ids),),
            device=self._device,
        )
        self._sample_commands(env_ids)
        self.reward_state.last_joint_vel[env_ids] = self.state_history[env_ids, -1, 21:33]
        self.reward_state.last_action[env_ids] = self.action_history[env_ids, -1]
        self.reward_state.base_lin_vel_xy_ema[env_ids] = self.state_history[env_ids, :, 0:2].mean(dim=1)
        self.reward_state.base_yaw_vel_ema[env_ids] = self.state_history[env_ids, :, 5].mean(dim=1)
        self._reset_interface_randomization_idx(env_ids)

    def _current_obs_t(self) -> torch.Tensor:
        full_obs = make_go2_policy_obs(self.state_history[:, -1], self.command, self.action_history[:, -1])
        return mask_tensor_features_t(full_obs, self._policy_observation_mask_indices)

    def _current_obs_np(self) -> np.ndarray:
        return self._apply_interface_obs_dr(self._current_obs_t()).detach().cpu().numpy().astype(np.float32)

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        del options
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self._reset_all()
        self._reward_buffer.clear()
        self._length_buffer.clear()
        return self._current_obs_np(), {}

    def _episode_info(self) -> dict[str, float]:
        info: dict[str, float] = dict(self._latest_log)
        if self._reward_buffer:
            info["Train/mean_reward"] = float(np.mean(self._reward_buffer))
            info["Train/mean_episode_length"] = float(np.mean(self._length_buffer))
        return info

    def step(self, actions: np.ndarray | torch.Tensor) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        if isinstance(actions, np.ndarray):
            policy_action_t = torch.from_numpy(actions).to(self._device).float()
        else:
            policy_action_t = actions.to(self._device).float()
        action_command_t = self._expand_policy_actions(policy_action_t)
        action_t = self._apply_interface_action_dr(action_command_t)

        self.action_history = torch.cat([self.action_history[:, 1:], action_t.unsqueeze(1)], dim=1)
        with torch.no_grad():
            next_state, _aleatoric, epistemic, contact_logits, term_logits = self.system_dynamics.predict(
                self.state_history,
                self.action_history,
                self.model_ids,
            )
        foot_contact = (torch.sigmoid(contact_logits) > 0.5).float()
        rewards, reward_terms = compute_go2_imagination_reward(
            state=next_state,
            action=action_t,
            command=self.command,
            foot_contact=foot_contact,
            episode_length=self.episode_length_buf,
            reward_state=self.reward_state,
            epistemic_uncertainty=epistemic,
        )

        self.state_history = torch.cat([self.state_history[:, 1:], next_state.unsqueeze(1)], dim=1)
        final_obs_t = make_go2_policy_obs(next_state, self.command, action_t)
        self.episode_length_buf += 1
        predicted_done = torch.sigmoid(term_logits).squeeze(-1) > 0.5
        bad_orientation = bad_orientation_from_state(next_state)
        time_outs = self.episode_length_buf >= self.cfg.max_episode_length
        terminated = predicted_done | bad_orientation
        truncated = time_outs
        dones = terminated | truncated
        termination_cost = terminated.float() * float(self.cfg.reward_termination_penalty)
        rewards = rewards + termination_cost
        reward_terms["termination_penalty"] = termination_cost
        self._ep_returns += rewards
        self._ep_lengths += 1
        final_obs = self._apply_interface_obs_dr(
            mask_tensor_features_t(
                final_obs_t,
                self._policy_observation_mask_indices,
            )
        ).detach().cpu().numpy().astype(np.float32)

        resample_ids = (self.episode_length_buf % self.command_intervals == 0).nonzero(as_tuple=False).squeeze(-1)
        self._sample_commands(resample_ids)

        done_ids = dones.nonzero(as_tuple=False).squeeze(-1)
        if len(done_ids) > 0:
            self._reward_buffer.extend(self._ep_returns[done_ids].detach().cpu().tolist())
            self._length_buffer.extend(self._ep_lengths[done_ids].detach().cpu().tolist())
            self._ep_returns[done_ids] = 0.0
            self._ep_lengths[done_ids] = 0
            self._reset_idx(done_ids)

        self._latest_log = {
            "Imagination/epistemic_uncertainty": float(epistemic.mean().detach().cpu()),
            "Imagination/predicted_done": float(predicted_done.float().mean().detach().cpu()),
            "Imagination/bad_orientation": float(bad_orientation.float().mean().detach().cpu()),
            "Imagination/num_valid_imagination_envs": float((~dones).float().sum().detach().cpu()),
            "Imagination/interface_action_delta_abs": float(
                torch.abs(action_t - action_command_t).mean().detach().cpu()
            ),
        }
        for key, value in reward_terms.items():
            self._latest_log[f"Imagination/{key}"] = float(value.mean().detach().cpu())

        next_obs = self._current_obs_np()
        infos = {
            "final_obs": final_obs,
            "episode_info": self._episode_info(),
        }
        return (
            next_obs,
            rewards.detach().cpu().numpy().astype(np.float32),
            terminated.detach().cpu().numpy(),
            truncated.detach().cpu().numpy(),
            infos,
        )

    def state_dict(self) -> dict[str, Any]:
        """Capture all mutable imagination state needed for exact continuation."""

        tensor_names = (
            "state_history",
            "action_history",
            "episode_length_buf",
            "model_ids",
            "command_intervals",
            "command",
            "_ep_returns",
            "_ep_lengths",
            "_interface_action_history",
            "_interface_action_scale",
            "_interface_action_bias",
            "_interface_obs_bias",
        )
        return {
            "format_version": "go2_rwm_flashsac_env_v1",
            "num_envs": self.num_envs,
            "tensors": {
                name: getattr(self, name).detach().cpu().clone()
                for name in tensor_names
            },
            "reward_state_tensors": {
                name: value.detach().cpu().clone()
                for name, value in vars(self.reward_state).items()
                if isinstance(value, torch.Tensor)
            },
            "reward_buffer": list(self._reward_buffer),
            "length_buffer": list(self._length_buffer),
            "latest_log": dict(self._latest_log),
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        if state.get("format_version") != "go2_rwm_flashsac_env_v1":
            raise ValueError("RWM environment checkpoint format mismatch.")
        if int(state["num_envs"]) != self.num_envs:
            raise ValueError("RWM environment count changed across resume.")
        tensors = state.get("tensors")
        if not isinstance(tensors, dict):
            raise ValueError("RWM environment checkpoint lacks tensors.")
        for name, value in tensors.items():
            target = getattr(self, name, None)
            if not isinstance(target, torch.Tensor):
                raise ValueError(f"RWM environment state field {name!r} changed.")
            source = torch.as_tensor(value, device=self._device)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    f"RWM environment state field {name!r} shape/dtype changed."
                )
            target.copy_(source)
        reward_tensors = state.get("reward_state_tensors")
        if not isinstance(reward_tensors, dict):
            raise ValueError("RWM environment checkpoint lacks reward state.")
        for name, value in reward_tensors.items():
            target = getattr(self.reward_state, name, None)
            if not isinstance(target, torch.Tensor):
                raise ValueError(f"RWM reward state field {name!r} changed.")
            source = torch.as_tensor(value, device=self._device)
            if source.shape != target.shape or source.dtype != target.dtype:
                raise ValueError(
                    f"RWM reward state field {name!r} shape/dtype changed."
                )
            target.copy_(source)
        self._reward_buffer.clear()
        self._reward_buffer.extend(float(value) for value in state["reward_buffer"])
        self._length_buffer.clear()
        self._length_buffer.extend(float(value) for value in state["length_buffer"])
        self._latest_log = {
            str(key): float(value)
            for key, value in dict(state["latest_log"]).items()
        }

    def close(self, **kwargs: Any) -> None:
        return None
