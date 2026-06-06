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
from src.tasks.rwm_velocity.mdp.extractors import make_go2_policy_obs
from src.tasks.rwm_velocity.mdp.rewards import (
    Go2RWMRewardState,
    bad_orientation_from_state,
    compute_go2_imagination_reward,
)


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
    uncertainty_penalty_weight: float = -1.0


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
        self._obs_dim = 48
        self._action_dim = dynamics.cfg.action_dim
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
            shape=(self._action_dim,),
            dtype=np.float32,
        )
        self.action_space = batch_space(self.single_action_space, self.num_envs)

        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self.model_ids = torch.randint(0, dynamics.ensemble_size, (self.num_envs,), device=self._device)
        self.command_intervals = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self.command = torch.zeros(self.num_envs, 3, device=self._device)
        self.state_history: torch.Tensor
        self.action_history: torch.Tensor
        self.reward_state = Go2RWMRewardState.create(
            num_envs=self.num_envs,
            action_dim=self._action_dim,
            device=self._device,
            step_dt=cfg.step_dt,
        )
        self.reward_state.weights.uncertainty = cfg.uncertainty_penalty_weight
        self._ep_returns = torch.zeros(self.num_envs, device=self._device)
        self._ep_lengths = torch.zeros(self.num_envs, dtype=torch.long, device=self._device)
        self._reward_buffer: deque[float] = deque(maxlen=100)
        self._length_buffer: deque[float] = deque(maxlen=100)
        self._latest_log: dict[str, float] = {}
        self._reset_all()

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
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

    def _reset_histories(self, env_ids: torch.Tensor) -> None:
        states, actions = self.dataset.sample_initial_history(
            batch_size=len(env_ids),
            history_horizon=self.system_dynamics.cfg.history_horizon,
            device=self._device,
        )
        self.state_history[env_ids] = states
        self.action_history[env_ids] = actions

    def _reset_all(self) -> None:
        self.state_history, self.action_history = self.dataset.sample_initial_history(
            batch_size=self.num_envs,
            history_horizon=self.system_dynamics.cfg.history_horizon,
            device=self._device,
        )
        env_ids = torch.arange(self.num_envs, device=self._device)
        self.episode_length_buf.zero_()
        self.model_ids = torch.randint(0, self.system_dynamics.ensemble_size, (self.num_envs,), device=self._device)
        self._sample_commands(env_ids)
        self.reward_state.last_joint_vel = self.state_history[:, -1, 21:33].clone()
        self.reward_state.last_action = self.action_history[:, -1].clone()
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

    def _current_obs_t(self) -> torch.Tensor:
        return make_go2_policy_obs(self.state_history[:, -1], self.command, self.action_history[:, -1])

    def _current_obs_np(self) -> np.ndarray:
        return self._current_obs_t().detach().cpu().numpy().astype(np.float32)

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
            action_t = torch.from_numpy(actions).to(self._device).float()
        else:
            action_t = actions.to(self._device).float()
        action_t = torch.clamp(action_t, -1.0, 1.0)

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
        self._ep_returns += rewards
        self._ep_lengths += 1

        predicted_done = torch.sigmoid(term_logits).squeeze(-1) > 0.5
        bad_orientation = bad_orientation_from_state(next_state)
        time_outs = self.episode_length_buf >= self.cfg.max_episode_length
        terminated = predicted_done | bad_orientation
        truncated = time_outs
        dones = terminated | truncated

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
        }
        for key, value in reward_terms.items():
            self._latest_log[f"Imagination/{key}"] = float(value.mean().detach().cpu())

        next_obs = self._current_obs_np()
        final_obs = final_obs_t.detach().cpu().numpy().astype(np.float32)
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

    def close(self, **kwargs: Any) -> None:
        return None
