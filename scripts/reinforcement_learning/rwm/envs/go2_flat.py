"""Go2 flat imagination environment backed by a learned RWM dynamics ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from rsl_rl.env import VecEnv
from tensordict import TensorDict

from scripts.reinforcement_learning.rwm.dynamics import SequenceReplayBuffer, SystemDynamicsEnsemble
from src.tasks.rwm_velocity.mdp.extractors import make_go2_policy_obs
from src.tasks.rwm_velocity.mdp.rewards import (
    Go2RWMRewardState,
    bad_orientation_from_state,
    compute_go2_imagination_reward,
)


@dataclass
class Go2ImaginationCfg:
    task: str = "go2_flat"
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


class Go2FlatRWMImaginationEnv(VecEnv):
    """RSL-RL VecEnv that rolls out Go2 states inside the learned dynamics."""

    def __init__(
        self,
        dynamics: SystemDynamicsEnsemble,
        dataset: SequenceReplayBuffer,
        cfg: Go2ImaginationCfg,
        device: torch.device | str,
    ) -> None:
        self.system_dynamics = dynamics.to(device).eval()
        self.dataset = dataset
        self.cfg = cfg
        self.num_envs = cfg.num_envs
        self.num_actions = dynamics.cfg.action_dim
        self.max_episode_length = cfg.max_episode_length
        self.device = torch.device(device)
        self.episode_length_buf = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.model_ids = torch.randint(0, dynamics.ensemble_size, (self.num_envs,), device=self.device)
        self.command_intervals = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.command = torch.zeros(self.num_envs, 3, device=self.device)
        self.state_history: torch.Tensor
        self.action_history: torch.Tensor
        self.reward_state = Go2RWMRewardState.create(
            num_envs=self.num_envs,
            action_dim=self.num_actions,
            device=self.device,
            step_dt=cfg.step_dt,
        )
        self.reward_state.weights.uncertainty = cfg.uncertainty_penalty_weight
        self._reset_all()

    def _sample_commands(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.rand(n, 3, device=self.device)
        self.command[env_ids, 0] = self.cfg.lin_vel_x_min + r[:, 0] * (self.cfg.lin_vel_x_max - self.cfg.lin_vel_x_min)
        self.command[env_ids, 1] = self.cfg.lin_vel_y_min + r[:, 1] * (self.cfg.lin_vel_y_max - self.cfg.lin_vel_y_min)
        self.command[env_ids, 2] = self.cfg.ang_vel_z_min + r[:, 2] * (self.cfg.ang_vel_z_max - self.cfg.ang_vel_z_min)
        standing = torch.rand(n, device=self.device) < self.cfg.rel_standing_envs
        self.command[env_ids[standing]] = 0.0
        self.command_intervals[env_ids] = torch.randint(
            self.cfg.command_resample_interval_min,
            self.cfg.command_resample_interval_max + 1,
            (n,),
            device=self.device,
        )

    def _reset_histories(self, env_ids: torch.Tensor) -> None:
        states, actions = self.dataset.sample_initial_history(
            batch_size=len(env_ids),
            history_horizon=self.system_dynamics.cfg.history_horizon,
            device=self.device,
        )
        self.state_history[env_ids] = states
        self.action_history[env_ids] = actions

    def _reset_all(self) -> None:
        self.state_history, self.action_history = self.dataset.sample_initial_history(
            batch_size=self.num_envs,
            history_horizon=self.system_dynamics.cfg.history_horizon,
            device=self.device,
        )
        env_ids = torch.arange(self.num_envs, device=self.device)
        self.episode_length_buf.zero_()
        self.model_ids = torch.randint(0, self.system_dynamics.ensemble_size, (self.num_envs,), device=self.device)
        self._sample_commands(env_ids)
        self.reward_state.last_joint_vel = self.state_history[:, -1, 21:33].clone()
        self.reward_state.last_action = self.action_history[:, -1].clone()

    def _reset_idx(self, env_ids: torch.Tensor) -> None:
        if len(env_ids) == 0:
            return
        self._reset_histories(env_ids)
        self.episode_length_buf[env_ids] = 0
        self.model_ids[env_ids] = torch.randint(0, self.system_dynamics.ensemble_size, (len(env_ids),), device=self.device)
        self._sample_commands(env_ids)
        self.reward_state.last_joint_vel[env_ids] = self.state_history[env_ids, -1, 21:33]
        self.reward_state.last_action[env_ids] = self.action_history[env_ids, -1]

    def get_observations(self) -> TensorDict:
        obs = make_go2_policy_obs(self.state_history[:, -1], self.command, self.action_history[:, -1])
        return TensorDict({"actor": obs, "critic": obs}, batch_size=[self.num_envs], device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[TensorDict, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions = actions.to(self.device).float()
        self.action_history = torch.cat([self.action_history[:, 1:], actions.unsqueeze(1)], dim=1)
        with torch.no_grad():
            next_state, _aleatoric, epistemic, contact_logits, term_logits = self.system_dynamics.predict(
                self.state_history,
                self.action_history,
                self.model_ids,
            )
        foot_contact = (torch.sigmoid(contact_logits) > 0.5).float()

        rewards, reward_terms = compute_go2_imagination_reward(
            state=next_state,
            action=actions,
            command=self.command,
            foot_contact=foot_contact,
            episode_length=self.episode_length_buf,
            reward_state=self.reward_state,
            epistemic_uncertainty=epistemic,
        )

        self.state_history = torch.cat([self.state_history[:, 1:], next_state.unsqueeze(1)], dim=1)
        self.episode_length_buf += 1

        predicted_done = torch.sigmoid(term_logits).squeeze(-1) > 0.5
        bad_orientation = bad_orientation_from_state(next_state)
        time_outs = self.episode_length_buf >= self.max_episode_length
        dones_bool = predicted_done | bad_orientation | time_outs
        dones = dones_bool.long()

        resample_ids = (self.episode_length_buf % self.command_intervals == 0).nonzero(as_tuple=False).squeeze(-1)
        self._sample_commands(resample_ids)

        reset_ids = dones_bool.nonzero(as_tuple=False).squeeze(-1)
        self._reset_idx(reset_ids)

        log = {
            "Imagination/epistemic_uncertainty": epistemic.mean(),
            "Imagination/predicted_done": predicted_done.float().mean(),
            "Imagination/bad_orientation": bad_orientation.float().mean(),
            "Imagination/num_valid_imagination_envs": (1.0 - dones.float()).sum(),
        }
        for key, value in reward_terms.items():
            log[f"Imagination/{key}"] = value.mean()

        extras = {"time_outs": time_outs, "log": log}
        return self.get_observations(), rewards, dones, extras

    def close(self) -> None:
        return None
