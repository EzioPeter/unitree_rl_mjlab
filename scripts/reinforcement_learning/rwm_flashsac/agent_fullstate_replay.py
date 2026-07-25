"""Full-state 48D FlashSAC agent with the audited replay mixer."""

from __future__ import annotations

from typing import Any, MutableMapping, cast

import gymnasium as gym
import torch

from flash_rl.agents.flashSAC.agent import (
    FlashSACAgent,
    FlashSACConfig,
    _sample_flashsac_actions,
    _update_networks,
)
from flash_rl.types import NDArray, Tensor
from scripts.reinforcement_learning.rwm_flashsac.agent_proprioceptive import (
    FlashSACProprioceptiveAgent,
)
from scripts.reinforcement_learning.rwm_flashsac.replay_mixer import (
    sample_mixed_replay_batch,
)


class FlashSACFullStateReplayAgent(FlashSACProprioceptiveAgent):
    """Keep all 48 observation dimensions for both actor and critic."""

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
        action_temperature: float | None = None,
    ) -> Tensor:
        del interaction_step
        if action_temperature is not None:
            temperature = float(action_temperature)
        elif training:
            temperature = 1.0
        else:
            temperature = 0.0
        observations = torch.as_tensor(
            prev_transition["next_observation"],
            dtype=torch.float32,
            device=self._device,
        )
        if observations.shape[-1] != 48:
            raise ValueError(
                f"V13 full-state actor requires 48D observations, got {observations.shape[-1]}."
            )
        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = _sample_flashsac_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )
        return actions.cpu().numpy()

    def update(self) -> dict[str, Any]:
        replay_info: dict[str, float] = {}
        if self._replay_mix_config is None:
            batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())
            for key, value in batch.items():
                batch[key] = value.to(self._device, non_blocking=True)
        else:
            if self._cfg.normalize_reward and not self._frozen_reward_normalizer:
                raise RuntimeError("Frozen reward normalizer has not been configured.")
            batch, replay_info, _source_ids = sample_mixed_replay_batch(
                config=self._replay_mix_config,
                observation_dim=self._critic_observation_dim,
                action_dim=self._action_dim,
                device=self._device,
                real_sampler=self._real_replay_sampler,
                rwm_buffer=self._replay_buffer if self.uses_rwm_replay else None,
                sim_sampler=self._sim_replay_sampler,
                generator=self._replay_mix_generator,
            )

        if batch["observation"].shape[-1] != 48:
            raise ValueError("V13 full-state replay observation must be 48D.")
        if batch["next_observation"].shape[-1] != 48:
            raise ValueError("V13 full-state replay next_observation must be 48D.")
        batch["actor_observation"] = batch["observation"]
        batch["actor_next_observation"] = batch["next_observation"]

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        actor_learning_starts_updates = self._actor_learning_starts_updates
        actor_update_enabled = self._update_step >= actor_learning_starts_updates
        do_actor_update = (
            actor_update_enabled
            and self._update_step % self._cfg.actor_update_period == 0
        )
        update_info_raw = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self._cfg,
            do_actor_update=do_actor_update,
            device=self._device,
            grad_scaler=self._grad_scaler,
        )
        self._update_step += 1

        update_info: dict[str, float] = {}
        for key, value in update_info_raw.items():
            if isinstance(value, torch.Tensor):
                update_info[key] = value.item()
            elif not isinstance(value, dict):
                update_info[key] = float(value)
        update_info.update(replay_info)
        if self._replay_mix_config is not None:
            update_info["Replay/normalizer_frozen"] = float(
                self._frozen_reward_normalizer
            )
        update_info["Warmup/actor_update_enabled"] = float(actor_update_enabled)
        update_info["Warmup/actor_update_performed"] = float(do_actor_update)
        update_info["Warmup/actor_learning_starts_updates"] = float(
            actor_learning_starts_updates
        )
        assert self._actor.optimizer is not None
        assert self._critic.optimizer is not None
        update_info["Schedule/actor_learning_rate_scale"] = float(
            self._actor_learning_rate_scale
        )
        update_info["Schedule/actor_learning_rate"] = float(
            self._actor.optimizer.param_groups[0]["lr"]
        )
        update_info["Schedule/critic_learning_rate"] = float(
            self._critic.optimizer.param_groups[0]["lr"]
        )
        return update_info


def create_go2_flashsac_fullstate_replay_agent(
    observation_space: gym.Space[NDArray],
    action_space: gym.Space[NDArray],
    cfg: FlashSACConfig,
) -> FlashSACAgent:
    observation_dim = int(observation_space.shape[-1])
    if observation_dim != 48:
        raise ValueError(
            f"V13 full-state policy requires observation_dim=48, got {observation_dim}."
        )
    env_info: dict[str, Any] = {}
    return FlashSACFullStateReplayAgent(
        observation_space,
        action_space,
        env_info,
        cfg,
    )
