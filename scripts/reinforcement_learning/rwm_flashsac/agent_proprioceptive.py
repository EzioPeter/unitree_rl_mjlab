"""FlashSAC agent with a proprioceptive actor observation."""

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
from scripts.reinforcement_learning.rwm_flashsac.world_model_env_proprioceptive import (
    PROPRIOCEPTIVE_ACTOR_OBS_DIM,
    proprioceptive_obs_t,
)


class FlashSACProprioceptiveAgent(FlashSACAgent):
    """Use full 48-dim RWM obs for critic and 45-dim proprioception for actor."""

    def sample_actions(
        self,
        interaction_step: int,
        prev_transition: MutableMapping[str, Tensor],
        training: bool,
    ) -> Tensor:
        del interaction_step
        temperature = 1.0 if training else 0.0
        observations = torch.as_tensor(
            prev_transition["next_observation"],
            dtype=torch.float32,
            device=self._device,
        )
        actor_observations = proprioceptive_obs_t(observations)

        with torch.no_grad():
            (
                self._cached_noise,
                actions,
                self._cur_noise_repeat_count,
                self._cur_noise_repeat_n,
            ) = _sample_flashsac_actions(
                actor=self._actor,
                noise=self._cached_noise,
                observations=actor_observations,
                temperature=temperature,
                cur_count=self._cur_noise_repeat_count,
                cur_n=self._cur_noise_repeat_n,
                zeta_cdf=self._zeta_cdf,
            )
        return actions.cpu().numpy()

    def update(self) -> dict[str, Any]:
        batch = cast(dict[str, torch.Tensor], self._replay_buffer.sample())
        for key, value in batch.items():
            batch[key] = value.to(self._device, non_blocking=True)

        batch["actor_observation"] = proprioceptive_obs_t(batch["observation"])
        batch["actor_next_observation"] = proprioceptive_obs_t(batch["next_observation"])

        if self._cfg.normalize_reward:
            assert self.reward_normalizer is not None
            batch["reward"] = self.reward_normalizer.normalize_rewards(batch["reward"])

        update_info_raw = _update_networks(
            batch=batch,
            actor=self._actor,
            critic=self._critic,
            target_critic=self._target_critic,
            temperature=self._temperature,
            cfg=self._cfg,
            do_actor_update=(self._update_step % self._cfg.actor_update_period == 0),
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
        return update_info


def create_go2_flashsac_proprioceptive_agent(
    observation_space: gym.Space[NDArray],
    action_space: gym.Space[NDArray],
    cfg: FlashSACConfig,
) -> FlashSACAgent:
    env_info: dict[str, Any] = {"actor_observation_size": (PROPRIOCEPTIVE_ACTOR_OBS_DIM,)}
    return FlashSACProprioceptiveAgent(observation_space, action_space, env_info, cfg)
