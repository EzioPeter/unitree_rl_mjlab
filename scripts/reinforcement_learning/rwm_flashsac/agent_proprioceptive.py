"""FlashSAC agent with a proprioceptive actor observation."""

from __future__ import annotations

import os
from dataclasses import asdict
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
from scripts.reinforcement_learning.rwm_flashsac.frozen_reward_normalizer import (
    load_frozen_reward_normalizer,
)
from scripts.reinforcement_learning.rwm_flashsac.replay_mixer import (
    ExternalReplaySampler,
    ReplayMixConfig,
    compute_source_counts,
    sample_mixed_replay_batch,
)
from scripts.reinforcement_learning.rwm_flashsac.world_model_env_proprioceptive import (
    proprioceptive_obs_t,
)


class FlashSACProprioceptiveAgent(FlashSACAgent):
    """Use full 48-dim RWM obs for critic and 45-dim proprioception for actor."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if "cfg" in kwargs:
            input_cfg = kwargs["cfg"]
        elif len(args) >= 4:
            input_cfg = args[3]
        else:
            raise TypeError("FlashSACProprioceptiveAgent requires a cfg argument.")
        # FlashSACAgent rebuilds its dataclass config with dataclasses.replace(),
        # which drops attributes added by V12's config adapter. Preserve this
        # V12-only scheduling option explicitly before calling the parent.
        self._actor_learning_starts_updates = int(
            getattr(input_cfg, "actor_learning_starts_updates", 0)
        )
        self._actor_learning_rate_scale = float(
            getattr(input_cfg, "actor_learning_rate_scale", 1.0)
        )
        if not 0.0 < self._actor_learning_rate_scale <= 1.0:
            raise ValueError(
                "actor_learning_rate_scale must be in (0, 1], got "
                f"{self._actor_learning_rate_scale}."
            )
        super().__init__(*args, **kwargs)
        assert self._actor.optimizer is not None
        for param_group in self._actor.optimizer.param_groups:
            param_group["lr"] = (
                float(param_group["lr"]) * self._actor_learning_rate_scale
            )
            if "initial_lr" in param_group:
                param_group["initial_lr"] = (
                    float(param_group["initial_lr"])
                    * self._actor_learning_rate_scale
                )
        if self._actor.scheduler is not None:
            self._actor.scheduler.base_lrs = [
                float(value) * self._actor_learning_rate_scale
                for value in self._actor.scheduler.base_lrs
            ]
            self._actor.scheduler._last_lr = [
                float(value) * self._actor_learning_rate_scale
                for value in self._actor.scheduler._last_lr
            ]
        self._replay_mix_config: ReplayMixConfig | None = None
        self._real_replay_sampler: ExternalReplaySampler | None = None
        self._sim_replay_sampler: ExternalReplaySampler | None = None
        self._replay_mix_generator = torch.Generator(device="cpu").manual_seed(
            int(self._cfg.seed) + 1701
        )
        self._frozen_reward_normalizer = False
        self._frozen_reward_normalizer_metadata: dict[str, Any] = {}

    def configure_replay_mix(
        self,
        *,
        config: ReplayMixConfig,
        real_sampler: ExternalReplaySampler | None,
        sim_sampler: ExternalReplaySampler | None,
        seed: int,
    ) -> None:
        counts = compute_source_counts(config)
        if counts.real and real_sampler is None:
            raise ValueError("Formal replay mix requires real_sampler.")
        if counts.sim and sim_sampler is None:
            raise ValueError("Configured simulator replay count requires sim_sampler.")
        self._replay_mix_config = config
        self._real_replay_sampler = real_sampler
        self._sim_replay_sampler = sim_sampler
        self._replay_mix_generator = torch.Generator(device="cpu").manual_seed(
            int(seed)
        )

    def configure_frozen_reward_normalizer(self, path: str) -> dict[str, Any]:
        if not self._cfg.normalize_reward or self.reward_normalizer is None:
            raise ValueError(
                "Frozen reward normalization requires agent.normalize_reward=true."
            )
        if self._real_replay_sampler is None:
            raise ValueError("Configure the real replay sampler before its normalizer.")
        source_dataset_sha256 = self._real_replay_sampler.metadata.get(
            "source_dataset_sha256"
        )
        reward_config_sha256 = self._real_replay_sampler.metadata.get(
            "reward_config_sha256"
        )
        if not isinstance(source_dataset_sha256, str) or not source_dataset_sha256:
            raise ValueError("Real replay metadata is missing source_dataset_sha256.")
        if not isinstance(reward_config_sha256, str) or not reward_config_sha256:
            raise ValueError("Real replay metadata is missing reward_config_sha256.")
        metadata = load_frozen_reward_normalizer(
            path=path,
            normalizer=self.reward_normalizer,
            expected_gamma=float(self._cfg.gamma),
            expected_normalized_g_max=float(self._cfg.normalized_G_max),
            expected_source_dataset_sha256=source_dataset_sha256,
            expected_reward_config_sha256=reward_config_sha256,
        )
        self._frozen_reward_normalizer = True
        self._frozen_reward_normalizer_metadata = metadata
        return dict(metadata)

    @property
    def uses_rwm_replay(self) -> bool:
        if self._replay_mix_config is None:
            return True
        configured = compute_source_counts(self._replay_mix_config).rwm > 0
        mutable_trace = bool(
            self._sim_replay_sampler is not None
            and getattr(self._sim_replay_sampler, "metadata", {}).get("mutable", False)
        )
        return configured or mutable_trace

    def process_transition(self, transition: MutableMapping[str, Tensor]) -> None:
        if self._replay_mix_config is None:
            super().process_transition(transition)
            return
        if self.uses_rwm_replay:
            self._replay_buffer.add(transition)
        if self._cfg.normalize_reward and not self._frozen_reward_normalizer:
            raise RuntimeError(
                "Formal mixed replay cannot update before the frozen normalizer is loaded."
            )

    def can_start_training(self) -> bool:
        if self._replay_mix_config is None:
            return super().can_start_training()
        counts = compute_source_counts(self._replay_mix_config)
        if counts.real and self._real_replay_sampler is None:
            return False
        if counts.sim and self._sim_replay_sampler is None:
            return False
        if (
            counts.sim
            and bool(getattr(self._sim_replay_sampler, "metadata", {}).get("mutable", False))
            and len(self._sim_replay_sampler) == 0
            and not self._replay_buffer.can_sample()
        ):
            return False
        if counts.rwm and not self._replay_buffer.can_sample():
            return False
        return True

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

        batch["actor_observation"] = proprioceptive_obs_t(batch["observation"])
        batch["actor_next_observation"] = proprioceptive_obs_t(batch["next_observation"])

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

    def save(self, path: str) -> None:
        super().save(path)
        if self._replay_mix_config is None:
            return
        state_path = os.path.join(path, "agent_state.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        state["exploration_state"] = {
            "cached_noise": self._cached_noise.detach().cpu(),
            "cur_noise_repeat_count": self._cur_noise_repeat_count.detach().cpu(),
            "cur_noise_repeat_n": self._cur_noise_repeat_n.detach().cpu(),
        }
        state["replay_mix_state"] = {
            "config": asdict(self._replay_mix_config),
            "mix_generator_state": self._replay_mix_generator.get_state(),
            "real_sampler_state": (
                self._real_replay_sampler.state_dict()
                if self._real_replay_sampler is not None
                else None
            ),
            "sim_sampler_state": (
                self._sim_replay_sampler.state_dict()
                if self._sim_replay_sampler is not None
                else None
            ),
            "real_replay_path": (
                str(self._real_replay_sampler.path)
                if self._real_replay_sampler is not None
                else None
            ),
            "sim_replay_path": (
                str(self._sim_replay_sampler.path)
                if self._sim_replay_sampler is not None
                else None
            ),
            "real_replay_sha256": (
                self._real_replay_sampler.sha256
                if self._real_replay_sampler is not None
                else None
            ),
            "sim_replay_sha256": (
                self._sim_replay_sampler.sha256
                if self._sim_replay_sampler is not None
                else None
            ),
            "frozen_reward_normalizer": self._frozen_reward_normalizer,
            "frozen_reward_normalizer_sha256": (
                self._frozen_reward_normalizer_metadata.get("artifact_sha256")
            ),
        }
        torch.save(state, state_path)

    def load(self, path: str) -> None:
        super().load(path)
        state_path = os.path.join(path, "agent_state.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        exploration_state = state.get("exploration_state")
        if isinstance(exploration_state, dict):
            self._cached_noise = torch.as_tensor(
                exploration_state["cached_noise"],
                device=self._device,
            ).clone()
            self._cur_noise_repeat_count = torch.as_tensor(
                exploration_state["cur_noise_repeat_count"],
                device=self._device,
            ).clone()
            self._cur_noise_repeat_n = torch.as_tensor(
                exploration_state["cur_noise_repeat_n"],
                device=self._device,
            ).clone()
        else:
            print(
                "[Go2-FlashSAC-RWM] legacy checkpoint has no exploration "
                "state; exact continuation is unavailable."
            )
        replay_state = state.get("replay_mix_state")
        if self._replay_mix_config is None:
            if replay_state is not None:
                raise ValueError(
                    "Checkpoint contains replay mixer state but current run does not."
                )
            return
        if not isinstance(replay_state, dict):
            raise ValueError("Formal replay-mix resume requires replay_mix_state.")
        if replay_state.get("config") != asdict(self._replay_mix_config):
            raise ValueError("Checkpoint replay mix configuration does not match.")
        if (
            self._real_replay_sampler is not None
            and replay_state.get("real_replay_path")
            != str(self._real_replay_sampler.path)
        ):
            raise ValueError("Checkpoint real replay path does not match.")
        if (
            self._sim_replay_sampler is not None
            and replay_state.get("sim_replay_path")
            != str(self._sim_replay_sampler.path)
        ):
            raise ValueError("Checkpoint simulator replay path does not match.")
        if (
            self._real_replay_sampler is not None
            and replay_state.get("real_replay_sha256")
            != self._real_replay_sampler.sha256
        ):
            raise ValueError("Checkpoint real replay hash does not match.")
        if (
            self._sim_replay_sampler is not None
            and replay_state.get("sim_replay_sha256")
            != self._sim_replay_sampler.sha256
        ):
            raise ValueError("Checkpoint simulator replay hash does not match.")
        self._replay_mix_generator.set_state(
            replay_state["mix_generator_state"].detach().cpu()
        )
        if self._real_replay_sampler is not None:
            self._real_replay_sampler.load_state_dict(
                replay_state["real_sampler_state"]
            )
        if self._sim_replay_sampler is not None:
            self._sim_replay_sampler.load_state_dict(
                replay_state["sim_sampler_state"]
            )
        expected_normalizer_hash = replay_state.get(
            "frozen_reward_normalizer_sha256"
        )
        actual_normalizer_hash = self._frozen_reward_normalizer_metadata.get(
            "artifact_sha256"
        )
        if expected_normalizer_hash != actual_normalizer_hash:
            raise ValueError("Checkpoint frozen reward normalizer hash does not match.")

def create_go2_flashsac_proprioceptive_agent(
    observation_space: gym.Space[NDArray],
    action_space: gym.Space[NDArray],
    cfg: FlashSACConfig,
) -> FlashSACAgent:
    obs_dim = int(observation_space.shape[-1])
    actor_obs_dim = obs_dim - 3
    if actor_obs_dim <= 0:
        raise ValueError(f"Expected RWM observation dim > 3, got {obs_dim}.")
    env_info: dict[str, Any] = {"actor_observation_size": (actor_obs_dim,)}
    return FlashSACProprioceptiveAgent(observation_space, action_space, env_info, cfg)
