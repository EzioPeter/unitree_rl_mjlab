from __future__ import annotations

from collections import deque
from typing import Any, Union

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector import VectorEnv
from gymnasium.vector.utils import batch_space

from ..types import F32NDArray, NDArray


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
) -> None:
    """Apply lightweight randomization switches to an mjlab env config."""
    if not use_domain_randomization:
        for event_name in ("foot_friction", "encoder_bias", "base_com"):
            env_cfg.events.pop(event_name, None)

    if not use_push_randomization:
        env_cfg.events.pop("push_robot", None)

    if not use_observation_noise:
        for group_name in ("actor", "critic"):
            obs_group = env_cfg.observations.get(group_name)
            if obs_group is not None:
                obs_group.enable_corruption = False


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
) -> MjlabVectorEnv:
    return MjlabVectorEnv(
        task_id=task_id,
        num_envs=num_envs,
        seed=seed,
        device=device,
        use_domain_randomization=use_domain_randomization,
        use_push_randomization=use_push_randomization,
        use_observation_noise=use_observation_noise,
        action_mask_indices=action_mask_indices,
        use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
    )
