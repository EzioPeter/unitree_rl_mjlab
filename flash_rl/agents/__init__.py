from typing import Any

import gymnasium as gym
from omegaconf import OmegaConf

from flash_rl.agents.base_agent import BaseAgent
from flash_rl.types import NDArray


def create_agent(
    observation_space: gym.spaces.Space[NDArray],
    action_space: gym.spaces.Space[NDArray],
    env_info: dict[str, Any],
    cfg: Any,
) -> BaseAgent[Any]:
    cfg_dict = OmegaConf.to_container(cfg, throw_on_missing=True, resolve=True)
    if not isinstance(cfg_dict, dict):
        raise ValueError("cfg must be a dictionary")
    cfg_dict = {str(k): v for k, v in cfg_dict.items()}
    agent_type = cfg_dict.pop("agent_type")

    agent: BaseAgent[Any]

    if agent_type == "flashSAC":
        from flash_rl.agents.flashSAC.agent import (
            FlashSACAgent,
            FlashSACConfig,
        )

        config = FlashSACConfig(**cfg_dict)  # type: ignore
        agent = FlashSACAgent(observation_space, action_space, env_info, config)

    else:
        raise NotImplementedError(f"Only flashSAC is supported, got agent_type={agent_type!r}.")

    return agent
