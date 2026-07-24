"""Thin construction helpers for reusing the existing FlashSACAgent."""

from __future__ import annotations

from typing import Any

import gymnasium as gym

from flash_rl.agents.flashSAC.agent import FlashSACAgent, FlashSACConfig
from flash_rl.types import NDArray


def create_go2_flashsac_agent(
    observation_space: gym.Space[NDArray],
    action_space: gym.Space[NDArray],
    cfg: FlashSACConfig,
) -> FlashSACAgent:
    """Create an existing FlashSACAgent for 48-dim Go2 RWM observations."""

    env_info: dict[str, Any] = {}
    return FlashSACAgent(observation_space, action_space, env_info, cfg)
