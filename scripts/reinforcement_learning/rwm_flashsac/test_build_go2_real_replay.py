from __future__ import annotations

import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from flash_rl.buffers.torch_buffer import TorchUniformBuffer


SCRIPT_PATH = Path(__file__).with_name("build_go2_real_replay.py")
SPEC = importlib.util.spec_from_file_location("build_go2_real_replay", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_n_step_matches_torch_uniform_buffer() -> None:
    gamma = 0.9
    n_step = 3
    time_steps = 7
    observations = torch.arange(time_steps * 2, dtype=torch.float32).reshape(time_steps, 1, 2)
    next_observations = observations + 0.5
    actions = torch.arange(time_steps, dtype=torch.float32).reshape(time_steps, 1, 1) / 10
    rewards = torch.arange(1, time_steps + 1, dtype=torch.float32).reshape(time_steps, 1)
    terminated = torch.tensor([[0], [0], [1], [0], [0], [1], [0]], dtype=torch.bool)
    truncated = torch.zeros_like(terminated)
    episode_ids = torch.tensor([[10], [10], [10], [11], [11], [11], [12]])
    timesteps = torch.tensor([[0], [1], [2], [0], [1], [2], [0]])

    offline = MODULE._build_n_step(
        observations=observations,
        actions=actions,
        rewards=rewards,
        terminated=terminated,
        truncated=truncated,
        next_observations=next_observations,
        episode_ids=episode_ids,
        timesteps=timesteps,
        eligible=torch.ones_like(terminated),
        gamma=gamma,
        n_step=n_step,
    )
    # The final non-terminal episode has insufficient lookahead and is omitted.
    assert offline["reward"].shape[0] == 6

    buffer = TorchUniformBuffer(
        observation_space=gym.spaces.Box(-np.inf, np.inf, shape=(2,), dtype=np.float32),
        action_space=gym.spaces.Box(-1, 1, shape=(1,), dtype=np.float32),
        n_step=n_step,
        gamma=gamma,
        max_length=32,
        min_length=1,
        sample_batch_size=1,
        device_type="cpu",
    )
    # Add two dummy rows after the second terminal so TorchUniformBuffer flushes
    # the terminal tail, exactly as a continuing vector environment would.
    for t in range(time_steps + 2):
        src = min(t, time_steps - 1)
        transition = {
            "observation": observations[src].numpy(),
            "action": actions[src].numpy(),
            "reward": rewards[src].numpy(),
            "terminated": terminated[src].numpy() if t < time_steps else np.array([False]),
            "truncated": truncated[src].numpy(),
            "next_observation": next_observations[src].numpy(),
        }
        buffer.add(transition)

    assert len(buffer) >= 6
    np.testing.assert_allclose(buffer._observations[:6].numpy(), offline["observation"].numpy())
    np.testing.assert_allclose(buffer._actions[:6].numpy(), offline["action"].numpy())
    np.testing.assert_allclose(buffer._rewards[:6].numpy(), offline["reward"].numpy(), rtol=1e-6)
    np.testing.assert_allclose(
        buffer._next_observations[:6].numpy(),
        offline["next_observation"].numpy(),
    )
    np.testing.assert_allclose(buffer._terminateds[:6].numpy(), offline["terminated"].numpy())
    np.testing.assert_allclose(buffer._truncateds[:6].numpy(), offline["truncated"].numpy())


def test_n_step_rejects_nonterminal_boundary_without_lookahead() -> None:
    observations = torch.zeros(3, 1, 2)
    try:
        MODULE._build_n_step(
            observations=observations,
            actions=torch.zeros(3, 1, 1),
            rewards=torch.ones(3, 1),
            terminated=torch.zeros(3, 1, dtype=torch.bool),
            truncated=torch.zeros(3, 1, dtype=torch.bool),
            next_observations=observations,
            episode_ids=torch.tensor([[1], [2], [2]]),
            timesteps=torch.tensor([[0], [0], [1]]),
            eligible=torch.ones(3, 1, dtype=torch.bool),
            gamma=0.99,
            n_step=3,
        )
    except ValueError as error:
        assert "No valid n-step transitions" in str(error)
    else:
        raise AssertionError("Expected a dataset-boundary failure.")


def test_ineligible_done_row_is_not_used() -> None:
    observations = torch.arange(8, dtype=torch.float32).reshape(4, 1, 2)
    terminated = torch.zeros(4, 1, dtype=torch.bool)
    truncated = torch.tensor([[False], [False], [False], [True]])
    eligible = ~truncated
    result = MODULE._build_n_step(
        observations=observations,
        actions=torch.zeros(4, 1, 1),
        rewards=torch.ones(4, 1),
        terminated=terminated,
        truncated=truncated,
        next_observations=observations + 1,
        episode_ids=torch.ones(4, 1, dtype=torch.long),
        timesteps=torch.arange(4).reshape(4, 1),
        eligible=eligible,
        gamma=0.99,
        n_step=3,
    )
    # Only t=0 has three valid transitions; windows depending on the auto-reset
    # done row are rejected.
    assert result["source_timestep"].tolist() == [0]


if __name__ == "__main__":
    test_n_step_matches_torch_uniform_buffer()
    print("test_n_step_matches_torch_uniform_buffer: PASS")
    test_n_step_rejects_nonterminal_boundary_without_lookahead()
    print("test_n_step_rejects_nonterminal_boundary_without_lookahead: PASS")
    test_ineligible_done_row_is_not_used()
    print("test_ineligible_done_row_is_not_used: PASS")
