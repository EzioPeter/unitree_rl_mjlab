from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import gymnasium as gym
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.agents.flashSAC.agent import FlashSACConfig
from scripts.reinforcement_learning.rwm_flashsac import agent_fullstate_replay as agent_module
from scripts.reinforcement_learning.rwm_flashsac.agent_fullstate_replay import (
    create_go2_flashsac_fullstate_replay_agent,
)
from scripts.reinforcement_learning.rwm_flashsac.replay_mixer import (
    ExternalReplaySampler,
    ReplayMixConfig,
)


def _config() -> FlashSACConfig:
    cfg = FlashSACConfig(
        seed=0,
        normalize_reward=False,
        normalized_G_max=5.0,
        asymmetric_observation=False,
        device_type="cpu",
        buffer_max_length=100,
        buffer_min_length=10,
        buffer_device_type="cpu",
        sample_batch_size=20,
        learning_rate_init=3.0e-4,
        learning_rate_peak=3.0e-4,
        learning_rate_end=1.5e-4,
        learning_rate_warmup_rate=0.0,
        learning_rate_warmup_step=0,
        learning_rate_decay_rate=1.0,
        learning_rate_decay_step=100,
        actor_num_blocks=1,
        actor_hidden_dim=16,
        actor_bc_alpha=0.0,
        actor_noise_zeta_mu=2.0,
        actor_noise_zeta_max=2,
        actor_update_period=2,
        critic_num_blocks=1,
        critic_hidden_dim=16,
        critic_num_bins=11,
        critic_min_v=-5.0,
        critic_max_v=5.0,
        critic_target_update_tau=0.01,
        temp_initial_value=0.01,
        temp_target_sigma=0.15,
        temp_target_entropy=0.0,
        gamma=0.99,
        n_step=3,
        use_compile=False,
        compile_mode="reduce-overhead",
        use_amp=False,
        load_optimizer=True,
        load_reward_normalizer=False,
    )
    setattr(cfg, "actor_learning_starts_updates", 0)
    setattr(cfg, "actor_learning_rate_scale", 1.0)
    return cfg


def _transition(count: int) -> dict[str, np.ndarray]:
    return {
        "observation": np.zeros((count, 48), dtype=np.float32),
        "action": np.zeros((count, 12), dtype=np.float32),
        "reward": np.zeros(count, dtype=np.float32),
        "terminated": np.zeros(count, dtype=np.float32),
        "truncated": np.zeros(count, dtype=np.float32),
        "next_observation": np.ones((count, 48), dtype=np.float32),
    }


def test_fullstate_actor_and_critic_use_48d_mixed_replay() -> None:
    observation_space = gym.spaces.Box(
        -np.inf,
        np.inf,
        shape=(48,),
        dtype=np.float32,
    )
    action_space = gym.spaces.Box(-1.0, 1.0, shape=(12,), dtype=np.float32)
    agent = create_go2_flashsac_fullstate_replay_agent(
        observation_space,
        action_space,
        _config(),
    )
    assert agent._actor_observation_dim == 48
    assert agent._critic_observation_dim == 48
    sampled_actions = agent.sample_actions(
        0,
        {"next_observation": np.zeros((2, 48), dtype=np.float32)},
        training=True,
    )
    assert sampled_actions.shape == (2, 12)

    with tempfile.TemporaryDirectory() as temporary:
        real_path = Path(temporary) / "real.pt"
        real = {
            "format_version": "go2_real_replay_v1",
            **{
                key: torch.from_numpy(value)
                for key, value in _transition(40).items()
            },
            "metadata": {
                "gamma": 0.99,
                "n_step": 3,
                "source_dataset_sha256": "dataset",
                "reward_config_sha256": "reward",
            },
        }
        torch.save(real, real_path)
        real_sampler = ExternalReplaySampler(
            real_path,
            seed=1,
            expected_observation_dim=48,
            expected_action_dim=12,
            expected_gamma=0.99,
            expected_n_step=3,
            expected_source="real",
        )
        agent.configure_replay_mix(
            config=ReplayMixConfig(
                batch_size=20,
                real_ratio=0.05,
                synthetic_mode="rwm",
                trace_ratio_within_synthetic=0.0,
            ),
            real_sampler=real_sampler,
            sim_sampler=None,
            seed=2,
        )
        for _ in range(21):
            agent.process_transition(_transition(1))
        assert agent.can_start_training()

        captured: dict[str, torch.Tensor] = {}
        original_update = agent_module._update_networks

        def _capture_update(**kwargs: object) -> dict[str, torch.Tensor]:
            batch = kwargs["batch"]
            assert isinstance(batch, dict)
            captured.update(batch)
            return {"critic/loss": torch.tensor(0.5)}

        agent_module._update_networks = _capture_update
        try:
            info = agent.update()
        finally:
            agent_module._update_networks = original_update

        assert captured["observation"].shape == (20, 48)
        assert captured["actor_observation"].shape == (20, 48)
        assert captured["actor_next_observation"].shape == (20, 48)
        torch.testing.assert_close(
            captured["actor_observation"],
            captured["observation"],
        )
        torch.testing.assert_close(
            captured["actor_next_observation"],
            captured["next_observation"],
        )
        assert info["Replay/real_count"] == 1.0
        assert info["Replay/rwm_count"] == 19.0
        assert info["Replay/sim_count"] == 0.0
        assert info["critic/loss"] == 0.5


if __name__ == "__main__":
    test_fullstate_actor_and_critic_use_48d_mixed_replay()
    print("test_fullstate_actor_and_critic_use_48d_mixed_replay: PASS")
