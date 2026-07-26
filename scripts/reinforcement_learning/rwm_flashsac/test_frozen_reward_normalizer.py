from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).with_name("frozen_reward_normalizer.py")
REPO_ROOT = SCRIPT_PATH.resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SPEC = importlib.util.spec_from_file_location("frozen_reward_normalizer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


class FakeRMS:
    def __init__(self) -> None:
        self.mean = torch.zeros(1)
        self.var = torch.ones(1)
        self.count = torch.tensor(0.0)


class FakeNormalizer:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.G_r = torch.zeros(1)
        self.G_r_max = torch.zeros(1)
        self.G_rms = FakeRMS()


def test_discounted_returns_respect_episode_boundaries() -> None:
    rewards = torch.tensor([[1.0], [2.0], [0.0], [3.0], [4.0]])
    eligible = torch.tensor([[True], [True], [False], [True], [True]])
    dones = torch.tensor([[False], [False], [True], [False], [False]])
    episode_ids = torch.tensor([[10], [10], [10], [11], [11]])
    timesteps = torch.tensor([[0], [1], [2], [0], [1]])
    actual = M._discounted_return_samples(
        one_step_rewards=rewards,
        eligible=eligible,
        dones=dones,
        episode_ids=episode_ids,
        timesteps=timesteps,
        gamma=0.5,
    )
    torch.testing.assert_close(actual, torch.tensor([1.0, 2.5, 3.0, 5.5]))


def test_loader_checks_hashes_and_loads_state() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "normalizer.pt"
        torch.save(
            {
                "format_version": M.FORMAT_VERSION,
                "state": {
                    "G_r": torch.zeros(1),
                    "G_r_max": torch.tensor([2.0]),
                    "G_rms_mean": torch.tensor([0.5]),
                    "G_rms_var": torch.tensor([0.25]),
                    "G_rms_count": torch.tensor(10.0),
                },
                "metadata": {
                    "gamma": 0.99,
                    "normalized_G_max": 5.0,
                    "source_dataset_sha256": "dataset",
                    "reward_config_sha256": "reward",
                    "frozen": True,
                },
            },
            path,
        )
        normalizer = FakeNormalizer()
        metadata = M.load_frozen_reward_normalizer(
            path=path,
            normalizer=normalizer,
            expected_gamma=0.99,
            expected_normalized_g_max=5.0,
            expected_source_dataset_sha256="dataset",
            expected_reward_config_sha256="reward",
        )
        torch.testing.assert_close(normalizer.G_r_max, torch.tensor([2.0]))
        torch.testing.assert_close(normalizer.G_rms.var, torch.tensor([0.25]))
        assert metadata["artifact_sha256"]

        try:
            M.load_frozen_reward_normalizer(
                path=path,
                normalizer=normalizer,
                expected_gamma=0.99,
                expected_normalized_g_max=5.0,
                expected_source_dataset_sha256="wrong",
                expected_reward_config_sha256="reward",
            )
        except ValueError as error:
            assert "dataset hash" in str(error)
        else:
            raise AssertionError("Mismatched dataset hash was accepted.")


if __name__ == "__main__":
    test_discounted_returns_respect_episode_boundaries()
    print("test_discounted_returns_respect_episode_boundaries: PASS")
    test_loader_checks_hashes_and_loads_state()
    print("test_loader_checks_hashes_and_loads_state: PASS")
