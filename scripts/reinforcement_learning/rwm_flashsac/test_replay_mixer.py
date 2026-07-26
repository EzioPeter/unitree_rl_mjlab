from __future__ import annotations

import importlib.util
import sys
import tempfile
from pathlib import Path

import torch


SCRIPT_PATH = Path(__file__).with_name("replay_mixer.py")
SPEC = importlib.util.spec_from_file_location("replay_mixer", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
M = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = M
SPEC.loader.exec_module(M)


def _batch(count: int, marker: float, obs_dim: int = 48, action_dim: int = 12):
    return {
        "observation": torch.full((count, obs_dim), marker),
        "action": torch.full((count, action_dim), marker / 10),
        "reward": torch.full((count,), marker),
        "terminated": torch.zeros(count),
        "truncated": torch.zeros(count),
        "next_observation": torch.full((count, obs_dim), marker + 0.5),
    }


class FakeRWMBuffer:
    def __init__(self, size: int = 100):
        self.size = size

    def __len__(self):
        return self.size

    def sample(self, sample_idxs=None):
        assert sample_idxs is not None
        return _batch(len(sample_idxs), 2.0)


class FakeExternalSampler:
    def __init__(self, marker: float):
        self.marker = marker

    def sample(self, count: int, *, device):
        return {key: value.to(device) for key, value in _batch(count, self.marker).items()}


class FakeMutableSampler(FakeExternalSampler):
    metadata = {"mutable": True}

    def __init__(self, marker: float, size: int):
        super().__init__(marker)
        self.size = size

    def __len__(self):
        return self.size


def test_formal_counts() -> None:
    cases = [
        (M.ReplayMixConfig(2048, 0.05, "rwm", 0.0), (102, 1946, 0)),
        (M.ReplayMixConfig(2048, 0.05, "mixed", 0.10), (102, 1752, 194)),
        (M.ReplayMixConfig(2048, 0.05, "mixed", 0.25), (102, 1460, 486)),
        (M.ReplayMixConfig(2048, 0.05, "trace", 1.0), (102, 0, 1946)),
    ]
    for config, expected in cases:
        counts = M.compute_source_counts(config)
        assert (counts.real, counts.rwm, counts.sim) == expected
        assert counts.total == 2048


def test_invalid_mode_ratios_fail_closed() -> None:
    invalid = [
        M.ReplayMixConfig(32, 0.05, "rwm", 0.1),
        M.ReplayMixConfig(32, 0.05, "trace", 0.9),
        M.ReplayMixConfig(32, 0.05, "mixed", 0.0),
        M.ReplayMixConfig(32, 0.05, "mixed", 1.0),
    ]
    for config in invalid:
        try:
            config.validate()
        except ValueError:
            pass
        else:
            raise AssertionError(f"Invalid config did not fail: {config}")


def test_mixed_batch_exact_composition_and_shuffle() -> None:
    config = M.ReplayMixConfig(20, 0.05, "mixed", 0.25, shuffle=True)
    batch, info, source_ids = M.sample_mixed_replay_batch(
        config=config,
        observation_dim=48,
        action_dim=12,
        device="cpu",
        real_sampler=FakeExternalSampler(1.0),
        rwm_buffer=FakeRWMBuffer(),
        sim_sampler=FakeExternalSampler(3.0),
        generator=torch.Generator().manual_seed(123),
    )
    # floor(20*0.05)=1 real; floor(19*0.25)=4 sim; 15 RWM.
    assert torch.bincount(source_ids, minlength=3).tolist() == [1, 15, 4]
    assert (batch["reward"] == 1).sum() == 1
    assert (batch["reward"] == 2).sum() == 15
    assert (batch["reward"] == 3).sum() == 4
    assert info["Replay/real_count"] == 1
    assert info["Replay/rwm_count"] == 15
    assert info["Replay/sim_count"] == 4
    # Shuffling must prevent source blocks from remaining in their input order.
    assert source_ids.tolist() != [0] + [1] * 15 + [2] * 4


def test_pure_trace_does_not_require_rwm() -> None:
    config = M.ReplayMixConfig(20, 0.05, "trace", 1.0)
    _batch_out, info, source_ids = M.sample_mixed_replay_batch(
        config=config,
        observation_dim=48,
        action_dim=12,
        device="cpu",
        real_sampler=FakeExternalSampler(1.0),
        rwm_buffer=None,
        sim_sampler=FakeExternalSampler(3.0),
        generator=torch.Generator().manual_seed(1),
    )
    assert info["Replay/rwm_count"] == 0
    assert sorted(set(source_ids.tolist())) == [0, 2]


def test_empty_mutable_trace_uses_explicit_rwm_warmup() -> None:
    config = M.ReplayMixConfig(20, 0.05, "mixed", 0.25, shuffle=False)
    _batch_out, info, source_ids = M.sample_mixed_replay_batch(
        config=config,
        observation_dim=48,
        action_dim=12,
        device="cpu",
        real_sampler=FakeExternalSampler(1.0),
        rwm_buffer=FakeRWMBuffer(),
        sim_sampler=FakeMutableSampler(3.0, size=0),
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.bincount(source_ids, minlength=3).tolist() == [1, 19, 0]
    assert info["Replay/trace_warmup"] == 1.0
    assert info["Replay/sim_ratio_actual"] == 0.0

    _batch_out, info, source_ids = M.sample_mixed_replay_batch(
        config=config,
        observation_dim=48,
        action_dim=12,
        device="cpu",
        real_sampler=FakeExternalSampler(1.0),
        rwm_buffer=FakeRWMBuffer(),
        sim_sampler=FakeMutableSampler(3.0, size=3),
        generator=torch.Generator().manual_seed(1),
    )
    assert torch.bincount(source_ids, minlength=3).tolist() == [1, 15, 4]
    assert info["Replay/trace_warmup"] == 0.0


def test_missing_required_source_fails() -> None:
    config = M.ReplayMixConfig(20, 0.05, "rwm", 0.0)
    try:
        M.sample_mixed_replay_batch(
            config=config,
            observation_dim=48,
            action_dim=12,
            device="cpu",
            real_sampler=FakeExternalSampler(1.0),
            rwm_buffer=None,
            sim_sampler=None,
            generator=torch.Generator().manual_seed(1),
        )
    except ValueError as error:
        assert "rwm_buffer is required" in str(error)
    else:
        raise AssertionError("Missing RWM buffer did not fail.")


def test_external_sampler_metadata_and_kind() -> None:
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "real.pt"
        torch.save(
            {
                "format_version": "go2_real_replay_v1",
                **_batch(10, 1.0),
                "metadata": {"gamma": 0.99, "n_step": 3},
            },
            path,
        )
        sampler = M.ExternalReplaySampler(
            path,
            seed=7,
            expected_observation_dim=48,
            expected_action_dim=12,
            expected_gamma=0.99,
            expected_n_step=3,
            expected_source="real",
        )
        assert sampler.num_transitions == 10
        assert sampler.sample(4, device="cpu")["reward"].shape == (4,)
        try:
            M.ExternalReplaySampler(
                path,
                seed=7,
                expected_observation_dim=48,
                expected_action_dim=12,
                expected_gamma=0.99,
                expected_n_step=3,
                expected_source="sim",
            )
        except ValueError as error:
            assert "TRACE replay" in str(error)
        else:
            raise AssertionError("Real artifact was accepted as simulator replay.")

        trace_path = Path(temp) / "trace.pt"
        torch.save(
            {
                "format_version": "go2_trace_replay_v2",
                **_batch(11, 3.0),
                "metadata": {"gamma": 0.99, "n_step": 3},
            },
            trace_path,
        )
        trace_sampler = M.ExternalReplaySampler(
            trace_path,
            seed=8,
            expected_observation_dim=48,
            expected_action_dim=12,
            expected_gamma=0.99,
            expected_n_step=3,
            expected_source="sim",
        )
        assert trace_sampler.num_transitions == 11
        assert trace_sampler.sample(5, device="cpu")["reward"].shape == (5,)


if __name__ == "__main__":
    tests = [
        test_formal_counts,
        test_invalid_mode_ratios_fail_closed,
        test_mixed_batch_exact_composition_and_shuffle,
        test_pure_trace_does_not_require_rwm,
        test_empty_mutable_trace_uses_explicit_rwm_warmup,
        test_missing_required_source_fails,
        test_external_sampler_metadata_and_kind,
    ]
    for test in tests:
        test()
        print(f"{test.__name__}: PASS")
