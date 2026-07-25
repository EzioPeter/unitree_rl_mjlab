"""Build and load the fixed reward normalizer used by formal V12 routes."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac.build_go2_real_replay import (  # noqa: E402
    _canonical_sha256,
    _recompute_one_step_rewards,
    _sha256_file,
    _stack_time_key,
    _validate_dataset_layout,
)
from scripts.reinforcement_learning.rwm_flashsac.world_model_env import (  # noqa: E402
    FlashSACWorldModelEnvConfig,
)


FORMAT_VERSION = "go2_frozen_reward_normalizer_v1"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-path", required=True)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--condition", required=True)
    parser.add_argument("--dataset-size", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _discounted_return_samples(
    *,
    one_step_rewards: torch.Tensor,
    eligible: torch.Tensor,
    dones: torch.Tensor,
    episode_ids: torch.Tensor,
    timesteps: torch.Tensor,
    gamma: float,
) -> torch.Tensor:
    """Compute source-ordered return statistics without using auto-reset rows."""

    time_steps, num_envs = one_step_rewards.shape
    samples: list[torch.Tensor] = []
    for env_id in range(num_envs):
        running_return = torch.zeros((), dtype=torch.float32)
        previous_episode_id: int | None = None
        for time_index in range(time_steps):
            episode_id = int(episode_ids[time_index, env_id])
            timestep = int(timesteps[time_index, env_id])
            if timestep == 0 or (
                previous_episode_id is not None and episode_id != previous_episode_id
            ):
                running_return.zero_()
            if bool(eligible[time_index, env_id]):
                running_return = (
                    float(gamma) * running_return
                    + one_step_rewards[time_index, env_id].float()
                )
                samples.append(running_return.clone())
            if bool(dones[time_index, env_id]):
                running_return.zero_()
            previous_episode_id = episode_id
    if not samples:
        raise ValueError("No eligible return samples were produced.")
    return torch.stack(samples).contiguous()


def build_frozen_reward_normalizer(
    *,
    dataset: dict[str, Any],
    config: Any,
    condition: str,
    dataset_size: str,
    source_dataset_path: Path,
    source_dataset_sha256: str,
    config_path: Path,
    config_sha256: str,
    device: torch.device,
    allowed_collector_types: tuple[int, ...] = (1,),
) -> dict[str, Any]:
    required = (
        "states",
        "actions",
        "next_states",
        "contacts",
        "observations",
        "next_observations",
        "commands",
        "dones",
        "timeouts",
        "prev_actions",
        "episode_ids",
        "timesteps",
        "collector_types",
        "trace_valid_masks",
    )
    tensors = {key: _stack_time_key(dataset, key) for key in required}
    _validate_dataset_layout(tensors)
    allowed_collectors = tuple(int(value) for value in allowed_collector_types)
    if not allowed_collectors:
        raise ValueError("allowed_collector_types must not be empty.")
    collector_types = tensors["collector_types"].long()
    collector_allowed = torch.zeros_like(collector_types, dtype=torch.bool)
    for collector_type in allowed_collectors:
        collector_allowed |= collector_types == collector_type
    if not bool(collector_allowed.all()):
        actual = sorted(int(value) for value in torch.unique(collector_types).tolist())
        raise ValueError(
            "Reward-normalizer source contains collector types outside the "
            f"explicitly allowed set: actual={actual}, "
            f"allowed={sorted(allowed_collectors)}."
        )
    if not bool(tensors["trace_valid_masks"].bool().all()):
        raise ValueError("Formal V12 normalizer requires trace-valid source transitions.")

    world_model = OmegaConf.to_container(
        config.world_model,
        resolve=True,
        throw_on_missing=True,
    )
    if not isinstance(world_model, dict):
        raise TypeError("config.world_model must resolve to a mapping.")
    wm_cfg = FlashSACWorldModelEnvConfig(**world_model)
    gamma = float(config.agent.gamma)
    normalized_g_max = float(config.agent.normalized_G_max)
    dones = tensors["dones"].bool()
    eligible = ~dones
    one_step_rewards, _reward_terms = _recompute_one_step_rewards(
        tensors,
        wm_cfg,
        eligible=eligible,
        device=device,
    )
    returns = _discounted_return_samples(
        one_step_rewards=one_step_rewards,
        eligible=eligible,
        dones=dones,
        episode_ids=tensors["episode_ids"].long(),
        timesteps=tensors["timesteps"].long(),
        gamma=gamma,
    )
    reward_config = {
        key: value
        for key, value in asdict(wm_cfg).items()
        if key.startswith("reward_")
        or key in {"step_dt", "uncertainty_penalty_weight"}
    }
    state = {
        "G_r": torch.zeros(1, dtype=torch.float32),
        "G_r_max": returns.abs().max().reshape(1).float(),
        "G_rms_mean": returns.mean().reshape(1).float(),
        "G_rms_var": returns.var(unbiased=False).reshape(1).float(),
        "G_rms_count": torch.tensor(float(returns.numel()), dtype=torch.float32),
    }
    metadata = {
        "condition": str(condition),
        "dataset_size": str(dataset_size),
        "source_dataset_path": str(source_dataset_path),
        "source_dataset_sha256": str(source_dataset_sha256),
        "config_path": str(config_path),
        "config_sha256": str(config_sha256),
        "reward_config": reward_config,
        "reward_config_sha256": _canonical_sha256(reward_config),
        "reward_version": str(wm_cfg.reward_version),
        "gamma": gamma,
        "normalized_G_max": normalized_g_max,
        "return_sample_count": int(returns.numel()),
        "return_mean": float(returns.mean()),
        "return_variance": float(returns.var(unbiased=False)),
        "return_abs_max": float(returns.abs().max()),
        "source_collector_types": sorted(
            int(value) for value in torch.unique(collector_types).tolist()
        ),
        "allowed_collector_types": sorted(allowed_collectors),
        "excluded_autoreset_done_transitions": int(dones.sum()),
        "source_reward_used": False,
        "frozen": True,
    }
    for key, value in state.items():
        if not torch.isfinite(value).all():
            raise ValueError(f"Frozen normalizer state {key!r} contains NaN/Inf.")
    return {
        "format_version": FORMAT_VERSION,
        "state": state,
        "metadata": metadata,
    }


def load_frozen_reward_normalizer(
    *,
    path: str | Path,
    normalizer: Any,
    expected_gamma: float,
    expected_normalized_g_max: float,
    expected_source_dataset_sha256: str,
    expected_reward_config_sha256: str,
) -> dict[str, Any]:
    checkpoint_path = Path(path).expanduser().resolve()
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if artifact.get("format_version") != FORMAT_VERSION:
        raise ValueError(f"Unsupported frozen normalizer artifact: {checkpoint_path}")
    metadata = dict(artifact.get("metadata") or {})
    checks = {
        "gamma": (metadata.get("gamma"), float(expected_gamma)),
        "normalized_G_max": (
            metadata.get("normalized_G_max"),
            float(expected_normalized_g_max),
        ),
    }
    for name, (actual, expected) in checks.items():
        if actual is None or abs(float(actual) - expected) > 1.0e-9:
            raise ValueError(
                f"Frozen normalizer {name}={actual!r} does not match expected {expected}."
            )
    if metadata.get("source_dataset_sha256") != expected_source_dataset_sha256:
        raise ValueError("Frozen normalizer source dataset hash does not match real replay.")
    if metadata.get("reward_config_sha256") != expected_reward_config_sha256:
        raise ValueError("Frozen normalizer reward config hash does not match real replay.")
    if not bool(metadata.get("frozen")):
        raise ValueError("Reward normalizer artifact is not marked frozen.")

    state = artifact.get("state")
    if not isinstance(state, dict):
        raise ValueError("Frozen normalizer artifact is missing state.")
    required = ("G_r", "G_r_max", "G_rms_mean", "G_rms_var", "G_rms_count")
    for key in required:
        value = state.get(key)
        if not isinstance(value, torch.Tensor) or not torch.isfinite(value).all():
            raise ValueError(f"Invalid frozen normalizer tensor {key!r}.")
    device = normalizer.device
    normalizer.G_r = state["G_r"].detach().to(device=device, dtype=torch.float32)
    normalizer.G_r_max = state["G_r_max"].detach().to(
        device=device,
        dtype=torch.float32,
    )
    normalizer.G_rms.mean = state["G_rms_mean"].detach().to(
        device=device,
        dtype=torch.float32,
    )
    normalizer.G_rms.var = state["G_rms_var"].detach().to(
        device=device,
        dtype=torch.float32,
    )
    normalizer.G_rms.count = state["G_rms_count"].detach().to(
        device=device,
        dtype=torch.float32,
    )
    metadata["artifact_path"] = str(checkpoint_path)
    metadata["artifact_sha256"] = _sha256_file(checkpoint_path)
    return metadata


def main() -> None:
    args = _parse_args()
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    config_path = Path(args.config_path).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(f"Refusing to overwrite existing artifact: {output_path}")
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    config = OmegaConf.load(config_path)
    OmegaConf.resolve(config)
    artifact = build_frozen_reward_normalizer(
        dataset=dataset,
        config=config,
        condition=args.condition,
        dataset_size=args.dataset_size,
        source_dataset_path=dataset_path,
        source_dataset_sha256=_sha256_file(dataset_path),
        config_path=config_path,
        config_sha256=_sha256_file(config_path),
        device=torch.device(args.device),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    torch.save(artifact, temporary_path)
    temporary_path.replace(output_path)
    print(json.dumps(artifact["metadata"], indent=2, sort_keys=True))
    print(f"[V12 frozen normalizer] wrote {output_path}")
    print(f"[V12 frozen normalizer] sha256={_sha256_file(output_path)}")


if __name__ == "__main__":
    main()
