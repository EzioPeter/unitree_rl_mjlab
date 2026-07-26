"""Write a fail-closed launch manifest for one V13 mixed-data condition."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


EXPECTED_CONDITIONS = {
    "g0": {"rr_calf_strength": 1.0, "payload_mass_kg": 0.0},
    "rr05": {"rr_calf_strength": 0.5, "payload_mass_kg": 0.0},
    "rr03": {"rr_calf_strength": 0.3, "payload_mass_kg": 0.0},
    "p5": {"rr_calf_strength": 1.0, "payload_mass_kg": 5.0},
    "p75": {"rr_calf_strength": 1.0, "payload_mass_kg": 7.5},
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo_root", required=True)
    parser.add_argument("--expert_policy_path", required=True)
    parser.add_argument("--condition_id", required=True, choices=tuple(EXPECTED_CONDITIONS))
    parser.add_argument("--rr_calf_strength", required=True, type=float)
    parser.add_argument("--payload_mass_kg", required=True, type=float)
    parser.add_argument("--device", required=True)
    parser.add_argument("--raw_dataset", required=True)
    parser.add_argument("--code_path", action="append", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    return {
        "path": str(resolved),
        "size_bytes": resolved.stat().st_size,
        "sha256": _sha256(resolved),
    }


def _git_record(repo_root: Path) -> dict[str, Any]:
    if not (repo_root / ".git").exists():
        return {
            "available": False,
            "reason": "not_a_git_worktree",
            "commit": None,
            "status_porcelain": None,
        }

    def run(*args: str) -> str:
        return subprocess.run(
            ["git", "-C", str(repo_root), *args],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()

    return {
        "available": True,
        "commit": run("rev-parse", "HEAD"),
        "status_porcelain": run("status", "--porcelain"),
    }


def main() -> None:
    args = _parse_args()
    output = Path(args.output).expanduser().resolve()
    repo_root = Path(args.repo_root).expanduser().resolve()
    expert_root = Path(args.expert_policy_path).expanduser().resolve()
    raw_dataset = Path(args.raw_dataset).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite launch manifest: {output}")
    if raw_dataset.exists():
        raise FileExistsError(f"Raw dataset already exists: {raw_dataset}")

    expected_target = EXPECTED_CONDITIONS[str(args.condition_id)]
    actual_target = {
        "rr_calf_strength": float(args.rr_calf_strength),
        "payload_mass_kg": float(args.payload_mass_kg),
    }
    if actual_target != expected_target:
        raise ValueError(
            f"Condition {args.condition_id} target {actual_target} != {expected_target}."
        )
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be explicitly set for formal collection.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    device = torch.device(str(args.device))
    gpu_name = torch.cuda.get_device_name(device)
    if "3090" not in gpu_name:
        raise RuntimeError(f"V13 collection requires an RTX 3090, got {gpu_name!r}.")

    artifacts = {
        "expert_actor": _file_record(expert_root / "actor.pt"),
        "expert_config": _file_record(expert_root / "flashsac_config.yaml"),
    }
    code = [_file_record(Path(path)) for path in args.code_path]
    manifest = {
        "schema": "go2_v13_mixed_collection_launch_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned_pre_collection",
        "experiment_id": f"v13_mixed_1024x1000_{args.condition_id}_seed0",
        "objective": "Collect a snapshot-enabled mixed simulator source pool for V13 Sim-side RWM and TRACE.",
        "reference": {
            "repository": "EzioPeter/unitree_rl_mjlab",
            "branch": "model_based",
            "commit": "daa2eae7324255d4011964d2e74199547f347f26",
        },
        "intentional_differences_from_reference": [
            "Collect 1024x1000 rectangular transitions instead of nominal 1M.",
            "Collect each of g0, rr05, rr03, p5 and p75 independently.",
            "Save exact-reset simulator snapshots required by TRACE.",
            "Use the currently frozen E0 expert artifact recorded below.",
        ],
        "invariants": {
            "collector_mix": {
                "expert": 0.45,
                "noisy_expert": 0.25,
                "medium": 0.10,
                "failure_border": 0.15,
                "random": 0.05,
            },
            "medium_policy_path": None,
            "medium_semantics": "expert_action_plus_gaussian_noise_std_0.25",
            "collector_action_noise_std": {
                "noisy_expert": 0.12,
                "medium": 0.25,
                "failure_border": 0.45,
            },
            "num_envs": 1024,
            "time_steps": 1000,
            "num_transitions": 1_024_000,
            "stored_collector_observation_dim": 45,
            "stored_collector_observation_semantics": (
                "legacy expert-policy view; full 48D policy observations are reconstructed "
                "from state, command and prev_action"
            ),
            "collector_live_actor_observation_dim": 45,
            "collector_live_critic_observation_dim": 48,
            "rwm_state_dim": 45,
            "rwm_input_state_dim": 45,
            "rwm_output_state_dim": 45,
            "rwm_base_lin_vel_in_input": True,
            "rwm_base_lin_vel_supervised": True,
            "downstream_actor_observation_dim": 48,
            "downstream_critic_observation_dim": 48,
            "downstream_actor_contains_base_lin_vel": True,
            "downstream_critic_contains_base_lin_vel": True,
            "save_trace_snapshots": True,
            "domain_randomization": False,
            "push_randomization": False,
            "observation_noise": False,
            "action_interface_noise_bias_delay": False,
            "command_modes": [
                "stand",
                "pure_x",
                "pure_y",
                "pure_yaw",
                "xy",
                "x_yaw",
                "y_yaw",
                "xy_yaw",
            ],
            "command_mode_weights": {
                "stand": 0.08,
                "pure_x": 0.25,
                "pure_y": 0.10,
                "pure_yaw": 0.08,
                "xy": 0.14,
                "x_yaw": 0.17,
                "y_yaw": 0.05,
                "xy_yaw": 0.13,
            },
            "command_sampling_semantics": (
                "random weighted sampling at environment initialization, 120-300-step "
                "expiry and episode reset; selected 25K occupancy is reported, not quota-gated"
            ),
            "command_ranges": {
                "vx_abs": [0.05, 0.5],
                "vy_abs": [0.03, 0.2],
                "yaw_abs": [0.05, 0.4],
                "resample_steps": [120, 300],
            },
        },
        "resolved_condition": {
            "condition_id": str(args.condition_id),
            **actual_target,
        },
        "seed": 0,
        "selection_seed": 0,
        "device": {
            "cli": str(args.device),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "gpu_name": gpu_name,
            "torch_version": torch.__version__,
        },
        "paths": {
            "repo_root": str(repo_root),
            "raw_dataset": str(raw_dataset),
        },
        "artifacts": artifacts,
        "code": code,
        "git": _git_record(repo_root),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[V13-LaunchManifest] wrote {output}")
    print(f"[V13-LaunchManifest] gpu={gpu_name}")
    print(f"[V13-LaunchManifest] expert_actor_sha256={artifacts['expert_actor']['sha256']}")


if __name__ == "__main__":
    main()
