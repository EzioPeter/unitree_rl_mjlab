"""Small real-MJLab smoke test for V13 mixed collection and exact reset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


SNAPSHOT_WIDTHS = {
    "sim_root_states_local": 13,
    "sim_joint_positions": 12,
    "sim_joint_velocities": 12,
    "sim_action_histories": 12,
    "sim_prev_action_histories": 12,
    "sim_prev_prev_action_histories": 12,
    "sim_snapshot_commands": 3,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expert-policy-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _stack(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset[key]
    return value if isinstance(value, torch.Tensor) else torch.stack(value, dim=0)


def _run(command: list[str], log_path: Path, repo_root: Path) -> None:
    with log_path.open("w", encoding="utf-8") as stream:
        subprocess.run(
            command,
            cwd=repo_root,
            check=True,
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )


def _base_command(
    python: Path,
    collector: Path,
    expert: Path,
    save_path: Path,
    *,
    device: str,
    num_transitions: int,
) -> list[str]:
    return [
        str(python),
        str(collector),
        "--task",
        "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
        "--device",
        device,
        "--seed",
        "0",
        "--num_envs",
        "32",
        "--num_transitions",
        str(num_transitions),
        "--save_path",
        str(save_path),
        "--dataset_obs_kind",
        "proprioceptive",
        "--expert_policy_path",
        str(expert),
        "--collector_mix",
        "expert:0.45,noisy_expert:0.25,medium:0.10,failure_border:0.15,random:0.05",
        "--action_noise_std",
        "0.12",
        "--medium_action_noise_std",
        "0.25",
        "--failure_action_noise_std",
        "0.45",
        "--command_modes",
        "stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw",
        "--command_mode_weights",
        "stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13",
        "--command_resample_interval_min",
        "120",
        "--command_resample_interval_max",
        "300",
        "--x_range",
        "-0.5",
        "0.5",
        "--signed_x",
        "--x_abs_range",
        "0.05",
        "0.5",
        "--y_abs_range",
        "0.03",
        "0.2",
        "--yaw_abs_range",
        "0.05",
        "0.4",
        "--rr_calf_strength_range",
        "1.0",
        "1.0",
        "--payload_mass_range_kg",
        "0.0",
        "0.0",
        "--env_action_noise_std",
        "0.0",
        "--env_action_bias_std",
        "0.0",
        "--env_action_scale_range",
        "1.0",
        "1.0",
        "--env_action_delay_steps",
        "0",
        "0",
        "--no-use_domain_randomization",
        "--no-use_push_randomization",
        "--no-use_observation_noise",
        "--save_trace_snapshots",
        "--chunk_size",
        str(num_transitions),
        "--headless",
    ]


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    expert = Path(args.expert_policy_path).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite smoke output: {output_root}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be explicitly set.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    gpu_name = torch.cuda.get_device_name(torch.device(args.device))
    if "3090" not in gpu_name:
        raise RuntimeError(f"Expected RTX 3090, got {gpu_name!r}.")

    python = repo_root / ".venv/bin/python"
    collector = (
        repo_root
        / "scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py"
    )
    source = output_root / "source_smoke.pt"
    trace = output_root / "exact_reset_smoke.pt"
    source_command = _base_command(
        python,
        collector,
        expert,
        source,
        device=str(args.device),
        num_transitions=4096,
    )
    trace_command = _base_command(
        python,
        collector,
        expert,
        trace,
        device=str(args.device),
        num_transitions=64,
    )
    trace_command.extend(
        [
            "--trace_reset_dataset",
            str(source),
            "--trace_reset_mode",
            "exact_snapshot",
            "--trace_rollout_length",
            "2",
            "--trace_trajectories_per_state",
            "4",
            "--trace_identity_tolerance",
            "0.005",
        ]
    )

    output_root.mkdir(parents=True)
    launch_manifest = {
        "schema": "go2_v13_mixed_exact_reset_smoke_launch_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "planned",
        "formal_training_input": False,
        "gpu": {
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "name": gpu_name,
            "device": str(args.device),
        },
        "source_command": source_command,
        "trace_command": trace_command,
        "artifacts": {
            "expert_actor": _record(expert / "actor.pt"),
            "expert_config": _record(expert / "flashsac_config.yaml"),
            "collector_code": _record(collector),
            "test_code": _record(Path(__file__)),
        },
    }
    (output_root / "launch_manifest.json").write_text(
        json.dumps(launch_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    try:
        _run(source_command, output_root / "source.log", repo_root)
        source_dataset = torch.load(source, map_location="cpu", weights_only=False)
        assert tuple(_stack(source_dataset, "states").shape) == (128, 32, 45)
        assert tuple(_stack(source_dataset, "observations").shape) == (128, 32, 45)
        assert bool(_stack(source_dataset, "trace_valid_masks").all())
        for key, width in SNAPSHOT_WIDTHS.items():
            tensor = _stack(source_dataset, key)
            assert tuple(tensor.shape) == (128, 32, width), (key, tensor.shape)
            assert bool(torch.isfinite(tensor).all())
        actor_observation = torch.cat(
            (
                _stack(source_dataset, "states")[..., 3:9],
                _stack(source_dataset, "commands"),
                _stack(source_dataset, "states")[..., 9:33],
                _stack(source_dataset, "prev_actions"),
            ),
            dim=-1,
        )
        actor_error = float(
            torch.max(
                torch.abs(actor_observation - _stack(source_dataset, "observations"))
            ).item()
        )
        assert actor_error <= 1.0e-6

        _run(trace_command, output_root / "exact_reset.log", repo_root)
        trace_dataset = torch.load(trace, map_location="cpu", weights_only=False)
        diagnostics = trace_dataset["metadata"]["trace_candidates"]
        repeated = diagnostics["one_step_identity"]
        source_identity = diagnostics["source_transition_identity"]
        assert repeated["passed"] is True
        assert repeated["max_state_abs_error"] <= repeated["tolerance"]
        assert source_identity["available"] is True
        assert source_identity["passed"] is True
        assert source_identity["max_state_abs_error"] <= source_identity["tolerance"]
        assert bool(_stack(trace_dataset, "trace_valid_masks").all())
        reset_errors = _stack(
            trace_dataset,
            "trace_reset_reconstruction_errors",
        )
        reset_error_max = float(reset_errors.max().item())
        assert reset_error_max <= 0.005

        result = {
            "schema": "go2_v13_mixed_exact_reset_smoke_result_v1",
            "status": "passed",
            "formal_training_input": False,
            "source_dataset": _record(source),
            "trace_dataset": _record(trace),
            "source_shape": [128, 32, 45],
            "trace_shape": list(_stack(trace_dataset, "states").shape),
            "actor_reconstruction_max_error": actor_error,
            "reset_reconstruction_error_state_0_33_max": reset_error_max,
            "reset_reconstruction_tolerance": 0.005,
            "one_step_identity": repeated,
            "source_transition_identity": source_identity,
            "all_source_trace_valid": True,
            "all_rollout_trace_valid": True,
            "snapshot_widths": SNAPSHOT_WIDTHS,
        }
        (output_root / "result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    except Exception as error:
        (output_root / "INVALID_DO_NOT_USE.txt").write_text(
            f"{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        raise

    print("[V13-MixedCollectorExactResetSmoke] PASS")
    print(f"[V13-MixedCollectorExactResetSmoke] gpu={gpu_name}")
    print(f"[V13-MixedCollectorExactResetSmoke] actor_error={actor_error:.9g}")
    print(
        "[V13-MixedCollectorExactResetSmoke] repeated_identity_error="
        f"{repeated['max_state_abs_error']:.9g}"
    )
    print(
        "[V13-MixedCollectorExactResetSmoke] source_identity_error="
        f"{source_identity['max_state_abs_error']:.9g}"
    )
    print(
        "[V13-MixedCollectorExactResetSmoke] reset_reconstruction_error="
        f"{reset_error_max:.9g}"
    )
    print(f"[V13-MixedCollectorExactResetSmoke] result={output_root / 'result.json'}")


if __name__ == "__main__":
    main()
