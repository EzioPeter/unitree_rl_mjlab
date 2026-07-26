"""Run exact-reset identity checks on all five formal V13 selected datasets."""

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


CONDITIONS = {
    "g0": (1.0, 0.0),
    "rr05": (0.5, 0.0),
    "rr03": (0.3, 0.0),
    "p5": (1.0, 5.0),
    "p75": (1.0, 7.5),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--dataset-root", required=True)
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


def _stack(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset[key]
    return value if isinstance(value, torch.Tensor) else torch.stack(value, dim=0)


def _command(
    python: Path,
    collector: Path,
    expert: Path,
    source: Path,
    output: Path,
    *,
    rr_strength: float,
    payload_kg: float,
    device: str,
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
        "64",
        "--save_path",
        str(output),
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
        str(rr_strength),
        str(rr_strength),
        "--payload_mass_range_kg",
        str(payload_kg),
        str(payload_kg),
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
        "64",
        "--headless",
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


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    dataset_root = Path(args.dataset_root).resolve()
    expert = Path(args.expert_policy_path).resolve()
    output_root = Path(args.output_root).resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite reset audit: {output_root}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be explicitly set.")
    gpu_name = torch.cuda.get_device_name(torch.device(args.device))
    if "3090" not in gpu_name:
        raise RuntimeError(f"Expected RTX 3090, got {gpu_name!r}.")

    python = repo_root / ".venv/bin/python"
    collector = (
        repo_root
        / "scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py"
    )
    commands = {}
    for condition_id, (rr_strength, payload_kg) in CONDITIONS.items():
        source = dataset_root / condition_id / "selected_25k/dataset.pt"
        if not source.is_file():
            raise FileNotFoundError(source)
        commands[condition_id] = _command(
            python,
            collector,
            expert,
            source,
            output_root / condition_id / "exact_reset.pt",
            rr_strength=rr_strength,
            payload_kg=payload_kg,
            device=str(args.device),
        )

    output_root.mkdir(parents=True)
    launch = {
        "schema": "go2_v13_formal_selected_exact_reset_launch_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "name": gpu_name,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "formal_training_input": False,
        "dataset_root": str(dataset_root),
        "commands": commands,
        "code_sha256": {
            str(collector): _sha256(collector),
            str(Path(__file__).resolve()): _sha256(Path(__file__).resolve()),
        },
    }
    (output_root / "launch_manifest.json").write_text(
        json.dumps(launch, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    results = {}
    try:
        for condition_id, command in commands.items():
            condition_root = output_root / condition_id
            condition_root.mkdir()
            log_path = condition_root / "exact_reset.log"
            with log_path.open("w", encoding="utf-8") as stream:
                subprocess.run(
                    command,
                    cwd=repo_root,
                    check=True,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
            output = condition_root / "exact_reset.pt"
            dataset = torch.load(output, map_location="cpu", weights_only=False)
            diagnostics = dataset["metadata"]["trace_candidates"]
            repeated = diagnostics["one_step_identity"]
            source_identity = diagnostics["source_transition_identity"]
            reset_error_max = float(
                _stack(dataset, "trace_reset_reconstruction_errors").max().item()
            )
            assert repeated["passed"] is True
            assert repeated["max_state_abs_error"] <= 0.005
            assert source_identity["available"] is True
            assert source_identity["passed"] is True
            assert source_identity["max_state_abs_error"] <= 0.005
            assert reset_error_max <= 0.005
            assert bool(_stack(dataset, "trace_valid_masks").all())
            source = dataset_root / condition_id / "selected_25k/dataset.pt"
            results[condition_id] = {
                "status": "passed",
                "source_dataset_path": str(source),
                "source_dataset_sha256": _sha256(source),
                "audit_dataset_path": str(output),
                "audit_dataset_sha256": _sha256(output),
                "reset_reconstruction_error_state_0_33_max": reset_error_max,
                "repeated_one_step_max_error": repeated["max_state_abs_error"],
                "source_transition_max_error": source_identity["max_state_abs_error"],
                "tolerance": 0.005,
                "all_trace_valid": True,
            }
            print(
                f"[V13-FormalExactReset] PASS {condition_id} "
                f"reset={reset_error_max:.9g} "
                f"repeat={repeated['max_state_abs_error']:.9g} "
                f"source={source_identity['max_state_abs_error']:.9g}"
            )
    except Exception as error:
        (output_root / "INVALID_DO_NOT_USE.txt").write_text(
            f"{type(error).__name__}: {error}\n",
            encoding="utf-8",
        )
        raise

    result = {
        "schema": "go2_v13_formal_selected_exact_reset_result_v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_training_input": False,
        "conditions": results,
    }
    result_path = output_root / "result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[V13-FormalExactReset] ALL PASS result={result_path}")


if __name__ == "__main__":
    main()
