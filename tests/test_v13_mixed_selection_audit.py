"""End-to-end synthetic test for V13 mixed selection and independent audit."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import torch


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expert-policy-path", required=True)
    return parser.parse_args()


def _run(*args: str) -> None:
    subprocess.run([sys.executable, *args], check=True)


def _rows(row: torch.Tensor, time_steps: int = 1000) -> list[torch.Tensor]:
    return [row] * time_steps


def _make_source(path: Path, expert_policy_path: Path) -> None:
    time_steps = 1000
    num_envs = 1024
    zero45 = torch.zeros(num_envs, 45)
    zero12 = torch.zeros(num_envs, 12)
    zero4 = torch.zeros(num_envs, 4)
    zero3 = torch.zeros(num_envs, 3)
    zero2 = torch.zeros(num_envs, 2)
    zero1 = torch.zeros(num_envs)
    false = torch.zeros(num_envs, dtype=torch.bool)
    true = torch.ones(num_envs, dtype=torch.bool)
    env_ids = torch.arange(num_envs, dtype=torch.long)
    collector_pattern = torch.tensor(
        [1] * 9 + [2] * 5 + [3] * 2 + [4] * 3 + [0],
        dtype=torch.long,
    )
    collector_rows = [
        collector_pattern[(env_ids + time_step) % collector_pattern.numel()]
        for time_step in range(time_steps)
    ]
    command_values = (
        (0.0, 0.0, 0.0),
        (0.25, 0.0, 0.0),
        (0.0, -0.10, 0.0),
        (0.0, 0.0, 0.20),
        (-0.25, 0.10, 0.0),
        (0.25, 0.0, -0.20),
        (0.0, -0.10, 0.20),
        (-0.25, 0.10, -0.20),
    )
    command_rows = [
        torch.tensor(command_values[time_step % len(command_values)]).repeat(num_envs, 1)
        for time_step in range(time_steps)
    ]
    observation_rows = []
    for command_row in command_rows:
        observation = zero45.clone()
        observation[:, 6:9] = command_row
        observation_rows.append(observation)
    timestep_rows = [
        torch.full((num_envs,), time_step, dtype=torch.long)
        for time_step in range(time_steps)
    ]
    dataset = {
        "format_version": "go2_mixed_rwm_dataset_v2",
        "state_dim": 45,
        "action_dim": 12,
        "contact_dim": 4,
        "termination_dim": 2,
        "num_envs": num_envs,
        "capacity": time_steps * num_envs,
        "states": _rows(zero45),
        "next_states": _rows(zero45),
        "actions": _rows(zero12),
        "raw_actions": _rows(zero12),
        "observations": observation_rows,
        "next_observations": _rows(zero45),
        "commands": command_rows,
        "prev_actions": _rows(zero12),
        "contacts": _rows(zero4),
        "terminations": _rows(zero2),
        "rewards": _rows(zero1),
        "dones": _rows(false),
        "timeouts": _rows(false),
        "episode_ids": _rows(env_ids),
        "timesteps": timestep_rows,
        "collector_types": collector_rows,
        "trace_valid_masks": _rows(true),
        "sim_root_states_local": _rows(torch.zeros(num_envs, 13)),
        "sim_joint_positions": _rows(zero12),
        "sim_joint_velocities": _rows(zero12),
        "sim_action_histories": _rows(zero12),
        "sim_prev_action_histories": _rows(zero12),
        "sim_prev_prev_action_histories": _rows(zero12),
        "sim_snapshot_commands": _rows(zero3),
        "metadata": {
            "task": "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
            "seed": 0,
            "obs_dim": 45,
            "dataset_obs_kind": "proprioceptive",
            "state_dim": 45,
            "action_dim": 12,
            "live_actor_obs_dim": 45,
            "live_critic_obs_dim": 48,
            "num_envs": num_envs,
            "requested_num_transitions": time_steps * num_envs,
            "actual_num_transitions": time_steps * num_envs,
            "collector_mix": {
                "expert": 0.45,
                "noisy_expert": 0.25,
                "medium": 0.10,
                "failure_border": 0.15,
                "random": 0.05,
            },
            "fixed_collector_assignment": False,
            "expert_policy_path": str(expert_policy_path),
            "medium_policy_path": None,
            "action_noise_std": 0.12,
            "medium_action_noise_std": 0.25,
            "failure_action_noise_std": 0.45,
            "dagger_relabel": {"enabled": False},
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
            "command_mode_counts": {
                "stand": 81_920,
                "pure_x": 256_000,
                "pure_y": 102_400,
                "pure_yaw": 81_920,
                "xy": 143_360,
                "x_yaw": 174_080,
                "y_yaw": 51_200,
                "xy_yaw": 133_120,
            },
            "command_ranges": {
                "x_range": [-0.5, 0.5],
                "signed_x": True,
                "x_abs_range": [0.05, 0.5],
                "y_abs_range": [0.03, 0.2],
                "yaw_abs_range": [0.05, 0.4],
            },
            "command_resample_interval": [120, 300],
            "env_randomization": {
                "use_domain_randomization": False,
                "use_push_randomization": False,
                "use_observation_noise": False,
            },
            "env_step_action_interface": {
                "env_action_noise_std": 0.0,
                "env_action_bias_std": 0.0,
                "env_action_scale_range": [1.0, 1.0],
                "env_action_delay_steps": [0, 0],
            },
            "target_gap": {
                "rr_calf_strength_range": [1.0, 1.0],
                "payload_mass_range_kg": [0.0, 0.0],
            },
            "save_trace_snapshots": True,
        },
    }
    torch.save(dataset, path)


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    expert = Path(args.expert_policy_path).resolve()
    scripts = repo_root / "scripts/reinforcement_learning/rwm_dataset"
    with tempfile.TemporaryDirectory(prefix="v13_mixed_selection_test_") as temporary:
        root = Path(temporary)
        source = root / "raw.pt"
        selected = root / "selected.pt"
        selection_report = root / "selection.json"
        audit = root / "audit.json"
        manifest = root / "launch_manifest.json"
        _make_source(source, expert)

        code_paths = [
            scripts / "collect_go2_expert_command_coverage_dataset.py",
            scripts / "slice_go2_dataset_by_env_trajectories.py",
            scripts / "audit_v13_mixed_25k.py",
            scripts / "prepare_v13_mixed_collection_manifest.py",
        ]
        manifest_args = [
            str(scripts / "prepare_v13_mixed_collection_manifest.py"),
            "--output",
            str(manifest),
            "--repo_root",
            str(repo_root),
            "--expert_policy_path",
            str(expert),
            "--condition_id",
            "g0",
            "--rr_calf_strength",
            "1.0",
            "--payload_mass_kg",
            "0.0",
            "--device",
            "cuda:0",
            "--raw_dataset",
            str(source),
        ]
        # The manifest writer normally runs before collection. Temporarily move
        # the synthetic source aside to preserve that production assertion.
        staged_source = root / "raw.staged.pt"
        source.rename(staged_source)
        for code_path in code_paths:
            manifest_args.extend(["--code_path", str(code_path)])
        _run(*manifest_args)
        staged_source.rename(source)

        _run(
            str(scripts / "slice_go2_dataset_by_env_trajectories.py"),
            "--source_path",
            str(source),
            "--save_path",
            str(selected),
            "--report_path",
            str(selection_report),
            "--condition_id",
            "g0",
            "--expected_expert_policy_path",
            str(expert),
            "--launch_manifest_path",
            str(manifest),
            "--num_trajectories",
            "25",
            "--expected_time_steps",
            "1000",
            "--seed",
            "0",
            "--selection",
            "stratified",
            "--require_trace_snapshots",
        )
        _run(
            str(scripts / "audit_v13_mixed_25k.py"),
            "--dataset",
            str(selected),
            "--selection_report",
            str(selection_report),
            "--output",
            str(audit),
            "--condition_id",
            "g0",
        )

        result = json.loads(audit.read_text(encoding="utf-8"))
        assert result["status"] == "passed"
        assert result["shape"]["transitions"] == 25_000
        assert result["all_trace_valid"] is True
        assert result["actor_reconstruction_max_error"] == 0.0
        assert result["shape"]["reconstructed_policy_observation_dim"] == 48
        assert result["command_ratio_acceptance"] == "reported_not_bounded_no_hard_quota"
        assert set(result["selected_command_counts"]) == set(result["requested_command_sampling_weights"])
        assert max(abs(value) for value in result["collector_ratio_errors"].values()) <= 0.08

        loaded_source = torch.load(source, map_location="cpu", weights_only=False)
        loaded_source["metadata"]["target_gap"]["rr_calf_strength_range"] = [0.5, 0.5]
        sys.path.insert(0, str(repo_root))
        from scripts.reinforcement_learning.rwm_dataset.slice_go2_dataset_by_env_trajectories import (
            _validate_source_protocol,
        )

        try:
            _validate_source_protocol(
                loaded_source,
                condition_id="g0",
                expected_expert_policy_path=str(expert),
            )
        except ValueError as error:
            assert "rr_calf_strength_range" in str(error)
        else:
            raise AssertionError("Wrong-condition source was not rejected.")

        print("[V13-SyntheticE2E] PASS")
        print("[V13-SyntheticE2E] wrong-condition source rejected")


if __name__ == "__main__":
    main()
