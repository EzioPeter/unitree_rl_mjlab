"""Create the active V13 Sim mixed-dataset manifest after all audits pass."""

from __future__ import annotations

import argparse
import hashlib
import json
import socket
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


CONDITIONS = {
    "g0": {"rr_calf_strength": 1.0, "payload_mass_kg": 0.0},
    "rr05": {"rr_calf_strength": 0.5, "payload_mass_kg": 0.0},
    "rr03": {"rr_calf_strength": 0.3, "payload_mass_kg": 0.0},
    "p5": {"rr_calf_strength": 1.0, "payload_mass_kg": 5.0},
    "p75": {"rr_calf_strength": 1.0, "payload_mass_kg": 7.5},
}
REQUESTED_MIX = {
    "expert": 0.45,
    "noisy_expert": 0.25,
    "medium": 0.10,
    "failure_border": 0.15,
    "random": 0.05,
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--superseded-manifest", required=True)
    parser.add_argument("--physical-audit", required=True)
    parser.add_argument("--reset-audit", required=True)
    parser.add_argument("--smoke-result", required=True)
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _stack(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset[key]
    return value if isinstance(value, torch.Tensor) else torch.stack(value, dim=0)


def main() -> None:
    args = _parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    output = Path(args.output).resolve()
    superseded_manifest = Path(args.superseded_manifest).resolve()
    physical_audit_path = Path(args.physical_audit).resolve()
    reset_audit_path = Path(args.reset_audit).resolve()
    smoke_result_path = Path(args.smoke_result).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite active manifest: {output}")

    superseded = _load_json(superseded_manifest)
    physical = _load_json(physical_audit_path)
    reset = _load_json(reset_audit_path)
    smoke = _load_json(smoke_result_path)
    if superseded.get("schema") != "go2_v13_frozen_sim_mixed_dataset_manifest_v2":
        raise ValueError("Superseded manifest is not the prior mixed-data V13 v2 manifest.")
    if physical.get("status") != "passed":
        raise ValueError("Five-condition physical audit did not pass.")
    if reset.get("status") != "passed":
        raise ValueError("Formal selected exact-reset audit did not pass.")
    if smoke.get("status") != "passed":
        raise ValueError("Mixed collector smoke did not pass.")

    condition_records = {}
    for condition_id, expected in CONDITIONS.items():
        condition_root = dataset_root / condition_id
        selected_path = condition_root / "selected_25k/dataset.pt"
        report_path = condition_root / "selected_25k/selection_report.json"
        audit_path = condition_root / "selected_25k/fullstate_audit.json"
        launch_path = condition_root / "raw_1024x1000_mixed/launch_manifest.json"
        report = _load_json(report_path)
        audit = _load_json(audit_path)
        launch = _load_json(launch_path)
        if audit.get("status") != "passed":
            raise ValueError(f"{condition_id} independent audit did not pass.")
        if audit.get("schema") != "go2_v13_mixed_25k_fullstate_audit_v2":
            raise ValueError(f"{condition_id} audit is not the full-state V13 audit.")
        selected_sha256 = _sha256(selected_path)
        if selected_sha256 != audit.get("dataset_sha256"):
            raise ValueError(f"{condition_id} selected SHA256 mismatch.")
        if report.get("output_sha256") != selected_sha256:
            raise ValueError(f"{condition_id} selection report SHA256 mismatch.")
        if launch.get("resolved_condition", {}).get("condition_id") != condition_id:
            raise ValueError(f"{condition_id} launch manifest condition mismatch.")

        dataset = torch.load(selected_path, map_location="cpu", weights_only=False)
        metadata = dict(dataset.get("metadata") or {})
        target = dict(metadata.get("target_gap") or {})
        rr_range = [float(value) for value in target.get("rr_calf_strength_range", [])]
        payload_range = [float(value) for value in target.get("payload_mass_range_kg", [])]
        if rr_range != [expected["rr_calf_strength"]] * 2:
            raise ValueError(f"{condition_id} RR calf metadata mismatch: {rr_range}")
        if payload_range != [expected["payload_mass_kg"]] * 2:
            raise ValueError(f"{condition_id} payload metadata mismatch: {payload_range}")
        mass_delta = dataset["startup_domain_table"]["base_mass_delta"].float()
        torch.testing.assert_close(
            mass_delta,
            torch.full_like(mass_delta, expected["payload_mass_kg"]),
            atol=1.0e-5,
            rtol=0.0,
        )
        collector_counts = audit["actual_collector_counts"]
        if any(int(collector_counts.get(name, 0)) <= 0 for name in REQUESTED_MIX):
            raise ValueError(f"{condition_id} does not contain all five collectors.")
        trace_valid = _stack(dataset, "trace_valid_masks").bool()
        if not bool(trace_valid.all()):
            raise ValueError(f"{condition_id} contains a non-resettable transition.")
        reset_record = reset["conditions"].get(condition_id)
        if not reset_record or reset_record.get("status") != "passed":
            raise ValueError(f"{condition_id} exact-reset record is missing.")
        physical_record = physical["conditions"].get(condition_id)
        if not physical_record:
            raise ValueError(f"{condition_id} physical audit record is missing.")

        condition_records[condition_id] = {
            "condition": expected,
            "raw_dataset_path": report["source_path"],
            "raw_dataset_sha256": report["source_sha256"],
            "selected_dataset_path": str(selected_path),
            "selected_dataset_sha256": selected_sha256,
            "selected_dataset_size_bytes": selected_path.stat().st_size,
            "selection_report_path": str(report_path),
            "selection_report_sha256": _sha256(report_path),
            "independent_audit_path": str(audit_path),
            "independent_audit_sha256": _sha256(audit_path),
            "launch_manifest_path": str(launch_path),
            "launch_manifest_sha256": _sha256(launch_path),
            "shape": [1000, 25],
            "transitions": 25_000,
            "requested_collector_mix": REQUESTED_MIX,
            "actual_collector_counts": collector_counts,
            "actual_collector_ratios": audit["actual_collector_ratios"],
            "requested_command_sampling_weights": audit[
                "requested_command_sampling_weights"
            ],
            "raw_source_command_counts": audit["raw_source_command_counts"],
            "raw_source_command_ratios": audit["raw_source_command_ratios"],
            "selected_command_counts": audit["selected_command_counts"],
            "selected_command_ratios": audit["selected_command_ratios"],
            "selected_command_sign_counts": audit["selected_command_sign_counts"],
            "selected_command_observed_abs_ranges": audit[
                "selected_command_observed_abs_ranges"
            ],
            "command_ratio_acceptance": audit["command_ratio_acceptance"],
            "unique_episode_ids": audit["unique_episode_ids"],
            "done_count": audit["done_count"],
            "timeout_count": audit["timeout_count"],
            "all_trace_valid": True,
            "legacy_collector_observation_max_error": audit[
                "legacy_collector_observation_max_error"
            ],
            "full_policy_observation_reconstructable": True,
            "realized_base_mass_delta_kg": [
                float(mass_delta.min()),
                float(mass_delta.max()),
            ],
            "physical_audit": physical_record,
            "exact_reset_audit": reset_record,
        }

    manifest = {
        "schema": "go2_v13_frozen_sim_mixed_fullstate_dataset_manifest_v3",
        "version": "V13",
        "status": "active_sim_dataset_input",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "storage_hostname": socket.gethostname(),
        "dataset_root": str(dataset_root),
        "supersedes": {
            "path": str(superseded_manifest),
            "sha256": _sha256(superseded_manifest),
            "route": "mixed collector, proprioceptive 42-to-45 downstream interface",
            "status": "SUPERSEDED_INTERFACE_DO_NOT_TRAIN",
        },
        "reference": {
            "repository": "EzioPeter/unitree_rl_mjlab",
            "branch": "model_based",
            "commit": "daa2eae7324255d4011964d2e74199547f347f26",
        },
        "dataset_policy": {
            "raw_shape_per_condition": [1000, 1024],
            "selected_shape_per_condition": [1000, 25],
            "selected_transitions_per_condition": 25_000,
            "selection": "stratified_environment_columns",
            "selection_seed": 0,
            "requested_collector_mix": REQUESTED_MIX,
            "actual_transition_occupancy_is_condition_dependent": True,
            "require_all_five_collectors": True,
            "allow_cross_condition_reuse": False,
            "allow_cross_episode_windows": False,
            "allow_cross_reset_windows": False,
            "command_sampling": (
                "weighted random sampling at initialization, 120-300-step expiry, "
                "and episode reset"
            ),
            "command_ratios_are_report_only": True,
            "command_ratio_hard_quota": None,
        },
        "rwm_interface": {
            "full_state_dim": 45,
            "input_state_dim": 45,
            "output_state_dim": 45,
            "input_dropped_state_indices": [],
            "output_dropped_state_indices": [],
            "state_loss_ignored_indices": [],
            "base_lin_vel_input_present": True,
            "base_lin_vel_target_supervised": True,
            "action_dim": 12,
        },
        "policy_interface": {
            "actor_observation_dim": 48,
            "critic_observation_dim": 48,
            "command_dim": 3,
            "actor_contains_base_lin_vel": True,
            "critic_contains_base_lin_vel": True,
            "observation_order": [
                "base_lin_vel[3]",
                "base_ang_vel[3]",
                "projected_gravity[3]",
                "command[3]",
                "joint_pos_rel[12]",
                "joint_vel[12]",
                "prev_action[12]",
            ],
        },
        "collector_interface_provenance": {
            "collector_task_actor_observation_dim": 45,
            "collector_task_critic_observation_dim": 48,
            "stored_observation_dim": 45,
            "stored_observation_is_legacy_projection": True,
            "stored_observation_used_as_final_policy_input": False,
            "full_48d_policy_observation_reconstructed_from": [
                "states",
                "commands",
                "prev_actions",
                "next_states",
                "next_observations",
            ],
        },
        "snapshot_contract": {
            "snapshot_version": "go2_trace_snapshot_v1",
            "all_selected_rows_trace_valid": True,
            "reset_reconstruction_state_slice": [0, 33],
            "actuator_force_state_slice": [33, 45],
            "actuator_force_resettable": False,
            "identity_tolerance": 0.005,
            "formal_five_condition_exact_reset_audit_path": str(reset_audit_path),
            "formal_five_condition_exact_reset_audit_sha256": _sha256(
                reset_audit_path
            ),
            "mixed_smoke_result_path": str(smoke_result_path),
            "mixed_smoke_result_sha256": _sha256(smoke_result_path),
        },
        "physical_audit": {
            "path": str(physical_audit_path),
            "sha256": _sha256(physical_audit_path),
            "status": "passed",
        },
        "conditions": condition_records,
        "downstream_binding": {
            "sim_rwm_training": "must match condition selected_dataset_sha256",
            "trace_reset_source": "must match condition selected_dataset_sha256",
            "run_manifest": "must record this manifest SHA256 and condition dataset SHA256",
        },
    }
    output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[V13-SimManifest] wrote {output}")
    print(f"[V13-SimManifest] sha256={_sha256(output)}")


if __name__ == "__main__":
    main()
