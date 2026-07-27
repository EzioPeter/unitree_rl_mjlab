"""Freeze four rule-TRACE launch configs, diffs, and provenance manifests."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO = Path(__file__).resolve().parents[2]
EXPERIMENT = Path(__file__).resolve().parent
SOURCE_ROOT = Path(
    "/data1/xjy/unitree_rl_mjlab_v13/logs/experiments/"
    "go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m_fresh_seed0_20260725"
)
OUTPUT_ROOT = Path(
    "/data1/xjy/unitree_rl_mjlab_v13/logs/experiments/"
    "go2_v13_rule_trace_T4_a010_r010_nosatgate_from_scratch_update97458_seed0_20260725"
)
HASH_LIST = Path(
    "/project/unitree_rl_mjlab_v13/REMOTE_HANDOFF_4090_225_20260725/"
    "CODE_STATE/minimal_run_artifacts_sha256.txt"
)
CONDITIONS = ("rr03", "rr05", "p5", "p75")
GPU_ASSIGNMENT = {"rr03": 1, "rr05": 1, "p5": 2, "p75": 2}
ALLOWED_CHANGED_PATHS = {
    "save_path",
    "save_checkpoint_per_interaction_step",
    "replay_mix.synthetic_mode",
    "replay_mix.trace_ratio_within_synthetic",
}
TRACE_INTERACTION_STEPS = 48828
TRACE_CONFIGURED_ENV_STEPS = 50000000
TRACE_REALIZED_ENV_STEPS = TRACE_INTERACTION_STEPS * 1024
EXPECTED_POLICY_UPDATES = 97458


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def flatten(value: Any, prefix: str = "") -> dict[str, Any]:
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            result.update(flatten(item, path))
        return result
    return {prefix: value}


def code_tree_hash() -> tuple[str, list[str]]:
    fixed = (
        REPO / "scripts/reinforcement_learning/rwm_flashsac/agent_proprioceptive.py",
        REPO / "scripts/reinforcement_learning/rwm_flashsac/replay_mixer.py",
        REPO
        / "scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2.py",
        REPO
        / "scripts/reinforcement_learning/rwm_flashsac/agent_fullstate_replay.py",
        REPO
        / "scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_v13_fullstate.py",
    )
    trace_files = tuple(
        sorted(
            path
            for path in (
                REPO / "scripts/reinforcement_learning/rwm_trace"
            ).rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        )
    )
    files = tuple(sorted((*fixed, *trace_files)))
    digest = hashlib.sha256()
    names: list[str] = []
    for path in files:
        relative = str(path.relative_to(REPO))
        names.append(relative)
        digest.update(relative.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest(), names


def artifact_hashes(condition: str) -> list[dict[str, str]]:
    rows = []
    for line in HASH_LIST.read_text(encoding="utf-8").splitlines():
        digest, raw_path = line.split(maxsplit=1)
        if f"/{condition}/" not in raw_path:
            continue
        direct_path = raw_path.replace(
            "/project/incoming_widecmd_runtime/mirror", ""
        )
        rows.append(
            {
                "sha256": digest,
                "packaged_path": raw_path,
                "direct_runtime_path": direct_path,
            }
        )
    return rows


def main() -> None:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--short"],
        cwd=REPO,
        check=True,
        text=True,
        capture_output=True,
    ).stdout.splitlines()
    code_hash, code_files = code_tree_hash()
    hash_list_sha = sha256_file(HASH_LIST)

    for condition in CONDITIONS:
        checkpoint = SOURCE_ROOT / condition / "run/step48828"
        baseline_path = checkpoint / "rwm_flashsac_config.yaml"
        overlay_path = EXPERIMENT / "configs" / f"{condition}.yaml"
        output = OUTPUT_ROOT / condition / "run"
        if output.exists() and any(output.iterdir()):
            raise FileExistsError(
                f"Refusing to overwrite non-empty TRACE output: {output}"
            )

        baseline = OmegaConf.load(baseline_path)
        resolved = OmegaConf.merge(baseline, OmegaConf.load(overlay_path))
        resolved.policy_resume_path = None
        resolved.save_path = str(output)
        OmegaConf.resolve(resolved)
        baseline_plain = OmegaConf.to_container(
            baseline, resolve=True, throw_on_missing=True
        )
        resolved_plain = OmegaConf.to_container(
            resolved, resolve=True, throw_on_missing=True
        )
        assert isinstance(baseline_plain, dict)
        assert isinstance(resolved_plain, dict)
        baseline_flat = flatten(baseline_plain)
        resolved_flat = flatten(resolved_plain)
        differences = {
            key: {
                "baseline": baseline_flat.get(key),
                "trace": resolved_flat.get(key),
            }
            for key in sorted(set(baseline_flat) | set(resolved_flat))
            if baseline_flat.get(key) != resolved_flat.get(key)
        }
        unexpected = [
            key
            for key in differences
            if not key.startswith("trace.") and key not in ALLOWED_CHANGED_PATHS
        ]
        if unexpected:
            raise ValueError(
                f"{condition}: unexpected baseline-to-TRACE changes: {unexpected}"
            )

        condition_manifest_dir = EXPERIMENT / "manifests" / condition
        condition_manifest_dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(
            resolved,
            condition_manifest_dir / "resolved_prelaunch.yaml",
            resolve=True,
        )
        (condition_manifest_dir / "baseline_to_trace_diff.json").write_text(
            json.dumps(
                {
                    "condition": condition,
                    "changed_paths": differences,
                    "unexpected_changed_paths": unexpected,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        command = [
            ".venv/bin/python",
            "scripts/reinforcement_learning/rwm_flashsac/"
            "train_flashsac_world_model_go2_v13_fullstate.py",
            "--config_path",
            str(baseline_path),
            "--trace_config_path",
            str(overlay_path),
            "--no-load_replay_buffer",
            "--device",
            "cuda",
            "--expected_policy_updates",
            str(EXPECTED_POLICY_UPDATES),
            "--save_path",
            str(output),
        ]
        manifest = {
            "format_version": "go2_v13_rule_trace_launch_manifest_v1",
            "condition": condition,
            "selection_backend": "rule_bootstrap",
            "llm_calls_required": 0,
            "physical_gpu": GPU_ASSIGNMENT[condition],
            "cuda_visible_devices": str(GPU_ASSIGNMENT[condition]),
            "baseline_config_path": str(baseline_path),
            "baseline_config_sha256": sha256_file(baseline_path),
            "trace_overlay_path": str(overlay_path),
            "trace_overlay_sha256": sha256_file(overlay_path),
            "baseline_evaluation_checkpoint": str(checkpoint),
            "policy_resume_path": None,
            "policy_initialization": "random_seed0_from_scratch",
            "initial_policy_update_step": 0,
            "expected_final_policy_update_step": EXPECTED_POLICY_UPDATES,
            "expected_completed_policy_updates": EXPECTED_POLICY_UPDATES,
            "load_replay_buffer": False,
            "output_path": str(output),
            "command": command,
            "git_head": head,
            "git_status": status,
            "formal_code_tree_sha256": code_hash,
            "formal_code_files": code_files,
            "minimal_artifact_hash_list_path": str(HASH_LIST),
            "minimal_artifact_hash_list_sha256": hash_list_sha,
            "artifact_hashes_preverified": artifact_hashes(condition),
            "reset_gate_disposition": {
                "certificate_scope": "normal-dog auxiliary proposal simulator",
                "certificate_status": "BLOCKED",
                "development_numerical_gate_passed": True,
                "accepted_nonblocking_differences": [
                    "qacc_warmstart",
                    "time",
                ],
                "user_authorized_stage_b": True,
                "physical37_linf": 6.4522e-06,
                "rwm45_linf": 2.2888e-05,
            },
            "default_parameter_evidence": {
                "proposal_interval_policy_updates": 1000,
                "updates_per_interaction_step": 2,
                "proposal_interval_interactions": 500,
                "trace_interaction_steps": TRACE_INTERACTION_STEPS,
                "trace_configured_env_steps": TRACE_CONFIGURED_ENV_STEPS,
                "trace_realized_env_steps": TRACE_REALIZED_ENV_STEPS,
                "matched_baseline_policy_updates": EXPECTED_POLICY_UPDATES,
                "num_start_states": 1024,
                "trajectories_per_start": 4,
                "rollout_horizon": 20,
                "actor_sample_temperature": 4.0,
                "select_alpha": 0.10,
                "trace_ratio_within_synthetic": 0.10,
                "action_saturation_handling": "stability_score_only_no_eligibility_gate",
                "buffer_retain_events": 5,
                "buffer_capacity": 41000,
                "checkpoint_interval_interactions": 5000,
                "checkpoint_writer": "forked_cpu_replay_copy_on_write",
            },
        }
        (condition_manifest_dir / "launch_manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
