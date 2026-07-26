#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "usage: $0 <condition> <physical_gpu>" >&2
  exit 2
fi

condition="$1"
physical_gpu="$2"
case "${condition}" in
  g0|rr03|rr05|p5|p75) ;;
  *)
    echo "Unsupported V13 condition: ${condition}" >&2
    exit 2
    ;;
esac
if [[ "${physical_gpu}" == "2" || "${physical_gpu}" == "7" ]]; then
  echo "GPU ${physical_gpu} is forbidden for V13." >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST="${V13_SIM_MANIFEST:-${REPO_ROOT}/V13_SIM_DATASET_MANIFEST.json}"
CONFIG="${SCRIPT_DIR}/configs/go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m.yaml"
RWM_EXPERIMENT_NAME="${V13_RWM_EXPERIMENT_NAME:-go2_rwm_v13_full45_25k}"
RWM_ROOT="${REPO_ROOT}/logs/experiments/${RWM_EXPERIMENT_NAME}/${condition}/rwm"
POLICY_EXPERIMENT_NAME="${V13_POLICY_EXPERIMENT_NAME:-go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m}"
POLICY_ROOT="${REPO_ROOT}/logs/experiments/${POLICY_EXPERIMENT_NAME}/${condition}"
INPUT_ROOT="${POLICY_ROOT}/policy_inputs"
SAVE_ROOT="${POLICY_ROOT}/run"
STATUS_ROOT="${V13_POLICY_STATUS_ROOT:-${REPO_ROOT}/logs/queues/v13_policy_fullstate_mixed1m_widecmd_policycfg_20260725/status}"
cd "${REPO_ROOT}"

EXPECTED_MANIFEST_SHA256="f507931ef4106d6c64952ddc25f4280d00c17330d8396bf30a788b6195bb28ff"
EXPECTED_CONFIG_SHA256="6884a9dd6b3f84ca61f0822062b42fdceb37f00085c4b179da89551f667685e0"

mkdir -p "${STATUS_ROOT}/${condition}"
status_path="${STATUS_ROOT}/${condition}/policy.status"

fail() {
  local message="$*"
  printf '%s\n' \
    "stage=policy" \
    "condition=${condition}" \
    "gpu=${physical_gpu}" \
    "status=failed" \
    "message=${message}" \
    "failed_at=$(date --iso-8601=seconds)" \
    > "${status_path}"
  echo "ERROR: ${message}" >&2
  exit 1
}

[[ -f "${MANIFEST}" ]] || fail "Missing V13 dataset manifest."
[[ -f "${CONFIG}" ]] || fail "Missing frozen V13 policy config."
[[ "$(sha256sum "${MANIFEST}" | awk '{print $1}')" == "${EXPECTED_MANIFEST_SHA256}" ]] \
  || fail "V13 dataset manifest hash changed."
[[ "$(sha256sum "${CONFIG}" | awk '{print $1}')" == "${EXPECTED_CONFIG_SHA256}" ]] \
  || fail "V13 policy config hash changed."
[[ ! -e "${POLICY_ROOT}" ]] || fail "Policy output root already exists: ${POLICY_ROOT}"

mapfile -t checkpoints < <(
  find "${RWM_ROOT}" -mindepth 2 -maxdepth 2 -type f -name model_5000.pt -print
)
[[ "${#checkpoints[@]}" -eq 1 ]] \
  || fail "Expected exactly one model_5000.pt, found ${#checkpoints[@]}."
checkpoint="${checkpoints[0]}"

dataset="$(
  "${REPO_ROOT}/.venv/bin/python" - "${MANIFEST}" "${condition}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("schema") != "go2_v13_frozen_sim_mixed_fullstate_dataset_manifest_v3":
    raise SystemExit("Unexpected V13 dataset manifest schema.")
entry = manifest["conditions"][sys.argv[2]]
print(entry["selected_dataset_path"])
PY
)"
[[ -f "${dataset}" ]] || fail "Condition dataset does not exist: ${dataset}"

mkdir -p "${INPUT_ROOT}"
real_replay="${INPUT_ROOT}/real_replay.pt"
reward_normalizer="${INPUT_ROOT}/frozen_reward_normalizer.pt"
preflight="${INPUT_ROOT}/policy_preflight.json"

CONDITION="${condition}" \
DATASET="${dataset}" \
CHECKPOINT="${checkpoint}" \
MANIFEST="${MANIFEST}" \
CONFIG="${CONFIG}" \
PREFLIGHT="${preflight}" \
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


condition = os.environ["CONDITION"]
dataset_path = Path(os.environ["DATASET"]).resolve()
checkpoint_path = Path(os.environ["CHECKPOINT"]).resolve()
manifest_path = Path(os.environ["MANIFEST"]).resolve()
config_path = Path(os.environ["CONFIG"]).resolve()
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
entry = manifest["conditions"][condition]
if sha256(dataset_path) != entry["selected_dataset_sha256"]:
    raise ValueError("Selected dataset SHA256 does not match the V13 manifest.")

checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
infos = checkpoint.get("infos") or {}
checkpoint_dataset_path = infos.get("dataset_path")
if not checkpoint_dataset_path:
    raise ValueError("RWM checkpoint does not record infos['dataset_path'].")
if Path(checkpoint_dataset_path).resolve() != dataset_path:
    raise ValueError(
        "RWM checkpoint dataset path does not match this condition: "
        f"{Path(checkpoint_dataset_path).resolve()} != {dataset_path}."
    )
if int(infos.get("state_dim", -1)) != 45:
    raise ValueError("RWM checkpoint state_dim is not 45.")
if int(infos.get("action_dim", -1)) != 12:
    raise ValueError("RWM checkpoint action_dim is not 12.")
normalizer = checkpoint.get("normalizer") or {}
if tuple(normalizer["state_mean"].shape) != (45,):
    raise ValueError("RWM checkpoint state normalizer is not 45D.")
if tuple(normalizer["obs_mean"].shape) != (48,):
    raise ValueError("RWM checkpoint policy observation normalizer is not 48D.")
interface = (checkpoint.get("config") or {}).get("v13_interface") or {}
expected_interface = {
    "rwm_input_state_dim": 45,
    "rwm_output_state_dim": 45,
    "actor_observation_dim": 48,
    "critic_observation_dim": 48,
}
for key, expected in expected_interface.items():
    if int(interface.get(key, -1)) != expected:
        raise ValueError(f"RWM checkpoint {key} does not equal {expected}.")

cfg = OmegaConf.load(config_path)
OmegaConf.resolve(cfg)
checks = {
    "num_imagination_envs": (int(cfg.num_imagination_envs), 1024),
    "num_env_steps": (int(cfg.num_env_steps), 50_000_000),
    "updates_per_interaction_step": (int(cfg.updates_per_interaction_step), 2),
    "real_ratio": (float(cfg.replay_mix.real_ratio), 0.05),
    "synthetic_mode": (str(cfg.replay_mix.synthetic_mode), "rwm"),
    "trace_ratio": (float(cfg.replay_mix.trace_ratio_within_synthetic), 0.0),
    "batch_size": (int(cfg.agent.sample_batch_size), 2048),
    "gamma": (float(cfg.agent.gamma), 0.99),
    "n_step": (int(cfg.agent.n_step), 3),
    "buffer_max_length": (int(cfg.agent.buffer_max_length), 10_000_000),
    "buffer_min_length": (int(cfg.agent.buffer_min_length), 100_000),
    "buffer_device_type": (str(cfg.agent.buffer_device_type), "cpu"),
    "normalize_reward": (bool(cfg.agent.normalize_reward), True),
    "normalized_G_max": (float(cfg.agent.normalized_G_max), 5.0),
    "critic_min_v": (float(cfg.agent.critic_min_v), -5.0),
    "critic_max_v": (float(cfg.agent.critic_max_v), 5.0),
    "actor_bc_alpha": (float(cfg.agent.actor_bc_alpha), 0.0),
    "asymmetric_observation": (bool(cfg.agent.asymmetric_observation), False),
    "reward_normalization_enabled": (bool(cfg.reward_normalization.enabled), True),
    "reward_normalization_frozen": (bool(cfg.reward_normalization.frozen), True),
    "reward_version": (str(cfg.world_model.reward_version), "v1"),
    "lin_vel_x_min": (float(cfg.world_model.lin_vel_x_min), -0.5),
    "lin_vel_x_max": (float(cfg.world_model.lin_vel_x_max), 0.5),
    "lin_vel_y_min": (float(cfg.world_model.lin_vel_y_min), -0.2),
    "lin_vel_y_max": (float(cfg.world_model.lin_vel_y_max), 0.2),
    "ang_vel_z_min": (float(cfg.world_model.ang_vel_z_min), -0.4),
    "ang_vel_z_max": (float(cfg.world_model.ang_vel_z_max), 0.4),
    "uncertainty_penalty_weight": (
        float(cfg.world_model.uncertainty_penalty_weight),
        -2.0,
    ),
    "reward_track_linear_velocity_weight": (
        float(cfg.world_model.reward_track_linear_velocity_weight),
        1.0,
    ),
    "reward_track_angular_velocity_weight": (
        float(cfg.world_model.reward_track_angular_velocity_weight),
        1.0,
    ),
    "reward_body_orientation_l2_weight": (
        float(cfg.world_model.reward_body_orientation_l2_weight),
        -1.0,
    ),
    "reward_body_ang_vel_weight": (
        float(cfg.world_model.reward_body_ang_vel_weight),
        -0.05,
    ),
    "reward_foot_gait_weight": (
        float(cfg.world_model.reward_foot_gait_weight),
        0.5,
    ),
    "reward_stand_still_weight": (
        float(cfg.world_model.reward_stand_still_weight),
        -1.0,
    ),
}
for name, (actual, expected) in checks.items():
    if actual != expected:
        raise ValueError(f"Frozen config {name}={actual!r}, expected {expected!r}.")

record = {
    "schema": "go2_v13_fullstate_policy_preflight_v1",
    "condition": condition,
    "dataset_path": str(dataset_path),
    "dataset_sha256": sha256(dataset_path),
    "rwm_checkpoint_path": str(checkpoint_path),
    "rwm_checkpoint_sha256": sha256(checkpoint_path),
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "config_path": str(config_path),
    "config_sha256": sha256(config_path),
    "policy_interface": expected_interface,
    "formal_config_checks": checks,
}
Path(os.environ["PREFLIGHT"]).write_text(
    json.dumps(record, indent=2, sort_keys=True, default=str) + "\n",
    encoding="utf-8",
)
PY

"${REPO_ROOT}/.venv/bin/python" \
  "${SCRIPT_DIR}/build_go2_real_replay_v13_fullstate.py" \
  --dataset-path "${dataset}" \
  --config-path "${CONFIG}" \
  --output-path "${real_replay}" \
  --condition "${condition}" \
  --dataset-size 25k \
  --device cpu

"${REPO_ROOT}/.venv/bin/python" -m \
  scripts.reinforcement_learning.rwm_flashsac.audit_go2_real_replay \
  --artifact-path "${real_replay}" \
  --dataset-path "${dataset}" \
  --config-path "${CONFIG}" \
  --device cpu \
  > "${INPUT_ROOT}/real_replay_audit.json"

"${REPO_ROOT}/.venv/bin/python" \
  "${SCRIPT_DIR}/frozen_reward_normalizer_v13_fullstate.py" \
  --dataset-path "${dataset}" \
  --config-path "${CONFIG}" \
  --output-path "${reward_normalizer}" \
  --condition "${condition}" \
  --dataset-size 25k \
  --device cpu

REAL_REPLAY="${real_replay}" \
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import os
import torch

real = torch.load(os.environ["REAL_REPLAY"], map_location="cpu", weights_only=False)
metadata = real["metadata"]
if tuple(real["observation"].shape[1:]) != (48,):
    raise ValueError("Real replay observations are not 48D.")
if tuple(real["next_observation"].shape[1:]) != (48,):
    raise ValueError("Real replay next observations are not 48D.")
if metadata.get("actor_observation_dim") != 48:
    raise ValueError("Real replay actor interface is not 48D.")
if metadata.get("critic_observation_dim") != 48:
    raise ValueError("Real replay critic interface is not 48D.")
if metadata.get("source_collector_types") != [0, 1, 2, 3, 4]:
    raise ValueError("Real replay is not bound to all five mixed collector types.")
if metadata.get("reward_version") != "v1":
    raise ValueError("Real replay reward is not the model_based reference v1.")
if int(metadata.get("n_step", -1)) != 3:
    raise ValueError("Real replay n_step is not 3.")
PY

REAL_REPLAY="${real_replay}" REWARD_NORMALIZER="${reward_normalizer}" \
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import os
import torch

real = torch.load(os.environ["REAL_REPLAY"], map_location="cpu", weights_only=False)
normalizer = torch.load(
    os.environ["REWARD_NORMALIZER"], map_location="cpu", weights_only=False
)
real_meta = real["metadata"]
norm_meta = normalizer["metadata"]
if norm_meta.get("reward_version") != "v1":
    raise ValueError("Frozen normalizer reward is not v1.")
if norm_meta.get("normalized_G_max") != 5.0:
    raise ValueError("Frozen normalizer normalized_G_max is not 5.")
if norm_meta.get("source_dataset_sha256") != real_meta.get("source_dataset_sha256"):
    raise ValueError("Normalizer and real replay dataset hashes differ.")
if norm_meta.get("reward_config_sha256") != real_meta.get("reward_config_sha256"):
    raise ValueError("Normalizer and real replay reward configs differ.")
if norm_meta.get("source_collector_types") != [0, 1, 2, 3, 4]:
    raise ValueError("Normalizer is not bound to all mixed collector types.")
PY

printf '%s\n' \
  "stage=policy" \
  "condition=${condition}" \
  "gpu=${physical_gpu}" \
  "status=running" \
  "rwm_checkpoint=${checkpoint}" \
  "real_replay=${real_replay}" \
  "reward_normalization=frozen" \
  "reward_normalizer=${reward_normalizer}" \
  "started_at=$(date --iso-8601=seconds)" \
  > "${status_path}"

CUDA_VISIBLE_DEVICES="${physical_gpu}" \
"${REPO_ROOT}/.venv/bin/python" \
  "${SCRIPT_DIR}/train_flashsac_world_model_go2_v13_fullstate.py" \
  --config_path "${CONFIG}" \
  --model_resume_path "${checkpoint}" \
  --dataset_path "${dataset}" \
  --device cuda:0 \
  --save_path "${SAVE_ROOT}" \
  --overrides "replay_mix.real_replay_path=${real_replay}" \
  --overrides "reward_normalization.checkpoint_path=${reward_normalizer}"

final_step=48828
[[ -f "${SAVE_ROOT}/step${final_step}/actor.pt" ]] \
  || fail "Final actor checkpoint is missing."
[[ -f "${SAVE_ROOT}/step${final_step}/critic.pt" ]] \
  || fail "Final critic checkpoint is missing."
[[ -f "${SAVE_ROOT}/v12_replay_mix_manifest.json" ]] \
  || fail "Replay-mix launch manifest is missing."

printf '%s\n' \
  "stage=policy" \
  "condition=${condition}" \
  "gpu=${physical_gpu}" \
  "status=completed" \
  "rwm_checkpoint=${checkpoint}" \
  "policy_checkpoint=${SAVE_ROOT}/step${final_step}" \
  "completed_at=$(date --iso-8601=seconds)" \
  > "${status_path}"
