#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
MANIFEST="${V13_SIM_MANIFEST:-${REPO_ROOT}/V13_SIM_DATASET_MANIFEST.json}"
EXPERIMENT_NAME="${V13_RWM_EXPERIMENT_NAME:-go2_rwm_v13_full45_25k_fresh_seed0_20260725}"
QUEUE_ROOT="${V13_RWM_QUEUE_ROOT:-${REPO_ROOT}/logs/queues/v13_rwm_full45_fresh_seed0_20260725}"
RUNNER="${SCRIPT_DIR}/run_v13_rwm45to45_fullstate.sh"
CONFIG="${SCRIPT_DIR}/configs/go2_offline_world_model_v13_full45.yaml"
TRAINER="${SCRIPT_DIR}/train_world_model_offline_go2_v13_fullstate.py"
BASE_TRAINER="${SCRIPT_DIR}/train_world_model_offline_go2.py"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

mark_failed() {
  local condition="$1"
  local gpu="$2"
  local exit_status="$3"
  shift 3
  local message="$*"
  local status_dir="${QUEUE_ROOT}/status/${condition}"
  mkdir -p "${status_dir}"
  printf '%s\n' \
    "stage=rwm" \
    "condition=${condition}" \
    "gpu=${gpu}" \
    "status=failed" \
    "exit_status=${exit_status}" \
    "message=${message}" \
    "failed_at=$(date --iso-8601=seconds)" \
    > "${status_dir}/rwm.status"
  printf 'condition=%s gpu=%s exit_status=%s message=%s\n' \
    "${condition}" "${gpu}" "${exit_status}" "${message}" \
    > "${QUEUE_ROOT}/FAILED_DO_NOT_CONTINUE.txt"
}

run_worker() {
  local gpu="$1"
  local condition="$2"
  [[ "${gpu}" != "2" && "${gpu}" != "7" ]] || fail "GPU ${gpu} is forbidden."
  local status_dir="${QUEUE_ROOT}/status/${condition}"
  local output_root="${REPO_ROOT}/logs/experiments/${EXPERIMENT_NAME}/${condition}/rwm"
  mkdir -p "${status_dir}"
  printf '%s\n' \
    "stage=rwm" \
    "condition=${condition}" \
    "gpu=${gpu}" \
    "status=running" \
    "mode=fresh" \
    "seed=0" \
    "experiment_name=${EXPERIMENT_NAME}" \
    "started_at=$(date --iso-8601=seconds)" \
    > "${status_dir}/rwm.status"

  local exit_status=0
  if ! V13_RWM_EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
    "${RUNNER}" "${condition}" "${gpu}" "${MANIFEST}"; then
    exit_status="$?"
    [[ "${exit_status}" -ne 0 ]] || exit_status=1
    mark_failed "${condition}" "${gpu}" "${exit_status}" "RWM runner failed."
    return "${exit_status}"
  fi

  mapfile -t checkpoints < <(
    find "${output_root}" -mindepth 2 -maxdepth 2 -type f -name model_5000.pt -print
  )
  if [[ "${#checkpoints[@]}" -ne 1 ]]; then
    mark_failed "${condition}" "${gpu}" 1 \
      "Expected exactly one model_5000.pt; found ${#checkpoints[@]}."
    return 1
  fi
  local checkpoint="${checkpoints[0]}"
  if ! CHECKPOINT="${checkpoint}" "${REPO_ROOT}/.venv/bin/python" - <<'PY'
import os
from pathlib import Path

import torch

path = Path(os.environ["CHECKPOINT"]).resolve()
checkpoint = torch.load(path, map_location="cpu", weights_only=False)
assert int(checkpoint.get("iter", -1)) == 5000
assert isinstance(checkpoint.get("system_dynamics_state_dict"), dict)
assert isinstance(checkpoint.get("system_dynamics_optimizer_state_dict"), dict)
rng = checkpoint.get("rng_state")
assert isinstance(rng, dict)
assert set(rng) == {"python", "numpy", "torch_cpu", "torch_cuda"}
infos = checkpoint.get("infos") or {}
assert int(infos.get("state_dim", -1)) == 45
assert int(infos.get("action_dim", -1)) == 12
normalizer = checkpoint.get("normalizer") or {}
assert tuple(normalizer["state_mean"].shape) == (45,)
assert tuple(normalizer["obs_mean"].shape) == (48,)
interface = (checkpoint.get("config") or {}).get("v13_interface") or {}
assert int(interface.get("rwm_input_state_dim", -1)) == 45
assert int(interface.get("rwm_output_state_dim", -1)) == 45
assert int(interface.get("actor_observation_dim", -1)) == 48
assert int(interface.get("critic_observation_dim", -1)) == 48
PY
  then
    mark_failed "${condition}" "${gpu}" 1 "Final RWM checkpoint validation failed."
    return 1
  fi

  printf '%s\n' \
    "stage=rwm" \
    "condition=${condition}" \
    "gpu=${gpu}" \
    "status=completed" \
    "mode=fresh" \
    "seed=0" \
    "checkpoint=${checkpoint}" \
    "checkpoint_sha256=$(sha256sum "${checkpoint}" | awk '{print $1}')" \
    "completed_at=$(date --iso-8601=seconds)" \
    > "${status_dir}/rwm.status"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ "$#" -ge 2 ]] || fail "worker mode requires <gpu> <condition> [condition...]."
  worker_gpu="$1"
  shift
  for worker_condition in "$@"; do
    run_worker "${worker_gpu}" "${worker_condition}"
  done
  exit 0
fi

[[ "$#" -eq 0 ]] || fail "This launcher takes no positional arguments."
for path in "${MANIFEST}" "${RUNNER}" "${CONFIG}" "${TRAINER}" "${BASE_TRAINER}"; do
  [[ -f "${path}" ]] || fail "Missing required file: ${path}"
done
[[ ! -e "${QUEUE_ROOT}" ]] || fail "Queue root already exists: ${QUEUE_ROOT}"

declare -A gpu_by_condition=(
  [g0]=0
  [rr03]=0
  [rr05]=5
  [p5]=5
  [p75]=6
)
conditions=(g0 rr03 rr05 p5 p75)

for condition in "${conditions[@]}"; do
  gpu="${gpu_by_condition[${condition}]}"
  [[ "${gpu}" != "2" && "${gpu}" != "7" ]] || fail "Forbidden GPU assignment."
  output_root="${REPO_ROOT}/logs/experiments/${EXPERIMENT_NAME}/${condition}/rwm"
  [[ ! -e "${output_root}" ]] || fail "Fresh output root already exists: ${output_root}"
done

mkdir -p "${QUEUE_ROOT}/logs" "${QUEUE_ROOT}/status"
QUEUE_ROOT="${QUEUE_ROOT}" \
MANIFEST="${MANIFEST}" \
RUNNER="${RUNNER}" \
CONFIG="${CONFIG}" \
TRAINER="${TRAINER}" \
BASE_TRAINER="${BASE_TRAINER}" \
EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
"${REPO_ROOT}/.venv/bin/python" - <<'PY'
import hashlib
import json
import os
from datetime import datetime
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


files = {
    name: Path(os.environ[name])
    for name in ("MANIFEST", "RUNNER", "CONFIG", "TRAINER", "BASE_TRAINER")
}
record = {
    "schema": "go2_v13_rwm_full45_fresh_queue_v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "experiment_name": os.environ["EXPERIMENT_NAME"],
    "all_fresh": True,
    "seed": 0,
    "rng_checkpointing": True,
    "files": {
        name.lower(): {"path": str(path), "sha256": sha256(path)}
        for name, path in files.items()
    },
    "forbidden_gpus": [2, 7],
    "workers": {
        "gpu0": ["g0", "rr03"],
        "gpu5": ["rr05", "p5"],
        "gpu6": ["p75"],
    },
}
path = Path(os.environ["QUEUE_ROOT"]) / "queue_manifest.json"
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

launch_worker() {
  local gpu="$1"
  shift
  local assigned=("$@")
  V13_RWM_EXPERIMENT_NAME="${EXPERIMENT_NAME}" \
  V13_RWM_QUEUE_ROOT="${QUEUE_ROOT}" \
  nohup "${BASH_SOURCE[0]}" --worker "${gpu}" "${assigned[@]}" \
    > "${QUEUE_ROOT}/logs/gpu${gpu}.log" 2>&1 &
  local worker_pid="$!"
  printf '%s\n' "${worker_pid}" > "${QUEUE_ROOT}/gpu${gpu}.pid"
  local condition
  for condition in "${assigned[@]}"; do
    printf '%s\n' "${worker_pid}" > "${QUEUE_ROOT}/${condition}.pid"
  done
}

launch_worker 0 g0 rr03
launch_worker 5 rr05 p5
launch_worker 6 p75

echo "V13 fresh RWM queue launched at ${QUEUE_ROOT}"
