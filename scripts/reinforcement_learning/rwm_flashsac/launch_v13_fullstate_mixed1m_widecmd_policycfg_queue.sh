#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
RWM_EXPERIMENT_NAME="${V13_RWM_EXPERIMENT_NAME:-go2_rwm_v13_full45_25k_fresh_seed0_20260725}"
POLICY_EXPERIMENT_NAME="${V13_POLICY_EXPERIMENT_NAME:-go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m_fresh_seed0_20260725}"
RWM_QUEUE="${V13_RWM_QUEUE_ROOT:-${REPO_ROOT}/logs/queues/v13_rwm_full45_fresh_seed0_20260725}"
POLICY_QUEUE="${V13_POLICY_QUEUE_ROOT:-${REPO_ROOT}/logs/queues/v13_policy_fullstate_mixed1m_widecmd_policycfg_fresh_seed0_20260725}"
RUNNER="${SCRIPT_DIR}/run_v13_fullstate_rwm_mixed1m_widecmd_policycfg.sh"
CONFIG="${SCRIPT_DIR}/configs/go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m.yaml"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

mark_failed() {
  local condition="$1"
  local gpu="$2"
  shift 2
  local message="$*"
  mkdir -p "${POLICY_QUEUE}/status/${condition}"
  printf '%s\n' \
    "stage=policy" \
    "condition=${condition}" \
    "gpu=${gpu}" \
    "status=failed" \
    "message=${message}" \
    "failed_at=$(date --iso-8601=seconds)" \
    > "${POLICY_QUEUE}/status/${condition}/policy.status"
  printf 'condition=%s gpu=%s message=%s\n' \
    "${condition}" "${gpu}" "${message}" \
    > "${POLICY_QUEUE}/FAILED_DO_NOT_CONTINUE.txt"
}

wait_for_rwm() {
  local condition="$1"
  local status="${RWM_QUEUE}/status/${condition}/rwm.status"
  local pid_file="${RWM_QUEUE}/${condition}.pid"
  while true; do
    if [[ -f "${RWM_QUEUE}/FAILED_DO_NOT_CONTINUE.txt" ]]; then
      return 1
    fi
    if [[ -f "${status}" ]] && grep -qx "status=completed" "${status}"; then
      return 0
    fi
    if [[ -f "${status}" ]] && grep -qx "status=failed" "${status}"; then
      return 1
    fi
    if [[ -f "${pid_file}" ]]; then
      local worker_pid
      worker_pid="$(<"${pid_file}")"
      if [[ "${worker_pid}" =~ ^[0-9]+$ ]] && ! kill -0 "${worker_pid}" 2>/dev/null; then
        return 1
      fi
    fi
    sleep 15
  done
}

run_worker() {
  local gpu="$1"
  [[ "${gpu}" != "2" && "${gpu}" != "7" ]] || fail "GPU ${gpu} is forbidden."
  shift
  local conditions=("$@")
  local condition
  for condition in "${conditions[@]}"; do
    if ! wait_for_rwm "${condition}"; then
      mark_failed "${condition}" "${gpu}" "RWM ${condition} failed or its worker died."
      return 1
    fi
  done
  for condition in "${conditions[@]}"; do
    if ! V13_RWM_EXPERIMENT_NAME="${RWM_EXPERIMENT_NAME}" \
      V13_POLICY_EXPERIMENT_NAME="${POLICY_EXPERIMENT_NAME}" \
      V13_POLICY_STATUS_ROOT="${POLICY_QUEUE}/status" \
      "${RUNNER}" "${condition}" "${gpu}"; then
      mark_failed "${condition}" "${gpu}" "Policy runner failed."
      return 1
    fi
  done
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  [[ "$#" -ge 2 ]] || fail "worker mode requires <gpu> <condition> [condition...]."
  run_worker "$@"
  exit 0
fi

[[ "$#" -eq 0 ]] || fail "This launcher takes no positional arguments."
[[ -d "${RWM_QUEUE}" ]] || fail "Missing fresh RWM queue."
[[ -x "${RUNNER}" ]] || fail "Missing executable policy runner."
[[ -f "${CONFIG}" ]] || fail "Missing frozen V13 policy config."
[[ ! -e "${POLICY_QUEUE}" ]] || fail "Policy queue root already exists."

declare -A gpu_by_condition=(
  [g0]=0
  [rr03]=0
  [rr05]=5
  [p5]=5
  [p75]=6
)
conditions=(g0 rr03 rr05 p5 p75)
for condition in "${conditions[@]}"; do
  policy_root="${REPO_ROOT}/logs/experiments/${POLICY_EXPERIMENT_NAME}/${condition}"
  [[ ! -e "${policy_root}" ]] || fail "Policy root already exists: ${policy_root}"
done

mkdir -p "${POLICY_QUEUE}/logs" "${POLICY_QUEUE}/status"
POLICY_QUEUE="${POLICY_QUEUE}" \
RWM_QUEUE="${RWM_QUEUE}" \
RWM_EXPERIMENT_NAME="${RWM_EXPERIMENT_NAME}" \
POLICY_EXPERIMENT_NAME="${POLICY_EXPERIMENT_NAME}" \
RUNNER="${RUNNER}" \
CONFIG="${CONFIG}" \
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


runner = Path(os.environ["RUNNER"])
config = Path(os.environ["CONFIG"])
record = {
    "schema": "go2_v13_fullstate_mixed1m_widecmd_policycfg_fresh_queue_v1",
    "created_at": datetime.now().astimezone().isoformat(),
    "rwm_queue": os.environ["RWM_QUEUE"],
    "rwm_experiment_name": os.environ["RWM_EXPERIMENT_NAME"],
    "policy_experiment_name": os.environ["POLICY_EXPERIMENT_NAME"],
    "runner_path": str(runner),
    "runner_sha256": sha256(runner),
    "config_path": str(config),
    "config_sha256": sha256(config),
    "model_based_reference_commit": "daa2eae7324255d4011964d2e74199547f347f26",
    "forbidden_gpus": [2, 7],
    "workers": {
        "gpu0": ["g0", "rr03"],
        "gpu5": ["rr05", "p5"],
        "gpu6": ["p75"],
    },
    "policy": {
        "num_imagination_envs": 1024,
        "num_env_steps": 50_000_000,
        "updates_per_interaction_step": 2,
        "actor_observation_dim": 48,
        "critic_observation_dim": 48,
        "batch_size": 2048,
        "buffer_max_length": 10_000_000,
        "buffer_min_length": 100_000,
        "buffer_device_type": "cpu",
        "n_step": 3,
        "reward_version": "v1",
        "normalize_reward": True,
        "reward_normalization_enabled": True,
        "reward_normalization_frozen": True,
        "normalized_G_max": 5.0,
        "critic_min_v": -5.0,
        "critic_max_v": 5.0,
        "uncertainty_penalty_weight": -2.0,
        "command_ranges": {"vx": [-0.5, 0.5], "vy": [-0.2, 0.2], "yaw": [-0.4, 0.4]},
        "real_rows": 102,
        "rwm_rows": 1946,
        "sim_rows": 0,
        "bc": False,
        "seed": 0,
    },
}
path = Path(os.environ["POLICY_QUEUE"]) / "queue_manifest.json"
path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

launch_worker() {
  local gpu="$1"
  shift
  local assigned=("$@")
  V13_RWM_EXPERIMENT_NAME="${RWM_EXPERIMENT_NAME}" \
  V13_POLICY_EXPERIMENT_NAME="${POLICY_EXPERIMENT_NAME}" \
  V13_RWM_QUEUE_ROOT="${RWM_QUEUE}" \
  V13_POLICY_QUEUE_ROOT="${POLICY_QUEUE}" \
  nohup "${BASH_SOURCE[0]}" --worker "${gpu}" "${assigned[@]}" \
    > "${POLICY_QUEUE}/logs/gpu${gpu}.log" 2>&1 &
  local worker_pid="$!"
  printf '%s\n' "${worker_pid}" > "${POLICY_QUEUE}/gpu${gpu}.pid"
  local condition
  for condition in "${assigned[@]}"; do
    printf '%s\n' "${worker_pid}" > "${POLICY_QUEUE}/${condition}.pid"
  done
}

launch_worker 0 g0 rr03
launch_worker 5 rr05 p5
launch_worker 6 p75

echo "V13 fresh full-state policy queue armed at ${POLICY_QUEUE}"
