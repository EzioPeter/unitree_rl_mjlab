#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

: "${EXPERT_POLICY_PATH:?Set EXPERT_POLICY_PATH to the 0.5-strength proprioceptive expert checkpoint directory.}"

DATASET_PATH="${DATASET_PATH:-logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt}"
COLLECT_TASK="${COLLECT_TASK:-Unitree-Go2-Flat-Broken-RR-Calf-Proprioceptive-Expert}"
COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-1024}"
COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS:-1000000}"
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-200000}"
COLLECT_DEVICE="${COLLECT_DEVICE:-cuda:0}"
DATASET_OBS_KIND="${DATASET_OBS_KIND:-proprioceptive}"

MEDIUM_POLICY_PATH="${MEDIUM_POLICY_PATH:-}"
JOINT_STRENGTH_SCALES="${JOINT_STRENGTH_SCALES:-RR_calf_joint=0.5}"
BROKEN_JOINT_NAMES="${BROKEN_JOINT_NAMES:-}"

COLLECTOR_MIX="${COLLECTOR_MIX:-expert:0.45,noisy_expert:0.25,medium:0.10,failure_border:0.15,random:0.05}"
ACTION_NOISE_STD="${ACTION_NOISE_STD:-0.12}"
MEDIUM_ACTION_NOISE_STD="${MEDIUM_ACTION_NOISE_STD:-0.25}"
FAILURE_ACTION_NOISE_STD="${FAILURE_ACTION_NOISE_STD:-0.45}"

COMMAND_MODES="${COMMAND_MODES:-stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw}"
COMMAND_MODE_WEIGHTS="${COMMAND_MODE_WEIGHTS:-stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13}"
X_ABS_RANGE_MIN="${X_ABS_RANGE_MIN:-0.05}"
X_ABS_RANGE_MAX="${X_ABS_RANGE_MAX:-0.5}"
Y_ABS_RANGE_MIN="${Y_ABS_RANGE_MIN:-0.03}"
Y_ABS_RANGE_MAX="${Y_ABS_RANGE_MAX:-0.2}"
YAW_ABS_RANGE_MIN="${YAW_ABS_RANGE_MIN:-0.05}"
YAW_ABS_RANGE_MAX="${YAW_ABS_RANGE_MAX:-0.4}"
COMMAND_RESAMPLE_INTERVAL_MIN="${COMMAND_RESAMPLE_INTERVAL_MIN:-120}"
COMMAND_RESAMPLE_INTERVAL_MAX="${COMMAND_RESAMPLE_INTERVAL_MAX:-300}"

USE_DOMAIN_RANDOMIZATION="${USE_DOMAIN_RANDOMIZATION:-false}"
USE_PUSH_RANDOMIZATION="${USE_PUSH_RANDOMIZATION:-false}"
USE_OBSERVATION_NOISE="${USE_OBSERVATION_NOISE:-false}"
RESET_PARTS="${RESET_PARTS:-true}"

BROKEN_JOINT_ARGS=()
if [[ -n "${BROKEN_JOINT_NAMES}" ]]; then
  read -r -a BROKEN_JOINT_VALUES <<< "${BROKEN_JOINT_NAMES}"
  BROKEN_JOINT_ARGS=(--broken_joint_names "${BROKEN_JOINT_VALUES[@]}")
fi

JOINT_STRENGTH_ARGS=()
if [[ -n "${JOINT_STRENGTH_SCALES}" ]]; then
  read -r -a JOINT_STRENGTH_VALUES <<< "${JOINT_STRENGTH_SCALES}"
  JOINT_STRENGTH_ARGS=(--joint_strength_scales "${JOINT_STRENGTH_VALUES[@]}")
fi

RANDOMIZATION_ARGS=()
if [[ "${USE_DOMAIN_RANDOMIZATION}" == "true" ]]; then
  RANDOMIZATION_ARGS+=(--use_domain_randomization)
else
  RANDOMIZATION_ARGS+=(--no-use_domain_randomization)
fi
if [[ "${USE_PUSH_RANDOMIZATION}" == "true" ]]; then
  RANDOMIZATION_ARGS+=(--use_push_randomization)
else
  RANDOMIZATION_ARGS+=(--no-use_push_randomization)
fi
if [[ "${USE_OBSERVATION_NOISE}" == "true" ]]; then
  RANDOMIZATION_ARGS+=(--use_observation_noise)
else
  RANDOMIZATION_ARGS+=(--no-use_observation_noise)
fi

mkdir -p "$(dirname "${DATASET_PATH}")"
if [[ "${RESET_PARTS}" == "true" ]]; then
  rm -rf "$(dirname "${DATASET_PATH}")/parts"
fi

uv run python scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \
  --task "${COLLECT_TASK}" \
  --device "${COLLECT_DEVICE}" \
  --num_envs "${COLLECT_NUM_ENVS}" \
  --num_transitions "${COLLECT_NUM_TRANSITIONS}" \
  --save_path "${DATASET_PATH}" \
  --expert_policy_path "${EXPERT_POLICY_PATH}" \
  --medium_policy_path "${MEDIUM_POLICY_PATH}" \
  --collector_mix "${COLLECTOR_MIX}" \
  --dataset_obs_kind "${DATASET_OBS_KIND}" \
  --action_noise_std "${ACTION_NOISE_STD}" \
  --medium_action_noise_std "${MEDIUM_ACTION_NOISE_STD}" \
  --failure_action_noise_std "${FAILURE_ACTION_NOISE_STD}" \
  --chunk_size "${COLLECT_CHUNK_SIZE}" \
  --command_modes "${COMMAND_MODES}" \
  --command_mode_weights "${COMMAND_MODE_WEIGHTS}" \
  --x_range "-${X_ABS_RANGE_MAX}" "${X_ABS_RANGE_MAX}" \
  --signed_x \
  --x_abs_range "${X_ABS_RANGE_MIN}" "${X_ABS_RANGE_MAX}" \
  --y_abs_range "${Y_ABS_RANGE_MIN}" "${Y_ABS_RANGE_MAX}" \
  --yaw_abs_range "${YAW_ABS_RANGE_MIN}" "${YAW_ABS_RANGE_MAX}" \
  --command_resample_interval_min "${COMMAND_RESAMPLE_INTERVAL_MIN}" \
  --command_resample_interval_max "${COMMAND_RESAMPLE_INTERVAL_MAX}" \
  "${BROKEN_JOINT_ARGS[@]}" \
  "${JOINT_STRENGTH_ARGS[@]}" \
  "${RANDOMIZATION_ARGS[@]}" \
  --headless
