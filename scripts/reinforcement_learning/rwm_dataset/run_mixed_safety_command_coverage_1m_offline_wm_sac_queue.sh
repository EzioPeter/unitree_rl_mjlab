#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

DATASET_PATH="${DATASET_PATH:-logs/rwm_datasets/go2_flat_mixed_safety_command_coverage_1m/dataset.pt}"
EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH:-logs/model_based/go2_flat_flashsac_rwm_sacwm/2026-06-05_17-27-59/step48828}"
MEDIUM_POLICY_PATH="${MEDIUM_POLICY_PATH:-}"
WM_SAVE_BASE="${WM_SAVE_BASE:-logs/rsl_rl/go2_flat_rwm_offline_mixed_safety_command_coverage_1m_aligned_hidden}"
SAC_SAVE_BASE="${SAC_SAVE_BASE:-logs/model_based/go2_flat_flashsac_rwm_mixed_safety_command_coverage_1m}"
QUEUE_ROOT="${QUEUE_ROOT:-logs/queues/offline_wm_sac_mixed_safety_cmdcov1m/$(date +%Y-%m-%d_%H-%M-%S)}"
SKIP_COLLECT="${SKIP_COLLECT:-false}"

COLLECT_TASK="${COLLECT_TASK:-Unitree-Go2-Flat-RWM-Pretrain-Ens}"
COLLECT_NUM_ENVS="${COLLECT_NUM_ENVS:-1024}"
COLLECT_NUM_TRANSITIONS="${COLLECT_NUM_TRANSITIONS:-1000000}"
COLLECT_CHUNK_SIZE="${COLLECT_CHUNK_SIZE:-200000}"
COLLECT_DEVICE="${COLLECT_DEVICE:-cuda:0}"

# This mix keeps mostly good locomotion data but adds enough off-manifold and
# near-failure transitions for the termination/contact head to become useful.
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

WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-5000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE:-256}"
WM_DEVICE="${WM_DEVICE:-cuda:0}"

SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS:-1024}"
SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS:-50000000}"
SAC_DEVICE="${SAC_DEVICE:-cuda:0}"
SAC_UPDATES_PER_INTERACTION_STEP="${SAC_UPDATES_PER_INTERACTION_STEP:-2}"
SAC_BUFFER_MAX_LENGTH="${SAC_BUFFER_MAX_LENGTH:-10000000}"
SAC_BUFFER_MIN_LENGTH="${SAC_BUFFER_MIN_LENGTH:-100000}"
SAC_BUFFER_DEVICE_TYPE="${SAC_BUFFER_DEVICE_TYPE:-cpu}"
SAC_SAMPLE_BATCH_SIZE="${SAC_SAMPLE_BATCH_SIZE:-2048}"
SAC_N_STEP="${SAC_N_STEP:-3}"
SAC_NORMALIZE_REWARD="${SAC_NORMALIZE_REWARD:-true}"
SAC_NORMALIZED_G_MAX="${SAC_NORMALIZED_G_MAX:-5.0}"
SAC_CRITIC_MIN_V="${SAC_CRITIC_MIN_V:--5.0}"
SAC_CRITIC_MAX_V="${SAC_CRITIC_MAX_V:-5.0}"
SAC_SAVE_INTERVAL="${SAC_SAVE_INTERVAL:-1000}"
SAC_UNCERTAINTY_PENALTY_WEIGHT="${SAC_UNCERTAINTY_PENALTY_WEIGHT:--2.0}"
SAC_LIN_VEL_X_MIN="${SAC_LIN_VEL_X_MIN:--0.3}"
SAC_LIN_VEL_X_MAX="${SAC_LIN_VEL_X_MAX:-0.3}"
SAC_LIN_VEL_Y_MIN="${SAC_LIN_VEL_Y_MIN:--0.15}"
SAC_LIN_VEL_Y_MAX="${SAC_LIN_VEL_Y_MAX:-0.15}"
SAC_ANG_VEL_Z_MIN="${SAC_ANG_VEL_Z_MIN:--0.3}"
SAC_ANG_VEL_Z_MAX="${SAC_ANG_VEL_Z_MAX:-0.3}"

mkdir -p "${QUEUE_ROOT}" "${WM_SAVE_BASE}" "${SAC_SAVE_BASE}" "$(dirname "${DATASET_PATH}")"

STATE_FILE="${QUEUE_ROOT}/state.txt"
SUMMARY_FILE="${QUEUE_ROOT}/summary.txt"

write_state() {
  local stage="$1"
  local status="$2"
  {
    echo "time=$(date --iso-8601=seconds)"
    echo "stage=${stage}"
    echo "status=${status}"
    echo "queue_root=${QUEUE_ROOT}"
  } > "${STATE_FILE}"
}

write_summary_header() {
  {
    echo "queue_root=${QUEUE_ROOT}"
    echo "dataset_path=${DATASET_PATH}"
    echo "expert_policy_path=${EXPERT_POLICY_PATH}"
    echo "medium_policy_path=${MEDIUM_POLICY_PATH}"
    echo "wm_save_base=${WM_SAVE_BASE}"
    echo "sac_save_base=${SAC_SAVE_BASE}"
    echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
    echo "collector_mix=${COLLECTOR_MIX}"
    echo "action_noise_std=${ACTION_NOISE_STD}"
    echo "medium_action_noise_std=${MEDIUM_ACTION_NOISE_STD}"
    echo "failure_action_noise_std=${FAILURE_ACTION_NOISE_STD}"
    echo "command_mode_weights=${COMMAND_MODE_WEIGHTS}"
    echo "collect_x_abs_range=${X_ABS_RANGE_MIN},${X_ABS_RANGE_MAX}"
    echo "collect_y_abs_range=${Y_ABS_RANGE_MIN},${Y_ABS_RANGE_MAX}"
    echo "collect_yaw_abs_range=${YAW_ABS_RANGE_MIN},${YAW_ABS_RANGE_MAX}"
    echo "sac_command_range_x=${SAC_LIN_VEL_X_MIN},${SAC_LIN_VEL_X_MAX}"
    echo "sac_command_range_y=${SAC_LIN_VEL_Y_MIN},${SAC_LIN_VEL_Y_MAX}"
    echo "sac_command_range_yaw=${SAC_ANG_VEL_Z_MIN},${SAC_ANG_VEL_Z_MAX}"
    echo "sac_uncertainty_penalty_weight=${SAC_UNCERTAINTY_PENALTY_WEIGHT}"
    echo "skip_collect=${SKIP_COLLECT}"
    echo "started_at=$(date --iso-8601=seconds)"
    echo
  } > "${SUMMARY_FILE}"
}

run_stage() {
  local stage="$1"
  local cmd_file="$2"
  local log_file="$3"
  write_state "${stage}" "running"
  echo "[queue] $(date --iso-8601=seconds) starting ${stage}" | tee -a "${SUMMARY_FILE}"
  bash "${cmd_file}" > "${log_file}" 2>&1
  echo "[queue] $(date --iso-8601=seconds) completed ${stage}" | tee -a "${SUMMARY_FILE}"
  write_state "${stage}" "completed"
}

write_summary_header
write_state "precheck" "running"

cat > "${QUEUE_ROOT}/01_collect_dataset_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
rm -rf "$(dirname "${DATASET_PATH}")/parts"
uv run python scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py \\
  --task "${COLLECT_TASK}" \\
  --device "${COLLECT_DEVICE}" \\
  --num_envs "${COLLECT_NUM_ENVS}" \\
  --num_transitions "${COLLECT_NUM_TRANSITIONS}" \\
  --save_path "${DATASET_PATH}" \\
  --expert_policy_path "${EXPERT_POLICY_PATH}" \\
  --medium_policy_path "${MEDIUM_POLICY_PATH}" \\
  --collector_mix "${COLLECTOR_MIX}" \\
  --action_noise_std "${ACTION_NOISE_STD}" \\
  --medium_action_noise_std "${MEDIUM_ACTION_NOISE_STD}" \\
  --failure_action_noise_std "${FAILURE_ACTION_NOISE_STD}" \\
  --chunk_size "${COLLECT_CHUNK_SIZE}" \\
  --command_modes "${COMMAND_MODES}" \\
  --command_mode_weights "${COMMAND_MODE_WEIGHTS}" \\
  --x_range "-${X_ABS_RANGE_MAX}" "${X_ABS_RANGE_MAX}" \\
  --signed_x \\
  --x_abs_range "${X_ABS_RANGE_MIN}" "${X_ABS_RANGE_MAX}" \\
  --y_abs_range "${Y_ABS_RANGE_MIN}" "${Y_ABS_RANGE_MAX}" \\
  --yaw_abs_range "${YAW_ABS_RANGE_MIN}" "${YAW_ABS_RANGE_MAX}" \\
  --command_resample_interval_min "${COMMAND_RESAMPLE_INTERVAL_MIN}" \\
  --command_resample_interval_max "${COMMAND_RESAMPLE_INTERVAL_MAX}" \\
  --no-use_domain_randomization \\
  --no-use_push_randomization \\
  --no-use_observation_noise \\
  --headless
EOF
chmod +x "${QUEUE_ROOT}/01_collect_dataset_command.sh"

if [[ "${SKIP_COLLECT}" == "true" && -f "${DATASET_PATH}" ]]; then
  echo "[queue] $(date --iso-8601=seconds) skipped collect_dataset; found ${DATASET_PATH}" | tee -a "${SUMMARY_FILE}"
  write_state "collect_dataset" "skipped"
else
  run_stage "collect_dataset" \
    "${QUEUE_ROOT}/01_collect_dataset_command.sh" \
    "${QUEUE_ROOT}/01_collect_dataset.log"
fi

if [[ ! -f "${DATASET_PATH}" ]]; then
  write_state "collect_dataset" "failed"
  echo "[queue] dataset not found after collection: ${DATASET_PATH}" | tee -a "${SUMMARY_FILE}"
  exit 1
fi

cat > "${QUEUE_ROOT}/02_train_offline_wm_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
uv run python scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2.py \\
  --config_path scripts/reinforcement_learning/rwm_dataset/configs/go2_offline_world_model.yaml \\
  --dataset_path "${DATASET_PATH}" \\
  --save_dir "${WM_SAVE_BASE}" \\
  --max_iterations "${WM_MAX_ITERATIONS}" \\
  --batch_size "${WM_BATCH_SIZE}" \\
  --micro_batch_size "${WM_MICRO_BATCH_SIZE}" \\
  --device "${WM_DEVICE}"
EOF
chmod +x "${QUEUE_ROOT}/02_train_offline_wm_command.sh"

run_stage "offline_world_model" \
  "${QUEUE_ROOT}/02_train_offline_wm_command.sh" \
  "${QUEUE_ROOT}/02_train_offline_wm.log"

WM_RUN_DIR="$(find "${WM_SAVE_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
if [[ -z "${WM_RUN_DIR}" ]]; then
  write_state "find_world_model" "failed"
  echo "[queue] no world-model run directory found under ${WM_SAVE_BASE}" | tee -a "${SUMMARY_FILE}"
  exit 1
fi

WM_MODEL_PATH="${WM_RUN_DIR}/model_${WM_MAX_ITERATIONS}.pt"
if [[ ! -f "${WM_MODEL_PATH}" ]]; then
  WM_MODEL_PATH="${WM_RUN_DIR}/latest.pt"
fi
if [[ ! -f "${WM_MODEL_PATH}" ]]; then
  write_state "find_world_model" "failed"
  echo "[queue] no model_${WM_MAX_ITERATIONS}.pt or latest.pt found in ${WM_RUN_DIR}" | tee -a "${SUMMARY_FILE}"
  exit 1
fi

{
  echo "wm_run_dir=${WM_RUN_DIR}"
  echo "wm_model_path=${WM_MODEL_PATH}"
  echo
} >> "${SUMMARY_FILE}"

cat > "${QUEUE_ROOT}/03_train_sac_command.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
cd "${REPO_ROOT}"
export WANDB_MODE="${WANDB_MODE}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}"
uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2.py \\
  --config_path scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm.yaml \\
  --model_resume_path "${WM_MODEL_PATH}" \\
  --dataset_path "${DATASET_PATH}" \\
  --num_imagination_envs "${SAC_NUM_IMAGINATION_ENVS}" \\
  --num_env_steps "${SAC_NUM_ENV_STEPS}" \\
  --device "${SAC_DEVICE}" \\
  --save_path "${SAC_SAVE_BASE}/TIMESTAMP" \\
  --overrides updates_per_interaction_step="${SAC_UPDATES_PER_INTERACTION_STEP}" \\
  --overrides save_checkpoint_per_interaction_step="${SAC_SAVE_INTERVAL}" \\
  --overrides world_model.uncertainty_penalty_weight="${SAC_UNCERTAINTY_PENALTY_WEIGHT}" \\
  --overrides world_model.lin_vel_x_min="${SAC_LIN_VEL_X_MIN}" \\
  --overrides world_model.lin_vel_x_max="${SAC_LIN_VEL_X_MAX}" \\
  --overrides world_model.lin_vel_y_min="${SAC_LIN_VEL_Y_MIN}" \\
  --overrides world_model.lin_vel_y_max="${SAC_LIN_VEL_Y_MAX}" \\
  --overrides world_model.ang_vel_z_min="${SAC_ANG_VEL_Z_MIN}" \\
  --overrides world_model.ang_vel_z_max="${SAC_ANG_VEL_Z_MAX}" \\
  --overrides agent.buffer_max_length="${SAC_BUFFER_MAX_LENGTH}" \\
  --overrides agent.buffer_min_length="${SAC_BUFFER_MIN_LENGTH}" \\
  --overrides agent.buffer_device_type="${SAC_BUFFER_DEVICE_TYPE}" \\
  --overrides agent.sample_batch_size="${SAC_SAMPLE_BATCH_SIZE}" \\
  --overrides agent.n_step="${SAC_N_STEP}" \\
  --overrides agent.normalize_reward="${SAC_NORMALIZE_REWARD}" \\
  --overrides agent.normalized_G_max="${SAC_NORMALIZED_G_MAX}" \\
  --overrides agent.critic_min_v="${SAC_CRITIC_MIN_V}" \\
  --overrides agent.critic_max_v="${SAC_CRITIC_MAX_V}"
EOF
chmod +x "${QUEUE_ROOT}/03_train_sac_command.sh"

run_stage "sac_policy" \
  "${QUEUE_ROOT}/03_train_sac_command.sh" \
  "${QUEUE_ROOT}/03_train_sac.log"

SAC_RUN_DIR="$(find "${SAC_SAVE_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
{
  echo "sac_run_dir=${SAC_RUN_DIR}"
  echo "completed_at=$(date --iso-8601=seconds)"
} >> "${SUMMARY_FILE}"

write_state "done" "completed"
echo "[queue] all done; summary=${SUMMARY_FILE}"
