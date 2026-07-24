#!/usr/bin/env bash
set -euo pipefail

# V12 50K data-size ablation:
# collect 2048 independent 1000-step columns (2,048,000 transitions) and
# select 50 columns (50,000 transitions) with the same real command envelope.

DEFAULT_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
REPO_ROOT="${REPO_ROOT:-${DEFAULT_REPO_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/runs/v12_e0_five_conditions_2m_to_50k}"
EXPERT="${EXPERT:-}"
DEVICE="${DEVICE:-cuda:0}"
PHYSICAL_GPU="${PHYSICAL_GPU:-}"
SEED="${SEED:-42}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
CONDITIONS="${CONDITIONS:-g0 rr05 p5 p75 rr03}"
COLLECTOR="${REPO_ROOT}/scripts/reinforcement_learning/rwm_dataset/collect_go2_expert_command_coverage_dataset.py"
SELECTOR="${REPO_ROOT}/scripts/reinforcement_learning/rwm_dataset/select_go2_full_trajectories_25k.py"
AUDITOR="${REPO_ROOT}/scripts/reinforcement_learning/rwm_dataset/audit_go2_realrange_25k.py"

readonly NUM_ENVS=2048
readonly NUM_TRANSITIONS=2048000
readonly NUM_SELECTED=50
readonly TRAJECTORY_LENGTH=1000
readonly X_MIN=-0.5
readonly X_MAX=0.5
readonly X_ABS_MIN=0.08
readonly X_ABS_MAX=0.5
readonly Y_ABS_MIN=0.05
readonly Y_ABS_MAX=0.2
readonly YAW_ABS_MIN=0.08
readonly YAW_ABS_MAX=0.4
readonly X_EDGES=(0.08 0.22 0.36 0.50)
readonly Y_EDGES=(0.05 0.10 0.15 0.20)
readonly YAW_EDGES=(0.08 0.186667 0.293333 0.40)

[[ -n "${EXPERT}" ]] || {
  echo "Set EXPERT to the E0 checkpoint directory containing actor.pt." >&2
  exit 2
}
for required in "${PYTHON}" "${COLLECTOR}" "${SELECTOR}" "${AUDITOR}" "${EXPERT}/actor.pt"; do
  [[ -e "${required}" ]] || { echo "Required path does not exist: ${required}" >&2; exit 2; }
done
mkdir -p "${DATA_ROOT}/logs"

collect_condition() {
  local condition="$1"
  local rr_strength="$2"
  local payload_kg="$3"
  local condition_dir="${DATA_ROOT}/${condition}"
  local raw="${condition_dir}/raw_2m/dataset.pt"
  local selected="${condition_dir}/selected_50k/dataset.pt"
  local report="${condition_dir}/selected_50k/selection_report.json"
  local audit="${condition_dir}/selected_50k/audit_report.json"
  local failure_args=()
  [[ "${condition}" == "rr03" ]] && failure_args+=(--allow-mid-trajectory-termination)
  mkdir -p "${condition_dir}/raw_2m" "${condition_dir}/selected_50k"

  if [[ ! -s "${raw}" ]]; then
    "${PYTHON}" "${COLLECTOR}" \
      --task Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --num_envs "${NUM_ENVS}" \
      --num_transitions "${NUM_TRANSITIONS}" \
      --save_path "${raw}" \
      --expert_policy_path "${EXPERT}" \
      --collector_mix expert:1.0 \
      --fixed_collector_assignment \
      --collector_assignment_seed "${SEED}" \
      --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
      --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
      --command_resample_interval_min 120 \
      --command_resample_interval_max 300 \
      --x_range "${X_MIN}" "${X_MAX}" \
      --signed_x \
      --x_abs_range "${X_ABS_MIN}" "${X_ABS_MAX}" \
      --y_abs_range "${Y_ABS_MIN}" "${Y_ABS_MAX}" \
      --yaw_abs_range "${YAW_ABS_MIN}" "${YAW_ABS_MAX}" \
      --dataset_obs_kind proprioceptive \
      --payload_mass_range_kg "${payload_kg}" "${payload_kg}" \
      --rr_calf_strength_range "${rr_strength}" "${rr_strength}" \
      --no-use_domain_randomization \
      --no-use_push_randomization \
      --no-use_observation_noise \
      --env_action_noise_std 0.0 \
      --env_action_bias_std 0.0 \
      --env_action_scale_range 1.0 1.0 \
      --env_action_delay_steps 0 0 \
      --save_trace_snapshots \
      --chunk_size 256000 \
      2>&1 | tee "${DATA_ROOT}/logs/${condition}_collect.log"
  fi

  if [[ ! -s "${selected}" ]]; then
    "${PYTHON}" "${SELECTOR}" \
      --input "${raw}" \
      --output "${selected}" \
      --report "${report}" \
      --condition-id "${condition}" \
      --num-trajectories "${NUM_SELECTED}" \
      --trajectory-length "${TRAJECTORY_LENGTH}" \
      --seed "${SEED}" \
      --x-edges "${X_EDGES[@]}" \
      --y-edges "${Y_EDGES[@]}" \
      --yaw-edges "${YAW_EDGES[@]}" \
      "${failure_args[@]}" \
      2>&1 | tee "${DATA_ROOT}/logs/${condition}_select.log"
  fi

  "${PYTHON}" "${AUDITOR}" \
    --dataset "${selected}" \
    --selection-report "${report}" \
    --output "${audit}" \
    --condition-id "${condition}" \
    --num-trajectories "${NUM_SELECTED}" \
    --trajectory-length "${TRAJECTORY_LENGTH}" \
    "${failure_args[@]}" \
    2>&1 | tee "${DATA_ROOT}/logs/${condition}_audit.log"

  # Raw part files duplicate raw_2m/dataset.pt and are safe to remove only
  # after the selected 50K dataset passes the fail-closed audit.
  [[ -s "${audit}" ]] && rm -rf "${condition_dir}/raw_2m/parts"
}

echo "physical_gpu=${PHYSICAL_GPU} visible_device=${DEVICE} conditions=${CONDITIONS}"
for condition in ${CONDITIONS}; do
  case "${condition}" in
    g0) collect_condition g0 1.0 0.0 ;;
    rr05) collect_condition rr05 0.5 0.0 ;;
    rr03) collect_condition rr03 0.3 0.0 ;;
    p5) collect_condition p5 1.0 5.0 ;;
    p75) collect_condition p75 1.0 7.5 ;;
    *) echo "Unknown condition: ${condition}" >&2; exit 2 ;;
  esac
done

echo "All requested 2M->50K datasets passed audit under ${DATA_ROOT}"
