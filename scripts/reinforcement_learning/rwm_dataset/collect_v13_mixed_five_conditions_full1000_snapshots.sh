#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv/bin/python}"
EXPERT_POLICY_PATH="${EXPERT_POLICY_PATH:-/data1/xjy/unitree_rl_mjlab_normal_fixstand_v2/logs/experiments/go2_normal_dr_3x3x4/normal_go2_fixstand_v2_moderate_seed0_20260712/experts/E0_calibrated_default/run/step97656}"
DATA_ROOT="${DATA_ROOT:-${REPO_ROOT}/logs/rwm_datasets_v13_mixed_full1000_snapshots_5conditions_realrange_20260724_v2}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-0}"
SELECTION_SEED="${SELECTION_SEED:-0}"
V13_PREFLIGHT_ONLY="${V13_PREFLIGHT_ONLY:-0}"
MEDIUM_POLICY_PATH="${MEDIUM_POLICY_PATH:-}"

COLLECTOR="${SCRIPT_DIR}/collect_go2_expert_command_coverage_dataset.py"
SELECTOR="${SCRIPT_DIR}/slice_go2_dataset_by_env_trajectories.py"
AUDITOR="${SCRIPT_DIR}/audit_v13_mixed_25k.py"
MANIFEST_WRITER="${SCRIPT_DIR}/prepare_v13_mixed_collection_manifest.py"

TASK_ID="Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert"
NUM_ENVS=1024
TIME_STEPS=1000
NUM_TRANSITIONS=$((NUM_ENVS * TIME_STEPS))
SELECTED_TRAJECTORIES=25

MIX_EXPERT="0.45"
MIX_NOISY_EXPERT="0.25"
MIX_MEDIUM="0.10"
MIX_FAILURE_BORDER="0.15"
MIX_RANDOM="0.05"
COLLECTOR_MIX="expert:${MIX_EXPERT},noisy_expert:${MIX_NOISY_EXPERT},medium:${MIX_MEDIUM},failure_border:${MIX_FAILURE_BORDER},random:${MIX_RANDOM}"
NOISY_ACTION_STD="0.12"
MEDIUM_ACTION_STD="0.25"
FAILURE_ACTION_STD="0.45"

COMMAND_X_MIN="0.05"
COMMAND_X_MAX="0.50"
COMMAND_Y_MIN="0.03"
COMMAND_Y_MAX="0.20"
COMMAND_YAW_MIN="0.05"
COMMAND_YAW_MAX="0.40"
COMMAND_RESAMPLE_MIN="120"
COMMAND_RESAMPLE_MAX="300"

CONDITIONS=(g0 rr05 rr03 p5 p75)
CURRENT_CONDITION=""
CURRENT_CONDITION_ROOT=""

fail() {
  printf 'ERROR: %s\n' "$*" >&2
  exit 1
}

mark_invalid_on_error() {
  failed_command="$1"
  status="$2"
  trap - ERR
  if [[ -n "${CURRENT_CONDITION_ROOT}" ]]; then
    mkdir -p "${CURRENT_CONDITION_ROOT}"
    printf '%s\n' \
      "V13 condition ${CURRENT_CONDITION} failed." \
      "exit_status=${status}" \
      "failed_command=${failed_command}" \
      "This condition is invalid and must not be used for training." \
      > "${CURRENT_CONDITION_ROOT}/INVALID_DO_NOT_USE.txt"
  fi
  exit "${status}"
}
trap 'mark_invalid_on_error "${BASH_COMMAND}" "$?"' ERR

[[ "${REPO_ROOT}" != *"unitree_rl_mjlab_v12"* ]] || fail "拒绝从 V12 仓库运行 V13 正式采集：${REPO_ROOT}"
[[ -x "${PYTHON}" ]] || fail "Python 不可执行：${PYTHON}"
[[ -f "${COLLECTOR}" ]] || fail "缺少采集脚本：${COLLECTOR}"
[[ -f "${SELECTOR}" ]] || fail "缺少筛选脚本：${SELECTOR}"
[[ -f "${AUDITOR}" ]] || fail "缺少独立审计脚本：${AUDITOR}"
[[ -f "${MANIFEST_WRITER}" ]] || fail "缺少 launch manifest 脚本：${MANIFEST_WRITER}"
[[ -e "${EXPERT_POLICY_PATH}" ]] || fail "E0 expert 不存在：${EXPERT_POLICY_PATH}"

[[ -z "${MEDIUM_POLICY_PATH}" ]] || fail "V13 当前未冻结独立 medium checkpoint；MEDIUM_POLICY_PATH 必须为空"

printf '%s\n' \
  "V13 mixed five-condition preflight" \
  "repo=${REPO_ROOT}" \
  "python=${PYTHON}" \
  "expert=${EXPERT_POLICY_PATH}" \
  "medium_policy=${MEDIUM_POLICY_PATH:-<empty: expert+0.25 noise>}" \
  "data_root=${DATA_ROOT}" \
  "device=${DEVICE}" \
  "conditions=${CONDITIONS[*]}" \
  "raw_per_condition=${NUM_ENVS}x${TIME_STEPS}=${NUM_TRANSITIONS}" \
  "selected_per_condition=${SELECTED_TRAJECTORIES}x${TIME_STEPS}" \
  "mix=${COLLECTOR_MIX}" \
  "action_noise_std=noisy:${NOISY_ACTION_STD},medium:${MEDIUM_ACTION_STD},failure:${FAILURE_ACTION_STD}" \
  "snapshots=enabled" \
  "DR/push/observation_noise/action_interface_noise_bias_delay=disabled"

if [[ "${V13_PREFLIGHT_ONLY}" == "1" ]]; then
  printf 'Preflight completed; no dataset was written.\n'
  exit 0
fi

for condition_id in "${CONDITIONS[@]}"; do
  CURRENT_CONDITION="${condition_id}"
  rr_strength="1.0"
  payload_mass="0.0"
  case "${condition_id}" in
    g0)
      ;;
    rr05)
      rr_strength="0.5"
      ;;
    rr03)
      rr_strength="0.3"
      ;;
    p5)
      payload_mass="5.0"
      ;;
    p75)
      payload_mass="7.5"
      ;;
    *)
      fail "未知工况：${condition_id}"
      ;;
  esac

  condition_root="${DATA_ROOT}/${condition_id}"
  CURRENT_CONDITION_ROOT="${condition_root}"
  raw_dir="${condition_root}/raw_1024x1000_mixed"
  selected_dir="${condition_root}/selected_25k"
  raw_dataset="${raw_dir}/dataset.pt"
  selected_dataset="${selected_dir}/dataset.pt"
  selection_report="${selected_dir}/selection_report.json"
  audit_report="${selected_dir}/fullstate_audit.json"
  launch_manifest="${raw_dir}/launch_manifest.json"
  collection_log="${raw_dir}/collection.log"

  if [[ -e "${selected_dataset}" || -e "${selection_report}" || -e "${audit_report}" ]]; then
    fail "${condition_id} 的筛选/审计产物已存在；为防止混写，请换一个 DATA_ROOT：${selected_dir}"
  fi

  if [[ ! -f "${raw_dataset}" ]]; then
    if [[ -d "${raw_dir}" ]] && find "${raw_dir}" -mindepth 1 -print -quit | grep -q .; then
      fail "${condition_id} 的 raw 输出目录非空；为防止混写，请换一个 DATA_ROOT：${raw_dir}"
    fi

    "${PYTHON}" "${MANIFEST_WRITER}" \
      --output "${launch_manifest}" \
      --repo_root "${REPO_ROOT}" \
      --expert_policy_path "${EXPERT_POLICY_PATH}" \
      --condition_id "${condition_id}" \
      --rr_calf_strength "${rr_strength}" \
      --payload_mass_kg "${payload_mass}" \
      --device "${DEVICE}" \
      --raw_dataset "${raw_dataset}" \
      --code_path "${COLLECTOR}" \
      --code_path "${SELECTOR}" \
      --code_path "${AUDITOR}" \
      --code_path "${MANIFEST_WRITER}" \
      --code_path "${SCRIPT_DIR}/$(basename -- "${BASH_SOURCE[0]}")"
    printf '\n[%s] collecting raw mixed dataset\n' "${condition_id}"
    # The frozen expert collector consumes its legacy 45D view. The selected
    # dataset still stores full 45D dynamics state and reconstructs the
    # downstream Actor/Critic 48D observations during audit/training.
    "${PYTHON}" "${COLLECTOR}" \
      --task "${TASK_ID}" \
      --device "${DEVICE}" \
      --seed "${SEED}" \
      --num_envs "${NUM_ENVS}" \
      --num_transitions "${NUM_TRANSITIONS}" \
      --save_path "${raw_dataset}" \
      --dataset_obs_kind proprioceptive \
      --expert_policy_path "${EXPERT_POLICY_PATH}" \
      --collector_mix "${COLLECTOR_MIX}" \
      --action_noise_std "${NOISY_ACTION_STD}" \
      --medium_action_noise_std "${MEDIUM_ACTION_STD}" \
      --failure_action_noise_std "${FAILURE_ACTION_STD}" \
      --command_modes stand,pure_x,pure_y,pure_yaw,xy,x_yaw,y_yaw,xy_yaw \
      --command_mode_weights stand:0.08,pure_x:0.25,pure_y:0.10,pure_yaw:0.08,xy:0.14,x_yaw:0.17,y_yaw:0.05,xy_yaw:0.13 \
      --command_resample_interval_min "${COMMAND_RESAMPLE_MIN}" \
      --command_resample_interval_max "${COMMAND_RESAMPLE_MAX}" \
      --x_range "-${COMMAND_X_MAX}" "${COMMAND_X_MAX}" \
      --signed_x \
      --x_abs_range "${COMMAND_X_MIN}" "${COMMAND_X_MAX}" \
      --y_abs_range "${COMMAND_Y_MIN}" "${COMMAND_Y_MAX}" \
      --yaw_abs_range "${COMMAND_YAW_MIN}" "${COMMAND_YAW_MAX}" \
      --rr_calf_strength_range "${rr_strength}" "${rr_strength}" \
      --payload_mass_range_kg "${payload_mass}" "${payload_mass}" \
      --env_action_noise_std 0.0 \
      --env_action_bias_std 0.0 \
      --env_action_scale_range 1.0 1.0 \
      --env_action_delay_steps 0 0 \
      --no-use_domain_randomization \
      --no-use_push_randomization \
      --no-use_observation_noise \
      --save_trace_snapshots \
      --chunk_size 256000 \
      --headless \
      2>&1 | tee "${collection_log}"
  else
    [[ -f "${launch_manifest}" ]] || fail "${condition_id} 原始数据缺少 launch_manifest.json"
    printf '\n[%s] reusing completed raw dataset: %s\n' "${condition_id}" "${raw_dataset}"
  fi

  mkdir -p "${selected_dir}"
  printf '[%s] selecting 25 complete snapshot trajectories\n' "${condition_id}"
  "${PYTHON}" "${SELECTOR}" \
    --source_path "${raw_dataset}" \
    --save_path "${selected_dataset}" \
    --report_path "${selection_report}" \
    --condition_id "${condition_id}" \
    --expected_expert_policy_path "${EXPERT_POLICY_PATH}" \
    --launch_manifest_path "${launch_manifest}" \
    --num_trajectories "${SELECTED_TRAJECTORIES}" \
    --expected_time_steps "${TIME_STEPS}" \
    --seed "${SELECTION_SEED}" \
    --selection stratified \
    --require_trace_snapshots

  printf '[%s] running independent audit\n' "${condition_id}"
  "${PYTHON}" "${AUDITOR}" \
    --dataset "${selected_dataset}" \
    --selection_report "${selection_report}" \
    --output "${audit_report}" \
    --condition_id "${condition_id}"
done

CURRENT_CONDITION=""
CURRENT_CONDITION_ROOT=""
printf '\nAll five V13 conditions passed selection and independent audit.\n'
