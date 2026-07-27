#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment_root="${repo_root}/experiments/trace_rule_4cond_20260725"
baseline_root="/data1/xjy/unitree_rl_mjlab_v13/logs/experiments/go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m_fresh_seed0_20260725"
output_root="/data1/xjy/unitree_rl_mjlab_v13/logs/experiments/go2_v13_rule_trace_T4_a010_r010_nosatgate_from_scratch_update97458_seed0_20260725"

mkdir -p "${experiment_root}/logs" "${experiment_root}/sessions"
cd "${repo_root}"

launch_one() {
  local condition="$1"
  local physical_gpu="$2"
  local session_name="trace_rule_${condition}_gpu${physical_gpu}"
  local output="${output_root}/${condition}/run"
  local log="${experiment_root}/logs/${condition}.log"
  local pid_file="${experiment_root}/sessions/${condition}.pid"
  local session_file="${experiment_root}/sessions/${condition}.json"

  if [[ -e "${output}" ]]; then
    echo "Refusing to overwrite existing output: ${output}" >&2
    return 1
  fi
  if [[ -e "${log}" || -e "${pid_file}" || -e "${session_file}" ]]; then
    echo "Refusing to overwrite existing session artifacts for ${condition}" >&2
    return 1
  fi

  nohup setsid env \
    -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
    "CUDA_VISIBLE_DEVICES=${physical_gpu}" \
    "TRACE_SESSION_NAME=${session_name}" \
    PYTHONUNBUFFERED=1 \
    .venv/bin/python \
    scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_v13_fullstate.py \
    --config_path "${baseline_root}/${condition}/run/step48828/rwm_flashsac_config.yaml" \
    --trace_config_path "${experiment_root}/configs/${condition}.yaml" \
    --no-load_replay_buffer \
    --device cuda \
    --expected_policy_updates 97458 \
    --save_path "${output}" \
    >"${log}" 2>&1 < /dev/null &
  local pid="$!"
  printf '%s\n' "${pid}" > "${pid_file}"
  printf '{"condition":"%s","physical_gpu":%s,"pid":%s,"session_name":"%s","log":"%s","output":"%s"}\n' \
    "${condition}" "${physical_gpu}" "${pid}" "${session_name}" "${log}" "${output}" \
    > "${session_file}"
  echo "${condition} pid=${pid} gpu=${physical_gpu} session=${session_name}"
}

launch_one rr03 1
launch_one rr05 1
launch_one p5 2
launch_one p75 2
