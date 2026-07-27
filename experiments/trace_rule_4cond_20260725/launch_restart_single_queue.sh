#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
experiment_root="${repo_root}/experiments/trace_rule_4cond_20260725"
baseline_root="/data1/xjy/unitree_rl_mjlab_v13/logs/experiments/go2_v13_fullstate_rwm_mixed1m_widecmd_policycfg_50m_fresh_seed0_20260725"
output_root="/data1/xjy/unitree_rl_mjlab_v13/logs/experiments/go2_v13_rule_trace_T4_a010_r010_nosatgate_from_scratch_update97458_seed0_20260726_restart1"
log_root="${experiment_root}/logs_restart1"
session_root="${experiment_root}/sessions_restart1"

mkdir -p "${log_root}" "${session_root}"
cd "${repo_root}"

launch_one() {
  local condition="$1"
  local physical_gpu="$2"
  local session_name="trace_rule_restart1_${condition}_gpu${physical_gpu}"
  local output="${output_root}/${condition}/run"
  local log="${log_root}/${condition}.log"
  local pid_file="${session_root}/${condition}.pid"
  local session_file="${session_root}/${condition}.json"

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

queue_after() {
  local predecessor="$1"
  local predecessor_pid="$2"
  local condition="$3"
  local physical_gpu="$4"
  local queue_log="${log_root}/${condition}.queue.log"

  while kill -0 "${predecessor_pid}" 2>/dev/null; do
    if [[ "$(ps -p "${predecessor_pid}" -o stat= 2>/dev/null || true)" == Z* ]]; then
      break
    fi
    sleep 30
  done

  printf '%s predecessor=%s pid=%s exited; launching %s on gpu=%s\n' \
    "$(date --iso-8601=seconds)" "${predecessor}" "${predecessor_pid}" \
    "${condition}" "${physical_gpu}" >>"${queue_log}"
  "${BASH_SOURCE[0]}" --launch-one "${condition}" "${physical_gpu}" >>"${queue_log}" 2>&1
}

case "${1:-}" in
  --launch-one)
    launch_one "$2" "$3"
    exit 0
    ;;
  --queue-after)
    queue_after "$2" "$3" "$4" "$5"
    exit 0
    ;;
  "")
    ;;
  *)
    echo "Unknown argument: $1" >&2
    exit 2
    ;;
esac

launch_one rr03 0
launch_one rr05 1
launch_one p5 2
rr03_pid="$(tr -d '\n' < "${session_root}/rr03.pid")"
nohup setsid "${BASH_SOURCE[0]}" --queue-after rr03 "${rr03_pid}" p75 0 \
  >"${log_root}/p75.queue.supervisor.log" 2>&1 < /dev/null &
queue_pid="$!"
printf '%s\n' "${queue_pid}" >"${session_root}/p75.queue.pid"
printf '{"condition":"p75","physical_gpu":0,"queue_pid":%s,"waits_for":"rr03","waits_for_pid":%s}\n' \
  "${queue_pid}" "${rr03_pid}" >"${session_root}/p75.queue.json"
echo "p75 queued pid=${queue_pid} waits_for=rr03:${rr03_pid} gpu=0"
