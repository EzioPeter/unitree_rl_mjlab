#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

DATASET_PATH="${DATASET_PATH:-logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt}"
WM_CONFIG_PATH="${WM_CONFIG_PATH:-scripts/reinforcement_learning/rwm_dataset/configs/go2_offline_world_model_proprioceptive.yaml}"
WM_SAVE_BASE="${WM_SAVE_BASE:-logs/rsl_rl/go2_rr_calf_strength_0p5_proprioceptive_unmasked}"
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-5000}"
WM_BATCH_SIZE="${WM_BATCH_SIZE:-1024}"
WM_MICRO_BATCH_SIZE="${WM_MICRO_BATCH_SIZE:-256}"
WM_DEVICE="${WM_DEVICE:-cuda:0}"
WM_SAVE_INTERVAL="${WM_SAVE_INTERVAL:-500}"
WM_LOG_INTERVAL="${WM_LOG_INTERVAL:-50}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

uv run python scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2_proprioceptive.py \
  --config_path "${WM_CONFIG_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --save_dir "${WM_SAVE_BASE}" \
  --max_iterations "${WM_MAX_ITERATIONS}" \
  --batch_size "${WM_BATCH_SIZE}" \
  --micro_batch_size "${WM_MICRO_BATCH_SIZE}" \
  --device "${WM_DEVICE}" \
  --save_interval "${WM_SAVE_INTERVAL}" \
  --log_interval "${WM_LOG_INTERVAL}" \
  --overrides "masked_joint_names=[]"
