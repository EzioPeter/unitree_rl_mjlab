#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

DATASET_PATH="${DATASET_PATH:-logs/rwm_datasets/go2_rr_calf_strength_0p5_proprioceptive_mixed_1m/dataset.pt}"
POLICY_CONFIG_PATH="${POLICY_CONFIG_PATH:-scripts/reinforcement_learning/rwm_flashsac/configs/go2_flashsac_rwm_proprioceptive.yaml}"
WM_SAVE_BASE="${WM_SAVE_BASE:-logs/rsl_rl/go2_rr_calf_strength_0p5_proprioceptive_unmasked}"
WM_MAX_ITERATIONS="${WM_MAX_ITERATIONS:-5000}"
WM_MODEL_PATH="${WM_MODEL_PATH:-}"
WM_RUN_DIR="${WM_RUN_DIR:-}"
SAC_SAVE_PATH="${SAC_SAVE_PATH:-logs/model_based/go2_rr_calf_strength_0p5_proprioceptive_unmasked/TIMESTAMP}"
SAC_NUM_IMAGINATION_ENVS="${SAC_NUM_IMAGINATION_ENVS:-1024}"
SAC_NUM_ENV_STEPS="${SAC_NUM_ENV_STEPS:-50000000}"
SAC_DEVICE="${SAC_DEVICE:-cuda:0}"

if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

if [[ -z "${WM_MODEL_PATH}" ]]; then
  if [[ -z "${WM_RUN_DIR}" && -d "${WM_SAVE_BASE}" ]]; then
    WM_RUN_DIR="$(find "${WM_SAVE_BASE}" -mindepth 1 -maxdepth 1 -type d -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)"
  fi
  if [[ -z "${WM_RUN_DIR}" ]]; then
    echo "World-model run dir not found. Set WM_MODEL_PATH or WM_RUN_DIR." >&2
    exit 1
  fi
  for candidate in "${WM_RUN_DIR}/model_${WM_MAX_ITERATIONS}.pt" "${WM_RUN_DIR}/latest.pt"; do
    if [[ -f "${candidate}" ]]; then
      WM_MODEL_PATH="${candidate}"
      break
    fi
  done
fi

if [[ ! -f "${WM_MODEL_PATH}" ]]; then
  echo "World-model checkpoint not found: ${WM_MODEL_PATH}" >&2
  exit 1
fi

uv run python scripts/reinforcement_learning/rwm_flashsac/train_flashsac_world_model_go2_proprioceptive.py \
  --config_path "${POLICY_CONFIG_PATH}" \
  --model_resume_path "${WM_MODEL_PATH}" \
  --dataset_path "${DATASET_PATH}" \
  --num_imagination_envs "${SAC_NUM_IMAGINATION_ENVS}" \
  --num_env_steps "${SAC_NUM_ENV_STEPS}" \
  --device "${SAC_DEVICE}" \
  --save_path "${SAC_SAVE_PATH}" \
  --overrides "world_model.policy_action_mask_indices=[]" \
  --overrides "world_model.world_model_action_mask_indices=[]" \
  --overrides "world_model.policy_observation_mask_indices=[]" \
  --overrides "world_model.broken_joint_names=[]"
