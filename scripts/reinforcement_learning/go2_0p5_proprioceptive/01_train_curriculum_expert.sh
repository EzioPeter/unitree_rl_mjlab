#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"
DEVICE="${DEVICE:-cuda:0}"
SAVE_PATH="${SAVE_PATH:-logs/model_based/go2_rr_calf_strength_0p5_flashsac_expert_proprioceptive/TIMESTAMP}"

uv run python scripts/train_flashsac.py \
  --config_name flashsac_go2_broken_rr_calf_proprioceptive_expert \
  --no-export_deploy_policy \
  --overrides "num_train_envs=${NUM_TRAIN_ENVS}" \
  --overrides "num_env_steps=${NUM_ENV_STEPS}" \
  --overrides "save_path=${SAVE_PATH}" \
  --overrides "env.device=${DEVICE}" \
  --overrides "env.broken_joint_names=[]" \
  --overrides "env.joint_strength_scales.RR_calf_joint=0.5" \
  --overrides "env.actuator_curriculum.enabled=true"
