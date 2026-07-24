#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

SEED="${SEED:-0}"
NUM_TRAIN_ENVS="${NUM_TRAIN_ENVS:-1024}"
NUM_ENV_STEPS="${NUM_ENV_STEPS:-100000000}"
DEVICE="${DEVICE:-cuda:0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/runs/go2_normal_flashsac_expert}"

uv run python scripts/train_flashsac.py \
  --config_name flashsac_go2_normal_proprioceptive_expert \
  --no-export_deploy_policy \
  --overrides "seed=${SEED}" \
  --overrides "num_train_envs=${NUM_TRAIN_ENVS}" \
  --overrides "num_env_steps=${NUM_ENV_STEPS}" \
  --overrides "save_path=${OUTPUT_DIR}" \
  --overrides "env.device=${DEVICE}"
