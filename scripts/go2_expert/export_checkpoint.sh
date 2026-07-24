#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

CHECKPOINT="${1:-${REPO_ROOT}/expert_deploy/artifacts/g0_d0_step97656}"
OUTPUT_POLICY_DIR="${2:-${REPO_ROOT}/expert_deploy/artifacts/g0_d0_step97656}"

uv run python scripts/export_flashsac_go2_normal_proprioceptive.py \
  --checkpoint_path "${CHECKPOINT}" \
  --config_path "${CHECKPOINT}/flashsac_config.yaml" \
  --output_policy_dir "${OUTPUT_POLICY_DIR}" \
  --deploy_yaml "${REPO_ROOT}/expert_deploy/deployment/policies/g0_d0_rrcalf_0p5/params/deploy.yaml" \
  --policy_name g0_d0_step97656
