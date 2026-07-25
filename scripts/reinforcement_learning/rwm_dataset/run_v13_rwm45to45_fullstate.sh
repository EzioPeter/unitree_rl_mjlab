#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 || $# -gt 4 ]]; then
  echo "usage: $0 <g0|rr05|rr03|p5|p75> <physical_gpu_index> <absolute_v13_sim_dataset_manifest.json> [resume_checkpoint.pt]" >&2
  exit 2
fi

condition="$1"
gpu="$2"
manifest_path="$3"
resume_path="${4:-}"
case "$condition" in
  g0|rr05|rr03|p5|p75) ;;
  *) echo "unsupported condition: $condition" >&2; exit 2 ;;
esac

if [[ "$manifest_path" != /* || ! -f "$manifest_path" ]]; then
  echo "manifest must be an existing absolute path: $manifest_path" >&2
  exit 2
fi
if [[ -n "$resume_path" && ( "$resume_path" != /* || ! -f "$resume_path" ) ]]; then
  echo "resume checkpoint must be an existing absolute path: $resume_path" >&2
  exit 2
fi

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"

config_path="scripts/reinforcement_learning/rwm_dataset/configs/go2_offline_world_model_v13_full45.yaml"
experiment_name="${V13_RWM_EXPERIMENT_NAME:-go2_rwm_v13_full45_25k}"
save_dir="logs/experiments/${experiment_name}/$condition/rwm"
launcher_log="$save_dir/launcher.log"

preflight_json="$(
  CONDITION="$condition" MANIFEST_PATH="$manifest_path" CONFIG_PATH="$config_path" \
    RESUME_PATH="$resume_path" \
    CUDA_VISIBLE_DEVICES="$gpu" \
    .venv/bin/python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


condition = os.environ["CONDITION"]
manifest_path = Path(os.environ["MANIFEST_PATH"]).resolve()
config_path = Path(os.environ["CONFIG_PATH"]).resolve()
repo_root = Path.cwd().resolve()
trainer_path = (
    repo_root
    / "scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2_v13_fullstate.py"
)
base_trainer_path = (
    repo_root
    / "scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2.py"
)
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
cfg = OmegaConf.load(config_path)

assert torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(torch.device("cuda:0"))
assert "3090" in gpu_name, gpu_name
assert manifest["schema"] == "go2_v13_frozen_sim_mixed_fullstate_dataset_manifest_v3"
assert manifest["status"] == "active_sim_dataset_input"
assert manifest["rwm_interface"]["input_state_dim"] == 45
assert manifest["rwm_interface"]["output_state_dim"] == 45
assert manifest["rwm_interface"]["input_dropped_state_indices"] == []
assert manifest["rwm_interface"]["state_loss_ignored_indices"] == []
assert manifest["rwm_interface"]["base_lin_vel_input_present"] is True
assert manifest["rwm_interface"]["base_lin_vel_target_supervised"] is True
assert manifest["policy_interface"]["actor_observation_dim"] == 48
assert manifest["policy_interface"]["critic_observation_dim"] == 48
assert int(cfg.v13_interface.rwm_input_state_dim) == 45
assert int(cfg.v13_interface.rwm_output_state_dim) == 45
assert bool(cfg.v13_interface.base_lin_vel_input_present) is True
assert bool(cfg.v13_interface.base_lin_vel_target_supervised) is True

entry = manifest["conditions"][condition]
dataset_path = Path(entry["selected_dataset_path"]).resolve()
assert dataset_path.is_file(), dataset_path
actual_dataset_sha256 = sha256(dataset_path)
assert actual_dataset_sha256 == entry["selected_dataset_sha256"]

dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
assert int(dataset["state_dim"]) == 45
assert int(dataset["action_dim"]) == 12
assert int(dataset["metadata"]["num_transitions"]) == 25_000
assert int(dataset["metadata"]["num_time_steps"]) == 1_000
states = torch.stack(dataset["states"])
next_states = torch.stack(dataset["next_states"])
assert tuple(states.shape) == (1_000, 25, 45)
assert tuple(next_states.shape) == (1_000, 25, 45)
assert torch.isfinite(states).all()
assert torch.isfinite(next_states).all()
assert torch.isfinite(states[..., :3]).all()
assert torch.isfinite(next_states[..., :3]).all()

resume_path_value = os.environ.get("RESUME_PATH", "")
resume_record = None
if resume_path_value:
    resume_path = Path(resume_path_value).resolve()
    checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
    iteration = int(checkpoint.get("iter", -1))
    assert 1 <= iteration < int(cfg.max_iterations), iteration
    infos = checkpoint.get("infos") or {}
    assert Path(infos["dataset_path"]).resolve() == dataset_path
    assert int(infos["state_dim"]) == 45
    assert int(infos["action_dim"]) == 12
    assert checkpoint.get("system_dynamics_optimizer_state_dict")
    normalizer = checkpoint["normalizer"]
    assert tuple(normalizer["state_mean"].shape) == (45,)
    assert tuple(normalizer["obs_mean"].shape) == (48,)
    interface = (checkpoint.get("config") or {}).get("v13_interface") or {}
    assert int(interface["rwm_input_state_dim"]) == 45
    assert int(interface["rwm_output_state_dim"]) == 45
    assert int(interface["actor_observation_dim"]) == 48
    assert int(interface["critic_observation_dim"]) == 48
    rng_state = checkpoint.get("rng_state")
    assert isinstance(rng_state, dict)
    assert set(rng_state) == {"python", "numpy", "torch_cpu", "torch_cuda"}
    resume_record = {
        "path": str(resume_path),
        "sha256": sha256(resume_path),
        "iteration": iteration,
        "optimizer_present": True,
        "rng_state_present": True,
    }

print(json.dumps({
    "condition": condition,
    "dataset_path": str(dataset_path),
    "dataset_sha256": actual_dataset_sha256,
    "manifest_path": str(manifest_path),
    "manifest_sha256": sha256(manifest_path),
    "config_sha256": sha256(config_path),
    "trainer_sha256": sha256(trainer_path),
    "base_trainer_sha256": sha256(base_trainer_path),
    "gpu_name": gpu_name,
    "resume": resume_record,
}, sort_keys=True))
PY
)"

dataset_path="$(
  PREFLIGHT_JSON="$preflight_json" .venv/bin/python - <<'PY'
import json
import os
print(json.loads(os.environ["PREFLIGHT_JSON"])["dataset_path"])
PY
)"

if [[ "${V13_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  echo "V13_RWM_FULL45_PREFLIGHT_PASS $preflight_json"
  exit 0
fi

if [[ -z "$resume_path" && -e "$save_dir" && -n "$(find "$save_dir" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "refusing to overwrite non-empty output directory: $save_dir" >&2
  exit 2
fi
if [[ -n "$resume_path" ]] && find "$save_dir" -mindepth 2 -maxdepth 2 \
  -type f -name model_5000.pt -print -quit | grep -q .; then
  echo "refusing to resume because model_5000.pt already exists in: $save_dir" >&2
  exit 2
fi
mkdir -p "$save_dir"

{
  echo "version=V13"
  echo "condition=$condition"
  echo "physical_gpu=$gpu"
  echo "dataset=$dataset_path"
  echo "manifest=$manifest_path"
  echo "preflight=$preflight_json"
  echo "config=$config_path"
  echo "save_dir=$save_dir"
  echo "semantics=full-state-45-input,45-output,BLV-input-and-target-supervised"
  echo "resume_path=${resume_path:-none}"
  echo "started_at=$(date --iso-8601=seconds)"
} | tee -a "$launcher_log"

resume_args=()
if [[ -n "$resume_path" ]]; then
  resume_args=(--resume_path "$resume_path")
fi

CUDA_VISIBLE_DEVICES="$gpu" .venv/bin/python \
  scripts/reinforcement_learning/rwm_dataset/train_world_model_offline_go2_v13_fullstate.py \
  --config_path "$config_path" \
  --dataset_path "$dataset_path" \
  --save_dir "$save_dir" \
  --device cuda:0 \
  "${resume_args[@]}" \
  2>&1 | tee -a "$launcher_log"
