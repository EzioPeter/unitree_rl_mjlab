"""Export a normal-strength 45D Go2 FlashSAC-RWM actor for deployment."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.export import export_flashsac_actor_to_onnx, read_onnx_signature


GO2_ACTOR_OBS_DIM = 45
GO2_ACTION_DIM = 12
DEFAULT_DEPLOY_TEMPLATE = (
    REPO_ROOT
    / "deployment/policies/g0_d0_rrcalf_0p5/params/deploy.yaml"
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--output_policy_dir", required=True)
    parser.add_argument("--deploy_yaml", default=str(DEFAULT_DEPLOY_TEMPLATE))
    parser.add_argument("--policy_name", default="go2_rwm_proprioceptive_normal")
    parser.add_argument("--onnx_ir_version", type=int, default=9)
    parser.add_argument("--parity_samples", type=int, default=32)
    parser.add_argument("--parity_atol", type=float, default=1.0e-5)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    checkpoint_path = Path(args.checkpoint_path).expanduser().resolve()
    config_path = (
        Path(args.config_path).expanduser().resolve()
        if args.config_path
        else checkpoint_path / "flashsac_config.yaml"
    )
    output_policy_dir = Path(args.output_policy_dir).expanduser().resolve()
    deploy_template = Path(args.deploy_yaml).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"FlashSAC config not found: {config_path}")
    if not deploy_template.is_file():
        raise FileNotFoundError(f"Normal Go2 deploy template not found: {deploy_template}")

    cfg = OmegaConf.load(config_path)
    OmegaConf.resolve(cfg)
    world_model = cfg.get("world_model")
    if world_model is not None:
        if list(world_model.policy_action_mask_indices) or list(world_model.world_model_action_mask_indices):
            raise ValueError("Normal Go2 export refuses masked action policies")
        if list(world_model.broken_joint_names):
            raise ValueError("Normal Go2 export refuses broken-joint policies")

    output_onnx = output_policy_dir / "exported" / "policy.onnx"
    output_yaml = output_policy_dir / "params" / "deploy.yaml"
    parity = export_flashsac_actor_to_onnx(
        checkpoint_path=checkpoint_path,
        agent_cfg=cfg.agent,
        output_onnx_path=output_onnx,
        obs_dim=GO2_ACTOR_OBS_DIM,
        action_dim=GO2_ACTION_DIM,
        onnx_ir_version=args.onnx_ir_version,
        parity_samples=args.parity_samples,
        parity_atol=args.parity_atol,
    )
    signature = read_onnx_signature(output_onnx)
    if int(args.onnx_ir_version) != 9:
        raise ValueError("Normal deployment contract requires ONNX IR version 9")
    output_yaml.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(deploy_template, output_yaml)

    metadata = {
        "policy_name": args.policy_name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "actor_observation_dim": GO2_ACTOR_OBS_DIM,
        "action_dim": GO2_ACTION_DIM,
        "onnx_ir_version": 9,
        "onnx_opset": 18,
        "normal_joint_strength": True,
        "rr_calf_strength_scale": 1.0,
        "observation_order": [
            "base_ang_vel[3]",
            "projected_gravity[3]",
            "velocity_commands[3]",
            "joint_pos_rel[12]",
            "joint_vel_rel[12]",
            "last_action[12]",
        ],
        "policy_joint_order": ["FL", "FR", "RL", "RR"],
        "hardware_joint_order": ["FR", "FL", "RR", "RL"],
        "joint_ids_map": [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8],
        "onnx_signature": {
            "inputs": [asdict(value) for value in signature.inputs],
            "outputs": [asdict(value) for value in signature.outputs],
        },
        "parity": asdict(parity),
    }
    output_policy_dir.mkdir(parents=True, exist_ok=True)
    (output_policy_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"policy_dir={output_policy_dir}")
    print(f"onnx={output_onnx}")
    print(f"deploy_yaml={output_yaml}")
    print(f"signature={signature}")
    print(f"parity={parity}")


if __name__ == "__main__":
    main()
