"""Export a FlashSAC checkpoint to the G1 deployment ONNX format."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import hydra
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from flash_rl.export import assert_matching_onnx_signature, export_flashsac_policy_to_onnx, read_onnx_signature


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
G1_REFERENCE_ONNX = REPO_ROOT / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
G1_DEPLOY_YAML = REPO_ROOT / "deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml"
G1_FLASHSAC_POLICY_DIR = REPO_ROOT / "deploy/robots/g1/config/policy/velocity/v1_flashsac"


def _compose_config(config_path: str, config_name: str, overrides: list[str]):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    GlobalHydra.instance().clear()
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = (REPO_ROOT / config_dir).resolve()
    hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config_name", type=str, default="flashsac_g1_velocity")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--output_policy_dir", type=str, default=str(G1_FLASHSAC_POLICY_DIR))
    parser.add_argument("--reference_onnx", type=str, default=str(G1_REFERENCE_ONNX))
    parser.add_argument("--deploy_yaml", type=str, default=str(G1_DEPLOY_YAML))
    args = parser.parse_args()

    cfg = _compose_config(args.config_path, args.config_name, args.overrides)
    output_policy_dir = Path(args.output_policy_dir).expanduser()
    if not output_policy_dir.is_absolute():
        output_policy_dir = (REPO_ROOT / output_policy_dir).resolve()
    output_onnx = output_policy_dir / "exported" / "policy.onnx"
    output_yaml = output_policy_dir / "params" / "deploy.yaml"

    export_flashsac_policy_to_onnx(
        checkpoint_path=args.checkpoint_path,
        agent_cfg=cfg.agent,
        reference_onnx_path=args.reference_onnx,
        output_onnx_path=output_onnx,
        deploy_yaml_path=args.deploy_yaml,
        output_deploy_yaml_path=output_yaml,
    )
    assert_matching_onnx_signature(output_onnx, args.reference_onnx)
    print(f"[FlashSAC] Exported: {output_onnx}")
    print(f"[FlashSAC] Deploy YAML: {output_yaml}")
    print(f"[FlashSAC] Signature: {read_onnx_signature(output_onnx)}")


if __name__ == "__main__":
    main()
