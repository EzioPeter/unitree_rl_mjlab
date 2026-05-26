from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf

from flash_rl.agents.flashSAC.network import FlashSACActor


@dataclass(frozen=True)
class OnnxValueInfo:
    name: str
    shape: tuple[int | str, ...]
    elem_type: int


@dataclass(frozen=True)
class OnnxSignature:
    inputs: tuple[OnnxValueInfo, ...]
    outputs: tuple[OnnxValueInfo, ...]


class FlashSACDeploymentActor(torch.nn.Module):
    """Deterministic FlashSAC actor used by the C++ ONNX Runtime deploy path."""

    def __init__(self, actor: FlashSACActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        mean, _std = self.actor.get_mean_and_std(obs, training=False)
        return torch.tanh(mean)


def read_onnx_signature(path: str | Path) -> OnnxSignature:
    import onnx

    model = onnx.load(str(path))

    def _value_info(value: Any) -> OnnxValueInfo:
        dims: list[int | str] = []
        for dim in value.type.tensor_type.shape.dim:
            if dim.dim_value:
                dims.append(int(dim.dim_value))
            else:
                dims.append(dim.dim_param or "?")
        return OnnxValueInfo(
            name=value.name,
            shape=tuple(dims),
            elem_type=int(value.type.tensor_type.elem_type),
        )

    return OnnxSignature(
        inputs=tuple(_value_info(value) for value in model.graph.input),
        outputs=tuple(_value_info(value) for value in model.graph.output),
    )


def assert_matching_onnx_signature(candidate: str | Path, reference: str | Path) -> None:
    candidate_sig = read_onnx_signature(candidate)
    reference_sig = read_onnx_signature(reference)
    if candidate_sig != reference_sig:
        raise ValueError(
            "Exported FlashSAC ONNX signature does not match the G1 deployment reference:\n"
            f"candidate={candidate_sig}\n"
            f"reference={reference_sig}"
        )


def _agent_cfg_value(agent_cfg: Any, key: str) -> Any:
    if OmegaConf.is_config(agent_cfg):
        return OmegaConf.select(agent_cfg, key)
    if isinstance(agent_cfg, dict):
        return agent_cfg[key]
    return getattr(agent_cfg, key)


def _load_actor_state_dict(actor: FlashSACActor, checkpoint_path: Path) -> None:
    actor_ckpt = torch.load(checkpoint_path / "actor.pt", map_location="cpu")
    state_dict = actor_ckpt["network_state_dict"]
    target_keys = set(actor.state_dict().keys())
    if not set(state_dict.keys()).issubset(target_keys):
        stripped = {}
        for key, value in state_dict.items():
            if key.startswith("_orig_mod."):
                stripped[key.removeprefix("_orig_mod.")] = value
            else:
                stripped[key] = value
        state_dict = stripped
    actor.load_state_dict(state_dict)


def export_flashsac_policy_to_onnx(
    checkpoint_path: str | Path,
    agent_cfg: Any,
    reference_onnx_path: str | Path,
    output_onnx_path: str | Path,
    deploy_yaml_path: str | Path | None = None,
    output_deploy_yaml_path: str | Path | None = None,
) -> None:
    """Export a FlashSAC checkpoint as a G1 deploy-compatible ONNX policy.

    The input/output names and dimensions are copied from ``reference_onnx_path``.
    For the current G1 velocity deploy policy that is ``obs [1, 98]`` to
    ``actions [1, 29]``.
    """

    checkpoint = Path(checkpoint_path)
    output_onnx = Path(output_onnx_path)
    reference_onnx = Path(reference_onnx_path)
    if not (checkpoint / "actor.pt").exists():
        raise FileNotFoundError(f"FlashSAC actor checkpoint not found: {checkpoint / 'actor.pt'}")
    if not reference_onnx.exists():
        raise FileNotFoundError(f"Reference ONNX policy not found: {reference_onnx}")

    reference_sig = read_onnx_signature(reference_onnx)
    if len(reference_sig.inputs) != 1 or len(reference_sig.outputs) != 1:
        raise ValueError(f"Expected one input and one output in reference ONNX, got: {reference_sig}")

    input_info = reference_sig.inputs[0]
    output_info = reference_sig.outputs[0]
    if len(input_info.shape) != 2 or len(output_info.shape) != 2:
        raise ValueError(f"Expected rank-2 input/output in reference ONNX, got: {reference_sig}")
    if not isinstance(input_info.shape[1], int) or not isinstance(output_info.shape[1], int):
        raise ValueError(f"Reference ONNX must have static feature dimensions, got: {reference_sig}")

    obs_dim = input_info.shape[1]
    action_dim = output_info.shape[1]

    actor = FlashSACActor(
        num_blocks=int(_agent_cfg_value(agent_cfg, "actor_num_blocks")),
        input_dim=obs_dim,
        hidden_dim=int(_agent_cfg_value(agent_cfg, "actor_hidden_dim")),
        action_dim=action_dim,
    )
    _load_actor_state_dict(actor, checkpoint)
    actor.eval()

    output_onnx.parent.mkdir(parents=True, exist_ok=True)
    wrapper = FlashSACDeploymentActor(actor).eval()
    dummy_obs = torch.zeros((1, obs_dim), dtype=torch.float32)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_obs,
            str(output_onnx),
            input_names=[input_info.name],
            output_names=[output_info.name],
            opset_version=18,
            external_data=False,
        )

    assert_matching_onnx_signature(output_onnx, reference_onnx)

    if deploy_yaml_path is not None:
        source_yaml = Path(deploy_yaml_path)
        if not source_yaml.exists():
            raise FileNotFoundError(f"Deploy YAML not found: {source_yaml}")
        if output_deploy_yaml_path is None:
            output_deploy_yaml_path = output_onnx.parent.parent / "params" / "deploy.yaml"
        target_yaml = Path(output_deploy_yaml_path)
        target_yaml.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_yaml, target_yaml)
