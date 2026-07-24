"""Train a history-conditioned CENet + deterministic BC policy for Go2."""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expert_deploy.v12_bc.cenet_bc import (  # noqa: E402
    CENetBCDeploymentPolicy,
    CENetBCModelConfig,
    CENetBCPolicy,
    normalize_unit_parameters,
)


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: str
    output_dir: str
    seed: int
    validation_fraction: float
    epochs: int
    refit_epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    velocity_loss_weight: float
    history_length: int
    cenet_hidden_dim: int
    latent_dim: int
    actor_hidden_dim: int
    actor_num_blocks: int
    device: str
    num_workers: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--refit_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--velocity_loss_weight", type=float, default=1.0)
    parser.add_argument("--history_length", type=int, default=10)
    parser.add_argument("--cenet_hidden_dim", type=int, default=128)
    parser.add_argument("--latent_dim", type=int, default=16)
    parser.add_argument("--actor_hidden_dim", type=int, default=128)
    parser.add_argument("--actor_num_blocks", type=int, default=2)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def stack_time_key(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset.get(key)
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, list) and value:
        tensor = torch.stack(value)
    else:
        raise ValueError(f"Dataset key {key!r} must be a non-empty tensor or list.")
    if tensor.ndim != 3:
        raise ValueError(f"Expected {key!r} [time, env, dim], got {tuple(tensor.shape)}.")
    return tensor.float()


def make_previous_history(observations: torch.Tensor, history_length: int) -> torch.Tensor:
    """Return [T,E,H,D] histories containing frames strictly before t."""

    if history_length < 1:
        raise ValueError("history_length must be positive.")
    time_steps = observations.shape[0]
    time_ids = torch.arange(time_steps).unsqueeze(1)
    offsets = torch.arange(history_length, 0, -1).unsqueeze(0)
    history_ids = (time_ids - offsets).clamp_min(0)
    return observations[history_ids].permute(0, 2, 1, 3).contiguous()


def trajectory_split(num_envs: int, fraction: float, seed: int) -> tuple[list[int], list[int]]:
    if num_envs < 2:
        raise ValueError("Trajectory validation requires at least two environments.")
    num_validation = min(num_envs - 1, max(1, round(num_envs * fraction)))
    permutation = torch.randperm(num_envs, generator=torch.Generator().manual_seed(seed)).tolist()
    return sorted(permutation[num_validation:]), sorted(permutation[:num_validation])


def select_trajectories(
    histories: torch.Tensor,
    observations: torch.Tensor,
    base_linear_velocities: torch.Tensor,
    actions: torch.Tensor,
    env_ids: list[int],
) -> TensorDataset:
    time_steps = observations.shape[0]
    selected_history = histories[:, env_ids].reshape(
        time_steps * len(env_ids),
        histories.shape[-2],
        histories.shape[-1],
    )
    selected_observation = observations[:, env_ids].reshape(-1, observations.shape[-1])
    selected_velocity = base_linear_velocities[:, env_ids].reshape(-1, 3)
    selected_action = actions[:, env_ids].reshape(-1, actions.shape[-1])
    finite = (
        torch.isfinite(selected_history).all(dim=(1, 2))
        & torch.isfinite(selected_observation).all(dim=1)
        & torch.isfinite(selected_velocity).all(dim=1)
        & torch.isfinite(selected_action).all(dim=1)
    )
    return TensorDataset(
        selected_history[finite].contiguous(),
        selected_observation[finite].contiguous(),
        selected_velocity[finite].contiguous(),
        selected_action[finite].contiguous(),
    )


@torch.no_grad()
def evaluate(
    policy: CENetBCPolicy,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, Any]:
    policy.eval()
    action_squared_error = 0.0
    action_absolute_error = 0.0
    action_target_squared = 0.0
    action_max_error = 0.0
    velocity_squared_error = torch.zeros(3, dtype=torch.float64)
    velocity_absolute_error = torch.zeros(3, dtype=torch.float64)
    velocity_count = 0
    action_count = 0
    for histories, observations, velocities, actions in loader:
        histories = histories.to(device, non_blocking=True)
        observations = observations.to(device, non_blocking=True)
        velocities = velocities.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        predictions, estimated_velocities = policy.predict(
            histories,
            observations,
            training=False,
        )
        action_error = predictions - actions
        velocity_error = estimated_velocities - velocities
        action_squared_error += action_error.square().sum().item()
        action_absolute_error += action_error.abs().sum().item()
        action_target_squared += actions.square().sum().item()
        action_max_error = max(action_max_error, action_error.abs().max().item())
        velocity_squared_error += velocity_error.square().sum(dim=0).double().cpu()
        velocity_absolute_error += velocity_error.abs().sum(dim=0).double().cpu()
        velocity_count += velocities.shape[0]
        action_count += action_error.numel()

    action_mse = action_squared_error / action_count
    velocity_mse_per_axis = velocity_squared_error / velocity_count
    velocity_mse = float(velocity_mse_per_axis.mean())
    return {
        "action": {
            "mse": action_mse,
            "rmse": math.sqrt(action_mse),
            "mae": action_absolute_error / action_count,
            "relative_l2": math.sqrt(action_squared_error / max(action_target_squared, 1e-12)),
            "max_abs_error": action_max_error,
        },
        "velocity": {
            "mse": velocity_mse,
            "rmse": math.sqrt(velocity_mse),
            "mae": float(velocity_absolute_error.sum() / (velocity_count * 3)),
            "mse_per_axis": velocity_mse_per_axis.tolist(),
            "rmse_per_axis": torch.sqrt(velocity_mse_per_axis).tolist(),
            "mae_per_axis": (velocity_absolute_error / velocity_count).tolist(),
        },
    }


def save_checkpoint(
    policy: CENetBCPolicy,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
    update_step: int,
    model_config: CENetBCModelConfig,
) -> None:
    torch.save(
        {
            "network_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "update_step": int(update_step),
            "model_config": asdict(model_config),
        },
        path,
    )


def train_epoch(
    policy: CENetBCPolicy,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    velocity_loss_weight: float,
) -> tuple[dict[str, float], int]:
    policy.train()
    action_squared_error = 0.0
    velocity_squared_error = 0.0
    action_count = 0
    velocity_count = 0
    updates = 0
    for histories, observations, velocities, actions in loader:
        histories = histories.to(device, non_blocking=True)
        observations = observations.to(device, non_blocking=True)
        velocities = velocities.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        predictions, estimated_velocities = policy.predict(histories, observations, training=True)
        action_loss = F.mse_loss(predictions, actions)
        velocity_loss = F.mse_loss(estimated_velocities, velocities)
        loss = action_loss + velocity_loss_weight * velocity_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=10.0)
        optimizer.step()
        normalize_unit_parameters(policy.actor)
        action_squared_error += action_loss.item() * actions.numel()
        velocity_squared_error += velocity_loss.item() * velocities.numel()
        action_count += actions.numel()
        velocity_count += velocities.numel()
        updates += 1
    action_mse = action_squared_error / action_count
    velocity_mse = velocity_squared_error / velocity_count
    return {
        "action_mse": action_mse,
        "velocity_mse": velocity_mse,
        "combined_loss": action_mse + velocity_loss_weight * velocity_mse,
    }, updates


def main() -> None:
    args = parse_args()
    cfg = TrainConfig(**vars(args))
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be between zero and one.")
    set_seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = Path(cfg.dataset_path).expanduser().resolve()
    dataset = torch.load(dataset_path, map_location="cpu", weights_only=False)
    if dataset.get("format_version") != "go2_mixed_rwm_dataset_v2":
        raise ValueError(f"Unsupported dataset format: {dataset.get('format_version')!r}.")
    observations = stack_time_key(dataset, "observations")
    states = stack_time_key(dataset, "states")
    actions = stack_time_key(dataset, "actions")
    if observations.shape[-1] != 45 or states.shape[-1] != 45 or actions.shape[-1] != 12:
        raise ValueError(
            "Expected observations/states/actions dimensions 45/45/12, got "
            f"{observations.shape[-1]}/{states.shape[-1]}/{actions.shape[-1]}."
        )
    if observations.shape[:2] != states.shape[:2] or observations.shape[:2] != actions.shape[:2]:
        raise ValueError("Dataset observation/state/action time and environment shapes differ.")
    base_linear_velocities = states[..., :3].contiguous()
    histories = make_previous_history(observations, cfg.history_length)
    train_envs, validation_envs = trajectory_split(
        observations.shape[1],
        cfg.validation_fraction,
        cfg.seed,
    )
    train_dataset = select_trajectories(
        histories,
        observations,
        base_linear_velocities,
        actions,
        train_envs,
    )
    validation_dataset = select_trajectories(
        histories,
        observations,
        base_linear_velocities,
        actions,
        validation_envs,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    model_config = CENetBCModelConfig(
        history_length=cfg.history_length,
        cenet_hidden_dim=cfg.cenet_hidden_dim,
        latent_dim=cfg.latent_dim,
        actor_hidden_dim=cfg.actor_hidden_dim,
        actor_num_blocks=cfg.actor_num_blocks,
    )
    policy = CENetBCPolicy(model_config).to(device)
    normalize_unit_parameters(policy.actor)
    optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.learning_rate * 0.05,
    )

    print(
        f"[CENet-BC] device={device}, observations={tuple(observations.shape)}, "
        f"actions={tuple(actions.shape)}, history={tuple(histories.shape)}"
    )
    print(
        f"[CENet-BC] train_envs={train_envs} ({len(train_dataset)} samples), "
        f"validation_envs={validation_envs} ({len(validation_dataset)} samples)"
    )
    history: list[dict[str, float | int]] = []
    best_validation_action_mse = math.inf
    best_epoch = -1
    update_step = 0
    for epoch in range(1, cfg.epochs + 1):
        train_metrics, updates = train_epoch(
            policy,
            train_loader,
            optimizer,
            device,
            cfg.velocity_loss_weight,
        )
        update_step += updates
        scheduler.step()
        validation = evaluate(policy, validation_loader, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_action_mse": train_metrics["action_mse"],
            "train_velocity_mse": train_metrics["velocity_mse"],
            "validation_action_mse": validation["action"]["mse"],
            "validation_velocity_mse": validation["velocity"]["mse"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if validation["action"]["mse"] < best_validation_action_mse:
            best_validation_action_mse = validation["action"]["mse"]
            best_epoch = epoch
            save_checkpoint(
                policy,
                optimizer,
                scheduler,
                output_dir / "policy.pt",
                update_step,
                model_config,
            )
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"[CENet-BC] epoch={epoch:04d} "
                f"train_action_mse={train_metrics['action_mse']:.8f} "
                f"train_velocity_mse={train_metrics['velocity_mse']:.8f} "
                f"val_action_mse={validation['action']['mse']:.8f} "
                f"val_velocity_rmse={validation['velocity']['rmse']:.6f}"
            )

    checkpoint = torch.load(output_dir / "policy.pt", map_location=device, weights_only=False)
    torch.save(checkpoint, output_dir / "policy_validation_best.pt")
    policy.load_state_dict(checkpoint["network_state_dict"])
    best_train_metrics = evaluate(
        policy,
        DataLoader(train_dataset, batch_size=cfg.batch_size * 2, shuffle=False),
        device,
    )
    best_validation_metrics = evaluate(policy, validation_loader, device)

    all_envs = list(range(observations.shape[1]))
    refit_dataset = select_trajectories(
        histories,
        observations,
        base_linear_velocities,
        actions,
        all_envs,
    )
    refit_loader = DataLoader(
        refit_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed + 1),
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )
    refit_optimizer = torch.optim.AdamW(
        policy.parameters(),
        lr=cfg.learning_rate * 0.1,
        weight_decay=cfg.weight_decay,
    )
    refit_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        refit_optimizer,
        T_max=max(1, cfg.refit_epochs),
        eta_min=cfg.learning_rate * 0.005,
    )
    for refit_epoch in range(1, cfg.refit_epochs + 1):
        refit_train, updates = train_epoch(
            policy,
            refit_loader,
            refit_optimizer,
            device,
            cfg.velocity_loss_weight,
        )
        update_step += updates
        refit_scheduler.step()
        if refit_epoch == 1 or refit_epoch % 10 == 0 or refit_epoch == cfg.refit_epochs:
            print(
                f"[CENet-BC] refit_epoch={refit_epoch:04d} "
                f"action_mse={refit_train['action_mse']:.8f} "
                f"velocity_mse={refit_train['velocity_mse']:.8f}"
            )
    save_checkpoint(
        policy,
        refit_optimizer,
        refit_scheduler,
        output_dir / "policy.pt",
        update_step,
        model_config,
    )
    refit_metrics = evaluate(
        policy,
        DataLoader(refit_dataset, batch_size=cfg.batch_size * 2, shuffle=False),
        device,
    )
    post_refit_validation_metrics = evaluate(policy, validation_loader, device)
    report = {
        "config": asdict(cfg),
        "model_config": asdict(model_config),
        "dataset_format": dataset["format_version"],
        "dataset_metadata": dataset.get("metadata", {}),
        "velocity_label": "states[...,0:3] (body-frame root_link_lin_vel_b)",
        "train_envs": train_envs,
        "validation_envs": validation_envs,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_train_metrics": best_train_metrics,
        "best_validation_metrics": best_validation_metrics,
        "refit_all_metrics": refit_metrics,
        "post_refit_validation_metrics": post_refit_validation_metrics,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "cenet_bc_config.json").write_text(
        json.dumps({"training": asdict(cfg), "model": asdict(model_config)}, indent=2),
        encoding="utf-8",
    )

    deployment_policy = CENetBCDeploymentPolicy(policy.eval()).cpu()
    example_history = torch.zeros(1, cfg.history_length, 45, dtype=torch.float32)
    example_observation = torch.zeros(1, 45, dtype=torch.float32)
    traced = torch.jit.trace(deployment_policy, (example_history, example_observation))
    traced.save(str(output_dir / "policy_jit.pt"))
    try:
        torch.onnx.export(
            deployment_policy,
            (example_history, example_observation),
            output_dir / "policy.onnx",
            input_names=["observation_history", "observation"],
            output_names=["actions", "estimated_base_linear_velocity"],
            dynamic_axes={
                "observation_history": {0: "batch"},
                "observation": {0: "batch"},
                "actions": {0: "batch"},
                "estimated_base_linear_velocity": {0: "batch"},
            },
            opset_version=17,
            dynamo=False,
        )
        (output_dir / "onnx_export_error.txt").unlink(missing_ok=True)
    except Exception as exc:
        (output_dir / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print(f"[CENet-BC] ONNX export skipped: {exc}")

    print(
        f"[CENet-BC] best_epoch={best_epoch}, "
        f"validation={best_validation_metrics}, refit_all={refit_metrics}"
    )
    print(f"[CENet-BC] saved policy to {output_dir}")


if __name__ == "__main__":
    main()
