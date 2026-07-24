"""Train a deterministic Go2 policy with behavior cloning.

The input is a ``go2_mixed_rwm_dataset_v2`` file produced by the RWM dataset
collector.  Environment trajectories, rather than individual transitions, are
split between train and validation sets to avoid temporal leakage.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from flash_rl.agents.flashSAC.network import FlashSACActor


@dataclass(frozen=True)
class TrainConfig:
    dataset_path: str
    output_dir: str
    observation_key: str
    action_key: str
    seed: int
    validation_fraction: float
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    hidden_dim: int
    num_blocks: int
    refit_epochs: int
    device: str
    num_workers: int


class DeterministicActor(nn.Module):
    """Deployment wrapper returning the tanh of the FlashSAC actor mean."""

    def __init__(self, actor: FlashSACActor) -> None:
        super().__init__()
        self.actor = actor

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        # Spell out RMSNorm because the legacy ONNX exporter does not support
        # aten::rms_norm. This is algebraically identical to UnitRMSNorm.
        x = self.actor.embedder(observations, training=False)
        for block in self.actor.encoder:
            x = block(x, training=False)
        x = x * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.actor.post_norm.eps)
        x = x * self.actor.post_norm.weight
        mean = F.linear(
            x,
            self.actor.predictor.mean_w.w.weight,
            self.actor.predictor.mean_bias,
        )
        return torch.tanh(mean)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--observation_key", default="observations")
    parser.add_argument("--action_key", default="actions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--validation_fraction", type=float, default=0.2)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--num_blocks", type=int, default=2)
    parser.add_argument(
        "--refit_epochs",
        type=int,
        default=100,
        help="After model selection, fine-tune on every trajectory for deployment.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=0)
    return parser.parse_args()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stack_time_key(dataset: dict[str, Any], key: str) -> torch.Tensor:
    value = dataset.get(key)
    if isinstance(value, torch.Tensor):
        tensor = value
    elif isinstance(value, list) and value:
        tensor = torch.stack(value)
    else:
        raise ValueError(f"Dataset key {key!r} must be a non-empty tensor or list.")
    if tensor.ndim != 3:
        raise ValueError(f"Expected {key!r} with shape [time, env, dim], got {tuple(tensor.shape)}.")
    return tensor.float()


def _normalize_unit_parameters(module: nn.Module) -> None:
    with torch.no_grad():
        for child in module.modules():
            normalize = getattr(child, "normalize_parameters", None)
            if callable(normalize):
                normalize()


def _predict(actor: FlashSACActor, observations: torch.Tensor, *, training: bool) -> torch.Tensor:
    mean, _std = actor.get_mean_and_std(observations, training=training)
    return torch.tanh(mean)


@torch.no_grad()
def _metrics(actor: FlashSACActor, loader: DataLoader, device: torch.device) -> dict[str, float]:
    actor.eval()
    squared_error = 0.0
    absolute_error = 0.0
    target_squared = 0.0
    max_error = 0.0
    count = 0
    for observations, actions in loader:
        observations = observations.to(device, non_blocking=True)
        actions = actions.to(device, non_blocking=True)
        predictions = _predict(actor, observations, training=False)
        error = predictions - actions
        squared_error += error.square().sum().item()
        absolute_error += error.abs().sum().item()
        target_squared += actions.square().sum().item()
        max_error = max(max_error, error.abs().max().item())
        count += error.numel()
    mse = squared_error / count
    return {
        "mse": mse,
        "rmse": math.sqrt(mse),
        "mae": absolute_error / count,
        "relative_l2": math.sqrt(squared_error / max(target_squared, 1e-12)),
        "max_abs_error": max_error,
    }


def _make_split(
    observations: torch.Tensor,
    actions: torch.Tensor,
    *,
    validation_fraction: float,
    seed: int,
) -> tuple[TensorDataset, TensorDataset, list[int], list[int]]:
    if observations.shape[:2] != actions.shape[:2]:
        raise ValueError(
            "Observation/action time and environment shapes differ: "
            f"{tuple(observations.shape)} versus {tuple(actions.shape)}."
        )
    num_envs = int(observations.shape[1])
    if num_envs < 2:
        raise ValueError("Trajectory-level validation requires at least two environments.")
    num_validation = min(num_envs - 1, max(1, round(num_envs * validation_fraction)))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_envs, generator=generator).tolist()
    validation_envs = sorted(permutation[:num_validation])
    train_envs = sorted(permutation[num_validation:])

    def select(env_ids: list[int]) -> TensorDataset:
        obs = observations[:, env_ids].reshape(-1, observations.shape[-1]).contiguous()
        act = actions[:, env_ids].reshape(-1, actions.shape[-1]).contiguous()
        finite = torch.isfinite(obs).all(dim=-1) & torch.isfinite(act).all(dim=-1)
        if not finite.all():
            obs, act = obs[finite], act[finite]
        return TensorDataset(obs, act)

    return select(train_envs), select(validation_envs), train_envs, validation_envs


def _save_actor_checkpoint(
    actor: FlashSACActor,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
    update_step: int,
) -> None:
    torch.save(
        {
            "network_state_dict": actor.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "update_step": update_step,
        },
        path,
    )


def main() -> None:
    args = _parse_args()
    cfg = TrainConfig(**vars(args))
    if not 0.0 < cfg.validation_fraction < 1.0:
        raise ValueError("--validation_fraction must be between zero and one.")
    _set_seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    output_dir = Path(cfg.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset = torch.load(
        Path(cfg.dataset_path).expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if dataset.get("format_version") != "go2_mixed_rwm_dataset_v2":
        raise ValueError(f"Unsupported dataset format: {dataset.get('format_version')!r}.")
    observations = _stack_time_key(dataset, cfg.observation_key)
    actions = _stack_time_key(dataset, cfg.action_key)
    if int(dataset.get("state_dim", -1)) != 45 or actions.shape[-1] != 12:
        raise ValueError(
            f"Expected Go2 45-dimensional data and 12 actions, got "
            f"state_dim={dataset.get('state_dim')}, action_dim={actions.shape[-1]}."
        )

    train_dataset, validation_dataset, train_envs, validation_envs = _make_split(
        observations,
        actions,
        validation_fraction=cfg.validation_fraction,
        seed=cfg.seed,
    )
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(cfg.seed),
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
    )

    actor = FlashSACActor(
        num_blocks=cfg.num_blocks,
        input_dim=observations.shape[-1],
        hidden_dim=cfg.hidden_dim,
        action_dim=actions.shape[-1],
    ).to(device)
    _normalize_unit_parameters(actor)
    optimizer = torch.optim.AdamW(
        actor.parameters(),
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=cfg.epochs,
        eta_min=cfg.learning_rate * 0.05,
    )

    history: list[dict[str, float | int]] = []
    best_validation_mse = math.inf
    best_epoch = -1
    update_step = 0
    print(
        f"[BC] device={device}, observations={tuple(observations.shape)}, "
        f"actions={tuple(actions.shape)}"
    )
    print(
        f"[BC] train_envs={train_envs} ({len(train_dataset)} samples), "
        f"validation_envs={validation_envs} ({len(validation_dataset)} samples)"
    )
    for epoch in range(1, cfg.epochs + 1):
        actor.train()
        train_squared_error = 0.0
        train_count = 0
        for batch_observations, batch_actions in train_loader:
            batch_observations = batch_observations.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            predictions = _predict(actor, batch_observations, training=True)
            loss = F.mse_loss(predictions, batch_actions)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
            optimizer.step()
            _normalize_unit_parameters(actor)
            train_squared_error += loss.item() * batch_actions.numel()
            train_count += batch_actions.numel()
            update_step += 1
        scheduler.step()

        validation = _metrics(actor, validation_loader, device)
        train_mse = train_squared_error / train_count
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_mse": train_mse,
            "validation_mse": validation["mse"],
            "validation_mae": validation["mae"],
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        if validation["mse"] < best_validation_mse:
            best_validation_mse = validation["mse"]
            best_epoch = epoch
            _save_actor_checkpoint(actor, optimizer, scheduler, output_dir / "actor.pt", update_step)
        if epoch == 1 or epoch % 10 == 0 or epoch == cfg.epochs:
            print(
                f"[BC] epoch={epoch:04d} train_mse={train_mse:.8f} "
                f"val_mse={validation['mse']:.8f} val_mae={validation['mae']:.6f}"
            )

    checkpoint = torch.load(output_dir / "actor.pt", map_location=device, weights_only=False)
    torch.save(checkpoint, output_dir / "actor_validation_best.pt")
    actor.load_state_dict(checkpoint["network_state_dict"])
    validation_best_train = _metrics(
        actor,
        DataLoader(train_dataset, batch_size=cfg.batch_size * 2, shuffle=False),
        device,
    )
    validation_best_validation = _metrics(actor, validation_loader, device)

    refit_dataset = TensorDataset(
        observations.reshape(-1, observations.shape[-1]).contiguous(),
        actions.reshape(-1, actions.shape[-1]).contiguous(),
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
        actor.parameters(),
        lr=cfg.learning_rate * 0.1,
        weight_decay=cfg.weight_decay,
    )
    refit_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        refit_optimizer,
        T_max=max(1, cfg.refit_epochs),
        eta_min=cfg.learning_rate * 0.005,
    )
    for refit_epoch in range(1, cfg.refit_epochs + 1):
        actor.train()
        refit_squared_error = 0.0
        refit_count = 0
        for batch_observations, batch_actions in refit_loader:
            batch_observations = batch_observations.to(device, non_blocking=True)
            batch_actions = batch_actions.to(device, non_blocking=True)
            predictions = _predict(actor, batch_observations, training=True)
            loss = F.mse_loss(predictions, batch_actions)
            refit_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(actor.parameters(), max_norm=10.0)
            refit_optimizer.step()
            _normalize_unit_parameters(actor)
            refit_squared_error += loss.item() * batch_actions.numel()
            refit_count += batch_actions.numel()
            update_step += 1
        refit_scheduler.step()
        if refit_epoch == 1 or refit_epoch % 10 == 0 or refit_epoch == cfg.refit_epochs:
            print(
                f"[BC] refit_epoch={refit_epoch:04d} "
                f"full_train_mse={refit_squared_error / refit_count:.8f}"
            )
    _save_actor_checkpoint(
        actor,
        refit_optimizer,
        refit_scheduler,
        output_dir / "actor.pt",
        update_step,
    )
    refit_metrics = _metrics(
        actor,
        DataLoader(refit_dataset, batch_size=cfg.batch_size * 2, shuffle=False),
        device,
    )
    post_refit_validation = _metrics(actor, validation_loader, device)
    zero_mse = actions[:, validation_envs].square().mean().item()
    report = {
        "config": asdict(cfg),
        "dataset_format": dataset["format_version"],
        "dataset_metadata": dataset.get("metadata", {}),
        "observation_dim": int(observations.shape[-1]),
        "action_dim": int(actions.shape[-1]),
        "train_envs": train_envs,
        "validation_envs": validation_envs,
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset),
        "best_epoch": best_epoch,
        "best_train_metrics": validation_best_train,
        "best_validation_metrics": validation_best_validation,
        "refit_all_metrics": refit_metrics,
        "post_refit_validation_metrics": post_refit_validation,
        "validation_zero_action_mse": zero_mse,
        "history": history,
    }
    (output_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_dir / "bc_config.json").write_text(
        json.dumps(asdict(cfg), indent=2),
        encoding="utf-8",
    )

    deployment_actor = DeterministicActor(actor.eval()).cpu()
    example = torch.zeros(1, observations.shape[-1], dtype=torch.float32)
    traced = torch.jit.trace(deployment_actor, example)
    traced.save(str(output_dir / "policy_jit.pt"))
    try:
        torch.onnx.export(
            deployment_actor,
            example,
            output_dir / "policy.onnx",
            input_names=["observations"],
            output_names=["actions"],
            dynamic_axes={"observations": {0: "batch"}, "actions": {0: "batch"}},
            opset_version=17,
            dynamo=False,
        )
        (output_dir / "onnx_export_error.txt").unlink(missing_ok=True)
    except Exception as exc:
        (output_dir / "onnx_export_error.txt").write_text(str(exc), encoding="utf-8")
        print(f"[BC] ONNX export skipped: {exc}")

    print(
        f"[BC] best_epoch={best_epoch}, validation={validation_best_validation}, "
        f"refit_all={refit_metrics}, "
        f"zero_action_mse={zero_mse:.8f}"
    )
    print(f"[BC] saved policy to {output_dir}")


if __name__ == "__main__":
    main()
