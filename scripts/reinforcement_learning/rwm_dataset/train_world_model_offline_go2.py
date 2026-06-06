"""Train Go2 RWM-U dynamics offline from a saved mixed simulator dataset."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm.dynamics import DynamicsConfig, SystemDynamicsEnsemble
from scripts.reinforcement_learning.rwm_dataset.dataset import (
    OfflineSamplerConfig,
    OfflineSequenceSampler,
    load_mixed_dataset,
    stack_time_key,
)
from scripts.reinforcement_learning.rwm_dataset.metrics import autoregressive_rollout_metrics
from scripts.reinforcement_learning.rwm_flashsac.utils import configure_low_thread_env, resolve_repo_path, set_seed


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "go2_offline_world_model.yaml"


class ScalarLogger:
    def __init__(self, log_dir: Path) -> None:
        self.writer = None
        try:
            from torch.utils.tensorboard import SummaryWriter

            self.writer = SummaryWriter(log_dir=str(log_dir / "tb"))
        except Exception:
            self.writer = None

    def log(self, values: dict[str, float], step: int) -> None:
        if self.writer is None:
            return
        for key, value in values.items():
            self.writer.add_scalar(key, value, step)
        self.writer.flush()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--save_dir", default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--micro_batch_size", type=int, default=None)
    parser.add_argument("--ensemble_size", type=int, default=None)
    parser.add_argument("--history_horizon", type=int, default=None)
    parser.add_argument("--forecast_horizon", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--save_interval", type=int, default=None)
    parser.add_argument("--log_interval", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(args.config_path)
    updates = list(args.overrides or [])
    scalar_overrides = {
        "dataset_path": args.dataset_path,
        "save_dir": args.save_dir,
        "max_iterations": args.max_iterations,
        "batch_size": args.batch_size,
        "micro_batch_size": args.micro_batch_size,
        "device": args.device,
        "save_interval": args.save_interval,
        "log_interval": args.log_interval,
        "system_dynamics.ensemble_size": args.ensemble_size,
        "system_dynamics.history_horizon": args.history_horizon,
        "system_dynamics.forecast_horizon": args.forecast_horizon,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            updates.append(f"{key}={value}")
    if updates:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(updates))
    OmegaConf.resolve(cfg)
    return cfg


def _obs_stats(dataset: dict[str, Any]) -> dict[str, torch.Tensor]:
    if "observations" not in dataset:
        return {}
    obs = stack_time_key(dataset, "observations").float().reshape(-1, int(dataset["metadata"].get("obs_dim", 48)))
    return {
        "obs_mean": obs.mean(dim=0),
        "obs_std": obs.std(dim=0).clamp_min(1e-6),
    }


def _make_checkpoint(
    *,
    dynamics: SystemDynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    infos: dict[str, Any],
    normalizer: dict[str, torch.Tensor],
    cfg: Any,
    dataset_metadata: dict[str, Any],
) -> dict[str, Any]:
    checkpoint = dynamics.checkpoint(optimizer=optimizer, iteration=iteration, infos=infos)
    checkpoint["normalizer"] = {key: value.detach().cpu() for key, value in normalizer.items()}
    checkpoint["config"] = OmegaConf.to_container(cfg, resolve=True)
    checkpoint["dataset_metadata"] = dataset_metadata
    return checkpoint


def _save_checkpoint(
    *,
    save_root: Path,
    dynamics: SystemDynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    infos: dict[str, Any],
    normalizer: dict[str, torch.Tensor],
    cfg: Any,
    dataset_metadata: dict[str, Any],
) -> None:
    checkpoint = _make_checkpoint(
        dynamics=dynamics,
        optimizer=optimizer,
        iteration=iteration,
        infos=infos,
        normalizer=normalizer,
        cfg=cfg,
        dataset_metadata=dataset_metadata,
    )
    torch.save(checkpoint, save_root / f"model_{iteration}.pt")
    torch.save(checkpoint, save_root / "latest.pt")


def _train_with_micro_batches(
    *,
    dynamics: SystemDynamicsEnsemble,
    optimizer: torch.optim.Optimizer,
    sampler: OfflineSequenceSampler,
    batch_size: int,
    micro_batch_size: int,
    device: torch.device,
) -> dict[str, float]:
    """Train one effective batch while bounding GRU activation memory."""

    dynamics.train()
    optimizer.zero_grad(set_to_none=True)
    micro_batch_size = max(1, min(int(micro_batch_size), int(batch_size)))
    num_micro_batches = int(np.ceil(int(batch_size) / micro_batch_size))
    totals: dict[str, float] = {}
    processed = 0
    for micro_idx in range(num_micro_batches):
        current = min(micro_batch_size, int(batch_size) - processed)
        if current <= 0:
            break
        batch = sampler.sample(current, device=device, split="train")
        loss_dict = dynamics.compute_loss(*batch, bootstrap=True)
        scale = float(current) / float(batch_size)
        (loss_dict["total_loss"] * scale).backward()
        processed += current
        for key, value in loss_dict.items():
            totals[key] = totals.get(key, 0.0) + float(value.detach().cpu()) * scale
        # Release references before the next GRU micro-batch.
        del batch, loss_dict
        if device.type == "cuda" and micro_idx + 1 < num_micro_batches:
            torch.cuda.empty_cache()
    torch.nn.utils.clip_grad_norm_(dynamics.parameters(), max_norm=10.0)
    optimizer.step()
    return totals


def main() -> None:
    configure_low_thread_env()
    args = _parse_args()
    cfg = _load_config(args)
    set_seed(int(cfg.seed))
    device = torch.device(str(cfg.device) if torch.cuda.is_available() or not str(cfg.device).startswith("cuda") else "cpu")

    dataset_path = resolve_repo_path(str(cfg.dataset_path))
    dataset = load_mixed_dataset(dataset_path)
    sampler_cfg = OfflineSamplerConfig(
        history_horizon=int(cfg.system_dynamics.history_horizon),
        forecast_horizon=int(cfg.system_dynamics.forecast_horizon),
        train_fraction=float(cfg.train_fraction),
        seed=int(cfg.seed),
    )
    sampler = OfflineSequenceSampler(dataset, sampler_cfg)
    state_mean, state_std, action_mean, action_std = sampler.stats()
    normalizer = {
        "state_mean": state_mean,
        "state_std": state_std,
        "action_mean": action_mean,
        "action_std": action_std,
        **_obs_stats(dataset),
    }

    dynamics_cfg = DynamicsConfig(
        state_dim=sampler.state_dim,
        action_dim=sampler.action_dim,
        contact_dim=sampler.contact_dim,
        termination_dim=sampler.termination_dim,
        ensemble_size=int(cfg.system_dynamics.ensemble_size),
        history_horizon=int(cfg.system_dynamics.history_horizon),
        forecast_horizon=int(cfg.system_dynamics.forecast_horizon),
        hidden_size=int(cfg.system_dynamics.hidden_size),
        num_layers=int(cfg.system_dynamics.num_layers),
        contact_loss_weight=float(cfg.system_dynamics.contact_loss_weight),
        termination_loss_weight=float(cfg.system_dynamics.termination_loss_weight),
    )
    dynamics = SystemDynamicsEnsemble(dynamics_cfg).to(device)
    dynamics.set_normalizers(state_mean, state_std, action_mean, action_std)
    optimizer = torch.optim.Adam(
        dynamics.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    save_base = resolve_repo_path(str(cfg.save_dir))
    save_root = save_base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_root / "go2_offline_world_model.yaml")
    with (save_root / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(sampler.metadata(), f, indent=2, default=str)
    logger = ScalarLogger(save_root)

    print(f"[Go2-OfflineRWM] dataset={dataset_path}")
    print(f"[Go2-OfflineRWM] save_root={save_root}")
    print(f"[Go2-OfflineRWM] device={device}, transitions={sampler.num_transitions}")
    print(f"[Go2-OfflineRWM] train_sequences={sampler.train_indices.shape[0]}, val_sequences={sampler.val_indices.shape[0]}")
    print(f"[Go2-OfflineRWM] batch_size={int(cfg.batch_size)}, micro_batch_size={int(cfg.micro_batch_size)}")
    print(f"[Go2-OfflineRWM] num_workers={args.num_workers} (sampling is in-process CPU tensor indexing)")

    start_time = time.perf_counter()
    latest_metrics: dict[str, float] = {}
    for iteration in tqdm.trange(1, int(cfg.max_iterations) + 1, smoothing=0.1, mininterval=0.5):
        loss_values = _train_with_micro_batches(
            dynamics=dynamics,
            optimizer=optimizer,
            sampler=sampler,
            batch_size=int(cfg.batch_size),
            micro_batch_size=int(cfg.micro_batch_size),
            device=device,
        )

        latest_metrics = {
            "Model/train_total_loss": float(loss_values["total_loss"]),
            "Model/train_state_loss": float(loss_values["state_loss"]),
            "Model/train_sequence_loss": float(loss_values["sequence_loss"]),
            "Model/train_contact_loss": float(loss_values["contact_loss"]),
            "Model/train_termination_loss": float(loss_values["termination_loss"]),
            "Dataset/replay_size": float(sampler.num_transitions),
            "Dataset/train_sequences": float(sampler.train_indices.shape[0]),
            "Dataset/val_sequences": float(sampler.val_indices.shape[0]),
        }

        if iteration % int(cfg.log_interval) == 0 or iteration == 1:
            dynamics.eval()
            with torch.no_grad():
                eval_batch_size = min(int(cfg.batch_size), int(cfg.micro_batch_size))
                eval_batch = sampler.sample(eval_batch_size, device=device, split="val")
                eval_loss = dynamics.compute_loss(*eval_batch, bootstrap=False)
                available_rollout = max(1, sampler.num_time_steps - dynamics.cfg.history_horizon)
                rollout_horizon = min(100, available_rollout)
                rollout_batch = sampler.sample(
                    min(eval_batch_size, 512),
                    device=device,
                    split="val",
                    forecast_horizon=max(rollout_horizon, int(cfg.system_dynamics.forecast_horizon)),
                )
                rollout = autoregressive_rollout_metrics(
                    dynamics,
                    rollout_batch,
                    rollout_horizon=rollout_horizon,
                )
            latest_metrics.update(
                {
                    "Model/eval_state_loss": float(eval_loss["state_loss"].detach().cpu()),
                    "Model/eval_sequence_loss": float(eval_loss["sequence_loss"].detach().cpu()),
                }
            )
            latest_metrics.update({f"Model/{key}": value for key, value in rollout.items()})
            latest_metrics["Perf/elapsed_seconds"] = float(time.perf_counter() - start_time)
            logger.log(latest_metrics, iteration)
            interesting = {
                key: latest_metrics[key]
                for key in (
                    "Model/train_state_loss",
                    "Model/eval_state_loss",
                    "Model/traj_autoregressive_error",
                    "Model/epistemic_uncertainty_mean",
                    "Dataset/replay_size",
                )
                if key in latest_metrics
            }
            print(f"[Go2-OfflineRWM] iter={iteration} {interesting}")

        if iteration % int(cfg.save_interval) == 0:
            _save_checkpoint(
                save_root=save_root,
                dynamics=dynamics,
                optimizer=optimizer,
                iteration=iteration,
                infos={
                    "dataset_path": str(dataset_path),
                    "metrics": dict(latest_metrics),
                    **sampler.metadata(),
                },
                normalizer=normalizer,
                cfg=cfg,
                dataset_metadata=sampler.metadata(),
            )

    _save_checkpoint(
        save_root=save_root,
        dynamics=dynamics,
        optimizer=optimizer,
        iteration=int(cfg.max_iterations),
        infos={
            "dataset_path": str(dataset_path),
            "metrics": dict(latest_metrics),
            **sampler.metadata(),
        },
        normalizer=normalizer,
        cfg=cfg,
        dataset_metadata=sampler.metadata(),
    )
    with (save_root / "final_metrics.json").open("w", encoding="utf-8") as f:
        json.dump(latest_metrics, f, indent=2)
    logger.close()
    print(f"[Go2-OfflineRWM] saved final checkpoint: {save_root / f'model_{int(cfg.max_iterations)}.pt'}")


if __name__ == "__main__":
    main()
