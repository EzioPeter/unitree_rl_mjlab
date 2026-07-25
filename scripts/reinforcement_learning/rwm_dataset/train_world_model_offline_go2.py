"""Train Go2 RWM-U dynamics offline from a saved mixed simulator dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
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
from scripts.reinforcement_learning.rwm_dataset.action_mask import mask_dataset_actions, normalize_action_mask_indices
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
    parser.add_argument("--resume_path", default=None)
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
    parser.add_argument("--action_mask_indices", nargs="*", type=int, default=None)
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


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
    if args.action_mask_indices is not None:
        indices = ",".join(str(int(idx)) for idx in args.action_mask_indices)
        updates.append(f"action_mask_indices=[{indices}]")
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
    checkpoint["rng_state"] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    return checkpoint


def _resume_config_view(config: dict[str, Any]) -> dict[str, Any]:
    """Return the fields that must remain identical for an exact continuation."""

    ignored = {"dataset_path", "save_dir", "device", "max_iterations"}
    return {key: value for key, value in config.items() if key not in ignored}


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
    include_base_lin_vel_confidence: bool = False,
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
        batch = sampler.sample(
            current,
            device=device,
            split="train",
            include_base_lin_vel_confidence=include_base_lin_vel_confidence,
        )
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
    action_mask_indices = normalize_action_mask_indices(cfg.get("action_mask_indices", []))
    action_mask_indices = mask_dataset_actions(dataset, action_mask_indices)
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
        state_loss_weight=float(cfg.system_dynamics.get("state_loss_weight", 1.0)),
        sequence_loss_weight=float(cfg.system_dynamics.get("sequence_loss_weight", 1.0)),
        bound_loss_weight=float(cfg.system_dynamics.get("bound_loss_weight", 1.0)),
        kl_loss_weight=float(cfg.system_dynamics.get("kl_loss_weight", 0.1)),
        extension_loss_weight=float(cfg.system_dynamics.get("extension_loss_weight", 1.0)),
        contact_loss_weight=float(cfg.system_dynamics.contact_loss_weight),
        termination_loss_weight=float(cfg.system_dynamics.termination_loss_weight),
        loss_mode=str(cfg.system_dynamics.get("loss_mode", "teacher_forced_nll")),
    )
    dynamics = SystemDynamicsEnsemble(dynamics_cfg).to(device)
    dynamics.set_normalizers(state_mean, state_std, action_mean, action_std)
    optimizer = torch.optim.Adam(
        dynamics.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )
    start_iteration = 0
    resume_metadata: dict[str, Any] = {}
    checkpoint: dict[str, Any] | None = None
    if args.resume_path is not None:
        resume_path = resolve_repo_path(str(args.resume_path))
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {resume_path}")
        checkpoint = torch.load(resume_path, map_location=device, weights_only=False)
        start_iteration = int(checkpoint.get("iter", -1))
        if start_iteration < 1 or start_iteration >= int(cfg.max_iterations):
            raise ValueError(
                f"Resume iteration {start_iteration} must be in "
                f"[1, {int(cfg.max_iterations) - 1}]."
            )
        checkpoint_infos = checkpoint.get("infos") or {}
        checkpoint_dataset = checkpoint_infos.get("dataset_path")
        if checkpoint_dataset is None:
            raise ValueError("Resume checkpoint does not record its dataset path.")
        if resolve_repo_path(str(checkpoint_dataset)) != dataset_path:
            raise ValueError(
                "Resume checkpoint dataset does not match the current dataset."
            )
        checkpoint_config = checkpoint.get("config") or {}
        current_config = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(checkpoint_config, dict) or not isinstance(current_config, dict):
            raise ValueError("Resume/current config must resolve to dictionaries.")
        if _resume_config_view(checkpoint_config) != _resume_config_view(current_config):
            raise ValueError(
                "Resume checkpoint training config differs from the current config "
                "outside dataset_path/save_dir/device/max_iterations."
            )
        checkpoint_mask = tuple(
            int(index) for index in checkpoint_config.get("action_mask_indices", [])
        )
        if checkpoint_mask != tuple(action_mask_indices):
            raise ValueError(
                "Resume checkpoint action mask does not match the current config."
            )
        checkpoint_normalizer = checkpoint.get("normalizer") or {}
        for name, expected in normalizer.items():
            actual = checkpoint_normalizer.get(name)
            if not isinstance(actual, torch.Tensor):
                raise ValueError(f"Resume checkpoint is missing normalizer {name!r}.")
            if actual.shape != expected.shape or not torch.allclose(
                actual.detach().cpu(),
                expected.detach().cpu(),
                rtol=1.0e-6,
                atol=1.0e-7,
            ):
                raise ValueError(
                    f"Resume checkpoint normalizer {name!r} does not match the dataset."
                )
        optimizer_state = checkpoint.get("system_dynamics_optimizer_state_dict")
        if not isinstance(optimizer_state, dict):
            raise ValueError("Resume checkpoint does not contain optimizer state.")
        dynamics.load_state_dict(
            checkpoint["system_dynamics_state_dict"],
            strict=True,
        )
        optimizer.load_state_dict(optimizer_state)
        rng_state = checkpoint.get("rng_state")
        if not isinstance(rng_state, dict):
            raise ValueError(
                "Resume checkpoint does not contain RNG state; exact continuation "
                "cannot be guaranteed."
            )
        required_rng_keys = {"python", "numpy", "torch_cpu", "torch_cuda"}
        missing_rng_keys = required_rng_keys.difference(rng_state)
        if missing_rng_keys:
            raise ValueError(
                f"Resume checkpoint RNG state is incomplete: {sorted(missing_rng_keys)}."
            )
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch_cpu"].cpu())
        cuda_rng_state = rng_state["torch_cuda"]
        if cuda_rng_state is not None:
            if not torch.cuda.is_available():
                raise ValueError(
                    "Resume checkpoint contains CUDA RNG state but CUDA is unavailable."
                )
            if len(cuda_rng_state) != torch.cuda.device_count():
                raise ValueError(
                    "Resume checkpoint CUDA RNG device count does not match the "
                    "current visible CUDA device count."
                )
            torch.cuda.set_rng_state_all(
                [state.detach().cpu() for state in cuda_rng_state]
            )
        resume_metadata = {
            "resumed": True,
            "resume_checkpoint_path": str(resume_path),
            "resume_checkpoint_sha256": _sha256_file(resume_path),
            "resume_iteration": start_iteration,
            "rng_state_restored": True,
        }

    save_base = resolve_repo_path(str(cfg.save_dir))
    save_root = save_base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_root / "go2_offline_world_model.yaml")
    with (save_root / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(sampler.metadata(), f, indent=2, default=str)
    if resume_metadata:
        with (save_root / "resume_metadata.json").open("w", encoding="utf-8") as f:
            json.dump(resume_metadata, f, indent=2, sort_keys=True)
    logger = ScalarLogger(save_root)

    print(f"[Go2-OfflineRWM] dataset={dataset_path}")
    print(f"[Go2-OfflineRWM] save_root={save_root}")
    print(f"[Go2-OfflineRWM] device={device}, transitions={sampler.num_transitions}")
    print(f"[Go2-OfflineRWM] train_sequences={sampler.train_indices.shape[0]}, val_sequences={sampler.val_indices.shape[0]}")
    print(f"[Go2-OfflineRWM] batch_size={int(cfg.batch_size)}, micro_batch_size={int(cfg.micro_batch_size)}")
    print(f"[Go2-OfflineRWM] num_workers={args.num_workers} (sampling is in-process CPU tensor indexing)")
    print(f"[Go2-OfflineRWM] action_mask_indices={list(action_mask_indices)}")
    if resume_metadata:
        print(
            "[Go2-OfflineRWM] "
            f"resume_checkpoint={resume_metadata['resume_checkpoint_path']}, "
            f"resume_iteration={start_iteration}, "
            "optimizer_restored=true, rng_state_restored=true"
        )

    start_time = time.perf_counter()
    latest_metrics: dict[str, float] = dict(
        (checkpoint.get("infos") or {}).get("metrics") or {}
    ) if checkpoint is not None else {}
    for iteration in tqdm.trange(
        start_iteration + 1,
        int(cfg.max_iterations) + 1,
        smoothing=0.1,
        mininterval=0.5,
    ):
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
            "Model/train_bound_loss": float(loss_values.get("bound_loss", 0.0)),
            "Model/train_kl_loss": float(loss_values.get("kl_loss", 0.0)),
            "Model/train_extension_loss": float(loss_values.get("extension_loss", 0.0)),
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
                    "Model/eval_bound_loss": float(eval_loss.get("bound_loss", torch.tensor(0.0)).detach().cpu()),
                    "Model/eval_kl_loss": float(eval_loss.get("kl_loss", torch.tensor(0.0)).detach().cpu()),
                    "Model/eval_extension_loss": float(eval_loss.get("extension_loss", torch.tensor(0.0)).detach().cpu()),
                    "Model/eval_contact_loss": float(eval_loss["contact_loss"].detach().cpu()),
                    "Model/eval_termination_loss": float(eval_loss["termination_loss"].detach().cpu()),
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
                    "action_mask_indices": list(action_mask_indices),
                    "metrics": dict(latest_metrics),
                    **resume_metadata,
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
            "action_mask_indices": list(action_mask_indices),
            "metrics": dict(latest_metrics),
            **resume_metadata,
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
