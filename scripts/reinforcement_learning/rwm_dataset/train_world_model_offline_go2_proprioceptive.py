"""Train a Go2 offline RWM with a true 42-dim state input.

The model receives ``full_state[..., 3:45]`` plus the 12-d action history and
predicts the original 45-d next state, contacts, and termination.
"""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import tqdm
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import OfflineSamplerConfig, OfflineSequenceSampler, load_mixed_dataset
from scripts.reinforcement_learning.rwm_dataset.metrics import autoregressive_rollout_metrics
from scripts.reinforcement_learning.rwm_dataset.proprioceptive_dynamics import (
    ProprioceptiveDynamicsConfig,
    ProprioceptiveSystemDynamicsEnsemble,
)
from scripts.reinforcement_learning.rwm_dataset.train_world_model_offline_go2 import (
    ScalarLogger,
    _load_config,
    _obs_stats,
    _parse_args,
    _save_checkpoint,
    _train_with_micro_batches,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import configure_low_thread_env, resolve_repo_path, set_seed


def main() -> None:
    configure_low_thread_env()
    args = _parse_args()
    cfg = _load_config(args)
    OmegaConf.update(cfg, "system_dynamics.input_state_dim", 42, merge=True)
    OmegaConf.update(cfg, "system_dynamics.output_state_dim", 45, merge=True)
    OmegaConf.update(cfg, "system_dynamics.dropped_state_indices", [0, 1, 2], merge=True)
    OmegaConf.update(cfg, "system_dynamics.model_type", "go2_proprioceptive", merge=True)
    OmegaConf.resolve(cfg)

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
        "output_state_mean": state_mean,
        "output_state_std": state_std,
        "input_state_mean": state_mean[3:],
        "input_state_std": state_std[3:],
        "action_mean": action_mean,
        "action_std": action_std,
        **_obs_stats(dataset),
    }

    dynamics_cfg = ProprioceptiveDynamicsConfig(
        input_state_dim=42,
        output_state_dim=sampler.state_dim,
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
        loss_mode=str(cfg.system_dynamics.get("loss_mode", "reference_autoregressive_mse")),
        dropped_state_indices=(0, 1, 2),
    )
    dynamics = ProprioceptiveSystemDynamicsEnsemble(dynamics_cfg).to(device)
    dynamics.set_normalizers(state_mean, state_std, action_mean, action_std)
    optimizer = torch.optim.Adam(
        dynamics.parameters(),
        lr=float(cfg.learning_rate),
        weight_decay=float(cfg.weight_decay),
    )

    save_base = resolve_repo_path(str(cfg.save_dir))
    save_root = save_base / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    save_root.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(cfg, save_root / "go2_offline_world_model_proprioceptive.yaml")
    with (save_root / "dataset_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(sampler.metadata(), f, indent=2, default=str)
    logger = ScalarLogger(save_root)

    print(f"[Go2-OfflineRWM-Proprioceptive] dataset={dataset_path}")
    print(f"[Go2-OfflineRWM-Proprioceptive] save_root={save_root}")
    print("[Go2-OfflineRWM-Proprioceptive] model input: state[..., 3:45] (42 dims) + action (12 dims)")
    print("[Go2-OfflineRWM-Proprioceptive] model output: full next_state (45 dims), contact, termination")
    print(f"[Go2-OfflineRWM-Proprioceptive] device={device}, transitions={sampler.num_transitions}")
    print(f"[Go2-OfflineRWM-Proprioceptive] train_sequences={sampler.train_indices.shape[0]}, val_sequences={sampler.val_indices.shape[0]}")
    print(f"[Go2-OfflineRWM-Proprioceptive] batch_size={int(cfg.batch_size)}, micro_batch_size={int(cfg.micro_batch_size)}")

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
            print(f"[Go2-OfflineRWM-Proprioceptive] iter={iteration} {interesting}")

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
    print(f"[Go2-OfflineRWM-Proprioceptive] saved final checkpoint: {save_root / f'model_{int(cfg.max_iterations)}.pt'}")


if __name__ == "__main__":
    main()
