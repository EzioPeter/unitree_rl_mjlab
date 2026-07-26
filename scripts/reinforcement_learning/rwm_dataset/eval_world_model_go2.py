"""Evaluate Go2 RWM-U checkpoints on held-out mixed dataset sequences."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm.dynamics import load_dynamics_checkpoint
from scripts.reinforcement_learning.rwm_dataset.dataset import (
    OfflineSamplerConfig,
    OfflineSequenceSampler,
    load_mixed_dataset,
)
from scripts.reinforcement_learning.rwm_dataset.metrics import autoregressive_rollout_metrics
from scripts.reinforcement_learning.rwm_flashsac.utils import configure_low_thread_env, resolve_repo_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--dataset_path", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--num_eval_sequences", type=int, default=4096)
    parser.add_argument("--rollout_horizon", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--train_fraction", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--compare_model_paths", nargs="*", default=[])
    return parser.parse_args()


def _evaluate_one(
    *,
    model_path: Path,
    sampler: OfflineSequenceSampler,
    num_eval_sequences: int,
    rollout_horizon: int,
    device: torch.device,
) -> dict[str, float | str]:
    dynamics, _checkpoint = load_dynamics_checkpoint(model_path, device=device)
    dynamics.eval()
    forecast = max(int(rollout_horizon), int(dynamics.cfg.forecast_horizon))
    batch = sampler.sample(
        batch_size=int(num_eval_sequences),
        device=device,
        split="val",
        forecast_horizon=forecast,
    )
    with torch.no_grad():
        loss_dict = dynamics.compute_loss(*batch, bootstrap=False)
        metrics = autoregressive_rollout_metrics(
            dynamics,
            batch,
            rollout_horizon=int(rollout_horizon),
            horizons=(1, 8, 32, int(rollout_horizon)),
        )
    out: dict[str, float | str] = {
        "model_path": str(model_path),
        "eval_state_loss": float(loss_dict["state_loss"].detach().cpu()),
        "eval_sequence_loss": float(loss_dict["sequence_loss"].detach().cpu()),
        "eval_contact_loss": float(loss_dict["contact_loss"].detach().cpu()),
        "eval_termination_loss": float(loss_dict["termination_loss"].detach().cpu()),
    }
    out.update(metrics)
    return out


def _save_outputs(metrics: list[dict[str, float | str]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "eval_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    csv_path = output_dir / "autoregressive_error.csv"
    horizon_keys = sorted(
        {key for row in metrics for key in row if key.endswith("_step_mse")},
        key=lambda key: int(key.split("_", maxsplit=1)[0]),
    )
    fieldnames = [
        "model_path",
        *horizon_keys,
        "traj_autoregressive_error",
        "epistemic_uncertainty_mean",
        "aleatoric_uncertainty_mean",
        "epistemic_error_correlation",
        "contact_accuracy",
        "termination_accuracy",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"[Go2-RWM-Eval] wrote {metrics_path}")
    print(f"[Go2-RWM-Eval] wrote {csv_path}")


def main() -> None:
    configure_low_thread_env()
    args = _parse_args()
    device = torch.device(args.device if torch.cuda.is_available() or not str(args.device).startswith("cuda") else "cpu")
    dataset_path = resolve_repo_path(args.dataset_path)
    model_path = resolve_repo_path(args.model_path)
    dataset = load_mixed_dataset(dataset_path)

    # Use the target model's horizons to build valid windows.
    target_dynamics, _ = load_dynamics_checkpoint(model_path, device=device)
    available_rollout = len(dataset["states"]) - int(target_dynamics.cfg.history_horizon)
    if available_rollout <= 0:
        raise ValueError(
            f"Dataset has only {len(dataset['states'])} time steps; "
            f"need more than history horizon {target_dynamics.cfg.history_horizon}."
        )
    effective_rollout_horizon = min(int(args.rollout_horizon), int(available_rollout))
    sampler = OfflineSequenceSampler(
        dataset,
        OfflineSamplerConfig(
            history_horizon=int(target_dynamics.cfg.history_horizon),
            forecast_horizon=max(effective_rollout_horizon, int(target_dynamics.cfg.forecast_horizon)),
            train_fraction=float(args.train_fraction),
            seed=int(args.seed),
        ),
    )

    all_paths = [model_path] + [resolve_repo_path(path) for path in args.compare_model_paths]
    metrics = []
    for path in all_paths:
        row = _evaluate_one(
            model_path=path,
            sampler=sampler,
            num_eval_sequences=int(args.num_eval_sequences),
            rollout_horizon=effective_rollout_horizon,
            device=device,
        )
        metrics.append(row)
        print(f"[Go2-RWM-Eval] model={path}")
        for key, value in row.items():
            if key != "model_path":
                print(f"  {key}: {value}")

    output_dir = model_path.parent
    _save_outputs(metrics, output_dir)


if __name__ == "__main__":
    main()
