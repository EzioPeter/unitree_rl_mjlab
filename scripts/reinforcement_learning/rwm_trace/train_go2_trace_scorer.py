#!/usr/bin/env python3
"""Train one task/dataset-specific Go2 TRACE Bradley-Terry scorer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.artifact_manifest import sha256_path
from scripts.reinforcement_learning.rwm_trace.schemas import canonical_sha256
from scripts.reinforcement_learning.rwm_trace.scorer import (
    Go2TraceScorer,
    ScorerBinding,
    fit_feature_stats,
    raw_feature_matrix,
    save_scorer_checkpoint,
    transform_features,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task_id", required=True)
    parser.add_argument("--dataset_id", required=True)
    parser.add_argument("--condition_id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--learning_rate", type=float, default=1.0e-3)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--allow_no_validation", action="store_true")
    return parser.parse_args()


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _group(summary: dict) -> str:
    return str(
        summary.get(
            "comparison_group_key",
            summary.get("start_state_key", summary.get("start_state_id", "")),
        )
    )


def _component_groups(labels: list[dict]) -> list[str]:
    parent: dict[str, str] = {}

    def find(node: str) -> str:
        parent.setdefault(node, node)
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        a, b = find(left), find(right)
        if a != b:
            parent[max(a, b)] = min(a, b)

    pair_nodes: list[str] = []
    for row in labels:
        left, right = row["trajectory_i"], row["trajectory_j"]
        left_group = f"group:{_group(left)}"
        right_group = f"group:{_group(right)}"
        left_trajectory = f"trajectory:{left['trajectory_id']}"
        right_trajectory = f"trajectory:{right['trajectory_id']}"
        union(left_group, left_trajectory)
        union(right_group, right_trajectory)
        union(left_trajectory, right_trajectory)
        pair_nodes.append(left_trajectory)
    return [find(node) for node in pair_nodes]


def _balanced_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    values = []
    for label in (False, True):
        mask = target.bool() == label
        if bool(mask.any()):
            values.append(float((prediction[mask] == target.bool()[mask]).float().mean()))
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = _args()
    if not 0.0 <= args.val_ratio < 1.0:
        raise ValueError("--val_ratio must be in [0,1).")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    pair_by_id = {str(row["pair_id"]): row for row in _read_jsonl(args.pairs)}
    labels = []
    for label in _read_jsonl(args.labels):
        pair_id = str(label.get("pair_id", ""))
        if (
            pair_id in pair_by_id
            and label.get("feedback") in {"i", "j"}
            and float(label.get("confidence", 0.0)) >= args.confidence_threshold
        ):
            labels.append({**pair_by_id[pair_id], **label})
    if len(labels) < 2:
        raise ValueError("Need at least two confidence-filtered non-tie labels.")

    summaries = [
        summary
        for row in labels
        for summary in (row["trajectory_i"], row["trajectory_j"])
    ]
    pair_groups = _component_groups(labels)
    unique_groups = sorted(set(pair_groups))
    rng = np.random.default_rng(args.seed)
    rng.shuffle(unique_groups)
    val_count = int(round(len(unique_groups) * args.val_ratio)) if len(unique_groups) > 1 else 0
    val_groups = set(unique_groups[:val_count])
    train_indices = [i for i, group in enumerate(pair_groups) if group not in val_groups]
    val_indices = [i for i, group in enumerate(pair_groups) if group in val_groups]
    if not train_indices:
        raise ValueError("Split produced no training pairs.")
    if not val_indices and not args.allow_no_validation:
        raise ValueError("Split produced no validation pairs.")

    raw = raw_feature_matrix(summaries)
    train_rows = [row for index in train_indices for row in (2 * index, 2 * index + 1)]
    stats = fit_feature_stats(raw[train_rows])
    features = torch.from_numpy(transform_features(raw, stats))
    x_i, x_j = features[0::2], features[1::2]
    targets = torch.tensor(
        [1.0 if row["feedback"] == "i" else 0.0 for row in labels],
        dtype=torch.float32,
    )
    train_index = torch.tensor(train_indices, dtype=torch.long)
    val_index = torch.tensor(val_indices, dtype=torch.long)

    model = Go2TraceScorer(features.shape[1], 256)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    final_loss = float("nan")
    for _ in range(args.epochs):
        logits = model(x_i[train_index]) - model(x_j[train_index])
        loss = F.binary_cross_entropy_with_logits(logits, targets[train_index])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())

    with torch.no_grad():
        scalar_scores = model(features).detach().cpu().numpy()
        predictions = model(x_i) > model(x_j)
        train_accuracy = float(
            (predictions[train_index] == targets[train_index].bool()).float().mean()
        )
        val_accuracy = (
            float((predictions[val_index] == targets[val_index].bool()).float().mean())
            if val_indices
            else float("nan")
        )
        train_balanced = _balanced_accuracy(predictions[train_index], targets[train_index])
        val_balanced = (
            _balanced_accuracy(predictions[val_index], targets[val_index])
            if val_indices
            else float("nan")
        )

    def correlation(left: np.ndarray, right: np.ndarray) -> float:
        mask = np.isfinite(left) & np.isfinite(right)
        if mask.sum() < 2 or np.std(left[mask]) < 1.0e-12 or np.std(right[mask]) < 1.0e-12:
            return float("nan")
        return float(np.corrcoef(left[mask], right[mask])[0, 1])

    returns = np.asarray(
        [float(summary.get("simulator_return", np.nan)) for summary in summaries]
    )
    tracking = np.asarray(
        [
            np.nanmean(
                [
                    float(summary.get(f"tracking_{axis}_normalized_mae_full", np.nan))
                    for axis in ("vx", "vy", "yaw")
                    if bool(summary.get("command_active", {}).get(axis, False))
                ]
            )
            if any(summary.get("command_active", {}).values())
            else np.nan
            for summary in summaries
        ]
    )
    posture = np.asarray(
        [float(summary.get("tilt_max", np.nan)) for summary in summaries]
    )
    top_count = max(1, int(np.ceil(0.25 * len(summaries))))
    score_top = set(np.argsort(-scalar_scores)[:top_count].tolist())
    return_top = set(np.argsort(-returns)[:top_count].tolist())

    split = {
        "seed": args.seed,
        "train_pair_ids": [labels[i]["pair_id"] for i in train_indices],
        "val_pair_ids": [labels[i]["pair_id"] for i in val_indices],
    }
    binding = ScorerBinding(
        task_id=args.task_id,
        dataset_id=args.dataset_id,
        dataset_sha256=sha256_path(args.dataset),
        condition_id=args.condition_id,
        labels_sha256=sha256_path(args.labels),
        split_sha256=canonical_sha256(split),
    )
    metrics = {
        "final_loss": final_loss,
        "train_accuracy": train_accuracy,
        "val_accuracy": val_accuracy,
        "train_balanced_accuracy": train_balanced,
        "val_balanced_accuracy": val_balanced,
        "num_pairs": len(labels),
        "num_train_pairs": len(train_indices),
        "num_val_pairs": len(val_indices),
        "split": split,
        "score_return_correlation": correlation(scalar_scores, returns),
        "score_tracking_error_correlation": correlation(scalar_scores, tracking),
        "score_posture_tilt_correlation": correlation(scalar_scores, posture),
        "score_return_top_alpha_overlap": float(
            len(score_top & return_top) / top_count
        ),
    }
    save_scorer_checkpoint(
        args.output,
        model,
        stats,
        binding=binding,
        training_metrics=metrics,
    )
    print(json.dumps({**metrics, "binding": binding.to_dict()}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
