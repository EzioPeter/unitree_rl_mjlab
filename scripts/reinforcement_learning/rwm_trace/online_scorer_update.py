"""In-process cumulative Bradley-Terry refresh used by online Go2 TRACE."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F

from .feedback_manager import CumulativeFeedbackManager
from .schemas import canonical_sha256
from .scorer import (
    Go2TraceScorer,
    ScorerBinding,
    fit_feature_stats,
    load_scorer_checkpoint,
    raw_feature_matrix,
    save_scorer_checkpoint,
    transform_features,
)


def decayed_feedback_budget(
    initial: int,
    minimum: int,
    decay: float,
    refresh_index: int,
) -> int:
    """D4RL TRACE query budget: floor exponential decay, clamped to a minimum."""

    if int(initial) < 1 or int(minimum) < 1 or int(minimum) > int(initial):
        raise ValueError("TRACE feedback budget bounds are invalid.")
    if not 0.0 < float(decay) <= 1.0:
        raise ValueError("TRACE feedback budget decay must be in (0,1].")
    if int(refresh_index) < 0:
        raise ValueError("TRACE feedback refresh index must be non-negative.")
    planned = int(
        np.floor(int(initial) * (float(decay) ** int(refresh_index)))
    )
    return max(int(minimum), planned)


def _group(summary: Mapping[str, Any]) -> str:
    return str(
        summary.get(
            "comparison_group_key",
            summary.get("start_state_key", summary.get("start_state_id", "")),
        )
    )


def _components(labels: Sequence[Mapping[str, Any]]) -> list[str]:
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

    nodes = []
    for row in labels:
        left = row["trajectory_i"]
        right = row["trajectory_j"]
        left_t = f"t:{left['trajectory_id']}"
        right_t = f"t:{right['trajectory_id']}"
        union(left_t, f"g:{_group(left)}")
        union(right_t, f"g:{_group(right)}")
        union(left_t, right_t)
        nodes.append(left_t)
    return [find(node) for node in nodes]


class OnlineScorerUpdater:
    def __init__(
        self,
        *,
        feedback: CumulativeFeedbackManager,
        feedback_budget_initial: int,
        feedback_budget_min: int,
        feedback_budget_decay: float,
        cross_region_fraction: float,
        task_id: str,
        dataset_id: str,
        dataset_sha256: str,
        condition_id: str,
        checkpoint_directory: str | Path,
        epochs: int,
        learning_rate: float,
        validation_ratio: float,
        seed: int,
        cpu_threads: int = 4,
    ) -> None:
        self.feedback = feedback
        self.feedback_budget_initial = int(feedback_budget_initial)
        self.feedback_budget_min = int(feedback_budget_min)
        self.feedback_budget_decay = float(feedback_budget_decay)
        decayed_feedback_budget(
            self.feedback_budget_initial,
            self.feedback_budget_min,
            self.feedback_budget_decay,
            0,
        )
        self.cross_region_fraction = float(cross_region_fraction)
        if not 0.0 <= self.cross_region_fraction <= 1.0:
            raise ValueError("cross_region_fraction must be in [0,1].")
        self.task_id = str(task_id)
        self.dataset_id = str(dataset_id)
        self.dataset_sha256 = str(dataset_sha256)
        self.condition_id = str(condition_id)
        self.directory = Path(checkpoint_directory).expanduser().resolve()
        self.epochs = int(epochs)
        self.learning_rate = float(learning_rate)
        self.validation_ratio = float(validation_ratio)
        self.seed = int(seed)
        self.cpu_threads = int(cpu_threads)
        if self.epochs < 1 or self.learning_rate <= 0.0:
            raise ValueError("Online scorer optimizer configuration is invalid.")
        if not 0.0 <= self.validation_ratio < 1.0:
            raise ValueError("Online scorer validation ratio must be in [0,1).")

    def __call__(
        self,
        summaries: Sequence[Mapping[str, Any]],
        _trajectories: Sequence[Mapping[str, Any]],
        proposal_event: int,
    ):
        feedback_budget = decayed_feedback_budget(
            self.feedback_budget_initial,
            self.feedback_budget_min,
            self.feedback_budget_decay,
            proposal_event,
        )
        self.feedback.request(
            summaries,
            feedback_budget=feedback_budget,
            cross_region_fraction=self.cross_region_fraction,
        )
        labels = self.feedback.cumulative_training_labels()
        if len(labels) < 2:
            raise ValueError("Cumulative feedback has fewer than two trainable labels.")
        components = _components(labels)
        unique = sorted(set(components))
        rng = np.random.default_rng(self.seed + int(proposal_event))
        rng.shuffle(unique)
        val_count = int(round(len(unique) * self.validation_ratio)) if len(unique) > 1 else 0
        val_groups = set(unique[:val_count])
        train_index = [index for index, key in enumerate(components) if key not in val_groups]
        val_index = [index for index, key in enumerate(components) if key in val_groups]
        if not train_index:
            raise ValueError("Online scorer component split has no training pairs.")

        trajectory_summaries = [
            summary
            for row in labels
            for summary in (row["trajectory_i"], row["trajectory_j"])
        ]
        raw = raw_feature_matrix(trajectory_summaries)
        train_rows = [
            row for index in train_index for row in (2 * index, 2 * index + 1)
        ]
        stats = fit_feature_stats(raw[train_rows])
        features = torch.from_numpy(transform_features(raw, stats))
        left, right = features[0::2], features[1::2]
        target = torch.tensor(
            [1.0 if row["feedback"] == "i" else 0.0 for row in labels]
        )
        train_t = torch.tensor(train_index, dtype=torch.long)
        val_t = torch.tensor(val_index, dtype=torch.long)
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(max(1, self.cpu_threads))
        try:
            torch.manual_seed(self.seed + int(proposal_event))
            model = Go2TraceScorer(features.shape[1], 256)
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
            loss_value = float("nan")
            for _ in range(self.epochs):
                logits = model(left[train_t]) - model(right[train_t])
                loss = F.binary_cross_entropy_with_logits(logits, target[train_t])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                loss_value = float(loss.detach())
        finally:
            torch.set_num_threads(previous_threads)
        if not np.isfinite(loss_value):
            raise ValueError("Online scorer training produced a non-finite loss.")
        with torch.no_grad():
            prediction = model(left) > model(right)
            train_accuracy = float(
                (prediction[train_t] == target[train_t].bool()).float().mean()
            )
            val_accuracy = (
                float((prediction[val_t] == target[val_t].bool()).float().mean())
                if val_index
                else float("nan")
            )
        split = {
            "proposal_event": int(proposal_event),
            "seed": self.seed + int(proposal_event),
            "train_pair_ids": [labels[index]["pair_id"] for index in train_index],
            "val_pair_ids": [labels[index]["pair_id"] for index in val_index],
        }
        binding = ScorerBinding(
            task_id=self.task_id,
            dataset_id=self.dataset_id,
            dataset_sha256=self.dataset_sha256,
            condition_id=self.condition_id,
            labels_sha256=self.feedback.labels_sha256,
            split_sha256=canonical_sha256(split),
        )
        metrics = {
            "feedback_budget": feedback_budget,
            "feedback_budget_initial": self.feedback_budget_initial,
            "feedback_budget_min": self.feedback_budget_min,
            "feedback_budget_decay": self.feedback_budget_decay,
            "loss": loss_value,
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "train_pairs": len(train_index),
            "val_pairs": len(val_index),
            "split": split,
        }
        self.directory.mkdir(parents=True, exist_ok=True)
        checkpoint = self.directory / f"scorer_refresh_{proposal_event:08d}.pt"
        save_scorer_checkpoint(
            checkpoint,
            model,
            stats,
            binding=binding,
            training_metrics=metrics,
        )
        loaded, loaded_stats, loaded_binding, loaded_metrics = load_scorer_checkpoint(
            checkpoint, expected_binding=binding
        )
        return loaded, loaded_stats, loaded_binding, {
            **loaded_metrics,
            "checkpoint": str(checkpoint),
            "binding_sha256": binding.sha256,
            "llm_provider": dict(
                getattr(
                    self.feedback.label_provider,
                    "last_request_metrics",
                    {},
                )
            ),
            "pair_sampling": dict(self.feedback.last_pair_metrics),
        }
