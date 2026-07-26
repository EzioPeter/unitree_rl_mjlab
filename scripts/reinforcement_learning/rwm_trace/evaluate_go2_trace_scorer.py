#!/usr/bin/env python3
"""Evaluate one Go2 TRACE scorer on component-disjoint heldout labels."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.scorer import (
    load_scorer_checkpoint,
    raw_feature_matrix,
    transform_features,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    return parser.parse_args()


def _read_jsonl(path: str | Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _balanced_accuracy(prediction: torch.Tensor, target: torch.Tensor) -> float:
    values = []
    for label in (False, True):
        mask = target.bool() == label
        if bool(mask.any()):
            values.append(
                float((prediction[mask] == target.bool()[mask]).float().mean())
            )
    return float(np.mean(values)) if values else float("nan")


def main() -> None:
    args = _args()
    pair_by_id = {
        str(row["pair_id"]): row for row in _read_jsonl(args.pairs)
    }
    labels = []
    for label in _read_jsonl(args.labels):
        pair_id = str(label.get("pair_id", ""))
        if (
            pair_id in pair_by_id
            and label.get("feedback") in {"i", "j"}
            and float(label.get("confidence", 0.0)) >= args.confidence_threshold
        ):
            labels.append({**pair_by_id[pair_id], **label})
    if not labels:
        raise ValueError("Heldout set has no confidence-filtered i/j labels.")
    model, stats, binding, training_metrics = load_scorer_checkpoint(
        args.checkpoint
    )
    summaries = [
        summary
        for row in labels
        for summary in (row["trajectory_i"], row["trajectory_j"])
    ]
    features = torch.from_numpy(
        transform_features(raw_feature_matrix(summaries), stats)
    )
    with torch.no_grad():
        scores = model(features)
        prediction = scores[0::2] > scores[1::2]
    target = torch.tensor(
        [row["feedback"] == "i" for row in labels],
        dtype=torch.bool,
    )
    result = {
        "heldout_label_count": len(labels),
        "accuracy": float((prediction == target).float().mean()),
        "balanced_accuracy": _balanced_accuracy(prediction, target),
        "feedback_i_count": int(target.sum()),
        "feedback_j_count": int((~target).sum()),
        "score_finite": bool(torch.isfinite(scores).all()),
        "binding": binding.to_dict(),
        "binding_sha256": binding.sha256,
        "training_metrics": training_metrics,
    }
    if not result["score_finite"]:
        raise ValueError("Heldout scorer outputs contain NaN/Inf.")
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
