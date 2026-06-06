"""Merge Go2 RWM dataset part files into one legacy-compatible dataset.pt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_dataset.dataset import merge_dataset_dicts, save_dataset_dict


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parts_dir", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--pattern", default="dataset_part_*.pt")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    parts_dir = Path(args.parts_dir).expanduser()
    output_path = Path(args.output_path).expanduser()
    part_paths = sorted(parts_dir.glob(args.pattern))
    if not part_paths:
        raise FileNotFoundError(f"No parts matching {args.pattern!r} under {parts_dir}")
    parts = [torch.load(path, map_location="cpu", weights_only=False) for path in part_paths]
    merged = merge_dataset_dicts(parts)
    save_dataset_dict(merged, output_path)
    print(f"[Go2-MixedDataset] merged {len(part_paths)} parts -> {output_path}")
    print(f"  transitions: {merged['metadata']['num_transitions']}")
    print(f"  file_size_mb: {output_path.stat().st_size / (1024 * 1024):.2f}")


if __name__ == "__main__":
    main()
