"""Minimal loader for Go2 RWM model-based policy checkpoints."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    checkpoint = torch.load(Path(args.checkpoint).expanduser(), map_location="cpu", weights_only=False)
    print(f"Loaded policy checkpoint: {args.checkpoint}")
    print(f"Iteration: {checkpoint.get('iter')}")
    print("Keys:", sorted(checkpoint.keys()))
    print("This checkpoint is intended for RWM imagination/RSL-RL loading; real-env playback can be added once the policy quality is validated.")


if __name__ == "__main__":
    main()
