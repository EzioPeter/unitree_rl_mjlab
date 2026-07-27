#!/usr/bin/env python3
"""Build fixed-quota Go2 TRACE feedback pairs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.feedback_pairs import (
    PairQuota,
    build_feedback_pairs,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--within_region_count", required=True, type=int)
    parser.add_argument("--cross_region_count", required=True, type=int)
    parser.add_argument("--pair_prefix", default="go2_pair")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--planar_command_scales",
        default=(0.5, 0.2),
        type=float,
        nargs=2,
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    with open(args.summaries, "r", encoding="utf-8") as handle:
        summaries = [json.loads(line) for line in handle if line.strip()]
    rows = build_feedback_pairs(
        summaries,
        quota=PairQuota(args.within_region_count, args.cross_region_count),
        pair_prefix=args.pair_prefix,
        seed=args.seed,
        planar_command_scales=args.planar_command_scales,
    )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "pair_count": len(rows),
                "within_region_quota": args.within_region_count,
                "cross_region_quota": args.cross_region_count,
                "scorer_used_for_pairing": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
