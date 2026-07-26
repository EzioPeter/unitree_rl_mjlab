#!/usr/bin/env python3
"""Build fixed-quota Go2 TRACE feedback pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.feedback_pairs import (
    PairQuota,
    build_feedback_pairs,
    build_global_feedback_pairs,
)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--summaries", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--heldout_output")
    parser.add_argument("--train_count", type=int)
    parser.add_argument("--heldout_summary_ratio", type=float, default=0.2)
    parser.add_argument("--sampling_mode", choices=("global_random", "region_quota"), default="region_quota")
    parser.add_argument("--pair_count", type=int)
    parser.add_argument("--within_region_count", type=int)
    parser.add_argument("--cross_region_count", type=int)
    parser.add_argument("--pair_prefix", default="go2_pair")
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument(
        "--planar_command_scales",
        required=True,
        type=float,
        nargs=2,
    )
    return parser.parse_args()


def main() -> None:
    args = _args()
    with open(args.summaries, "r", encoding="utf-8") as handle:
        summaries = [json.loads(line) for line in handle if line.strip()]
    if (args.heldout_output is None) != (args.train_count is None):
        raise ValueError("--heldout_output and --train_count must be provided together.")
    if args.sampling_mode == "global_random":
        if args.pair_count is None or args.pair_count < 1:
            raise ValueError("--pair_count is required for global_random.")
        rows = build_global_feedback_pairs(
            summaries,
            pair_count=args.pair_count,
            pair_prefix=args.pair_prefix,
            seed=args.seed,
            planar_command_scales=args.planar_command_scales,
        )
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
        print(json.dumps({"output": str(output), "pair_count": len(rows), "sampling_mode": "global_random"}, indent=2))
        return
    if args.within_region_count is None or args.cross_region_count is None:
        raise ValueError("region_quota requires within/cross counts.")
    total_count = args.within_region_count + args.cross_region_count
    if args.train_count is not None and not 0 < args.train_count < total_count:
        raise ValueError("--train_count must be between zero and the total pair count.")
    if args.train_count is None:
        train_rows = build_feedback_pairs(
            summaries,
            quota=PairQuota(
                args.within_region_count,
                args.cross_region_count,
            ),
            pair_prefix=args.pair_prefix,
            seed=args.seed,
            planar_command_scales=args.planar_command_scales,
        )
        heldout_rows = []
    else:
        if not 0.0 < args.heldout_summary_ratio < 1.0:
            raise ValueError("--heldout_summary_ratio must be in (0,1).")
        groups = sorted(
            {
                str(
                    row.get(
                        "comparison_group_key",
                        row.get("start_state_key", row.get("start_state_id", "")),
                    )
                )
                for row in summaries
            },
            key=lambda group: hashlib.sha256(
                f"{args.seed}\0{group}".encode("utf-8")
            ).hexdigest(),
        )
        heldout_group_count = max(
            1, min(len(groups) - 1, int(round(len(groups) * args.heldout_summary_ratio)))
        )
        heldout_groups = set(groups[:heldout_group_count])
        train_summaries = [
            row
            for row in summaries
            if str(
                row.get(
                    "comparison_group_key",
                    row.get("start_state_key", row.get("start_state_id", "")),
                )
            )
            not in heldout_groups
        ]
        heldout_summaries = [
            row
            for row in summaries
            if str(
                row.get(
                    "comparison_group_key",
                    row.get("start_state_key", row.get("start_state_id", "")),
                )
            )
            in heldout_groups
        ]
        heldout_count = total_count - args.train_count
        train_cross = int(
            round(
                args.train_count
                * args.cross_region_count
                / total_count
            )
        )
        train_quota = PairQuota(
            within_region=args.train_count - train_cross,
            cross_region=train_cross,
        )
        heldout_quota = PairQuota(
            within_region=args.within_region_count - train_quota.within_region,
            cross_region=args.cross_region_count - train_quota.cross_region,
        )
        if (
            heldout_quota.within_region + heldout_quota.cross_region
            != heldout_count
        ):
            raise RuntimeError("Offline train/heldout region quotas do not add up.")
        train_rows = build_feedback_pairs(
            train_summaries,
            quota=train_quota,
            pair_prefix=f"{args.pair_prefix}_train",
            seed=args.seed,
            planar_command_scales=args.planar_command_scales,
        )
        heldout_rows = build_feedback_pairs(
            heldout_summaries,
            quota=heldout_quota,
            pair_prefix=f"{args.pair_prefix}_heldout",
            seed=args.seed + 1,
            planar_command_scales=args.planar_command_scales,
        )
    output = Path(args.output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for row in train_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    heldout_output = None
    if args.heldout_output is not None:
        heldout_output = Path(args.heldout_output).expanduser().resolve()
        heldout_output.parent.mkdir(parents=True, exist_ok=True)
        with heldout_output.open("w", encoding="utf-8") as handle:
            for row in heldout_rows:
                handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "output": str(output),
                "pair_count": len(train_rows),
                "heldout_output": (
                    str(heldout_output) if heldout_output is not None else None
                ),
                "heldout_pair_count": len(heldout_rows),
                "total_pair_count": len(train_rows) + len(heldout_rows),
                "within_region_quota": args.within_region_count,
                "cross_region_quota": args.cross_region_count,
                "scorer_used_for_pairing": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
