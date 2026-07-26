#!/usr/bin/env python3
"""Resumable Codex labeling with per-pair salvage and targeted repair."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_trace.go2_feedback_prompt import validate_label


DEFAULT_SCHEMA = Path(__file__).with_name("feedback_label_batch.schema.json")


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--pairs", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--schema", default=str(DEFAULT_SCHEMA))
    parser.add_argument("--repo_root", default=str(REPO_ROOT))
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--max_repair_rounds", type=int, default=2)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--timeout_sec", type=float, default=240.0)
    parser.add_argument("--codex_model", default=None)
    parser.add_argument(
        "--codex_reasoning_effort",
        choices=("low", "medium", "high", "xhigh"),
        default=None,
    )
    parser.add_argument("--provider_workers", type=int, default=16)
    parser.add_argument("--max_retries", type=int, default=2)
    parser.add_argument("--retry_backoff_sec", type=float, default=2.0)
    return parser.parse_args()


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")


def build_batch_prompt(rows: list[Mapping[str, Any]]) -> str:
    return (
        "Label every supplied Go2 TRACE pair independently. Each embedded prompt "
        "contains the complete task criteria. Return strict JSON matching the output "
        "schema. Do not omit a pair and do not invent pair IDs.\n"
        + json.dumps(
            {
                "pairs": [
                    {"pair_id": row["pair_id"], "prompt": row["prompt"]}
                    for row in rows
                ]
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )


def salvage_label_response(
    response: Mapping[str, Any],
    expected_pair_ids: Iterable[str],
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    """Keep every individually valid label; report only missing/invalid IDs."""

    expected = set(map(str, expected_pair_ids))
    valid: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    labels = response.get("labels", [])
    if not isinstance(labels, list):
        return {}, {pair_id: "response.labels is not a list" for pair_id in expected}
    for item in labels:
        if not isinstance(item, Mapping):
            continue
        pair_id = str(item.get("pair_id", ""))
        if pair_id not in expected or pair_id in valid:
            continue
        try:
            valid[pair_id] = validate_label(item, expected_pair_id=pair_id)
        except Exception as exc:
            failures[pair_id] = str(exc)
    for pair_id in expected - set(valid):
        failures.setdefault(pair_id, "label missing from response")
    return valid, failures


def _call_codex(
    rows: list[Mapping[str, Any]],
    *,
    response_path: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    command = [
        "codex",
        "exec",
        "--cd",
        str(Path(args.repo_root).resolve()),
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--output-schema",
        str(Path(args.schema).resolve()),
        "-o",
        str(response_path),
        "-",
    ]
    if args.codex_model:
        command[2:2] = ["--model", str(args.codex_model)]
    if args.codex_reasoning_effort:
        command[2:2] = [
            "--config",
            f'model_reasoning_effort="{args.codex_reasoning_effort}"',
        ]
    try:
        result = subprocess.run(
            command,
            input=build_batch_prompt(rows),
            text=True,
            capture_output=True,
            timeout=float(args.timeout_sec),
        )
        log_path.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
        if result.returncode != 0 or not response_path.exists():
            return None
        return json.loads(response_path.read_text(encoding="utf-8"))
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        log_path.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
        return None


def _call_codex_with_retries(
    rows: list[Mapping[str, Any]],
    *,
    response_path: Path,
    log_path: Path,
    args: argparse.Namespace,
) -> Mapping[str, Any] | None:
    for attempt in range(args.max_retries + 1):
        attempt_response = response_path.with_name(
            f"{response_path.stem}_try{attempt:02d}{response_path.suffix}"
        )
        attempt_log = log_path.with_name(
            f"{log_path.stem}_try{attempt:02d}{log_path.suffix}"
        )
        if attempt_response.is_file():
            try:
                cached = json.loads(
                    attempt_response.read_text(encoding="utf-8")
                )
                if isinstance(cached, Mapping):
                    return cached
            except json.JSONDecodeError:
                attempt_response.unlink(missing_ok=True)
        response = _call_codex(
            rows,
            response_path=attempt_response,
            log_path=attempt_log,
            args=args,
        )
        if response is not None:
            return response
        if attempt < args.max_retries and args.retry_backoff_sec:
            time.sleep(args.retry_backoff_sec * (2**attempt))
    return None


def main() -> None:
    args = _args()
    if (
        args.batch_size < 1
        or args.max_repair_rounds < 0
        or args.provider_workers < 1
        or args.max_retries < 0
        or args.retry_backoff_sec < 0.0
    ):
        raise ValueError("Batch/concurrency/retry settings are invalid.")
    pairs = _read_jsonl(args.pairs)
    pair_by_id = {str(row["pair_id"]): row for row in pairs}
    if len(pair_by_id) != len(pairs):
        raise ValueError("Pair file contains duplicate pair IDs.")
    output = Path(args.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    accepted_path = output / "labels_salvaged.jsonl"
    accepted: dict[str, dict[str, Any]] = {}
    if accepted_path.exists():
        for row in _read_jsonl(accepted_path):
            pair_id = str(row.get("pair_id", ""))
            if pair_id in pair_by_id:
                accepted[pair_id] = validate_label(row, expected_pair_id=pair_id)
    pending = [pair_id for pair_id in pair_by_id if pair_id not in accepted]

    if shutil.which("codex") is None:
        _write_jsonl(output / "manual_label_todo.jsonl", (pair_by_id[key] for key in pending))
        print("codex executable unavailable; wrote manual_label_todo.jsonl")
        return

    failures: dict[str, str] = {}
    call_index = 0
    for repair_round in range(args.max_repair_rounds + 1):
        if not pending:
            break
        next_pending: list[str] = []
        jobs = []
        for start in range(0, len(pending), args.batch_size):
            ids = pending[start : start + args.batch_size]
            jobs.append(
                (
                    ids,
                    output / f"response_{call_index:05d}.json",
                    output / f"response_{call_index:05d}.log",
                )
            )
            call_index += 1

        def run_job(
            job: tuple[list[str], Path, Path],
        ) -> tuple[list[str], Mapping[str, Any] | None]:
            ids, response_path, log_path = job
            return ids, _call_codex_with_retries(
                [pair_by_id[pair_id] for pair_id in ids],
                response_path=response_path,
                log_path=log_path,
                args=args,
            )

        with ThreadPoolExecutor(
            max_workers=min(args.provider_workers, len(jobs)),
            thread_name_prefix="go2-trace-label",
        ) as executor:
            results = list(executor.map(run_job, jobs))
        for ids, response in results:
            if response is None:
                next_pending.extend(ids)
                failures.update({pair_id: "Codex call or JSON parse failed" for pair_id in ids})
                continue
            valid, invalid = salvage_label_response(response, ids)
            accepted.update(valid)
            failures.update(invalid)
            next_pending.extend(pair_id for pair_id in ids if pair_id not in valid)
            _write_jsonl(
                accepted_path,
                (accepted[key] for key in sorted(accepted)),
            )
        pending = next_pending

    enriched = [
        {**pair_by_id[pair_id], **accepted[pair_id]}
        for pair_id in sorted(accepted)
    ]
    filtered = [
        row
        for row in enriched
        if row["feedback"] in {"i", "j"}
        and float(row["confidence"]) >= args.confidence_threshold
    ]
    _write_jsonl(output / "labels_raw_enriched.jsonl", enriched)
    _write_jsonl(output / "labels_filtered.jsonl", filtered)
    _write_jsonl(output / "manual_label_todo.jsonl", (pair_by_id[key] for key in pending))
    (output / "labeling_report.json").write_text(
        json.dumps(
            {
                "pair_count": len(pairs),
                "valid_label_count": len(enriched),
                "training_label_count": len(filtered),
                "pending_pair_ids": pending,
                "last_failures": {key: failures[key] for key in pending if key in failures},
                "partial_batch_salvage": True,
                "targeted_repair_rounds": args.max_repair_rounds,
                "provider_workers": args.provider_workers,
                "max_retries": args.max_retries,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"accepted={len(enriched)} filtered={len(filtered)} pending={len(pending)}")


if __name__ == "__main__":
    main()
