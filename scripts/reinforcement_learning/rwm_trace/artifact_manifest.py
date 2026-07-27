#!/usr/bin/env python3
"""Create and verify immutable TRACE input manifests."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


def sha256_path(path: str | Path) -> str:
    path = Path(path).resolve()
    digest = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    if not path.is_dir():
        raise FileNotFoundError(path)
    files = sorted(item for item in path.rglob("*") if item.is_file())
    if not files:
        raise ValueError(f"Cannot hash empty directory: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode()
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def build_manifest(entries: dict[str, str | Path]) -> dict:
    result = {"format_version": "go2_trace_manifest_v1", "artifacts": {}}
    for name, raw_path in sorted(entries.items()):
        path = Path(raw_path).resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        result["artifacts"][name] = {
            "path": str(path),
            "kind": "directory" if path.is_dir() else "file",
            "sha256": sha256_path(path),
        }
    return result


def verify_manifest(manifest: dict) -> None:
    if manifest.get("format_version") not in {
        "go2_trace_manifest_v1",
        "go2_trace_stage_b_manifest_v1",
    }:
        raise ValueError("Unsupported TRACE manifest format.")
    errors = []
    for name, row in sorted((manifest.get("artifacts") or {}).items()):
        path = Path(row["path"])
        if not path.exists():
            errors.append(f"{name}: missing {path}")
            continue
        actual = sha256_path(path)
        if actual != row.get("sha256"):
            errors.append(f"{name}: sha256 {actual} != {row.get('sha256')}")
    if errors:
        raise RuntimeError("TRACE manifest verification failed: " + "; ".join(errors))


def build_stage_b_manifest(
    *,
    task_id: str,
    dataset_id: str,
    condition_id: str,
    artifacts: Mapping[str, str | Path],
    schema_manifest: Mapping[str, Any],
    clean_core_commit: str,
    clean_scorer_commit: str,
    v13_commit: str,
    device_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the immutable binding consumed by a formal Stage-B run."""

    required = {
        "dataset",
        "initial_feedback_labels",
        "reset_certificate",
        "initial_scorer_checkpoint",
        "training_config",
        "trace_code",
    }
    missing = sorted(required - set(artifacts))
    if missing:
        raise ValueError(f"Stage-B manifest artifacts are missing {missing}.")
    base = build_manifest(dict(artifacts))
    base.update(
        {
            "format_version": "go2_trace_stage_b_manifest_v1",
            "task_id": str(task_id),
            "dataset_id": str(dataset_id),
            "condition_id": str(condition_id),
            "schemas": dict(schema_manifest),
            "source_pins": {
                "trace_core_clean": str(clean_core_commit),
                "trace_scorer_clean": str(clean_scorer_commit),
                "canonical_v13": str(v13_commit),
            },
            "device_identity": (
                dict(device_identity) if device_identity is not None else None
            ),
        }
    )
    if any(not value for value in (task_id, dataset_id, condition_id)):
        raise ValueError("Stage-B task/dataset/condition identities must be non-empty.")
    return base


def main() -> None:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--output", required=True)
    create.add_argument("--input", action="append", default=[], metavar="NAME=PATH")
    verify = sub.add_parser("verify")
    verify.add_argument("--manifest", required=True)
    args = parser.parse_args()
    if args.command == "create":
        entries = {}
        for item in args.input:
            name, path = item.split("=", 1)
            if not name or name in entries:
                raise ValueError(f"Invalid or duplicate manifest name: {name!r}")
            entries[name] = path
        manifest = build_manifest(entries)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        print(output)
    else:
        manifest = json.loads(Path(args.manifest).read_text())
        verify_manifest(manifest)
        print("TRACE manifest verified")


if __name__ == "__main__":
    main()
