"""Cumulative, deduplicated feedback state used by online TRACE refreshes."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .feedback_pairs import (
    PairQuota,
    build_batch_global_random_pairs,
    build_feedback_pairs,
    build_trace_feedback_pairs,
)
from .go2_feedback_prompt import validate_label
from .label_feedback_with_codex import build_batch_prompt, salvage_label_response
from .schemas import PAIR_SCHEMA_HASH, PAIR_SCHEMA_VERSION, PROMPT_HASH


def pair_quota(feedback_budget: int, cross_region_fraction: float) -> PairQuota:
    if int(feedback_budget) < 1:
        raise ValueError("feedback_budget must be positive.")
    if not 0.0 <= float(cross_region_fraction) <= 1.0:
        raise ValueError("cross_region_fraction must be in [0,1].")
    cross = int(round(int(feedback_budget) * float(cross_region_fraction)))
    return PairQuota(
        within_region=int(feedback_budget) - cross,
        cross_region=cross,
    )


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            default=str,
        ).encode("utf-8")
    ).hexdigest()


class CumulativeFeedbackManager:
    """Pair sampling + atomic cumulative-label store.

    The label provider receives pair rows and returns any subset of label rows.
    Valid rows are salvaged immediately; missing/invalid rows cause this refresh
    to fail without corrupting previously accepted labels.
    """

    def __init__(
        self,
        *,
        label_store_path: str | Path,
        confidence_threshold: float,
        label_provider: Callable[
            [Sequence[Mapping[str, Any]]], Sequence[Mapping[str, Any]]
        ],
        pair_seed: int,
        planar_command_scales: Sequence[float],
        pair_sampling_mode: str = "region_quota",
        initial_labels_path: str | Path | None = None,
    ) -> None:
        self.path = Path(label_store_path).expanduser().resolve()
        self.confidence_threshold = float(confidence_threshold)
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be in [0,1].")
        self.label_provider = label_provider
        self.pair_seed = int(pair_seed)
        self.pair_sampling_mode = str(pair_sampling_mode)
        if self.pair_sampling_mode not in {
            "trace_original",
            "region_quota",
            "batch_global_random",
        }:
            raise ValueError(
                "pair_sampling_mode must be trace_original, region_quota, "
                "or batch_global_random."
            )
        self.planar_command_scales = tuple(map(float, planar_command_scales))
        if (
            len(self.planar_command_scales) != 2
            or any(
                not np.isfinite(value) or value <= 0.0
                for value in self.planar_command_scales
            )
        ):
            raise ValueError(
                "planar_command_scales must contain two finite positive values."
            )
        self.refresh_count = 0
        self.last_pair_metrics: dict[str, Any] = {}
        self._labels: dict[str, dict[str, Any]] = {}
        self.initial_labels_path = (
            Path(initial_labels_path).expanduser().resolve()
            if initial_labels_path is not None
            else None
        )
        if self.initial_labels_path is not None and not self.initial_labels_path.is_file():
            raise FileNotFoundError(self.initial_labels_path)
        self.initial_labels_file_sha256 = (
            hashlib.sha256(self.initial_labels_path.read_bytes()).hexdigest()
            if self.initial_labels_path is not None
            else None
        )
        initial_labels: dict[str, dict[str, Any]] = {}
        if self.initial_labels_path is not None:
            with self.initial_labels_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (
                        row.get("pair_schema_version") != PAIR_SCHEMA_VERSION
                        or row.get("pair_schema_hash") != PAIR_SCHEMA_HASH
                        or row.get("prompt_hash") != PROMPT_HASH
                    ):
                        raise ValueError(
                            "Initial feedback labels are incompatible with the "
                            "active pair/prompt semantics."
                        )
                    pair_id = str(row.get("pair_id", ""))
                    if not pair_id or pair_id in initial_labels:
                        raise ValueError(
                            "Initial feedback labels contain an empty or duplicate pair ID."
                        )
                    initial_labels[pair_id] = dict(row)
            if not initial_labels:
                raise ValueError("Initial feedback label file is empty.")
        self.initial_label_count = len(initial_labels)
        if self.path.exists():
            with self.path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if (
                        row.get("pair_schema_version") != PAIR_SCHEMA_VERSION
                        or row.get("pair_schema_hash") != PAIR_SCHEMA_HASH
                        or row.get("prompt_hash") != PROMPT_HASH
                    ):
                        raise ValueError(
                            "Existing cumulative labels use incompatible pre-region "
                            "pair semantics. Start a new output directory; automatic "
                            "migration is unsafe."
                        )
                    pair_id = str(row["pair_id"])
                    self._labels[pair_id] = dict(row)
            for pair_id, row in initial_labels.items():
                if self._labels.get(pair_id) != row:
                    raise ValueError(
                        "Existing cumulative feedback does not contain the exact "
                        "configured initial-label prefix."
                    )
        elif initial_labels:
            self._labels.update(initial_labels)
            self._atomic_write()

    @property
    def label_count(self) -> int:
        return len(self._labels)

    @property
    def labels_sha256(self) -> str:
        return _canonical_hash([self._labels[key] for key in sorted(self._labels)])

    def _atomic_write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor, temporary = tempfile.mkstemp(
            prefix=f".{self.path.name}.", suffix=".tmp", dir=self.path.parent
        )
        os.close(descriptor)
        try:
            with open(temporary, "w", encoding="utf-8") as handle:
                for key in sorted(self._labels):
                    handle.write(
                        json.dumps(
                            self._labels[key], ensure_ascii=False, sort_keys=True
                        )
                        + "\n"
                    )
            os.replace(temporary, self.path)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)

    def request(
        self,
        summaries: Sequence[Mapping[str, Any]],
        *,
        feedback_budget: int,
        cross_region_fraction: float,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        if self.pair_sampling_mode == "trace_original":
            pairs = build_trace_feedback_pairs(
                summaries,
                pair_count=feedback_budget,
                pair_prefix=f"refresh{self.refresh_count:08d}",
                seed=self.pair_seed + self.refresh_count,
                planar_command_scales=self.planar_command_scales,
            )
        elif self.pair_sampling_mode == "batch_global_random":
            pairs = build_batch_global_random_pairs(
                summaries,
                pair_count=feedback_budget,
                pair_prefix=f"refresh{self.refresh_count:08d}",
                seed=self.pair_seed + self.refresh_count,
                planar_command_scales=self.planar_command_scales,
            )
        else:
            quota = pair_quota(feedback_budget, cross_region_fraction)
            pairs = build_feedback_pairs(
                summaries,
                quota=quota,
                pair_prefix=f"refresh{self.refresh_count:08d}",
                seed=self.pair_seed + self.refresh_count,
                planar_command_scales=self.planar_command_scales,
            )
        region_counts: dict[str, int] = {}
        for pair in pairs:
            key = (
                f"{pair['command_region_i']}->{pair['command_region_j']}"
                if not pair["same_command_region"]
                else str(pair["command_region_i"])
            )
            region_counts[key] = region_counts.get(key, 0) + 1
        self.last_pair_metrics = {
            "feedback_budget": int(feedback_budget),
            "pair_sampling_mode": self.pair_sampling_mode,
            "cross_region_fraction": float(cross_region_fraction),
            "fixed_cross_region_quota_applied": (
                self.pair_sampling_mode == "region_quota"
            ),
            "within_region_count": sum(
                bool(pair["same_command_region"]) for pair in pairs
            ),
            "cross_region_count": sum(
                not bool(pair["same_command_region"]) for pair in pairs
            ),
            "region_pair_counts": region_counts,
            "sampling_seed": self.pair_seed + self.refresh_count,
            "balance_regions": False,
        }
        response = list(self.label_provider(pairs))
        expected = {str(pair["pair_id"]): pair for pair in pairs}
        accepted: dict[str, dict[str, Any]] = {}
        for raw in response:
            pair_id = str(raw.get("pair_id", ""))
            if pair_id not in expected or pair_id in accepted:
                continue
            try:
                label = validate_label(raw, expected_pair_id=pair_id)
            except Exception:
                continue
            pair = expected[pair_id]
            accepted[pair_id] = {
                **label,
                "trajectory_i": pair["trajectory_i"],
                "trajectory_j": pair["trajectory_j"],
                "comparison_type": pair["comparison_type"],
                "command_region_i": pair["command_region_i"],
                "command_region_j": pair["command_region_j"],
                "same_command_region": pair["same_command_region"],
                "planar_command_scales": pair["planar_command_scales"],
                "pair_schema_version": PAIR_SCHEMA_VERSION,
                "pair_schema_hash": PAIR_SCHEMA_HASH,
                "prompt_hash": PROMPT_HASH,
                "pair_sha256": _canonical_hash(
                    {key: value for key, value in pair.items() if key != "prompt"}
                ),
            }
        missing = sorted(set(expected) - set(accepted))
        # Persist every valid label even when the provider exhausted its repair
        # rounds for other pairs.  The scorer refresh remains fail-closed, but
        # already-paid feedback is retained for later cumulative training.
        self._labels.update(accepted)
        if accepted:
            self._atomic_write()
        self.refresh_count += 1
        if missing:
            raise RuntimeError(
                f"Feedback refresh missing/invalid {len(missing)} pair labels: {missing[:10]}"
            )
        trainable = [
            row
            for row in accepted.values()
            if row["feedback"] in {"i", "j"}
            and float(row["confidence"]) >= self.confidence_threshold
        ]
        return pairs, trainable

    def cumulative_training_labels(self) -> list[dict[str, Any]]:
        return [
            row
            for row in self._labels.values()
            if row.get("feedback") in {"i", "j"}
            and float(row.get("confidence", 0.0)) >= self.confidence_threshold
        ]

    def state_dict(self) -> dict[str, Any]:
        return {
            "refresh_count": self.refresh_count,
            "pair_seed": self.pair_seed,
            "pair_sampling_mode": self.pair_sampling_mode,
            "labels_sha256": self.labels_sha256,
            "label_count": self.label_count,
            "path": str(self.path),
            "pair_schema_version": PAIR_SCHEMA_VERSION,
            "pair_schema_hash": PAIR_SCHEMA_HASH,
            "prompt_hash": PROMPT_HASH,
            "planar_command_scales": self.planar_command_scales,
            "last_pair_metrics": self.last_pair_metrics,
            "initial_labels_path": (
                str(self.initial_labels_path)
                if self.initial_labels_path is not None
                else None
            ),
            "initial_labels_file_sha256": self.initial_labels_file_sha256,
            "initial_label_count": self.initial_label_count,
            "labels": [self._labels[key] for key in sorted(self._labels)],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if (
            state.get("pair_schema_version") != PAIR_SCHEMA_VERSION
            or state.get("pair_schema_hash") != PAIR_SCHEMA_HASH
            or state.get("prompt_hash") != PROMPT_HASH
        ):
            raise ValueError(
                "Feedback checkpoint predates six-region pair semantics and cannot "
                "be migrated safely. Resume from a new six-region checkpoint."
            )
        if int(state["pair_seed"]) != self.pair_seed:
            raise ValueError("Feedback pair seed changed across resume.")
        if state.get("pair_sampling_mode", "region_quota") != self.pair_sampling_mode:
            raise ValueError("Feedback pair sampling mode changed across resume.")
        if (
            state.get("initial_labels_file_sha256")
            != self.initial_labels_file_sha256
        ):
            raise ValueError("Initial feedback labels changed across resume.")
        if int(state.get("initial_label_count", -1)) != self.initial_label_count:
            raise ValueError("Initial feedback label count changed across resume.")
        if tuple(state["planar_command_scales"]) != self.planar_command_scales:
            raise ValueError("Feedback planar-command scales changed across resume.")
        labels = state.get("labels")
        if not isinstance(labels, list):
            raise ValueError("Feedback checkpoint does not contain cumulative labels.")
        restored = {str(row["pair_id"]): dict(row) for row in labels}
        restored_hash = _canonical_hash(
            [restored[key] for key in sorted(restored)]
        )
        if restored_hash != state["labels_sha256"]:
            raise ValueError("Cumulative labels in the checkpoint are corrupt.")
        if int(state["label_count"]) != len(restored):
            raise ValueError("Cumulative label count in the checkpoint is corrupt.")
        if self._labels:
            for pair_id, row in self._labels.items():
                restored_row = restored.get(pair_id)
                if (
                    restored_row is None
                    or _canonical_hash([restored_row])
                    != _canonical_hash([row])
                ):
                    raise ValueError(
                        "Existing cumulative label store conflicts with checkpoint."
                    )
        self._labels = restored
        if restored:
            self._atomic_write()
        self.refresh_count = int(state["refresh_count"])
        self.last_pair_metrics = dict(state.get("last_pair_metrics", {}))


class CodexBatchLabelProvider:
    """Configurable strict-JSON Codex provider with partial repair."""

    def __init__(
        self,
        *,
        repo_root: str | Path,
        schema_path: str | Path,
        control_dir: str | Path,
        batch_size: int = 20,
        repair_rounds: int = 2,
        timeout_seconds: float = 240.0,
        model: str | None = None,
        reasoning_effort: str | None = None,
        workers: int = 16,
        max_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
    ) -> None:
        self.repo_root = Path(repo_root).expanduser().resolve()
        self.schema_path = Path(schema_path).expanduser().resolve()
        self.control_dir = Path(control_dir).expanduser().resolve()
        self.batch_size = int(batch_size)
        self.repair_rounds = int(repair_rounds)
        self.timeout_seconds = float(timeout_seconds)
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.workers = int(workers)
        self.max_retries = int(max_retries)
        self.retry_backoff_seconds = float(retry_backoff_seconds)
        self.last_request_metrics: dict[str, Any] = {}
        if shutil.which("codex") is None:
            raise FileNotFoundError("Online TRACE feedback requires the codex executable.")
        if not self.schema_path.is_file():
            raise FileNotFoundError(self.schema_path)
        if self.reasoning_effort not in {None, "low", "medium", "high", "xhigh"}:
            raise ValueError("Codex reasoning_effort is unsupported.")
        if (
            self.batch_size < 1
            or self.repair_rounds < 0
            or self.workers < 1
            or self.max_retries < 0
            or self.retry_backoff_seconds < 0.0
        ):
            raise ValueError("Codex batch/concurrency/retry settings are invalid.")

    def _request_hash(self, batch_prompt: str) -> str:
        return _canonical_hash(
            {
                "prompt": batch_prompt,
                "schema_sha256": hashlib.sha256(
                    self.schema_path.read_bytes()
                ).hexdigest(),
                "model": self.model,
                "reasoning_effort": self.reasoning_effort,
            }
        )

    def _call(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        call_id: str,
    ) -> Mapping[str, Any] | None:
        self.control_dir.mkdir(parents=True, exist_ok=True)
        batch_prompt = build_batch_prompt(list(rows))
        request_hash = self._request_hash(batch_prompt)
        artifact_id = f"{call_id}_{request_hash}"
        response = self.control_dir / f"{artifact_id}.json"
        log = self.control_dir / f"{artifact_id}.log"
        if response.is_file():
            try:
                cached = json.loads(response.read_text(encoding="utf-8"))
                if isinstance(cached, Mapping):
                    return cached
            except json.JSONDecodeError:
                response.unlink(missing_ok=True)
        command = [
            "codex",
            "exec",
            "--cd",
            str(self.repo_root),
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--output-schema",
            str(self.schema_path),
            "-o",
            str(response),
            "-",
        ]
        if self.model:
            command[2:2] = ["--model", self.model]
        if self.reasoning_effort:
            command[2:2] = [
                "--config",
                f'model_reasoning_effort="{self.reasoning_effort}"',
            ]
        try:
            result = subprocess.run(
                command,
                input=batch_prompt,
                text=True,
                capture_output=True,
                timeout=self.timeout_seconds,
            )
            log.write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
            if result.returncode != 0 or not response.exists():
                return None
            return json.loads(response.read_text(encoding="utf-8"))
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
            log.write_text(f"{type(exc).__name__}: {exc}\n", encoding="utf-8")
            return None

    def _call_with_retries(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        call_id: str,
    ) -> Mapping[str, Any] | None:
        for attempt in range(self.max_retries + 1):
            response = self._call(
                rows,
                call_id=f"{call_id}_try{attempt:02d}",
            )
            if response is not None:
                return response
            if attempt < self.max_retries and self.retry_backoff_seconds:
                time.sleep(self.retry_backoff_seconds * (2**attempt))
        return None

    def __call__(
        self, pairs: Sequence[Mapping[str, Any]]
    ) -> Sequence[Mapping[str, Any]]:
        started_at = time.perf_counter()
        pair_by_id = {str(row["pair_id"]): row for row in pairs}
        accepted: dict[str, dict[str, Any]] = {}
        pending = list(pair_by_id)
        batch_count = 0
        repair_rounds_used = 0
        for repair in range(self.repair_rounds + 1):
            if not pending:
                break
            repair_rounds_used = repair + 1
            next_pending: list[str] = []
            batches = [
                pending[start : start + self.batch_size]
                for start in range(0, len(pending), self.batch_size)
            ]
            batch_count += len(batches)

            def run_batch(
                item: tuple[int, list[str]],
            ) -> tuple[list[str], Mapping[str, Any] | None]:
                batch_index, ids = item
                return ids, self._call_with_retries(
                    [pair_by_id[pair_id] for pair_id in ids],
                    call_id=f"r{repair:02d}_batch{batch_index:05d}",
                )

            worker_count = min(self.workers, len(batches))
            with ThreadPoolExecutor(
                max_workers=worker_count,
                thread_name_prefix="go2-trace-label",
            ) as executor:
                results = list(executor.map(run_batch, enumerate(batches)))
            for ids, response in results:
                if response is None:
                    next_pending.extend(ids)
                    continue
                valid, _failures = salvage_label_response(response, ids)
                accepted.update(valid)
                next_pending.extend(pair_id for pair_id in ids if pair_id not in valid)
            pending = next_pending
        self.last_request_metrics = {
            "pair_count": len(pair_by_id),
            "accepted_count": len(accepted),
            "pending_count": len(pending),
            "batch_count": batch_count,
            "batch_size": self.batch_size,
            "workers": self.workers,
            "repair_rounds_used": repair_rounds_used,
            "elapsed_seconds": time.perf_counter() - started_at,
        }
        return [accepted[key] for key in sorted(accepted)]
