"""Formal Go2 online TRACE implementation for canonical V13."""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORTS = {
    "ResetResult": (".simulator_reset", "ResetResult"),
    "capture_snapshot_v1": (".simulator_reset", "capture_snapshot_v1"),
    "capture_physical37": (".simulator_reset", "capture_physical37"),
    "capture_rwm45": (".simulator_reset", "capture_rwm45"),
    "reset_from_snapshot": (".simulator_reset", "reset_from_snapshot"),
    "reset_native_random": (".simulator_reset", "reset_native_random"),
    "CONDITION_REGISTRY": (".condition_registry", "CONDITION_REGISTRY"),
    "guard_cuda_device": (".device_guard", "guard_cuda_device"),
    "summarize_go2_trajectory": (".trajectory", "summarize_go2_trajectory"),
    "build_go2_feedback_prompt": (
        ".go2_feedback_prompt",
        "build_go2_feedback_prompt",
    ),
    "Go2TraceScorer": (".scorer", "Go2TraceScorer"),
    "ScorerBinding": (".scorer", "ScorerBinding"),
    "score_summaries": (".scorer", "score_summaries"),
    "global_top_alpha": (".selection", "global_top_alpha"),
    "RuleBootstrapConfig": (".rule_bootstrap", "RuleBootstrapConfig"),
    "score_rule_summaries": (".rule_bootstrap", "score_rule_summaries"),
    "MutableTraceReplayBuffer": (".replay", "MutableTraceReplayBuffer"),
    "mix_trace_within_synthetic": (
        ".replay",
        "mix_trace_within_synthetic",
    ),
    "Go2OnlineTraceManager": (
        ".online_trace_manager",
        "Go2OnlineTraceManager",
    ),
    "CanonicalV13ReplaySemantics": (
        ".v13_replay_adapter",
        "CanonicalV13ReplaySemantics",
    ),
    # Legacy read-only compatibility only.
    "TraceReplaySampler": (".replay", "TraceReplaySampler"),
    "mix_trace_replay_batch": (".replay", "mix_trace_replay_batch"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value
