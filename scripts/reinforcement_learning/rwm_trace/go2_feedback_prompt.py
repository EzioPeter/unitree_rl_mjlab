"""Task-specific LLM comparison prompt for Go2 TRACE."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .schemas import LABEL_SCHEMA_VERSION, PROMPT_HASH, PROMPT_VERSION
from .trajectory import build_go2_llm_display


def build_go2_feedback_prompt(pair: Mapping[str, Any]) -> str:
    """Build a strict-JSON prompt without D4RL-specific wording.

    Velocity tracking is primary. Severe posture/survival failure is a veto.
    Reward and return are retained as diagnostics, never hidden or forced to be
    ignored.
    """

    left = pair["trajectory_i"]
    right = pair["trajectory_j"]
    payload = {
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "pair_id": str(pair["pair_id"]),
        "comparison_type": str(pair["comparison_type"]),
        "pairing_context": {
            "command_region_i": str(pair.get("command_region_i", "")),
            "command_region_j": str(pair.get("command_region_j", "")),
            "within_same_command_region": (
                pair.get("command_region_i") == pair.get("command_region_j")
            ),
        },
        "task": (
            "Choose which short Go2 trajectory is the better TRACE replay "
            "candidate for locomotion policy improvement."
        ),
        "evaluation_order": [
            (
                "First evaluate commanded base velocity tracking. For every active "
                "vx, vy, or yaw axis, prefer low full/steady tracking error, correct "
                "direction, adequate realization, and commanded displacement. For "
                "inactive axes, prefer low drift."
            ),
            (
                "Then apply the posture and survival veto. A trajectory with severe "
                "tilt, growing roll/pitch instability, height collapse, early "
                "termination, or physically implausible motion must not win merely "
                "because its velocity or return looks better."
            ),
            (
                "Among trajectories that track commands and remain viable, use "
                "smooth non-saturated actions, bounded joint/actuator behavior, and "
                "plausible contact switching as secondary evidence."
            ),
            (
                "Reward and simulator return are required diagnostics. Use them as "
                "supporting evidence, but inspect whether they agree with velocity, "
                "posture, and control evidence before making the decision."
            ),
        ],
        "source_guidance": {
            "offline_window": (
                "These are observed dataset windows and need not share a start. "
                "Compare task quality using the supplied command-aware diagnostics."
            ),
            "same_start_candidate": (
                "These candidates share the same reset state, so their outcome "
                "differences are directly comparable."
            ),
            "cross_start_candidate": (
                "These candidates start from different states. Do not treat small "
                "raw metric differences as counterfactual proof; require a clear "
                "task-quality difference."
            ),
        },
        "trajectory_i": build_go2_llm_display(left),
        "trajectory_j": build_go2_llm_display(right),
        "output_contract": {
            "pair_id": str(pair["pair_id"]),
            "feedback": "exactly one of: i, j, tie, invalid",
            "confidence": "finite number in [0, 1]",
            "reason": "brief task-grounded explanation",
            "used_criteria": "non-empty list of metric groups used",
            "return_used_as_primary": "boolean",
            "motion_quality_overrode_return": "boolean",
            "velocity_tracking_used_as_primary": "boolean",
            "posture_veto_applied": "boolean",
        },
    }
    return (
        "Return exactly one JSON object and no Markdown or surrounding text.\n"
        + json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True)
    )


def validate_label(label: Mapping[str, Any], *, expected_pair_id: str) -> dict[str, Any]:
    required = {
        "pair_id",
        "feedback",
        "confidence",
        "reason",
        "used_criteria",
        "return_used_as_primary",
        "motion_quality_overrode_return",
        "velocity_tracking_used_as_primary",
        "posture_veto_applied",
    }
    if set(label) != required:
        raise ValueError(
            f"Label fields differ from contract: missing={sorted(required-set(label))}, "
            f"extra={sorted(set(label)-required)}."
        )
    if str(label["pair_id"]) != expected_pair_id:
        raise ValueError("Label pair_id does not match the requested pair.")
    if label["feedback"] not in {"i", "j", "tie", "invalid"}:
        raise ValueError("feedback must be i, j, tie, or invalid.")
    confidence = float(label["confidence"])
    if not 0.0 <= confidence <= 1.0:
        raise ValueError("confidence must be in [0,1].")
    if not str(label["reason"]).strip():
        raise ValueError("reason must be non-empty.")
    criteria = label["used_criteria"]
    if not isinstance(criteria, list) or not criteria or not all(
        isinstance(item, str) and item.strip() for item in criteria
    ):
        raise ValueError("used_criteria must be a non-empty list of strings.")
    if not isinstance(label["return_used_as_primary"], bool) or not isinstance(
        label["motion_quality_overrode_return"], bool
    ):
        raise ValueError("The two return-use fields must be booleans.")
    if not isinstance(label["velocity_tracking_used_as_primary"], bool) or not isinstance(
        label["posture_veto_applied"], bool
    ):
        raise ValueError("The velocity-primary and posture-veto fields must be booleans.")
    allowed_criteria = {
        "validity",
        "survival",
        "velocity_tracking",
        "stand_drift",
        "posture",
        "action_continuity",
        "actuation",
        "contact_gait",
        "return",
    }
    unknown = sorted(set(criteria) - allowed_criteria)
    if unknown:
        raise ValueError(f"used_criteria contains unsupported values: {unknown}.")
    return {
        **dict(label),
        "pair_id": expected_pair_id,
        "confidence": confidence,
        "reason": str(label["reason"]).strip(),
        "used_criteria": [str(item).strip() for item in criteria],
    }
