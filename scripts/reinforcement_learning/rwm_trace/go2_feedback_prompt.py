"""Task-specific LLM comparison prompt for Go2 TRACE."""

from __future__ import annotations

import json
from typing import Any, Mapping

from .schemas import LABEL_SCHEMA_VERSION, PROMPT_HASH, PROMPT_VERSION
from .trajectory import build_go2_llm_display


def build_go2_feedback_prompt(pair: Mapping[str, Any]) -> str:
    """Build a strict-JSON prompt without D4RL-specific wording.

    Marginal replay value for coverage and corrective learning is primary.
    Severe posture/survival failure remains a veto. Reward and return are
    retained as diagnostics, never hidden or forced to be ignored.
    """

    left = pair["trajectory_i"]
    right = pair["trajectory_j"]
    payload = {
        "prompt_version": PROMPT_VERSION,
        "prompt_hash": PROMPT_HASH,
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "pair_id": str(pair["pair_id"]),
        "comparison_type": str(pair["comparison_type"]),
        "command_region_i": str(pair["command_region_i"]),
        "command_region_j": str(pair["command_region_j"]),
        "same_command_region": bool(pair["same_command_region"]),
        "planar_command_scales": list(pair["planar_command_scales"]),
        "task": (
            "Choose which short Go2 trajectory has greater marginal TRACE "
            "replay value for filling missing behavior coverage and improving "
            "a broadly capable locomotion policy. Do not simply choose the "
            "easier task or the trajectory that already performs best."
        ),
        "selection_objective": {
            "primary": (
                "Prefer viable trajectories that add corrective learning signal "
                "for under-covered commands, directions, command combinations, "
                "and nontrivial but recoverable tracking deficits."
            ),
            "task_difficulty_not_failure": (
                "A difficult command or systematic tracking weakness can be high "
                "value. A physically broken, collapsed, terminal, or implausible "
                "trajectory is not useful merely because it performs badly."
            ),
            "coverage_shortcut_guard": (
                "Do not reward an easy or common stand/front/back example merely "
                "for having low error. A stable stand or already-mastered simple "
                "motion usually has low marginal replay value. Lateral, yaw, "
                "combined-axis, and other nontrivial viable behavior can have "
                "higher value when it exposes a learnable coverage gap."
            ),
            "quality_floor": (
                "The selected trajectory must remain physically viable enough to "
                "provide usable replay. Prefer informative imperfection over either "
                "a trivial success or catastrophic failure."
            ),
        },
        "evaluation_order": [
            (
                "First apply validity plus the posture and survival veto. Severe "
                "tilt, growing roll/pitch instability, height collapse, early "
                "termination, or physically implausible motion is unusable replay."
            ),
            (
                "Among viable trajectories, evaluate marginal coverage value. "
                "Prefer behavior that covers a nontrivial direction, command mode, "
                "axis combination, or recoverable policy weakness over an easy, "
                "common, already-mastered example."
            ),
            (
                "Use commanded velocity diagnostics to identify useful corrective "
                "signal. Moderate under-realization, tracking error, or direction "
                "weakness on a viable difficult task can make a trajectory more "
                "valuable; do not automatically prefer the lowest-error trajectory. "
                "For inactive axes, inspect whether drift reveals a meaningful gap."
            ),
            (
                "Use smoothness, saturation, joint/actuator behavior, and contact "
                "switching to judge whether the corrective example remains "
                "learnable and physically plausible, not to favor trivial motion."
            ),
            (
                "Reward and simulator return are required diagnostics. Use them as "
                "supporting evidence only. High return or low tracking error does "
                "not imply high marginal replay value."
            ),
        ],
        "source_guidance": {
            "offline_window": (
                "These are observed dataset windows and need not share a start. "
                "Compare their likely marginal contribution to missing behavior "
                "coverage using the supplied command-aware diagnostics."
            ),
            "within_command_region": (
                "Both trajectories belong to the same signed command region. "
                "Prefer the viable example with the more useful corrective signal "
                "or broader command-mode/axis coverage, not automatically the one "
                "with the best current performance."
            ),
            "cross_command_region": (
                "The trajectories belong to different signed command regions. "
                "Compare marginal coverage and corrective value after the universal "
                "viability veto. Do not prefer easy/common stand or simple forward/"
                "back motion merely because its normalized errors are lower."
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
    if len(criteria) != len(set(criteria)):
        raise ValueError("used_criteria must not contain duplicates.")
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
