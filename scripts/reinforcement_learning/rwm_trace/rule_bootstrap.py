"""Token-free Go2 rule scorer used only before the learned TRACE stage.

This adapts the legacy response-tracking/stability rule to the current Go2
summary schema and adds normalized simulator return as an equal third term.
It exercises six-region top-alpha, V13 reward/n-step, mutable replay, and
trainer integration as the learned stage without calling an LLM.  It is not
the final TRACE scorer and never trains the policy through its scalar score.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from .schemas import AXES, SUMMARY_SCHEMA_HASH, canonical_sha256


@dataclass(frozen=True)
class RuleBootstrapConfig:
    response_tracking_weight: float = 1.0 / 3.0
    dynamic_stability_weight: float = 1.0 / 3.0
    reward_weight: float = 1.0 / 3.0
    minimum_component_response_ratio: float = 0.02
    minimum_direction_correct_fraction: float = 0.50
    maximum_direction_violation_rate: float = 0.50
    minimum_survival_fraction: float = 0.01
    tilt_scale_rad: float = 0.20
    roll_pitch_rate_scale: float = 1.0
    action_delta_scale: float = 1.0

    def validate(self) -> None:
        if not math.isclose(
            self.response_tracking_weight
            + self.dynamic_stability_weight
            + self.reward_weight,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("Rule-bootstrap three-way weights must sum to one.")
        if any(
            not math.isclose(
                weight, 1.0 / 3.0, rel_tol=0.0, abs_tol=1.0e-12
            )
            for weight in (
                self.response_tracking_weight,
                self.dynamic_stability_weight,
                self.reward_weight,
            )
        ):
            raise ValueError(
                "Rule bootstrap requires tracking/stability/reward at one third each."
            )
        for name, value in asdict(self).items():
            value = float(value)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"Rule-bootstrap field {name!r} is invalid.")
        if self.minimum_direction_correct_fraction > 1.0:
            raise ValueError("minimum_direction_correct_fraction must be in [0,1].")
        if self.maximum_direction_violation_rate > 1.0:
            raise ValueError("maximum_direction_violation_rate must be in [0,1].")
        if self.minimum_survival_fraction > 1.0:
            raise ValueError("minimum_survival_fraction must be in [0,1].")
        if min(
            self.tilt_scale_rad,
            self.roll_pitch_rate_scale,
            self.action_delta_scale,
        ) <= 0.0:
            raise ValueError("Rule-bootstrap quality scales must be positive.")

    @property
    def sha256(self) -> str:
        self.validate()
        return canonical_sha256(
            {
                "version": "go2_trace_rule_bootstrap_v5",
                "summary_schema_hash": SUMMARY_SCHEMA_HASH,
                "config": asdict(self),
                "return_used": True,
                "return_normalization": "eligible_candidate_minmax",
                "selection": "command_region_stratified_top_alpha",
            }
        )


@dataclass(frozen=True)
class RuleBootstrapResult:
    scores: np.ndarray
    eligible: tuple[bool, ...]
    diagnostics: tuple[dict[str, Any], ...]
    config_sha256: str


def _finite(summary: Mapping[str, Any], name: str) -> float | None:
    try:
        value = float(summary.get(name, float("nan")))
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _quality_from_error(value: float, scale: float = 1.0) -> float:
    return 1.0 / (1.0 + max(0.0, value) / max(scale, 1.0e-8))


def _response_quality(value: float) -> float:
    # A response ratio of one is ideal; zero and large overshoot are worse.
    return 1.0 / (1.0 + abs(value - 1.0))


def _score_one(
    summary: Mapping[str, Any],
    config: RuleBootstrapConfig,
) -> tuple[float, bool, dict[str, Any]]:
    if summary.get("summary_schema_hash") != SUMMARY_SCHEMA_HASH:
        return 0.0, False, {"rejection_reason": "summary_schema"}
    survival = _finite(summary, "survival_fraction")
    saturation = _finite(summary, "action_saturation_fraction")
    if survival is None or survival < config.minimum_survival_fraction:
        return 0.0, False, {"rejection_reason": "survival"}
    if saturation is None:
        return 0.0, False, {"rejection_reason": "action_saturation_nonfinite"}
    simulator_return = _finite(summary, "simulator_return")
    if simulator_return is None:
        return 0.0, False, {"rejection_reason": "simulator_return_nonfinite"}

    active = {
        axis: bool((summary.get("command_active") or {}).get(axis, False))
        for axis in AXES
    }
    primary_components: dict[str, float] = {}
    if any(active.values()):
        response = _finite(summary, "minimum_active_axis_response_full")
        direction_correct = _finite(
            summary, "command_direction_correct_fraction"
        )
        direction_violation = _finite(
            summary, "command_direction_violation_rate"
        )
        if None in (response, direction_correct, direction_violation):
            return 0.0, False, {"rejection_reason": "active_motion_nonfinite"}
        assert response is not None
        assert direction_correct is not None
        assert direction_violation is not None
        if response < config.minimum_component_response_ratio:
            return 0.0, False, {"rejection_reason": "component_response"}
        if direction_correct < config.minimum_direction_correct_fraction:
            return 0.0, False, {"rejection_reason": "direction_correct"}
        if direction_violation > config.maximum_direction_violation_rate:
            return 0.0, False, {"rejection_reason": "direction_violation"}
        tracking_quality = []
        for axis in AXES:
            if not active[axis]:
                continue
            full = _finite(summary, f"tracking_{axis}_normalized_mae_full")
            steady = _finite(
                summary, f"tracking_{axis}_normalized_mae_steady"
            )
            if full is None or steady is None:
                return 0.0, False, {
                    "rejection_reason": f"tracking_{axis}_nonfinite"
                }
            tracking_quality.append(_quality_from_error(max(full, steady)))
        progress = _finite(summary, "command_displacement_realization_ratio")
        yaw_progress = _finite(summary, "yaw_displacement_realization_ratio")
        progress_quality = []
        if any(active[axis] for axis in ("vx", "vy")) and progress is not None:
            progress_quality.append(_response_quality(progress))
        if active["yaw"] and yaw_progress is not None:
            progress_quality.append(_response_quality(yaw_progress))
        primary_components = {
            "tracking": min(tracking_quality),
            "component_response": _response_quality(response),
            "direction": min(
                max(0.0, min(1.0, direction_correct)),
                1.0 - max(0.0, min(1.0, direction_violation)),
            ),
            "transport": (
                min(progress_quality) if progress_quality else 1.0
            ),
        }
    else:
        drift_quality = []
        floors = summary.get("command_normalization_floors") or {}
        for axis in AXES:
            drift = _finite(summary, f"drift_abs_mean_{axis}")
            floor = float(floors.get(axis, 0.05))
            if drift is None or not math.isfinite(floor) or floor <= 0.0:
                return 0.0, False, {
                    "rejection_reason": f"stand_drift_{axis}_nonfinite"
                }
            drift_quality.append(_quality_from_error(drift, floor))
        primary_components = {"stand_drift": min(drift_quality)}
    primary = min(primary_components.values())

    tilt = _finite(summary, "tilt_mean")
    roll_pitch_rate = _finite(summary, "roll_pitch_rate_rms")
    action_delta = _finite(summary, "action_delta_norm_mean")
    if None in (tilt, roll_pitch_rate, action_delta):
        return 0.0, False, {"rejection_reason": "stability_nonfinite"}
    assert tilt is not None
    assert roll_pitch_rate is not None
    assert action_delta is not None
    stability_components = {
        "survival": max(0.0, min(1.0, survival)),
        "tilt": math.exp(-max(0.0, tilt) / config.tilt_scale_rad),
        "roll_pitch_rate": math.exp(
            -max(0.0, roll_pitch_rate) / config.roll_pitch_rate_scale
        ),
        "action_delta": math.exp(
            -max(0.0, action_delta) / config.action_delta_scale
        ),
        "action_saturation": 1.0 - max(0.0, min(1.0, saturation)),
    }
    stability = float(np.mean(list(stability_components.values())))
    return 0.0, True, {
        "rejection_reason": None,
        "profile": "active_motion" if any(active.values()) else "stand",
        "primary_quality": float(primary),
        "stability_quality": stability,
        "primary_components": primary_components,
        "stability_components": stability_components,
        "simulator_return": simulator_return,
        "return_used": True,
    }


def score_rule_summaries(
    summaries: Sequence[Mapping[str, Any]],
    config: RuleBootstrapConfig,
) -> RuleBootstrapResult:
    """Return token-free rule scores for command-stratified top-alpha selection."""

    config.validate()
    rows = [_score_one(summary, config) for summary in summaries]
    eligible_returns = [
        float(row[2]["simulator_return"]) for row in rows if row[1]
    ]
    if eligible_returns:
        reward_min = min(eligible_returns)
        reward_max = max(eligible_returns)
    else:
        reward_min = reward_max = 0.0
    reward_range = reward_max - reward_min
    scores: list[float] = []
    diagnostics: list[dict[str, Any]] = []
    for _old_score, eligible, raw_diagnostics in rows:
        row = dict(raw_diagnostics)
        if not eligible:
            scores.append(0.0)
            diagnostics.append(row)
            continue
        reward_quality = (
            0.5
            if reward_range <= 1.0e-12
            else (
                float(row["simulator_return"]) - reward_min
            )
            / reward_range
        )
        score = (
            config.response_tracking_weight * float(row["primary_quality"])
            + config.dynamic_stability_weight * float(
                row["stability_quality"]
            )
            + config.reward_weight * reward_quality
        )
        row.update(
            {
                "reward_quality": float(reward_quality),
                "reward_normalization_min": float(reward_min),
                "reward_normalization_max": float(reward_max),
                "weights": {
                    "response_tracking": config.response_tracking_weight,
                    "dynamic_stability": config.dynamic_stability_weight,
                    "reward": config.reward_weight,
                },
            }
        )
        scores.append(float(score))
        diagnostics.append(row)
    return RuleBootstrapResult(
        scores=np.asarray(scores, dtype=np.float32),
        eligible=tuple(bool(row[1]) for row in rows),
        diagnostics=tuple(diagnostics),
        config_sha256=config.sha256,
    )
