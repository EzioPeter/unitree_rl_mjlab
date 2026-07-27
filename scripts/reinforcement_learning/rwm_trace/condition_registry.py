"""Canonical target/perfect/imperfect simulator conditions for Go2 TRACE."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


REGISTRY_SCHEMA_VERSION = "go2_trace_condition_registry_v1"
FORBIDDEN_FORMAL_SWITCHES = frozenset(
    {"gravity", "friction", "joint_noise", "gravity_scale", "friction_scale"}
)


class ConditionRegistryError(ValueError):
    """Raised when condition metadata is unknown, stale, or inconsistent."""


@dataclass(frozen=True)
class RobotCondition:
    """Physical task condition relevant to the formal Go2 experiments."""

    profile: str
    payload_mass_kg: float
    rear_right_leg_strength_scale: float

    def to_dict(self) -> dict[str, str | float]:
        return asdict(self)


@dataclass(frozen=True)
class ConditionSpec:
    """Target/perfect and imperfect simulator mapping for one dataset/task."""

    condition_id: str
    target: RobotCondition
    perfect: RobotCondition
    imperfect: RobotCondition

    def __post_init__(self) -> None:
        if self.target != self.perfect:
            raise ConditionRegistryError(
                f"{self.condition_id}: target and perfect simulator must match."
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "condition_id": self.condition_id,
            "target": self.target.to_dict(),
            "perfect": self.perfect.to_dict(),
            "imperfect": self.imperfect.to_dict(),
        }


def _robot(
    profile: str,
    *,
    payload_mass_kg: float = 0.0,
    rear_right_leg_strength_scale: float = 1.0,
) -> RobotCondition:
    return RobotCondition(
        profile=profile,
        payload_mass_kg=payload_mass_kg,
        rear_right_leg_strength_scale=rear_right_leg_strength_scale,
    )


_NORMAL = _robot("normal")
_REGISTRY = {
    "g0": ConditionSpec("g0", _NORMAL, _NORMAL, _NORMAL),
    "p5": ConditionSpec(
        "p5",
        _robot("payload_5kg", payload_mass_kg=5.0),
        _robot("payload_5kg", payload_mass_kg=5.0),
        _NORMAL,
    ),
    "p75": ConditionSpec(
        "p75",
        _robot("payload_7_5kg", payload_mass_kg=7.5),
        _robot("payload_7_5kg", payload_mass_kg=7.5),
        _NORMAL,
    ),
    "rr05": ConditionSpec(
        "rr05",
        _robot("rear_right_weak_0_5", rear_right_leg_strength_scale=0.5),
        _robot("rear_right_weak_0_5", rear_right_leg_strength_scale=0.5),
        _NORMAL,
    ),
    "rr03": ConditionSpec(
        "rr03",
        _robot("rear_right_weak_0_3", rear_right_leg_strength_scale=0.3),
        _robot("rear_right_weak_0_3", rear_right_leg_strength_scale=0.3),
        _NORMAL,
    ),
}
CONDITION_REGISTRY: Mapping[str, ConditionSpec] = MappingProxyType(_REGISTRY)


def canonical_registry_payload() -> dict[str, Any]:
    return {
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "conditions": {
            condition_id: CONDITION_REGISTRY[condition_id].to_dict()
            for condition_id in sorted(CONDITION_REGISTRY)
        },
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_registry_hash() -> str:
    return hashlib.sha256(_canonical_json(canonical_registry_payload()).encode()).hexdigest()


def get_condition(condition_id: str) -> ConditionSpec:
    try:
        return CONDITION_REGISTRY[condition_id]
    except KeyError as exc:
        raise ConditionRegistryError(
            f"Unknown Go2 TRACE condition {condition_id!r}; "
            f"expected one of {sorted(CONDITION_REGISTRY)}."
        ) from exc


def canonical_condition_metadata(condition_id: str) -> dict[str, Any]:
    spec = get_condition(condition_id)
    return {
        "registry_schema_version": REGISTRY_SCHEMA_VERSION,
        "registry_sha256": canonical_registry_hash(),
        "condition": spec.to_dict(),
    }


def _reject_forbidden_switches(metadata: Mapping[str, Any]) -> None:
    stack: list[tuple[str, Mapping[str, Any]]] = [("", metadata)]
    while stack:
        prefix, current = stack.pop()
        for key, value in current.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key in FORBIDDEN_FORMAL_SWITCHES:
                raise ConditionRegistryError(
                    f"Formal condition metadata contains forbidden switch {path!r}."
                )
            if isinstance(value, Mapping):
                stack.append((path, value))


def assert_condition_metadata(
    metadata: Mapping[str, Any],
    *,
    expected_condition_id: str | None = None,
) -> ConditionSpec:
    """Fail closed unless metadata exactly matches the canonical registry."""

    if not isinstance(metadata, Mapping):
        raise ConditionRegistryError("Condition metadata must be a mapping.")
    _reject_forbidden_switches(metadata)
    condition = metadata.get("condition")
    if not isinstance(condition, Mapping):
        raise ConditionRegistryError("Condition metadata lacks a condition mapping.")
    condition_id = condition.get("condition_id")
    if not isinstance(condition_id, str):
        raise ConditionRegistryError("Condition metadata lacks a string condition_id.")
    if expected_condition_id is not None and condition_id != expected_condition_id:
        raise ConditionRegistryError(
            f"Condition {condition_id!r} does not match expected "
            f"{expected_condition_id!r}."
        )
    expected = canonical_condition_metadata(condition_id)
    if dict(metadata) != expected:
        raise ConditionRegistryError(
            f"Condition metadata for {condition_id!r} is stale or non-canonical."
        )
    return get_condition(condition_id)


def assert_simulator_condition(
    condition_id: str,
    *,
    role: str,
    actual: Mapping[str, Any],
) -> RobotCondition:
    """Assert target/perfect/imperfect simulator metadata for a formal task."""

    if role not in {"target", "perfect", "imperfect"}:
        raise ConditionRegistryError(
            f"Unsupported simulator role {role!r}; expected target/perfect/imperfect."
        )
    if not isinstance(actual, Mapping):
        raise ConditionRegistryError("Simulator condition metadata must be a mapping.")
    _reject_forbidden_switches(actual)
    expected = getattr(get_condition(condition_id), role)
    if dict(actual) != expected.to_dict():
        raise ConditionRegistryError(
            f"{condition_id}/{role} simulator condition does not match registry."
        )
    return expected


__all__ = [
    "CONDITION_REGISTRY",
    "ConditionRegistryError",
    "ConditionSpec",
    "FORBIDDEN_FORMAL_SWITCHES",
    "REGISTRY_SCHEMA_VERSION",
    "RobotCondition",
    "assert_condition_metadata",
    "assert_simulator_condition",
    "canonical_condition_metadata",
    "canonical_registry_hash",
    "canonical_registry_payload",
    "get_condition",
]
