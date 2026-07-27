"""Controlled Go2 simulator reset primitives for TRACE.

There are deliberately two reset modes:

``exact_snapshot``
    Restores a simulator snapshot captured from MJLab.  This is the only mode
    allowed to claim an exact simulator reset.

``canonical_real_projection``
    Projects a 45-D real-robot RWM state into a deterministic MJLab state.  It
    is useful for real-data TRACE proposals, but is never reported as exact
    because the real log does not contain root height/yaw or simulator hidden
    state.

All proposal branches must be created from one captured snapshot.  Resetting
each vector environment independently is not a controlled counterfactual when
domain randomization is enabled.
"""

from __future__ import annotations

import hashlib
import inspect
from dataclasses import dataclass, field
from typing import Any, Mapping

import torch


SNAPSHOT_VERSION = "go2_trace_snapshot_v1"
SNAPSHOT_REQUIRED_KEYS = (
    "root_state_local",
    "joint_position",
    "joint_velocity",
    "action",
    "prev_action",
    "prev_prev_action",
    "command",
)
SNAPSHOT_FIELD_WIDTHS = {
    "root_state_local": 13,
    "joint_position": 12,
    "joint_velocity": 12,
    "action": 12,
    "prev_action": 12,
    "prev_prev_action": 12,
    "command": 3,
}
SNAPSHOT_SCHEMA_SHA256 = hashlib.sha256(
    repr((SNAPSHOT_VERSION, tuple(SNAPSHOT_FIELD_WIDTHS.items()))).encode("utf-8")
).hexdigest()
FULL_SIMULATOR_STATE_VERSION = "mujoco_integration_state_v1"
CONTROLLED_MODEL_FIELDS = (
    "body_mass",
    "body_ipos",
    "body_inertia",
    "body_iquat",
    "geom_friction",
    "actuator_gainprm",
    "actuator_biasprm",
    "actuator_forcerange",
)
CONTROL_TARGET_FIELDS = (
    "joint_pos_target",
    "joint_vel_target",
    "joint_effort_target",
    "tendon_len_target",
    "tendon_vel_target",
    "tendon_effort_target",
    "site_effort_target",
)


SnapshotV1 = dict[str, torch.Tensor | str]


@dataclass(frozen=True)
class _SubsetRuntimeState:
    """Runtime rows that a global MJLab control/forward pass may overwrite."""

    unselected: torch.Tensor
    entity_targets: dict[tuple[str, str], tuple[Any, torch.Tensor]]
    ctrl_storage: Any
    ctrl: torch.Tensor
    actuator_force_storage: Any
    actuator_force: torch.Tensor


@dataclass(frozen=True)
class _ObservationHistoryState:
    buffer: Any
    data: torch.Tensor
    num_pushes: torch.Tensor
    pointer: int


@dataclass(frozen=True)
class _SubsetObservationState:
    """Observation-manager state to preserve for non-target worlds."""

    unselected: torch.Tensor
    cache_was_none: bool
    cache: Any
    histories: tuple[_ObservationHistoryState, ...]


@dataclass(frozen=True)
class ResetResult:
    """Structured output shared by manual and simulator-native reset paths.

    ``manual_restore_completed`` reports that the write/synchronize path ran.
    ``exact_reset`` remains false until an external, hash-matched H-step
    certificate has passed; executing this function alone never proves exactness.
    """

    reset_kind: str
    manual_restore_completed: bool
    exact_reset: bool
    source_certification_applicable: bool
    snapshot_version: str
    env_ids: torch.Tensor
    snapshot: SnapshotV1
    physical37: torch.Tensor
    rwm45: torch.Tensor
    observable_reconstruction_errors: dict[str, float] = field(default_factory=dict)
    observations: Any = None
    info: Any = None
    seed: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    def as_legacy_dict(self) -> dict[str, Any]:
        """Return the small dictionary produced by the historical restore API."""

        return {
            "exact_reset": self.exact_reset,
            "reset_kind": self.reset_kind,
            "snapshot_version": self.snapshot_version,
        }


def _base_env(env: Any) -> Any:
    unwrapped = getattr(env, "unwrapped", None)
    return env if unwrapped is None else unwrapped


def _normalize_env_ids(env: Any, env_ids: torch.Tensor) -> torch.Tensor:
    env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.ndim != 1 or env_ids.numel() == 0:
        raise ValueError("env_ids must be a non-empty one-dimensional tensor.")
    if bool((env_ids < 0).any()) or bool((env_ids >= int(env.num_envs)).any()):
        raise ValueError(f"env_ids are outside [0, {int(env.num_envs)}).")
    if int(torch.unique(env_ids).numel()) != int(env_ids.numel()):
        raise ValueError("env_ids must not contain duplicates.")
    return env_ids


def _snapshot_sha256(snapshot: Mapping[str, Any]) -> str:
    digest = hashlib.sha256()
    digest.update(SNAPSHOT_VERSION.encode("utf-8"))
    for key in SNAPSHOT_REQUIRED_KEYS:
        value = torch.as_tensor(snapshot[key]).detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(repr(tuple(value.shape)).encode("utf-8"))
        digest.update(bytes(value.view(torch.uint8).reshape(-1).tolist()))
    return digest.hexdigest()


def _snapshot_reconstruction_errors(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> dict[str, float]:
    result: dict[str, float] = {}
    for key in SNAPSHOT_REQUIRED_KEYS:
        expected_t = torch.as_tensor(expected[key], dtype=torch.float32)
        actual_t = torch.as_tensor(actual[key], device=expected_t.device, dtype=torch.float32)
        result[key] = float(torch.max(torch.abs(actual_t - expected_t)).detach().cpu())
    result["overall_max"] = max(result.values(), default=0.0)
    return result


def _snapshot_action_term_tensors(manager: Any) -> dict[tuple[str, str], torch.Tensor]:
    """Capture per-environment action-term tensors before a subset restore."""

    result: dict[tuple[str, str], torch.Tensor] = {}
    terms = getattr(manager, "_terms", {})
    if not isinstance(terms, Mapping):
        return result
    for term_name, term in terms.items():
        for attribute, value in vars(term).items():
            if (
                isinstance(value, torch.Tensor)
                and value.ndim >= 1
                and int(value.shape[0]) == int(manager.num_envs)
            ):
                result[(str(term_name), str(attribute))] = value.detach().clone()
    return result


def _restore_unselected_action_term_rows(
    manager: Any,
    saved: Mapping[tuple[str, str], torch.Tensor],
    env_ids: torch.Tensor,
) -> None:
    """Undo process-action side effects outside ``env_ids``."""

    if not saved or int(env_ids.numel()) == int(manager.num_envs):
        return
    selected = torch.zeros(int(manager.num_envs), dtype=torch.bool, device=env_ids.device)
    selected[env_ids] = True
    unselected = ~selected
    terms = getattr(manager, "_terms", {})
    for (term_name, attribute), previous in saved.items():
        term = terms.get(term_name)
        current = getattr(term, attribute, None) if term is not None else None
        if (
            isinstance(current, torch.Tensor)
            and current.shape == previous.shape
            and current.device == previous.device
        ):
            current[unselected] = previous[unselected]


def _snapshot_batched_storage(
    owner: Any,
    attribute: str,
    *,
    num_envs: int,
    context: str,
) -> tuple[Any, torch.Tensor]:
    """Return a writable MJLab storage proxy and a detached tensor snapshot."""

    storage = getattr(owner, attribute, None)
    if storage is None:
        raise RuntimeError(
            f"Subset manual reset cannot preserve {context}.{attribute}: "
            "the runtime field is unavailable."
        )
    try:
        tensor = storage if isinstance(storage, torch.Tensor) else storage[:]
    except (IndexError, TypeError, RuntimeError) as exc:
        raise RuntimeError(
            f"Subset manual reset cannot read {context}.{attribute}."
        ) from exc
    if (
        not isinstance(tensor, torch.Tensor)
        or tensor.ndim < 1
        or int(tensor.shape[0]) != int(num_envs)
    ):
        shape = getattr(tensor, "shape", None)
        raise RuntimeError(
            f"Subset manual reset requires batched {context}.{attribute} with "
            f"leading dimension {num_envs}, got {shape!r}."
        )
    return storage, tensor.detach().clone()


def _capture_subset_runtime_state(
    env: Any,
    env_ids: torch.Tensor,
) -> _SubsetRuntimeState | None:
    """Fail closed unless non-target control state can be preserved exactly.

    MJLab v1.2.0 action terms and ``Scene.write_data_to_sim`` operate over every
    vector world. A subset restore therefore snapshots the public entity target
    tensors plus the low-level ``ctrl`` and ``actuator_force`` arrays. Stateful
    custom actuators are rejected because V1 has no schema for their hidden
    buffers. The pinned Go2 runtime uses only stateless built-in position
    actuators and is supported.
    """

    if int(env_ids.numel()) == int(env.num_envs):
        return None

    selected = torch.zeros(
        int(env.num_envs),
        dtype=torch.bool,
        device=env_ids.device,
    )
    selected[env_ids] = True
    unselected = ~selected

    entities = getattr(env.scene, "entities", None)
    if not isinstance(entities, Mapping):
        raise RuntimeError(
            "Subset manual reset requires MJLab scene.entities so non-target "
            "actuator targets can be preserved."
        )

    entity_targets: dict[tuple[str, str], tuple[Any, torch.Tensor]] = {}
    for entity_name, entity in entities.items():
        actuators = getattr(entity, "actuators", ())
        try:
            has_actuators = len(actuators) > 0
        except TypeError as exc:
            raise RuntimeError(
                f"Subset manual reset cannot inspect actuators for entity "
                f"{entity_name!r}."
            ) from exc
        if not has_actuators:
            continue
        custom_actuators = getattr(entity, "_custom_actuators", None)
        if custom_actuators is None:
            raise RuntimeError(
                f"Subset manual reset cannot prove that entity {entity_name!r} "
                "uses only stateless built-in actuators."
            )
        if len(custom_actuators) != 0:
            raise RuntimeError(
                f"Subset manual reset does not support stateful custom actuators "
                f"on entity {entity_name!r}."
            )
        entity_data = getattr(entity, "data", None)
        if entity_data is None:
            raise RuntimeError(
                f"Subset manual reset cannot access data for actuated entity "
                f"{entity_name!r}."
            )
        for field_name in CONTROL_TARGET_FIELDS:
            entity_targets[(str(entity_name), field_name)] = (
                _snapshot_batched_storage(
                    entity_data,
                    field_name,
                    num_envs=int(env.num_envs),
                    context=f"scene.entities[{entity_name!r}].data",
                )
            )

    sim_data = getattr(getattr(env, "sim", None), "data", None)
    if sim_data is None:
        raise RuntimeError(
            "Subset manual reset requires MJLab sim.data control storage."
        )
    ctrl_storage, ctrl = _snapshot_batched_storage(
        sim_data,
        "ctrl",
        num_envs=int(env.num_envs),
        context="sim.data",
    )
    actuator_force_storage, actuator_force = _snapshot_batched_storage(
        sim_data,
        "actuator_force",
        num_envs=int(env.num_envs),
        context="sim.data",
    )
    return _SubsetRuntimeState(
        unselected=unselected,
        entity_targets=entity_targets,
        ctrl_storage=ctrl_storage,
        ctrl=ctrl,
        actuator_force_storage=actuator_force_storage,
        actuator_force=actuator_force,
    )


def _restore_subset_entity_targets(state: _SubsetRuntimeState | None) -> None:
    if state is None:
        return
    for storage, previous in state.entity_targets.values():
        storage[state.unselected] = previous[state.unselected]


def _restore_subset_control(state: _SubsetRuntimeState | None) -> None:
    if state is None:
        return
    state.ctrl_storage[state.unselected] = state.ctrl[state.unselected]


def _restore_subset_actuator_force(state: _SubsetRuntimeState | None) -> None:
    if state is None:
        return
    state.actuator_force_storage[state.unselected] = state.actuator_force[
        state.unselected
    ]


def _clone_observation_cache(value: Any, num_envs: int, path: str) -> Any:
    if isinstance(value, torch.Tensor):
        if value.ndim < 1 or int(value.shape[0]) != int(num_envs):
            raise RuntimeError(
                f"Subset manual reset cannot isolate observation cache {path}: "
                f"expected leading dimension {num_envs}, got {tuple(value.shape)}."
            )
        return value.detach().clone()
    if isinstance(value, Mapping):
        return {
            key: _clone_observation_cache(
                child,
                num_envs,
                f"{path}.{key}",
            )
            for key, child in value.items()
        }
    raise RuntimeError(
        f"Subset manual reset cannot isolate observation cache {path}: "
        f"unsupported value type {type(value).__qualname__}."
    )


def _restore_observation_cache_rows(
    current: Any,
    previous: Any,
    unselected: torch.Tensor,
    path: str,
) -> None:
    if isinstance(previous, torch.Tensor):
        if not isinstance(current, torch.Tensor) or current.shape != previous.shape:
            raise RuntimeError(
                f"Observation cache structure changed during subset reset at {path}."
            )
        current[unselected] = previous[unselected]
        return
    if not isinstance(current, Mapping) or not isinstance(previous, Mapping):
        raise RuntimeError(
            f"Observation cache structure changed during subset reset at {path}."
        )
    if set(current) != set(previous):
        raise RuntimeError(
            f"Observation cache keys changed during subset reset at {path}."
        )
    for key in previous:
        _restore_observation_cache_rows(
            current[key],
            previous[key],
            unselected,
            f"{path}.{key}",
        )


def _capture_subset_observation_state(
    env: Any,
    env_ids: torch.Tensor,
) -> _SubsetObservationState | None:
    """Preflight and snapshot the pinned MJLab observation runtime.

    CircularBuffer uses a global time pointer and an environment axis at index
    one. Exact row isolation is therefore only provable for ``max_len == 1``.
    DelayBuffer updates lags, counters and RNG globally, so any configured delay
    is rejected. These restrictions match the pinned Go2 TRACE configuration.
    """

    if int(env_ids.numel()) == int(env.num_envs):
        return None
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        return None

    selected = torch.zeros(
        int(env.num_envs),
        dtype=torch.bool,
        device=env_ids.device,
    )
    selected[env_ids] = True
    unselected = ~selected

    term_groups = getattr(manager, "_group_obs_term_cfgs", None)
    history_groups = getattr(manager, "_group_obs_term_history_buffer", None)
    delay_groups = getattr(manager, "_group_obs_term_delay_buffer", None)
    class_term_groups = getattr(manager, "_group_obs_class_term_cfgs", None)
    noise_instances = getattr(manager, "_group_obs_class_instances", None)
    for name, value in (
        ("_group_obs_term_cfgs", term_groups),
        ("_group_obs_term_history_buffer", history_groups),
        ("_group_obs_term_delay_buffer", delay_groups),
        ("_group_obs_class_term_cfgs", class_term_groups),
        ("_group_obs_class_instances", noise_instances),
    ):
        if not isinstance(value, Mapping):
            raise RuntimeError(
                f"Subset manual reset cannot inspect ObservationManager.{name}."
            )

    for group_name, term_cfgs in term_groups.items():
        for term_cfg in term_cfgs:
            history_length = int(getattr(term_cfg, "history_length", 0))
            delay_max_lag = int(getattr(term_cfg, "delay_max_lag", 0))
            if history_length > 1:
                raise RuntimeError(
                    "Subset manual reset requires observation history_length <= 1; "
                    f"group {group_name!r} has {history_length}."
                )
            if delay_max_lag != 0:
                raise RuntimeError(
                    "Subset manual reset requires zero observation delay; "
                    f"group {group_name!r} has delay_max_lag={delay_max_lag}."
                )
            if getattr(term_cfg, "noise", None) is not None:
                raise RuntimeError(
                    "Subset manual reset requires observation corruption/noise "
                    f"disabled; group {group_name!r} still has a noise term."
                )
    if any(bool(group) for group in delay_groups.values()):
        raise RuntimeError(
            "Subset manual reset cannot isolate MJLab DelayBuffer state."
        )
    if any(bool(group) for group in class_term_groups.values()):
        raise RuntimeError(
            "Subset manual reset cannot isolate stateful observation term instances."
        )
    if bool(noise_instances):
        raise RuntimeError(
            "Subset manual reset cannot isolate observation noise-model instances."
        )

    histories: list[_ObservationHistoryState] = []
    for group_name, group_buffers in history_groups.items():
        if not isinstance(group_buffers, Mapping):
            raise RuntimeError(
                f"Observation history group {group_name!r} is not a mapping."
            )
        for term_name, buffer in group_buffers.items():
            max_length = int(getattr(buffer, "max_length", -1))
            batch_size = int(getattr(buffer, "batch_size", -1))
            data = getattr(buffer, "_buffer", None)
            num_pushes = getattr(buffer, "_num_pushes", None)
            pointer = getattr(buffer, "_pointer", None)
            if max_length != 1 or batch_size != int(env.num_envs):
                raise RuntimeError(
                    "Subset manual reset can only isolate initialized one-frame "
                    f"history; {group_name}.{term_name} has max_length={max_length}, "
                    f"batch_size={batch_size}."
                )
            if (
                not isinstance(data, torch.Tensor)
                or data.ndim < 2
                or int(data.shape[0]) != 1
                or int(data.shape[1]) != int(env.num_envs)
                or not isinstance(num_pushes, torch.Tensor)
                or tuple(num_pushes.shape) != (int(env.num_envs),)
                or pointer != 0
            ):
                raise RuntimeError(
                    "Subset manual reset requires an initialized MJLab v1.2.0 "
                    f"CircularBuffer for {group_name}.{term_name}."
                )
            histories.append(
                _ObservationHistoryState(
                    buffer=buffer,
                    data=data.detach().clone(),
                    num_pushes=num_pushes.detach().clone(),
                    pointer=int(pointer),
                )
            )

    if not hasattr(manager, "_obs_buffer"):
        raise RuntimeError(
            "Subset manual reset cannot access ObservationManager._obs_buffer."
        )
    cache_value = manager._obs_buffer
    cache = (
        None
        if cache_value is None
        else _clone_observation_cache(cache_value, int(env.num_envs), "_obs_buffer")
    )
    return _SubsetObservationState(
        unselected=unselected,
        cache_was_none=cache_value is None,
        cache=cache,
        histories=tuple(histories),
    )


def _restore_subset_observation_state(
    observation_manager: Any,
    state: _SubsetObservationState | None,
) -> None:
    if state is None:
        return
    for history in state.histories:
        current_data = getattr(history.buffer, "_buffer", None)
        current_num_pushes = getattr(history.buffer, "_num_pushes", None)
        current_pointer = getattr(history.buffer, "_pointer", None)
        if (
            not isinstance(current_data, torch.Tensor)
            or current_data.shape != history.data.shape
            or not isinstance(current_num_pushes, torch.Tensor)
            or current_num_pushes.shape != history.num_pushes.shape
            or current_pointer != history.pointer
        ):
            raise RuntimeError(
                "Observation history structure changed during subset reset."
            )
        current_data[:, state.unselected] = history.data[:, state.unselected]
        current_num_pushes[state.unselected] = history.num_pushes[state.unselected]

    if state.cache_was_none:
        observation_manager._obs_buffer = None
        return
    current_cache = getattr(observation_manager, "_obs_buffer", None)
    if current_cache is None:
        raise RuntimeError(
            "Observation cache disappeared during subset reset."
        )
    _restore_observation_cache_rows(
        current_cache,
        state.cache,
        state.unselected,
        "_obs_buffer",
    )


def _observation_sync_provenance(observation_manager: Any) -> dict[str, Any]:
    """Describe what V1 can and cannot reconstruct for managed observations."""

    term_groups = getattr(observation_manager, "_group_obs_term_cfgs", None)
    inspected = isinstance(term_groups, Mapping)
    max_history_length = 0
    max_delay_lag = 0
    if inspected:
        for term_cfgs in term_groups.values():
            for term_cfg in term_cfgs:
                max_history_length = max(
                    max_history_length,
                    int(getattr(term_cfg, "history_length", 0)),
                )
                max_delay_lag = max(
                    max_delay_lag,
                    int(getattr(term_cfg, "delay_max_lag", 0)),
                )
    current_observation_exact = (
        inspected and max_history_length <= 1 and max_delay_lag == 0
    )
    return {
        "manager_config_inspected": inspected,
        "compute_update_history": True,
        "stale_cache_bypassed": True,
        "history_restored_from_snapshot": False,
        "delay_buffer_restored_from_snapshot": False,
        "max_history_length": max_history_length if inspected else None,
        "max_delay_lag": max_delay_lag if inspected else None,
        "returned_selected_observation_is_current_state_exact": (
            current_observation_exact
        ),
        "returned_observation_is_current_state_exact": current_observation_exact,
        "limitation": (
            "TRACE V1 does not store observation history or delay buffers. "
            "The current sample is recomputed and appended without resetting "
            "those buffers; histories longer than one or nonzero delay retain "
            "pre-restore runtime context and are not exact snapshot state."
        ),
    }


def roll_pitch_from_projected_gravity(projected_gravity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    gravity = projected_gravity / torch.clamp(torch.linalg.norm(projected_gravity, dim=-1, keepdim=True), min=1.0e-8)
    pitch = torch.asin(torch.clamp(gravity[..., 0], -1.0, 1.0))
    roll = torch.atan2(-gravity[..., 1], -gravity[..., 2])
    return roll, pitch


def quaternion_wxyz_from_roll_pitch(roll: torch.Tensor, pitch: torch.Tensor) -> torch.Tensor:
    half_roll = 0.5 * roll
    half_pitch = 0.5 * pitch
    cr, sr = torch.cos(half_roll), torch.sin(half_roll)
    cp, sp = torch.cos(half_pitch), torch.sin(half_pitch)
    return torch.stack([cr * cp, sr * cp, cr * sp, -sr * sp], dim=-1)


def _quat_apply_wxyz(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    scalar = quaternion[..., :1]
    xyz = quaternion[..., 1:]
    first_cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (scalar * first_cross + torch.cross(xyz, first_cross, dim=-1))


def _force_commands(env: Any, command: torch.Tensor, env_ids: torch.Tensor) -> None:
    term = env.command_manager.get_term("twist")
    command = command.to(device=env.device, dtype=torch.float32)
    if not hasattr(term, "vel_command_b"):
        raise AttributeError("twist command term does not expose vel_command_b.")
    term.vel_command_b[env_ids] = command
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[env_ids] = torch.linalg.norm(command, dim=-1) < 1.0e-8
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[env_ids] = False


def capture_snapshot_v1(
    env: Any,
    env_ids: torch.Tensor,
    *,
    command: torch.Tensor | None = None,
) -> SnapshotV1:
    """Capture the reset-relevant MJLab state for selected environments."""

    env = _base_env(env)
    env_ids = _normalize_env_ids(env, env_ids)
    robot = env.scene["robot"]
    root_state = torch.cat([robot.data.root_link_pose_w, robot.data.root_link_vel_w], dim=-1)[env_ids].clone()
    root_state[:, :3] -= env.scene.env_origins[env_ids]
    if command is None:
        command = env.command_manager.get_command("twist")[env_ids]
    manager = env.action_manager
    return {
        "snapshot_version": SNAPSHOT_VERSION,
        "root_state_local": root_state.detach().clone(),
        "joint_position": robot.data.joint_pos[env_ids].detach().clone(),
        "joint_velocity": robot.data.joint_vel[env_ids].detach().clone(),
        "action": manager.action[env_ids].detach().clone(),
        "prev_action": manager.prev_action[env_ids].detach().clone(),
        "prev_prev_action": manager.prev_prev_action[env_ids].detach().clone(),
        "command": command.to(device=env.device, dtype=torch.float32).detach().clone(),
    }


def capture_go2_simulator_snapshot(
    env: Any,
    env_ids: torch.Tensor,
    *,
    command: torch.Tensor | None = None,
) -> SnapshotV1:
    """Backward-compatible alias for :func:`capture_snapshot_v1`."""

    return capture_snapshot_v1(env, env_ids, command=command)


def capture_physical37(env: Any, env_ids: torch.Tensor) -> torch.Tensor:
    """Capture the 37-D writable physical state used by V1 certification."""

    env = _base_env(env)
    env_ids = _normalize_env_ids(env, env_ids)
    robot = env.scene["robot"]
    root_state = torch.cat(
        [robot.data.root_link_pose_w, robot.data.root_link_vel_w],
        dim=-1,
    )[env_ids].clone()
    root_state[:, :3] -= env.scene.env_origins[env_ids]
    return torch.cat(
        [
            root_state,
            robot.data.joint_pos[env_ids],
            robot.data.joint_vel[env_ids],
        ],
        dim=-1,
    ).detach().float().clone()


def full_simulator_state_layout(env: Any) -> dict[str, Any]:
    """Describe MuJoCo's complete integration state for this simulator.

    This follows ``mjSTATE_INTEGRATION`` rather than an observation or a
    robot-specific projection.  MJWarp currently does not implement MuJoCo
    userdata/plugin state, so those dimensions must be zero for this capture
    path to be complete.
    """

    import mujoco

    env = _base_env(env)
    model = env.sim.mj_model
    unsupported = {
        "userdata": int(getattr(model, "nuserdata", 0)),
        "plugin": int(getattr(model, "npluginstate", 0)),
    }
    if any(unsupported.values()):
        raise RuntimeError(
            "MuJoCo integration-state capture cannot omit userdata/plugin "
            f"dimensions: {unsupported}."
        )

    widths = (
        ("time", 1),
        ("qpos", int(model.nq)),
        ("qvel", int(model.nv)),
        ("act", int(model.na)),
        ("qacc_warmstart", int(model.nv)),
        ("ctrl", int(model.nu)),
        ("qfrc_applied", int(model.nv)),
        ("xfrc_applied", 6 * int(model.nbody)),
        ("eq_active", int(model.neq)),
        ("mocap_pos", 3 * int(model.nmocap)),
        ("mocap_quat", 4 * int(model.nmocap)),
    )
    fields: dict[str, dict[str, int]] = {}
    offset = 0
    for name, width in widths:
        fields[name] = {"start": offset, "stop": offset + width, "width": width}
        offset += width
    official_size = int(
        mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION)
    )
    if offset != official_size:
        raise RuntimeError(
            "MJWarp-supported integration layout does not match MuJoCo "
            f"mjSTATE_INTEGRATION size: supported={offset}, official={official_size}."
        )
    return {
        "version": FULL_SIMULATOR_STATE_VERSION,
        "mujoco_state_signature": "mjSTATE_INTEGRATION",
        "dimension": official_size,
        "fields": fields,
        "unsupported_dimensions": unsupported,
    }


def capture_full_simulator_state(
    env: Any,
    env_ids: torch.Tensor,
) -> torch.Tensor:
    """Capture the complete persistent MuJoCo integration state.

    The result is used only as reset-certification ground truth.  It is not
    consumed by ``reset_from_snapshot`` and is not an RWM observation.
    """

    import mujoco_warp as mjwarp
    import warp as wp

    env = _base_env(env)
    env_ids = _normalize_env_ids(env, env_ids)
    layout = full_simulator_state_layout(env)
    state_wp = wp.empty(
        (int(env.num_envs), int(layout["dimension"])),
        dtype=wp.float32,
        device=env.sim.wp_device,
    )
    mjwarp.get_state(
        env.sim.wp_model,
        env.sim.wp_data,
        state_wp,
        mjwarp.State.INTEGRATION,
    )
    state = wp.to_torch(state_wp)
    return state[env_ids].detach().float().clone()


def capture_rwm45(env: Any, env_ids: torch.Tensor) -> torch.Tensor:
    """Capture the 45-D Go2 state used by the RWM and reset validator."""

    env = _base_env(env)
    env_ids = _normalize_env_ids(env, env_ids)
    data = env.scene["robot"].data
    return torch.cat(
        [
            data.root_link_lin_vel_b[env_ids],
            data.root_link_ang_vel_b[env_ids],
            data.projected_gravity_b[env_ids],
            data.joint_pos[env_ids] - data.default_joint_pos[env_ids],
            data.joint_vel[env_ids],
            data.actuator_force[env_ids],
        ],
        dim=-1,
    ).detach().float().clone()


def validate_snapshot(snapshot: Mapping[str, Any], batch_size: int | None = None) -> None:
    if "snapshot_version" not in snapshot:
        raise ValueError("TRACE V1 snapshot is missing required key: snapshot_version.")
    version = snapshot["snapshot_version"]
    if version != SNAPSHOT_VERSION:
        raise ValueError(f"Unsupported TRACE snapshot version: {version!r}.")
    missing = [key for key in SNAPSHOT_REQUIRED_KEYS if snapshot.get(key) is None]
    if missing:
        raise ValueError(f"TRACE exact snapshot is missing required keys: {missing}.")
    nonfloating = [
        key
        for key in SNAPSHOT_REQUIRED_KEYS
        if not torch.is_floating_point(torch.as_tensor(snapshot[key]))
    ]
    if nonfloating:
        raise ValueError(f"TRACE V1 snapshot fields must be floating point: {nonfloating}.")
    shapes = {key: tuple(torch.as_tensor(snapshot[key]).shape) for key in SNAPSHOT_REQUIRED_KEYS}
    bad_widths = {
        key: shape
        for key, shape in shapes.items()
        if len(shape) != 2 or shape[1] != SNAPSHOT_FIELD_WIDTHS[key]
    }
    if bad_widths:
        raise ValueError(f"TRACE V1 snapshot field shapes are invalid: {bad_widths}.")
    inferred_batch = shapes[SNAPSHOT_REQUIRED_KEYS[0]][0]
    expected_batch = inferred_batch if batch_size is None else int(batch_size)
    bad_batches = {
        key: shape
        for key, shape in shapes.items()
        if shape[0] != expected_batch
    }
    if bad_batches:
        raise ValueError(
            f"TRACE snapshot batch mismatch, expected {expected_batch}: {bad_batches}."
        )
    nonfinite = [
        key
        for key in SNAPSHOT_REQUIRED_KEYS
        if not bool(torch.isfinite(torch.as_tensor(snapshot[key], dtype=torch.float32)).all())
    ]
    if nonfinite:
        raise ValueError(f"TRACE V1 snapshot contains non-finite values: {nonfinite}.")


def repeat_snapshot_rows(snapshot: Mapping[str, Any], repeats: int) -> dict[str, Any]:
    """Repeat every source row contiguously for controlled proposal branches."""

    if repeats < 1:
        raise ValueError("repeats must be positive")
    validate_snapshot(snapshot)
    result: dict[str, Any] = {"snapshot_version": SNAPSHOT_VERSION}
    for key in SNAPSHOT_REQUIRED_KEYS:
        result[key] = torch.as_tensor(snapshot[key]).repeat_interleave(int(repeats), dim=0)
    return result


def synchronize_branch_domain_parameters(env: Any, branches_per_state: int) -> None:
    """Make DR parameters identical inside every contiguous branch group."""

    if branches_per_state < 1 or env.num_envs % branches_per_state:
        raise ValueError(
            f"num_envs={env.num_envs} must be divisible by branches_per_state={branches_per_state}."
        )
    leaders = torch.arange(0, env.num_envs, branches_per_state, device=env.device)
    source = leaders.repeat_interleave(branches_per_state)
    for field_name in CONTROLLED_MODEL_FIELDS:
        value = getattr(env.sim.model, field_name, None)
        if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == env.num_envs:
            value.copy_(value[source].clone())
    runtime = getattr(env, "_friend_dr_runtime", None)
    if isinstance(runtime, dict):
        for value in runtime.values():
            if isinstance(value, torch.Tensor) and value.ndim >= 1 and value.shape[0] == env.num_envs:
                value.copy_(value[source].clone())


def step_without_automatic_reset(env: Any, action: torch.Tensor):
    """MJLab policy step that preserves the physical terminal next state.

    ``ManagerBasedRlEnv.step`` resets terminated worlds before returning its
    observation.  TRACE needs the terminal transition itself and must stop the
    proposal at that boundary, so candidate collection uses this equivalent
    step with only the reset block omitted.
    """

    env.action_manager.process_action(action.to(env.device))
    for _ in range(env.cfg.decimation):
        env._sim_step_counter += 1
        env.action_manager.apply_action()
        env.scene.write_data_to_sim()
        env.sim.step()
        env.scene.update(dt=env.physics_dt)
    env.episode_length_buf += 1
    env.common_step_counter += 1
    env.reset_buf = env.termination_manager.compute()
    env.reset_terminated = env.termination_manager.terminated
    env.reset_time_outs = env.termination_manager.time_outs
    env.reward_buf = env.reward_manager.compute(dt=env.step_dt)
    env.metrics_manager.compute()
    env.sim.forward()
    env.command_manager.compute(dt=env.step_dt)
    if "step" in env.event_manager.available_modes:
        env.event_manager.apply(mode="step", dt=env.step_dt)
    if "interval" in env.event_manager.available_modes:
        env.event_manager.apply(mode="interval", dt=env.step_dt)
    env.sim.sense()
    env.obs_buf = env.observation_manager.compute(update_history=True)
    return (
        env.obs_buf,
        env.reward_buf,
        env.reset_terminated,
        env.reset_time_outs,
        env.extras,
    )


def reset_from_snapshot(
    env: Any,
    snapshot: Mapping[str, Any],
    env_ids: torch.Tensor,
) -> ResetResult:
    """Manually restore V1 without invoking any simulator reset routine."""

    env = _base_env(env)
    env_ids = _normalize_env_ids(env, env_ids)
    validate_snapshot(snapshot, len(env_ids))
    robot = env.scene["robot"]
    root_state = torch.as_tensor(snapshot["root_state_local"], device=env.device, dtype=torch.float32).clone()
    root_state[:, :3] += env.scene.env_origins[env_ids]
    joint_position = torch.as_tensor(snapshot["joint_position"], device=env.device, dtype=torch.float32)
    joint_velocity = torch.as_tensor(snapshot["joint_velocity"], device=env.device, dtype=torch.float32)
    action = torch.as_tensor(snapshot["action"], device=env.device, dtype=torch.float32)
    prev_action = torch.as_tensor(snapshot["prev_action"], device=env.device, dtype=torch.float32)
    prev_prev_action = torch.as_tensor(snapshot["prev_prev_action"], device=env.device, dtype=torch.float32)
    command = torch.as_tensor(snapshot["command"], device=env.device, dtype=torch.float32)

    # This preflight must happen before the first state write. A subset request
    # that cannot preserve global MJLab control side effects fails atomically.
    subset_runtime_state = _capture_subset_runtime_state(env, env_ids)
    subset_observation_state = _capture_subset_observation_state(env, env_ids)

    robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_position, joint_velocity, env_ids=env_ids)

    # Process the held action so action terms reconstruct their position targets.
    manager = env.action_manager
    saved_action = manager.action.clone()
    saved_prev_action = manager.prev_action.clone()
    saved_prev_prev_action = manager.prev_prev_action.clone()
    saved_term_tensors = _snapshot_action_term_tensors(manager)
    all_actions = saved_action.clone()
    all_actions[env_ids] = action
    manager.process_action(all_actions)
    manager._action.copy_(saved_action)
    manager._prev_action.copy_(saved_prev_action)
    manager._prev_prev_action.copy_(saved_prev_prev_action)
    manager._action[env_ids] = action
    manager._prev_action[env_ids] = prev_action
    manager._prev_prev_action[env_ids] = prev_prev_action
    _restore_unselected_action_term_rows(manager, saved_term_tensors, env_ids)
    _force_commands(env, command, env_ids)

    env.episode_length_buf[env_ids] = 0
    # Reconstruct held PD targets and derived actuator forces before the
    # forward/sense pass. Otherwise forces can leak from the previous world.
    manager.apply_action()
    _restore_subset_entity_targets(subset_runtime_state)
    env.scene.write_data_to_sim()
    _restore_subset_control(subset_runtime_state)
    env.sim.forward()
    env.sim.sense()
    # MuJoCo forward recomputes actuator_force globally. Preserve the exact
    # pre-call value for non-target worlds; selected rows retain the newly
    # reconstructed force used by RWM45.
    _restore_subset_actuator_force(subset_runtime_state)
    observations = None
    observation_sync = {
        "manager_available": False,
        "compute_update_history": False,
        "stale_cache_bypassed": False,
        "history_restored_from_snapshot": False,
        "delay_buffer_restored_from_snapshot": False,
        "returned_selected_observation_is_current_state_exact": False,
        "returned_observation_is_current_state_exact": False,
        "limitation": "No observation manager was available.",
    }
    observation_manager = getattr(env, "observation_manager", None)
    if observation_manager is not None:
        observation_sync = {
            "manager_available": True,
            **_observation_sync_provenance(observation_manager),
        }
        # MJLab v1.2.0 returns _obs_buffer unchanged from compute() when
        # update_history=False. update_history=True bypasses that stale cache,
        # recomputes every term after forward/sense, and appends the current
        # sample without resetting history that V1 does not contain.
        observations = observation_manager.compute(update_history=True)
        _restore_subset_observation_state(
            observation_manager,
            subset_observation_state,
        )
        observation_sync["subset_unselected_runtime_state_preserved"] = (
            subset_observation_state is not None
        )
        if subset_observation_state is not None:
            # The selected rows are the freshly recomputed reset state. The
            # non-selected rows are deliberately restored to their pre-call
            # cache, so it would be incorrect to certify the whole returned
            # batch as one globally current observation.
            observation_sync[
                "returned_observation_is_current_state_exact"
            ] = False
        if hasattr(env, "obs_buf"):
            env.obs_buf = observations

    realized_snapshot = capture_snapshot_v1(env, env_ids)
    return ResetResult(
        reset_kind="manual_snapshot_restore",
        manual_restore_completed=True,
        exact_reset=False,
        source_certification_applicable=True,
        snapshot_version=SNAPSHOT_VERSION,
        env_ids=env_ids.detach().clone(),
        snapshot=realized_snapshot,
        physical37=capture_physical37(env, env_ids),
        rwm45=capture_rwm45(env, env_ids),
        observable_reconstruction_errors=_snapshot_reconstruction_errors(
            snapshot,
            realized_snapshot,
        ),
        observations=observations,
        provenance={
            "schema_sha256": SNAPSHOT_SCHEMA_SHA256,
            "source_snapshot_sha256": _snapshot_sha256(snapshot),
            "realized_snapshot_sha256": _snapshot_sha256(realized_snapshot),
            "batch_size": int(env_ids.numel()),
            "device": str(env.device),
            "certification_status": "not_evaluated",
            "native_reset_calls": 0,
            "observation_sync": observation_sync,
        },
    )


def restore_go2_simulator_snapshot(
    env: Any,
    snapshot: Mapping[str, Any],
    env_ids: torch.Tensor,
) -> dict[str, Any]:
    """Backward-compatible dictionary-returning V1 restore wrapper."""

    return reset_from_snapshot(env, snapshot, env_ids).as_legacy_dict()


def reset_native_random(
    env: Any,
    *,
    seed: int | None = None,
    env_ids: torch.Tensor | None = None,
) -> ResetResult:
    """Reset a dedicated vector environment from native episode-0 state.

    The public ``env.reset`` method is invoked exactly once. A subset is used
    only when that public signature explicitly exposes ``env_ids``; otherwise
    the request fails before the call and never falls back to private methods.
    """

    base_env = _base_env(env)
    all_env_ids = torch.arange(
        int(base_env.num_envs),
        device=base_env.device,
        dtype=torch.long,
    )
    requested_ids = all_env_ids
    if env_ids is not None:
        requested_ids = _normalize_env_ids(base_env, env_ids)

    reset_method = getattr(env, "reset", None)
    if not callable(reset_method):
        raise TypeError("env must expose a callable public reset method.")
    subset_requested = not torch.equal(requested_ids, all_env_ids)
    kwargs: dict[str, Any] = {}
    if seed is not None:
        kwargs["seed"] = int(seed)
    if subset_requested:
        try:
            parameters = inspect.signature(reset_method).parameters
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "Cannot prove that public env.reset supports subset env_ids."
            ) from exc
        if "env_ids" not in parameters:
            raise ValueError(
                "Public env.reset does not explicitly support subset env_ids."
            )
        kwargs["env_ids"] = requested_ids
    reset_output = reset_method(**kwargs)
    if isinstance(reset_output, tuple) and len(reset_output) == 2:
        observations, info = reset_output
    else:
        observations, info = reset_output, {}

    realized_snapshot = capture_snapshot_v1(base_env, requested_ids)
    return ResetResult(
        reset_kind="simulator_native_random",
        manual_restore_completed=False,
        exact_reset=False,
        source_certification_applicable=False,
        snapshot_version=SNAPSHOT_VERSION,
        env_ids=requested_ids.detach().clone(),
        snapshot=realized_snapshot,
        physical37=capture_physical37(base_env, requested_ids),
        rwm45=capture_rwm45(base_env, requested_ids),
        observations=observations,
        info=info,
        seed=seed,
        provenance={
            "schema_sha256": SNAPSHOT_SCHEMA_SHA256,
            "realized_snapshot_sha256": _snapshot_sha256(realized_snapshot),
            "batch_size": int(requested_ids.numel()),
            "device": str(base_env.device),
            "reset_scope": "subset" if subset_requested else "all_envs",
            "native_reset_calls": 1,
        },
    )


def canonical_snapshot_from_rwm_state(
    env: Any,
    states: torch.Tensor,
    commands: torch.Tensor,
    previous_actions: torch.Tensor,
    env_ids: torch.Tensor,
) -> dict[str, Any]:
    """Create a deterministic, explicitly approximate snapshot from real data."""

    if states.ndim != 2 or states.shape[-1] != 45:
        raise ValueError(f"Expected [batch, 45] RWM states, got {tuple(states.shape)}.")
    if len(states) != len(env_ids):
        raise ValueError("states and env_ids must have the same batch length.")
    env_ids = env_ids.to(device=env.device, dtype=torch.long)
    states = states.to(device=env.device, dtype=torch.float32)
    commands = commands.to(device=env.device, dtype=torch.float32)
    previous_actions = previous_actions.to(device=env.device, dtype=torch.float32)
    robot = env.scene["robot"]
    roll, pitch = roll_pitch_from_projected_gravity(states[:, 6:9])
    quaternion = quaternion_wxyz_from_roll_pitch(roll, pitch)
    root_state = robot.data.default_root_state[env_ids].clone()
    root_state[:, :2] = 0.0
    root_state[:, 3:7] = quaternion
    root_state[:, 7:10] = _quat_apply_wxyz(quaternion, states[:, 0:3])
    root_state[:, 10:13] = _quat_apply_wxyz(quaternion, states[:, 3:6])
    snapshot = {
        "snapshot_version": SNAPSHOT_VERSION,
        "root_state_local": root_state,
        "joint_position": robot.data.default_joint_pos[env_ids] + states[:, 9:21],
        "joint_velocity": states[:, 21:33],
        "action": previous_actions,
        "prev_action": previous_actions,
        "prev_prev_action": previous_actions,
        "command": commands,
    }
    restore_go2_simulator_snapshot(env, snapshot, env_ids)
    # Re-capture the realized simulator state; all branches are cloned from this
    # one canonical state rather than independently reconstructed.
    return capture_go2_simulator_snapshot(env, env_ids, command=commands)


def reset_go2_from_rwm_state(
    env: Any,
    states: torch.Tensor,
    env_ids: torch.Tensor,
    *,
    robot_name: str = "robot",
) -> dict[str, Any]:
    """Deprecated compatibility wrapper; never claims an exact reset."""

    del robot_name
    zeros_command = torch.zeros(len(states), 3, device=env.device)
    zeros_action = torch.zeros(len(states), env.action_manager.total_action_dim, device=env.device)
    snapshot = canonical_snapshot_from_rwm_state(env, states, zeros_command, zeros_action, env_ids)
    restore_go2_simulator_snapshot(env, snapshot, env_ids)
    return {
        "exact_reset": False,
        "reset_kind": "canonical_real_projection",
        "fixed_fields": ["root_x", "root_y", "root_z", "root_yaw"],
        "unrecoverable_fields": ["actuator_force", "simulator_hidden_state"],
        "recoverable_state_slice": [0, 33],
    }
