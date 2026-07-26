#!/usr/bin/env python3
"""Fail-closed g0 MJLab runner for Go2 reset certification.

Heavy simulator imports are intentionally delayed until the adapter is called,
so parser/provenance tests remain CPU-only.  The runner performs two independent
manual restores and rollouts and refuses to return data unless they agree.
"""

from __future__ import annotations

import hashlib
import inspect
import json
import math
import platform
import subprocess
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Callable, Mapping

import torch

from scripts.reinforcement_learning.rwm_trace.device_guard import guard_cuda_device
from scripts.reinforcement_learning.rwm_trace.validate_go2_source_reset import (
    ResetRunnerOutput,
    ResetValidationRequest,
)


RUNNER_CONFIG_VERSION = "go2_perfect_sim_runner_config_v1"
RUNNER_CONTRACT_VERSION = "go2_perfect_sim_runner_v1"
SOFTWARE_VERSIONS_SCHEMA_VERSION = "go2_runner_software_versions_v1"
PRECISION_SCHEMA_VERSION = "go2_runner_precision_v1"
INDEPENDENT_RESETS_SCHEMA_VERSION = "go2_runner_independent_resets_v1"
RUNTIME_SCOPE = "pinned_v10_adapter"
V10_BASE_COMMIT = "31a03f5e89d5de9ca40bb5ad7d9c85205b67f9ab"
G0_TASK = "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert"
COMMAND_RESAMPLING_DISABLED_SECONDS = 1.0e9
ROLLOUT_EVENT_MODES = frozenset({"startup", "step", "interval"})
ACTION_MANAGER_TYPE = "mjlab.managers.action_manager.ActionManager"
G0_ACTION_CONFIG_TYPE = "mjlab.envs.mdp.actions.actions.JointPositionActionCfg"
G0_ACTION_RUNTIME_TYPE = "mjlab.envs.mdp.actions.actions.JointPositionAction"
G0_ACTUATOR_CONFIG_TYPE = (
    "mjlab.actuator.builtin_actuator.BuiltinPositionActuatorCfg"
)
G0_ACTUATOR_RUNTIME_TYPE = (
    "mjlab.actuator.builtin_actuator.BuiltinPositionActuator"
)
REQUIRED_SOFTWARE_DISTRIBUTIONS = {
    "mjlab": "mjlab",
    "mujoco": "mujoco",
    "mujoco_warp": "mujoco-warp",
    "warp": "warp-lang",
}


def guard_execution_device(device: str) -> dict[str, Any]:
    """Resolve a logical CUDA ordinal to a non-denied physical GPU."""

    value = str(device).strip().lower()
    if not value.startswith("cuda:"):
        raise ValueError(
            "Go2 simulator certification requires an explicit cuda:N device."
        )
    identity = guard_cuda_device(value).to_metadata()
    if not str(identity["visible_token"]).startswith("GPU-"):
        raise ValueError(
            "Formal certification requires CUDA_VISIBLE_DEVICES to contain "
            "a verified GPU UUID token."
        )
    return identity


def verify_device_identity(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
) -> None:
    required = (
        "logical_index",
        "visible_token",
        "cuda_visible_devices",
        "physical_index",
        "uuid",
        "pci_bus_id",
    )
    missing = [key for key in required if key not in expected]
    if missing:
        raise ValueError(f"Runner config device_identity is missing {missing}.")
    mismatches = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in required
        if expected.get(key) != actual.get(key)
    }
    if mismatches:
        raise ValueError(f"CUDA device identity changed: {mismatches}")


def sha256_artifact(path: str | Path) -> str:
    path = Path(path).expanduser().resolve()
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
        raise ValueError(f"Cannot hash empty artifact directory: {path}")
    for item in files:
        relative = item.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "little"))
        digest.update(relative)
        with item.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def parse_named_paths(values: list[str] | tuple[str, ...], label: str) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for raw in values:
        if "=" not in str(raw):
            raise ValueError(f"{label} entry must use NAME=PATH: {raw!r}")
        name, path_value = str(raw).split("=", 1)
        name = name.strip()
        if not name or name in result:
            raise ValueError(f"Invalid or duplicate {label} name: {name!r}")
        path = Path(path_value).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"{label} {name!r} does not exist: {path}")
        result[name] = path
    return result


def _git_output(root: str | Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(Path(root).expanduser().resolve()), *args],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def collect_source_provenance(
    *,
    repo_root: str | Path,
    expected_repo_commit: str,
    mjlab_repo_root: str | Path,
    expected_mjlab_commit: str,
    config_artifacts: Mapping[str, str | Path],
    asset_artifacts: Mapping[str, str | Path],
    source_artifacts: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Resolve commits and artifact hashes; any missing provenance is fatal."""

    if not config_artifacts:
        raise ValueError("At least one config artifact is required.")
    if not asset_artifacts:
        raise ValueError("At least one robot asset artifact is required.")
    if not source_artifacts:
        raise ValueError("At least one source artifact is required.")
    repo_root = Path(repo_root).expanduser().resolve()
    mjlab_repo_root = Path(mjlab_repo_root).expanduser().resolve()
    repo_commit = _git_output(repo_root, "rev-parse", "HEAD")
    mjlab_commit = _git_output(mjlab_repo_root, "rev-parse", "HEAD")
    if repo_commit != str(expected_repo_commit):
        raise ValueError(
            f"Repository commit mismatch: expected {expected_repo_commit}, got {repo_commit}"
        )
    if mjlab_commit != str(expected_mjlab_commit):
        raise ValueError(
            f"MJLab commit mismatch: expected {expected_mjlab_commit}, got {mjlab_commit}"
        )

    def artifact_rows(values: Mapping[str, str | Path]) -> dict[str, dict[str, str]]:
        rows = {}
        for name, raw_path in sorted(values.items()):
            path = Path(raw_path).expanduser().resolve()
            if not path.exists():
                raise FileNotFoundError(path)
            rows[str(name)] = {
                "path": str(path),
                "sha256": sha256_artifact(path),
                "kind": "directory" if path.is_dir() else "file",
            }
        return rows

    return {
        "repo": {
            "root": str(repo_root),
            "commit": repo_commit,
            "dirty": bool(_git_output(repo_root, "status", "--porcelain")),
        },
        "mjlab": {
            "root": str(mjlab_repo_root),
            "commit": mjlab_commit,
            "dirty": bool(_git_output(mjlab_repo_root, "status", "--porcelain")),
        },
        "config_artifacts": artifact_rows(config_artifacts),
        "asset_artifacts": artifact_rows(asset_artifacts),
        "source_artifacts": artifact_rows(source_artifacts),
    }


def verify_source_provenance(provenance: Mapping[str, Any]) -> None:
    for repo_name in ("repo", "mjlab"):
        row = provenance.get(repo_name)
        if not isinstance(row, Mapping):
            raise ValueError(f"Missing {repo_name} commit provenance.")
        actual = _git_output(str(row.get("root", "")), "rev-parse", "HEAD")
        if actual != row.get("commit"):
            raise ValueError(
                f"{repo_name} commit changed: expected {row.get('commit')}, got {actual}"
            )
    for group in ("config_artifacts", "asset_artifacts", "source_artifacts"):
        rows = provenance.get(group)
        if not isinstance(rows, Mapping) or not rows:
            raise ValueError(f"Missing non-empty {group} provenance.")
        for name, row in rows.items():
            if not isinstance(row, Mapping):
                raise ValueError(f"Malformed {group}.{name} provenance.")
            path = Path(str(row.get("path", ""))).expanduser().resolve()
            actual = sha256_artifact(path)
            if actual != row.get("sha256"):
                raise ValueError(
                    f"{group}.{name} hash changed: expected {row.get('sha256')}, got {actual}"
                )


def derive_runtime_scope_from_verified_provenance(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the only scope this adapter can claim after fail-closed checks."""

    verify_source_provenance(provenance)
    for repo_name in ("repo", "mjlab"):
        row = provenance.get(repo_name)
        if not isinstance(row, Mapping):
            raise ValueError(f"Missing {repo_name} provenance.")
        if row.get("dirty") is not False:
            raise ValueError(
                f"{repo_name} provenance must have been collected from a clean worktree."
            )
        root = str(row.get("root", ""))
        if _git_output(root, "status", "--porcelain"):
            raise ValueError(f"{repo_name} worktree became dirty after collection.")
        commit = str(row.get("commit", ""))
        if len(commit) != 40 or any(char not in "0123456789abcdef" for char in commit):
            raise ValueError(f"{repo_name} provenance has an invalid commit: {commit!r}")

    repo_root = str(provenance["repo"]["root"])
    proc = subprocess.run(
        [
            "git",
            "-C",
            repo_root,
            "merge-base",
            "--is-ancestor",
            V10_BASE_COMMIT,
            str(provenance["repo"]["commit"]),
        ],
        check=False,
        text=True,
        capture_output=True,
    )
    if proc.returncode != 0:
        raise ValueError(
            "Repository provenance is not descended from the pinned Go2 TRACE "
            f"V10 base commit {V10_BASE_COMMIT}."
        )
    return {
        "runtime_scope": RUNTIME_SCOPE,
        "derivation": "code_constant_plus_verified_clean_v10_provenance",
        "v10_base_commit": V10_BASE_COMMIT,
        "repo_commit": str(provenance["repo"]["commit"]),
        "mjlab_commit": str(provenance["mjlab"]["commit"]),
    }


def validate_runner_config(config: Mapping[str, Any]) -> None:
    if config.get("schema_version") != RUNNER_CONFIG_VERSION:
        raise ValueError(
            f"runner config schema must be {RUNNER_CONFIG_VERSION!r}"
        )
    if config.get("condition_id") != "g0":
        raise ValueError("This Phase-A perfect runner supports condition_id=g0 only.")
    if config.get("task") != G0_TASK:
        raise ValueError(f"g0 runner requires task={G0_TASK!r}.")
    if config.get("runtime_scope") != RUNTIME_SCOPE:
        raise ValueError(
            "This runner is only a pinned_v10_adapter and cannot claim "
            f"runtime_scope={config.get('runtime_scope')!r}."
        )
    if not str(config.get("device", "")).strip():
        raise ValueError("runner config requires an explicit device.")
    if not isinstance(config.get("device_identity"), Mapping):
        raise ValueError("runner config requires resolved physical device_identity.")
    if not isinstance(config.get("provenance"), Mapping):
        raise ValueError("runner config requires provenance.")
    burn_in_steps = config.get("history_burn_in_steps")
    if (
        isinstance(burn_in_steps, bool)
        or not isinstance(burn_in_steps, int)
        or burn_in_steps < 3
    ):
        raise ValueError("runner config requires history_burn_in_steps >= 3.")
    physics = config.get("physics")
    if not isinstance(physics, Mapping):
        raise ValueError("runner config requires collected physics provenance.")
    for key in (
        "physics_dt",
        "step_dt",
        "decimation",
        "event_terms",
        "rollout_event_modes_disabled",
        "command_resampling_time_range",
        "command_resampling_disabled",
        "actuator_groups",
        "actuator_delay_max_lag",
        "action_interface",
        "action_delay_disabled",
        "action_noise_disabled",
        "observation_interface",
        "observation_history_at_most_one",
        "observation_delay_disabled",
        "solver",
        "domain_randomization_disabled",
        "push_randomization_disabled",
        "observation_noise_disabled",
    ):
        if key not in physics:
            raise ValueError(f"runner config physics is missing {key!r}.")
    if physics.get("actuator_delay_max_lag") != 0:
        raise ValueError("runner config requires actuator_delay_max_lag=0.")
    if physics.get("action_delay_disabled") is not True:
        raise ValueError("runner config must prove action_delay_disabled=true.")
    if physics.get("action_noise_disabled") is not True:
        raise ValueError("runner config must prove action_noise_disabled=true.")
    action_interface = physics.get("action_interface")
    if not isinstance(action_interface, Mapping):
        raise ValueError("runner config requires action_interface evidence.")
    if action_interface.get("action_delay_disabled") is not True:
        raise ValueError("action_interface does not prove zero action delay.")
    if action_interface.get("action_noise_disabled") is not True:
        raise ValueError("action_interface does not prove zero action noise.")
    observation_interface = physics.get("observation_interface")
    if not isinstance(observation_interface, Mapping):
        raise ValueError("runner config requires observation_interface evidence.")
    if physics.get("observation_history_at_most_one") is not True:
        raise ValueError("runner config must prove observation history <= 1.")
    if physics.get("observation_delay_disabled") is not True:
        raise ValueError("runner config must prove observation delay=0.")
    if observation_interface.get("history_at_most_one") is not True:
        raise ValueError("observation_interface does not prove history <= 1.")
    if int(observation_interface.get("maximum_history_length", -1)) not in {0, 1}:
        raise ValueError("observation_interface maximum_history_length must be 0 or 1.")
    if observation_interface.get("delay_disabled") is not True:
        raise ValueError("observation_interface does not prove delay=0.")
    if int(observation_interface.get("maximum_delay_lag", -1)) != 0:
        raise ValueError("observation_interface maximum_delay_lag must be 0.")
    tolerance = float(config.get("determinism_tolerance", 1.0e-4))
    if not math.isfinite(tolerance) or tolerance <= 0.0 or tolerance > 1.0e-4:
        raise ValueError("determinism_tolerance must be in (0, 1e-4].")


def _configure_g0_env(num_envs: int, config: Mapping[str, Any]):
    """Create the target-matched normal-dog simulator with all DR disabled."""

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from flash_rl.envs.mjlab import configure_mjlab_randomization
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from src.tasks.rwm_velocity.mdp.extractors import Go2RWMExtractor

    configure_torch_backends()
    env_cfg = load_env_cfg(G0_TASK)
    env_cfg.scene.num_envs = int(num_envs)
    env_cfg.seed = int(config["seed"])
    if hasattr(env_cfg, "auto_reset"):
        env_cfg.auto_reset = True
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=False,
        use_push_randomization=False,
        use_observation_noise=False,
        randomization_preset="default",
        randomization_components=None,
        randomization_scale=1.0,
        payload_mass_range_kg=(0.0, 0.0),
        payload_position_body_m=(0.0, 0.0, 0.1),
        payload_box_size_m=(0.2, 0.12, 0.05),
        rr_calf_strength_range=(1.0, 1.0),
    )
    # The certification rollout uses a fixed caller-supplied command.
    twist_cfg = env_cfg.commands.get("twist")
    if twist_cfg is None or not hasattr(twist_cfg, "resampling_time_range"):
        raise RuntimeError("g0 config does not expose the required twist command.")
    twist_cfg.resampling_time_range = (
        COMMAND_RESAMPLING_DISABLED_SECONDS,
        COMMAND_RESAMPLING_DISABLED_SECONDS,
    )

    # Native reset events remain for source collection.  Any event that can
    # mutate an H-step rollout is rejected before environment construction.
    bad_events = {
        str(name): str(getattr(term_cfg, "mode", ""))
        for name, term_cfg in env_cfg.events.items()
        if str(getattr(term_cfg, "mode", "")) in ROLLOUT_EVENT_MODES
    }
    if bad_events:
        raise RuntimeError(
            f"g0 certification config contains rollout-changing events: {bad_events}"
        )
    noisy_groups = [
        str(name)
        for name, group_cfg in env_cfg.observations.items()
        if bool(getattr(group_cfg, "enable_corruption", False))
    ]
    if noisy_groups:
        raise RuntimeError(
            f"g0 certification observation corruption remains enabled: {noisy_groups}"
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=str(config["device"]))
    return env, Go2RWMExtractor(env.unwrapped), env_cfg


def _json_scalar(value: Any) -> Any:
    if isinstance(value, (str, bool, int, float)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError, RuntimeError):
            pass
    return str(value)


def _validate_software_versions(value: Mapping[str, Any]) -> None:
    required = (
        "schema_version",
        "python",
        "python_implementation",
        "torch",
        "torch_git_version",
        "cuda_build",
        *REQUIRED_SOFTWARE_DISTRIBUTIONS,
    )
    missing = [key for key in required if not str(value.get(key, "")).strip()]
    if missing:
        raise RuntimeError(f"Incomplete runner software_versions: {missing}.")
    if value.get("schema_version") != SOFTWARE_VERSIONS_SCHEMA_VERSION:
        raise RuntimeError("Unexpected runner software_versions schema.")


def collect_software_versions() -> dict[str, str]:
    """Collect installed versions without importing additional GPU packages."""

    versions = {
        "schema_version": SOFTWARE_VERSIONS_SCHEMA_VERSION,
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "torch": str(torch.__version__),
        "torch_git_version": str(getattr(torch.version, "git_version", "")),
        "cuda_build": str(getattr(torch.version, "cuda", "") or ""),
    }
    for key, distribution in REQUIRED_SOFTWARE_DISTRIBUTIONS.items():
        try:
            versions[key] = importlib_metadata.version(distribution)
        except importlib_metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"Required software distribution {distribution!r} is not installed."
            ) from exc
    _validate_software_versions(versions)
    return versions


def _tensor_dtype(value: Any, label: str) -> str:
    if not isinstance(value, torch.Tensor):
        raise RuntimeError(f"Cannot record precision: {label} is not a tensor.")
    return str(value.dtype)


def collect_precision_metadata(
    request: ResetValidationRequest,
    output: ResetRunnerOutput,
    physics: Mapping[str, Any],
) -> dict[str, Any]:
    model_hashes = physics.get("realized_model_hashes")
    if not isinstance(model_hashes, Mapping) or not model_hashes:
        raise RuntimeError("Cannot record model precision without realized model hashes.")
    model_tensor_dtypes = {
        str(name): str(row.get("dtype", ""))
        for name, row in model_hashes.items()
        if isinstance(row, Mapping)
    }
    if len(model_tensor_dtypes) != len(model_hashes) or any(
        not value for value in model_tensor_dtypes.values()
    ):
        raise RuntimeError("Realized model precision metadata is incomplete.")

    state_dtypes = {
        "expected_physical_states": _tensor_dtype(
            request.expected_physical_states, "expected_physical_states"
        ),
        "expected_rwm_states": _tensor_dtype(
            request.expected_rwm_states, "expected_rwm_states"
        ),
        "rollout_physical_states": _tensor_dtype(
            output.physical_states, "rollout_physical_states"
        ),
        "rollout_rwm_states": _tensor_dtype(
            output.rwm_states, "rollout_rwm_states"
        ),
    }
    action_dtypes = {
        "request_actions": _tensor_dtype(request.actions, "request_actions"),
        "rollout_action_histories": _tensor_dtype(
            output.action_histories, "rollout_action_histories"
        ),
    }
    required_state_action_dtype = "torch.float32"
    mismatched = {
        name: dtype
        for name, dtype in {**state_dtypes, **action_dtypes}.items()
        if dtype != required_state_action_dtype
    }
    if mismatched:
        raise RuntimeError(
            "Reset certification requires float32 state/action tensors; "
            f"mismatched={mismatched}."
        )
    result = {
        "schema_version": PRECISION_SCHEMA_VERSION,
        "required_state_action_dtype": required_state_action_dtype,
        "state": state_dtypes,
        "action": action_dtypes,
        "model": {
            "tensor_dtypes": model_tensor_dtypes,
            "unique_dtypes": sorted(set(model_tensor_dtypes.values())),
        },
        "torch_default_dtype": str(torch.get_default_dtype()),
        # PyTorch 2.9 raises merely from reading either the old or new TF32
        # global when a dependency has mixed both API families.  The concrete
        # state/action/model dtypes above are the stable precision evidence.
        "tf32_global_status": "not_queried_mixed_pytorch_api",
    }
    if result["schema_version"] != PRECISION_SCHEMA_VERSION:
        raise RuntimeError("Unexpected runner precision schema.")
    return result


def _runtime_tensor_hash(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    # MuJoCo-Warp may expose model fields through tensor-compatible wrappers
    # whose concrete class is not ``torch.Tensor``.  Prefer their native
    # detach/cpu path before asking Warp to reinterpret the object.
    if callable(getattr(value, "detach", None)) and callable(
        getattr(value, "cpu", None)
    ):
        value = value.detach()
    elif not isinstance(value, torch.Tensor):
        try:
            import warp as wp

            value = wp.to_torch(value)
        except (AttributeError, ImportError, TypeError, ValueError, RuntimeError):
            return None
    cpu = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(cpu.dtype).encode("utf-8"))
    digest.update(repr(tuple(cpu.shape)).encode("utf-8"))
    digest.update(bytes(cpu.view(torch.uint8).reshape(-1).tolist()))
    return {
        "sha256": digest.hexdigest(),
        "shape": list(cpu.shape),
        "dtype": str(cpu.dtype),
    }


def _qualified_type(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__qualname__}"


def _runtime_type_source(value: Any) -> dict[str, str]:
    source = inspect.getsourcefile(type(value))
    if source is None:
        raise RuntimeError(f"Cannot resolve source for runtime type {_qualified_type(value)}.")
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise RuntimeError(
            f"Runtime type source does not exist for {_qualified_type(value)}: {path}"
        )
    return {"path": str(path), "sha256": sha256_artifact(path)}


def _action_interface_provenance(env: Any, env_cfg: Any) -> dict[str, Any]:
    """Prove the pinned g0 action path has neither stochasticity nor delay."""

    manager = env.unwrapped.action_manager
    manager_type = _qualified_type(manager)
    if manager_type != ACTION_MANAGER_TYPE:
        raise RuntimeError(
            f"Unsupported action manager {manager_type!r}; expected {ACTION_MANAGER_TYPE!r}."
        )
    config_terms = getattr(env_cfg, "actions", None)
    if not isinstance(config_terms, Mapping):
        raise RuntimeError("g0 env config does not expose an actions mapping.")
    active_terms = [str(name) for name in manager.active_terms]
    configured_terms = [
        str(name) for name, term_cfg in config_terms.items() if term_cfg is not None
    ]
    if active_terms != ["joint_pos"] or configured_terms != active_terms:
        raise RuntimeError(
            "Pinned g0 requires exactly the joint_pos action term; "
            f"configured={configured_terms}, active={active_terms}."
        )
    if int(manager.total_action_dim) != 12:
        raise RuntimeError(
            f"Pinned Go2 action dimension must be 12, got {manager.total_action_dim}."
        )

    config_term = config_terms["joint_pos"]
    runtime_term = manager.get_term("joint_pos")
    config_type = _qualified_type(config_term)
    runtime_type = _qualified_type(runtime_term)
    if config_type != G0_ACTION_CONFIG_TYPE or runtime_type != G0_ACTION_RUNTIME_TYPE:
        raise RuntimeError(
            "Cannot prove a deterministic action path for unknown term types: "
            f"config={config_type!r}, runtime={runtime_type!r}."
        )
    if getattr(runtime_term, "cfg", None) is not config_term:
        raise RuntimeError("Runtime joint_pos term is not bound to the collected config.")
    if int(runtime_term.action_dim) != 12:
        raise RuntimeError(
            f"Pinned Go2 joint_pos action dimension must be 12, got {runtime_term.action_dim}."
        )

    config_fields = sorted(str(name) for name in vars(config_term))
    stochastic_fields = [
        name
        for name in config_fields
        if any(
            token in name.lower()
            for token in ("noise", "random", "stochastic", "delay", "latency")
        )
    ]
    if stochastic_fields:
        raise RuntimeError(
            "Pinned deterministic joint_pos config contains unverified stochastic "
            f"or delayed fields: {stochastic_fields}."
        )

    raw_action = getattr(runtime_term, "raw_action", None)
    processed_action = getattr(runtime_term, "_processed_actions", None)
    if not isinstance(raw_action, torch.Tensor) or not isinstance(
        processed_action, torch.Tensor
    ):
        raise RuntimeError("joint_pos term does not expose raw/processed action tensors.")
    expected_shape = (int(env.unwrapped.num_envs), 12)
    if (
        tuple(raw_action.shape) != expected_shape
        or tuple(processed_action.shape) != expected_shape
    ):
        raise RuntimeError(
            "joint_pos runtime tensors have unexpected shapes: "
            f"raw={tuple(raw_action.shape)}, processed={tuple(processed_action.shape)}, "
            f"expected={expected_shape}."
        )

    robot = env.unwrapped.scene["robot"]
    target_ids = getattr(runtime_term, "target_ids", None)
    encoder_bias = getattr(robot.data, "encoder_bias", None)
    if not isinstance(target_ids, torch.Tensor) or not isinstance(
        encoder_bias, torch.Tensor
    ):
        raise RuntimeError("Cannot prove zero action noise without target ids/encoder bias.")
    selected_bias = encoder_bias[:, target_ids]
    if selected_bias.numel() == 0 or not bool(torch.isfinite(selected_bias).all()):
        raise RuntimeError("Runtime encoder bias is empty or non-finite.")
    encoder_bias_max_abs = float(selected_bias.detach().abs().max().cpu())
    if encoder_bias_max_abs != 0.0:
        raise RuntimeError(
            "Perfect g0 requires exactly zero encoder bias; "
            f"max_abs={encoder_bias_max_abs}."
        )

    return {
        "manager_type": manager_type,
        "manager_source": _runtime_type_source(manager),
        "active_terms": active_terms,
        "total_action_dim": int(manager.total_action_dim),
        "terms": [
            {
                "name": "joint_pos",
                "config_type": config_type,
                "runtime_type": runtime_type,
                "config_source": _runtime_type_source(config_term),
                "runtime_source": _runtime_type_source(runtime_term),
                "config_fields": config_fields,
                "action_dim": int(runtime_term.action_dim),
            }
        ],
        "processing_path": (
            "request_action->ActionManager.process_action->"
            "JointPositionAction.scale_offset->zero_encoder_bias->actuator_target"
        ),
        "runner_action_transform": "direct_request_action_no_noise_no_delay",
        "encoder_bias_max_abs": encoder_bias_max_abs,
        "action_noise_disabled": True,
        "action_delay_disabled": True,
    }


def _observation_interface_provenance(env: Any, env_cfg: Any) -> dict[str, Any]:
    """Prove all active observation terms have history <= 1 and zero delay."""

    manager = env.unwrapped.observation_manager
    config_groups = getattr(env_cfg, "observations", None)
    if not isinstance(config_groups, Mapping) or not config_groups:
        raise RuntimeError("g0 env config does not expose observation groups.")
    runtime_terms = getattr(manager, "active_terms", None)
    if not isinstance(runtime_terms, Mapping):
        raise RuntimeError("Observation manager does not expose active terms.")
    runtime_groups = getattr(manager, "cfg", None)
    if not isinstance(runtime_groups, Mapping):
        raise RuntimeError("Observation manager does not expose runtime group config.")
    delay_buffers = getattr(manager, "_group_obs_term_delay_buffer", None)
    if not isinstance(delay_buffers, Mapping):
        raise RuntimeError("Cannot inspect runtime observation delay buffers.")
    history_buffers = getattr(manager, "_group_obs_term_history_buffer", None)
    if not isinstance(history_buffers, Mapping):
        raise RuntimeError("Cannot inspect runtime observation history buffers.")

    group_rows = []
    maximum_history = 0
    maximum_delay = 0
    for group_name, group_cfg in config_groups.items():
        group_name = str(group_name)
        if bool(getattr(group_cfg, "enable_corruption", False)):
            raise RuntimeError(
                f"Observation corruption remains enabled for group {group_name!r}."
            )
        runtime_group_cfg = runtime_groups.get(group_name)
        if runtime_group_cfg is None or bool(
            getattr(runtime_group_cfg, "enable_corruption", True)
        ):
            raise RuntimeError(
                f"Runtime observation corruption is not proven disabled for "
                f"group {group_name!r}."
            )
        configured_terms = getattr(group_cfg, "terms", None)
        active_names = [str(name) for name in runtime_terms.get(group_name, ())]
        if not isinstance(configured_terms, Mapping) or not configured_terms:
            raise RuntimeError(
                f"Observation group {group_name!r} has no inspectable terms."
            )
        configured_names = [
            str(name) for name, term_cfg in configured_terms.items() if term_cfg is not None
        ]
        if configured_names != active_names:
            raise RuntimeError(
                f"Observation terms changed for {group_name!r}: "
                f"configured={configured_names}, active={active_names}."
            )
        runtime_delay_names = sorted(
            str(name) for name in delay_buffers.get(group_name, {})
        )
        if runtime_delay_names:
            raise RuntimeError(
                f"Runtime observation delay buffers exist for {group_name!r}: "
                f"{runtime_delay_names}."
            )

        term_rows = []
        expected_history_names = []
        for term_name in active_names:
            term_cfg = manager.get_term_cfg(group_name, term_name)
            history_length = int(getattr(term_cfg, "history_length", -1))
            delay_min_lag = int(getattr(term_cfg, "delay_min_lag", -1))
            delay_max_lag = int(getattr(term_cfg, "delay_max_lag", -1))
            if history_length < 0 or history_length > 1:
                raise RuntimeError(
                    f"Observation {group_name}.{term_name} history_length must be "
                    f"0 or 1, got {history_length}."
                )
            if delay_min_lag != 0 or delay_max_lag != 0:
                raise RuntimeError(
                    f"Observation {group_name}.{term_name} must have zero delay, got "
                    f"[{delay_min_lag}, {delay_max_lag}]."
                )
            maximum_history = max(maximum_history, history_length)
            maximum_delay = max(maximum_delay, delay_min_lag, delay_max_lag)
            if history_length > 0:
                expected_history_names.append(term_name)
            term_rows.append(
                {
                    "name": term_name,
                    "config_type": _qualified_type(term_cfg),
                    "history_length": history_length,
                    "delay_min_lag": delay_min_lag,
                    "delay_max_lag": delay_max_lag,
                    "noise_configured": getattr(term_cfg, "noise", None) is not None,
                    "noise_active": False,
                }
            )
        runtime_history_names = sorted(
            str(name) for name in history_buffers.get(group_name, {})
        )
        if runtime_history_names != sorted(expected_history_names):
            raise RuntimeError(
                f"Runtime observation history buffers differ for {group_name!r}: "
                f"expected={sorted(expected_history_names)}, "
                f"actual={runtime_history_names}."
            )
        group_rows.append(
            {
                "name": group_name,
                "config_type": _qualified_type(group_cfg),
                "enable_corruption": False,
                "runtime_enable_corruption": False,
                "configured_history_length": getattr(
                    group_cfg, "history_length", None
                ),
                "runtime_history_buffer_terms": runtime_history_names,
                "terms": term_rows,
            }
        )
    if set(runtime_terms) != {str(name) for name in config_groups}:
        raise RuntimeError(
            "Runtime/config observation group sets differ: "
            f"runtime={sorted(runtime_terms)}, config={sorted(config_groups)}."
        )
    return {
        "manager_type": _qualified_type(manager),
        "manager_source": _runtime_type_source(manager),
        "groups": group_rows,
        "maximum_history_length": maximum_history,
        "maximum_delay_lag": maximum_delay,
        "history_at_most_one": True,
        "delay_disabled": True,
        "corruption_disabled": True,
    }


def physics_provenance(env: Any, env_cfg: Any) -> dict[str, Any]:
    gravity = None
    model = getattr(env.unwrapped.sim, "model", None)
    option = getattr(model, "opt", None)
    if option is not None and getattr(option, "gravity", None) is not None:
        gravity_value = getattr(option, "gravity")
        gravity = torch.as_tensor(gravity_value).detach().cpu().reshape(-1).tolist()

    event_terms = {
        str(mode): sorted(str(name) for name in names)
        for mode, names in env.unwrapped.event_manager.active_terms.items()
    }
    rollout_events = {
        mode: names
        for mode, names in event_terms.items()
        if mode in ROLLOUT_EVENT_MODES and names
    }
    if rollout_events:
        raise RuntimeError(
            f"Runtime event manager contains rollout-changing events: {rollout_events}"
        )

    command_term = env.unwrapped.command_manager.get_term("twist")
    command_range = tuple(
        float(value) for value in command_term.cfg.resampling_time_range
    )
    command_resampling_disabled = (
        len(command_range) == 2
        and min(command_range) >= COMMAND_RESAMPLING_DISABLED_SECONDS
    )
    if not command_resampling_disabled:
        raise RuntimeError(f"Command resampling is not disabled: {command_range}")

    robot = env.unwrapped.scene["robot"]
    actuator_groups = []
    maximum_delay_lag = 0
    for actuator in robot.actuators:
        cfg = actuator.cfg
        runtime_type = _qualified_type(actuator)
        config_type = _qualified_type(cfg)
        if (
            runtime_type != G0_ACTUATOR_RUNTIME_TYPE
            or config_type != G0_ACTUATOR_CONFIG_TYPE
        ):
            raise RuntimeError(
                "Cannot prove a noise-free, delay-free actuator path for unknown "
                f"types: runtime={runtime_type!r}, config={config_type!r}."
            )
        config_fields = sorted(str(name) for name in vars(cfg))
        stochastic_fields = [
            name
            for name in config_fields
            if any(
                token in name.lower()
                for token in ("noise", "random", "stochastic", "delay", "latency")
            )
        ]
        if stochastic_fields:
            raise RuntimeError(
                "Pinned g0 actuator config contains unverified stochastic or delayed "
                f"fields: {stochastic_fields}."
            )
        minimum_lag = int(getattr(cfg, "delay_min_lag", 0))
        maximum_lag = int(getattr(cfg, "delay_max_lag", 0))
        maximum_delay_lag = max(maximum_delay_lag, maximum_lag)
        actuator_groups.append(
            {
                "runtime_type": runtime_type,
                "config_type": config_type,
                "runtime_source": _runtime_type_source(actuator),
                "config_source": _runtime_type_source(cfg),
                "config_fields": config_fields,
                "delay_min_lag": minimum_lag,
                "delay_max_lag": maximum_lag,
            }
        )
    delayed_types = [
        row
        for row in actuator_groups
        if "Delayed" in row["runtime_type"] or "Delayed" in row["config_type"]
    ]
    if maximum_delay_lag != 0 or delayed_types:
        raise RuntimeError(
            "Perfect g0 requires zero actuator delay; "
            f"max_lag={maximum_delay_lag}, delayed_types={delayed_types}"
        )
    action_interface = _action_interface_provenance(env, env_cfg)
    observation_interface = _observation_interface_provenance(env, env_cfg)

    mujoco_cfg = env_cfg.sim.mujoco
    solver = {
        key: _json_scalar(getattr(mujoco_cfg, key))
        for key in (
            "integrator",
            "impratio",
            "cone",
            "jacobian",
            "solver",
            "iterations",
            "tolerance",
            "ls_iterations",
            "ls_tolerance",
            "ccd_iterations",
            "multiccd",
        )
    }
    model_hashes = {
        name: hashed
        for name in (
            "body_mass",
            "body_ipos",
            "body_inertia",
            "body_iquat",
            "geom_friction",
            "actuator_gainprm",
            "actuator_biasprm",
            "actuator_forcerange",
        )
        if (
            hashed := _runtime_tensor_hash(
                getattr(env.unwrapped.sim.model, name, None)
            )
        )
        is not None
    }
    if not model_hashes:
        raise RuntimeError("Could not hash any realized simulator model tensors.")
    return {
        "physics_dt": float(env.unwrapped.physics_dt),
        "step_dt": float(env.unwrapped.step_dt),
        "decimation": int(env.unwrapped.cfg.decimation),
        "gravity": gravity,
        "num_envs": int(env.unwrapped.num_envs),
        "event_terms": event_terms,
        "rollout_event_modes_disabled": True,
        "command_resampling_time_range": list(command_range),
        "command_resampling_disabled": command_resampling_disabled,
        "actuator_groups": actuator_groups,
        "actuator_delay_max_lag": maximum_delay_lag,
        "action_interface": action_interface,
        "action_delay_disabled": True,
        "action_noise_disabled": True,
        "observation_interface": observation_interface,
        "observation_history_at_most_one": True,
        "observation_delay_disabled": True,
        "solver": solver,
        "realized_model_hashes": model_hashes,
        "domain_randomization_disabled": True,
        "push_randomization_disabled": True,
        "observation_noise_disabled": True,
        "payload_mass_kg": 0.0,
        "rr_calf_strength": 1.0,
        "task": G0_TASK,
        "env_cfg_type": type(env_cfg).__qualname__,
    }


def _verify_physics(expected: Mapping[str, Any], actual: Mapping[str, Any]) -> None:
    for key in (
        "physics_dt",
        "step_dt",
        "decimation",
        "event_terms",
        "rollout_event_modes_disabled",
        "command_resampling_time_range",
        "command_resampling_disabled",
        "actuator_groups",
        "actuator_delay_max_lag",
        "action_interface",
        "action_delay_disabled",
        "action_noise_disabled",
        "observation_interface",
        "observation_history_at_most_one",
        "observation_delay_disabled",
        "solver",
        "realized_model_hashes",
        "domain_randomization_disabled",
        "push_randomization_disabled",
        "observation_noise_disabled",
        "payload_mass_kg",
        "rr_calf_strength",
        "task",
    ):
        if expected.get(key) != actual.get(key):
            raise ValueError(
                f"Perfect-simulator physics mismatch for {key}: "
                f"expected {expected.get(key)!r}, got {actual.get(key)!r}"
            )
    if expected.get("gravity") is not None and expected.get("gravity") != actual.get(
        "gravity"
    ):
        raise ValueError(
            f"Perfect-simulator gravity mismatch: {expected.get('gravity')} != "
            f"{actual.get('gravity')}"
        )


def _force_commands(env: Any, command: torch.Tensor, env_ids: torch.Tensor) -> None:
    term = env.command_manager.get_term("twist")
    command = command.to(device=env.device, dtype=torch.float32)
    if hasattr(term, "vel_command_b"):
        term.vel_command_b[env_ids] = command
    if hasattr(term, "is_standing_env"):
        term.is_standing_env[env_ids] = (
            torch.linalg.norm(command, dim=-1) < 1.0e-8
        )
    if hasattr(term, "is_heading_env"):
        term.is_heading_env[env_ids] = False


def _capture_point(env: Any, extractor: Any, env_ids: torch.Tensor) -> dict[str, torch.Tensor]:
    from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
        capture_go2_simulator_snapshot,
    )

    snapshot = capture_go2_simulator_snapshot(env, env_ids)
    return {
        "physical": torch.cat(
            [
                snapshot["root_state_local"],
                snapshot["joint_position"],
                snapshot["joint_velocity"],
            ],
            dim=-1,
        ).detach(),
        "rwm": extractor.extract_state()[env_ids].detach().clone(),
        "action": snapshot["action"].detach(),
        "prev_action": snapshot["prev_action"].detach(),
        "prev_prev_action": snapshot["prev_prev_action"].detach(),
        "command": snapshot["command"].detach(),
    }


def _single_manual_rollout(
    env: Any,
    extractor: Any,
    request: ResetValidationRequest,
    env_ids: torch.Tensor,
) -> ResetRunnerOutput:
    from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
        restore_go2_simulator_snapshot,
        step_without_automatic_reset,
    )

    snapshot = {
        key: (
            value.to(env.device)
            if isinstance(value, torch.Tensor)
            else value
        )
        for key, value in request.snapshot.items()
    }
    restore_go2_simulator_snapshot(env, snapshot, env_ids)
    _force_commands(
        env,
        request.expected_commands[:, 0].to(env.device),
        env_ids,
    )
    points = [_capture_point(env, extractor, env_ids)]
    contacts = []
    terminations = []
    rewards = []
    for step in range(request.horizon):
        _force_commands(
            env,
            request.transition_commands[:, step].to(env.device),
            env_ids,
        )
        _, reward, terminated, timeout, _ = step_without_automatic_reset(
            env,
            request.actions[:, step].to(env.device),
        )
        contacts.append(extractor.extract_contact()[env_ids].detach().clone())
        terminations.append(
            (terminated[env_ids].bool() | timeout[env_ids].bool()).detach().clone()
        )
        rewards.append(reward[env_ids].detach().clone())
        _force_commands(
            env,
            request.expected_commands[:, step + 1].to(env.device),
            env_ids,
        )
        points.append(_capture_point(env, extractor, env_ids))

    def stack_point(name: str) -> torch.Tensor:
        return torch.stack([point[name] for point in points], dim=1).cpu()

    return ResetRunnerOutput(
        physical_states=stack_point("physical"),
        rwm_states=stack_point("rwm"),
        action_histories=stack_point("action"),
        prev_action_histories=stack_point("prev_action"),
        prev_prev_action_histories=stack_point("prev_prev_action"),
        commands=stack_point("command"),
        contacts=torch.stack(contacts, dim=1).cpu(),
        terminations=torch.stack(terminations, dim=1).cpu(),
        rewards=torch.stack(rewards, dim=1).cpu(),
    )


def _run_clean_manual_replay(
    env: Any,
    env_cfg: Any,
    extractor: Any,
    request: ResetValidationRequest,
    env_ids: torch.Tensor,
    *,
    seed: int,
    replay_index: int,
    expected_physics: Mapping[str, Any],
    reset_fn: Callable[..., Any],
    physics_fn: Callable[[Any, Any], dict[str, Any]] = physics_provenance,
    verify_physics_fn: Callable[[Mapping[str, Any], Mapping[str, Any]], None] = _verify_physics,
    rollout_fn: Callable[..., ResetRunnerOutput] = _single_manual_rollout,
) -> tuple[ResetRunnerOutput, dict[str, Any], dict[str, Any]]:
    """Public native reset, verify physics, then manually restore one replay."""

    reset_result = reset_fn(env, seed=int(seed))
    provenance = getattr(reset_result, "provenance", None)
    if getattr(reset_result, "reset_kind", None) != "simulator_native_random":
        raise RuntimeError("Independent replay cleanup did not use native random reset.")
    if getattr(reset_result, "seed", None) != int(seed):
        raise RuntimeError("Independent replay cleanup did not preserve the requested seed.")
    if not isinstance(provenance, Mapping):
        raise RuntimeError("Independent replay reset returned no provenance.")
    if provenance.get("native_reset_calls") != 1:
        raise RuntimeError("Each independent replay must call public env.reset exactly once.")
    if provenance.get("reset_scope") != "all_envs":
        raise RuntimeError("Independent replay cleanup must reset every dedicated slot.")
    if int(provenance.get("batch_size", -1)) != len(request.windows):
        raise RuntimeError("Independent replay reset batch size does not match request.")
    snapshot_sha256 = str(provenance.get("realized_snapshot_sha256", ""))
    if len(snapshot_sha256) != 64:
        raise RuntimeError("Independent replay reset lacks realized snapshot hash evidence.")

    actual_physics = physics_fn(env, env_cfg)
    verify_physics_fn(expected_physics, actual_physics)
    output = rollout_fn(env.unwrapped, extractor, request, env_ids)
    evidence = {
        "replay_index": int(replay_index),
        "seed": int(seed),
        "reset_kind": "simulator_native_random",
        "reset_scope": "all_envs",
        "public_native_reset_calls": 1,
        "manual_restore_after_native_reset": True,
        "batch_size": int(provenance["batch_size"]),
        "native_reset_snapshot_sha256": snapshot_sha256,
    }
    return output, evidence, actual_physics


def _independent_reset_metadata(
    records: list[Mapping[str, Any]],
    *,
    seed: int,
) -> dict[str, Any]:
    if len(records) != 2:
        raise RuntimeError("Double replay requires exactly two independent resets.")
    for expected_index, row in enumerate(records, start=1):
        if row.get("replay_index") != expected_index:
            raise RuntimeError("Independent reset replay indices are not ordered.")
        if row.get("seed") != int(seed):
            raise RuntimeError("Independent resets did not use the same seed.")
        if row.get("public_native_reset_calls") != 1:
            raise RuntimeError("Each replay must contain exactly one public native reset.")
        if row.get("manual_restore_after_native_reset") is not True:
            raise RuntimeError("Manual restore must follow every independent native reset.")
    return {
        "schema_version": INDEPENDENT_RESETS_SCHEMA_VERSION,
        "replay_count": 2,
        "same_seed": True,
        "seed": int(seed),
        "public_native_reset_calls_total": 2,
        "clean_start_before_every_replay": True,
        "records": [dict(row) for row in records],
    }


def _validate_runner_artifact_metadata(metadata: Mapping[str, Any]) -> None:
    _validate_software_versions(metadata.get("software_versions", {}))
    precision = metadata.get("precision")
    if not isinstance(precision, Mapping):
        raise RuntimeError("Runner artifact lacks precision metadata.")
    if precision.get("schema_version") != PRECISION_SCHEMA_VERSION:
        raise RuntimeError("Runner artifact has an invalid precision schema.")
    if precision.get("required_state_action_dtype") != "torch.float32":
        raise RuntimeError("Runner artifact does not enforce float32 state/action.")
    native_resets = metadata.get("independent_native_resets")
    if not isinstance(native_resets, Mapping):
        raise RuntimeError("Runner artifact lacks independent native reset evidence.")
    if native_resets.get("schema_version") != INDEPENDENT_RESETS_SCHEMA_VERSION:
        raise RuntimeError("Runner artifact has an invalid independent-reset schema.")
    if native_resets.get("public_native_reset_calls_total") != 2:
        raise RuntimeError("Runner artifact must record exactly two public native resets.")
    if native_resets.get("clean_start_before_every_replay") is not True:
        raise RuntimeError("Runner artifact does not prove clean replay starts.")
    burn_in_steps = metadata.get("history_burn_in_steps")
    if isinstance(burn_in_steps, bool) or not isinstance(burn_in_steps, int):
        raise RuntimeError("Runner artifact lacks integer history_burn_in_steps.")
    if burn_in_steps < 3:
        raise RuntimeError("Runner artifact requires history_burn_in_steps >= 3.")
    physics = metadata.get("physics")
    if not isinstance(physics, Mapping):
        raise RuntimeError("Runner artifact lacks physics provenance.")
    if physics.get("observation_history_at_most_one") is not True:
        raise RuntimeError("Runner artifact does not prove observation history <= 1.")
    if physics.get("observation_delay_disabled") is not True:
        raise RuntimeError("Runner artifact does not prove observation delay=0.")


def _assert_deterministic(
    first: ResetRunnerOutput,
    second: ResetRunnerOutput,
    tolerance: float,
) -> dict[str, Any]:
    continuous_names = (
        "physical_states",
        "rwm_states",
        "action_histories",
        "prev_action_histories",
        "prev_prev_action_histories",
        "commands",
        "rewards",
    )
    continuous_linf = {}
    for name in continuous_names:
        left = getattr(first, name)
        right = getattr(second, name)
        if left is None or right is None:
            raise ValueError(f"Double replay is missing {name}.")
        error = float(torch.max(torch.abs(left.float() - right.float())))
        continuous_linf[name] = error
        if not math.isfinite(error) or error > tolerance:
            raise RuntimeError(
                f"Double replay is nondeterministic for {name}: {error} > {tolerance}"
            )
    exact = {}
    for name in ("contacts", "terminations"):
        matched = bool(
            torch.equal(getattr(first, name).bool(), getattr(second, name).bool())
        )
        exact[name] = matched
        if not matched:
            raise RuntimeError(f"Double replay is nondeterministic for {name}.")
    return {
        "repetitions": 2,
        "passed": True,
        "tolerance": float(tolerance),
        "continuous_linf": continuous_linf,
        "exact_match": exact,
    }


def run_perfect_simulator(
    request: ResetValidationRequest,
    runner_config: Mapping[str, Any],
) -> Mapping[str, Any]:
    """Validator callback for the condition-matched g0 perfect simulator."""

    validate_runner_config(runner_config)
    software_versions = collect_software_versions()
    runtime_scope_evidence = derive_runtime_scope_from_verified_provenance(
        runner_config["provenance"]
    )
    device_identity = guard_execution_device(str(runner_config["device"]))
    verify_device_identity(runner_config["device_identity"], device_identity)
    env = None
    try:
        env, extractor, env_cfg = _configure_g0_env(
            len(request.windows),
            runner_config,
        )
        from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
            reset_native_random,
        )

        env_ids = torch.arange(
            len(request.windows),
            device=env.unwrapped.device,
            dtype=torch.long,
        )
        seed = int(runner_config["seed"])
        first, first_reset, actual_physics = _run_clean_manual_replay(
            env,
            env_cfg,
            extractor,
            request,
            env_ids,
            seed=seed,
            replay_index=1,
            expected_physics=runner_config["physics"],
            reset_fn=reset_native_random,
        )
        second, second_reset, _ = _run_clean_manual_replay(
            env,
            env_cfg,
            extractor,
            request,
            env_ids,
            seed=seed,
            replay_index=2,
            expected_physics=runner_config["physics"],
            reset_fn=reset_native_random,
        )
        independent_native_resets = _independent_reset_metadata(
            [first_reset, second_reset],
            seed=seed,
        )
        determinism = _assert_deterministic(
            first,
            second,
            float(runner_config.get("determinism_tolerance", 1.0e-4)),
        )
        precision = collect_precision_metadata(request, first, actual_physics)
        metadata = {
            "runner_contract_version": RUNNER_CONTRACT_VERSION,
            "condition_id": "g0",
            "task": G0_TASK,
            "manual_snapshot_restore": True,
            "automatic_reset_during_rollout": False,
            "snapshot_action_history_overwritten": False,
            "physics": actual_physics,
            "software_versions": software_versions,
            "precision": precision,
            "device_identity": device_identity,
            "determinism": determinism,
            "independent_native_resets": independent_native_resets,
            "history_burn_in_steps": int(runner_config["history_burn_in_steps"]),
            "provenance_verified": True,
            "runtime_scope": RUNTIME_SCOPE,
            "runtime_scope_evidence": runtime_scope_evidence,
        }
        _validate_runner_artifact_metadata(metadata)
        # Return a structural mapping so the standalone CLI works even when
        # its validator dataclass is loaded once as ``__main__`` and once via
        # the package-qualified runner import.
        return {
            "physical_states": first.physical_states,
            "rwm_states": first.rwm_states,
            "action_histories": first.action_histories,
            "prev_action_histories": first.prev_action_histories,
            "prev_prev_action_histories": first.prev_prev_action_histories,
            "commands": first.commands,
            "contacts": first.contacts,
            "terminations": first.terminations,
            "rewards": first.rewards,
            "metadata": metadata,
        }
    finally:
        if env is not None:
            env.close()
