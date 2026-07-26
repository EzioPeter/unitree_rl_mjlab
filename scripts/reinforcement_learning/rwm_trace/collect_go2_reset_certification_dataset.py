#!/usr/bin/env python3
"""Collect a minimal condition-matched g0 V1 reset-certification dataset."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


COLLECTOR_FORMAT_VERSION = "go2_reset_certification_dataset_v1"
RUNNER_CONFIG_VERSION = "go2_perfect_sim_runner_config_v1"
G0_TASK = "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert"
RUNTIME_SCOPE = "v13_canonical"
HISTORY_BURN_IN_STEPS = 3


def _guard_requested_device(device: str) -> dict[str, Any]:
    """Load the stdlib-only guard before importing torch or MJLab."""

    if not str(device).strip().lower().startswith("cuda:"):
        raise ValueError("Formal simulator collection requires explicit cuda:N.")
    path = Path(__file__).resolve().with_name("device_guard.py")
    spec = importlib.util.spec_from_file_location(
        "_go2_trace_pre_torch_device_guard",
        path,
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load device guard from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    identity = module.guard_cuda_device(device).to_metadata()
    if not str(identity["visible_token"]).startswith("GPU-"):
        raise ValueError(
            "Formal collection requires CUDA_VISIBLE_DEVICES to use a verified "
            "GPU UUID token, not an unset/numeric mapping."
        )
    return identity


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-dataset", required=True)
    parser.add_argument("--output-runner-config", required=True)
    parser.add_argument("--condition-id", choices=("g0",), default="g0")
    parser.add_argument("--task", choices=(G0_TASK,), default=G0_TASK)
    parser.add_argument("--device", required=True)
    parser.add_argument("--seed", type=int, default=9102)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--action-std", type=float, default=0.02)
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=(0.2, 0.0, 0.0),
        metavar=("VX", "VY", "YAW"),
    )
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--expected-repo-commit", required=True)
    parser.add_argument("--mjlab-repo-root", required=True)
    parser.add_argument("--expected-mjlab-commit", required=True)
    parser.add_argument(
        "--config-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--asset-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument(
        "--source-artifact",
        action="append",
        default=[],
        metavar="NAME=PATH",
    )
    parser.add_argument("--determinism-tolerance", type=float, default=1.0e-4)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def validate_args(args: argparse.Namespace) -> None:
    if args.condition_id != "g0" or args.task != G0_TASK:
        raise ValueError("This collector supports the target-matched g0 task only.")
    if args.num_envs < 1 or args.horizon < 1:
        raise ValueError("num-envs and horizon must be positive.")
    if not 0.0 < float(args.action_std) <= 0.1:
        raise ValueError("action-std must be in (0, 0.1] for certification data.")
    if not 0.0 < float(args.determinism_tolerance) <= 1.0e-4:
        raise ValueError("determinism-tolerance must be in (0, 1e-4].")
    if not args.config_artifact:
        raise ValueError("At least one --config-artifact NAME=PATH is required.")
    if not args.asset_artifact:
        raise ValueError("At least one --asset-artifact NAME=PATH is required.")


def _default_source_artifacts() -> dict[str, Path]:
    root = Path(__file__).resolve().parent
    return {
        "collector": Path(__file__).resolve(),
        "runner": root / "go2_perfect_sim_runner.py",
        "validator": root / "validate_go2_source_reset.py",
        "manual_reset": root / "simulator_reset.py",
    }


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: Any) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _cpu_actions(
    *,
    point: int,
    num_envs: int,
    action_dim: int,
    std: float,
    seed: int,
) -> torch.Tensor:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 104729 * int(point))
    return torch.clamp(
        torch.randn(num_envs, action_dim, generator=generator) * float(std),
        -1.0,
        1.0,
    )


def collect_dataset(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    validate_args(args)
    device_identity = _guard_requested_device(str(args.device))

    import torch

    from scripts.reinforcement_learning.rwm_trace.condition_registry import (
        canonical_condition_metadata,
    )
    from scripts.reinforcement_learning.rwm_trace.go2_perfect_sim_runner import (
        _configure_g0_env,
        _force_commands,
        collect_source_provenance,
        parse_named_paths,
        physics_provenance,
    )
    from scripts.reinforcement_learning.rwm_trace.validate_go2_source_reset import (
        preflight_dataset,
        select_source_windows,
    )

    config_artifacts = parse_named_paths(args.config_artifact, "config-artifact")
    asset_artifacts = parse_named_paths(args.asset_artifact, "asset-artifact")
    source_artifacts = _default_source_artifacts()
    source_artifacts.update(
        parse_named_paths(args.source_artifact, "source-artifact")
    )
    provenance = collect_source_provenance(
        repo_root=args.repo_root,
        expected_repo_commit=args.expected_repo_commit,
        mjlab_repo_root=args.mjlab_repo_root,
        expected_mjlab_commit=args.expected_mjlab_commit,
        config_artifacts=config_artifacts,
        asset_artifacts=asset_artifacts,
        source_artifacts=source_artifacts,
    )

    from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
        capture_go2_simulator_snapshot,
        reset_native_random,
        step_without_automatic_reset,
    )
    from src.tasks.rwm_velocity.mdp.extractors import make_go2_policy_obs

    env = None
    try:
        runner_seed_config = {
            "seed": int(args.seed),
            "device": str(args.device),
        }
        env, extractor, env_cfg = _configure_g0_env(
            int(args.num_envs),
            runner_seed_config,
        )
        native_reset = reset_native_random(env, seed=int(args.seed))
        del native_reset
        physics = physics_provenance(env, env_cfg)
        env_ids = torch.arange(
            int(args.num_envs),
            device=env.unwrapped.device,
            dtype=torch.long,
        )
        commands = torch.tensor(
            args.command,
            device=env.unwrapped.device,
            dtype=torch.float32,
        ).repeat(int(args.num_envs), 1)
        _force_commands(env.unwrapped, commands, env_ids)
        episode_ids = torch.arange(
            int(args.num_envs),
            device=env.unwrapped.device,
            dtype=torch.long,
        )
        last_actions = torch.zeros(
            int(args.num_envs),
            int(env.single_action_space.shape[0]),
            device=env.unwrapped.device,
        )
        for burn_in_step in range(HISTORY_BURN_IN_STEPS):
            _force_commands(env.unwrapped, commands, env_ids)
            burn_in_action = _cpu_actions(
                point=burn_in_step,
                num_envs=int(args.num_envs),
                action_dim=int(env.single_action_space.shape[0]),
                std=float(args.action_std),
                seed=int(args.seed),
            ).to(env.unwrapped.device)
            _, _, terminated, timeout, _ = step_without_automatic_reset(
                env.unwrapped,
                burn_in_action,
            )
            burn_in_done = terminated.bool() | timeout.bool()
            if bool(burn_in_done.any()):
                bad = (
                    burn_in_done.nonzero(as_tuple=False)
                    .flatten()
                    .detach()
                    .cpu()
                    .tolist()
                )
                raise RuntimeError(
                    "g0 source collection terminated during the three-step "
                    f"action-history burn-in in envs {bad}"
                )
            last_actions = burn_in_action.clone()

        keys = (
            "states",
            "actions",
            "next_states",
            "contacts",
            "terminations",
            "observations",
            "next_observations",
            "commands",
            "rewards",
            "dones",
            "timeouts",
            "prev_actions",
            "episode_ids",
            "timesteps",
            "sim_root_states_local",
            "sim_joint_positions",
            "sim_joint_velocities",
            "sim_action_histories",
            "sim_prev_action_histories",
            "sim_prev_prev_action_histories",
            "sim_snapshot_commands",
        )
        data: dict[str, Any] = {key: [] for key in keys}
        for point in range(int(args.horizon) + 1):
            _force_commands(env.unwrapped, commands, env_ids)
            state = extractor.extract_state().clone()
            snapshot = capture_go2_simulator_snapshot(
                env.unwrapped,
                env_ids,
                command=commands,
            )
            action = _cpu_actions(
                point=point + HISTORY_BURN_IN_STEPS,
                num_envs=int(args.num_envs),
                action_dim=int(env.single_action_space.shape[0]),
                std=float(args.action_std),
                seed=int(args.seed),
            ).to(env.unwrapped.device)
            observation = make_go2_policy_obs(state, commands, last_actions)
            _, reward, terminated, timeout, _ = step_without_automatic_reset(
                env.unwrapped,
                action,
            )
            next_state = extractor.extract_state().clone()
            contact = extractor.extract_contact().clone()
            done = terminated.bool() | timeout.bool()
            next_observation = make_go2_policy_obs(
                next_state,
                commands,
                action,
            )
            if point < int(args.horizon) and bool(done.any()):
                bad = done.nonzero(as_tuple=False).flatten().detach().cpu().tolist()
                raise RuntimeError(
                    f"g0 source collection terminated before H steps in envs {bad}"
                )

            values = {
                "states": state,
                "actions": action,
                "next_states": next_state,
                "contacts": contact,
                "terminations": done.float().unsqueeze(-1),
                "observations": observation,
                "next_observations": next_observation,
                "commands": commands,
                "rewards": reward,
                "dones": done,
                "timeouts": timeout.bool(),
                "prev_actions": last_actions,
                "episode_ids": episode_ids,
                "timesteps": torch.full_like(
                    episode_ids,
                    point + HISTORY_BURN_IN_STEPS,
                ),
                "sim_root_states_local": snapshot["root_state_local"],
                "sim_joint_positions": snapshot["joint_position"],
                "sim_joint_velocities": snapshot["joint_velocity"],
                "sim_action_histories": snapshot["action"],
                "sim_prev_action_histories": snapshot["prev_action"],
                "sim_prev_prev_action_histories": snapshot["prev_prev_action"],
                "sim_snapshot_commands": snapshot["command"],
            }
            for key, value in values.items():
                data[key].append(value.detach().cpu())
            last_actions = action.clone()

        initial_history_max_abs = {
            "action": float(data["sim_action_histories"][0].abs().max()),
            "prev_action": float(
                data["sim_prev_action_histories"][0].abs().max()
            ),
            "prev_prev_action": float(
                data["sim_prev_prev_action_histories"][0].abs().max()
            ),
        }
        if any(value <= 0.0 for value in initial_history_max_abs.values()):
            raise RuntimeError(
                "Certification burn-in did not exercise all three nonzero "
                f"action-history levels: {initial_history_max_abs}"
            )
        metadata = {
            "format_version": COLLECTOR_FORMAT_VERSION,
            "condition_id": "g0",
            "task": G0_TASK,
            "target_environment": "normal_go2",
            "perfect_simulator": "normal_go2",
            "imperfect_simulator": "normal_go2",
            "seed": int(args.seed),
            "num_envs": int(args.num_envs),
            "horizon": int(args.horizon),
            "history_burn_in_steps": HISTORY_BURN_IN_STEPS,
            "recorded_timestep_start": HISTORY_BURN_IN_STEPS,
            "initial_action_history_max_abs": initial_history_max_abs,
            "num_time_steps": int(args.horizon) + 1,
            "num_transitions": (int(args.horizon) + 1) * int(args.num_envs),
            "unrecorded_history_burn_in_transitions": (
                HISTORY_BURN_IN_STEPS * int(args.num_envs)
            ),
            "save_trace_snapshots": True,
            "snapshot_version": "go2_trace_snapshot_v1",
            "condition_registry": canonical_condition_metadata("g0"),
            "certification_controls": {
                "action_delay": 0,
                "actuator_delay": 0,
                "domain_randomization": False,
                "push_randomization": False,
                "observation_noise": False,
                "action_noise": False,
                "physics_startup_events": False,
                "physics_interval_events": False,
                "command_random_resample": False,
            },
            "action_source": "deterministic_cpu_gaussian_clipped",
            "action_std": float(args.action_std),
            "command": [float(value) for value in args.command],
            "physics": physics,
            "provenance": provenance,
            "device_identity": device_identity,
            "runtime_scope": RUNTIME_SCOPE,
        }
        dataset = {
            "format_version": "go2_mixed_rwm_dataset_v2",
            "state_dim": 45,
            "action_dim": 12,
            "contact_dim": 4,
            "termination_dim": 1,
            "num_envs": int(args.num_envs),
            "capacity": (int(args.horizon) + 1) * int(args.num_envs),
            **data,
            "metadata": metadata,
        }
        view, preflight = preflight_dataset(
            dataset,
            expected_condition_id="g0",
            expected_task=G0_TASK,
        )
        if view is None:
            raise RuntimeError(f"Collected dataset failed preflight: {preflight}")
        _, windows = select_source_windows(
            view,
            horizon=int(args.horizon),
            count=int(args.num_envs),
            seed=int(args.seed),
            min_source_timestep=HISTORY_BURN_IN_STEPS,
        )
        if not windows["passed"]:
            raise RuntimeError(
                f"Collected dataset has insufficient certification windows: {windows}"
            )
        runner_config = {
            "schema_version": RUNNER_CONFIG_VERSION,
            "condition_id": "g0",
            "task": G0_TASK,
            "horizon": int(args.horizon),
            "history_burn_in_steps": HISTORY_BURN_IN_STEPS,
            "device": str(args.device),
            "device_identity": device_identity,
            "seed": int(args.seed),
            "physics": physics,
            "provenance": provenance,
            "determinism_tolerance": float(args.determinism_tolerance),
            "runtime_scope": RUNTIME_SCOPE,
        }
        return dataset, runner_config
    finally:
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    dataset, runner_config = collect_dataset(args)
    dataset_path = Path(args.output_dataset).expanduser().resolve()
    runner_config_path = Path(args.output_runner_config).expanduser().resolve()
    _atomic_torch_save(dataset_path, dataset)
    _atomic_json(runner_config_path, runner_config)
    manifest = {
        "status": "COLLECTED",
        "dataset": str(dataset_path),
        "runner_config": str(runner_config_path),
        "condition_id": "g0",
        "task": G0_TASK,
        "num_envs": int(args.num_envs),
        "horizon": int(args.horizon),
    }
    _atomic_json(dataset_path.with_suffix(".collection.json"), manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
