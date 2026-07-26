#!/usr/bin/env python3
"""Probe buffer-snapshot reset against MuJoCo's full integration state."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--output-buffer", required=True)
    parser.add_argument("--output-report", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=9102)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--action-std", type=float, default=0.02)
    parser.add_argument("--tolerance", type=float, default=1.0e-4)
    parser.add_argument(
        "--command",
        type=float,
        nargs=3,
        default=(0.2, 0.0, 0.0),
        metavar=("VX", "VY", "YAW"),
    )
    return parser


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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _cpu_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    import torch

    return {
        key: value.detach().cpu().clone()
        if isinstance(value, torch.Tensor)
        else value
        for key, value in snapshot.items()
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_envs < 1 or args.horizon < 1:
        raise ValueError("num-envs and horizon must be positive.")
    if not 0.0 < float(args.tolerance) <= 1.0e-4:
        raise ValueError("tolerance must be in (0, 1e-4].")

    # This guard is stdlib-only and runs before torch/MJLab imports.
    from scripts.reinforcement_learning.rwm_trace.collect_go2_reset_certification_dataset import (
        HISTORY_BURN_IN_STEPS,
        _cpu_actions,
        _guard_requested_device,
    )

    device_identity = _guard_requested_device(str(args.device))

    import torch

    from scripts.reinforcement_learning.rwm_trace.go2_perfect_sim_runner import (
        _configure_g0_env,
        _force_commands,
    )
    from scripts.reinforcement_learning.rwm_trace.simulator_reset import (
        SNAPSHOT_REQUIRED_KEYS,
        capture_full_simulator_state,
        capture_go2_simulator_snapshot,
        capture_physical37,
        capture_rwm45,
        full_simulator_state_layout,
        reset_from_snapshot,
        reset_native_random,
        step_without_automatic_reset,
    )

    env = None
    try:
        env, _, _ = _configure_g0_env(
            int(args.num_envs),
            {"seed": int(args.seed), "device": str(args.device)},
        )
        base_env = env.unwrapped
        env_ids = torch.arange(
            int(args.num_envs),
            device=base_env.device,
            dtype=torch.long,
        )
        commands = torch.tensor(
            args.command,
            device=base_env.device,
            dtype=torch.float32,
        ).repeat(int(args.num_envs), 1)
        action_dim = int(env.single_action_space.shape[0])

        # Build the source buffer in one uninterrupted perfect-simulator run.
        reset_native_random(env, seed=int(args.seed))
        for burn_in_step in range(HISTORY_BURN_IN_STEPS):
            _force_commands(base_env, commands, env_ids)
            burn_action = _cpu_actions(
                point=burn_in_step,
                num_envs=int(args.num_envs),
                action_dim=action_dim,
                std=float(args.action_std),
                seed=int(args.seed),
            ).to(base_env.device)
            _, _, terminated, timeout, _ = step_without_automatic_reset(
                base_env,
                burn_action,
            )
            if bool((terminated.bool() | timeout.bool()).any()):
                raise RuntimeError("Source rollout terminated during history burn-in.")

        _force_commands(base_env, commands, env_ids)
        source_snapshot = capture_go2_simulator_snapshot(
            base_env,
            env_ids,
            command=commands,
        )
        layout = full_simulator_state_layout(base_env)
        source_full = [capture_full_simulator_state(base_env, env_ids)]
        source_physical = [capture_physical37(base_env, env_ids)]
        source_rwm = [capture_rwm45(base_env, env_ids)]
        source_actions = []
        source_terminations = []
        for step in range(int(args.horizon)):
            action = _cpu_actions(
                point=step + HISTORY_BURN_IN_STEPS,
                num_envs=int(args.num_envs),
                action_dim=action_dim,
                std=float(args.action_std),
                seed=int(args.seed),
            ).to(base_env.device)
            source_actions.append(action.detach().cpu())
            _force_commands(base_env, commands, env_ids)
            _, _, terminated, timeout, _ = step_without_automatic_reset(
                base_env,
                action,
            )
            done = terminated.bool() | timeout.bool()
            source_terminations.append(done.detach().cpu())
            if bool(done.any()):
                raise RuntimeError(f"Source rollout terminated at step {step}.")
            source_full.append(capture_full_simulator_state(base_env, env_ids))
            source_physical.append(capture_physical37(base_env, env_ids))
            source_rwm.append(capture_rwm45(base_env, env_ids))

        source_buffer = {
            "format_version": "go2_full_state_reset_probe_buffer_v1",
            "reset_snapshot": _cpu_snapshot(source_snapshot),
            "actions": torch.stack(source_actions, dim=1),
            "commands": commands.detach().cpu().unsqueeze(1).repeat(
                1, int(args.horizon) + 1, 1
            ),
            "full_simulator_states": torch.stack(source_full, dim=1).cpu(),
            "physical37": torch.stack(source_physical, dim=1).cpu(),
            "rwm45": torch.stack(source_rwm, dim=1).cpu(),
            "terminations": torch.stack(source_terminations, dim=1),
            "metadata": {
                "full_state_layout": layout,
                "reset_fields": list(SNAPSHOT_REQUIRED_KEYS),
                "reset_uses_full_state": False,
                "full_state_is_evaluation_only": True,
                "history_burn_in_steps": HISTORY_BURN_IN_STEPS,
                "horizon": int(args.horizon),
                "num_envs": int(args.num_envs),
                "seed": int(args.seed),
                "device_identity": device_identity,
            },
        }
        buffer_path = Path(args.output_buffer).resolve()
        _atomic_torch_save(buffer_path, source_buffer)
        buffer_sha256 = _sha256(buffer_path)

        # Reload from disk so the reset input is the serialized buffer artifact.
        loaded = torch.load(buffer_path, map_location="cpu", weights_only=False)
        reset_native_random(env, seed=int(args.seed))
        reset_result = reset_from_snapshot(
            base_env,
            loaded["reset_snapshot"],
            env_ids,
        )
        replay_commands = loaded["commands"].to(base_env.device)
        _force_commands(base_env, replay_commands[:, 0], env_ids)
        replay_full = [capture_full_simulator_state(base_env, env_ids)]
        replay_physical = [capture_physical37(base_env, env_ids)]
        replay_rwm = [capture_rwm45(base_env, env_ids)]
        replay_terminations = []
        for step in range(int(args.horizon)):
            _force_commands(base_env, replay_commands[:, step], env_ids)
            _, _, terminated, timeout, _ = step_without_automatic_reset(
                base_env,
                loaded["actions"][:, step].to(base_env.device),
            )
            replay_terminations.append(
                (terminated.bool() | timeout.bool()).detach().cpu()
            )
            _force_commands(base_env, replay_commands[:, step + 1], env_ids)
            replay_full.append(capture_full_simulator_state(base_env, env_ids))
            replay_physical.append(capture_physical37(base_env, env_ids))
            replay_rwm.append(capture_rwm45(base_env, env_ids))

        expected_full = loaded["full_simulator_states"].float()
        actual_full = torch.stack(replay_full, dim=1).cpu().float()
        full_error = torch.abs(actual_full - expected_full)
        expected_physical = loaded["physical37"].float()
        actual_physical = torch.stack(replay_physical, dim=1).cpu().float()
        expected_rwm = loaded["rwm45"].float()
        actual_rwm = torch.stack(replay_rwm, dim=1).cpu().float()

        per_field = {}
        for name, bounds in layout["fields"].items():
            start, stop = int(bounds["start"]), int(bounds["stop"])
            if stop == start:
                per_field[name] = {"width": 0, "linf": 0.0}
                continue
            values = full_error[..., start:stop]
            per_field[name] = {
                "width": stop - start,
                "linf": float(values.max()),
                "mean": float(values.mean()),
            }

        flat_index = int(torch.argmax(full_error))
        env_index, step_index, dim_index = (
            int(value)
            for value in torch.unravel_index(
                torch.tensor(flat_index),
                full_error.shape,
            )
        )
        worst_field = next(
            name
            for name, bounds in layout["fields"].items()
            if int(bounds["start"]) <= dim_index < int(bounds["stop"])
        )
        time_stop = int(layout["fields"]["time"]["stop"])
        without_time = full_error[..., time_stop:]
        termination_match = torch.equal(
            torch.stack(replay_terminations, dim=1).bool(),
            loaded["terminations"].bool(),
        )
        full_linf = float(full_error.max())
        report = {
            "format_version": "go2_full_state_reset_probe_report_v1",
            "status": "PASS" if full_linf <= float(args.tolerance) else "FAIL",
            "passed": bool(full_linf <= float(args.tolerance)),
            "definition": {
                "reset_source": "serialized_buffer.reset_snapshot",
                "reset_fields": list(SNAPSHOT_REQUIRED_KEYS),
                "evaluation_state": "MuJoCo mjSTATE_INTEGRATION",
                "full_state_layout": layout,
                "full_state_is_not_used_for_reset": True,
            },
            "config": {
                "seed": int(args.seed),
                "num_envs": int(args.num_envs),
                "horizon": int(args.horizon),
                "tolerance": float(args.tolerance),
                "command": [float(value) for value in args.command],
                "action_std": float(args.action_std),
                "device_identity": device_identity,
            },
            "source_buffer": {
                "path": str(buffer_path),
                "sha256": buffer_sha256,
            },
            "metrics": {
                "full_integration_state_linf": full_linf,
                "full_integration_state_without_time_linf": float(
                    without_time.max()
                ),
                "physical37_diagnostic_linf": float(
                    torch.abs(actual_physical - expected_physical).max()
                ),
                "rwm45_diagnostic_linf": float(
                    torch.abs(actual_rwm - expected_rwm).max()
                ),
                "termination_exact_match": termination_match,
                "per_step_full_state_linf": [
                    float(value)
                    for value in full_error.amax(dim=(0, 2)).tolist()
                ],
                "per_field": per_field,
            },
            "worst": {
                "env_index": env_index,
                "step": step_index,
                "dimension": dim_index,
                "field": worst_field,
                "expected": float(expected_full[env_index, step_index, dim_index]),
                "actual": float(actual_full[env_index, step_index, dim_index]),
                "absolute_error": float(
                    full_error[env_index, step_index, dim_index]
                ),
            },
            "reset_result": {
                "manual_restore_completed": bool(
                    reset_result.manual_restore_completed
                ),
                "exact_reset": bool(reset_result.exact_reset),
                "observable_reconstruction_errors": dict(
                    reset_result.observable_reconstruction_errors
                ),
            },
        }
        _atomic_json(Path(args.output_report).resolve(), report)
        return report
    finally:
        if env is not None:
            env.close()


def main(argv: Sequence[str] | None = None) -> int:
    report = run(build_parser().parse_args(argv))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
