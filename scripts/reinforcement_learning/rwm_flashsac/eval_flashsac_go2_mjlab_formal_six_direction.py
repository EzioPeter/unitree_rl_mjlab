"""Formal six-direction Go2 policy RMSE evaluation in one continuous MJLab rollout."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.eval_flashsac_go2_mjlab import (
    _actor_obs,
    _apply_command_to_obs,
    _attach_fixed_payload_brick,
    _configure_fixed_command_range,
    _disable_randomization,
    _force_fixed_command,
)
from scripts.reinforcement_learning.rwm_dataset.broken_go2 import (
    apply_go2_pd_joint_strength_scales,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    load_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    select_device,
    set_seed,
)


COMMANDS: tuple[tuple[str, tuple[float, float, float]], ...] = (
    ("forward", (0.5, 0.0, 0.0)),
    ("backward", (-0.5, 0.0, 0.0)),
    ("leftward", (0.0, 0.2, 0.0)),
    ("rightward", (0.0, -0.2, 0.0)),
    ("yaw_positive", (0.0, 0.0, 0.4)),
    ("yaw_negative", (0.0, 0.0, -0.4)),
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument(
        "--task",
        default="Unitree-Go2-Flat-Normal-FixStand-RWM-Pretrain-Ens",
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_envs", type=int, default=256)
    parser.add_argument("--segment_steps", type=int, default=300)
    parser.add_argument("--settling_steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=400)
    parser.add_argument("--payload_mass_kg", type=float, default=0.0)
    parser.add_argument("--rr_calf_strength", type=float, default=1.0)
    parser.add_argument(
        "--clean", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def _environment_step_dt(env: Any) -> float:
    for owner in (env.unwrapped, env):
        value = getattr(owner, "step_dt", None)
        if value is not None:
            return float(value)
    return 0.02


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse_args()
    if args.segment_steps <= args.settling_steps:
        raise ValueError("segment_steps must be greater than settling_steps")
    if args.num_envs <= 0:
        raise ValueError("num_envs must be positive")

    checkpoint_path = resolve_repo_path(args.checkpoint_path)
    config_path = (
        Path(args.config_path).expanduser()
        if args.config_path
        else checkpoint_path / "rwm_flashsac_config.yaml"
    )
    if not config_path.is_absolute():
        config_path = resolve_repo_path(config_path)
    cfg = load_config(config_path, overrides=args.overrides)
    device = select_device(args.device or cfg.agent.device_type)
    set_seed(args.seed)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(args.task, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    if args.clean:
        _disable_randomization(env_cfg)
    if args.payload_mass_kg < 0.0:
        raise ValueError("payload_mass_kg must be non-negative")
    if not 0.0 <= args.rr_calf_strength <= 1.0:
        raise ValueError("rr_calf_strength must lie in [0, 1]")
    _attach_fixed_payload_brick(env_cfg, mass_kg=float(args.payload_mass_kg))
    if args.rr_calf_strength != 1.0:
        apply_go2_pd_joint_strength_scales(
            env_cfg, {"RR_calf_joint": float(args.rr_calf_strength)}
        )

    # Keep the command manager's declared range inside the formal evaluation range.
    _configure_fixed_command_range(env_cfg, COMMANDS[0][1])
    twist = env_cfg.commands.get("twist")
    if twist is not None:
        twist.ranges.lin_vel_x = (-0.5, 0.5)
        twist.ranges.lin_vel_y = (-0.2, 0.2)
        twist.ranges.ang_vel_z = (-0.4, 0.4)

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    robot = env.unwrapped.scene["robot"]
    payload_ids = [
        idx
        for idx, name in enumerate(robot.body_names)
        if str(name) == "fixed_payload_brick"
    ]
    realized_payload_mass_kg = 0.0
    if args.payload_mass_kg == 0.0:
        if payload_ids:
            raise RuntimeError("unexpected payload body in zero-payload evaluation")
    else:
        if len(payload_ids) != 1:
            raise RuntimeError(
                f"expected one payload body, found {len(payload_ids)}"
            )
        body_id = robot.indexing.body_ids[payload_ids[0]].long()
        realized = env.unwrapped.sim.model.body_mass[:, body_id]
        expected = torch.full_like(realized, float(args.payload_mass_kg))
        if not torch.allclose(realized, expected, atol=1.0e-5, rtol=0.0):
            raise RuntimeError("realized payload mass does not match request")
        realized_payload_mass_kg = float(realized[0].item())

    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    action_dim = int(env.single_action_space.shape[0])
    if actor_dim != 48:
        raise RuntimeError(f"expected 48 actor inputs, got {actor_dim}")
    if action_dim != 12:
        raise RuntimeError(f"expected 12 actor outputs, got {action_dim}")
    obs_space, action_space = make_vector_spaces(
        args.num_envs, obs_dim=actor_dim, action_dim=action_dim
    )
    agent = create_go2_flashsac_agent(
        obs_space,
        action_space,
        make_flashsac_config(cfg, device=device),
    )
    agent.load(str(checkpoint_path))

    obs_dict, _ = env.reset()
    observations = _actor_obs(obs_dict)
    segment_results: dict[str, dict[str, Any]] = {}
    total_terminated = 0
    total_truncated = 0

    with torch.no_grad():
        for segment_index, (name, command) in enumerate(COMMANDS):
            linear_samples: list[np.ndarray] = []
            yaw_samples: list[np.ndarray] = []
            terminated_count = 0
            truncated_count = 0
            post_settling_terminated_count = 0
            post_settling_truncated_count = 0
            ever_terminated = np.zeros(args.num_envs, dtype=bool)

            for local_step in range(args.segment_steps):
                _force_fixed_command(env, command)
                observations = _apply_command_to_obs(observations, command)
                if local_step >= args.settling_steps:
                    linear_samples.append(observations[:, 0:2].copy())
                    yaw_samples.append(observations[:, 5].copy())

                actions_np = agent.sample_actions(
                    interaction_step=0,
                    prev_transition={"next_observation": observations},
                    training=False,
                )
                actions = torch.from_numpy(actions_np).to(
                    device=device, dtype=torch.float32
                )
                obs_dict, _rewards, terminateds, truncateds, _extras = env.step(
                    actions
                )
                term = terminateds.detach().cpu().numpy().astype(bool)
                trunc = truncateds.detach().cpu().numpy().astype(bool)
                terminated_count += int(term.sum())
                truncated_count += int(trunc.sum())
                if local_step >= args.settling_steps:
                    post_settling_terminated_count += int(term.sum())
                    post_settling_truncated_count += int(trunc.sum())
                ever_terminated |= term
                _force_fixed_command(env, command)
                observations = _apply_command_to_obs(
                    _actor_obs(obs_dict), command
                )

            linear = np.concatenate(linear_samples, axis=0)
            yaw = np.concatenate(yaw_samples, axis=0)
            command_xy = np.asarray(command[:2], dtype=np.float32)
            command_yaw = float(command[2])
            command_speed = float(np.linalg.norm(command_xy))
            if command_speed > 0.0:
                direction = command_xy / command_speed
                projected = linear @ direction
                axis_error = projected - command_speed
                axis_rmse = _rmse(axis_error)
                yaw_rmse = None
            else:
                yaw_direction = math.copysign(1.0, command_yaw)
                projected_yaw = yaw * yaw_direction
                axis_error = projected_yaw - abs(command_yaw)
                axis_rmse = _rmse(axis_error)
                yaw_rmse = axis_rmse

            vector_error = linear - command_xy
            xy_vector_rmse = float(
                np.sqrt(
                    np.mean(
                        np.sum(np.square(vector_error), axis=1),
                        dtype=np.float64,
                    )
                )
            )
            total_terminated += terminated_count
            total_truncated += truncated_count
            segment_results[name] = {
                "segment_index": segment_index + 1,
                "command": list(command),
                "segment_steps": args.segment_steps,
                "settling_steps": args.settling_steps,
                "sampled_steps": args.segment_steps - args.settling_steps,
                "sample_count": int(linear.shape[0]),
                "axis_rmse": axis_rmse,
                "projected_velocity_mean": float(np.mean(projected)),
                "projected_velocity_std": float(np.std(projected)),
                "base_velocity_xy_mean": [
                    float(value) for value in np.mean(linear, axis=0)
                ],
                "base_velocity_xy_std": [
                    float(value) for value in np.std(linear, axis=0)
                ],
                "yaw_rmse": yaw_rmse,
                "xy_vector_rmse": xy_vector_rmse,
                "terminated_count": terminated_count,
                "truncated_count": truncated_count,
                "post_settling_terminated_count": (
                    post_settling_terminated_count
                ),
                "post_settling_truncated_count": post_settling_truncated_count,
                "ever_terminated_env_count": int(ever_terminated.sum()),
                "termination_rate_per_env_step": (
                    terminated_count / (args.num_envs * args.segment_steps)
                ),
            }

    control_dt = _environment_step_dt(env)
    env.close()
    yaw_positive = float(segment_results["yaw_positive"]["axis_rmse"])
    yaw_negative = float(segment_results["yaw_negative"]["axis_rmse"])
    output = {
        "protocol": {
            "name": "formal_continuous_six_direction_rmse_v1",
            "task": args.task,
            "simulator": "MJLab/MuJoCo",
            "clean": bool(args.clean),
            "auto_reset": True,
            "seed": args.seed,
            "num_envs": args.num_envs,
            "segment_steps": args.segment_steps,
            "settling_steps": args.settling_steps,
            "sampled_steps_per_segment": (
                args.segment_steps - args.settling_steps
            ),
            "control_dt_seconds": control_dt,
            "control_frequency_hz": 1.0 / control_dt,
            "command_order": [name for name, _command in COMMANDS],
            "actor_training": False,
            "actor_observation_dim": actor_dim,
            "actor_action_dim": action_dim,
            "velocity_source": "actor simulator observation base-frame truth",
            "command_observation_slice": [9, 12],
        },
        "checkpoint_path": str(checkpoint_path),
        "config_path": str(config_path),
        "physical_condition": {
            "payload_mass_kg": float(args.payload_mass_kg),
            "realized_payload_mass_kg": realized_payload_mass_kg,
            "payload_position_m": [0.0, 0.0, 0.10],
            "payload_box_full_size_m": [0.20, 0.12, 0.05],
            "rr_calf_strength": float(args.rr_calf_strength),
        },
        "segments": segment_results,
        "table_rmse": {
            "forward_mps": segment_results["forward"]["axis_rmse"],
            "backward_mps": segment_results["backward"]["axis_rmse"],
            "leftward_mps": segment_results["leftward"]["axis_rmse"],
            "rightward_mps": segment_results["rightward"]["axis_rmse"],
            "yaw_positive_radps": yaw_positive,
            "yaw_negative_radps": yaw_negative,
            "yaw_pooled_radps": math.sqrt(
                (yaw_positive * yaw_positive + yaw_negative * yaw_negative)
                / 2.0
            ),
        },
        "termination": {
            "terminated_count": total_terminated,
            "truncated_count": total_truncated,
            "termination_rate_per_env_step": (
                total_terminated
                / (args.num_envs * args.segment_steps * len(COMMANDS))
            ),
        },
    }
    output_path = Path(args.output_json).expanduser()
    if not output_path.is_absolute():
        output_path = resolve_repo_path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)
    print(json.dumps(output["table_rmse"], sort_keys=True))
    print(json.dumps(output["termination"], sort_keys=True))
    print(f"wrote {output_path}")


if __name__ == "__main__":
    main()
