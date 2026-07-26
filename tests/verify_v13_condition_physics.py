"""Independently verify the five V13 simulator condition realizations."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


CONDITIONS = {
    "g0": (1.0, 0.0),
    "rr05": (0.5, 0.0),
    "rr03": (0.3, 0.0),
    "p5": (1.0, 5.0),
    "p75": (1.0, 7.5),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = _parse_args()
    repo_root = Path(args.repo_root).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite physical audit: {output}")
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
        raise ValueError("CUDA_VISIBLE_DEVICES must be explicitly set.")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from flash_rl.envs.mjlab import configure_mjlab_randomization
    from flash_rl.envs.mjlab_dr import friend_dr_runtime_snapshot
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    device = torch.device(args.device)
    gpu_name = torch.cuda.get_device_name(device)
    if "3090" not in gpu_name:
        raise RuntimeError(f"Expected RTX 3090, got {gpu_name!r}.")

    results = {}
    for condition_id, (rr_strength, payload_kg) in CONDITIONS.items():
        env_cfg = load_env_cfg(
            "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert"
        )
        env_cfg.scene.num_envs = 4
        env_cfg.seed = 0
        configure_mjlab_randomization(
            env_cfg,
            use_domain_randomization=False,
            use_push_randomization=False,
            use_observation_noise=False,
            randomization_preset="default",
            randomization_components=None,
            randomization_scale=1.0,
            payload_mass_range_kg=(payload_kg, payload_kg),
            payload_position_body_m=(0.0, 0.0, 0.10),
            payload_box_size_m=(0.20, 0.12, 0.05),
            rr_calf_strength_range=(rr_strength, rr_strength),
        )
        env = ManagerBasedRlEnv(cfg=env_cfg, device=str(args.device))
        try:
            observations, _ = env.reset()
            robot = env.scene["robot"]
            actuator_names = list(robot.actuator_names)
            rr_ctrl_id = actuator_names.index("RR_calf_joint")
            runtime = friend_dr_runtime_snapshot(env.unwrapped)
            gap = runtime.get(
                "gap_joint_motor_strength",
                torch.ones(env.num_envs, len(actuator_names), device=env.device),
            )
            realized_rr = gap[:, rr_ctrl_id]
            torch.testing.assert_close(
                realized_rr,
                torch.full_like(realized_rr, rr_strength),
                atol=1.0e-6,
                rtol=0.0,
            )
            other_ids = [index for index in range(len(actuator_names)) if index != rr_ctrl_id]
            torch.testing.assert_close(
                gap[:, other_ids],
                torch.ones_like(gap[:, other_ids]),
                atol=1.0e-6,
                rtol=0.0,
            )

            default_gain = env.sim.get_default_field("actuator_gainprm")
            current_gain = env.sim.model.actuator_gainprm[:, rr_ctrl_id, 0]
            gain_ratio = current_gain / default_gain[rr_ctrl_id, 0]
            torch.testing.assert_close(
                gain_ratio,
                torch.full_like(gain_ratio, rr_strength),
                atol=1.0e-6,
                rtol=0.0,
            )
            default_force = env.sim.get_default_field("actuator_forcerange")
            current_force = env.sim.model.actuator_forcerange[:, rr_ctrl_id]
            force_ratio = current_force / default_force[rr_ctrl_id]
            torch.testing.assert_close(
                force_ratio,
                torch.full_like(force_ratio, rr_strength),
                atol=1.0e-6,
                rtol=0.0,
            )

            base_id = int(robot.indexing.body_ids[0])
            default_mass = env.sim.get_default_field("body_mass")[base_id]
            mass_delta = env.sim.model.body_mass[:, base_id] - default_mass
            torch.testing.assert_close(
                mass_delta,
                torch.full_like(mass_delta, payload_kg),
                atol=1.0e-5,
                rtol=0.0,
            )
            results[condition_id] = {
                "actor_shape": list(observations["actor"].shape),
                "critic_shape": list(observations["critic"].shape),
                "rr_calf_control_id": rr_ctrl_id,
                "rr_calf_runtime_strength_min": float(realized_rr.min()),
                "rr_calf_runtime_strength_max": float(realized_rr.max()),
                "rr_calf_gain_ratio_min": float(gain_ratio.min()),
                "rr_calf_gain_ratio_max": float(gain_ratio.max()),
                "rr_calf_force_ratio_min": float(force_ratio.min()),
                "rr_calf_force_ratio_max": float(force_ratio.max()),
                "base_mass_delta_min_kg": float(mass_delta.min()),
                "base_mass_delta_max_kg": float(mass_delta.max()),
            }
        finally:
            env.close()

    code_paths = [
        Path(__file__).resolve(),
        repo_root / "flash_rl/envs/mjlab.py",
        repo_root / "flash_rl/envs/mjlab_dr.py",
    ]
    audit = {
        "schema": "go2_v13_five_condition_physical_audit_v1",
        "status": "passed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu": {
            "name": gpu_name,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        },
        "conditions": results,
        "code_sha256": {
            str(path): _sha256(path) for path in code_paths
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("[V13-ConditionPhysics] PASS")
    print(json.dumps(results, indent=2, sort_keys=True))
    print(f"[V13-ConditionPhysics] output={output}")


if __name__ == "__main__":
    main()
