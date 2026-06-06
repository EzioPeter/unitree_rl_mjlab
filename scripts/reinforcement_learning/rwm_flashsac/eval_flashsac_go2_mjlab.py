"""Evaluate a Go2 FlashSAC-RWM policy in the real mjlab simulator."""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    load_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    scalarize,
    select_device,
    set_seed,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--task", default="Unitree-Go2-Flat-RWM-Pretrain-Ens")
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_envs", type=int, default=1024)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--clean", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _disable_randomization(env_cfg: Any) -> None:
    """Turn the RWM task into a clean deterministic-ish evaluation config."""
    env_cfg.events.pop("push_robot", None)
    for event_name in ("foot_friction", "encoder_bias", "base_com"):
        env_cfg.events.pop(event_name, None)
    for group_name in ("actor", "critic"):
        obs_group = env_cfg.observations.get(group_name)
        if obs_group is not None:
            obs_group.enable_corruption = False
    if hasattr(env_cfg, "curriculum"):
        env_cfg.curriculum = {}
    if hasattr(env_cfg, "curriculums"):
        env_cfg.curriculums = {}


def _actor_obs(obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
    return obs_dict["actor"].detach().cpu().numpy().astype(np.float32)


def _mean(values: list[float], fallback: float = 0.0) -> float:
    return float(np.mean(values)) if values else fallback


def _summarize_metric(metrics: dict[str, list[float]], key: str) -> float | None:
    values = metrics.get(key)
    if not values:
        return None
    return float(np.mean(values))


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse_args()

    checkpoint_path = resolve_repo_path(args.checkpoint_path)
    config_path = Path(args.config_path).expanduser() if args.config_path else checkpoint_path / "rwm_flashsac_config.yaml"
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

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    action_dim = int(env.single_action_space.shape[0])
    if actor_dim != 48:
        raise RuntimeError(f"Expected 48-dim RWM actor observation, got {actor_dim}.")

    obs_space, action_space = make_vector_spaces(args.num_envs, obs_dim=actor_dim, action_dim=action_dim)
    agent_cfg = make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_agent(obs_space, action_space, agent_cfg)
    agent.load(str(checkpoint_path))

    obs_dict, _ = env.reset()
    observations = _actor_obs(obs_dict)
    ep_returns = np.zeros(args.num_envs, dtype=np.float64)
    ep_lengths = np.zeros(args.num_envs, dtype=np.int64)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    terminated_count = 0
    timeout_count = 0
    logged_metrics: dict[str, list[float]] = defaultdict(list)

    print(f"[Go2-FlashSAC-RWM-Eval] checkpoint={checkpoint_path}")
    print(f"[Go2-FlashSAC-RWM-Eval] task={args.task}, clean={args.clean}, device={device}, num_envs={args.num_envs}")

    with torch.no_grad():
        for _step in range(args.steps):
            actions_np = agent.sample_actions(
                interaction_step=0,
                prev_transition={"next_observation": observations},
                training=False,
            )
            actions_t = torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)
            obs_dict, rewards, terminateds, truncateds, extras = env.step(actions_t)
            rewards_np = rewards.detach().cpu().numpy().astype(np.float64)
            term_np = terminateds.detach().cpu().numpy().astype(bool)
            trunc_np = truncateds.detach().cpu().numpy().astype(bool)
            done_np = np.logical_or(term_np, trunc_np)

            ep_returns += rewards_np
            ep_lengths += 1

            if done_np.any():
                completed_returns.extend(ep_returns[done_np].tolist())
                completed_lengths.extend(ep_lengths[done_np].tolist())
                terminated_count += int(term_np.sum())
                timeout_count += int(trunc_np.sum())
                ep_returns[done_np] = 0.0
                ep_lengths[done_np] = 0

            for key, value in (extras.get("log") or {}).items():
                logged_metrics[key].append(float(scalarize(value)))

            observations = _actor_obs(obs_dict)

    env.close()

    mean_return = _mean(completed_returns, fallback=float(ep_returns.mean()))
    std_return = float(np.std(completed_returns)) if completed_returns else float(np.std(ep_returns))
    mean_episode_length = _mean(completed_lengths, fallback=float(ep_lengths.mean()))
    summary = {
        "mean_return": mean_return,
        "std_return": std_return,
        "mean_episode_length": mean_episode_length,
        "completed_episodes": len(completed_returns),
        "terminated_count": terminated_count,
        "timeout_count": timeout_count,
        "non_timeout_termination_count": terminated_count,
    }

    print("[Go2-FlashSAC-RWM-Eval] Summary")
    for key, value in summary.items():
        print(f"  {key}: {value}")

    key_metrics = (
        "Episode_Reward/track_linear_velocity",
        "Episode_Reward/track_angular_velocity",
        "Episode_Reward/body_orientation_l2",
        "Episode_Reward/pose",
        "Episode_Reward/action_rate_l2",
        "Episode_Reward/foot_gait",
        "Episode_Termination/fell_over",
        "Episode_Termination/illegal_contact",
        "Metrics/twist/error_vel_xy",
        "Metrics/twist/error_vel_yaw",
    )
    for key in key_metrics:
        value = _summarize_metric(logged_metrics, key)
        if value is not None:
            print(f"  {key}: {value:.6f}")


if __name__ == "__main__":
    main()
