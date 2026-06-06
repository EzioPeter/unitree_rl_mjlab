"""Collect a Go2 sim-mixed dataset for offline RWM-U training."""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flash_rl.envs.mjlab import configure_mjlab_randomization
from scripts.reinforcement_learning.rwm_dataset.dataset import (
    COLLECTOR_ID_TO_NAME,
    COLLECTOR_NAME_TO_ID,
    Go2MixedDatasetBuilder,
    merge_dataset_dicts,
    parse_collector_mix,
    sample_collector_ids,
    save_dataset_dict,
)
from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    load_config as load_flashsac_config,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    select_device,
    set_seed,
)
from src.tasks.rwm_velocity.mdp.extractors import Go2RWMExtractor


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "go2_mixed_dataset.yaml"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_transitions", type=int, default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--expert_policy_path", default=None)
    parser.add_argument("--medium_policy_path", default=None)
    parser.add_argument("--collector_mix", default=None)
    parser.add_argument("--action_noise_std", type=float, default=None)
    parser.add_argument("--medium_action_noise_std", type=float, default=None)
    parser.add_argument("--failure_action_noise_std", type=float, default=None)
    parser.add_argument("--chunk_size", type=int, default=None)
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_domain_randomization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_push_randomization", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_observation_noise", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(args.config_path)
    updates = list(args.overrides or [])
    scalar_overrides = {
        "task": args.task,
        "seed": args.seed,
        "num_envs": args.num_envs,
        "num_transitions": args.num_transitions,
        "save_path": args.save_path,
        "expert_policy_path": args.expert_policy_path,
        "medium_policy_path": args.medium_policy_path,
        "collector_mix": args.collector_mix,
        "action_noise_std": args.action_noise_std,
        "medium_action_noise_std": args.medium_action_noise_std,
        "failure_action_noise_std": args.failure_action_noise_std,
        "chunk_size": args.chunk_size,
        "headless": args.headless,
    }
    for key, value in scalar_overrides.items():
        if value is not None:
            updates.append(f"{key}={value}")
    env_overrides = {
        "env.use_domain_randomization": args.use_domain_randomization,
        "env.use_push_randomization": args.use_push_randomization,
        "env.use_observation_noise": args.use_observation_noise,
    }
    for key, value in env_overrides.items():
        if value is not None:
            updates.append(f"{key}={str(value).lower()}")
    if updates:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(updates))
    OmegaConf.resolve(cfg)
    return cfg


def _actor_obs_np(obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
    return obs_dict["actor"].detach().cpu().numpy().astype(np.float32)


def _maybe_load_agent(
    checkpoint_path: str | None,
    *,
    num_envs: int,
    actor_dim: int,
    action_dim: int,
    device: str,
) -> Any | None:
    if checkpoint_path is None or str(checkpoint_path).lower() in {"", "none", "null"}:
        return None
    ckpt_dir = resolve_repo_path(checkpoint_path)
    cfg_path = ckpt_dir / "rwm_flashsac_config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"FlashSAC checkpoint config not found: {cfg_path}")
    cfg = load_flashsac_config(cfg_path)
    cfg.agent.load_optimizer = False
    cfg.agent.load_reward_normalizer = False
    cfg.agent.device_type = device
    obs_space, action_space = make_vector_spaces(num_envs, obs_dim=actor_dim, action_dim=action_dim)
    agent = create_go2_flashsac_agent(obs_space, action_space, make_flashsac_config(cfg, device=device))
    agent.load(str(ckpt_dir))
    return agent


def _policy_actions(agent: Any | None, observations: np.ndarray, action_dim: int) -> np.ndarray | None:
    if agent is None:
        return None
    return agent.sample_actions(
        interaction_step=0,
        prev_transition={"next_observation": observations},
        training=False,
    ).astype(np.float32)


def _mixed_actions(
    *,
    collector_ids: torch.Tensor,
    observations: np.ndarray,
    expert_agent: Any | None,
    medium_agent: Any | None,
    action_dim: int,
    device: torch.device,
    action_noise_std: float,
    medium_action_noise_std: float,
    failure_action_noise_std: float,
) -> torch.Tensor:
    num_envs = int(collector_ids.numel())
    actions = torch.empty(num_envs, action_dim, device=device).uniform_(-1.0, 1.0)
    expert_np = _policy_actions(expert_agent, observations, action_dim)
    medium_np = _policy_actions(medium_agent, observations, action_dim)
    if medium_np is None:
        medium_np = expert_np
    expert_t = torch.from_numpy(expert_np).to(device) if expert_np is not None else None
    medium_t = torch.from_numpy(medium_np).to(device) if medium_np is not None else None

    expert_mask = collector_ids == COLLECTOR_NAME_TO_ID["expert"]
    if expert_t is not None and expert_mask.any():
        actions[expert_mask] = expert_t[expert_mask]

    noisy_mask = collector_ids == COLLECTOR_NAME_TO_ID["noisy_expert"]
    if expert_t is not None and noisy_mask.any():
        actions[noisy_mask] = expert_t[noisy_mask] + action_noise_std * torch.randn_like(actions[noisy_mask])

    medium_mask = collector_ids == COLLECTOR_NAME_TO_ID["medium"]
    if medium_t is not None and medium_mask.any():
        actions[medium_mask] = medium_t[medium_mask] + medium_action_noise_std * torch.randn_like(actions[medium_mask])

    failure_mask = collector_ids == COLLECTOR_NAME_TO_ID["failure_border"]
    if expert_t is not None and failure_mask.any():
        actions[failure_mask] = expert_t[failure_mask] + failure_action_noise_std * torch.randn_like(actions[failure_mask])

    return actions.clamp(-1.0, 1.0)


def _new_builder(cfg: Any, dims: Any, num_envs: int, metadata: dict[str, Any]) -> Go2MixedDatasetBuilder:
    return Go2MixedDatasetBuilder(
        state_dim=int(dims.state_dim),
        action_dim=int(dims.action_dim),
        contact_dim=int(dims.contact_dim),
        termination_dim=int(dims.termination_dim),
        num_envs=num_envs,
        capacity=int(cfg.num_transitions),
        metadata=metadata,
    )


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")
    args = _parse_args()
    cfg = _load_config(args)
    device = select_device(args.device)
    set_seed(int(cfg.seed))

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(str(cfg.task))
    env_cfg.scene.num_envs = int(cfg.num_envs)
    env_cfg.seed = int(cfg.seed)
    if hasattr(env_cfg, "auto_reset"):
        env_cfg.auto_reset = True
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=bool(cfg.env.use_domain_randomization),
        use_push_randomization=bool(cfg.env.use_push_randomization),
        use_observation_noise=bool(cfg.env.use_observation_noise),
    )

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    extractor = Go2RWMExtractor(env.unwrapped)
    dims = extractor.dims
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    action_dim = int(env.single_action_space.shape[0])
    if actor_dim != int(dims.policy_obs_dim):
        raise RuntimeError(f"Expected {dims.policy_obs_dim}-dim RWM obs, got {actor_dim}.")

    expert_agent = _maybe_load_agent(
        None if cfg.expert_policy_path is None else str(cfg.expert_policy_path),
        num_envs=int(cfg.num_envs),
        actor_dim=actor_dim,
        action_dim=action_dim,
        device=device,
    )
    medium_agent = _maybe_load_agent(
        None if cfg.medium_policy_path is None else str(cfg.medium_policy_path),
        num_envs=int(cfg.num_envs),
        actor_dim=actor_dim,
        action_dim=action_dim,
        device=device,
    )

    mix = parse_collector_mix(str(cfg.collector_mix))
    save_path = resolve_repo_path(str(cfg.save_path))
    part_dir = save_path.parent / "parts"
    part_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "task": str(cfg.task),
        "robot": "Unitree-Go2",
        "dt": float(getattr(env_cfg, "step_dt", 0.02)),
        "obs_dim": actor_dim,
        "action_dim": action_dim,
        "state_dim": int(dims.state_dim),
        "contact_dim": int(dims.contact_dim),
        "termination_dim": int(dims.termination_dim),
        "num_envs": int(cfg.num_envs),
        "requested_num_transitions": int(cfg.num_transitions),
        "collector_mix": dict(mix),
        "collector_id_to_name": dict(COLLECTOR_ID_TO_NAME),
        "expert_policy_path": None if cfg.expert_policy_path is None else str(cfg.expert_policy_path),
        "medium_policy_path": None if cfg.medium_policy_path is None else str(cfg.medium_policy_path),
        "env_randomization": {
            "use_domain_randomization": bool(cfg.env.use_domain_randomization),
            "use_push_randomization": bool(cfg.env.use_push_randomization),
            "use_observation_noise": bool(cfg.env.use_observation_noise),
        },
    }
    builder = _new_builder(cfg, dims, int(cfg.num_envs), metadata)
    part_paths: list[Path] = []

    obs_dict, _ = env.reset()
    observations = _actor_obs_np(obs_dict)
    last_actions = torch.zeros(int(cfg.num_envs), action_dim, device=env.device)
    episode_ids = torch.arange(int(cfg.num_envs), device=env.device, dtype=torch.long)
    next_episode_id = int(cfg.num_envs)
    timesteps = torch.zeros(int(cfg.num_envs), device=env.device, dtype=torch.long)
    collector_ids = sample_collector_ids(mix, int(cfg.num_envs), env.device)
    collector_counts = torch.zeros(len(COLLECTOR_ID_TO_NAME), dtype=torch.long)
    returns = torch.zeros(int(cfg.num_envs), device=env.device)
    lengths = torch.zeros(int(cfg.num_envs), device=env.device, dtype=torch.long)
    completed_returns: list[float] = []
    completed_lengths: list[float] = []
    termination_count = 0
    timeout_count = 0

    total_steps = int(np.ceil(int(cfg.num_transitions) / int(cfg.num_envs)))
    chunk_size = int(cfg.chunk_size) if int(cfg.chunk_size) > 0 else 0
    start_time = time.perf_counter()

    def flush_part(force: bool = False) -> None:
        nonlocal builder
        if builder.num_time_steps == 0:
            return
        if not force and (chunk_size <= 0 or builder.num_transitions < chunk_size):
            return
        part_path = part_dir / f"dataset_part_{len(part_paths):03d}.pt"
        builder.save(part_path)
        part_paths.append(part_path)
        print(f"[Go2-MixedDataset] saved part {part_path} ({builder.num_transitions} transitions)")
        builder = _new_builder(cfg, dims, int(cfg.num_envs), metadata)

    print(f"[Go2-MixedDataset] task={cfg.task}")
    print(f"[Go2-MixedDataset] save_path={save_path}")
    print(f"[Go2-MixedDataset] mix={mix}, num_envs={cfg.num_envs}, target_transitions={cfg.num_transitions}")
    for _ in tqdm.trange(total_steps, smoothing=0.1, mininterval=0.5):
        state = extractor.extract_state()
        command = extractor.extract_command()
        prev_action = last_actions.clone()
        action_t = _mixed_actions(
            collector_ids=collector_ids,
            observations=observations,
            expert_agent=expert_agent,
            medium_agent=medium_agent,
            action_dim=action_dim,
            device=torch.device(env.device),
            action_noise_std=float(cfg.action_noise_std),
            medium_action_noise_std=float(cfg.medium_action_noise_std),
            failure_action_noise_std=float(cfg.failure_action_noise_std),
        )
        obs_dict, rewards, terminateds, truncateds, _extras = env.step(action_t)
        next_obs_np = _actor_obs_np(obs_dict)
        next_obs_t = torch.from_numpy(next_obs_np).to(env.device)
        obs_t = torch.from_numpy(observations).to(env.device)
        next_state = extractor.extract_state()
        contact = extractor.extract_contact()
        termination = extractor.extract_termination()
        done = terminateds | truncateds

        builder.add(
            obs=obs_t,
            next_obs=next_obs_t,
            state=state,
            action=action_t,
            next_state=next_state,
            contact=contact,
            termination=termination,
            command=command,
            reward=rewards,
            done=done,
            timeout=truncateds,
            prev_action=prev_action,
            episode_id=episode_ids,
            timestep=timesteps,
            collector_type=collector_ids,
        )

        collector_counts += torch.bincount(
            collector_ids.detach().cpu(),
            minlength=len(COLLECTOR_ID_TO_NAME),
        )
        returns += rewards
        lengths += 1
        done_ids = done.nonzero(as_tuple=False).flatten()
        if done_ids.numel() > 0:
            completed_returns.extend(returns[done_ids].detach().cpu().tolist())
            completed_lengths.extend(lengths[done_ids].detach().cpu().float().tolist())
            termination_count += int(terminateds.sum().item())
            timeout_count += int(truncateds.sum().item())
            returns[done_ids] = 0.0
            lengths[done_ids] = 0
            new_ids = torch.arange(
                next_episode_id,
                next_episode_id + int(done_ids.numel()),
                device=env.device,
                dtype=torch.long,
            )
            next_episode_id += int(done_ids.numel())
            episode_ids[done_ids] = new_ids
            timesteps[done_ids] = 0
            collector_ids[done_ids] = sample_collector_ids(mix, int(done_ids.numel()), env.device)
        not_done = ~done
        timesteps[not_done] += 1
        last_actions = action_t
        observations = next_obs_np
        flush_part(force=False)

    flush_part(force=True)
    env.close()

    if len(part_paths) == 1:
        dataset = torch.load(part_paths[0], map_location="cpu", weights_only=False)
    else:
        dataset = merge_dataset_dicts(
            [torch.load(path, map_location="cpu", weights_only=False) for path in part_paths]
        )
    dataset["metadata"].update(
        {
            "actual_num_transitions": len(dataset["states"]) * int(dataset["num_envs"]),
            "collector_counts": {
                COLLECTOR_ID_TO_NAME[idx]: int(count)
                for idx, count in enumerate(collector_counts.tolist())
                if idx in COLLECTOR_ID_TO_NAME
            },
            "mean_reward": float(np.mean(completed_returns)) if completed_returns else float(returns.mean().item()),
            "mean_episode_length": float(np.mean(completed_lengths)) if completed_lengths else float(lengths.float().mean().item()),
            "termination_count": int(termination_count),
            "timeout_count": int(timeout_count),
            "collection_seconds": float(time.perf_counter() - start_time),
        }
    )
    save_dataset_dict(dataset, save_path)
    file_mb = save_path.stat().st_size / (1024 * 1024)
    print("[Go2-MixedDataset] complete")
    print(f"  dataset: {save_path}")
    print(f"  transitions: {dataset['metadata']['actual_num_transitions']}")
    print(f"  file_size_mb: {file_mb:.2f}")
    print(f"  collector_counts: {dataset['metadata']['collector_counts']}")
    print(f"  mean_reward: {dataset['metadata']['mean_reward']:.4f}")
    print(f"  mean_episode_length: {dataset['metadata']['mean_episode_length']:.2f}")
    print(f"  termination_count: {termination_count}, timeout_count: {timeout_count}")


if __name__ == "__main__":
    main()
