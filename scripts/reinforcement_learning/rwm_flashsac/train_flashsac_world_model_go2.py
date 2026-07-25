"""Train Go2 FlashSAC entirely inside the learned RWM imagination env."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import tqdm
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.reinforcement_learning.rwm.dynamics import SequenceReplayBuffer, load_dynamics_checkpoint
from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.replay_mixer import (
    ExternalReplaySampler,
    ReplayMixConfig,
    compute_source_counts,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    load_config,
    make_flashsac_config,
    resolve_repo_path,
    save_config,
    select_device,
    set_seed,
)
from scripts.reinforcement_learning.rwm_flashsac.world_model_env import (
    FlashSACWorldModelEnvConfig,
    Go2RWMFlashSACWorldModelEnv,
)

class ScalarLogger:
    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        self._writer = None
        self._values: dict[str, list[float]] = {}
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(log_dir / "tb"))

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                value = float(value.detach().float().mean().cpu())
            elif isinstance(value, np.ndarray):
                value = float(value.astype(np.float32).mean()) if value.size else 0.0
            elif isinstance(value, (float, int, np.floating, np.integer)):
                value = float(value)
            else:
                continue
            self._values.setdefault(key, []).append(float(value))

    def log(self, step: int) -> dict[str, float]:
        averages = {key: float(np.mean(vals)) for key, vals in self._values.items() if vals}
        if self._writer is not None:
            for key, value in averages.items():
                self._writer.add_scalar(key, value, step)
            self._writer.flush()
        self._values.clear()
        return averages

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", default=None)
    parser.add_argument("--model_resume_path", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--policy_resume_path", default=None)
    parser.add_argument("--policy_resume_mode", choices=("full", "actor_only"), default="full")
    parser.add_argument("--load_replay_buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_imagination_envs", type=int, default=None)
    parser.add_argument("--num_env_steps", type=int, default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--save_replay_buffer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _apply_arg_overrides(cfg: Any, args: argparse.Namespace) -> Any:
    updates: list[str] = list(args.overrides or [])
    if args.model_resume_path is not None:
        updates.append(f"model_resume_path={args.model_resume_path}")
    if args.dataset_path is not None:
        updates.append(f"dataset_path={args.dataset_path}")
    if args.policy_resume_path is not None:
        updates.append(f"policy_resume_path={args.policy_resume_path}")
    if args.num_imagination_envs is not None:
        updates.append(f"num_imagination_envs={args.num_imagination_envs}")
    if args.num_env_steps is not None:
        updates.append(f"num_env_steps={args.num_env_steps}")
    if args.save_path is not None:
        updates.append(f"save_path={args.save_path}")
    if args.save_replay_buffer is not None:
        updates.append(f"save_replay_buffer={str(args.save_replay_buffer).lower()}")
    if not updates:
        return cfg
    return OmegaConf.merge(cfg, OmegaConf.from_dotlist(updates))


def _save_checkpoint(agent: Any, save_dir: Path, cfg: Any, save_replay: bool) -> None:
    agent.save(str(save_dir))
    save_config(cfg, save_dir / "rwm_flashsac_config.yaml")
    if save_replay:
        agent.save_replay_buffer(str(save_dir))


def _load_actor_only(agent: Any, checkpoint_path: Path, device: str) -> None:
    actor_path = checkpoint_path / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor-only resume requires actor.pt: {checkpoint_path}")
    actor = getattr(agent, "_actor", None) or getattr(agent, "actor", None)
    if actor is None:
        raise AttributeError("FlashSAC agent has no accessible actor module for actor-only resume.")
    if hasattr(actor, "load"):
        actor.load(str(actor_path), load_optimizer=False)
    else:
        state = torch.load(actor_path, map_location=device)
        if isinstance(state, dict) and "network_state_dict" in state:
            state = state["network_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state and len(state) == 1:
            state = state["state_dict"]
        actor.load_state_dict(state)
    print(f"[Go2-FlashSAC-RWM] loaded actor-only checkpoint={actor_path}")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_formal_replay_mix(
    *,
    cfg: Any,
    agent: Any,
    observation_dim: int,
    action_dim: int,
    agent_cfg: Any,
) -> tuple[ReplayMixConfig | None, ExternalReplaySampler | None, ExternalReplaySampler | None, dict[str, Any]]:
    enabled = bool(OmegaConf.select(cfg, "replay_mix.enabled", default=False))
    if not enabled:
        if bool(OmegaConf.select(cfg, "trace.enabled", default=False)):
            raise ValueError(
                "Legacy trace.enabled is not supported by the formal V12 trainer; "
                "use replay_mix instead."
            )
        return None, None, None, {}
    if not hasattr(agent, "configure_replay_mix"):
        raise TypeError("Formal replay mixing requires FlashSACProprioceptiveAgent.")

    mix_config = ReplayMixConfig(
        batch_size=int(agent_cfg.sample_batch_size),
        real_ratio=float(OmegaConf.select(cfg, "replay_mix.real_ratio")),
        synthetic_mode=str(OmegaConf.select(cfg, "replay_mix.synthetic_mode")),
        trace_ratio_within_synthetic=float(
            OmegaConf.select(
                cfg,
                "replay_mix.trace_ratio_within_synthetic",
            )
        ),
        shuffle=bool(
            OmegaConf.select(
                cfg,
                "replay_mix.shuffle_mixed_batch",
                default=True,
            )
        ),
    )
    if not np.isclose(mix_config.real_ratio, 0.05):
        raise ValueError("Formal V12 routes require replay_mix.real_ratio=0.05.")
    counts = compute_source_counts(mix_config)
    real_path_value = OmegaConf.select(cfg, "replay_mix.real_replay_path")
    if counts.real and not real_path_value:
        raise ValueError("Formal V12 replay mix requires real_replay_path.")
    real_sampler = (
        ExternalReplaySampler(
            resolve_repo_path(str(real_path_value)),
            seed=int(OmegaConf.select(cfg, "replay_mix.seed", default=int(cfg.seed))) + 1,
            expected_observation_dim=observation_dim,
            expected_action_dim=action_dim,
            expected_gamma=float(agent_cfg.gamma),
            expected_n_step=int(agent_cfg.n_step),
            expected_source="real",
        )
        if counts.real
        else None
    )
    sim_path_value = OmegaConf.select(
        cfg,
        "replay_mix.trace_replay_path",
        default=None,
    )
    if counts.sim and not sim_path_value:
        raise ValueError("Configured simulator replay count requires trace_replay_path.")
    sim_sampler = (
        ExternalReplaySampler(
            resolve_repo_path(str(sim_path_value)),
            seed=int(OmegaConf.select(cfg, "replay_mix.seed", default=int(cfg.seed))) + 2,
            expected_observation_dim=observation_dim,
            expected_action_dim=action_dim,
            expected_gamma=float(agent_cfg.gamma),
            expected_n_step=int(agent_cfg.n_step),
            expected_source="sim",
        )
        if counts.sim
        else None
    )
    if sim_sampler is not None:
        allow_legacy = bool(
            OmegaConf.select(cfg, "replay_mix.allow_legacy_trace", default=False)
        )
        protocol = sim_sampler.metadata.get("trace_protocol_version")
        if protocol != "go2_trace_v5_controlled" and not allow_legacy:
            raise ValueError(
                "Formal TRACE training requires trace_protocol_version="
                f"'go2_trace_v5_controlled', got {protocol!r}."
            )

    agent.configure_replay_mix(
        config=mix_config,
        real_sampler=real_sampler,
        sim_sampler=sim_sampler,
        seed=int(OmegaConf.select(cfg, "replay_mix.seed", default=int(cfg.seed))),
    )
    normalization_enabled = bool(
        OmegaConf.select(cfg, "reward_normalization.enabled", default=False)
    )
    normalization_frozen = bool(
        OmegaConf.select(cfg, "reward_normalization.frozen", default=False)
    )
    normalizer_path_value = OmegaConf.select(
        cfg,
        "reward_normalization.checkpoint_path",
        default=None,
    )
    if bool(agent_cfg.normalize_reward):
        if not normalization_enabled or not normalization_frozen or not normalizer_path_value:
            raise ValueError(
                "Normalized formal replay mixing requires an enabled, frozen "
                "reward normalizer checkpoint."
            )
        normalizer_metadata = agent.configure_frozen_reward_normalizer(
            str(resolve_repo_path(str(normalizer_path_value)))
        )
    else:
        if normalization_enabled or normalization_frozen or normalizer_path_value:
            raise ValueError(
                "Reference unnormalized reward requires reward_normalization to be "
                "disabled with no checkpoint."
            )
        normalizer_metadata = {
            "enabled": False,
            "frozen": False,
            "reason": "model_based_reference_reward_is_unnormalized",
        }
    return mix_config, real_sampler, sim_sampler, normalizer_metadata


def _write_formal_replay_manifest(
    *,
    path: Path,
    mix_config: ReplayMixConfig,
    real_sampler: ExternalReplaySampler | None,
    sim_sampler: ExternalReplaySampler | None,
    normalizer_metadata: dict[str, Any],
    model_path: Path,
    dataset_path: Path,
    policy_resume_path: Path | None,
    policy_resume_mode: str,
    actor_bc_alpha: float,
    actor_learning_starts_updates: int,
    actor_learning_rate_scale: float,
) -> None:
    counts = compute_source_counts(mix_config)
    manifest = {
        "format_version": "go2_v12_replay_mix_manifest_v1",
        "replay_mix_config": asdict(mix_config),
        "source_counts": asdict(counts),
        "real_replay_path": str(real_sampler.path) if real_sampler else None,
        "real_replay_sha256": real_sampler.sha256 if real_sampler else None,
        "real_replay_metadata": real_sampler.metadata if real_sampler else None,
        "trace_replay_path": str(sim_sampler.path) if sim_sampler else None,
        "trace_replay_sha256": sim_sampler.sha256 if sim_sampler else None,
        "trace_replay_metadata": sim_sampler.metadata if sim_sampler else None,
        "normalizer_path": normalizer_metadata.get("artifact_path"),
        "normalizer_sha256": normalizer_metadata.get("artifact_sha256"),
        "normalizer_metadata": normalizer_metadata,
        "rwm_checkpoint_path": str(model_path),
        "rwm_checkpoint_sha256": _sha256_file(model_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "policy_resume_path": (
            str(policy_resume_path) if policy_resume_path is not None else None
        ),
        "policy_resume_mode": str(policy_resume_mode),
        "policy_resume_actor_sha256": (
            _sha256_file(policy_resume_path / "actor.pt")
            if policy_resume_path is not None
            else None
        ),
        "actor_bc_alpha": float(actor_bc_alpha),
        "actor_learning_starts_updates": int(actor_learning_starts_updates),
        "actor_learning_rate_scale": float(actor_learning_rate_scale),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")


def main() -> None:
    configure_low_thread_env()
    args = _parse_args()
    cfg = load_config(args.config_path)
    cfg = _apply_arg_overrides(cfg, args)
    OmegaConf.resolve(cfg)

    device = select_device(args.device or cfg.agent.device_type)
    set_seed(int(cfg.seed))

    model_path = resolve_repo_path(str(cfg.model_resume_path))
    dataset_path = resolve_repo_path(str(cfg.dataset_path))
    dynamics, _checkpoint = load_dynamics_checkpoint(model_path, device=device)
    dataset = SequenceReplayBuffer.load(dataset_path, device=device)

    checkpoint_infos = _checkpoint.get("infos") or {}
    checkpoint_action_mask = checkpoint_infos.get("action_mask_indices") or []
    checkpoint_world_model_mask = checkpoint_infos.get("world_model_action_mask_indices") or checkpoint_action_mask
    checkpoint_policy_mask = checkpoint_infos.get("policy_action_mask_indices") or checkpoint_world_model_mask
    checkpoint_policy_observation_mask = checkpoint_infos.get("policy_observation_mask_indices") or []
    checkpoint_broken_joints = (
        checkpoint_infos.get("broken_pd_joint_names")
        or checkpoint_infos.get("broken_joint_names")
        or []
    )
    configured_policy_mask = OmegaConf.select(cfg, "world_model.policy_action_mask_indices") or []
    configured_wm_mask = OmegaConf.select(cfg, "world_model.world_model_action_mask_indices") or []
    configured_policy_observation_mask = OmegaConf.select(cfg, "world_model.policy_observation_mask_indices") or []
    configured_broken_joints = OmegaConf.select(cfg, "world_model.broken_joint_names") or []
    if checkpoint_policy_mask and not configured_policy_mask:
        OmegaConf.update(
            cfg,
            "world_model.policy_action_mask_indices",
            list(checkpoint_policy_mask),
            merge=True,
        )
    if checkpoint_world_model_mask and not configured_wm_mask:
        OmegaConf.update(
            cfg,
            "world_model.world_model_action_mask_indices",
            list(checkpoint_world_model_mask),
            merge=True,
        )
    if checkpoint_policy_observation_mask and not configured_policy_observation_mask:
        OmegaConf.update(
            cfg,
            "world_model.policy_observation_mask_indices",
            list(checkpoint_policy_observation_mask),
            merge=True,
        )
    if checkpoint_broken_joints and not configured_broken_joints:
        OmegaConf.update(
            cfg,
            "world_model.broken_joint_names",
            list(checkpoint_broken_joints),
            merge=True,
        )

    wm_cfg_dict = OmegaConf.to_container(cfg.world_model, resolve=True, throw_on_missing=True)
    assert isinstance(wm_cfg_dict, dict)
    wm_cfg = FlashSACWorldModelEnvConfig(**{str(k): v for k, v in wm_cfg_dict.items()})
    wm_cfg.num_envs = int(cfg.num_imagination_envs)
    env = Go2RWMFlashSACWorldModelEnv(dynamics=dynamics, dataset=dataset, cfg=wm_cfg, device=device)
    policy_action_dim = int(env.single_action_space.shape[0])

    agent_cfg = make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_agent(env.observation_space, env.action_space, agent_cfg)
    replay_mix_config, real_sampler, sim_sampler, normalizer_metadata = (
        _configure_formal_replay_mix(
            cfg=cfg,
            agent=agent,
            observation_dim=int(env.single_observation_space.shape[-1]),
            action_dim=policy_action_dim,
            agent_cfg=agent_cfg,
        )
    )
    policy_resume_value = OmegaConf.select(cfg, "policy_resume_path", default=None)
    policy_resume_path: Path | None = None
    if policy_resume_value:
        policy_resume_path = resolve_repo_path(str(policy_resume_value))
        if args.policy_resume_mode == "actor_only":
            _load_actor_only(agent, policy_resume_path, device=device)
        else:
            agent.load(str(policy_resume_path))
        uses_rwm_replay = bool(getattr(agent, "uses_rwm_replay", True))
        if (
            args.load_replay_buffer
            and uses_rwm_replay
            and (policy_resume_path / "replay_buffer.pt").exists()
        ):
            agent.load_replay_buffer(str(policy_resume_path))
        elif args.load_replay_buffer and uses_rwm_replay:
            print(f"[Go2-FlashSAC-RWM] replay buffer not found in resume checkpoint={policy_resume_path}")

    save_path = str(cfg.save_path).replace("TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    save_root = resolve_repo_path(save_path)
    save_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, save_root / "rwm_flashsac_config.yaml")
    if replay_mix_config is not None:
        _write_formal_replay_manifest(
            path=save_root / "v12_replay_mix_manifest.json",
            mix_config=replay_mix_config,
            real_sampler=real_sampler,
            sim_sampler=sim_sampler,
            normalizer_metadata=normalizer_metadata,
            model_path=model_path,
            dataset_path=dataset_path,
            policy_resume_path=policy_resume_path,
            policy_resume_mode=args.policy_resume_mode,
            actor_bc_alpha=float(agent_cfg.actor_bc_alpha),
            actor_learning_starts_updates=int(
                getattr(agent_cfg, "actor_learning_starts_updates", 0)
            ),
            actor_learning_rate_scale=float(
                getattr(agent_cfg, "actor_learning_rate_scale", 1.0)
            ),
        )
    logger = ScalarLogger(save_root, enabled=str(cfg.logger_type).lower() == "tensorboard")

    num_envs = int(cfg.num_imagination_envs)
    total_interaction_steps = max(1, int(int(cfg.num_env_steps) // num_envs))
    update_counter = 0.0
    observations, _ = env.reset(seed=int(cfg.seed))
    transition: dict[str, Any] | None = (
        {"next_observation": observations}
        if policy_resume_value
        else None
    )
    collection_time_acc = 0.0
    learning_time_acc = 0.0
    env_steps_since_log = 0

    print(f"[Go2-FlashSAC-RWM] model={model_path}")
    print(f"[Go2-FlashSAC-RWM] dataset={dataset_path}")
    print(f"[Go2-FlashSAC-RWM] save_root={save_root}")
    print(f"[Go2-FlashSAC-RWM] device={device}, num_envs={num_envs}, interaction_steps={total_interaction_steps}")
    print(f"[Go2-FlashSAC-RWM] full_action_dim={env.full_action_dim}, policy_action_dim={policy_action_dim}")
    print(f"[Go2-FlashSAC-RWM] policy_resume_path={policy_resume_value}")
    print(
        "[Go2-FlashSAC-RWM] "
        f"policy_action_mask_indices={list(wm_cfg.policy_action_mask_indices)}, "
        f"world_model_action_mask_indices={list(wm_cfg.world_model_action_mask_indices)}, "
        f"policy_observation_mask_indices={list(wm_cfg.policy_observation_mask_indices)}"
    )
    if replay_mix_config is not None:
        replay_counts = compute_source_counts(replay_mix_config)
        print(
            "[Go2-FlashSAC-RWM] "
            f"replay_mix={asdict(replay_mix_config)}, "
            f"source_counts={asdict(replay_counts)}, "
            f"real_transitions={real_sampler.num_transitions if real_sampler else 0}, "
            f"sim_transitions={sim_sampler.num_transitions if sim_sampler else 0}, "
            f"normalizer_sha256={normalizer_metadata.get('artifact_sha256')}"
        )
    else:
        print("[Go2-FlashSAC-RWM] replay_mix=legacy_internal_rwm_only")

    for interaction_step in tqdm.tqdm(range(1, total_interaction_steps + 1), smoothing=0.1, mininterval=0.5):
        env_step = interaction_step * num_envs
        start = time.perf_counter()
        support_preserving_warmup = bool(agent_cfg.actor_support_preserving_warmup)
        if (agent.can_start_training() or policy_resume_value or support_preserving_warmup) and (
            transition is not None or support_preserving_warmup
        ):
            # Uniform random warm-up is catastrophically off-support for a
            # 12-D learned dynamics model.  In the opt-in mode, the still
            # randomly initialized actor explores residuals around the
            # last_action already present in each dataset-reset observation.
            # No expert actor, target action, BC loss, or checkpoint is used.
            action_input = transition or {"next_observation": observations}
            actions = agent.sample_actions(
                interaction_step,
                prev_transition=action_input,
                training=True,
            )
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(num_envs, policy_action_dim)).astype(np.float32)

        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
        next_buffer_observations = next_observations.copy()
        final_obs = infos.get("final_obs")
        if final_obs is not None:
            done_mask = np.logical_or(terminateds, truncateds)
            next_buffer_observations[done_mask] = final_obs[done_mask]

        transition = {
            "observation": observations,
            "action": actions,
            "reward": rewards,
            "terminated": terminateds,
            "truncated": truncateds,
            "next_observation": next_buffer_observations,
        }
        agent.process_transition(transition)
        transition["next_observation"] = next_observations
        observations = next_observations
        collection_time_acc += time.perf_counter() - start
        env_steps_since_log += num_envs
        if "episode_info" in infos:
            logger.update(infos["episode_info"])

        if agent.can_start_training():
            start = time.perf_counter()
            update_counter += float(cfg.updates_per_interaction_step)
            while update_counter >= 1.0:
                logger.update(agent.update())
                update_counter -= 1.0
            learning_time_acc += time.perf_counter() - start

        if int(cfg.logging_per_interaction_step) and interaction_step % int(cfg.logging_per_interaction_step) == 0:
            total_time = collection_time_acc + learning_time_acc
            if total_time > 0.0:
                logger.update(
                    {
                        "Perf/total_fps": env_steps_since_log / total_time,
                        "Perf/collection_time": collection_time_acc,
                        "Perf/learning_time": learning_time_acc,
                    }
                )
            logged = logger.log(env_step)
            interesting = {
                k: logged[k]
                for k in (
                    "Train/mean_reward",
                    "Train/mean_episode_length",
                    "critic/loss",
                    "actor/loss",
                    "Imagination/epistemic_uncertainty",
                    "Imagination/track_linear_velocity",
                    "Imagination/action_rate_l2",
                    "trace/replay_batch_fraction",
                    "Replay/real_count",
                    "Replay/rwm_count",
                    "Replay/sim_count",
                    "Replay/normalizer_frozen",
                    "Replay/mixed_reward_mean",
                    "Perf/total_fps",
                )
                if k in logged
            }
            print(f"[Go2-FlashSAC-RWM] step={interaction_step} env_step={env_step} {interesting}")
            collection_time_acc = 0.0
            learning_time_acc = 0.0
            env_steps_since_log = 0

        if (
            int(cfg.save_checkpoint_per_interaction_step)
            and interaction_step % int(cfg.save_checkpoint_per_interaction_step) == 0
        ):
            _save_checkpoint(
                agent,
                save_root / f"step{interaction_step}",
                cfg,
                # Replay continuity is only needed at the final resumable
                # checkpoint.  Serializing the 10M-row buffer at every
                # diagnostic checkpoint wastes hundreds of GiB per branch.
                save_replay=False,
            )

    _save_checkpoint(
        agent,
        save_root / f"step{total_interaction_steps}",
        cfg,
        save_replay=bool(cfg.save_replay_buffer)
        and bool(getattr(agent, "uses_rwm_replay", True)),
    )
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
