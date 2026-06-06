"""Train a Go2 RWM dynamics ensemble from live mjlab FlashSAC rollouts."""

from __future__ import annotations

import argparse
import os
import sys
import time
from collections import deque
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

from flash_rl.envs.mjlab import configure_mjlab_randomization
from scripts.reinforcement_learning.rwm.dynamics import (
    DynamicsConfig,
    ReplayConfig,
    SequenceReplayBuffer,
    SystemDynamicsEnsemble,
    WorldModelConfig,
    train_world_model_steps,
)
from scripts.reinforcement_learning.rwm_flashsac.agent import create_go2_flashsac_agent
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    configure_low_thread_env,
    make_flashsac_config,
    make_vector_spaces,
    resolve_repo_path,
    save_config,
    scalarize,
    select_device,
    set_seed,
)
from src.tasks.rwm_velocity.mdp.extractors import Go2RWMExtractor


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "go2_flashsac_rwm_pretrain.yaml"


class ScalarLogger:
    def __init__(self, log_dir: Path, enabled: bool = True) -> None:
        self._writer = None
        self._values: dict[str, list[float]] = {}
        if enabled:
            from torch.utils.tensorboard import SummaryWriter

            self._writer = SummaryWriter(log_dir=str(log_dir / "tb"))

    def update(self, values: dict[str, Any]) -> None:
        for key, value in values.items():
            scalar = scalarize(value)
            if isinstance(scalar, (float, int, np.floating, np.integer)):
                self._values.setdefault(key, []).append(float(scalar))

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
    parser.add_argument("--config_path", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--task", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--num_envs", type=int, default=None)
    parser.add_argument("--num_env_steps", type=int, default=None)
    parser.add_argument("--save_path", default=None)
    parser.add_argument("--overrides", action="append", default=[])
    return parser.parse_args()


def _load_config(args: argparse.Namespace) -> Any:
    cfg = OmegaConf.load(args.config_path)
    updates = list(args.overrides or [])
    if args.task is not None:
        updates.append(f"task={args.task}")
    if args.seed is not None:
        updates.append(f"seed={args.seed}")
    if args.num_envs is not None:
        updates.append(f"num_train_envs={args.num_envs}")
    if args.num_env_steps is not None:
        updates.append(f"num_env_steps={args.num_env_steps}")
    if args.save_path is not None:
        updates.append(f"save_path={args.save_path}")
    if updates:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(updates))
    OmegaConf.resolve(cfg)
    return cfg


def _make_world_model_config(cfg: Any, dims: Any) -> WorldModelConfig:
    sd = cfg.system_dynamics
    return WorldModelConfig(
        dynamics=DynamicsConfig(
            state_dim=int(dims.state_dim),
            action_dim=int(dims.action_dim),
            contact_dim=int(dims.contact_dim),
            termination_dim=int(dims.termination_dim),
            ensemble_size=int(sd.ensemble_size),
            history_horizon=int(sd.history_horizon),
            forecast_horizon=int(sd.forecast_horizon),
            hidden_size=int(sd.hidden_size),
            num_layers=int(sd.num_layers),
        ),
        replay=ReplayConfig(
            capacity=int(sd.replay_capacity),
            min_transitions=int(sd.min_transitions),
            batch_size=int(sd.batch_size),
        ),
        learning_rate=float(sd.learning_rate),
        weight_decay=float(sd.weight_decay),
        updates_per_iter=int(sd.updates_per_interaction_step),
        save_dataset=bool(cfg.save_dataset),
    )


def _actor_obs_np(obs_dict: dict[str, torch.Tensor]) -> np.ndarray:
    return obs_dict["actor"].detach().cpu().numpy().astype(np.float32)


def _save_checkpoint(
    *,
    save_root: Path,
    interaction_step: int,
    env_step: int,
    agent: Any,
    dynamics: SystemDynamicsEnsemble,
    dynamics_optimizer: torch.optim.Optimizer,
    replay: SequenceReplayBuffer,
    world_model_cfg: WorldModelConfig,
    cfg: Any,
) -> None:
    step_dir = save_root / f"step{interaction_step}"
    agent.save(str(step_dir))
    save_config(cfg, step_dir / "rwm_flashsac_pretrain_config.yaml")

    infos = {
        "replay_size": len(replay),
        "num_time_steps": replay.num_time_steps,
        "num_env_steps": env_step,
        "task": str(cfg.task),
        "collector": "FlashSAC",
    }
    checkpoint = dynamics.checkpoint(
        optimizer=dynamics_optimizer,
        iteration=interaction_step,
        infos=infos,
    )
    torch.save(checkpoint, save_root / f"model_{interaction_step}.pt")
    if world_model_cfg.save_dataset:
        replay.save(save_root / "dataset.pt")


def _extract_env_metrics(extras: dict[str, Any]) -> dict[str, float]:
    raw_log = extras.get("log") or {}
    return {key: float(scalarize(value)) for key, value in raw_log.items()}


def main() -> None:
    configure_low_thread_env()
    os.environ.setdefault("MUJOCO_GL", "egl")

    args = _parse_args()
    cfg = _load_config(args)
    device = select_device(args.device or cfg.agent.device_type)
    set_seed(int(cfg.seed))

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    import src.tasks.rwm_velocity  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    env_cfg = load_env_cfg(str(cfg.task))
    env_cfg.scene.num_envs = int(cfg.num_train_envs)
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
        raise RuntimeError(f"Expected {dims.policy_obs_dim}-dim RWM actor observation, got {actor_dim}.")
    if action_dim != int(dims.action_dim):
        raise RuntimeError(f"Action dim mismatch: env={action_dim}, extractor={dims.action_dim}.")

    obs_space, action_space = make_vector_spaces(int(cfg.num_train_envs), obs_dim=actor_dim, action_dim=action_dim)
    agent_cfg = make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_agent(obs_space, action_space, agent_cfg)

    world_model_cfg = _make_world_model_config(cfg, dims)
    dynamics = SystemDynamicsEnsemble(world_model_cfg.dynamics).to(device)
    dynamics_optimizer = torch.optim.Adam(
        dynamics.parameters(),
        lr=world_model_cfg.learning_rate,
        weight_decay=world_model_cfg.weight_decay,
    )
    replay = SequenceReplayBuffer(
        state_dim=int(dims.state_dim),
        action_dim=action_dim,
        contact_dim=int(dims.contact_dim),
        termination_dim=int(dims.termination_dim),
        num_envs=int(cfg.num_train_envs),
        capacity=world_model_cfg.replay.capacity,
        device="cpu",
    )

    save_path = str(cfg.save_path).replace("TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    save_root = resolve_repo_path(save_path)
    save_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, save_root / "rwm_flashsac_pretrain_config.yaml")
    logger = ScalarLogger(save_root, enabled=str(cfg.logger_type).lower() == "tensorboard")

    obs_dict, _ = env.reset()
    observations = _actor_obs_np(obs_dict)
    transition: dict[str, Any] | None = None
    update_counter = 0.0
    ep_returns = np.zeros(int(cfg.num_train_envs), dtype=np.float64)
    ep_lengths = np.zeros(int(cfg.num_train_envs), dtype=np.int64)
    reward_buffer: deque[float] = deque(maxlen=100)
    length_buffer: deque[float] = deque(maxlen=100)
    collection_time_acc = 0.0
    learning_time_acc = 0.0
    env_steps_since_log = 0

    num_envs = int(cfg.num_train_envs)
    total_interaction_steps = max(1, int(int(cfg.num_env_steps) // num_envs))

    print(f"[Go2-FlashSAC-RWM-Pretrain] task={cfg.task}")
    print(f"[Go2-FlashSAC-RWM-Pretrain] save_root={save_root}")
    print(f"[Go2-FlashSAC-RWM-Pretrain] device={device}, num_envs={num_envs}, interaction_steps={total_interaction_steps}")
    print(f"[Go2-FlashSAC-RWM-Pretrain] dims state={dims.state_dim}, action={action_dim}, contact={dims.contact_dim}")

    for interaction_step in tqdm.tqdm(range(1, total_interaction_steps + 1), smoothing=0.1, mininterval=0.5):
        env_step = interaction_step * num_envs
        start = time.perf_counter()

        state = extractor.extract_state().to(device)
        if agent.can_start_training() and transition is not None:
            actions_np = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            actions_np = np.random.uniform(-1.0, 1.0, size=(num_envs, action_dim)).astype(np.float32)

        actions_t = torch.from_numpy(actions_np).to(device=device, dtype=torch.float32)
        obs_dict, rewards_t, terminateds_t, truncateds_t, extras = env.step(actions_t)
        next_state = extractor.extract_state().to(device)
        contact = extractor.extract_contact().to(device)
        termination = extractor.extract_termination().to(device)
        replay.add(
            state=state,
            action=actions_t,
            next_state=next_state,
            contact=contact,
            termination=termination,
        )

        next_observations = _actor_obs_np(obs_dict)
        rewards = rewards_t.detach().cpu().numpy().astype(np.float32)
        terminateds = terminateds_t.detach().cpu().numpy().astype(bool)
        truncateds = truncateds_t.detach().cpu().numpy().astype(bool)
        next_buffer_observations = next_observations.copy()
        done_mask = np.logical_or(terminateds, truncateds)

        ep_returns += rewards.astype(np.float64)
        ep_lengths += 1
        if done_mask.any():
            reward_buffer.extend(ep_returns[done_mask].tolist())
            length_buffer.extend(ep_lengths[done_mask].tolist())
            ep_returns[done_mask] = 0.0
            ep_lengths[done_mask] = 0

        env_metrics = _extract_env_metrics(extras)
        if reward_buffer:
            env_metrics["Train/mean_reward"] = float(np.mean(reward_buffer))
            env_metrics["Train/mean_episode_length"] = float(np.mean(length_buffer))
        logger.update(env_metrics)

        transition = {
            "observation": observations,
            "action": actions_np,
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

        start = time.perf_counter()
        if agent.can_start_training():
            update_counter += float(cfg.updates_per_interaction_step)
            while update_counter >= 1.0:
                logger.update(agent.update())
                update_counter -= 1.0

        model_info = train_world_model_steps(
            dynamics=dynamics,
            optimizer=dynamics_optimizer,
            replay=replay,
            cfg=world_model_cfg,
            device=device,
            updates=int(cfg.system_dynamics.updates_per_interaction_step),
        )
        if model_info:
            logger.update({f"Model/{key}": value for key, value in model_info.items()})
        logger.update({"Model/replay_size": float(len(replay))})
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
                key: logged[key]
                for key in (
                    "Train/mean_reward",
                    "Train/mean_episode_length",
                    "critic/loss",
                    "actor/loss",
                    "Model/state_loss",
                    "Model/eval_state_loss",
                    "Model/traj_autoregressive_error",
                    "Model/replay_size",
                    "Perf/total_fps",
                )
                if key in logged
            }
            print(f"[Go2-FlashSAC-RWM-Pretrain] step={interaction_step} env_step={env_step} {interesting}")
            collection_time_acc = 0.0
            learning_time_acc = 0.0
            env_steps_since_log = 0

        if (
            int(cfg.save_checkpoint_per_interaction_step)
            and interaction_step % int(cfg.save_checkpoint_per_interaction_step) == 0
        ):
            _save_checkpoint(
                save_root=save_root,
                interaction_step=interaction_step,
                env_step=env_step,
                agent=agent,
                dynamics=dynamics,
                dynamics_optimizer=dynamics_optimizer,
                replay=replay,
                world_model_cfg=world_model_cfg,
                cfg=cfg,
            )

    final_step = total_interaction_steps
    _save_checkpoint(
        save_root=save_root,
        interaction_step=final_step,
        env_step=final_step * num_envs,
        agent=agent,
        dynamics=dynamics,
        dynamics_optimizer=dynamics_optimizer,
        replay=replay,
        world_model_cfg=world_model_cfg,
        cfg=cfg,
    )
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
