"""Train Go2 FlashSAC entirely inside the learned RWM imagination env."""

from __future__ import annotations

import argparse
import sys
import time
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

    wm_cfg_dict = OmegaConf.to_container(cfg.world_model, resolve=True, throw_on_missing=True)
    assert isinstance(wm_cfg_dict, dict)
    wm_cfg = FlashSACWorldModelEnvConfig(**{str(k): v for k, v in wm_cfg_dict.items()})
    wm_cfg.num_envs = int(cfg.num_imagination_envs)
    env = Go2RWMFlashSACWorldModelEnv(dynamics=dynamics, dataset=dataset, cfg=wm_cfg, device=device)

    agent_cfg = make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_agent(env.observation_space, env.action_space, agent_cfg)

    save_path = str(cfg.save_path).replace("TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    save_root = resolve_repo_path(save_path)
    save_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, save_root / "rwm_flashsac_config.yaml")
    logger = ScalarLogger(save_root, enabled=str(cfg.logger_type).lower() == "tensorboard")

    num_envs = int(cfg.num_imagination_envs)
    total_interaction_steps = max(1, int(int(cfg.num_env_steps) // num_envs))
    update_counter = 0.0
    observations, _ = env.reset(seed=int(cfg.seed))
    transition: dict[str, Any] | None = None
    collection_time_acc = 0.0
    learning_time_acc = 0.0
    env_steps_since_log = 0

    print(f"[Go2-FlashSAC-RWM] model={model_path}")
    print(f"[Go2-FlashSAC-RWM] dataset={dataset_path}")
    print(f"[Go2-FlashSAC-RWM] save_root={save_root}")
    print(f"[Go2-FlashSAC-RWM] device={device}, num_envs={num_envs}, interaction_steps={total_interaction_steps}")

    for interaction_step in tqdm.tqdm(range(1, total_interaction_steps + 1), smoothing=0.1, mininterval=0.5):
        env_step = interaction_step * num_envs
        start = time.perf_counter()
        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(num_envs, dynamics.cfg.action_dim)).astype(np.float32)

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
                    "Perf/total_fps",
                )
                if k in logged
            }
            print(f"[Go2-FlashSAC-RWM] step={interaction_step} env_step={env_step} {interesting}")
            collection_time_acc = 0.0
            learning_time_acc = 0.0
            env_steps_since_log = 0

        if int(cfg.save_checkpoint_per_interaction_step) and interaction_step % int(cfg.save_checkpoint_per_interaction_step) == 0:
            _save_checkpoint(
                agent,
                save_root / f"step{interaction_step}",
                cfg,
                save_replay=bool(cfg.save_replay_buffer),
            )

    _save_checkpoint(
        agent,
        save_root / f"step{total_interaction_steps}",
        cfg,
        save_replay=bool(cfg.save_replay_buffer),
    )
    logger.close()
    env.close()


if __name__ == "__main__":
    main()
