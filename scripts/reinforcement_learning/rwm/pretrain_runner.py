"""RSL-RL runner that trains a Go2 world model alongside PPO rollouts."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch

from mjlab.rl import MjlabOnPolicyRunner
from src.tasks.rwm_velocity.mdp.extractors import Go2RWMExtractor

from .dynamics import (
    DynamicsConfig,
    ReplayConfig,
    SequenceReplayBuffer,
    SystemDynamicsEnsemble,
    WorldModelConfig,
    train_world_model_steps,
)


def _default_world_model_config(
    state_dim: int,
    action_dim: int,
    contact_dim: int,
    termination_dim: int,
    overrides: dict[str, Any] | None = None,
) -> WorldModelConfig:
    cfg = WorldModelConfig(
        dynamics=DynamicsConfig(
            state_dim=state_dim,
            action_dim=action_dim,
            contact_dim=contact_dim,
            termination_dim=termination_dim,
        ),
        replay=ReplayConfig(),
    )
    overrides = overrides or {}
    for key, value in overrides.items():
        if hasattr(cfg.dynamics, key):
            setattr(cfg.dynamics, key, value)
        elif hasattr(cfg.replay, key):
            setattr(cfg.replay, key, value)
        elif hasattr(cfg, key):
            setattr(cfg, key, value)
        else:
            raise KeyError(f"Unknown RWM config override: {key}")
    return cfg


class Go2RWMPretrainRunner(MjlabOnPolicyRunner):
    """PPO runner with online system-dynamics ensemble training."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.rwm_cfg_overrides = dict(train_cfg.pop("rwm", {}))
        super().__init__(env, train_cfg, log_dir, device)

        self.extractor = Go2RWMExtractor(self.env.unwrapped)
        dims = self.extractor.dims
        self.world_model_cfg = _default_world_model_config(
            state_dim=dims.state_dim,
            action_dim=dims.action_dim,
            contact_dim=dims.contact_dim,
            termination_dim=dims.termination_dim,
            overrides=self.rwm_cfg_overrides,
        )
        self.system_dynamics = SystemDynamicsEnsemble(self.world_model_cfg.dynamics).to(self.device)
        self.system_dynamics_optimizer = torch.optim.Adam(
            self.system_dynamics.parameters(),
            lr=self.world_model_cfg.learning_rate,
            weight_decay=self.world_model_cfg.weight_decay,
        )
        self.replay = SequenceReplayBuffer(
            state_dim=dims.state_dim,
            action_dim=dims.action_dim,
            contact_dim=dims.contact_dim,
            termination_dim=dims.termination_dim,
            num_envs=self.env.num_envs,
            capacity=self.world_model_cfg.replay.capacity,
            device="cpu",
        )

    @property
    def _log_dir_path(self) -> Path | None:
        if self.logger.log_dir is None:
            return None
        return Path(self.logger.log_dir)

    def _save_policy(self, path: Path, infos: dict[str, Any] | None = None) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)

    def _save_world_model(self, iteration: int) -> None:
        log_dir = self._log_dir_path
        if log_dir is None:
            return
        log_dir.mkdir(parents=True, exist_ok=True)
        infos = {
            "replay_size": len(self.replay),
            "num_time_steps": self.replay.num_time_steps,
            "task": "Unitree-Go2-Flat-RWM-Pretrain-Ens",
        }
        checkpoint = self.system_dynamics.checkpoint(
            optimizer=self.system_dynamics_optimizer,
            iteration=iteration,
            infos=infos,
        )
        torch.save(checkpoint, log_dir / f"model_{iteration}.pt")
        if self.world_model_cfg.save_dataset:
            self.replay.save(log_dir / "dataset.pt")

    def _log_world_model_metrics(self, iteration: int, model_info: dict[str, float]) -> None:
        if not model_info:
            return
        renamed = {
            "total_loss": "train_total_loss",
            "state_loss": "train_state_loss",
            "sequence_loss": "train_sequence_loss",
            "contact_loss": "train_contact_loss",
            "termination_loss": "train_termination_loss",
            "eval_state_loss": "eval_state_loss",
            "traj_autoregressive_error": "traj_autoregressive_error",
        }
        metrics = {renamed.get(key, key): value for key, value in model_info.items()}
        metrics["replay_size"] = float(len(self.replay))

        writer = getattr(self.logger, "writer", None)
        if writer is not None:
            for key, value in metrics.items():
                writer.add_scalar(f"Model/{key}", value, iteration)

        summary_keys = [
            "train_state_loss",
            "eval_state_loss",
            "traj_autoregressive_error",
            "replay_size",
        ]
        summary = ", ".join(f"{key}={metrics[key]:.4f}" for key in summary_keys if key in metrics)
        print(f"[Go2-RWM] World model metrics: {summary}")

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf,
                high=int(self.env.max_episode_length),
            )

        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        self.logger.init_logging_writer()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    state = self.extractor.extract_state().to(self.device)
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    next_state = self.extractor.extract_state().to(self.device)
                    contact = self.extractor.extract_contact().to(self.device)
                    termination = self.extractor.extract_termination().to(self.device)
                    self.replay.add(
                        state=state,
                        action=actions,
                        next_state=next_state,
                        contact=contact,
                        termination=termination,
                    )
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)

                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
            model_info = train_world_model_steps(
                dynamics=self.system_dynamics,
                optimizer=self.system_dynamics_optimizer,
                replay=self.replay,
                cfg=self.world_model_cfg,
                device=self.device,
            )

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.get_policy().output_std,
                rnd_weight=self.alg.rnd.weight if self.cfg["algorithm"]["rnd_cfg"] else None,
            )
            self._log_world_model_metrics(it, model_info)

            log_dir = self._log_dir_path
            if log_dir is not None and it % self.cfg["save_interval"] == 0:
                self._save_policy(log_dir / f"policy_{it}.pt", infos={"rwm_replay_size": len(self.replay)})
                self._save_world_model(it)

        log_dir = self._log_dir_path
        if log_dir is not None:
            self._save_policy(log_dir / f"policy_{self.current_learning_iteration}.pt")
            self._save_world_model(self.current_learning_iteration)
            if self.logger.writer is not None:
                self.logger.stop_logging_writer()
