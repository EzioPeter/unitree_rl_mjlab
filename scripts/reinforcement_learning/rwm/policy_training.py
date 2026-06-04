"""Policy runner for RWM imagination training."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

import torch
from rsl_rl.runners import OnPolicyRunner


class RWMPolicyRunner(OnPolicyRunner):
    """OnPolicyRunner variant that saves policy_<iter>.pt checkpoints."""

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        for key in ("actor", "critic"):
            if key in train_cfg:
                for opt in ("cnn_cfg", "distribution_cfg"):
                    if train_cfg[key].get(opt) is None:
                        train_cfg[key].pop(opt, None)
        super().__init__(env, train_cfg, log_dir, device)

    def save_policy(self, path: str | Path, infos: dict[str, Any] | None = None) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        saved_dict = self.alg.save()
        saved_dict["iter"] = self.current_learning_iteration
        saved_dict["infos"] = infos
        torch.save(saved_dict, path)

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
                    actions = self.alg.act(obs)
                    obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                    obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    intrinsic_rewards = self.alg.intrinsic_rewards if self.cfg["algorithm"]["rnd_cfg"] else None
                    self.logger.process_env_step(rewards, dones, extras, intrinsic_rewards)
                stop = time.time()
                collect_time = stop - start
                start = stop
                self.alg.compute_returns(obs)

            loss_dict = self.alg.update()
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
            if self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0:
                self.save_policy(Path(self.logger.log_dir) / f"policy_{it}.pt")

        if self.logger.log_dir is not None:
            self.save_policy(Path(self.logger.log_dir) / f"policy_{self.current_learning_iteration}.pt")
            if self.logger.writer is not None:
                self.logger.stop_logging_writer()
