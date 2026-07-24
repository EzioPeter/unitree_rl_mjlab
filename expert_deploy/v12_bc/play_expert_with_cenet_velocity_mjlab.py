"""Play the E0 FlashSAC expert while evaluating CENet velocity estimates."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expert_deploy.v12_bc.play_bc_go2_mjlab import _configure_dataset_random_commands  # noqa: E402
from expert_deploy.v12_bc.play_cenet_bc_go2_mjlab import load_policy as load_cenet_policy  # noqa: E402
from flash_rl.agents import create_agent  # noqa: E402
from flash_rl.envs.mjlab import (  # noqa: E402
    configure_mjlab_randomization,
    normalize_action_mask_indices,
)
from scripts.flashsac_mjlab_overrides import apply_mjlab_env_overrides  # noqa: E402
from scripts.play_flashsac_mjlab import _FlashSACPolicy, _MjlabViewerEnv  # noqa: E402


class VelocityMetrics:
    def __init__(self, device: torch.device) -> None:
        self.count = 0
        self.sum_true = torch.zeros(3, device=device, dtype=torch.float64)
        self.sum_predicted = torch.zeros(3, device=device, dtype=torch.float64)
        self.sum_true_squared = torch.zeros(3, device=device, dtype=torch.float64)
        self.sum_predicted_squared = torch.zeros(3, device=device, dtype=torch.float64)
        self.sum_cross = torch.zeros(3, device=device, dtype=torch.float64)
        self.sum_squared_error = torch.zeros(3, device=device, dtype=torch.float64)
        self.sum_absolute_error = torch.zeros(3, device=device, dtype=torch.float64)

    def update(self, predicted: torch.Tensor, target: torch.Tensor) -> None:
        predicted = predicted.double()
        target = target.double()
        error = predicted - target
        self.count += target.shape[0]
        self.sum_true += target.sum(dim=0)
        self.sum_predicted += predicted.sum(dim=0)
        self.sum_true_squared += target.square().sum(dim=0)
        self.sum_predicted_squared += predicted.square().sum(dim=0)
        self.sum_cross += (target * predicted).sum(dim=0)
        self.sum_squared_error += error.square().sum(dim=0)
        self.sum_absolute_error += error.abs().sum(dim=0)

    def report(self) -> dict[str, Any]:
        if self.count == 0:
            return {"samples": 0}
        count = float(self.count)
        mse = self.sum_squared_error / count
        mae = self.sum_absolute_error / count
        bias = (self.sum_predicted - self.sum_true) / count
        target_ss = self.sum_true_squared - self.sum_true.square() / count
        covariance_numerator = count * self.sum_cross - self.sum_true * self.sum_predicted
        correlation_denominator = torch.sqrt(
            torch.clamp(
                (count * self.sum_true_squared - self.sum_true.square())
                * (count * self.sum_predicted_squared - self.sum_predicted.square()),
                min=1.0e-12,
            )
        )
        r2 = 1.0 - self.sum_squared_error / torch.clamp(target_ss, min=1.0e-12)
        correlation = covariance_numerator / correlation_denominator
        return {
            "samples": self.count,
            "mse": float(mse.mean()),
            "rmse": float(torch.sqrt(mse.mean())),
            "mae": float(mae.mean()),
            "rmse_per_axis": torch.sqrt(mse).tolist(),
            "mae_per_axis": mae.tolist(),
            "bias_per_axis": bias.tolist(),
            "r2_per_axis": r2.tolist(),
            "correlation_per_axis": correlation.tolist(),
            "target_mean_per_axis": (self.sum_true / count).tolist(),
            "predicted_mean_per_axis": (self.sum_predicted / count).tolist(),
        }


class ExpertWithCENetMonitor:
    """Use expert actions while running CENet as a read-only sidecar."""

    def __init__(
        self,
        *,
        expert_policy: _FlashSACPolicy,
        cenet_policy: torch.nn.Module,
        env: Any,
        warmup_steps: int,
        log_interval: int,
    ) -> None:
        self.expert_policy = expert_policy
        self.cenet = cenet_policy.cenet
        self.history_length = cenet_policy.config.history_length
        self.env = env
        self.warmup_steps = int(warmup_steps)
        self.log_interval = int(log_interval)
        self.history: torch.Tensor | None = None
        self.step_count = 0
        self.metrics = VelocityMetrics(torch.device(env.device))

    def reset(self, observations: torch.Tensor) -> None:
        self.history = observations.unsqueeze(1).repeat(1, self.history_length, 1)

    def reset_indices(self, env_ids: torch.Tensor, observations: torch.Tensor) -> None:
        if self.history is None or env_ids.numel() == 0:
            return
        self.history[env_ids] = observations[env_ids].unsqueeze(1).repeat(
            1,
            self.history_length,
            1,
        )

    @torch.no_grad()
    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        actor_observation = observations["actor"]
        if self.history is None or self.history.shape[0] != actor_observation.shape[0]:
            self.reset(actor_observation)
        assert self.history is not None
        estimated_velocity, _latent = self.cenet(self.history)
        true_velocity = self.env.scene["robot"].data.root_link_lin_vel_b
        if self.step_count >= self.warmup_steps:
            self.metrics.update(estimated_velocity, true_velocity)
        self.history = torch.cat((self.history[:, 1:], actor_observation.unsqueeze(1)), dim=1)
        self.step_count += 1
        if (
            self.log_interval > 0
            and self.step_count > self.warmup_steps
            and self.step_count % self.log_interval == 0
        ):
            report = self.metrics.report()
            print(
                f"[Expert+CENet] step={self.step_count} "
                f"velocity_rmse={report['rmse']:.6f} m/s "
                f"axis_rmse={report['rmse_per_axis']} "
                f"r2={report['r2_per_axis']}",
                flush=True,
            )
        return self.expert_policy(observations)


def build_expert_policy(
    *,
    cfg: Any,
    checkpoint_path: Path,
    env: Any,
    device: torch.device,
) -> _FlashSACPolicy:
    obs_groups = list(env.single_observation_space.spaces.keys())
    has_critic_obs = "actor" in obs_groups and "critic" in obs_groups
    use_critic_as_full = bool(
        cfg.env.get("use_critic_observation_as_full_observation", False)
    ) and has_critic_obs
    actor_dim = int(env.single_observation_space.spaces["actor"].shape[0])
    if use_critic_as_full:
        flat_dim = int(env.single_observation_space.spaces["critic"].shape[0])
    else:
        flat_dim = (
            actor_dim + int(env.single_observation_space.spaces["critic"].shape[0])
            if has_critic_obs
            else actor_dim
        )
    full_action_dim = int(env.single_action_space.shape[0])
    action_mask_indices = normalize_action_mask_indices(
        cfg.env.get("action_mask_indices", []),
        action_dim=full_action_dim,
    )
    policy_action_dim = full_action_dim - len(action_mask_indices)
    obs_space = batch_space(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32),
        env.num_envs,
    )
    action_space = batch_space(
        gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(policy_action_dim,),
            dtype=np.float32,
        ),
        env.num_envs,
    )
    env_info: dict[str, Any] = {}
    if has_critic_obs:
        env_info["actor_observation_size"] = (actor_dim,)
    agent = create_agent(
        observation_space=obs_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(str(checkpoint_path))
    return _FlashSACPolicy(
        agent,
        device=str(device),
        has_critic_obs=has_critic_obs,
        use_critic_observation_as_full_observation=use_critic_as_full,
        full_action_dim=full_action_dim,
        action_mask_indices=action_mask_indices,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expert_checkpoint", required=True)
    parser.add_argument("--expert_config", required=True)
    parser.add_argument("--cenet_checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--viewer", choices=("native", "none"), default="native")
    parser.add_argument("--frame_rate", type=float, default=50.0)
    parser.add_argument("--warmup_steps", type=int, default=10)
    parser.add_argument("--log_interval", type=int, default=250)
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--enable_randomization", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    expert_checkpoint = (REPO_ROOT / args.expert_checkpoint).resolve()
    expert_config = (REPO_ROOT / args.expert_config).resolve()
    cenet_checkpoint = (REPO_ROOT / args.cenet_checkpoint).resolve()
    dataset_path = (REPO_ROOT / args.dataset).resolve()
    cfg = OmegaConf.load(expert_config)
    OmegaConf.resolve(cfg)

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(cfg.env.env_name, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    apply_mjlab_env_overrides(env_cfg, cfg)
    _configure_dataset_random_commands(
        env_cfg,
        dataset_path,
        forward_only=args.forward_only,
    )
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=args.enable_randomization,
        use_push_randomization=args.enable_randomization,
        use_observation_noise=args.enable_randomization,
        randomization_preset=(
            str(cfg.env.get("randomization_preset", "default"))
            if args.enable_randomization
            else "default"
        ),
        randomization_components=cfg.env.get("randomization_components", None),
        randomization_scale=float(cfg.env.get("randomization_scale", 1.0)),
    )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
    expert_policy = build_expert_policy(
        cfg=cfg,
        checkpoint_path=expert_checkpoint,
        env=env,
        device=device,
    )
    cenet_policy = load_cenet_policy(cenet_checkpoint, device)
    monitor = ExpertWithCENetMonitor(
        expert_policy=expert_policy,
        cenet_policy=cenet_policy,
        env=env,
        warmup_steps=args.warmup_steps,
        log_interval=args.log_interval,
    )
    env.reset()
    observations = env.observation_manager.compute()
    monitor.reset(observations["actor"])
    terminated_count = 0
    timeout_count = 0
    print(f"[Expert+CENet] expert={expert_checkpoint}")
    print(f"[Expert+CENet] cenet={cenet_checkpoint}")
    print("[Expert+CENet] BC actor is not used; CENet is a read-only velocity monitor.")
    if args.viewer == "native":
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(_MjlabViewerEnv(env), monitor, frame_rate=args.frame_rate).run()
    else:
        for _step in range(args.num_steps):
            actions = monitor(observations)
            observations, _rewards, terminated, truncated, _extras = env.step(actions)
            terminated_count += int(terminated.sum().item())
            timeout_count += int(truncated.sum().item())
            done_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
            monitor.reset_indices(done_ids, observations["actor"])
    report = {
        "expert_checkpoint": str(expert_checkpoint),
        "cenet_checkpoint": str(cenet_checkpoint),
        "dataset": str(dataset_path),
        "seed": args.seed,
        "num_envs": args.num_envs,
        "steps": monitor.step_count,
        "warmup_steps": args.warmup_steps,
        "forward_only": args.forward_only,
        "randomization": args.enable_randomization,
        "bc_actor_used": False,
        "terminations": terminated_count,
        "timeouts": timeout_count,
        "velocity_estimation": monitor.metrics.report(),
    }
    print(json.dumps(report, indent=2), flush=True)
    if args.output:
        output_path = (REPO_ROOT / args.output).resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    env.close()


if __name__ == "__main__":
    main()
