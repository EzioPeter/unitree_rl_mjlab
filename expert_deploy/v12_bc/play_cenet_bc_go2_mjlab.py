"""Play a history-conditioned CENet-BC Go2 policy in MJLab."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from expert_deploy.v12_bc.cenet_bc import CENetBCModelConfig, CENetBCPolicy  # noqa: E402
from expert_deploy.v12_bc.play_bc_go2_mjlab import _configure_dataset_random_commands  # noqa: E402
from flash_rl.envs.mjlab import configure_mjlab_randomization  # noqa: E402


def load_policy(checkpoint_path: Path, device: torch.device) -> CENetBCPolicy:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model_config = CENetBCModelConfig(**checkpoint["model_config"])
    policy = CENetBCPolicy(model_config).to(device)
    policy.load_state_dict(checkpoint["network_state_dict"])
    return policy.eval()


class OnlineCENetPolicy:
    """Maintain the observation-history state required by CENet."""

    def __init__(self, policy: CENetBCPolicy) -> None:
        self.policy = policy
        self.history: torch.Tensor | None = None
        self.last_estimated_velocity: torch.Tensor | None = None

    def reset(self, observations: torch.Tensor) -> None:
        self.history = observations.unsqueeze(1).repeat(1, self.policy.config.history_length, 1)

    def reset_indices(self, env_ids: torch.Tensor, observations: torch.Tensor) -> None:
        if self.history is None or env_ids.numel() == 0:
            return
        self.history[env_ids] = observations[env_ids].unsqueeze(1).repeat(
            1,
            self.policy.config.history_length,
            1,
        )

    @torch.no_grad()
    def act(self, observations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.history is None or self.history.shape[0] != observations.shape[0]:
            self.reset(observations)
        assert self.history is not None
        actions, estimated_velocity = self.policy.predict(
            self.history,
            observations,
            training=False,
        )
        self.history = torch.cat((self.history[:, 1:], observations.unsqueeze(1)), dim=1)
        self.last_estimated_velocity = estimated_velocity
        return actions, estimated_velocity

    def __call__(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        actions, _estimated_velocity = self.act(observations["actor"])
        return actions


class MjlabViewerEnv:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.num_envs = env.num_envs

    @property
    def device(self) -> Any:
        return self._env.device

    @property
    def cfg(self) -> Any:
        return self._env.cfg

    @property
    def unwrapped(self) -> Any:
        return self._env.unwrapped if hasattr(self._env, "unwrapped") else self._env

    def get_observations(self) -> dict[str, torch.Tensor]:
        return self._env.observation_manager.compute()

    def step(self, actions: torch.Tensor) -> Any:
        return self._env.step(actions)

    def reset(self, **kwargs: Any) -> Any:
        return self._env.reset(**kwargs)

    def close(self) -> None:
        self._env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--frame_rate", type=float, default=50.0)
    parser.add_argument("--viewer", choices=("native", "none"), default="native")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--forward_only", action="store_true")
    parser.add_argument("--enable_randomization", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = torch.device(args.device or ("cuda:0" if torch.cuda.is_available() else "cpu"))
    checkpoint_path = (REPO_ROOT / args.checkpoint).resolve()
    dataset_path = (REPO_ROOT / args.dataset).resolve()

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(
        "Unitree-Go2-Flat-Normal-FixStand-Proprioceptive-Expert",
        play=True,
    )
    env_cfg.scene.num_envs = 1
    env_cfg.seed = args.seed
    env_cfg.auto_reset = True
    _configure_dataset_random_commands(
        env_cfg,
        dataset_path,
        forward_only=args.forward_only,
    )
    if not args.enable_randomization:
        configure_mjlab_randomization(
            env_cfg,
            use_domain_randomization=False,
            use_push_randomization=False,
            use_observation_noise=False,
        )
    env = ManagerBasedRlEnv(cfg=env_cfg, device=str(device))
    policy = OnlineCENetPolicy(load_policy(checkpoint_path, device))
    env.reset()
    initial_observation = env.observation_manager.compute()["actor"]
    policy.reset(initial_observation)
    print(f"[CENet-BC Play] checkpoint={checkpoint_path}")
    print(
        "[CENet-BC Play] commands="
        + ("forward-only" if args.forward_only else "dataset-random")
        + ", randomization="
        + ("enabled" if args.enable_randomization else "disabled (nominal G0)")
    )
    if args.viewer == "native":
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(MjlabViewerEnv(env), policy, frame_rate=args.frame_rate).run()
    else:
        for _ in range(args.num_steps):
            observations = env.observation_manager.compute()
            env.step(policy(observations))
    env.close()


if __name__ == "__main__":
    main()
