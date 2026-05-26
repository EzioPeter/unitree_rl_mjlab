"""Play a trained FlashSAC checkpoint in mjlab."""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"

import gymnasium as gym
import hydra
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.envs.mjlab import configure_mjlab_randomization


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"


class _MjlabViewerEnv:
    def __init__(self, env: Any) -> None:
        self._env = env
        self.num_envs: int = env.num_envs

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


class _FlashSACPolicy:
    def __init__(self, agent: Any, device: str, has_critic_obs: bool) -> None:
        self._agent = agent
        self._device = device
        self._has_critic_obs = has_critic_obs

    def __call__(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        flat = torch.cat([obs_dict["actor"], obs_dict["critic"]], dim=-1) if self._has_critic_obs else obs_dict["actor"]
        actions_np = self._agent.sample_actions(
            interaction_step=0,
            prev_transition={"next_observation": flat.cpu().numpy()},
            training=False,
        )
        return torch.from_numpy(actions_np).to(self._device)


def _compose_config(config_path: str, config_name: str, overrides: list[str]):
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda s: eval(s))
    GlobalHydra.instance().clear()
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = (REPO_ROOT / config_dir).resolve()
    hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def play(args: argparse.Namespace) -> None:
    cfg = _compose_config(args.config_path, args.config_name, args.overrides)

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(cfg.seed)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs.manager_based_rl_env import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg

    env_cfg = load_env_cfg(cfg.env.env_name, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = cfg.seed
    env_cfg.auto_reset = True
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=cfg.env.use_domain_randomization,
        use_push_randomization=cfg.env.use_push_randomization,
        use_observation_noise=cfg.env.use_observation_noise,
    )

    render_mode = "rgb_array" if args.video else None
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    env: Any = raw_env
    if args.video:
        from mjlab.utils.wrappers import VideoRecorder

        env = VideoRecorder(
            raw_env,
            video_folder=args.video,
            episode_trigger=lambda ep: ep == 0,
            disable_logger=True,
        )
        print(f"[INFO] Video will be saved to: {args.video}")

    viewer_env = _MjlabViewerEnv(env)

    obs_groups = list(raw_env.single_observation_space.spaces.keys())
    has_critic_obs = "actor" in obs_groups and "critic" in obs_groups
    actor_dim = int(raw_env.single_observation_space.spaces["actor"].shape[0])
    flat_dim = (
        actor_dim + int(raw_env.single_observation_space.spaces["critic"].shape[0])
        if has_critic_obs
        else actor_dim
    )
    action_dim = int(raw_env.single_action_space.shape[0])

    obs_space = batch_space(gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32), args.num_envs)
    act_space = batch_space(gym.spaces.Box(low=-np.inf, high=np.inf, shape=(action_dim,), dtype=np.float32), args.num_envs)
    env_info: dict[str, Any] = {}
    if has_critic_obs:
        env_info["actor_observation_size"] = (actor_dim,)

    agent = create_agent(
        observation_space=obs_space,
        action_space=act_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(args.checkpoint_path)
    policy = _FlashSACPolicy(agent, device=device, has_critic_obs=has_critic_obs)

    env.reset()
    with torch.no_grad():
        obs = env.observation_manager.compute()
        env.step(policy(obs))

    viewer_type = args.viewer
    if viewer_type == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        viewer_type = "native" if has_display else "viser"
        print(f"[INFO] Viewer auto-selected: {viewer_type}")

    if viewer_type == "none":
        env.reset()
        for _ in range(args.num_steps):
            obs_dict = env.observation_manager.compute()
            env.step(policy(obs_dict))
    elif viewer_type == "native":
        from mjlab.viewer import NativeMujocoViewer

        NativeMujocoViewer(viewer_env, policy).run()
    elif viewer_type == "viser":
        from mjlab.viewer import ViserPlayViewer

        ViserPlayViewer(viewer_env, policy).run()
    else:
        raise ValueError(f"Unknown viewer: {viewer_type!r}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent in mjlab")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config_name", type=str, default="flashsac_g1_velocity")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", type=str, default="auto", choices=["auto", "native", "viser", "none"])
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--video", type=str, default=None, metavar="OUTPUT_DIR")
    play(parser.parse_args())
