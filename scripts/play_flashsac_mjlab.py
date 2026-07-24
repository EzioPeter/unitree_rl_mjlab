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
from flash_rl.envs.mjlab import (
    configure_mjlab_randomization,
    expand_masked_actions_t,
    normalize_action_mask_indices,
)
from scripts.flashsac_mjlab_overrides import apply_mjlab_env_overrides


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
FLASHSAC_CONFIG_FILENAME = "flashsac_config.yaml"


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
    def __init__(
        self,
        agent: Any,
        device: str,
        has_critic_obs: bool,
        use_critic_observation_as_full_observation: bool,
        full_action_dim: int,
        action_mask_indices: tuple[int, ...],
        manual_command: bool = False,
        env: Any | None = None,
    ) -> None:
        self._agent = agent
        self._device = device
        self._has_critic_obs = has_critic_obs
        self._use_critic_observation_as_full_observation = use_critic_observation_as_full_observation
        self._full_action_dim = full_action_dim
        self._action_mask_indices = action_mask_indices
        self._manual_command = manual_command
        self._env = env

    def _apply_manual_command(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self._manual_command or self._env is None:
            return obs_dict
        from scripts.reinforcement_learning.rwm_flashsac import play_flashsac_go2_mjlab as viewer_helpers

        command = viewer_helpers._manual_gui_command(self._env)
        viewer_helpers._force_fixed_command(self._env, command)
        actor = obs_dict["actor"].clone()
        actor_dim = int(actor.shape[-1])
        command_start = 6 if actor_dim == 45 else 9
        actor[:, command_start : command_start + 3] = torch.tensor(
            command,
            dtype=actor.dtype,
            device=actor.device,
        )
        updated = dict(obs_dict)
        updated["actor"] = actor
        return updated

    def __call__(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        obs_dict = self._apply_manual_command(obs_dict)
        if self._has_critic_obs and self._use_critic_observation_as_full_observation:
            flat = obs_dict["critic"]
        elif self._has_critic_obs:
            flat = torch.cat([obs_dict["actor"], obs_dict["critic"]], dim=-1)
        else:
            flat = obs_dict["actor"]
        actions_np = self._agent.sample_actions(
            interaction_step=0,
            prev_transition={"next_observation": flat.cpu().numpy()},
            training=False,
        )
        actions_t = torch.from_numpy(actions_np).to(self._device)
        return expand_masked_actions_t(
            actions_t,
            full_action_dim=self._full_action_dim,
            action_mask_indices=self._action_mask_indices,
        )


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


def _load_saved_config(config_file: Path, overrides: list[str]):
    if not config_file.is_absolute():
        config_file = (REPO_ROOT / config_file).resolve()
    cfg = OmegaConf.load(config_file)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(overrides))
    OmegaConf.resolve(cfg)
    return cfg


def _find_checkpoint_config(checkpoint_path: Path) -> Path | None:
    candidates = (
        checkpoint_path / FLASHSAC_CONFIG_FILENAME,
        checkpoint_path.parent / FLASHSAC_CONFIG_FILENAME,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _load_play_config(args: argparse.Namespace):
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if args.config_file is not None:
        return _load_saved_config(Path(args.config_file).expanduser(), args.overrides), checkpoint_path
    checkpoint_config = _find_checkpoint_config(checkpoint_path)
    if checkpoint_config is not None:
        print(f"[INFO] Loading FlashSAC config: {checkpoint_config}")
        return _load_saved_config(checkpoint_config, args.overrides), checkpoint_path
    return _compose_config(args.config_path, args.config_name, args.overrides), checkpoint_path


def play(args: argparse.Namespace) -> None:
    cfg, checkpoint_path = _load_play_config(args)

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
    from scripts.reinforcement_learning.rwm_flashsac import play_flashsac_go2_mjlab as viewer_helpers

    env_cfg = load_env_cfg(cfg.env.env_name, play=True)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = cfg.seed
    env_cfg.auto_reset = True
    apply_mjlab_env_overrides(env_cfg, cfg)
    if args.manual_command:
        viewer_helpers._configure_command_ranges(
            env_cfg,
            fixed_command=None,
            command_sequence=[],
            random_ranges=(
                tuple(args.manual_lin_vel_x),
                tuple(args.manual_lin_vel_y),
                tuple(args.manual_ang_vel_z),
            ),
            manual_command=True,
        )
    joint_strength_scales = {
        str(name): float(scale)
        for name, scale in (cfg.env.get("joint_strength_scales", {}) or {}).items()
    }
    joint_strength_scales.update(viewer_helpers._parse_joint_strength_scales(args.joint_strength_scales))
    broken_joint_names = tuple(cfg.env.get("broken_joint_names", []) or ())
    for joint_name in broken_joint_names:
        joint_strength_scales[str(joint_name)] = 0.0
    if joint_strength_scales:
        from scripts.reinforcement_learning.rwm_dataset.broken_go2 import apply_go2_pd_joint_strength_scales

        apply_go2_pd_joint_strength_scales(env_cfg, joint_strength_scales)
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=cfg.env.use_domain_randomization,
        use_push_randomization=cfg.env.use_push_randomization,
        use_observation_noise=cfg.env.use_observation_noise,
    )

    render_mode = "rgb_array" if args.video else None
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if args.manual_command:
        viewer_helpers._force_fixed_command(raw_env, (0.0, 0.0, 0.0))

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
    use_critic_observation_as_full_observation = bool(
        cfg.env.get("use_critic_observation_as_full_observation", False)
    ) and has_critic_obs
    actor_dim = int(raw_env.single_observation_space.spaces["actor"].shape[0])
    if use_critic_observation_as_full_observation:
        flat_dim = int(raw_env.single_observation_space.spaces["critic"].shape[0])
    else:
        flat_dim = (
            actor_dim + int(raw_env.single_observation_space.spaces["critic"].shape[0])
            if has_critic_obs
            else actor_dim
        )
    full_action_dim = int(raw_env.single_action_space.shape[0])
    action_mask_indices = normalize_action_mask_indices(
        cfg.env.get("action_mask_indices", []),
        action_dim=full_action_dim,
    )
    policy_action_dim = full_action_dim - len(action_mask_indices)

    obs_space = batch_space(gym.spaces.Box(low=-np.inf, high=np.inf, shape=(flat_dim,), dtype=np.float32), args.num_envs)
    act_space = batch_space(
        gym.spaces.Box(low=-np.inf, high=np.inf, shape=(policy_action_dim,), dtype=np.float32),
        args.num_envs,
    )
    env_info: dict[str, Any] = {}
    if has_critic_obs:
        env_info["actor_observation_size"] = (actor_dim,)

    agent = create_agent(
        observation_space=obs_space,
        action_space=act_space,
        env_info=env_info,
        cfg=cfg.agent,
    )
    agent.load(str(checkpoint_path))
    policy = _FlashSACPolicy(
        agent,
        device=device,
        has_critic_obs=has_critic_obs,
        use_critic_observation_as_full_observation=use_critic_observation_as_full_observation,
        full_action_dim=full_action_dim,
        action_mask_indices=action_mask_indices,
        manual_command=args.manual_command,
        env=viewer_env,
    )

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

        NativeMujocoViewer(viewer_env, policy, frame_rate=args.frame_rate).run()
    elif viewer_type == "viser":
        import viser
        from mjlab.viewer import ViserPlayViewer

        server = viser.ViserServer(port=args.viser_port, label="mjlab")
        ViserPlayViewer(viewer_env, policy, frame_rate=args.frame_rate, viser_server=server).run()
    else:
        raise ValueError(f"Unknown viewer: {viewer_type!r}")

    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play a trained FlashSAC agent in mjlab")
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config_name", type=str, default="flashsac_g1_velocity")
    parser.add_argument("--config_file", type=str, default=None)
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--viewer", type=str, default="auto", choices=["auto", "native", "viser", "none"])
    parser.add_argument("--frame_rate", type=float, default=60.0)
    parser.add_argument("--viser_port", type=int, default=8080)
    parser.add_argument("--manual_command", action="store_true")
    parser.add_argument("--manual_lin_vel_x", type=float, nargs=2, default=(-0.5, 0.5))
    parser.add_argument("--manual_lin_vel_y", type=float, nargs=2, default=(-0.2, 0.2))
    parser.add_argument("--manual_ang_vel_z", type=float, nargs=2, default=(-0.4, 0.4))
    parser.add_argument("--joint_strength_scales", nargs="*", default=(), metavar="JOINT=SCALE")
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--video", type=str, default=None, metavar="OUTPUT_DIR")
    play(parser.parse_args())
