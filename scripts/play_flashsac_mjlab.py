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


def _attach_fixed_payload_brick(
    env_cfg: Any,
    *,
    mass_kg: float,
    position_body_m: tuple[float, float, float] = (0.0, 0.0, 0.10),
    box_size_m: tuple[float, float, float] = (0.20, 0.12, 0.05),
) -> None:
    """Attach a visible, rigid MuJoCo box body to Go2's base_link."""

    if mass_kg < 0.0:
        raise ValueError(f"payload mass must be non-negative, got {mass_kg}")
    if mass_kg == 0.0:
        return
    robot_cfg = env_cfg.scene.entities["robot"]
    original_spec_fn = robot_cfg.spec_fn
    half_size = tuple(float(value) * 0.5 for value in box_size_m)

    def spec_with_payload_brick():
        import mujoco

        spec = original_spec_fn()
        base_body = spec.body("base_link")
        payload_body = base_body.add_body(
            name="fixed_payload_brick",
            pos=position_body_m,
        )
        payload_body.add_geom(
            name="fixed_payload_brick_geom",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=half_size,
            mass=float(mass_kg),
            contype=0,
            conaffinity=0,
            group=2,
            rgba=(0.72, 0.22, 0.08, 1.0),
        )
        return spec

    robot_cfg.spec_fn = spec_with_payload_brick


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
        random_command: bool = False,
        command_switch_steps: int = 250,
        random_ranges: tuple[
            tuple[float, float],
            tuple[float, float],
            tuple[float, float],
        ] = ((-0.8, 0.8), (-0.3, 0.3), (-0.6, 0.6)),
        random_stand_prob: float = 0.1,
        env: Any | None = None,
    ) -> None:
        self._agent = agent
        self._device = device
        self._has_critic_obs = has_critic_obs
        self._use_critic_observation_as_full_observation = use_critic_observation_as_full_observation
        self._full_action_dim = full_action_dim
        self._action_mask_indices = action_mask_indices
        self._manual_command = manual_command
        self._random_command = random_command
        self._command_switch_steps = max(1, int(command_switch_steps))
        self._random_ranges = random_ranges
        self._random_stand_prob = float(random_stand_prob)
        self._step_count = 0
        self._sampled_command: tuple[float, float, float] | None = None
        self._env = env

    def _current_command(self) -> tuple[float, float, float] | None:
        if self._env is None:
            return None
        if self._manual_command:
            from scripts.reinforcement_learning.rwm_flashsac import play_flashsac_go2_mjlab as viewer_helpers

            return viewer_helpers._manual_gui_command(self._env)
        if not self._random_command:
            return None
        if self._sampled_command is None or self._step_count % self._command_switch_steps == 0:
            if random.random() < self._random_stand_prob:
                self._sampled_command = (0.0, 0.0, 0.0)
            else:
                self._sampled_command = tuple(
                    random.uniform(*limits) for limits in self._random_ranges
                )
            print(f"[FlashSAC-Play] sampled_command={self._sampled_command}")
        return self._sampled_command

    def _apply_command(self, obs_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        command = self._current_command()
        if command is None or self._env is None:
            return obs_dict
        from scripts.reinforcement_learning.rwm_flashsac import play_flashsac_go2_mjlab as viewer_helpers

        viewer_helpers._force_fixed_command(self._env, command)
        updated = dict(obs_dict)
        actor_dim = int(obs_dict["actor"].shape[-1])
        command_start = 6 if actor_dim == 45 else 9
        for group_name in ("actor", "critic"):
            observation = obs_dict.get(group_name)
            if observation is None:
                continue
            observation = observation.clone()
            observation[:, command_start : command_start + 3] = torch.tensor(
                command,
                dtype=observation.dtype,
                device=observation.device,
            )
            updated[group_name] = observation
        return updated

    def __call__(self, obs_dict: dict[str, torch.Tensor]) -> torch.Tensor:
        obs_dict = self._apply_command(obs_dict)
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
        self._step_count += 1
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
    env_cfg.events.pop("randomize_terrain", None)
    _attach_fixed_payload_brick(env_cfg, mass_kg=float(args.payload_brick_mass_kg))
    if args.manual_command and args.random_command:
        raise ValueError("--manual_command and --random_command cannot be used together.")
    random_ranges = (
        tuple(args.random_lin_vel_x),
        tuple(args.random_lin_vel_y),
        tuple(args.random_ang_vel_z),
    )
    # Viser's velocity GUI requires each displayed maximum to be at least 0.1.
    # Keep the policy sampler's exact ranges (including fixed-zero axes), while
    # widening only the command-manager GUI bounds when an axis is fixed at zero.
    gui_safe_random_ranges = tuple(
        (float(limits[0]), max(0.1, float(limits[1]))) for limits in random_ranges
    )
    if args.manual_command or args.random_command:
        viewer_helpers._configure_command_ranges(
            env_cfg,
            fixed_command=None,
            command_sequence=[],
            random_ranges=gui_safe_random_ranges,
            manual_command=bool(args.manual_command),
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
    if not 0.0 <= float(args.rr_calf_strength) <= 1.0:
        raise ValueError(f"--rr_calf_strength must be in [0, 1], got {args.rr_calf_strength}")
    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=cfg.env.use_domain_randomization,
        use_push_randomization=cfg.env.use_push_randomization,
        use_observation_noise=cfg.env.use_observation_noise,
        rr_calf_strength_range=(args.rr_calf_strength, args.rr_calf_strength),
    )

    render_mode = "rgb_array" if args.video else None
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    if args.manual_command or args.random_command:
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
        random_command=args.random_command,
        command_switch_steps=args.command_switch_steps,
        random_ranges=random_ranges,
        random_stand_prob=args.random_stand_prob,
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
    parser.add_argument("--random_command", action="store_true")
    parser.add_argument("--command_switch_steps", type=int, default=250)
    parser.add_argument("--random_lin_vel_x", type=float, nargs=2, default=(-0.8, 0.8))
    parser.add_argument("--random_lin_vel_y", type=float, nargs=2, default=(-0.3, 0.3))
    parser.add_argument("--random_ang_vel_z", type=float, nargs=2, default=(-0.6, 0.6))
    parser.add_argument("--random_stand_prob", type=float, default=0.1)
    parser.add_argument("--manual_lin_vel_x", type=float, nargs=2, default=(-0.5, 0.5))
    parser.add_argument("--manual_lin_vel_y", type=float, nargs=2, default=(-0.2, 0.2))
    parser.add_argument("--manual_ang_vel_z", type=float, nargs=2, default=(-0.4, 0.4))
    parser.add_argument("--joint_strength_scales", nargs="*", default=(), metavar="JOINT=SCALE")
    parser.add_argument(
        "--rr_calf_strength",
        type=float,
        default=1.0,
        help="Fixed RR calf strength applied by the same MJLab gap event used for collection and evaluation.",
    )
    parser.add_argument(
        "--payload_brick_mass_kg",
        type=float,
        default=0.0,
        help="Mass of a visible rigid box body attached above base_link.",
    )
    parser.add_argument("--num_steps", type=int, default=1000)
    parser.add_argument("--video", type=str, default=None, metavar="OUTPUT_DIR")
    play(parser.parse_args())
