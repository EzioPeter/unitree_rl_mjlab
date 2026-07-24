"""Train FlashSAC on mjlab Unitree tasks and export G1 deploy ONNX policies."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ["OMP_NUM_THREADS"] = "2"
os.environ["MKL_NUM_THREADS"] = "2"
os.environ["NUMEXPR_NUM_THREADS"] = "2"
os.environ["JAX_DEFAULT_MATMUL_PRECISION"] = "highest"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

import hydra
import numpy as np
import torch
import tqdm
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from flash_rl.agents import create_agent
from flash_rl.common import create_logger
from flash_rl.envs.mjlab import MjlabVectorEnv, configure_mjlab_randomization
from flash_rl.evaluation import evaluate, record_video
from flash_rl.export import export_flashsac_policy_to_onnx
from flash_rl.types import Tensor
from scripts.flashsac_mjlab_overrides import apply_mjlab_env_overrides


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
FLASHSAC_CONFIG_FILENAME = "flashsac_config.yaml"
SAVE_LOG_KEYS = (
    "Train/mean_reward",
    "Train/mean_episode_length",
    "Reward/track_linear_velocity",
    "Reward/track_angular_velocity",
    "Episode_Reward/track_linear_velocity",
    "Episode_Reward/track_angular_velocity",
    "Metrics/lin_vel_xy_error_mean",
    "Metrics/lin_vel_x_error_mean",
    "Metrics/lin_vel_y_error_mean",
    "Metrics/yaw_vel_error_mean",
    "Metrics/yaw_drift_when_no_yaw",
    "Metrics/command_lin_vel_xy_mean",
    "Metrics/base_lin_vel_xy_mean",
    "critic/loss",
    "actor/loss",
    "Perf/total_fps",
)
G1_REFERENCE_ONNX = REPO_ROOT / "deploy/robots/g1/config/policy/velocity/v0/exported/policy.onnx"
G1_DEPLOY_YAML = REPO_ROOT / "deploy/robots/g1/config/policy/velocity/v0/params/deploy.yaml"
G1_FLASHSAC_POLICY_DIR = REPO_ROOT / "deploy/robots/g1/config/policy/velocity/v1_flashsac"


def _register_resolvers() -> None:
    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", lambda s: eval(s))


def _compose_config(config_path: str, config_name: str, overrides: list[str]):
    _register_resolvers()
    GlobalHydra.instance().clear()
    config_dir = Path(config_path)
    if not config_dir.is_absolute():
        config_dir = (REPO_ROOT / config_dir).resolve()
    hydra.initialize_config_dir(version_base=None, config_dir=str(config_dir))
    cfg = hydra.compose(config_name=config_name, overrides=overrides)
    OmegaConf.resolve(cfg)
    return cfg


def _save_config(cfg, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=path)


def _perf_metrics(env_steps: int, collection_time: float, learning_time: float) -> dict[str, float]:
    total_time = collection_time + learning_time
    if total_time <= 0.0:
        return {}
    return {
        "Perf/total_fps": float(env_steps / total_time),
        "Perf/collection_time": float(collection_time),
        "Perf/learning_time": float(learning_time),
    }


def _logger_scalar_snapshot(logger) -> dict[str, float]:
    average_meter_dict = getattr(logger, "average_meter_dict", None)
    if average_meter_dict is None:
        return {}
    return {
        key: float(value)
        for key, value in average_meter_dict.averages().items()
        if isinstance(value, (float, int))
    }


def _checkpoint_log_summary(metrics: dict[str, float]) -> dict[str, float]:
    summary = {key: metrics[key] for key in SAVE_LOG_KEYS if key in metrics}
    if summary:
        return summary
    return {
        key: value
        for key, value in metrics.items()
        if key.startswith(("Train/", "Reward/", "Episode_Reward/", "Metrics/", "Perf/"))
    }


def _print_checkpoint_log(interaction_step: int, env_step: int, metrics: dict[str, float]) -> None:
    print(
        f"[FlashSAC] step={interaction_step} env_step={env_step} {_checkpoint_log_summary(metrics)}",
        flush=True,
    )


def _export_g1_policy(checkpoint_path: Path, cfg, deploy_policy_dir: Path) -> None:
    checkpoint_onnx = checkpoint_path / "exported" / "policy.onnx"
    export_flashsac_policy_to_onnx(
        checkpoint_path=checkpoint_path,
        agent_cfg=cfg.agent,
        reference_onnx_path=G1_REFERENCE_ONNX,
        output_onnx_path=checkpoint_onnx,
        deploy_yaml_path=G1_DEPLOY_YAML,
    )

    deploy_exported = deploy_policy_dir / "exported"
    deploy_params = deploy_policy_dir / "params"
    deploy_exported.mkdir(parents=True, exist_ok=True)
    deploy_params.mkdir(parents=True, exist_ok=True)
    shutil.copy2(checkpoint_onnx, deploy_exported / "policy.onnx")
    checkpoint_external_data = checkpoint_onnx.with_suffix(checkpoint_onnx.suffix + ".data")
    if checkpoint_external_data.exists():
        shutil.copy2(checkpoint_external_data, deploy_exported / "policy.onnx.data")
    shutil.copy2(G1_DEPLOY_YAML, deploy_params / "deploy.yaml")
    print(f"[FlashSAC] Exported G1 deploy policy: {deploy_exported / 'policy.onnx'}")


def _maybe_evaluate(agent, env, num_episodes: int, env_type: str) -> dict[str, float]:
    if num_episodes <= 0:
        return {}
    return evaluate(agent, env, num_episodes, env_type)


def _maybe_record_video(agent, env, num_episodes: int, env_type: str) -> dict:
    if num_episodes <= 0:
        return {}
    return record_video(agent, env, num_episodes, env_type)


def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    if node is None:
        return default
    if hasattr(node, "get"):
        return node.get(key, default)
    return getattr(node, key, default)


def _joint_strength_scales_from_cfg(value: Any) -> dict[str, float]:
    if value is None:
        return {}
    if not hasattr(value, "items"):
        raise ValueError(f"joint_strength_scales must be a mapping, got {type(value).__name__}.")
    scales = {str(name): float(scale) for name, scale in value.items() if str(name)}
    bad = {name: scale for name, scale in scales.items() if scale < 0.0 or scale > 1.0}
    if bad:
        raise ValueError(f"joint strength scales must be in [0, 1], got {bad}.")
    return scales


@dataclass(frozen=True)
class ActuatorCurriculumStage:
    index: int
    name: str
    end_fraction: float
    joint_strength_scales: dict[str, float]


def _build_actuator_curriculum(cfg) -> tuple[ActuatorCurriculumStage, ...]:
    curriculum = _cfg_get(cfg.env, "actuator_curriculum", None)
    if curriculum is None or not bool(_cfg_get(curriculum, "enabled", False)):
        return ()

    joint_names = tuple(str(name) for name in (_cfg_get(curriculum, "joint_names", []) or ()) if str(name))
    stage_cfgs = _cfg_get(curriculum, "stages", []) or []
    if not stage_cfgs:
        raise ValueError("env.actuator_curriculum.enabled=true requires at least one stage.")

    stages: list[ActuatorCurriculumStage] = []
    previous_end_fraction = 0.0
    for idx, stage_cfg in enumerate(stage_cfgs):
        name = str(_cfg_get(stage_cfg, "name", f"stage{idx}"))
        end_fraction = float(_cfg_get(stage_cfg, "end_fraction"))
        if end_fraction <= previous_end_fraction or end_fraction > 1.0:
            raise ValueError(
                "actuator_curriculum stage end_fraction values must be increasing and <= 1.0; "
                f"got {end_fraction} after {previous_end_fraction}."
            )

        scales = _joint_strength_scales_from_cfg(_cfg_get(stage_cfg, "joint_strength_scales", None))
        if not scales:
            if not joint_names:
                raise ValueError(
                    "actuator_curriculum stages need joint_strength_scales or parent joint_names."
                )
            strength = _cfg_get(stage_cfg, "strength", None)
            if strength is None:
                raise ValueError(
                    f"actuator_curriculum stage {name!r} needs strength or joint_strength_scales."
                )
            strength_f = float(strength)
            scales = {joint_name: strength_f for joint_name in joint_names}
            _joint_strength_scales_from_cfg(scales)

        stages.append(
            ActuatorCurriculumStage(
                index=idx,
                name=name,
                end_fraction=end_fraction,
                joint_strength_scales=scales,
            )
        )
        previous_end_fraction = end_fraction
    return tuple(stages)


def _actuator_curriculum_stage_for_step(
    stages: tuple[ActuatorCurriculumStage, ...],
    interaction_step: int,
    num_interaction_steps: int,
) -> ActuatorCurriculumStage | None:
    if not stages:
        return None
    progress = float(interaction_step) / float(max(1, num_interaction_steps))
    for stage in stages:
        if progress <= stage.end_fraction + 1e-12:
            return stage
    return stages[-1]


def _set_active_actuator_curriculum_stage(cfg, stage: ActuatorCurriculumStage | None) -> None:
    if stage is None:
        return
    OmegaConf.update(
        cfg,
        "env.joint_strength_scales",
        dict(stage.joint_strength_scales),
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "env.actuator_curriculum.active_stage_index",
        int(stage.index),
        merge=True,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "env.actuator_curriculum.active_stage_name",
        stage.name,
        merge=True,
        force_add=True,
    )


def _actuator_curriculum_metrics(stage: ActuatorCurriculumStage | None) -> dict[str, float]:
    if stage is None:
        return {}
    metrics = {
        "Curriculum/actuator_stage_index": float(stage.index),
        "Curriculum/actuator_stage_end_fraction": float(stage.end_fraction),
    }
    for joint_name, scale in stage.joint_strength_scales.items():
        metrics[f"Curriculum/joint_strength/{joint_name}"] = float(scale)
    return metrics


def _reset_buffer_on_actuator_stage_change(cfg) -> bool:
    curriculum = _cfg_get(cfg.env, "actuator_curriculum", None)
    return bool(_cfg_get(curriculum, "reset_buffer_on_stage_change", False))


def _close_envs(*envs) -> None:
    closed_env_ids: set[int] = set()
    for env in envs:
        if env is not None and id(env) not in closed_env_ids:
            env.close()
            closed_env_ids.add(id(env))


def _assert_same_env_spaces(reference_env, candidate_env) -> None:
    if candidate_env.observation_space.shape != reference_env.observation_space.shape:
        raise RuntimeError(
            "Actuator curriculum changed observation space from "
            f"{reference_env.observation_space.shape} to {candidate_env.observation_space.shape}."
        )
    if candidate_env.action_space.shape != reference_env.action_space.shape:
        raise RuntimeError(
            "Actuator curriculum changed action space from "
            f"{reference_env.action_space.shape} to {candidate_env.action_space.shape}."
        )


def _create_mjlab_envs_like_ppo(
    cfg,
    joint_strength_scales: dict[str, float] | None = None,
):
    """Create mjlab envs through the same ManagerBasedRlEnv path as scripts/train.py."""
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    os.environ["MUJOCO_GL"] = "egl"

    device = cfg.env.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    env_cfg = load_env_cfg(cfg.env.env_name)
    env_cfg.scene.num_envs = cfg.num_train_envs
    env_cfg.seed = cfg.seed

    strength_scales = _joint_strength_scales_from_cfg(cfg.env.get("joint_strength_scales", None))
    if joint_strength_scales is not None:
        strength_scales.update(joint_strength_scales)
    broken_joint_names = tuple(cfg.env.get("broken_joint_names", []) or ())
    for joint_name in broken_joint_names:
        strength_scales[str(joint_name)] = 0.0
    if strength_scales:
        from scripts.reinforcement_learning.rwm_dataset.broken_go2 import apply_go2_pd_joint_strength_scales

        strength_scales = apply_go2_pd_joint_strength_scales(env_cfg, strength_scales)
    apply_mjlab_env_overrides(env_cfg, cfg)

    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=cfg.env.get("use_domain_randomization", True),
        use_push_randomization=cfg.env.get("use_push_randomization", True),
        use_observation_noise=cfg.env.get("use_observation_noise", True),
        randomization_preset=cfg.env.get("randomization_preset", "default"),
        randomization_components=cfg.env.get("randomization_components", None),
        randomization_scale=cfg.env.get("randomization_scale", 1.0),
        payload_mass_range_kg=cfg.env.get("payload_mass_range_kg", None),
        payload_position_body_m=cfg.env.get("payload_position_body_m", None),
        payload_box_size_m=cfg.env.get("payload_box_size_m", None),
        rr_calf_strength_range=cfg.env.get("rr_calf_strength_range", None),
    )

    print(
        "[FlashSAC] Creating mjlab env via PPO path: "
        f"task={cfg.env.env_name}, device={device}, joint_strength_scales={strength_scales}"
    )
    manager_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    train_env = MjlabVectorEnv.from_env(
        manager_env,
        action_mask_indices=cfg.env.get("action_mask_indices", []),
        use_critic_observation_as_full_observation=cfg.env.get(
            "use_critic_observation_as_full_observation",
            False,
        ),
    )
    return train_env, train_env, train_env


def _create_training_envs(
    cfg,
    joint_strength_scales: dict[str, float] | None = None,
):
    if cfg.env.env_type != "mjlab":
        raise ValueError(f"Only mjlab FlashSAC training is supported, got env_type={cfg.env.env_type!r}.")
    return _create_mjlab_envs_like_ppo(cfg, joint_strength_scales=joint_strength_scales)


def run(args: argparse.Namespace) -> None:
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

    num_interaction_steps = int(cfg.num_interaction_steps)
    total_interaction_steps = num_interaction_steps + 1
    actuator_curriculum = _build_actuator_curriculum(cfg)
    current_actuator_stage = _actuator_curriculum_stage_for_step(
        actuator_curriculum,
        interaction_step=0,
        num_interaction_steps=num_interaction_steps,
    )
    _set_active_actuator_curriculum_stage(cfg, current_actuator_stage)
    if current_actuator_stage is not None:
        print(
            "[FlashSAC] Actuator curriculum initial stage: "
            f"{current_actuator_stage.name} "
            f"scales={current_actuator_stage.joint_strength_scales}",
            flush=True,
        )

    train_env, eval_env, record_env = _create_training_envs(
        cfg,
        joint_strength_scales=(
            current_actuator_stage.joint_strength_scales
            if current_actuator_stage is not None
            else None
        ),
    )
    reference_train_env = train_env
    observation_space = train_env.observation_space
    action_space = train_env.action_space

    _, env_info = train_env.reset()
    agent = create_agent(
        observation_space=observation_space,
        action_space=action_space,
        env_info=env_info,
        cfg=cfg.agent,
    )

    logger = create_logger(cfg)

    save_path_resolved = cfg.save_path.replace("TIMESTAMP", datetime.now().strftime("%m%d-%H%M%S"))
    save_path_base = REPO_ROOT / save_path_resolved
    _save_config(cfg, save_path_base / FLASHSAC_CONFIG_FILENAME)
    if cfg.agent_load_path is not None:
        agent.load(str((REPO_ROOT / cfg.agent_load_path).resolve()))
    if cfg.buffer_load_path is not None:
        agent.load_replay_buffer(str((REPO_ROOT / cfg.buffer_load_path).resolve()))

    eval_info = _maybe_evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = _maybe_record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
    logger.update_metric(**_actuator_curriculum_metrics(current_actuator_stage))
    logger.update_metric(**eval_info)
    logger.update_metric(**video_info)
    logger.log_metric(step=0)
    logger.reset()

    observations, _env_infos = train_env.reset()
    actions: Optional[Tensor] = None
    transition: Optional[dict[str, Tensor]] = None
    update_counter = 0.0
    update_info = {}
    collection_time_acc = 0.0
    learning_time_acc = 0.0
    env_steps_since_log = 0
    last_metric_snapshot: dict[str, float] = {}

    deploy_policy_dir = Path(args.deploy_policy_dir).expanduser()
    if not deploy_policy_dir.is_absolute():
        deploy_policy_dir = (REPO_ROOT / deploy_policy_dir).resolve()

    for interaction_step in tqdm.tqdm(range(1, total_interaction_steps), smoothing=0.1, mininterval=0.5):
        next_actuator_stage = _actuator_curriculum_stage_for_step(
            actuator_curriculum,
            interaction_step=interaction_step,
            num_interaction_steps=num_interaction_steps,
        )
        if (
            next_actuator_stage is not None
            and current_actuator_stage is not None
            and next_actuator_stage.index != current_actuator_stage.index
        ):
            current_actuator_stage = next_actuator_stage
            _set_active_actuator_curriculum_stage(cfg, current_actuator_stage)
            print(
                "[FlashSAC] Actuator curriculum stage switch: "
                f"step={interaction_step}, stage={current_actuator_stage.name}, "
                f"scales={current_actuator_stage.joint_strength_scales}",
                flush=True,
            )
            _close_envs(train_env, eval_env, record_env)
            train_env, eval_env, record_env = _create_training_envs(
                cfg,
                joint_strength_scales=current_actuator_stage.joint_strength_scales,
            )
            _assert_same_env_spaces(reference_train_env, train_env)
            observations, _env_infos = train_env.reset()
            actions = None
            transition = None
            update_counter = 0.0
            if _reset_buffer_on_actuator_stage_change(cfg):
                reset_buffer = getattr(agent, "reset_replay_buffer", None)
                if reset_buffer is None:
                    raise RuntimeError("Agent does not support reset_replay_buffer().")
                reset_buffer()

        env_step = interaction_step * cfg.num_train_envs
        collection_start = time.perf_counter()

        if agent.can_start_training() and transition is not None:
            actions = agent.sample_actions(interaction_step, prev_transition=transition, training=True)
        else:
            actions = train_env.action_space.sample()

        assert actions is not None
        actions = np.array(actions)
        next_observations, rewards, terminateds, truncateds, env_infos = train_env.step(actions)
        next_buffer_observations = next_observations.copy()
        for env_idx in range(cfg.num_train_envs):
            if terminateds[env_idx] or truncateds[env_idx]:
                next_buffer_observations[env_idx] = env_infos["final_obs"][env_idx]

        if "episode_info" in env_infos:
            logger.update_metric(**env_infos["episode_info"])
        logger.update_metric(**_actuator_curriculum_metrics(current_actuator_stage))

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
        collection_time_acc += time.perf_counter() - collection_start
        env_steps_since_log += cfg.num_train_envs

        if agent.can_start_training():
            learning_start = time.perf_counter()
            update_counter += cfg.updates_per_interaction_step
            while update_counter >= 1:
                update_info = agent.update()
                logger.update_metric(**update_info)
                update_counter -= 1
            learning_time_acc += time.perf_counter() - learning_start

            if cfg.evaluation_per_interaction_step and interaction_step % cfg.evaluation_per_interaction_step == 0:
                logger.update_metric(**_maybe_evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type))

            if cfg.metrics_per_interaction_step and interaction_step % cfg.metrics_per_interaction_step == 0:
                logger.update_metric(**agent.get_metrics())

            if cfg.recording_per_interaction_step and interaction_step % cfg.recording_per_interaction_step == 0:
                logger.update_metric(**_maybe_record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type))

            if cfg.logging_per_interaction_step and interaction_step % cfg.logging_per_interaction_step == 0:
                logger.update_metric(**_perf_metrics(env_steps_since_log, collection_time_acc, learning_time_acc))
                last_metric_snapshot = _logger_scalar_snapshot(logger)
                logger.log_metric(step=env_step)
                logger.reset()
                collection_time_acc = 0.0
                learning_time_acc = 0.0
                env_steps_since_log = 0

            if cfg.save_checkpoint_per_interaction_step and interaction_step % cfg.save_checkpoint_per_interaction_step == 0:
                save_path = save_path_base / f"step{interaction_step}"
                agent.save(str(save_path))
                _save_config(cfg, save_path / FLASHSAC_CONFIG_FILENAME)
                save_metrics = _logger_scalar_snapshot(logger)
                if save_metrics:
                    save_metrics.update(_perf_metrics(env_steps_since_log, collection_time_acc, learning_time_acc))
                else:
                    save_metrics = dict(last_metric_snapshot)
                _print_checkpoint_log(interaction_step, env_step, save_metrics)
                if args.export_deploy_policy:
                    _export_g1_policy(save_path, cfg, deploy_policy_dir)

    final_step = total_interaction_steps - 1
    final_checkpoint = save_path_base / f"step{final_step}"
    agent.save(str(final_checkpoint))
    _save_config(cfg, final_checkpoint / FLASHSAC_CONFIG_FILENAME)
    if args.export_deploy_policy:
        _export_g1_policy(final_checkpoint, cfg, deploy_policy_dir)

    logger.update_metric(**_perf_metrics(env_steps_since_log, collection_time_acc, learning_time_acc))
    logger.update_metric(**_maybe_evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type))
    logger.update_metric(**_maybe_record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type))
    logger.log_metric(step=final_step * cfg.num_train_envs)
    logger.reset()

    closed_env_ids: set[int] = set()
    for env in (train_env, eval_env, record_env):
        if id(env) not in closed_env_ids:
            env.close()
            closed_env_ids.add(id(env))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", type=str, default=str(DEFAULT_CONFIG_DIR))
    parser.add_argument("--config_name", type=str, default="flashsac_g1_velocity")
    parser.add_argument("--overrides", action="append", default=[])
    parser.add_argument("--export_deploy_policy", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--deploy_policy_dir", type=str, default=str(G1_FLASHSAC_POLICY_DIR))
    run(parser.parse_args())
