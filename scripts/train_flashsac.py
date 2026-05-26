"""Train FlashSAC on mjlab Unitree tasks and export G1 deploy ONNX policies."""

from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

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
from flash_rl.envs import create_envs
from flash_rl.envs.mjlab import MjlabVectorEnv, configure_mjlab_randomization
from flash_rl.evaluation import evaluate, record_video
from flash_rl.export import export_flashsac_policy_to_onnx
from flash_rl.types import Tensor


DEFAULT_CONFIG_DIR = REPO_ROOT / "configs"
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


def _create_mjlab_envs_like_ppo(cfg):
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

    configure_mjlab_randomization(
        env_cfg,
        use_domain_randomization=cfg.env.get("use_domain_randomization", True),
        use_push_randomization=cfg.env.get("use_push_randomization", True),
        use_observation_noise=cfg.env.get("use_observation_noise", True),
    )

    print(f"[FlashSAC] Creating mjlab env via PPO path: task={cfg.env.env_name}, device={device}")
    manager_env = ManagerBasedRlEnv(cfg=env_cfg, device=device)
    train_env = MjlabVectorEnv.from_env(manager_env)
    return train_env, train_env, train_env


def _create_training_envs(cfg):
    if cfg.env.env_type == "mjlab":
        return _create_mjlab_envs_like_ppo(cfg)
    return create_envs(**cfg.env)


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

    train_env, eval_env, record_env = _create_training_envs(cfg)
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
    if cfg.agent_load_path is not None:
        agent.load(str((REPO_ROOT / cfg.agent_load_path).resolve()))
    if cfg.buffer_load_path is not None:
        agent.load_replay_buffer(str((REPO_ROOT / cfg.buffer_load_path).resolve()))

    eval_info = _maybe_evaluate(agent, eval_env, cfg.num_eval_episodes, cfg.env.env_type)
    video_info = _maybe_record_video(agent, record_env, cfg.num_record_episodes, cfg.env.env_type)
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

    total_interaction_steps = int(cfg.num_interaction_steps + 1)
    deploy_policy_dir = Path(args.deploy_policy_dir).expanduser()
    if not deploy_policy_dir.is_absolute():
        deploy_policy_dir = (REPO_ROOT / deploy_policy_dir).resolve()

    for interaction_step in tqdm.tqdm(range(1, total_interaction_steps), smoothing=0.1, mininterval=0.5):
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
                total_time = collection_time_acc + learning_time_acc
                if total_time > 0.0:
                    logger.update_metric(
                        **{
                            "Perf/total_fps": float(env_steps_since_log / total_time),
                            "Perf/collection_time": float(collection_time_acc),
                            "Perf/learning_time": float(learning_time_acc),
                        }
                    )
                logger.log_metric(step=env_step)
                logger.reset()
                collection_time_acc = 0.0
                learning_time_acc = 0.0
                env_steps_since_log = 0

            if cfg.save_checkpoint_per_interaction_step and interaction_step % cfg.save_checkpoint_per_interaction_step == 0:
                save_path = save_path_base / f"step{interaction_step}"
                agent.save(str(save_path))
                if args.export_deploy_policy:
                    _export_g1_policy(save_path, cfg, deploy_policy_dir)

    final_step = total_interaction_steps - 1
    final_checkpoint = save_path_base / f"step{final_step}"
    agent.save(str(final_checkpoint))
    if args.export_deploy_policy:
        _export_g1_policy(final_checkpoint, cfg, deploy_policy_dir)

    total_time = collection_time_acc + learning_time_acc
    if total_time > 0.0:
        logger.update_metric(
            **{
                "Perf/total_fps": float(env_steps_since_log / total_time),
                "Perf/collection_time": float(collection_time_acc),
                "Perf/learning_time": float(learning_time_acc),
            }
        )
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
