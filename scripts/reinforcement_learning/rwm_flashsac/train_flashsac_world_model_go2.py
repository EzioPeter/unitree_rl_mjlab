"""Train Go2 FlashSAC entirely inside the learned RWM imagination env."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import shutil
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, dataclass
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
from scripts.reinforcement_learning.rwm_flashsac.replay_mixer import (
    ExternalReplaySampler,
    ReplayMixConfig,
    compute_source_counts,
)
from scripts.reinforcement_learning.rwm_flashsac.utils import (
    DEFAULT_CONFIG_PATH,
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
from scripts.reinforcement_learning.rwm_trace.lateral_reward_shaping import (
    shifted_lateral_quality_delta_numpy,
)
from scripts.reinforcement_learning.rwm_trace.multi_axis_reward_shaping import (
    aligned_multi_axis_quality_delta_numpy,
)


def _apply_aligned_rwm_simulator_reward(
    *,
    cfg: Any,
    observations: np.ndarray,
    final_observations: np.ndarray,
    rewards: np.ndarray,
) -> tuple[np.ndarray, float]:
    enabled = bool(
        OmegaConf.select(
            cfg,
            "trace.align_world_model_reward_with_simulator",
            default=False,
        )
    )
    if not enabled:
        return rewards, 0.0
    simulator_reward = OmegaConf.select(
        cfg,
        "trace.simulator_reward",
        default={},
    )
    if not bool(OmegaConf.select(simulator_reward, "enabled", default=False)):
        raise ValueError(
            "Aligned RWM/simulator reward requires simulator_reward.enabled=true."
        )
    aligned_multi_axis_mode = str(
        OmegaConf.select(
            simulator_reward,
            "aligned_multi_axis_mode",
            default="none",
        )
    ).strip().lower()
    unsupported_additive_fields = (
        "additive_lateral_progress_weight",
        "additive_lateral_gaussian_advantage_weight",
        "additive_lateral_below_threshold_penalty",
        "additive_lateral_rmse_weight",
        "additive_lateral_velocity_delta_weight",
    )
    nonzero_unsupported = {
        name: float(OmegaConf.select(simulator_reward, name, default=0.0))
        for name in unsupported_additive_fields
        if float(OmegaConf.select(simulator_reward, name, default=0.0)) != 0.0
    }
    if nonzero_unsupported:
        raise ValueError(
            "Aligned RWM/simulator reward only supports the shared shifted "
            f"exp/tanh implementation, got {nonzero_unsupported}."
        )
    observation_mask = tuple(
        int(index)
        for index in OmegaConf.select(
            cfg,
            "world_model.policy_observation_mask_indices",
            default=[],
        )
    )
    if observation_mask:
        raise ValueError(
            "Aligned RWM/simulator reward currently requires an unmasked "
            "full-state observation layout."
        )
    if observations.ndim != 2 or observations.shape[1] < 12:
        raise ValueError(
            "Aligned RWM/simulator reward requires command indices 9:12."
        )
    if (
        final_observations.ndim != 2
        or final_observations.shape[1] < 3
        or final_observations.shape[0] != observations.shape[0]
    ):
        raise ValueError(
            "Aligned RWM/simulator reward requires full predicted final observations."
        )

    world_model = cfg.world_model
    signed_exp_weight = float(
        OmegaConf.select(
            simulator_reward,
            "additive_lateral_signed_exp_weight",
            default=0.0,
        )
    )
    shifted_tanh_weight = float(
        OmegaConf.select(
            simulator_reward,
            "additive_lateral_shifted_tanh_weight",
            default=0.0,
        )
    )
    if (
        aligned_multi_axis_mode == "none"
        and signed_exp_weight == 0.0
        and shifted_tanh_weight == 0.0
    ):
        raise ValueError(
            "Aligned RWM/simulator reward requires a shifted exp or tanh term."
        )
    command_scale_floor = float(
        OmegaConf.select(
            world_model,
            "reward_response_command_scale_floor",
            default=0.05,
        )
    )
    if aligned_multi_axis_mode != "none":
        if signed_exp_weight != 0.0 or shifted_tanh_weight != 0.0:
            raise ValueError(
                "Aligned multi-axis reward cannot be combined with legacy "
                "lateral exp/tanh shaping."
            )
        delta = aligned_multi_axis_quality_delta_numpy(
            velocity=final_observations[:, [0, 1, 5]],
            command=observations[:, 9:12],
            active_thresholds=(
                float(
                    OmegaConf.select(
                        world_model,
                        "reward_command_active_threshold_x",
                        default=0.03,
                    )
                ),
                float(
                    OmegaConf.select(
                        world_model,
                        "reward_command_active_threshold_y",
                        default=0.02,
                    )
                ),
                float(
                    OmegaConf.select(
                        world_model,
                        "reward_command_active_threshold_yaw",
                        default=0.03,
                    )
                ),
            ),
            axis_weights=tuple(
                float(value)
                for value in OmegaConf.select(
                    simulator_reward,
                    "aligned_multi_axis_axis_weights",
                    default=[1.0, 1.0, 1.0],
                )
            ),
            command_scale_floor=command_scale_floor,
            tracking_stds=(
                float(OmegaConf.select(simulator_reward, "std_x", default=0.25)),
                float(OmegaConf.select(simulator_reward, "std_y", default=0.10)),
                float(OmegaConf.select(simulator_reward, "std_yaw", default=0.20)),
            ),
            mode=aligned_multi_axis_mode,
            weight=float(
                OmegaConf.select(
                    simulator_reward,
                    "aligned_multi_axis_weight",
                    default=0.0,
                )
            ),
            tanh_gain=float(
                OmegaConf.select(
                    simulator_reward,
                    "aligned_multi_axis_tanh_gain",
                    default=2.0,
                )
            ),
            overspeed_weight=float(
                OmegaConf.select(
                    simulator_reward,
                    "aligned_multi_axis_overspeed_weight",
                    default=4.0,
                )
            ),
        )
    else:
        delta = shifted_lateral_quality_delta_numpy(
            velocity_y=final_observations[:, 1],
            command_y=observations[:, 10],
            active_threshold_y=float(
                OmegaConf.select(
                    world_model,
                    "reward_command_active_threshold_y",
                    default=0.02,
                )
            ),
            command_scale_floor=command_scale_floor,
            signed_exp_weight=signed_exp_weight,
            signed_exp_clip=float(
                OmegaConf.select(
                    simulator_reward,
                    "additive_lateral_signed_exp_clip",
                    default=2.0,
                )
            ),
            shifted_tanh_weight=shifted_tanh_weight,
            shifted_tanh_gain=float(
                OmegaConf.select(
                    simulator_reward,
                    "additive_lateral_shifted_tanh_gain",
                    default=2.0,
                )
            ),
        ).astype(np.float32, copy=False)
    aligned_rewards = rewards + float(world_model.step_dt) * delta
    return aligned_rewards.astype(np.float32, copy=False), float(delta.mean())

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


def _release_trace_proposal_cuda_cache(device: str | torch.device) -> dict[str, float]:
    """Release temporary proposal allocations shared by PyTorch and Warp.

    TRACE proposal collection creates substantially larger temporary tensors
    than the regular RWM update. PyTorch's caching allocator otherwise keeps
    those blocks reserved between refreshes, starving Warp's independent CUDA
    graph allocator even though no live TRACE tensor needs the memory.
    """

    cuda_device = torch.device(device)
    if cuda_device.type != "cuda" or not torch.cuda.is_available():
        return {}

    torch.cuda.synchronize(cuda_device)
    allocated_before = int(torch.cuda.memory_allocated(cuda_device))
    reserved_before = int(torch.cuda.memory_reserved(cuda_device))

    gc.collect()
    try:
        import warp as wp

        wp.synchronize_device(str(cuda_device))
    except (ImportError, RuntimeError, ValueError):
        # The proposal runtime may use a non-Warp backend in unit tests.
        pass
    torch.cuda.empty_cache()
    torch.cuda.synchronize(cuda_device)

    allocated_after = int(torch.cuda.memory_allocated(cuda_device))
    reserved_after = int(torch.cuda.memory_reserved(cuda_device))
    mib = float(1024**2)
    return {
        "TRACE/cuda_allocated_before_cleanup_mib": allocated_before / mib,
        "TRACE/cuda_reserved_before_cleanup_mib": reserved_before / mib,
        "TRACE/cuda_allocated_after_cleanup_mib": allocated_after / mib,
        "TRACE/cuda_reserved_after_cleanup_mib": reserved_after / mib,
        "TRACE/cuda_cache_released_mib": max(
            0.0, (reserved_before - reserved_after) / mib
        ),
    }


def _debug_live_cuda_tensors(device: str | torch.device) -> None:
    """Print live Python-owned CUDA storages for an opt-in memory diagnosis."""

    if os.environ.get("TRACE_DEBUG_CUDA_TENSORS") != "1":
        return
    cuda_device = torch.device(device)
    storages: dict[tuple[int, int], tuple[int, tuple[int, ...], str]] = {}
    for value in gc.get_objects():
        try:
            if not torch.is_tensor(value) or value.device != cuda_device:
                continue
            storage = value.untyped_storage()
            key = (int(storage.data_ptr()), int(storage.nbytes()))
            storages.setdefault(
                key,
                (int(storage.nbytes()), tuple(value.shape), str(value.dtype)),
            )
        except (AttributeError, RuntimeError):
            continue
    rows = sorted(storages.values(), reverse=True)
    total_mib = sum(row[0] for row in rows) / float(1024**2)
    grouped: dict[tuple[tuple[int, ...], str], list[int]] = {}
    for nbytes, shape, dtype in rows:
        aggregate = grouped.setdefault((shape, dtype), [0, 0])
        aggregate[0] += 1
        aggregate[1] += nbytes
    top_groups = sorted(
        (
            (total_bytes, count, shape, dtype)
            for (shape, dtype), (count, total_bytes) in grouped.items()
        ),
        reverse=True,
    )
    top = ", ".join(
        f"{nbytes / float(1024**2):.1f}MiB:{shape}:{dtype}"
        for nbytes, shape, dtype in rows[:12]
    )
    group_top = ", ".join(
        f"{total_bytes / float(1024**2):.1f}MiB/{count}x:{shape}:{dtype}"
        for total_bytes, count, shape, dtype in top_groups[:16]
    )
    print(
        "[Go2-FlashSAC-RWM][TRACE-Memory-Debug] "
        f"live_python_cuda_storages={len(rows)}, total_mib={total_mib:.1f}, "
        f"top=[{top}], grouped_top=[{group_top}]"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--config_path", default=None)
    parser.add_argument(
        "--trace_config_path",
        default=None,
        help="Optional formal online TRACE overlay; merged before CLI overrides.",
    )
    parser.add_argument("--model_resume_path", default=None)
    parser.add_argument("--dataset_path", default=None)
    parser.add_argument("--policy_resume_path", default=None)
    parser.add_argument("--policy_resume_mode", choices=("full", "actor_only"), default="full")
    parser.add_argument("--load_replay_buffer", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num_imagination_envs", type=int, default=None)
    parser.add_argument("--num_env_steps", type=int, default=None)
    parser.add_argument(
        "--expected_policy_updates",
        type=int,
        default=None,
        help="Fail closed unless this run executes exactly this many agent updates.",
    )
    parser.add_argument(
        "--allow_training_horizon_extension",
        action="store_true",
        help=(
            "Allow a full-state checkpoint to continue under a strictly larger "
            "interaction-step and policy-update budget. All learned, replay, "
            "environment, TRACE, optimizer, and RNG state is still restored."
        ),
    )
    parser.add_argument(
        "--freeze_trace_scorer_after_resume",
        action="store_true",
        help=(
            "After loading a co-located TRACE manager checkpoint, disable its "
            "online scorer updater while preserving the checkpointed scorer "
            "weights, replay state, and feedback cursor."
        ),
    )
    parser.add_argument(
        "--stop_after_interaction_step",
        type=int,
        default=None,
        help=(
            "Create a resumable checkpoint and stop at this interaction step "
            "without changing the checkpoint's original total-step budget."
        ),
    )
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
    if args.policy_resume_path is not None:
        updates.append(f"policy_resume_path={args.policy_resume_path}")
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


def _completed_step_checkpoints(root: Path) -> list[Path]:
    return sorted(
        (
            row
            for row in root.iterdir()
            if row.is_dir()
            and row.name.startswith("step")
            and row.name[4:].isdigit()
            and (row / "CHECKPOINT_COMPLETE").is_file()
        ),
        key=lambda row: int(row.name[4:]),
    )


def _stage_reusable_replay_slot(
    save_dir: Path, temporary: Path
) -> Path | None:
    completed = _completed_step_checkpoints(save_dir.parent)
    if len(completed) < 2:
        return None
    reusable = completed[-2]
    if not (
        (reusable / "replay_buffer.pt").is_file()
        and (reusable / "replay_buffer_data").is_dir()
    ):
        return None
    retired_holder = temporary / "_retired_checkpoint"
    os.replace(reusable, retired_holder)
    os.replace(
        retired_holder / "replay_buffer.pt",
        temporary / "replay_buffer.pt",
    )
    os.replace(
        retired_holder / "replay_buffer_data",
        temporary / "replay_buffer_data",
    )
    return retired_holder


def _commit_checkpoint_directory(
    temporary: Path,
    save_dir: Path,
    retired_holder: Path | None,
) -> None:
    if retired_holder is not None:
        shutil.rmtree(retired_holder)
    (temporary / "CHECKPOINT_COMPLETE").write_text(
        "complete\n", encoding="utf-8"
    )
    os.replace(temporary, save_dir)
    completed = _completed_step_checkpoints(save_dir.parent)
    for obsolete in completed[:-2]:
        shutil.rmtree(obsolete)


def _save_checkpoint(
    agent: Any,
    save_dir: Path,
    cfg: Any,
    save_replay: bool,
    trace_manager: Any | None = None,
    training_state: dict[str, Any] | None = None,
) -> None:
    if save_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {save_dir}")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{save_dir.name}.", dir=save_dir.parent)
    )
    retired_holder: Path | None = None
    try:
        if save_replay:
            retired_holder = _stage_reusable_replay_slot(
                save_dir, temporary
            )
        agent.save(str(temporary))
        save_config(cfg, temporary / "rwm_flashsac_config.yaml")
        if save_replay:
            agent.save_replay_buffer(str(temporary))
        if trace_manager is not None:
            trace_manager.save_checkpoint(temporary / "trace_manager")
        if training_state is not None:
            torch.save(training_state, temporary / "training_state.pt")
        _commit_checkpoint_directory(
            temporary, save_dir, retired_holder
        )
        retired_holder = None
        if save_replay:
            replay_buffer = getattr(agent, "_replay_buffer", None)
            mark_saved = getattr(
                replay_buffer, "mark_saved_checkpoint", None
            )
            if callable(mark_saved):
                mark_saved(str(save_dir / "replay_buffer.pt"))
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


@dataclass(frozen=True)
class AsyncCheckpointWriter:
    pid: int
    save_dir: Path
    started_at: float


def _start_async_checkpoint(
    agent: Any,
    save_dir: Path,
    cfg: Any,
    trace_manager: Any,
    training_state: dict[str, Any],
) -> AsyncCheckpointWriter:
    """Freeze state, then fork only a CPU replay disk writer."""

    replay_buffer = getattr(agent, "_replay_buffer", None)
    replay_device = getattr(replay_buffer, "_device", None)
    replay_snapshot = None
    if replay_device is not None and torch.device(replay_device).type != "cpu":
        snapshot_to_cpu = getattr(replay_buffer, "snapshot_to_cpu", None)
        if not callable(snapshot_to_cpu):
            raise ValueError(
                "CUDA replay buffer does not support a CPU checkpoint snapshot."
            )
        replay_snapshot = snapshot_to_cpu()
    if save_dir.exists():
        raise FileExistsError(f"Refusing to overwrite checkpoint: {save_dir}")
    save_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{save_dir.name}.", dir=save_dir.parent)
    )
    retired_holder: Path | None = None
    replay_write_plan = None
    try:
        retired_holder = _stage_reusable_replay_slot(
            save_dir, temporary
        )
        deferred_source = (
            replay_snapshot
            if replay_snapshot is not None
            else replay_buffer
        )
        deferred_save = getattr(deferred_source, "save", None)
        if callable(deferred_save):
            replay_write_plan = deferred_save(
                str(temporary / "replay_buffer.pt"),
                defer_raw_writes=True,
            )
        # Capture every non-replay component before fork.  This is the short
        # consistency pause; the large raw replay write happens in the child.
        agent.save(str(temporary))
        save_config(cfg, temporary / "rwm_flashsac_config.yaml")
        trace_manager.save_checkpoint(temporary / "trace_manager")
        torch.save(training_state, temporary / "training_state.pt")
        sys.stdout.flush()
        sys.stderr.flush()
        started_at = time.perf_counter()
        pid = os.fork()
        if pid == 0:
            try:
                # The child never invokes CUDA.  A CUDA-backed replay is
                # copied to one consistent CPU generation before fork; a
                # CPU-backed replay is frozen by fork copy-on-write.
                if replay_write_plan is not None:
                    replay_write_plan.execute()
                    print(
                        "\033[32m[FlashSAC]\033[0m Successfully saved "
                        f"replay buffer at {temporary}."
                    )
                elif replay_snapshot is None:
                    agent.save_replay_buffer(str(temporary))
                else:
                    replay_snapshot.save(
                        str(temporary / "replay_buffer.pt")
                    )
                    print(
                        "\033[32m[FlashSAC]\033[0m Successfully saved "
                        f"replay buffer at {temporary}."
                    )
                _commit_checkpoint_directory(
                    temporary, save_dir, retired_holder
                )
            except BaseException:
                traceback.print_exc()
                if temporary.exists():
                    shutil.rmtree(temporary)
                os._exit(1)
            os._exit(0)
        return AsyncCheckpointWriter(
            pid=pid,
            save_dir=save_dir,
            started_at=started_at,
        )
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _wait_async_checkpoint(
    writer: AsyncCheckpointWriter | None,
    block: bool,
) -> AsyncCheckpointWriter | None:
    if writer is None:
        return None
    flags = 0 if block else os.WNOHANG
    pid, status = os.waitpid(writer.pid, flags)
    if pid == 0:
        return writer
    elapsed = time.perf_counter() - writer.started_at
    if not os.WIFEXITED(status) or os.WEXITSTATUS(status) != 0:
        raise RuntimeError(
            "Asynchronous checkpoint writer failed for "
            f"{writer.save_dir} (status={status})."
        )
    print(
        "[Go2-FlashSAC-RWM] async_checkpoint_complete="
        f"{writer.save_dir}, writer_seconds={elapsed:.2f}"
    )
    return None


def _capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.random.get_rng_state(),
        "torch_cuda": (
            [state.cpu() for state in torch.cuda.get_rng_state_all()]
            if torch.cuda.is_available()
            else []
        ),
    }


def _restore_rng_state(state: dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(tuple(state["numpy"]))
    torch.random.set_rng_state(state["torch_cpu"])
    cuda_states = list(state.get("torch_cuda") or [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise ValueError("Checkpoint requires CUDA RNG state.")
        if len(cuda_states) != torch.cuda.device_count():
            raise ValueError("Visible CUDA device count changed across resume.")
        torch.cuda.set_rng_state_all(cuda_states)


def _make_training_state(
    *,
    interaction_step: int,
    total_interaction_steps: int,
    update_counter: float,
    target_policy_updates: int | None,
    agent: Any,
    env: Any,
    observations: np.ndarray,
    transition: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "format_version": "go2_flashsac_training_state_v1",
        "interaction_step": int(interaction_step),
        "total_interaction_steps": int(total_interaction_steps),
        "update_counter": float(update_counter),
        "target_policy_updates": (
            int(target_policy_updates)
            if target_policy_updates is not None
            else None
        ),
        "policy_update_step": int(getattr(agent, "_update_step")),
        "observations": np.asarray(observations).copy(),
        "transition": transition,
        "environment": env.state_dict(),
        "rng": _capture_rng_state(),
    }


def _load_training_state(checkpoint: Path) -> dict[str, Any] | None:
    state_path = checkpoint / "training_state.pt"
    complete_path = checkpoint / "CHECKPOINT_COMPLETE"
    if not state_path.exists() and not complete_path.exists():
        return None
    if not state_path.is_file() or not complete_path.is_file():
        raise ValueError("Resumable checkpoint is incomplete.")
    if not (checkpoint / "replay_buffer.pt").is_file():
        raise ValueError("Resumable checkpoint lacks the RWM replay buffer.")
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    if state.get("format_version") != "go2_flashsac_training_state_v1":
        raise ValueError("Training-state checkpoint format mismatch.")
    return state


def _load_actor_only(agent: Any, checkpoint_path: Path, device: str) -> None:
    actor_path = checkpoint_path / "actor.pt"
    if not actor_path.exists():
        raise FileNotFoundError(f"Actor-only resume requires actor.pt: {checkpoint_path}")
    actor = getattr(agent, "_actor", None) or getattr(agent, "actor", None)
    if actor is None:
        raise AttributeError("FlashSAC agent has no accessible actor module for actor-only resume.")
    if hasattr(actor, "load"):
        actor.load(str(actor_path), load_optimizer=False)
    else:
        state = torch.load(actor_path, map_location=device)
        if isinstance(state, dict) and "network_state_dict" in state:
            state = state["network_state_dict"]
        elif isinstance(state, dict) and "state_dict" in state and len(state) == 1:
            state = state["state_dict"]
        actor.load_state_dict(state)
    print(f"[Go2-FlashSAC-RWM] loaded actor-only checkpoint={actor_path}")


def _sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _configure_formal_replay_mix(
    *,
    cfg: Any,
    agent: Any,
    observation_dim: int,
    action_dim: int,
    agent_cfg: Any,
    online_sim_sampler: Any | None = None,
) -> tuple[ReplayMixConfig | None, ExternalReplaySampler | None, ExternalReplaySampler | None, dict[str, Any]]:
    enabled = bool(OmegaConf.select(cfg, "replay_mix.enabled", default=False))
    if not enabled:
        if bool(OmegaConf.select(cfg, "trace.enabled", default=False)):
            raise ValueError(
                "Legacy trace.enabled is not supported by the formal V12 trainer; "
                "use replay_mix instead."
            )
        return None, None, None, {}
    if not hasattr(agent, "configure_replay_mix"):
        raise TypeError("Formal replay mixing requires FlashSACProprioceptiveAgent.")

    mix_config = ReplayMixConfig(
        batch_size=int(agent_cfg.sample_batch_size),
        real_ratio=float(OmegaConf.select(cfg, "replay_mix.real_ratio")),
        synthetic_mode=str(OmegaConf.select(cfg, "replay_mix.synthetic_mode")),
        trace_ratio_within_synthetic=float(
            OmegaConf.select(
                cfg,
                "replay_mix.trace_ratio_within_synthetic",
            )
        ),
        shuffle=bool(
            OmegaConf.select(
                cfg,
                "replay_mix.shuffle_mixed_batch",
                default=True,
            )
        ),
    )
    if not np.isclose(mix_config.real_ratio, 0.05):
        raise ValueError("Formal V12 routes require replay_mix.real_ratio=0.05.")
    counts = compute_source_counts(mix_config)
    real_path_value = OmegaConf.select(cfg, "replay_mix.real_replay_path")
    if counts.real and not real_path_value:
        raise ValueError("Formal V12 replay mix requires real_replay_path.")
    real_sampler = (
        ExternalReplaySampler(
            resolve_repo_path(str(real_path_value)),
            seed=int(OmegaConf.select(cfg, "replay_mix.seed", default=int(cfg.seed))) + 1,
            expected_observation_dim=observation_dim,
            expected_action_dim=action_dim,
            expected_gamma=float(agent_cfg.gamma),
            expected_n_step=int(agent_cfg.n_step),
            expected_source="real",
        )
        if counts.real
        else None
    )
    sim_path_value = OmegaConf.select(
        cfg,
        "replay_mix.trace_replay_path",
        default=None,
    )
    if counts.sim and online_sim_sampler is None and not sim_path_value:
        raise ValueError("Configured simulator replay count requires trace_replay_path.")
    if counts.sim and online_sim_sampler is not None and sim_path_value:
        raise ValueError(
            "Online TRACE uses its mutable buffer and forbids replay_mix.trace_replay_path."
        )
    sim_sampler = (
        online_sim_sampler
        if counts.sim and online_sim_sampler is not None
        else ExternalReplaySampler(
            resolve_repo_path(str(sim_path_value)),
            seed=int(OmegaConf.select(cfg, "replay_mix.seed", default=int(cfg.seed))) + 2,
            expected_observation_dim=observation_dim,
            expected_action_dim=action_dim,
            expected_gamma=float(agent_cfg.gamma),
            expected_n_step=int(agent_cfg.n_step),
            expected_source="sim",
        )
        if counts.sim
        else None
    )
    if sim_sampler is not None:
        allow_legacy = bool(
            OmegaConf.select(cfg, "replay_mix.allow_legacy_trace", default=False)
        )
        protocol = sim_sampler.metadata.get("trace_protocol_version")
        mutable_online = bool(sim_sampler.metadata.get("mutable", False))
        if (
            protocol != "go2_trace_online_v1"
            and protocol != "go2_trace_v5_controlled"
            and not allow_legacy
        ):
            raise ValueError(
                "Formal TRACE training requires an online-v1 or controlled-v5 "
                f"TRACE protocol, got {protocol!r} (mutable={mutable_online})."
            )

    agent.configure_replay_mix(
        config=mix_config,
        real_sampler=real_sampler,
        sim_sampler=sim_sampler,
        seed=int(OmegaConf.select(cfg, "replay_mix.seed", default=int(cfg.seed))),
    )
    normalization_enabled = bool(
        OmegaConf.select(cfg, "reward_normalization.enabled", default=False)
    )
    normalization_frozen = bool(
        OmegaConf.select(cfg, "reward_normalization.frozen", default=False)
    )
    normalizer_path_value = OmegaConf.select(
        cfg,
        "reward_normalization.checkpoint_path",
        default=None,
    )
    if bool(agent_cfg.normalize_reward):
        if not normalization_enabled or not normalization_frozen or not normalizer_path_value:
            raise ValueError(
                "Normalized formal replay mixing requires an enabled, frozen "
                "reward normalizer checkpoint."
            )
        normalizer_metadata = agent.configure_frozen_reward_normalizer(
            str(resolve_repo_path(str(normalizer_path_value)))
        )
    else:
        if normalization_enabled or normalization_frozen or normalizer_path_value:
            raise ValueError(
                "Reference unnormalized reward requires reward_normalization to be "
                "disabled with no checkpoint."
            )
        normalizer_metadata = {
            "enabled": False,
            "frozen": False,
            "reason": "model_based_reference_reward_is_unnormalized",
        }
    return mix_config, real_sampler, sim_sampler, normalizer_metadata


def _write_formal_replay_manifest(
    *,
    path: Path,
    mix_config: ReplayMixConfig,
    real_sampler: ExternalReplaySampler | None,
    sim_sampler: ExternalReplaySampler | None,
    normalizer_metadata: dict[str, Any],
    model_path: Path,
    dataset_path: Path,
    policy_resume_path: Path | None,
    policy_resume_mode: str,
    actor_bc_alpha: float,
    actor_learning_starts_updates: int,
    actor_learning_rate_scale: float,
) -> None:
    counts = compute_source_counts(mix_config)
    manifest = {
        "format_version": "go2_v12_replay_mix_manifest_v1",
        "replay_mix_config": asdict(mix_config),
        "source_counts": asdict(counts),
        "real_replay_path": str(real_sampler.path) if real_sampler else None,
        "real_replay_sha256": real_sampler.sha256 if real_sampler else None,
        "real_replay_metadata": real_sampler.metadata if real_sampler else None,
        "trace_replay_path": str(sim_sampler.path) if sim_sampler else None,
        "trace_replay_sha256": sim_sampler.sha256 if sim_sampler else None,
        "trace_replay_metadata": sim_sampler.metadata if sim_sampler else None,
        "normalizer_path": normalizer_metadata.get("artifact_path"),
        "normalizer_sha256": normalizer_metadata.get("artifact_sha256"),
        "normalizer_metadata": normalizer_metadata,
        "rwm_checkpoint_path": str(model_path),
        "rwm_checkpoint_sha256": _sha256_file(model_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": _sha256_file(dataset_path),
        "policy_resume_path": (
            str(policy_resume_path) if policy_resume_path is not None else None
        ),
        "policy_resume_mode": str(policy_resume_mode),
        "policy_resume_actor_sha256": (
            _sha256_file(policy_resume_path / "actor.pt")
            if policy_resume_path is not None
            else None
        ),
        "actor_bc_alpha": float(actor_bc_alpha),
        "actor_learning_starts_updates": int(actor_learning_starts_updates),
        "actor_learning_rate_scale": float(actor_learning_rate_scale),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str) + "\n")


def main() -> None:
    configure_low_thread_env()
    args = _parse_args()
    cfg = load_config(args.config_path)
    if args.trace_config_path is not None:
        cfg = OmegaConf.merge(cfg, OmegaConf.load(args.trace_config_path))
    cfg = _apply_arg_overrides(cfg, args)
    OmegaConf.resolve(cfg)

    device = select_device(args.device or cfg.agent.device_type)
    set_seed(int(cfg.seed))

    model_path = resolve_repo_path(str(cfg.model_resume_path))
    dataset_path = resolve_repo_path(str(cfg.dataset_path))
    dynamics, _checkpoint = load_dynamics_checkpoint(model_path, device=device)
    dataset = SequenceReplayBuffer.load(dataset_path, device=device)

    checkpoint_infos = _checkpoint.get("infos") or {}
    checkpoint_action_mask = checkpoint_infos.get("action_mask_indices") or []
    checkpoint_world_model_mask = checkpoint_infos.get("world_model_action_mask_indices") or checkpoint_action_mask
    checkpoint_policy_mask = checkpoint_infos.get("policy_action_mask_indices") or checkpoint_world_model_mask
    checkpoint_policy_observation_mask = checkpoint_infos.get("policy_observation_mask_indices") or []
    checkpoint_broken_joints = (
        checkpoint_infos.get("broken_pd_joint_names")
        or checkpoint_infos.get("broken_joint_names")
        or []
    )
    configured_policy_mask = OmegaConf.select(cfg, "world_model.policy_action_mask_indices") or []
    configured_wm_mask = OmegaConf.select(cfg, "world_model.world_model_action_mask_indices") or []
    configured_policy_observation_mask = OmegaConf.select(cfg, "world_model.policy_observation_mask_indices") or []
    configured_broken_joints = OmegaConf.select(cfg, "world_model.broken_joint_names") or []
    if checkpoint_policy_mask and not configured_policy_mask:
        OmegaConf.update(
            cfg,
            "world_model.policy_action_mask_indices",
            list(checkpoint_policy_mask),
            merge=True,
        )
    if checkpoint_world_model_mask and not configured_wm_mask:
        OmegaConf.update(
            cfg,
            "world_model.world_model_action_mask_indices",
            list(checkpoint_world_model_mask),
            merge=True,
        )
    if checkpoint_policy_observation_mask and not configured_policy_observation_mask:
        OmegaConf.update(
            cfg,
            "world_model.policy_observation_mask_indices",
            list(checkpoint_policy_observation_mask),
            merge=True,
        )
    if checkpoint_broken_joints and not configured_broken_joints:
        OmegaConf.update(
            cfg,
            "world_model.broken_joint_names",
            list(checkpoint_broken_joints),
            merge=True,
        )

    wm_cfg_dict = OmegaConf.to_container(cfg.world_model, resolve=True, throw_on_missing=True)
    assert isinstance(wm_cfg_dict, dict)
    wm_cfg = FlashSACWorldModelEnvConfig(**{str(k): v for k, v in wm_cfg_dict.items()})
    wm_cfg.num_envs = int(cfg.num_imagination_envs)
    env = Go2RWMFlashSACWorldModelEnv(dynamics=dynamics, dataset=dataset, cfg=wm_cfg, device=device)
    policy_action_dim = int(env.single_action_space.shape[0])

    agent_cfg = make_flashsac_config(cfg, device=device)
    agent = create_go2_flashsac_agent(env.observation_space, env.action_space, agent_cfg)
    save_path = str(cfg.save_path).replace(
        "TIMESTAMP", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    )
    save_root = resolve_repo_path(save_path)
    save_root.mkdir(parents=True, exist_ok=True)
    save_config(cfg, save_root / "rwm_flashsac_config.yaml")

    trace_manager = None
    proposal_env = None
    if bool(OmegaConf.select(cfg, "trace.enabled", default=False)):
        from scripts.reinforcement_learning.rwm_trace.v13_runtime_factory import (
            create_v13_online_trace_manager,
        )

        trace_manager, proposal_env = create_v13_online_trace_manager(
            cfg=cfg,
            agent=agent,
            training_config_path=Path(args.config_path or DEFAULT_CONFIG_PATH),
            save_root=save_root,
            device=device,
            v13_repo_root=REPO_ROOT,
        )
    replay_mix_config, real_sampler, sim_sampler, normalizer_metadata = (
        _configure_formal_replay_mix(
            cfg=cfg,
            agent=agent,
            observation_dim=int(env.single_observation_space.shape[-1]),
            action_dim=policy_action_dim,
            agent_cfg=agent_cfg,
            online_sim_sampler=(
                trace_manager.buffer if trace_manager is not None else None
            ),
        )
    )
    policy_resume_value = OmegaConf.select(cfg, "policy_resume_path", default=None)
    policy_resume_path: Path | None = None
    resume_training_state: dict[str, Any] | None = None
    if policy_resume_value:
        policy_resume_path = resolve_repo_path(str(policy_resume_value))
        resume_training_state = _load_training_state(policy_resume_path)
        if (
            resume_training_state is not None
            and args.policy_resume_mode != "full"
        ):
            raise ValueError(
                "Resumable checkpoint requires policy_resume_mode=full."
            )
        if args.policy_resume_mode == "actor_only":
            _load_actor_only(agent, policy_resume_path, device=device)
        else:
            agent.load(str(policy_resume_path))
        uses_rwm_replay = bool(getattr(agent, "uses_rwm_replay", True))
        if (
            args.load_replay_buffer
            and uses_rwm_replay
            and (policy_resume_path / "replay_buffer.pt").exists()
        ):
            agent.load_replay_buffer(str(policy_resume_path))
        elif resume_training_state is not None and uses_rwm_replay:
            raise ValueError(
                "Resumable checkpoint requires --load_replay_buffer."
            )
        elif args.load_replay_buffer and uses_rwm_replay:
            print(f"[Go2-FlashSAC-RWM] replay buffer not found in resume checkpoint={policy_resume_path}")

    trace_resume_value = OmegaConf.select(
        cfg, "trace.resume_checkpoint", default=None
    )
    if trace_manager is not None:
        derived_trace_resume = (
            policy_resume_path / "trace_manager"
            if policy_resume_path is not None
            else None
        )
        if resume_training_state is not None:
            if derived_trace_resume is None or not (
                derived_trace_resume / "COMPLETE"
            ).is_file():
                raise ValueError(
                    "Resumable TRACE checkpoint lacks its co-located manager."
                )
            if (
                trace_resume_value
                and resolve_repo_path(str(trace_resume_value))
                != derived_trace_resume
            ):
                raise ValueError(
                    "Policy and TRACE resume checkpoints must be co-located."
                )
            trace_resume_value = str(derived_trace_resume)
        elif policy_resume_value and not trace_resume_value:
            if (
                derived_trace_resume is not None
                and (derived_trace_resume / "COMPLETE").is_file()
            ):
                trace_resume_value = str(derived_trace_resume)
            else:
                raise ValueError(
                    "Online TRACE policy resume requires a complete "
                    "trace_manager checkpoint."
                )
        if trace_resume_value:
            trace_manager.load_checkpoint(resolve_repo_path(str(trace_resume_value)))
            if args.freeze_trace_scorer_after_resume:
                if policy_resume_path is None:
                    raise ValueError(
                        "--freeze_trace_scorer_after_resume requires a policy resume."
                    )
                trace_manager.scorer_updater = None
                print(
                    "[Go2-FlashSAC-RWM][TRACE] resumed scorer frozen; "
                    "online updater and LLM interaction disabled, "
                    f"binding_sha256={trace_manager.scorer_binding.sha256}"
                )
    if replay_mix_config is not None:
        _write_formal_replay_manifest(
            path=save_root / "v12_replay_mix_manifest.json",
            mix_config=replay_mix_config,
            real_sampler=real_sampler,
            sim_sampler=sim_sampler,
            normalizer_metadata=normalizer_metadata,
            model_path=model_path,
            dataset_path=dataset_path,
            policy_resume_path=policy_resume_path,
            policy_resume_mode=args.policy_resume_mode,
            actor_bc_alpha=float(agent_cfg.actor_bc_alpha),
            actor_learning_starts_updates=int(
                getattr(agent_cfg, "actor_learning_starts_updates", 0)
            ),
            actor_learning_rate_scale=float(
                getattr(agent_cfg, "actor_learning_rate_scale", 1.0)
            ),
        )
    logger = ScalarLogger(save_root, enabled=str(cfg.logger_type).lower() == "tensorboard")

    num_envs = int(cfg.num_imagination_envs)
    total_interaction_steps = max(1, int(int(cfg.num_env_steps) // num_envs))
    initial_policy_update_step = int(getattr(agent, "_update_step"))
    if resume_training_state is not None:
        completed_interaction_step = int(
            resume_training_state["interaction_step"]
        )
        saved_total_interaction_steps = int(
            resume_training_state["total_interaction_steps"]
        )
        horizon_extended = (
            bool(args.allow_training_horizon_extension)
            and total_interaction_steps > saved_total_interaction_steps
            and completed_interaction_step <= saved_total_interaction_steps
        )
        if saved_total_interaction_steps != total_interaction_steps:
            if not horizon_extended:
                raise ValueError(
                    "Total interaction-step budget changed across resume. "
                    "A continuation requires --allow_training_horizon_extension "
                    "and a strictly larger budget."
                )
            print(
                "[Go2-FlashSAC-RWM] extending_training_horizon="
                f"{saved_total_interaction_steps}->{total_interaction_steps}"
            )
        if not 0 <= completed_interaction_step < total_interaction_steps:
            raise ValueError("Checkpoint interaction step is outside the run.")
        if (
            int(resume_training_state["policy_update_step"])
            != initial_policy_update_step
        ):
            raise ValueError(
                "Policy update counter differs between agent and runner state."
            )
        saved_target_updates = resume_training_state.get(
            "target_policy_updates"
        )
        if (
            saved_target_updates is not None
            and args.expected_policy_updates is not None
            and int(saved_target_updates) != int(args.expected_policy_updates)
        ):
            if not (
                horizon_extended
                and int(args.expected_policy_updates) > int(saved_target_updates)
            ):
                raise ValueError(
                    "Target policy-update budget changed across resume. "
                    "A continuation requires a strictly larger update target."
                )
            print(
                "[Go2-FlashSAC-RWM] extending_policy_update_target="
                f"{int(saved_target_updates)}->"
                f"{int(args.expected_policy_updates)}"
            )
        env.load_state_dict(resume_training_state["environment"])
        observations = np.asarray(
            resume_training_state["observations"], dtype=np.float32
        ).copy()
        transition = resume_training_state["transition"]
        update_counter = float(resume_training_state["update_counter"])
        _restore_rng_state(resume_training_state["rng"])
        first_interaction_step = completed_interaction_step + 1
        print(
            "[Go2-FlashSAC-RWM] resumed_at_interaction_step="
            f"{completed_interaction_step}, "
            f"policy_update_step={initial_policy_update_step}"
        )
    else:
        completed_interaction_step = 0
        first_interaction_step = 1
        update_counter = 0.0
        observations, _ = env.reset(seed=int(cfg.seed))
        transition = (
            {"next_observation": observations}
            if policy_resume_value
            else None
        )
    collection_time_acc = 0.0
    learning_time_acc = 0.0
    env_steps_since_log = 0
    checkpoint_writer: AsyncCheckpointWriter | None = None
    checkpoint_interval = int(cfg.save_checkpoint_per_interaction_step)
    checkpoint_offset = int(
        OmegaConf.select(
            cfg,
            "save_checkpoint_interaction_offset",
            default=0,
        )
    )
    if not 0 <= checkpoint_offset < max(1, checkpoint_interval):
        raise ValueError(
            "save_checkpoint_interaction_offset must be in "
            "[0, save_checkpoint_per_interaction_step)."
        )

    print(f"[Go2-FlashSAC-RWM] model={model_path}")
    print(f"[Go2-FlashSAC-RWM] dataset={dataset_path}")
    print(f"[Go2-FlashSAC-RWM] save_root={save_root}")
    print(f"[Go2-FlashSAC-RWM] device={device}, num_envs={num_envs}, interaction_steps={total_interaction_steps}")
    print(f"[Go2-FlashSAC-RWM] full_action_dim={env.full_action_dim}, policy_action_dim={policy_action_dim}")
    print(f"[Go2-FlashSAC-RWM] policy_resume_path={policy_resume_value}")
    print(
        "[Go2-FlashSAC-RWM] "
        f"policy_action_mask_indices={list(wm_cfg.policy_action_mask_indices)}, "
        f"world_model_action_mask_indices={list(wm_cfg.world_model_action_mask_indices)}, "
        f"policy_observation_mask_indices={list(wm_cfg.policy_observation_mask_indices)}"
    )
    if replay_mix_config is not None:
        replay_counts = compute_source_counts(replay_mix_config)
        print(
            "[Go2-FlashSAC-RWM] "
            f"replay_mix={asdict(replay_mix_config)}, "
            f"source_counts={asdict(replay_counts)}, "
            f"real_transitions={real_sampler.num_transitions if real_sampler else 0}, "
            f"sim_transitions={sim_sampler.num_transitions if sim_sampler else 0}, "
            f"normalizer_sha256={normalizer_metadata.get('artifact_sha256')}"
        )
    else:
        print("[Go2-FlashSAC-RWM] replay_mix=legacy_internal_rwm_only")

    run_end_interaction_step = total_interaction_steps
    if args.stop_after_interaction_step is not None:
        run_end_interaction_step = int(args.stop_after_interaction_step)
        if not (
            completed_interaction_step
            < run_end_interaction_step
            <= total_interaction_steps
        ):
            raise ValueError(
                "--stop_after_interaction_step must be after the resumed step "
                "and no greater than the original total interaction-step budget."
            )
        print(
            "[Go2-FlashSAC-RWM] controlled_stop_interaction_step="
            f"{run_end_interaction_step}"
        )

    for interaction_step in tqdm.tqdm(
        range(first_interaction_step, run_end_interaction_step + 1),
        total=total_interaction_steps,
        initial=completed_interaction_step,
        smoothing=0.1,
        mininterval=0.5,
    ):
        checkpoint_writer = _wait_async_checkpoint(
            checkpoint_writer, block=False
        )
        env_step = interaction_step * num_envs
        if trace_manager is not None:
            trace_report = trace_manager.maybe_propose(interaction_step)
            if trace_report is not None:
                logger.update(
                    {
                        "TRACE/candidate_count": trace_report["candidate_count"],
                        "TRACE/selected_count": trace_report["selected_count"],
                        "TRACE/buffer_size": trace_report["trace_buffer_size"],
                        "TRACE/candidate_score_mean": trace_report[
                            "candidate_score_mean"
                        ],
                        "TRACE/selected_score_mean": trace_report[
                            "selected_score_mean"
                        ],
                        "TRACE/rule_bootstrap": float(
                            trace_report["selection_backend"]
                            == "rule_bootstrap"
                        ),
                    }
                )
                print(
                    "[Go2-FlashSAC-RWM][TRACE] "
                    + json.dumps(
                        {
                            key: trace_report.get(key)
                            for key in (
                                "training_step",
                                "proposal_event",
                                "candidate_count",
                                "selected_count",
                                "candidate_command_region_counts",
                                "selected_command_region_counts",
                                "candidate_score_mean",
                                "selected_score_mean",
                                "feedback_status",
                                "trace_buffer_size",
                            )
                        },
                        sort_keys=True,
                    )
                )
        start = time.perf_counter()
        support_preserving_warmup = bool(agent_cfg.actor_support_preserving_warmup)
        if (agent.can_start_training() or policy_resume_value or support_preserving_warmup) and (
            transition is not None or support_preserving_warmup
        ):
            # Uniform random warm-up is catastrophically off-support for a
            # 12-D learned dynamics model.  In the opt-in mode, the still
            # randomly initialized actor explores residuals around the
            # last_action already present in each dataset-reset observation.
            # No expert actor, target action, BC loss, or checkpoint is used.
            action_input = transition or {"next_observation": observations}
            actions = agent.sample_actions(
                interaction_step,
                prev_transition=action_input,
                training=True,
            )
        else:
            actions = np.random.uniform(-1.0, 1.0, size=(num_envs, policy_action_dim)).astype(np.float32)

        next_observations, rewards, terminateds, truncateds, infos = env.step(actions)
        rewards, aligned_reward_delta_mean = (
            _apply_aligned_rwm_simulator_reward(
                cfg=cfg,
                observations=observations,
                final_observations=infos["final_obs"],
                rewards=rewards,
            )
        )
        if bool(
            OmegaConf.select(
                cfg,
                "trace.align_world_model_reward_with_simulator",
                default=False,
            )
        ):
            infos["episode_info"][
                "Imagination/aligned_simulator_quality_delta"
            ] = aligned_reward_delta_mean
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
                    "trace/replay_batch_fraction",
                    "Replay/real_count",
                    "Replay/rwm_count",
                    "Replay/sim_count",
                    "Replay/normalizer_frozen",
                    "Replay/mixed_reward_mean",
                    "Perf/total_fps",
                )
                if k in logged
            }
            print(f"[Go2-FlashSAC-RWM] step={interaction_step} env_step={env_step} {interesting}")
            collection_time_acc = 0.0
            learning_time_acc = 0.0
            env_steps_since_log = 0

        checkpoint_due = bool(
            checkpoint_interval
            and interaction_step >= checkpoint_offset
            and (interaction_step - checkpoint_offset)
            % checkpoint_interval
            == 0
        )
        if checkpoint_due:
            checkpoint_training_state = (
                _make_training_state(
                    interaction_step=interaction_step,
                    total_interaction_steps=total_interaction_steps,
                    update_counter=update_counter,
                    target_policy_updates=args.expected_policy_updates,
                    agent=agent,
                    env=env,
                    observations=observations,
                    transition=transition,
                )
                if trace_manager is not None
                else None
            )
            if trace_manager is not None:
                assert checkpoint_training_state is not None
                checkpoint_writer = _wait_async_checkpoint(
                    checkpoint_writer, block=True
                )
                checkpoint_writer = _start_async_checkpoint(
                    agent,
                    save_root / f"step{interaction_step}",
                    cfg,
                    trace_manager,
                    checkpoint_training_state,
                )
            else:
                _save_checkpoint(
                    agent,
                    save_root / f"step{interaction_step}",
                    cfg,
                    save_replay=False,
                )

    checkpoint_writer = _wait_async_checkpoint(
        checkpoint_writer, block=True
    )
    final_training_state = (
        _make_training_state(
            interaction_step=run_end_interaction_step,
            total_interaction_steps=total_interaction_steps,
            update_counter=update_counter,
            target_policy_updates=args.expected_policy_updates,
            agent=agent,
            env=env,
            observations=observations,
            transition=transition,
        )
        if trace_manager is not None
        else None
    )
    final_checkpoint_path = save_root / f"step{run_end_interaction_step}"
    final_checkpoint_already_complete = (
        final_checkpoint_path / "CHECKPOINT_COMPLETE"
    ).is_file()
    if trace_manager is not None:
        assert final_training_state is not None
        if not final_checkpoint_already_complete:
            checkpoint_writer = _start_async_checkpoint(
                agent,
                final_checkpoint_path,
                cfg,
                trace_manager,
                final_training_state,
            )
            checkpoint_writer = _wait_async_checkpoint(
                checkpoint_writer, block=True
            )
    elif not final_checkpoint_already_complete:
        _save_checkpoint(
            agent,
            final_checkpoint_path,
            cfg,
            save_replay=(
                bool(cfg.save_replay_buffer)
                and bool(getattr(agent, "uses_rwm_replay", True))
            ),
        )
    completed_policy_updates = (
        int(getattr(agent, "_update_step")) - initial_policy_update_step
    )
    final_policy_update_step = int(getattr(agent, "_update_step"))
    if (
        args.expected_policy_updates is not None
        and run_end_interaction_step == total_interaction_steps
    ):
        actual_policy_updates = (
            final_policy_update_step
            if resume_training_state is not None
            else completed_policy_updates
        )
        if actual_policy_updates != int(args.expected_policy_updates):
            raise RuntimeError(
                "Policy-update budget mismatch: expected "
                f"{int(args.expected_policy_updates)}, completed "
                f"{actual_policy_updates}."
            )
    elif args.expected_policy_updates is not None:
        print(
            "[Go2-FlashSAC-RWM] controlled partial run; deferred final "
            "policy-update budget assertion until original total step, "
            f"target={int(args.expected_policy_updates)}, "
            f"current={final_policy_update_step}"
        )
    print(
        "[Go2-FlashSAC-RWM] "
        f"completed_policy_updates={completed_policy_updates}, "
        f"initial_update_step={initial_policy_update_step}, "
        f"final_update_step={final_policy_update_step}"
    )
    logger.close()
    env.close()
    if proposal_env is not None:
        proposal_env.close()


if __name__ == "__main__":
    main()
