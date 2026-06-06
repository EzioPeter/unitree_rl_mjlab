"""Utilities shared by Go2 FlashSAC-RWM scripts."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium.vector.utils import batch_space
from omegaconf import DictConfig, OmegaConf

from flash_rl.agents.flashSAC.agent import FlashSACConfig


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "go2_flashsac_rwm.yaml"


def resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (REPO_ROOT / p).resolve()


def load_config(config_path: str | Path | None = None, overrides: list[str] | None = None) -> DictConfig:
    cfg = OmegaConf.load(config_path or DEFAULT_CONFIG_PATH)
    if overrides:
        override_cfg = OmegaConf.from_dotlist(overrides)
        cfg = OmegaConf.merge(cfg, override_cfg)
    OmegaConf.resolve(cfg)
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(device_type: str | None = None) -> str:
    if device_type:
        if device_type.startswith("cuda") and torch.cuda.is_available():
            return device_type if ":" in device_type else "cuda:0"
        if device_type.startswith("cpu"):
            return "cpu"
    return "cuda:0" if torch.cuda.is_available() else "cpu"


def make_vector_spaces(num_envs: int, obs_dim: int = 48, action_dim: int = 12) -> tuple[gym.Space, gym.Space]:
    single_obs = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
    single_action = gym.spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    return batch_space(single_obs, num_envs), batch_space(single_action, num_envs)


def make_flashsac_config(cfg: Any, device: str) -> FlashSACConfig:
    agent_cfg = OmegaConf.to_container(cfg.agent, resolve=True, throw_on_missing=True)
    if not isinstance(agent_cfg, dict):
        raise ValueError("cfg.agent must be a mapping")
    agent_cfg = {str(k): v for k, v in agent_cfg.items()}
    agent_cfg["device_type"] = device
    if device.startswith("cpu"):
        agent_cfg["buffer_device_type"] = "cpu"
        agent_cfg["use_amp"] = False
    elif not str(agent_cfg.get("buffer_device_type", "")).startswith("cuda"):
        agent_cfg["buffer_device_type"] = device
    return FlashSACConfig(**agent_cfg)  # type: ignore[arg-type]


def save_config(cfg: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config=cfg, f=path)


def scalarize(value: Any) -> float | int | Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 0:
            return 0.0
        return float(value.float().mean().item())
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return 0.0
        return float(value.astype(np.float32).mean())
    if isinstance(value, (float, int)):
        return value
    return value


def latest_checkpoint(root: str | Path) -> Path:
    root = Path(root)
    candidates = [p for p in root.glob("step*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No step* checkpoint directories found under {root}")

    def step_num(path: Path) -> int:
        try:
            return int(path.name.replace("step", ""))
        except ValueError:
            return -1

    return max(candidates, key=step_num)


def configure_low_thread_env() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    os.environ.setdefault("MKL_NUM_THREADS", "2")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "2")
