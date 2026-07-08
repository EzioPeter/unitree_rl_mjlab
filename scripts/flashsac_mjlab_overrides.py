"""Optional mjlab env overrides driven by FlashSAC Hydra config."""

from __future__ import annotations

from typing import Any


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _as_tuple2(value: Any) -> tuple[float, float]:
    if isinstance(value, str):
        parts = value.replace("[", "").replace("]", "").split(",")
        if len(parts) != 2:
            raise ValueError(f"Expected two comma-separated values, got {value!r}.")
        return float(parts[0]), float(parts[1])
    if len(value) != 2:
        raise ValueError(f"Expected two values, got {value!r}.")
    return float(value[0]), float(value[1])


def apply_mjlab_env_overrides(env_cfg: Any, cfg: Any) -> None:
    """Apply optional command/reward overrides after ``load_env_cfg``.

    This keeps the registered task stable while allowing staged FlashSAC runs to
    tune command distribution and reward coefficients through Hydra overrides.
    """

    env_node = _cfg_get(cfg, "env", None)
    if env_node is None:
        return

    command_distribution = _cfg_get(env_node, "command_distribution", None)
    twist_cmd = env_cfg.commands.get("twist") if hasattr(env_cfg, "commands") else None
    if command_distribution is not None and twist_cmd is not None:
        for key in (
            "xy_command_prob",
            "yaw_command_prob",
            "mixed_command_prob",
            "stand_command_prob",
            "min_lin_speed",
            "min_yaw_speed",
        ):
            value = _cfg_get(command_distribution, key, None)
            if value is not None:
                setattr(twist_cmd, key, float(value))

    command_ranges = _cfg_get(env_node, "command_ranges", None)
    if command_ranges is not None and twist_cmd is not None:
        ranges = getattr(twist_cmd, "ranges", None)
        for key in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            value = _cfg_get(command_ranges, key, None)
            if value is not None:
                setattr(ranges, key, _as_tuple2(value))

    expert_rewards = _cfg_get(env_node, "expert_rewards", None)
    if expert_rewards is None:
        return
    for reward_name, reward_override in expert_rewards.items():
        reward_term = env_cfg.rewards.get(str(reward_name))
        if reward_term is None:
            continue
        weight = _cfg_get(reward_override, "weight", None)
        if weight is not None:
            reward_term.weight = float(weight)
        std = _cfg_get(reward_override, "std", None)
        if std is not None:
            reward_term.params["std"] = float(std)
        command_threshold = _cfg_get(reward_override, "command_threshold", None)
        if command_threshold is not None:
            reward_term.params["command_threshold"] = float(command_threshold)
