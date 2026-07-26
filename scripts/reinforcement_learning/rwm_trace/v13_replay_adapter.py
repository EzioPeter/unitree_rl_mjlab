"""Narrow reward/n-step bridge to canonical V13 commit b704130.

This module deliberately imports and calls V13's reward and n-step functions.
TRACE owns no reward formula and score never enters this adapter.
"""

from __future__ import annotations

import ast
import hashlib
import importlib
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


V13_EXPECTED_COMMIT = "b704130b848db1835b6a0102043f3b6cb0e46e03"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), default=str
        ).encode("utf-8")
    ).hexdigest()


def _git_commit(root: Path) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout.strip()


def _verify_portable_snapshot(
    root: Path,
    *,
    expected_commit: str,
    protected_sources: Sequence[str],
) -> str:
    """Verify protected source files in a git-archive handoff snapshot."""

    package_root = root.parent
    code_state_path = package_root / "manifests" / "CODE_STATE.txt"
    checksums_path = package_root / "manifests" / "SHA256SUMS"
    if not code_state_path.is_file() or not checksums_path.is_file():
        raise ValueError(
            "Canonical V13 source has no .git metadata and no portable "
            "handoff manifests."
        )
    state = {}
    for line in code_state_path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            state[key.strip()] = value.strip()
    if state.get("canonical_base_commit") != expected_commit:
        raise ValueError(
            "Portable V13 snapshot commit differs from the configured pin."
        )
    if state.get("source_snapshot_kind") != (
        "base_git_archive_plus_current_dirty_overlay"
    ):
        raise ValueError("Unsupported portable V13 snapshot provenance.")

    expected_hashes: dict[str, str] = {}
    for line in checksums_path.read_text(encoding="utf-8").splitlines():
        digest, separator, relative = line.partition("  ")
        if separator:
            expected_hashes[relative.removeprefix("./")] = digest
    for relative in protected_sources:
        manifest_key = f"source/{relative}"
        expected_hash = expected_hashes.get(manifest_key)
        source_path = root / relative
        if (
            expected_hash is None
            or not source_path.is_file()
            or _sha256_file(source_path) != expected_hash
        ):
            raise ValueError(
                "Protected V13 source differs from the verified portable "
                f"snapshot: {relative}"
            )
    return expected_commit


def _world_env_semantics_ast(source: str) -> str:
    """Hash the protected environment semantics, excluding checkpoint I/O."""

    tree = ast.parse(source)
    for node in tree.body:
        if (
            isinstance(node, ast.ClassDef)
            and node.name == "Go2RWMFlashSACWorldModelEnv"
        ):
            node.body = [
                child
                for child in node.body
                if not (
                    isinstance(
                        child, (ast.FunctionDef, ast.AsyncFunctionDef)
                    )
                    and child.name in {"state_dict", "load_state_dict"}
                )
            ]
    return ast.dump(tree, include_attributes=False)


def _extend_package_path(package_name: str, path: Path) -> None:
    package = importlib.import_module(package_name)
    package_path = getattr(package, "__path__", None)
    if package_path is None:
        raise RuntimeError(f"{package_name!r} is not a package.")
    value = str(path)
    if value not in package_path:
        package_path.append(value)


def _bind_v13_imports(root: Path) -> dict[str, Any]:
    """Expose the independent V13 worktree through existing namespace packages."""

    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    _extend_package_path("scripts", root / "scripts")
    _extend_package_path(
        "scripts.reinforcement_learning",
        root / "scripts" / "reinforcement_learning",
    )
    world_env = importlib.import_module(
        "scripts.reinforcement_learning.rwm_flashsac.world_model_env"
    )
    replay_builder = importlib.import_module(
        "scripts.reinforcement_learning.rwm_flashsac.build_go2_real_replay"
    )
    rewards = importlib.import_module("src.tasks.rwm_velocity.mdp.rewards")
    extractors = importlib.import_module("src.tasks.rwm_velocity.mdp.extractors")
    return {
        "world_env": world_env,
        "replay_builder": replay_builder,
        "rewards": rewards,
        "extractors": extractors,
    }


class CanonicalV13ReplaySemantics:
    """Exact V13 public reward + configured n-step materialization."""

    def __init__(
        self,
        *,
        v13_repo_root: str | Path,
        training_config_path: str | Path,
        resolved_config: Any | None = None,
        expected_commit: str = V13_EXPECTED_COMMIT,
        device: torch.device | str = "cpu",
    ) -> None:
        from omegaconf import OmegaConf

        self.repo_root = Path(v13_repo_root).expanduser().resolve()
        self.config_path = Path(training_config_path).expanduser().resolve()
        protected_sources = (
            "scripts/reinforcement_learning/rwm_flashsac/build_go2_real_replay.py",
            "src/tasks/rwm_velocity/mdp/rewards.py",
            "src/tasks/rwm_velocity/mdp/extractors.py",
        )
        world_env_relative = (
            "scripts/reinforcement_learning/rwm_flashsac/world_model_env.py"
        )
        if (self.repo_root / ".git").exists():
            head_commit = _git_commit(self.repo_root)
            ancestor = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_root),
                    "merge-base",
                    "--is-ancestor",
                    expected_commit,
                    head_commit,
                ],
                check=False,
            )
            if ancestor.returncode != 0:
                raise ValueError(
                    "Canonical V13 worktree is not descended from the configured pin."
                )
            protected_diff = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_root),
                    "diff",
                    "--quiet",
                    expected_commit,
                    "--",
                    *protected_sources,
                ],
                check=False,
            )
            if protected_diff.returncode != 0:
                raise ValueError(
                    "Protected V13 reward/n-step/observation sources differ "
                    "from the pin."
                )
            pinned_world_env = subprocess.run(
                [
                    "git",
                    "-C",
                    str(self.repo_root),
                    "show",
                    f"{expected_commit}:{world_env_relative}",
                ],
                check=True,
                text=True,
                capture_output=True,
            ).stdout
            current_world_env = (
                self.repo_root / world_env_relative
            ).read_text(encoding="utf-8")
            if _world_env_semantics_ast(
                current_world_env
            ) != _world_env_semantics_ast(pinned_world_env):
                raise ValueError(
                    "Protected V13 world-environment semantics differ from the pin."
                )
        else:
            head_commit = _verify_portable_snapshot(
                self.repo_root,
                expected_commit=expected_commit,
                protected_sources=(*protected_sources, world_env_relative),
            )
        if resolved_config is None and not self.config_path.is_file():
            raise FileNotFoundError(self.config_path)
        self.source_commit = head_commit
        self.reward_base_commit = expected_commit
        self.config_sha256 = (
            _sha256_file(self.config_path)
            if resolved_config is None
            else _canonical_sha256(
                OmegaConf.to_container(
                    resolved_config, resolve=True, throw_on_missing=True
                )
            )
        )
        self.device = torch.device(device)
        modules = _bind_v13_imports(self.repo_root)
        self._configure_reward_state = modules["world_env"].configure_go2_reward_state
        self._wm_config_type = modules["world_env"].FlashSACWorldModelEnvConfig
        self._compute_reward = modules["rewards"].compute_go2_imagination_reward
        self._make_policy_obs = modules["extractors"].make_go2_policy_obs
        self._build_n_step = modules["replay_builder"]._build_n_step

        config = (
            OmegaConf.load(self.config_path)
            if resolved_config is None
            else resolved_config
        )
        OmegaConf.resolve(config)
        wm_value = OmegaConf.to_container(
            config.world_model, resolve=True, throw_on_missing=True
        )
        if not isinstance(wm_value, dict):
            raise TypeError("V13 config.world_model must resolve to a mapping.")
        self._wm_cfg = self._wm_config_type(**wm_value)
        self._gamma = float(config.agent.gamma)
        self._n_step = int(config.agent.n_step)
        if not 0.0 < self._gamma <= 1.0 or self._n_step < 1:
            raise ValueError("V13 training config exposes invalid gamma/n_step.")
        reward_config = {
            key: value
            for key, value in asdict(self._wm_cfg).items()
            if key.startswith("reward_")
            or key in {"step_dt", "uncertainty_penalty_weight"}
        }
        self._reward_config_sha256 = _canonical_sha256(reward_config)

    @property
    def gamma(self) -> float:
        return self._gamma

    @property
    def n_step(self) -> int:
        return self._n_step

    @property
    def reward_config_sha256(self) -> str:
        return self._reward_config_sha256

    def _tensor(
        self,
        trajectory: Mapping[str, Any],
        key: str,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        if key not in trajectory:
            raise KeyError(f"V13 replay materialization requires {key!r}.")
        device_fields = trajectory.get("_device_fields")
        value = (
            device_fields[key]
            if isinstance(device_fields, Mapping) and key in device_fields
            else trajectory[key]
        )
        return torch.as_tensor(value, dtype=dtype, device=self.device)

    def _stack_field(
        self,
        trajectories: Sequence[Mapping[str, Any]],
        key: str,
        *,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        values: list[Any] = []
        all_on_target = True
        for trajectory in trajectories:
            if key not in trajectory:
                raise KeyError(f"V13 replay materialization requires {key!r}.")
            device_fields = trajectory.get("_device_fields")
            value = (
                device_fields[key]
                if isinstance(device_fields, Mapping) and key in device_fields
                else trajectory[key]
            )
            values.append(value)
            all_on_target = (
                all_on_target
                and isinstance(value, torch.Tensor)
                and value.device == self.device
            )
        if all_on_target:
            return torch.stack(
                [value.to(dtype=dtype) for value in values]
            )
        # CPU-backed unit tests/offline materialization take one bulk transfer
        # per field rather than one transfer per selected trajectory.
        cpu = torch.stack(
            [
                torch.as_tensor(value, dtype=dtype, device="cpu")
                for value in values
            ]
        )
        return cpu.to(self.device)

    def _materialize_same_length(
        self,
        trajectories: Sequence[Mapping[str, Any]],
        *,
        length: int,
    ) -> list[Mapping[str, torch.Tensor]]:
        if not trajectories or length < 1:
            raise ValueError("V13 batch materializer requires non-empty trajectories.")
        states = self._stack_field(trajectories, "states")
        actions = self._stack_field(trajectories, "actions")
        next_states = self._stack_field(trajectories, "next_states")
        commands = self._stack_field(trajectories, "commands")
        contacts = self._stack_field(trajectories, "contacts")
        previous_actions = self._stack_field(
            trajectories, "prev_actions"
        )
        terminated = self._stack_field(
            trajectories, "terminations", dtype=torch.bool
        ).reshape(len(trajectories), -1)
        truncated_rows = [
            (
                row
                if "truncations" in row
                else {
                    **row,
                    "truncations": torch.zeros(
                        length, dtype=torch.bool
                    ),
                }
            )
            for row in trajectories
        ]
        truncated = self._stack_field(
            truncated_rows, "truncations", dtype=torch.bool
        ).reshape(len(trajectories), -1)
        tensors = (
            states,
            actions,
            next_states,
            commands,
            contacts,
            previous_actions,
            terminated,
            truncated,
        )
        if any(value.shape[1] != length for value in tensors):
            raise ValueError("V13 candidate trajectory fields have inconsistent lengths.")
        if states.shape[2:] != (45,) or next_states.shape != states.shape:
            raise ValueError("V13 reward adapter requires 45D state and next_state.")
        if actions.shape[2:] != (12,) or previous_actions.shape != actions.shape:
            raise ValueError("V13 reward adapter requires 12D full actions.")

        batch_size = len(trajectories)
        reward_state = self._configure_reward_state(
            self._wm_cfg,
            num_envs=batch_size,
            action_dim=12,
            device=self.device,
        )
        reward_state.last_joint_vel.copy_(states[:, 0, 21:33])
        reward_state.last_action.copy_(previous_actions[:, 0])
        reward_state.base_lin_vel_xy_ema.copy_(states[:, 0, 0:2])
        reward_state.base_yaw_vel_ema.copy_(states[:, 0, 5])
        rewards: list[torch.Tensor] = []
        for step in range(length):
            reward, _ = self._compute_reward(
                state=next_states[:, step],
                action=actions[:, step],
                command=commands[:, step],
                foot_contact=contacts[:, step],
                episode_length=torch.full(
                    (batch_size,),
                    step,
                    dtype=torch.long,
                    device=self.device,
                ),
                reward_state=reward_state,
                epistemic_uncertainty=torch.zeros(
                    batch_size, device=self.device
                ),
            )
            reward = reward + terminated[:, step].float() * float(
                self._wm_cfg.reward_termination_penalty
            )
            rewards.append(reward)
        one_step_reward = torch.stack(rewards, dim=1)

        critic_observation = self._make_policy_obs(
            states,
            commands,
            previous_actions,
        )
        critic_next_observation = self._make_policy_obs(
            next_states,
            commands,
            actions,
        )
        policy_mask = tuple(int(index) for index in self._wm_cfg.policy_action_mask_indices)
        keep = [index for index in range(12) if index not in set(policy_mask)]
        if not keep:
            raise ValueError("V13 policy action mask removes every action.")

        episode_ids = torch.zeros(length, batch_size, dtype=torch.long)
        timesteps = torch.arange(length, dtype=torch.long).unsqueeze(1).expand(
            -1, batch_size
        )
        replay = self._build_n_step(
            observations=critic_observation.detach().cpu().transpose(0, 1),
            actions=actions.detach().cpu()[:, :, keep].transpose(0, 1),
            rewards=one_step_reward.detach().cpu().transpose(0, 1),
            terminated=terminated.detach().cpu().transpose(0, 1),
            truncated=truncated.detach().cpu().transpose(0, 1),
            next_observations=critic_next_observation.detach().cpu().transpose(
                0, 1
            ),
            episode_ids=episode_ids,
            timesteps=timesteps,
            eligible=None,
            gamma=self.gamma,
            n_step=self.n_step,
        )
        keys = (
            "observation",
            "action",
            "reward",
            "terminated",
            "truncated",
            "next_observation",
        )
        source_env_index = replay["source_env_index"].long()
        return [
            {
                key: replay[key][source_env_index == env_index]
                .detach()
                .to("cpu", dtype=torch.float32)
                .contiguous()
                for key in keys
            }
            for env_index in range(batch_size)
        ]

    def materialize_many(
        self,
        trajectories: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, torch.Tensor]]:
        if not trajectories:
            return []
        groups: dict[int, list[int]] = {}
        for index, trajectory in enumerate(trajectories):
            states = trajectory.get("states")
            if states is None:
                raise KeyError("V13 replay materialization requires 'states'.")
            length = int(len(states))
            if length < 1:
                raise ValueError("V13 candidate trajectory is empty.")
            groups.setdefault(length, []).append(index)
        result: list[Mapping[str, torch.Tensor] | None] = [
            None
        ] * len(trajectories)
        for length, indices in groups.items():
            materialized = self._materialize_same_length(
                [trajectories[index] for index in indices],
                length=length,
            )
            for index, replay in zip(indices, materialized, strict=True):
                result[index] = replay
        if any(replay is None for replay in result):
            raise AssertionError("V13 batch materializer lost a trajectory.")
        return [replay for replay in result if replay is not None]

    def materialize(
        self,
        trajectory: Mapping[str, Any],
    ) -> Mapping[str, torch.Tensor]:
        return self.materialize_many([trajectory])[0]
