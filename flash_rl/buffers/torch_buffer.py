import os
from collections import deque
from dataclasses import dataclass
from typing import Any, Optional, cast

import gymnasium as gym
import numpy as np
import torch

from flash_rl.buffers.base_buffer import BaseBuffer, Batch
from flash_rl.types import NDArray

# Mapping from numpy dtypes to torch dtypes
_NP_TO_TORCH_DTYPE: dict[np.dtype[Any], torch.dtype] = {
    np.dtype(np.float64): torch.float32,  # enforce float32
    np.dtype(np.float32): torch.float32,
    np.dtype(np.int32): torch.int32,
    np.dtype(np.int64): torch.int64,
    np.dtype(np.bool_): torch.bool,
    np.dtype(np.uint8): torch.uint8,
}


@dataclass(frozen=True)
class ReplayBufferWritePlan:
    """Raw NumPy writes safe to execute in a post-CUDA forked child."""

    writes: tuple[
        tuple[str, np.dtype[Any], tuple[int, ...], np.ndarray, int, int],
        ...,
    ]
    max_length: int

    def execute(self) -> None:
        for raw_path, dtype, shape, value, start_idx, count in self.writes:
            mapped = np.memmap(
                raw_path,
                mode="r+",
                dtype=dtype,
                shape=shape,
            )
            if count:
                first = min(count, self.max_length - start_idx)
                second = count - first
                mapped[start_idx : start_idx + first] = value[
                    start_idx : start_idx + first
                ]
                if second:
                    mapped[:second] = value[:second]
            mapped.flush()
            del mapped


def _numpy_dtype_to_torch(dtype: Any) -> torch.dtype:
    """Convert a numpy dtype to a torch dtype, enforcing float32 for float64."""
    dtype = np.dtype(dtype)
    if dtype in _NP_TO_TORCH_DTYPE:
        return _NP_TO_TORCH_DTYPE[dtype]
    return torch.float32


class TorchUniformBuffer(BaseBuffer):
    """
    A uniform experience replay buffer using PyTorch tensors.
    Uniform replay buffer backed by torch tensors on the selected device.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Space[NDArray],
        action_space: gym.spaces.Space[NDArray],
        n_step: int,
        gamma: float,
        max_length: int,
        min_length: int,
        sample_batch_size: int,
        device_type: str,
        obs_storage_dtype: Optional[torch.dtype] = None,
    ):
        super(TorchUniformBuffer, self).__init__(
            observation_space,
            action_space,
            n_step,
            gamma,
            max_length,
            min_length,
            sample_batch_size,
        )
        device_type = (
            device_type
            if device_type.startswith("cuda") and ":" in device_type
            else ("cuda:0" if device_type.startswith("cuda") else "cpu")
        )
        self._device = torch.device(device_type)
        self._obs_storage_dtype = obs_storage_dtype
        self.reset()

    def __len__(self) -> int:
        return self._num_in_buffer

    def reset(self) -> None:
        m = self._max_length
        pin = self._device.type == "cpu" and torch.cuda.is_available()

        observation_shape = (self._observation_space.shape[-1],) if self._observation_space.shape is not None else (0,)
        observation_dtype = _numpy_dtype_to_torch(
            self._observation_space.dtype if self._observation_space.dtype is not None else np.float32
        )

        action_shape = (self._action_space.shape[-1],) if self._action_space.shape is not None else (0,)
        action_dtype = _numpy_dtype_to_torch(
            self._action_space.dtype if self._action_space.dtype is not None else np.float32
        )

        obs_storage_dtype = self._obs_storage_dtype or observation_dtype
        self._observations = torch.empty(
            (m,) + observation_shape, dtype=obs_storage_dtype, device=self._device, pin_memory=pin
        )
        self._next_observations = torch.empty(
            (m,) + observation_shape, dtype=obs_storage_dtype, device=self._device, pin_memory=pin
        )
        self._actions = torch.empty((m,) + action_shape, dtype=action_dtype, device=self._device, pin_memory=pin)
        self._rewards = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._terminateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)
        self._truncateds = torch.empty((m,), dtype=torch.float32, device=self._device, pin_memory=pin)

        self._n_step_transitions: deque[dict[str, Any]] = deque(maxlen=self._n_step)
        self._num_in_buffer = 0
        self._current_idx = 0
        self._total_added = 0
        self._last_saved_total_added = 0
        self._last_checkpoint_path: str | None = None

    def _to_tensor(self, value: Any) -> torch.Tensor:
        """Convert a value to a tensor on the buffer device (cloned if already a tensor)."""
        if isinstance(value, torch.Tensor):
            return value.detach().to(self._device, copy=True)
        return torch.tensor(value, device=self._device)

    def _get_n_step_prev_transition(self) -> Batch:
        """
        Processes n_step_transitions to compute the n-step return, done status,
        and next observation.
        """
        n_step_prev_transition = self._n_step_transitions[0]
        curr_transition = self._n_step_transitions[-1]

        # clone last transition
        n_step_reward = curr_transition["reward"].clone()
        n_step_terminated = curr_transition["terminated"].clone()
        n_step_truncated = curr_transition["truncated"].clone()
        n_step_next_observation = curr_transition["next_observation"].clone()

        for n_step_idx in reversed(range(self._n_step - 1)):
            transition = self._n_step_transitions[n_step_idx]
            reward = transition["reward"]  # (n,)
            terminated = transition["terminated"]  # (n,)
            truncated = transition["truncated"]  # (n,)
            next_observation = transition["next_observation"]  # (n, *obs_shape)

            # compute n-step return
            done = (terminated.bool() | truncated.bool()).float()
            n_step_reward = reward + self._gamma * n_step_reward * (1 - done)

            # assign next observation starting from done
            done_mask = done.bool()
            n_step_terminated[done_mask] = terminated[done_mask]
            n_step_truncated[done_mask] = truncated[done_mask]
            n_step_next_observation[done_mask] = next_observation[done_mask]

        n_step_prev_transition["reward"] = n_step_reward
        n_step_prev_transition["terminated"] = n_step_terminated
        n_step_prev_transition["truncated"] = n_step_truncated
        n_step_prev_transition["next_observation"] = n_step_next_observation

        return cast(Batch, n_step_prev_transition)

    def add(self, transition: Batch) -> None:
        self._n_step_transitions.append({key: self._to_tensor(value) for key, value in transition.items()})

        if len(self._n_step_transitions) >= self._n_step:
            n_step_prev_transition = cast(dict[str, torch.Tensor], self._get_n_step_prev_transition())

            add_batch_size = len(n_step_prev_transition["observation"])
            end_idx = self._current_idx + add_batch_size

            if end_idx <= self._max_length:
                # Contiguous slice — avoids scatter and tensor allocation
                idxs: Any = slice(self._current_idx, end_idx)
            else:
                idxs = (torch.arange(add_batch_size, device=self._device) + self._current_idx) % self._max_length

            self._observations[idxs] = n_step_prev_transition["observation"].to(self._observations.dtype)
            self._next_observations[idxs] = n_step_prev_transition["next_observation"].to(self._next_observations.dtype)
            self._actions[idxs] = n_step_prev_transition["action"].to(self._actions.dtype)
            self._rewards[idxs] = n_step_prev_transition["reward"].to(self._rewards.dtype)
            self._terminateds[idxs] = n_step_prev_transition["terminated"].to(self._terminateds.dtype)
            self._truncateds[idxs] = n_step_prev_transition["truncated"].to(self._truncateds.dtype)

            self._num_in_buffer = min(self._num_in_buffer + add_batch_size, self._max_length)
            self._current_idx = (self._current_idx + add_batch_size) % self._max_length
            self._total_added += add_batch_size

    def can_sample(self) -> bool:
        return self._num_in_buffer >= self._min_length

    def sample(self, sample_idxs: Optional[NDArray] = None) -> Batch:
        if sample_idxs is None:
            idxs = torch.randint(0, self._num_in_buffer, (self._sample_batch_size,), device=self._device)
        else:
            idxs = torch.as_tensor(sample_idxs, device=self._device)

        batch: Batch = {}
        batch["observation"] = self._observations[idxs]
        batch["action"] = self._actions[idxs]
        batch["reward"] = self._rewards[idxs]
        batch["terminated"] = self._terminateds[idxs]
        batch["truncated"] = self._truncateds[idxs]
        batch["next_observation"] = self._next_observations[idxs]

        if self._obs_storage_dtype is not None:
            batch["observation"] = cast(torch.Tensor, batch["observation"]).to(torch.float32)
            batch["next_observation"] = cast(torch.Tensor, batch["next_observation"]).to(torch.float32)

        return batch

    def snapshot_to_cpu(self) -> "TorchUniformBuffer":
        """Copy one consistent ring generation to CPU for an async writer."""

        snapshot = TorchUniformBuffer(
            observation_space=self._observation_space,
            action_space=self._action_space,
            n_step=self._n_step,
            gamma=self._gamma,
            max_length=self._max_length,
            min_length=self._min_length,
            sample_batch_size=self._sample_batch_size,
            device_type="cpu",
            obs_storage_dtype=self._obs_storage_dtype,
        )
        fields = (
            "_observations",
            "_actions",
            "_rewards",
            "_terminateds",
            "_truncateds",
            "_next_observations",
        )
        count = (
            self._max_length
            if self._num_in_buffer == self._max_length
            else self._num_in_buffer
        )
        for name in fields:
            target = cast(torch.Tensor, getattr(snapshot, name))
            source = cast(torch.Tensor, getattr(self, name))
            if count:
                target[:count].copy_(
                    source[:count], non_blocking=self._device.type == "cuda"
                )
        if self._device.type == "cuda":
            torch.cuda.current_stream(self._device).synchronize()
        snapshot._num_in_buffer = self._num_in_buffer
        snapshot._current_idx = self._current_idx
        snapshot._total_added = self._total_added
        snapshot._last_saved_total_added = self._last_saved_total_added
        snapshot._last_checkpoint_path = self._last_checkpoint_path
        snapshot._n_step_transitions.clear()
        for transition in self._n_step_transitions:
            snapshot._n_step_transitions.append(
                {
                    key: torch.as_tensor(value).detach().cpu().clone()
                    for key, value in transition.items()
                }
            )
        return snapshot

    def save(
        self, path: str, *, defer_raw_writes: bool = False
    ) -> ReplayBufferWritePlan | None:
        """
        Save a self-contained rolling replay snapshot.

        The tensor files are fixed-size ring slots.  When a retired checkpoint
        slot is moved into the new checkpoint directory by the runner, only
        rows written since that slot's generation are updated in place.
        args:
            path (str): The full file path (e.g. "checkpoints/replay_buffer.pt").
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        storage_name = "replay_buffer_data"
        storage_dir = os.path.join(os.path.dirname(path), storage_name)
        fields = {
            "observation": self._observations,
            "action": self._actions,
            "reward": self._rewards,
            "terminated": self._terminateds,
            "truncated": self._truncateds,
            "next_observation": self._next_observations,
        }
        previous_total: int | None = None
        if os.path.isfile(path) and os.path.isdir(storage_dir):
            previous = torch.load(
                path, map_location="cpu", weights_only=False
            )
            if (
                previous.get("format_version")
                == "flashsac_torch_uniform_buffer_v4_rolling_raw"
                and int(previous["max_length"]) == self._max_length
                and int(previous["n_step"]) == self._n_step
                and float(previous["gamma"]) == self._gamma
            ):
                previous_layout = previous.get("fields", {})
                storage_complete = all(
                    key in previous_layout
                    and tuple(previous_layout[key].get("shape", ()))
                    == tuple(value.shape)
                    and np.dtype(
                        previous_layout[key].get("numpy_dtype")
                    )
                    == np.dtype(
                        value[:0].detach().cpu().numpy().dtype
                    )
                    and os.path.isfile(
                        os.path.join(
                            storage_dir,
                            str(previous_layout[key].get("file", "")),
                        )
                    )
                    and os.path.getsize(
                        os.path.join(
                            storage_dir,
                            str(previous_layout[key].get("file", "")),
                        )
                    )
                    == int(previous_layout[key].get("bytes", -1))
                    for key, value in fields.items()
                )
                if storage_complete:
                    previous_total = int(previous["total_added"])

        initialize = (
            previous_total is None
            or previous_total < 0
            or previous_total > self._total_added
        )
        if initialize:
            os.makedirs(storage_dir, exist_ok=True)
            previous_total = None
            count = self._num_in_buffer
            start_idx = (
                self._current_idx
                if self._num_in_buffer == self._max_length
                else 0
            )
        else:
            count = min(
                self._total_added - previous_total, self._max_length
            )
            start_idx = (self._current_idx - count) % self._max_length

        layout: dict[str, dict[str, Any]] = {}
        raw_writes: list[
            tuple[
                str,
                np.dtype[Any],
                tuple[int, ...],
                np.ndarray,
                int,
                int,
            ]
        ] = []
        for key, value in fields.items():
            try:
                numpy_dtype = np.dtype(
                    value[:0].detach().cpu().numpy().dtype
                )
            except TypeError as exc:
                raise ValueError(
                    f"Replay field {key} has no raw NumPy dtype."
                ) from exc
            file_name = f"{key}.bin"
            raw_path = os.path.join(storage_dir, file_name)
            expected_bytes = int(value.numel() * value.element_size())
            if initialize or not os.path.isfile(raw_path):
                with open(raw_path, "wb") as handle:
                    handle.truncate(expected_bytes)
            elif os.path.getsize(raw_path) != expected_bytes:
                raise ValueError(
                    f"Rolling replay field size changed for {key}."
                )
            if self._device.type == "cpu":
                raw_writes.append(
                    (
                        raw_path,
                        numpy_dtype,
                        tuple(value.shape),
                        value.detach().numpy(),
                        start_idx,
                        count,
                    )
                )
            else:
                if defer_raw_writes:
                    raise ValueError(
                        "Deferred replay writes require a CPU snapshot."
                    )
                mapped = np.memmap(
                    raw_path,
                    mode="r+",
                    dtype=numpy_dtype,
                    shape=tuple(value.shape),
                )
                if count:
                    first = min(count, self._max_length - start_idx)
                    second = count - first
                    mapped[start_idx : start_idx + first] = (
                        value[start_idx : start_idx + first]
                        .detach()
                        .cpu()
                        .numpy()
                    )
                    if second:
                        mapped[:second] = (
                            value[:second].detach().cpu().numpy()
                        )
                mapped.flush()
                del mapped
            layout[key] = {
                "file": file_name,
                "shape": tuple(value.shape),
                "numpy_dtype": numpy_dtype.str,
                "bytes": expected_bytes,
            }

        dataset: dict[str, Any] = {
            "format_version": "flashsac_torch_uniform_buffer_v4_rolling_raw",
            "storage_directory": storage_name,
            "updated_from_total_added": previous_total,
            "updated_start_idx": start_idx,
            "updated_count": count,
            "fields": layout,
            "num_in_buffer": self._num_in_buffer,
            "current_idx": self._current_idx,
            "total_added": self._total_added,
            "max_length": self._max_length,
            "n_step": self._n_step,
            "gamma": self._gamma,
            "n_step_transitions": [
                {
                    key: torch.as_tensor(value).detach().cpu().clone()
                    for key, value in transition.items()
                }
                for transition in self._n_step_transitions
            ],
        }
        torch.save(dataset, path)
        plan = ReplayBufferWritePlan(
            writes=tuple(raw_writes),
            max_length=self._max_length,
        )
        if defer_raw_writes:
            return plan
        plan.execute()
        return None

    def mark_saved_checkpoint(self, path: str) -> None:
        """Record the committed generation for diagnostics."""

        self._last_checkpoint_path = os.path.abspath(path)
        self._last_saved_total_added = self._total_added

    def load(self, path: str) -> None:
        """
        Load buffer contents and metadata.
        args:
            path (str): The full file path (e.g. "checkpoints/replay_buffer.pt").
        """
        source = os.path.abspath(path)

        def load_one(current: str, seen: set[str]) -> None:
            current = os.path.abspath(current)
            if current in seen:
                raise ValueError("Replay checkpoint parent chain contains a cycle.")
            seen.add(current)
            dataset = torch.load(
                current, map_location="cpu", weights_only=False
            )
            format_version = dataset.get("format_version")
            if format_version == "flashsac_torch_uniform_buffer_v4_rolling_raw":
                if int(dataset["max_length"]) != self._max_length:
                    raise ValueError("Replay checkpoint capacity changed.")
                if int(dataset["n_step"]) != self._n_step:
                    raise ValueError("Replay checkpoint n-step changed.")
                if float(dataset["gamma"]) != self._gamma:
                    raise ValueError("Replay checkpoint gamma changed.")
                storage_name = str(dataset["storage_directory"])
                if storage_name != os.path.basename(storage_name):
                    raise ValueError(
                        "Replay storage directory must be checkpoint-relative."
                    )
                storage_dir = os.path.join(
                    os.path.dirname(current), storage_name
                )
                fields = {
                    "observation": self._observations,
                    "action": self._actions,
                    "reward": self._rewards,
                    "terminated": self._terminateds,
                    "truncated": self._truncateds,
                    "next_observation": self._next_observations,
                }
                layout = dataset["fields"]
                n = int(dataset["num_in_buffer"])
                current_idx = int(dataset["current_idx"])
                total_added = int(dataset["total_added"])
                if not 0 <= n <= self._max_length:
                    raise ValueError(
                        "Replay checkpoint has an invalid row count."
                    )
                if not 0 <= current_idx < self._max_length:
                    raise ValueError(
                        "Replay checkpoint has an invalid write index."
                    )
                if total_added < n:
                    raise ValueError(
                        "Replay checkpoint total-added cursor is invalid."
                    )
                for key, target in fields.items():
                    row = layout[key]
                    if tuple(row["shape"]) != tuple(target.shape):
                        raise ValueError(
                            f"Replay field shape changed for {key}."
                        )
                    raw_path = os.path.join(
                        storage_dir, str(row["file"])
                    )
                    if (
                        not os.path.isfile(raw_path)
                        or os.path.getsize(raw_path) != int(row["bytes"])
                    ):
                        raise ValueError(
                            f"Replay raw storage is incomplete for {key}."
                        )
                    mapped = np.memmap(
                        raw_path,
                        mode="c",
                        dtype=np.dtype(row["numpy_dtype"]),
                        shape=tuple(target.shape),
                    )
                    if n == self._max_length:
                        target.copy_(
                            torch.from_numpy(np.asarray(mapped)).to(
                                self._device
                            )
                        )
                    elif n:
                        target[:n].copy_(
                            torch.from_numpy(np.asarray(mapped[:n])).to(
                                self._device
                            )
                        )
                    del mapped
                self._num_in_buffer = n
                self._current_idx = current_idx
                self._total_added = total_added
            elif format_version == "flashsac_torch_uniform_buffer_v3_incremental":
                if int(dataset["max_length"]) != self._max_length:
                    raise ValueError("Replay checkpoint capacity changed.")
                if int(dataset["n_step"]) != self._n_step:
                    raise ValueError("Replay checkpoint n-step changed.")
                if float(dataset["gamma"]) != self._gamma:
                    raise ValueError("Replay checkpoint gamma changed.")
                parent = dataset.get("parent_checkpoint")
                if parent is not None:
                    if not os.path.isfile(parent):
                        raise FileNotFoundError(
                            f"Replay parent checkpoint is missing: {parent}"
                        )
                    load_one(str(parent), seen)
                else:
                    self._num_in_buffer = 0
                    self._current_idx = 0
                    self._total_added = 0
                    self._n_step_transitions.clear()
                count = int(dataset["delta_count"])
                start_idx = int(dataset["delta_start_idx"])
                if not 0 <= count <= self._max_length:
                    raise ValueError("Replay delta count is invalid.")
                if not 0 <= start_idx < self._max_length:
                    raise ValueError("Replay delta start index is invalid.")
                first = min(count, self._max_length - start_idx)
                second = count - first
                fields = {
                    "observation": self._observations,
                    "action": self._actions,
                    "reward": self._rewards,
                    "terminated": self._terminateds,
                    "truncated": self._truncateds,
                    "next_observation": self._next_observations,
                }
                for key, target in fields.items():
                    value = dataset[key]
                    if int(value.shape[0]) != count:
                        raise ValueError("Replay delta tensor length is invalid.")
                    target[start_idx : start_idx + first] = value[:first]
                    if second:
                        target[:second] = value[first:]
                self._num_in_buffer = int(dataset["num_in_buffer"])
                self._current_idx = int(dataset["current_idx"])
                self._total_added = int(dataset["total_added"])
            else:
                # Backward-compatible full checkpoint.
                n = int(dataset["num_in_buffer"])
                if n < 0 or n > self._max_length:
                    raise ValueError("Replay checkpoint has an invalid row count.")
                if format_version is not None:
                    if format_version != "flashsac_torch_uniform_buffer_v2":
                        raise ValueError("Replay checkpoint format is unsupported.")
                    if int(dataset["max_length"]) != self._max_length:
                        raise ValueError("Replay checkpoint capacity changed.")
                    if int(dataset["n_step"]) != self._n_step:
                        raise ValueError("Replay checkpoint n-step changed.")
                    if float(dataset["gamma"]) != self._gamma:
                        raise ValueError("Replay checkpoint gamma changed.")
                self._observations[:n] = dataset["observation"]
                self._next_observations[:n] = dataset["next_observation"]
                self._actions[:n] = dataset["action"]
                self._rewards[:n] = dataset["reward"]
                self._terminateds[:n] = dataset["terminated"]
                self._truncateds[:n] = dataset["truncated"]
                self._num_in_buffer = n
                self._current_idx = int(dataset["current_idx"])
                self._total_added = n
            self._n_step_transitions.clear()
            for transition in dataset.get("n_step_transitions", []):
                self._n_step_transitions.append(
                    {
                        key: torch.as_tensor(value).to(
                            self._device, copy=True
                        )
                        for key, value in transition.items()
                    }
                )

        load_one(source, set())
        if not 0 <= self._num_in_buffer <= self._max_length:
            raise ValueError("Replay checkpoint has an invalid final row count.")
        if not 0 <= self._current_idx < self._max_length:
            raise ValueError("Replay checkpoint has an invalid final write index.")
        self._last_checkpoint_path = source
        self._last_saved_total_added = self._total_added

    def get_observations(self) -> torch.Tensor:
        return self._observations[: self._num_in_buffer]
