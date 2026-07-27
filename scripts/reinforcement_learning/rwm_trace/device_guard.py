"""Fail-closed physical GPU guard for TRACE processes.

This module intentionally does not import torch or initialize CUDA.  Device
identity is resolved from ``nvidia-smi`` inventory plus
``CUDA_VISIBLE_DEVICES`` before a CUDA-capable process is started.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import os
import re
import subprocess
from typing import Mapping, Sequence


FORBIDDEN_PHYSICAL_GPU_INDICES = frozenset({2, 7})
_GPU_UUID_RE = re.compile(r"^GPU-[A-Za-z0-9-]+$")


class DeviceGuardError(RuntimeError):
    """Base error for device identity and policy failures."""


class UnverifiableCudaDeviceError(DeviceGuardError):
    """Raised when a logical CUDA device cannot be proven physical."""


class ForbiddenCudaDeviceError(DeviceGuardError):
    """Raised when a selected logical device resolves to GPU 2 or 7."""


@dataclass(frozen=True)
class PhysicalGpu:
    """One physical device reported by ``nvidia-smi``."""

    physical_index: int
    uuid: str
    pci_bus_id: str

    def to_metadata(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ResolvedCudaDevice:
    """Auditable logical-to-physical CUDA device resolution."""

    logical_index: int
    visible_token: str
    cuda_visible_devices: str | None
    physical_index: int
    uuid: str
    pci_bus_id: str

    def to_metadata(self) -> dict[str, int | str | None]:
        return asdict(self)


def parse_nvidia_smi_inventory(output: str) -> tuple[PhysicalGpu, ...]:
    """Parse ``index,uuid,pci.bus_id`` CSV output from ``nvidia-smi``.

    The result is ordered by the reported physical index.  Duplicate or
    incomplete identities are rejected because they cannot support a
    fail-closed mapping.
    """

    devices: list[PhysicalGpu] = []
    for line_number, raw_line in enumerate(output.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3:
            raise UnverifiableCudaDeviceError(
                f"Invalid nvidia-smi inventory line {line_number}: {raw_line!r}"
            )
        index_text, uuid, pci_bus_id = fields
        try:
            physical_index = int(index_text)
        except ValueError as exc:
            raise UnverifiableCudaDeviceError(
                f"Invalid physical GPU index on line {line_number}: {index_text!r}"
            ) from exc
        if physical_index < 0 or not _GPU_UUID_RE.fullmatch(uuid) or not pci_bus_id:
            raise UnverifiableCudaDeviceError(
                f"Incomplete GPU identity on line {line_number}: {raw_line!r}"
            )
        devices.append(
            PhysicalGpu(
                physical_index=physical_index,
                uuid=uuid,
                pci_bus_id=pci_bus_id,
            )
        )

    if not devices:
        raise UnverifiableCudaDeviceError("nvidia-smi returned no physical GPUs.")

    indices = [device.physical_index for device in devices]
    uuids = [device.uuid for device in devices]
    pci_ids = [device.pci_bus_id.lower() for device in devices]
    if len(indices) != len(set(indices)):
        raise UnverifiableCudaDeviceError("Duplicate physical GPU indices in inventory.")
    if len(uuids) != len(set(uuids)):
        raise UnverifiableCudaDeviceError("Duplicate GPU UUIDs in inventory.")
    if len(pci_ids) != len(set(pci_ids)):
        raise UnverifiableCudaDeviceError("Duplicate GPU PCI bus IDs in inventory.")
    return tuple(sorted(devices, key=lambda device: device.physical_index))


def query_nvidia_smi_inventory() -> tuple[PhysicalGpu, ...]:
    """Query GPU identity without importing a CUDA runtime."""

    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,pci.bus_id",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise UnverifiableCudaDeviceError(
            "Unable to query physical GPU identity with nvidia-smi."
        ) from exc
    return parse_nvidia_smi_inventory(completed.stdout)


def _resolve_uuid_token(
    token: str, inventory: Sequence[PhysicalGpu]
) -> PhysicalGpu:
    matches = [device for device in inventory if device.uuid.startswith(token)]
    if len(matches) != 1:
        raise UnverifiableCudaDeviceError(
            f"CUDA_VISIBLE_DEVICES UUID token {token!r} matched {len(matches)} GPUs."
        )
    return matches[0]


def _visible_devices(
    inventory: Sequence[PhysicalGpu],
    cuda_visible_devices: str | None,
) -> tuple[tuple[str, PhysicalGpu], ...]:
    if not inventory:
        raise UnverifiableCudaDeviceError("Physical GPU inventory is empty.")

    by_index = {device.physical_index: device for device in inventory}
    if len(by_index) != len(inventory):
        raise UnverifiableCudaDeviceError("Physical GPU inventory has duplicate indices.")

    if cuda_visible_devices is None:
        return tuple(
            (str(device.physical_index), device)
            for device in sorted(inventory, key=lambda item: item.physical_index)
        )

    raw_tokens = [token.strip() for token in cuda_visible_devices.split(",")]
    if not raw_tokens or any(not token for token in raw_tokens):
        raise UnverifiableCudaDeviceError(
            "CUDA_VISIBLE_DEVICES is empty or contains an empty token."
        )
    if raw_tokens == ["-1"]:
        raise UnverifiableCudaDeviceError("CUDA_VISIBLE_DEVICES disables all GPUs.")
    if any(token == "-1" for token in raw_tokens):
        raise UnverifiableCudaDeviceError(
            "CUDA_VISIBLE_DEVICES mixes the disable token with device tokens."
        )

    visible: list[tuple[str, PhysicalGpu]] = []
    for token in raw_tokens:
        if token.isdecimal():
            physical_index = int(token)
            device = by_index.get(physical_index)
            if device is None:
                raise UnverifiableCudaDeviceError(
                    f"CUDA_VISIBLE_DEVICES index {physical_index} is absent from inventory."
                )
        elif token.startswith("GPU-"):
            device = _resolve_uuid_token(token, inventory)
        else:
            raise UnverifiableCudaDeviceError(
                f"Unsupported CUDA_VISIBLE_DEVICES token: {token!r}"
            )
        visible.append((token, device))

    identities = [device.uuid for _, device in visible]
    if len(identities) != len(set(identities)):
        raise UnverifiableCudaDeviceError(
            "CUDA_VISIBLE_DEVICES resolves multiple tokens to the same physical GPU."
        )
    return tuple(visible)


def _parse_logical_index(logical_device: int | str) -> int:
    if isinstance(logical_device, bool):
        raise UnverifiableCudaDeviceError("Boolean CUDA device identifiers are invalid.")
    if isinstance(logical_device, int):
        logical_index = logical_device
    elif isinstance(logical_device, str):
        value = logical_device.strip()
        if value.startswith("cuda:"):
            value = value[5:]
        if not value.isdecimal():
            raise UnverifiableCudaDeviceError(
                f"Invalid logical CUDA device: {logical_device!r}"
            )
        logical_index = int(value)
    else:
        raise UnverifiableCudaDeviceError(
            f"Unsupported logical CUDA device type: {type(logical_device).__name__}"
        )
    if logical_index < 0:
        raise UnverifiableCudaDeviceError("Logical CUDA index must be non-negative.")
    return logical_index


def resolve_cuda_device(
    logical_device: int | str,
    *,
    inventory: Sequence[PhysicalGpu],
    cuda_visible_devices: str | None,
) -> ResolvedCudaDevice:
    """Resolve one logical CUDA ordinal and enforce the physical denylist."""

    logical_index = _parse_logical_index(logical_device)
    visible = _visible_devices(inventory, cuda_visible_devices)
    if logical_index >= len(visible):
        raise UnverifiableCudaDeviceError(
            f"Logical cuda:{logical_index} is outside {len(visible)} visible GPUs."
        )
    token, physical = visible[logical_index]
    if physical.physical_index in FORBIDDEN_PHYSICAL_GPU_INDICES:
        raise ForbiddenCudaDeviceError(
            f"Logical cuda:{logical_index} resolves to forbidden physical GPU "
            f"{physical.physical_index} ({physical.uuid}, {physical.pci_bus_id})."
        )
    return ResolvedCudaDevice(
        logical_index=logical_index,
        visible_token=token,
        cuda_visible_devices=cuda_visible_devices,
        physical_index=physical.physical_index,
        uuid=physical.uuid,
        pci_bus_id=physical.pci_bus_id,
    )


def guard_cuda_device(
    logical_device: int | str,
    *,
    environ: Mapping[str, str] | None = None,
    inventory: Sequence[PhysicalGpu] | None = None,
) -> ResolvedCudaDevice:
    """Query, resolve, and guard a CUDA device before importing CUDA libraries."""

    environment = os.environ if environ is None else environ
    known_inventory = query_nvidia_smi_inventory() if inventory is None else inventory
    return resolve_cuda_device(
        logical_device,
        inventory=known_inventory,
        cuda_visible_devices=environment.get("CUDA_VISIBLE_DEVICES"),
    )


__all__ = [
    "DeviceGuardError",
    "ForbiddenCudaDeviceError",
    "FORBIDDEN_PHYSICAL_GPU_INDICES",
    "PhysicalGpu",
    "ResolvedCudaDevice",
    "UnverifiableCudaDeviceError",
    "guard_cuda_device",
    "parse_nvidia_smi_inventory",
    "query_nvidia_smi_inventory",
    "resolve_cuda_device",
]
