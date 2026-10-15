"""Discover visible NVIDIA devices without importing torch or starting CUDA work."""
from __future__ import annotations

import csv
from dataclasses import asdict, dataclass
import os
import subprocess

from robustops.launch.config import LaunchConfigError


@dataclass(frozen=True)
class GPU:
    index: int
    uuid: str
    name: str
    memory_mib: int


def visible_gpus() -> list[GPU]:
    mask = os.environ.get("CUDA_VISIBLE_DEVICES")
    if mask is not None and mask.strip() in ("", "-1"):
        return []
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,uuid,name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=False,
        )
    except FileNotFoundError:
        return []
    if result.returncode:
        raise LaunchConfigError(f"GPU discovery failed: {result.stderr.strip()}")
    devices = [GPU(int(index), uuid.strip(), name.strip(), int(memory))
               for index, uuid, name, memory in csv.reader(result.stdout.splitlines(), skipinitialspace=True)]
    if mask is None:
        return devices
    visible = []
    for token in mask.split(","):
        token = token.strip()
        matches = [gpu for gpu in devices
                   if str(gpu.index) == token or (token.startswith("GPU-") and gpu.uuid.startswith(token))]
        if len(matches) != 1 or matches[0] in visible:
            raise LaunchConfigError(f"CUDA_VISIBLE_DEVICES contains an unknown or duplicate GPU: {token!r}")
        visible.append(matches[0])
    return visible


@dataclass(frozen=True)
class GPUSelection:
    detected: list[GPU]
    count: int
    explicit: bool

    @property
    def devices(self) -> list[GPU]:
        return self.detected[:self.count]

    @property
    def cuda_visible_devices(self) -> str:
        return ",".join(gpu.uuid for gpu in self.devices)

    def record(self) -> dict:
        return dict(detected_gpu_count=len(self.detected),
                    visible_devices=[asdict(gpu) for gpu in self.detected],
                    selected_gpu_count=self.count,
                    selection="cli_override" if self.explicit else "auto",
                    selected_devices=[asdict(gpu) for gpu in self.devices],
                    offline_plan=self.count > len(self.detected))


def select_gpus(override: int | None, *, dry_run: bool) -> GPUSelection:
    if override is not None and (type(override) is not int or override <= 0):
        raise LaunchConfigError("--gpus must be a positive integer")
    devices = visible_gpus()
    count = len(devices) if override is None else override
    if not count:
        raise LaunchConfigError("No visible NVIDIA GPUs; use --gpus N --dry-run for an offline plan")
    if count > len(devices) and not dry_run:
        raise LaunchConfigError(f"Requested {count} GPUs, but only {len(devices)} are visible")
    return GPUSelection(devices, count, override is not None)


def device_for_worker(logical_index: int) -> GPU:
    devices = visible_gpus()
    if logical_index < 0 or logical_index >= len(devices):
        raise LaunchConfigError(f"GPU worker {logical_index} is outside the visible device set")
    return devices[logical_index]
