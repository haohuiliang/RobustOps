"""Optional host-memory checks for ordinary hosts and constrained containers."""

import math
import os
from pathlib import Path

from robustops.launch.config import LaunchConfigError


def host_memory_bytes():
    capacities = []
    try:
        capacities.append(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError, AttributeError):
        pass
    for path in (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    ):
        try:
            value = path.read_text().strip()
            if value != "max":
                capacities.append(int(value))
        except (OSError, ValueError):
            continue
    return min((n for n in capacities if n > 0), default=None)


def check_host_memory(minimum_gib=0):
    if type(minimum_gib) not in (int, float) or not math.isfinite(minimum_gib) or minimum_gib < 0:
        raise LaunchConfigError("min_host_memory_gib must be finite and nonnegative")
    capacity = host_memory_bytes()
    if minimum_gib and (capacity is None or capacity < minimum_gib * 1024**3):
        detected = "unknown" if capacity is None else f"{capacity / 1024**3:.1f} GiB"
        raise LaunchConfigError(f"Configured host memory minimum is {minimum_gib} GiB; detected {detected}")
    return capacity
