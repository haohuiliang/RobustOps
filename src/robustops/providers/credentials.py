"""Read an API key from the environment or an explicit local env file."""

import os
import shlex
from pathlib import Path


def credential(provider):
    name = provider["api_key_env"]
    value = os.environ.get(name)
    if value:
        return value
    path = Path(provider["local_env_file"]) if provider.get("local_env_file") else None
    if path is not None and path.is_file():
        for line in path.read_text().splitlines():
            parts = shlex.split(line, comments=True)
            if parts[:1] == ["export"]:
                parts = parts[1:]
            if len(parts) == 1 and parts[0].startswith(name + "="):
                value = parts[0].split("=", 1)[1]
                if value:
                    return value
    raise ValueError(
        "set " + name + (" or fill " + str(path) if path is not None else "")
    )
