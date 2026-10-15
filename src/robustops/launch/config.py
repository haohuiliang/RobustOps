from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


class LaunchConfigError(ValueError):
    """Raised when a launch recipe is incomplete or inconsistent."""


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_project_path(root: Path, value: str | os.PathLike[str]) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else root / path


def load_recipe(root: Path, value: str | os.PathLike[str]) -> tuple[Path, dict[str, Any]]:
    path = resolve_project_path(root, value)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise LaunchConfigError(f"launch recipe does not exist: {path}") from exc
    except json.JSONDecodeError as exc:
        raise LaunchConfigError(f"invalid JSON launch recipe {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise LaunchConfigError(f"launch recipe must be a JSON object: {path}")
    if payload.get("schema_version") != 1:
        raise LaunchConfigError(f"unsupported schema_version in {path}: {payload.get('schema_version')!r}")
    return path, payload


def require_mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise LaunchConfigError(f"{key} must be an object")
    return value


def require_string(payload: dict[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise LaunchConfigError(f"{key} must be a non-empty string")
    return value


def resolve_model_path(root: Path, logical_path: str) -> Path:
    return resolve_project_path(root, os.environ.get("ROBUSTOPS_MODEL_PATH", logical_path))


def resolve_training_output_parent(root: Path, logical_path: str) -> Path:
    return resolve_project_path(root, logical_path)


def require_string_list(value: Any, name: str) -> list[str]:
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(item, str) and item for item in value)
        or len(value) != len(set(value))
    ):
        raise LaunchConfigError(f"{name} must be a non-empty list of unique strings")
    return value
