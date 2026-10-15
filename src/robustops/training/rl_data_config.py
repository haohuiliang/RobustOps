"""Resolve RL data inputs and verify their contents across continuations."""

import hashlib
import json
from pathlib import Path

DEFAULT_DATA_SOURCES = {
    "task_manifest": "data/task_sets/enterpriseops.jsonl",
    "train_pool": "data/task_sets/recovery_train_pool.json",
    "fault_selector": "data/fault_selector/train.jsonl",
    "sft_records": "data/sft/train/records.jsonl",
    "sft_provenance": "data/sft/train/provenance.jsonl",
    "sft_fault_exposure": "data/sft/train/fault_exposure.json",
    "sft_validation": "data/sft/validation/validation.parquet",
    "validation_recipe": "recipes/support/validation.json",
    "validation_policy": None,
}
OPTIONAL_SOURCES = {"sft_provenance", "sft_fault_exposure", "validation_policy"}


def resolve_data_sources(root, sources=None):
    if sources is not None and not isinstance(sources, dict):
        raise ValueError("RL data must be an object of input paths")
    sources = sources or {}
    unknown = set(sources) - set(DEFAULT_DATA_SOURCES)
    if unknown:
        raise ValueError(f"unsupported RL data inputs: {sorted(unknown)}")
    result = {}
    for name, value in dict(DEFAULT_DATA_SOURCES, **sources).items():
        if value is None and name in OPTIONAL_SOURCES:
            result[name] = None
        elif not isinstance(value, str) or not value.strip():
            raise ValueError(f"data.{name} must be a nonempty path")
        else:
            result[name] = str((Path(root) / Path(value).expanduser()).resolve())
    return result


def file_fingerprints(paths):
    result = {}
    for value in paths:
        if value is None:
            continue
        path = Path(value).resolve()
        # Record absence as well: a new exposure manifest changes fallback semantics.
        if path.is_file():
            with path.open('rb') as stream:
                result[str(path)] = hashlib.file_digest(stream, 'sha256').hexdigest()
        else:
            result[str(path)] = None
    return result


def verify_saved_inputs(directory):
    summary = json.loads((Path(directory) / 'data_summary.json').read_text())
    expected = summary.get('input_fingerprints')
    if expected is not None and file_fingerprints(expected) != expected:
        raise ValueError('RL input files changed since data preparation; start a fresh run')
