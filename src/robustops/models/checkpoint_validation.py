"""Inspect local model exports and select checkpoints without fixed experiment identities."""

import hashlib
import json
from pathlib import Path

from robustops.launch.config import resolve_project_path
from robustops.training.recovery_rl_storage import validate_hf_export


def inspect_local_model(root, model, *, lineage=None, allow_missing=False):
    """Validate a loadable export and record its actual metadata without loading weights."""
    model = resolve_project_path(root, str(model)).resolve()
    ready = bool(list(model.glob("*.safetensors")))
    if not ready and not allow_missing:
        raise ValueError(f"Missing model weights: {model}")
    details = validate_hf_export(model, allowed_dtypes=("F32", "F16", "BF16")) if ready else None
    metadata, hashes = {}, {}
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "MODEL_PROVENANCE.json"):
        path = model / name
        if path.is_file():
            payload = json.loads(path.read_text())
            if not isinstance(payload, dict):
                raise ValueError(f"Model metadata must be an object: {path}")
            metadata[name] = payload
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    source = None
    if lineage is not None:
        path = resolve_project_path(root, lineage)
        source = json.loads(path.read_text())
        if not isinstance(source, dict):
            raise ValueError("model.lineage must contain an object")
    return dict(model_path=str(model), ready=ready, export=details,
                metadata=metadata, metadata_sha256=hashes, declared_lineage=source)


def sft_checkpoint_path(train, step=None):
    """Select an explicit step or the run's advertised latest checkpoint."""
    train = Path(train).resolve()
    config_path = train / "resolved_launch_config.json"
    config = json.loads(config_path.read_text()) if config_path.is_file() else {}
    directory = config.get("checkpoint", {}).get("directory_name", "checkpoints")
    checkpoints = train / directory
    if step is None:
        marker = checkpoints / "latest_checkpointed_iteration.txt"
        if not marker.is_file():
            raise ValueError("Specify step for sft_train_run or provide its latest checkpoint marker")
        step = int(marker.read_text().strip())
    if type(step) is not int or step < 1:
        raise ValueError("SFT checkpoint step must be a positive integer")
    return (checkpoints / f"global_step_{step}" / "huggingface").resolve(), step


def validate_sft_training_export(train, *, step=None):
    """Validate the selected export, including sharded weights, without loading it."""
    model, step = sft_checkpoint_path(train, step)
    details = inspect_local_model(Path(train), model)
    return model, dict(details, source_train_run=str(Path(train).resolve()),
                       checkpoint=f"global_step_{step}", step=step)
