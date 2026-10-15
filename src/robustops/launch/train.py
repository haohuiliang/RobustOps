from __future__ import annotations

import datetime as dt
import hashlib
import json
import math
import os
import shlex
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from robustops.launch.config import (
    require_string_list as _string_list,
    LaunchConfigError,
    require_mapping,
    require_string,
    resolve_model_path,
    resolve_project_path,
    resolve_training_output_parent,
)
from robustops.launch.hardware import select_gpus


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LaunchConfigError(f"{name} must be a positive integer")
    return value


def _nonzero_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value == 0:
        raise LaunchConfigError(f"{name} must be a non-zero integer")
    return value


def _finite_float(value: Any, name: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not math.isfinite(value):
        raise LaunchConfigError(f"{name} must be a finite number")
    return float(value)


def _optimizer_betas(value: Any) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise LaunchConfigError("training.optimizer_betas must contain exactly two numbers")
    beta1 = _finite_float(value[0], "training.optimizer_betas[0]")
    beta2 = _finite_float(value[1], "training.optimizer_betas[1]")
    if not (0.0 <= beta1 < 1.0 and 0.0 <= beta2 < 1.0):
        raise LaunchConfigError("training.optimizer_betas values must be in [0, 1)")
    return beta1, beta2


def _python_path(root: Path) -> str:
    parts = [str(root / "src"), str(root / "verl")]
    inherited = os.environ.get("PYTHONPATH")
    if inherited:
        parts.append(inherited)
    return os.pathsep.join(parts)


def _git_value(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else "unavailable"


def _gpu_inventory() -> list[str]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=index,name,memory.total,compute_cap",
            "--format=csv,noheader",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip().splitlines() if result.returncode == 0 else ["unavailable"]


def _run_data_builder(
    root: Path,
    data: dict[str, Any],
    python_bin: Path,
    *,
    builder_config_key: str,
) -> None:
    builder_value = data.get(builder_config_key)
    if not isinstance(builder_value, str):
        raise LaunchConfigError(f"data.{builder_config_key} must be set when its parquet is absent")
    builder = resolve_project_path(root, builder_value)
    builder_module = data.get(
        "builder_module",
        "robustops.data.teacher_sft",
    )
    if not isinstance(builder_module, str) or not builder_module:
        raise LaunchConfigError("data.builder_module must be a non-empty string")
    env = os.environ.copy()
    env["PYTHONPATH"] = _python_path(root)
    subprocess.run(
        [
            str(python_bin),
            "-m",
            builder_module,
            "--config",
            str(builder),
        ],
        check=True,
        cwd=root,
        env=env,
    )


def _select_train_data(root: Path, data: dict[str, Any], python_bin: Path) -> Path:
    override_name = data.get("train_data_env", "ROBUSTOPS_SFT_TRAIN_DATA")
    if not isinstance(override_name, str) or not override_name:
        raise LaunchConfigError("data.train_data_env must be a non-empty string")
    override = os.environ.get(override_name)
    canonical = resolve_project_path(root, require_string(data, "train_parquet"))
    provided_value = data.get("provided_train_parquet")
    provided = resolve_project_path(root, provided_value) if isinstance(provided_value, str) else None
    if override:
        selected = Path(override).expanduser()
        if not selected.is_file():
            raise LaunchConfigError(f"{override_name} does not exist: {selected}")
        return selected
    if canonical.is_file():
        return canonical
    if provided is not None and provided.is_file():
        return provided

    _run_data_builder(
        root,
        data,
        python_bin,
        builder_config_key="builder_config",
    )
    if not canonical.is_file():
        raise LaunchConfigError(f"training data builder did not create {canonical}")
    print(f"training_data_rebuilt={canonical}")
    return canonical


def _select_validation_data(
    root: Path,
    data: dict[str, Any],
    python_bin: Path,
) -> Path | None:
    configured = data.get("validation_parquet")
    if configured is None:
        return None
    if not isinstance(configured, str) or not configured:
        raise LaunchConfigError("data.validation_parquet must be a non-empty string")
    override_name = data.get(
        "validation_data_env",
        "ROBUSTOPS_SFT_VALIDATION_DATA",
    )
    if not isinstance(override_name, str) or not override_name:
        raise LaunchConfigError("data.validation_data_env must be a non-empty string")
    override = os.environ.get(override_name)
    selected = Path(override).expanduser() if override else resolve_project_path(root, configured)
    if selected.is_file():
        return selected
    if override:
        raise LaunchConfigError(f"{override_name} does not exist: {selected}")
    _run_data_builder(
        root,
        data,
        python_bin,
        builder_config_key="validation_builder_config",
    )
    if not selected.is_file():
        raise LaunchConfigError(f"validation data builder did not create {selected}")
    print(f"validation_data_rebuilt={selected}")
    return selected


def _write_resolved_config(
    path: Path,
    *,
    recipe_path: Path,
    recipe: dict[str, Any],
    root: Path,
    run_id: str,
    model_path: Path,
    train_path: Path,
    validation_path: Path | None,
    rows: int,
    validation_rows: int,
    effective_rows: int,
    padding_rows: int,
    gpus: int,
    command: Sequence[str],
) -> None:
    training = require_mapping(recipe, "training")
    payload = {
        "schema_version": 1,
        "created_at_utc": dt.datetime.now(dt.UTC).isoformat(),
        "interface": "train",
        "pipeline": recipe["pipeline"],
        "experiment_id": recipe.get("experiment_id"),
        "launch_recipe": str(recipe_path),
        "run_id": run_id,
        "recorded_hardware": recipe.get("hardware", {}).get("recorded"),
        "runtime_hardware": recipe["runtime_hardware"],
        "gpu_count": gpus,
        "gpu_inventory": _gpu_inventory(),
        "root_dir": str(root),
        "model_path": str(model_path),
        "model_revision": require_mapping(recipe, "model").get("revision"),
        "train_parquet": str(train_path),
        "train_parquet_sha256": hashlib.sha256(train_path.read_bytes()).hexdigest(),
        "train_records": rows,
        "checkpoint": {"directory_name": "final_model", **recipe.get("checkpoint", {})},
        "validation_parquet": str(validation_path) if validation_path is not None else None,
        "validation_records": validation_rows,
        "training": {
            **training,
            "records_per_epoch": rows,
            "distributed_padding_records_per_epoch": padding_rows,
            "effective_samples_per_epoch": effective_rows,
            "dropped_records_per_epoch": 0,
        },
        "git_revision": _git_value(root, "rev-parse", "HEAD"),
        "git_dirty": bool(_git_value(root, "status", "--porcelain", "--untracked-files=normal")),
        "tensorboard_dir": str(path.parent / "tensorboard"),
        "command": shlex.join(command),
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _launch_sft(
    root: Path,
    recipe_path: Path,
    recipe: dict[str, Any],
    *,
    gpu_override: int | None,
    dry_run: bool,
    hydra_overrides: Sequence[str],
) -> int:
    hardware = require_mapping(recipe, "hardware")
    model = require_mapping(recipe, "model")
    data = require_mapping(recipe, "data")
    training = require_mapping(recipe, "training")
    checkpoint = require_mapping(recipe, "checkpoint")
    logging = recipe.get("logging", {"backends": ["console"]})
    evaluation = recipe.get("evaluation", {"frequency": -1})
    if not isinstance(logging, dict):
        raise LaunchConfigError("logging must be an object")
    if not isinstance(evaluation, dict):
        raise LaunchConfigError("evaluation must be an object")
    log_backends = _string_list(logging.get("backends", ["console"]), "logging.backends")
    unsupported_backends = sorted(set(log_backends) - {"console", "tensorboard"})
    if unsupported_backends:
        raise LaunchConfigError(f"unsupported logging backends: {unsupported_backends}")
    test_frequency = _nonzero_int(evaluation.get("frequency", -1), "evaluation.frequency")

    selected = select_gpus(gpu_override, dry_run=dry_run)
    gpus = selected.count
    recipe = dict(recipe, runtime_hardware=selected.record())
    global_batch = _positive_int(training.get("global_batch_size"), "training.global_batch_size")
    if global_batch < gpus or global_batch % gpus:
        raise LaunchConfigError(
            f"global batch {global_batch} must be at least and divisible by {gpus} GPU(s)"
        )

    python_bin = Path(os.environ.get("ROBUSTOPS_PYTHON", sys.executable))
    torchrun = python_bin.parent / "torchrun"
    model_path = resolve_model_path(root, require_string(model, "path"))
    dataset_class = resolve_project_path(root, require_string(data, "dataset_class"))
    for required in (python_bin, torchrun, model_path, dataset_class):
        if not required.exists() and not (dry_run and required in (torchrun, model_path)):
            raise LaunchConfigError(f"required training path does not exist: {required}")

    train_path = _select_train_data(root, data, python_bin)
    validation_path = _select_validation_data(root, data, python_bin)
    rows = pq.read_metadata(train_path).num_rows
    validation_rows = (
        pq.read_metadata(validation_path).num_rows if validation_path is not None else 0
    )
    expected_train_rows = data.get("expected_train_records")
    if expected_train_rows is not None:
        expected_train_rows = _positive_int(
            expected_train_rows, "data.expected_train_records"
        )
        if rows != expected_train_rows:
            raise LaunchConfigError(
                f"training data row count changed: expected {expected_train_rows}, found {rows}"
            )
    expected_validation_rows = data.get("expected_validation_records")
    if expected_validation_rows is not None:
        expected_validation_rows = _positive_int(
            expected_validation_rows, "data.expected_validation_records"
        )
        if validation_path is None or validation_rows != expected_validation_rows:
            raise LaunchConfigError(
                "validation data row count changed: "
                f"expected {expected_validation_rows}, found {validation_rows}"
            )
    pad_to_multiple = _positive_int(data.get("pad_to_multiple", 1), "data.pad_to_multiple")
    effective_rows = ((rows + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
    effective_validation_rows = (
        ((validation_rows + pad_to_multiple - 1) // pad_to_multiple) * pad_to_multiple
        if validation_path is not None
        else 0
    )
    padding_rows = effective_rows - rows
    allow_partial = bool(training.get("allow_partial_epoch", False))
    if rows <= 0:
        raise LaunchConfigError(f"training data has no rows: {train_path}")
    if validation_path is not None and validation_rows <= 0:
        raise LaunchConfigError(f"validation data has no rows: {validation_path}")
    if validation_path is None and test_frequency > 0:
        raise LaunchConfigError("positive evaluation.frequency requires validation data")
    if validation_path is not None and effective_validation_rows % global_batch:
        raise LaunchConfigError(
            f"validation data has {validation_rows} real rows and {effective_validation_rows} rows "
            f"after dataset padding, which is not divisible by global batch {global_batch}"
        )
    if effective_rows % global_batch and not allow_partial:
        raise LaunchConfigError(
            f"training data has {rows} real rows and {effective_rows} rows after dataset "
            f"padding, but global batch {global_batch} would still drop "
            f"{effective_rows % global_batch}; change data.pad_to_multiple instead of "
            "silently dropping records"
        )

    prefix = str(recipe.get("run_id_prefix", "training"))
    run_id = os.environ.get(
        "ROBUSTOPS_TRAIN_RUN_ID",
        f"{prefix}-{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}",
    )
    output_parent = resolve_training_output_parent(root, require_string(checkpoint, "output_root"))
    output_root = output_parent / run_id
    tensorboard_dir = output_root / "tensorboard"

    resume_path_value = os.environ.get("ROBUSTOPS_TRAIN_RESUME_FROM")
    if resume_path_value:
        resume_path = Path(resume_path_value).expanduser()
        if not resume_path.is_dir():
            raise LaunchConfigError(
                f"ROBUSTOPS_TRAIN_RESUME_FROM is not a checkpoint directory: {resume_path}"
            )
        resume_mode = "resume_path"
    else:
        resume_path = None
        resume_mode = str(checkpoint.get("resume_mode", "disable"))
        if resume_mode != "disable":
            raise LaunchConfigError(
                "checkpoint.resume_mode must be disable; use ROBUSTOPS_TRAIN_RESUME_FROM for an explicit resume"
            )

    checkpoint_dir_name = checkpoint.get("directory_name", "final_model")
    if not isinstance(checkpoint_dir_name, str) or not checkpoint_dir_name:
        raise LaunchConfigError("checkpoint.directory_name must be a non-empty string")
    total_training_steps = training.get("total_training_steps")
    if total_training_steps is not None:
        total_training_steps = _positive_int(
            total_training_steps,
            "training.total_training_steps",
        )
    save_frequency = checkpoint.get("save_frequency", "after_each_epoch")
    if isinstance(save_frequency, int) and not isinstance(save_frequency, bool):
        _nonzero_int(save_frequency, "checkpoint.save_frequency")
    elif save_frequency != "after_each_epoch":
        raise LaunchConfigError(
            "checkpoint.save_frequency must be a non-zero integer or after_each_epoch"
        )
    max_to_keep = checkpoint.get("max_to_keep")
    if max_to_keep is not None:
        max_to_keep = _positive_int(max_to_keep, "checkpoint.max_to_keep")
    save_contents = _string_list(
        checkpoint.get("save_contents", ["model"]),
        "checkpoint.save_contents",
    )
    unsupported_contents = sorted(set(save_contents) - {"model", "optimizer", "extra", "hf_model"})
    if unsupported_contents:
        raise LaunchConfigError(f"unsupported checkpoint contents: {unsupported_contents}")

    trainer_module = training.get("trainer_module", "verl.trainer.sft_trainer")
    supported_trainer_modules = {
        "verl.trainer.sft_trainer",
        "robustops.training.sft_base_trainer",
        "robustops.training.decision_sft_trainer",
    }
    if trainer_module not in supported_trainer_modules:
        raise LaunchConfigError(f"unsupported SFT trainer module: {trainer_module!r}")
    sampler_shuffle = data.get("sampler_shuffle", True)
    if not isinstance(sampler_shuffle, bool):
        raise LaunchConfigError("data.sampler_shuffle must be a boolean")
    optimizer_betas = _optimizer_betas(training.get("optimizer_betas", [0.9, 0.999]))
    clip_grad = _finite_float(training.get("clip_grad", 1.0), "training.clip_grad")
    if clip_grad <= 0.0:
        raise LaunchConfigError("training.clip_grad must be positive")
    min_lr_ratio = _finite_float(
        training.get("min_lr_ratio", 0.0), "training.min_lr_ratio"
    )
    if not 0.0 <= min_lr_ratio <= 1.0:
        raise LaunchConfigError("training.min_lr_ratio must be in [0, 1]")
    pytorch_cuda_alloc_conf = training.get("pytorch_cuda_alloc_conf")
    if pytorch_cuda_alloc_conf is not None and (
        not isinstance(pytorch_cuda_alloc_conf, str) or not pytorch_cuda_alloc_conf
    ):
        raise LaunchConfigError("training.pytorch_cuda_alloc_conf must be a non-empty string")

    overrides = [
        f"hydra.run.dir={output_root}/hydra",
        f"data.train_files={train_path}",
        f"data.val_files={validation_path}" if validation_path is not None else "data.val_files=null",
        f"data.messages_key={data.get('messages_key', 'messages_json')}",
        f"data.tools_key={data.get('tools_key', 'tools_json')}",
        f"data.enable_thinking_default={data.get('enable_thinking_default', False)}",
        f"data.custom_cls.path={dataset_class}",
        f"data.custom_cls.name={data.get('dataset_class_name', 'FinalAssistantOnlySFTDataset')}",
        f"data.train_batch_size={global_batch}",
        f"data.micro_batch_size_per_gpu={training['micro_batch_size_per_gpu']}",
        f"data.max_token_len_per_gpu={training['max_token_len_per_gpu']}",
        f"data.use_dynamic_bsz={training.get('use_dynamic_batch_size', True)}",
        f"data.max_length={training['max_length']}",
        f"data.truncation={training.get('truncation', 'error')}",
        f"data.pad_mode={training.get('pad_mode', 'no_padding')}",
        f"data.num_workers={training.get('num_workers', 4)}",
        f"+data.pad_to_multiple={pad_to_multiple}",
        f"+data.sampler_shuffle={str(sampler_shuffle).lower()}",
        "data.ignore_input_ids_mismatch=False",
        f"model.path={model_path}",
        f"+model.override_config.attn_implementation={training.get('attention', 'flash_attention_2')}",
        f"model.enable_gradient_checkpointing={training.get('gradient_checkpointing', True)}",
        f"model.use_remove_padding={training.get('use_remove_padding', False)}",
        "model.lora_rank=0",
        f"engine={training.get('engine', 'fsdp')}",
        f"engine.dtype={training.get('dtype', 'bfloat16')}",
        f"engine.use_torch_compile={training.get('use_torch_compile', False)}",
        f"optim.lr={training['learning_rate']}",
        f"optim.lr_warmup_steps_ratio={training['lr_warmup_steps_ratio']}",
        f"optim.lr_scheduler_type={training['lr_scheduler_type']}",
        f"optim.weight_decay={training['weight_decay']}",
        f"optim.betas=[{optimizer_betas[0]},{optimizer_betas[1]}]",
        f"optim.clip_grad={clip_grad}",
        f"optim.min_lr_ratio={min_lr_ratio}",
        f"trainer.default_local_dir={output_root}/{checkpoint_dir_name}",
        f"trainer.project_name={recipe.get('project_name', 'robustops')}",
        f"trainer.experiment_name={run_id}",
        f"trainer.total_epochs={training['epochs']}",
        f"trainer.logger=[{','.join(log_backends)}]",
        f"trainer.save_freq={save_frequency}",
        f"trainer.test_freq={test_frequency}",
        f"trainer.max_ckpt_to_keep={json.dumps(max_to_keep)}",
        f"trainer.resume_mode={resume_mode}",
        f"trainer.seed={training['seed']}",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={gpus}",
        f"checkpoint.save_contents={json.dumps(save_contents, separators=(',', ':'))}",
    ]
    if resume_path is not None:
        overrides.append(f"trainer.resume_from_path={resume_path}")
    if total_training_steps is not None:
        overrides.append(f"trainer.total_training_steps={total_training_steps}")
    overrides.extend(hydra_overrides)
    command = [
        str(torchrun),
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={gpus}",
        "-m",
        trainer_module,
        *overrides,
    ]

    print("pipeline=sft")
    print(f"runtime_hardware={json.dumps(selected.record())}")
    print(f"gpus={gpus}")
    print(f"model_path={model_path}")
    print(f"training_data={train_path}")
    print(f"training_rows={rows}")
    print(f"validation_data={validation_path}")
    print(f"validation_rows={validation_rows}")
    print(f"distributed_padding_rows={padding_rows}")
    print(f"effective_training_samples={effective_rows}")
    print(f"epochs={training['epochs']}")
    print(f"total_training_steps={total_training_steps or 'full_epoch'}")
    print(f"validation_frequency={test_frequency}")
    print(f"logging_backends={','.join(log_backends)}")
    print(f"pytorch_cuda_alloc_conf={pytorch_cuda_alloc_conf}")
    print(f"tensorboard_dir={tensorboard_dir}")
    print(f"resume_from={resume_path}")
    print(f"output_root={output_root}")
    if dry_run:
        print(f"command={shlex.join(command)}")
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    resolved_path = output_root / "resolved_launch_config.json"
    _write_resolved_config(
        resolved_path,
        recipe_path=recipe_path,
        recipe=recipe,
        root=root,
        run_id=run_id,
        model_path=model_path,
        train_path=train_path,
        validation_path=validation_path,
        rows=rows,
        validation_rows=validation_rows,
        effective_rows=effective_rows,
        padding_rows=padding_rows,
        gpus=gpus,
        command=command,
    )
    print(f"resolved_config={resolved_path}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = selected.cuda_visible_devices
    env["PYTHONPATH"] = _python_path(root)
    env["TENSORBOARD_DIR"] = str(tensorboard_dir)
    if pytorch_cuda_alloc_conf is not None:
        env["PYTORCH_CUDA_ALLOC_CONF"] = pytorch_cuda_alloc_conf
    with (output_root / "train.log").open("wb") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while chunk := process.stdout.read1(65536):
            log_handle.write(chunk)
            log_handle.flush()
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
        return process.wait()


def _launch_rl(
    root: Path,
    recipe: dict[str, Any],
    *,
    gpu_override: int | None,
    dry_run: bool,
    hydra_overrides: Sequence[str],
) -> int:
    launcher = require_mapping(recipe, "launcher")
    module = require_string(launcher, "module")
    if module == "robustops.training.rl_launcher":
        if hydra_overrides:
            raise LaunchConfigError("Edit the RL recipe instead of supplying unpaired backend overrides")
        selected = select_gpus(gpu_override, dry_run=dry_run)
        recipe = dict(recipe, runtime_hardware=selected.record())
        python_bin = os.environ.get("ROBUSTOPS_PYTHON", sys.executable)
        command = [python_bin, "-m", module, "--root", str(root),
                   "--recipe-json", json.dumps(recipe)]
        if dry_run:
            command.append("--dry-run")
        env = dict(os.environ, PYTHONPATH=_python_path(root))
        if not dry_run:
            env["CUDA_VISIBLE_DEVICES"] = selected.cuda_visible_devices
        return subprocess.run(command, cwd=root, env=env, check=False).returncode
    if module != "verl.trainer.main_ppo":
        raise LaunchConfigError(f"unsupported RL launcher module: {module}")
    base_overrides = launcher.get("hydra_overrides")
    if not isinstance(base_overrides, list) or not all(isinstance(item, str) for item in base_overrides):
        raise LaunchConfigError("RL launcher.hydra_overrides must be a list of strings")
    selected = select_gpus(gpu_override, dry_run=dry_run)
    gpus = selected.count
    python_bin = Path(os.environ.get("ROBUSTOPS_PYTHON", sys.executable))
    command = [str(python_bin), "-m", module, *base_overrides, *hydra_overrides]
    print("pipeline=rl")
    print(f"gpus={gpus}")
    print(f"command={shlex.join(command)}")
    if dry_run:
        return 0
    env = os.environ.copy()
    env["PYTHONPATH"] = _python_path(root)
    env["CUDA_VISIBLE_DEVICES"] = selected.cuda_visible_devices
    env["ROBUSTOPS_GPU_COUNT"] = str(gpus)
    return subprocess.run(command, cwd=root, env=env, check=False).returncode


def launch(
    root: Path,
    recipe_path: Path,
    recipe: dict[str, Any],
    *,
    gpu_override: int | None,
    dry_run: bool,
    hydra_overrides: Sequence[str],
) -> int:
    if recipe.get("interface") != "train":
        raise LaunchConfigError(f"train interface received recipe for {recipe.get('interface')!r}")
    pipeline = recipe.get("pipeline")
    if pipeline == "sft":
        return _launch_sft(
            root,
            recipe_path,
            recipe,
            gpu_override=gpu_override,
            dry_run=dry_run,
            hydra_overrides=hydra_overrides,
        )
    if pipeline == "rl":
        return _launch_rl(
            root,
            recipe,
            gpu_override=gpu_override,
            dry_run=dry_run,
            hydra_overrides=hydra_overrides,
        )
    raise LaunchConfigError(f"unsupported training pipeline: {pipeline!r}")
