"""Reuse task plans and restore committed training state for RL continuations."""

from copy import deepcopy
import json
from pathlib import Path
import re
import shutil
import tempfile
import time
from robustops.training.recovery_rl_core import write_json


def reuse_task_plan(source, destination, config, *, include_initial_states=True):
    """Copy the frozen dataset, including the original complete-window reserves."""
    from robustops.training.recovery_rl_data import require_current_task_contract

    source, destination = Path(source).resolve(), Path(destination)
    require_current_task_contract(source / "data")
    from robustops.training.rl_data_config import verify_saved_inputs
    verify_saved_inputs(source / "data")
    summary = json.loads((source / "data/data_summary.json").read_text())
    if summary.get("data_sources") is not None and summary["data_sources"] != config.get("data_sources"):
        raise ValueError("task-plan source uses different RL data inputs")
    original = json.loads((source / "recipe.json").read_text())["training"]
    if (
        original.get("advantage_estimator") != "grpo"
        or original["seed"] != config["seed"]
    ):
        raise ValueError("task-plan source must be a GRPO run with the same seed")
    schedule = json.loads((source / "data/task_schedule.json").read_text())
    if schedule["seed"] != config["seed"] or len(schedule["batches"]) < config["steps"]:
        raise ValueError("source task plan is too short or uses a different seed")
    if (original["batch_size"], original["group_size"]) != (32, 4):
        raise ValueError("source must contain 32 G4 groups per step")
    # Copy, do not truncate/regenerate: reserves still exclude all original 30 steps.
    shutil.copytree(source / "data", destination / "data")
    initial = source / "plans/initial_states"
    if include_initial_states and initial.exists():
        shutil.copytree(initial, destination / "plans/initial_states")
    result = dict(
        source=str(source),
        reused_plan_steps=len(schedule["batches"]),
        training_steps=config["steps"],
        task_plan_equal=True,
        faults_required_equal=False,
        initialization="fresh SFT; no weights or replay queue reused",
    )
    write_json(destination / "task_plan_source.json", result)
    return result


def preserve_uncommitted(suite, step, stamp):
    """Move abandoned step evidence aside before replaying from a saved state."""
    root = Path(suite) / "trainer"
    archive = root / "continuations" / stamp / "uncommitted"
    moved = []
    for relative in (
        "episodes/train",
        "episodes/donors",
        "anchors",
        "alignment",
        "diagnostics",
        "metrics",
        "credit",
        "updates",
        "val_loss",
        "recovery_states",
        "model_exports",
        "emergency_checkpoints",
        "member_retries",
        "final/steps",
    ):
        for item in (root / relative).glob("*"):
            match = re.fullmatch(
                r"(?:step_)?(\d+)(?:_schedule|_failed|_replacement_exhausted)?(?:\.jsonl|\.json|\.log)?",
                item.name,
            )
            if match and int(match[1]) > step:
                target = archive / relative / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(item), str(target))
                moved.append(str(target))
    for name in (
        "failed.json",
        "process_cleanup.json",
        "command.json",
        "stop_request.json",
    ):
        item = root / name
        if item.exists():
            archive.mkdir(parents=True, exist_ok=True)
            shutil.move(str(item), str(archive / name))
    # Preserve unpublished generations from failed saves or model exports.
    for directory in (root / "checkpoints", root / "models"):
        for item in directory.glob("global_step_*"):
            match = re.fullmatch(r"global_step_(\d+)(?:\.partial)?", item.name)
            if match and int(match[1]) > step:
                target = directory / "abandoned" / stamp / item.name
                target.parent.mkdir(parents=True, exist_ok=True)
                item.rename(target)
                moved.append(str(target))
    write_json(archive / "preserved.json", dict(resume_step=step, paths=moved))
    # TensorBoard's restart marker hides abandoned >= step+1 scalar events;
    # original event files remain available for failure analysis.
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto import event_pb2

    for directory in (root / "tensorboard", root / "focus", root / "final/tensorboard"):
        if directory.exists():
            writer = EventFileWriter(str(directory))
            writer.add_event(
                event_pb2.Event(
                    wall_time=time.time(),
                    step=step + 1,
                    session_log=event_pb2.SessionLog(status=event_pb2.SessionLog.START),
                )
            )
            writer.close()


def fork_data(saved, destination):
    """Private data/seed copies; inherited replay snapshots stay read-only sources.

    Model, optimizer, scheduler and RNG load directly from the source full
    checkpoint, with del_local_ckpt_after_load=False. Never copy old run logs.
    """
    destination = Path(destination)
    shutil.copytree(saved["data_directory"], destination / "data")
    shutil.copytree(saved["shared_plans"], destination / "plans")


def extend_data(root, directory, config):
    """Extend only the unconsumed future of a sequential task-paired dataset."""
    from robustops.training.recovery_rl_data import prepare

    directory = Path(directory)
    saved = json.loads((directory / "task_schedule.json").read_text())
    if len(saved["batches"]) >= config["steps"]:
        return
    if config.get("task_sampling") == "queue":
        import pyarrow as pa
        import pyarrow.parquet as pq
        from robustops.training.recovery_task_queue import carrier_batches, load_pool

        extra = carrier_batches(
            load_pool(directory), len(saved["batches"]) + 1, config["steps"]
        )
        table = pq.read_table(directory / "train.parquet")
        rows = table.to_pylist()[: len(saved["batches"]) * 32]
        if len(rows) != len(saved["batches"]) * 32:
            raise ValueError(
                "queue carrier dataset is shorter than its committed schedule"
            )
        for batch in extra:
            for row in batch["groups"]:
                item = deepcopy(rows[0])
                item["extra_info"] = dict(index=row["visit"], recovery=row)
                rows.append(item)
        # Preserve the consumed dataloader prefix; future rows only carry slots.
        pq.write_table(
            pa.Table.from_pylist(rows, schema=table.schema),
            directory / "train.parquet.writing",
        )
        (directory / "train.parquet.writing").replace(directory / "train.parquet")
        saved["batches"].extend(extra)
        write_json(directory / "task_schedule.json", saved)
        summary = json.loads((directory / "data_summary.json").read_text())
        summary["train_rows"] = len(rows)
        write_json(directory / "data_summary.json", summary)
        return
    with tempfile.TemporaryDirectory(
        prefix="task-schedule-", dir=directory.parent
    ) as tmp:
        prepare(root, tmp, config)
        new = json.loads((Path(tmp) / "task_schedule.json").read_text())
        if (
            new["seed"] != saved["seed"]
            or new["batches"][: len(saved["batches"])] != saved["batches"]
        ):
            raise ValueError("task schedule prefix changed; refusing resume")
        old_reserves = json.loads((directory / "fault_reserves.json").read_text())
        new_reserves = json.loads((Path(tmp) / "fault_reserves.json").read_text())
        if any(new_reserves.get(k) != v for k, v in old_reserves.items()):
            raise ValueError("fault reserve prefix changed; refusing resume")
        # A crash here can be retried: publication is offline, before the trainer.
        for name in (
            "train.parquet",
            "validation.parquet",
            "data_summary.json",
            "task_schedule.json",
            "fault_reserves.json",
        ):
            (Path(tmp) / name).replace(directory / name)


def latest_checkpoint(suite, step=None):
    suite = Path(suite).resolve()
    directory = suite / "trainer" / "checkpoints"
    if step is None:
        marker = directory / "latest_checkpointed_iteration.txt"
        if not marker.is_file():
            raise ValueError(f"no resumable checkpoint found in {suite}")
        step = int(marker.read_text())
    selected = directory / f"global_step_{step}"
    if not selected.is_dir():
        raise ValueError(f"no resumable checkpoint found at {selected}")
    from robustops.training.recovery_rl_storage import validate_resumable_checkpoint

    validate_resumable_checkpoint(selected)
    state = json.loads((selected / "recovery_state.json").read_text())
    if state.get("format_version") != 2 or state.get("schedule") != "independent_16_16":
        raise ValueError("checkpoint uses an incompatible training-state schema")
    if Path(state["run_directory"]).resolve() != suite:
        raise ValueError("checkpoint belongs to a different run")
    return selected, state


def resolve_continuation(root, recipe, gpu_count):
    """Resolve a fresh/resumed/forked recipe without mutating run artifacts."""
    from robustops.launch.config import resolve_project_path
    from robustops.training.rl_config import (
        RESUME_FIELDS,
        FORK_FIXED_FIELDS,
        validate_training_config,
        RUNTIME_DEFAULTS,
    )
    from robustops.training.recovery_rl_storage import validate_resumable_checkpoint
    from robustops.training.rl_data_config import resolve_data_sources, verify_saved_inputs

    requested = dict(RUNTIME_DEFAULTS, **deepcopy(recipe["training"]))
    if requested.get("verifier_graph_dir"):
        requested["verifier_graph_dir"] = str(
            resolve_project_path(root, requested["verifier_graph_dir"]).resolve()
        )
    requested["data_sources"] = resolve_data_sources(root, recipe.get("data", requested.get("data_sources")))
    requested["gpu_count"] = gpu_count
    validate_training_config(requested)
    resume = recipe.get("resume")
    if "resume_checkpoint" in requested:
        raise ValueError("use the top-level resume configuration to load a checkpoint")
    if resume is None:
        requested.pop("resume_history_output", None)
        requested["metrics_start_step"] = 0
        return requested, None, None
    if not isinstance(resume, dict) or set(resume) - {
        "run",
        "checkpoint",
        "step",
        "fork_output",
    }:
        raise ValueError(
            "resume accepts run or checkpoint, optional step, and fork_output"
        )
    if bool(resume.get("run")) == bool(resume.get("checkpoint")):
        raise ValueError("resume requires exactly one of run or checkpoint")
    if resume.get("step") is not None and (
        type(resume["step"]) is not int or resume["step"] < 1
    ):
        raise ValueError("resume.step must be a positive integer")
    if resume.get("step") is not None and not resume.get("fork_output"):
        raise ValueError("selecting a checkpoint step requires a separate fork_output")
    if resume.get("checkpoint"):
        if resume.get("step") is not None:
            raise ValueError("checkpoint path and step cannot be combined")
        checkpoint = resolve_project_path(root, resume["checkpoint"]).resolve()
        validate_resumable_checkpoint(checkpoint, expected_world_size=gpu_count)
        saved = json.loads((checkpoint / "recovery_state.json").read_text())
        if (
            saved.get("format_version") != 2
            or saved.get("schedule") != "independent_16_16"
        ):
            raise ValueError("checkpoint uses an incompatible training-state schema")
        source = Path(saved["run_directory"]).resolve()
        if not resume.get("fork_output"):
            latest, _ = latest_checkpoint(source)
            if latest.resolve() != checkpoint:
                raise ValueError(
                    "resuming an earlier checkpoint requires a separate fork_output"
                )
    else:
        source = resolve_project_path(root, resume["run"]).resolve()
        checkpoint, saved = latest_checkpoint(source, resume.get("step"))
        validate_resumable_checkpoint(checkpoint, expected_world_size=gpu_count)
    previous = dict(RUNTIME_DEFAULTS, **deepcopy(saved["training_config"]))
    # Legacy checkpoint metadata remains resumable without permanent exceptions.
    previous.pop("checkpoint_protected_steps", None)
    if previous.get("verifier_graph_dir"):
        previous["verifier_graph_dir"] = str(
            resolve_project_path(root, previous["verifier_graph_dir"]).resolve()
        )
    previous["data_sources"] = resolve_data_sources(root, previous.get("data_sources"))
    validate_training_config(previous)
    changed = {
        k
        for k, v in requested.items()
        if k not in RESUME_FIELDS and v != previous.get(k)
    }
    if resume.get("fork_output"):
        fixed = changed & FORK_FIXED_FIELDS
        if fixed:
            raise ValueError(f"full checkpoint forks must preserve {sorted(fixed)}")
    elif changed:
        raise ValueError(f"changing {sorted(changed)} requires a fork_output")
    config = dict(previous, **requested, resume_checkpoint=str(checkpoint))
    if config["steps"] <= saved["step"]:
        raise ValueError("target steps must exceed the saved cumulative step")
    if saved["next_visit"] != saved["step"] * config["batch_size"]:
        raise ValueError(
            "saved sampler cursor does not match checkpoint step and batch size"
        )
    if Path(saved["data_directory"]).resolve() != source / "data":
        raise ValueError("saved data directory does not match the source run")
    if Path(saved["shared_plans"]).resolve() != source / "plans":
        raise ValueError("saved plan directory does not match the source run")
    if (Path(saved["data_directory"]) / "data_summary.json").is_file():
        verify_saved_inputs(saved["data_directory"])
    return config, saved, source
