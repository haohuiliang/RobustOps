"""Launch one configured GRPO run, resume it, or fork a full checkpoint."""

import argparse
import asyncio
from contextlib import suppress
from copy import deepcopy
import datetime
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

from robustops.launch.hardware import visible_gpus, select_gpus
from robustops.training.recovery_rl_core import write_json
from robustops.training.recovery_rl_storage import (
    validate_resumable_checkpoint,
    validate_hf_export,
)
from robustops.training.rl_config import validate_training_config, RUNTIME_DEFAULTS


def retained_checkpoint_steps(config):
    if not config.get("save_checkpoint", True):
        return []
    steps = config["steps"]
    frequency = config.get("checkpoint_save_frequency", steps)
    keep = config.get("checkpoint_max_to_keep")
    if (
        type(frequency) is not int
        or frequency <= 0
        or (keep is not None and (type(keep) is not int or keep <= 0))
    ):
        raise ValueError("checkpoint frequency must be positive; retention must be null or positive")
    exports = {s for s in config.get("hf_export_steps", []) if s <= steps}
    latest = sorted(
        set(range(frequency, steps + 1, frequency)) | {steps} | exports
    )
    return latest if keep is None else latest[-keep:]


def checkpoint_directory(suite, config):
    return Path(suite) / "trainer" / "checkpoints"


def hardware_gate(expected_count=None, *, min_host_memory_gib=0):
    from robustops.launch.resources import check_host_memory
    memory = check_host_memory(min_host_memory_gib)
    devices = visible_gpus()
    if not devices:
        raise RuntimeError(
            "RL requires at least one visible GPU; use --gpus N --dry-run offline."
        )
    if expected_count is not None and len(devices) != expected_count:
        raise RuntimeError(
            f"RL selected {expected_count} GPUs, but {len(devices)} are visible in the child process"
        )
    used = subprocess.check_output(
        [
            "nvidia-smi",
            "-i",
            ",".join(gpu.uuid for gpu in devices),
            "--query-compute-apps=pid",
            "--format=csv,noheader",
        ],
        text=True,
    )
    if used.strip():
        raise RuntimeError(
            "GPUs already have compute processes; refusing competing training"
        )
    return dict(
        gpu_count=len(devices),
        devices=[
            dict(uuid=gpu.uuid, name=gpu.name, memory_mib=gpu.memory_mib)
            for gpu in devices
        ],
        host_memory_bytes=memory,
    )


def validate_gpu_count(config):
    gpu_count = config.get("gpu_count", 1)
    batch = config["batch_size"]
    if type(gpu_count) is not int or gpu_count <= 0 or batch % gpu_count:
        raise ValueError(
            f"RL group batch {batch} must be divisible by the selected GPU count {gpu_count}; use --gpus N"
        )
    return gpu_count


def hydra_command(root, suite, model, config, urls):
    root, suite = Path(root), Path(suite)
    from robustops.training.rl_data_config import resolve_data_sources

    config = dict(RUNTIME_DEFAULTS, **config)
    data_sources = resolve_data_sources(root, config.get("data_sources"))
    output = suite / "trainer"
    validate_training_config(config)
    steps = config["steps"]
    batch = config["batch_size"]
    gpu_count = validate_gpu_count(config)
    values = {
        "hydra.run.dir": str(output / "hydra" / "${now:%Y-%m-%d_%H-%M-%S}"),
        "trainer.use_v1": False,
        "transfer_queue.enable": False,
        "algorithm.adv_estimator": "grpo",
        "algorithm.norm_adv_by_std_in_grpo": True,
        "algorithm.use_kl_in_reward": False,
        "data.train_files": str(suite / "data/train.parquet"),
        "data.val_files": str(suite / "data/validation.parquet"),
        "data.train_batch_size": batch,
        "data.val_batch_size": 16,
        "data.max_prompt_length": config["max_context_tokens"],
        "data.max_response_length": config["max_context_tokens"],
        "data.shuffle": False,
        "data.validation_shuffle": False,
        "data.filter_overlong_prompts": False,
        "data.dataloader_num_workers": 0,
        "actor_rollout_ref.model.path": str(model),
        "actor_rollout_ref.model.use_remove_padding": True,
        "actor_rollout_ref.model.enable_gradient_checkpointing": True,
        "+actor_rollout_ref.model.override_config.attn_implementation": "flash_attention_2",
        "actor_rollout_ref.actor.strategy": "fsdp",
        "actor_rollout_ref.actor.optim.lr": config["learning_rate"],
        "actor_rollout_ref.actor.optim.lr_warmup_steps": 0,
        "actor_rollout_ref.actor.optim.lr_scheduler_type": "constant",
        "actor_rollout_ref.actor.ppo_mini_batch_size": batch,
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu": 1,
        "actor_rollout_ref.actor.use_dynamic_bsz": True,
        "actor_rollout_ref.actor.ppo_max_token_len_per_gpu": config.get(
            "actor_token_batch_per_gpu", config["max_context_tokens"]
        ),
        "actor_rollout_ref.actor.ppo_epochs": 1,
        "actor_rollout_ref.actor.loss_agg_mode": "token-mean",
        "actor_rollout_ref.actor.clip_ratio": 0.2,
        "actor_rollout_ref.actor.use_kl_loss": True,
        "actor_rollout_ref.actor.kl_loss_coef": config.get("kl_loss_coef", 0.001),
        "actor_rollout_ref.actor.entropy_coeff": 0.0,
        "actor_rollout_ref.actor.entropy_from_logits_with_chunking": True,
        "actor_rollout_ref.actor.entropy_checkpointing": True,
        "actor_rollout_ref.actor.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.actor.fsdp_config.dtype": "bfloat16",
        "actor_rollout_ref.actor.fsdp_config.param_offload": True,
        "actor_rollout_ref.actor.fsdp_config.optimizer_offload": True,
        "actor_rollout_ref.actor.checkpoint.save_contents": config.get(
            "checkpoint_save_contents", ["model", "optimizer", "extra"]
        ),
        "actor_rollout_ref.ref.fsdp_config.param_offload": True,
        "actor_rollout_ref.ref.fsdp_config.use_torch_compile": False,
        "actor_rollout_ref.ref.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.ref.log_prob_max_token_len_per_gpu": config["max_context_tokens"],
        "actor_rollout_ref.rollout.name": "vllm",
        "actor_rollout_ref.rollout.tensor_model_parallel_size": 1,
        "actor_rollout_ref.rollout.gpu_memory_utilization": config["rollout_gpu_memory_utilization"],
        "actor_rollout_ref.rollout.enforce_eager": True,
        "actor_rollout_ref.rollout.free_cache_engine": True,
        "actor_rollout_ref.rollout.max_model_len": config["max_context_tokens"],
        "actor_rollout_ref.rollout.max_num_seqs": config.get(
            "rollout_concurrency_per_worker", 4
        ),
        "actor_rollout_ref.rollout.max_num_batched_tokens": config["rollout_max_num_batched_tokens"],
        "actor_rollout_ref.rollout.n": config["group_size"],
        "actor_rollout_ref.rollout.temperature": config.get("rollout_temperature", 0.6),
        "actor_rollout_ref.rollout.top_p": 1.0,
        "actor_rollout_ref.rollout.top_k": -1,
        "actor_rollout_ref.rollout.calculate_log_probs": True,
        "actor_rollout_ref.rollout.log_prob_use_dynamic_bsz": True,
        "actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu": config["max_context_tokens"],
        "actor_rollout_ref.rollout.val_kwargs.n": 1,
        "actor_rollout_ref.rollout.val_kwargs.temperature": 0.0,
        "actor_rollout_ref.rollout.val_kwargs.do_sample": False,
        "actor_rollout_ref.rollout.agent.num_workers": gpu_count,
        "actor_rollout_ref.rollout.agent.default_agent_loop": "recovery_agent",
        "actor_rollout_ref.rollout.agent.agent_loop_config_path": str(
            root / "recipes/support/rl_agent.yaml"
        ),
        "+actor_rollout_ref.rollout.agent.agent_loop_manager_class": "robustops.training.rl_manager.RecoveryManager",
        "actor_rollout_ref.rollout.multi_turn.enable": True,
        "reward.num_workers": 1,
        "reward.reward_model.enable": False,
        "trainer.nnodes": 1,
        "trainer.n_gpus_per_node": gpu_count,
        "trainer.total_epochs": config.get("epochs", 1),
        "trainer.total_training_steps": steps,
        "trainer.project_name": "robustops-recovery-rl",
        "trainer.experiment_name": suite.name,
        "trainer.default_local_dir": str(checkpoint_directory(suite, config)),
        "trainer.logger": ["console", "tensorboard"],
        "trainer.val_before_train": False,
        "trainer.test_freq": -1,
        "trainer.save_freq": config.get("checkpoint_save_frequency", steps)
        if config.get("save_checkpoint", True)
        else -1,
        "trainer.max_actor_ckpt_to_keep": config.get("checkpoint_max_to_keep"),
        "trainer.resume_mode": "resume_path"
        if config.get("resume_checkpoint")
        else "disable",
        "+ray_kwargs.ray_init.address": "local",
        "+recovery": dict(
            root=str(root),
            output=str(output),
            shared_plans=str(suite / "plans"),
            seed=config["seed"],
            advantage_estimator="grpo",
            mcp_urls=urls,
            concurrency_per_worker=config.get("rollout_concurrency_per_worker", 4),
            max_turns=config["max_turns"],
            max_generation_tokens=config["max_generation_tokens"],
            max_context_tokens=config["max_context_tokens"],
            episode_timeout_seconds=config["episode_timeout_seconds"],
        ),
    }
    if config["ray_num_cpus"] is not None:
        values["ray_kwargs.ray_init.num_cpus"] = config["ray_num_cpus"]
    if config["ray_object_store_memory_gib"] is not None:
        values["+ray_kwargs.ray_init.object_store_memory"] = int(config["ray_object_store_memory_gib"] * 1024**3)
    values["+recovery"].update(
        joint_rollout=True,
        alignment_monitoring=True,
        sampler_json=json.dumps(
            dict(
                temperature=config.get("rollout_temperature", 1.0),
                top_p=1.0,
                top_k=-1,
                repetition_penalty=1.0,
                max_generation_tokens=config["max_generation_tokens"],
                stop=["</tool_call>"],
                include_stop_str_in_output=True,
            )
        ),
        reward_weights=config["reward_weights"],
        local_bad_eta=config["local_bad_eta"],
        post_update_probe=True,
        focus_logdir=str(root / "runs/tensorboard"),
        q_temporal_credit=config.get("q_temporal_credit", False),
        reward_normalization=config["reward_normalization"],
        task_sampling=config.get("task_sampling", "queue"),
        local_read_repeat_from=config.get("local_read_repeat_from", 0),
        terminal_stop_audit=config.get("terminal_stop_audit", False),
        terminal_stop_penalty=config.get("terminal_stop_penalty", False),
        terminal_stop_advantage_cap=config.get("terminal_stop_advantage_cap"),
        verified_reference_repeats=config.get("verified_reference_repeats", False),
        exclude_all_success_groups=config.get("exclude_all_success_groups", False),
        verifier_graph_dir=(
            str((root / config["verifier_graph_dir"]).resolve())
            if config.get("verifier_graph_dir")
            else None
        ),
        resume_history_output=config.get("resume_history_output", str(output)),
        focus_run_label=config.get("focus_run_label", suite.name),
        metrics_start_step=config.get("metrics_start_step", 0),
        sft_val_every_step=config.get("sft_val_every_step", True),
        sft_val_file=data_sources["sft_validation"],
        fault_point_policy=config["fault_point_policy"],
        fault_strategy=config.get(
            "fault_strategy", "recipes/enterpriseops/faults/strategy.json"
        ),
        zero_pg_replay_delay=config.get("zero_pg_replay_delay", 0),
        zero_pg_replay_max_zero_groups=config.get("zero_pg_replay_max_zero_groups", 5),
        task_schedule_file=str(suite / "data/task_schedule.json"),
    )
    if config.get("resume_checkpoint"):
        values["trainer.resume_from_path"] = config["resume_checkpoint"]
        values["trainer.del_local_ckpt_after_load"] = False
        values["+recovery"]["resume_checkpoint"] = config["resume_checkpoint"]
    if config.get("save_checkpoint", True):
        values["+recovery"]["checkpoint_state_json"] = json.dumps(
            dict(
                directory=str(checkpoint_directory(suite, config)),
                frequency=config.get("checkpoint_save_frequency", steps),
                final_step=steps,
                max_to_keep=config.get("checkpoint_max_to_keep"),
                hf_export_steps=config.get("hf_export_steps", []),
                metadata_json=json.dumps(
                    dict(
                        format_version=2,
                        run_directory=str(suite),
                        data_directory=str(suite / "data"),
                        shared_plans=str(suite / "plans"),
                        training_config={
                            k: v for k, v in config.items() if k != "resume_checkpoint"
                        },
                    )
                ),
            )
        )

    # Hydra mappings require bare keys at every nesting level, unlike JSON.
    # String-valued JSON (sampler/checkpoint metadata) must stay quoted strings.
    def hydra_value(value):
        if isinstance(value, dict):
            return (
                "{" + ",".join(f"{k}:{hydra_value(v)}" for k, v in value.items()) + "}"
            )
        if isinstance(value, list):
            return "[" + ",".join(hydra_value(v) for v in value) + "]"
        return json.dumps(value, separators=(",", ":"))

    command = [sys.executable, "-m", "verl.trainer.main_ppo"]
    for key, value in values.items():
        command.append(f"{key}={hydra_value(value)}")
    return command


async def run_child(command, output, config, env):
    import psutil

    output.mkdir(parents=True, exist_ok=True)
    tracked = {}
    started = time.monotonic()
    last_output = started
    with (output / "train.log").open("ab") as log:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env=env,
            start_new_session=True,
        )

        async def copy_output(*, check_metrics=True):
            nonlocal last_output
            tail = b""
            while chunk := await process.stdout.read(65536):
                log.write(chunk)
                log.flush()
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
                text = tail + chunk
                if check_metrics and re.search(
                    rb"actor/(?:grad_norm|pg_loss|kl_loss|ppo_kl)\s*:\s*(?:np\.float\d+\()?\s*(?:nan|[+-]?inf)\b",
                    text,
                    re.I,
                ):
                    raise RuntimeError(
                        "Non-finite actor training metric; stopping before further updates"
                    )
                if any(
                    marker in text
                    for marker in (
                        b"RECOVERY_TURN",
                        b"RECOVERY_BATCH",
                        b"RECOVERY_SEED_READY",
                        b"RECOVERY_HF_EXPORT_",
                        b"Training Progress",
                        b"Saving checkpoint",
                        b"Writing model shards",
                    )
                ):
                    last_output = time.monotonic()
                tail = text[-100:]

        copy_task = asyncio.create_task(copy_output())
        try:
            while process.returncode is None:
                if copy_task.done() and copy_task.exception() is not None:
                    raise copy_task.exception()
                try:
                    for child in psutil.Process(process.pid).children(recursive=True):
                        tracked[child.pid] = child.create_time()
                except psutil.NoSuchProcess:
                    pass
                if time.monotonic() - last_output > config["no_output_timeout_seconds"]:
                    raise TimeoutError(
                        "No rollout/trainer progress output; terminating owned run"
                    )
                if time.monotonic() - started > config["run_timeout_seconds"]:
                    raise TimeoutError("Run wall-clock budget exceeded")
                await asyncio.sleep(2)
            await copy_task
            if process.returncode:
                raise RuntimeError(
                    f"Training exited {process.returncode}: {output / 'train.log'}"
                )
        finally:
            if process.returncode is None and config.get("save_checkpoint", True):
                from robustops.training.recovery_rl_emergency import (
                    request_training_stop,
                )

                # asyncio.run cancels sibling tasks on Ctrl+C. Keep draining
                # the child pipe while it finishes episodes and saves weights.
                if copy_task.done() or copy_task.cancelling():
                    await asyncio.gather(copy_task, return_exceptions=True)
                    copy_task = asyncio.create_task(copy_output(check_metrics=False))
                await request_training_stop(
                    process, output, config, str(sys.exception() or "launcher shutdown")
                )
            try:
                for child in psutil.Process(process.pid).children(recursive=True):
                    tracked[child.pid] = child.create_time()
            except psutil.NoSuchProcess:
                pass
            copy_task.cancel()
            await asyncio.gather(copy_task, return_exceptions=True)
            owned = []
            for pid, created in tracked.items():
                try:
                    p = psutil.Process(pid)
                    if p.create_time() == created:
                        p.terminate()
                        owned.append(p)
                except psutil.NoSuchProcess:
                    pass
            if process.returncode is None:
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGTERM)
            try:
                # Drain pipes after terminating descendants that may inherit them.
                await asyncio.wait_for(process.communicate(), timeout=10)
            except asyncio.TimeoutError:
                for p in owned:
                    with suppress(psutil.NoSuchProcess):
                        p.kill()
                with suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                await asyncio.wait_for(process.communicate(), timeout=10)
            _, alive = psutil.wait_procs(owned, timeout=2)
            for p in alive:
                with suppress(psutil.NoSuchProcess):
                    p.kill()
            _, remaining = psutil.wait_procs(alive, timeout=30)
            live = []
            for p in remaining:
                with suppress(psutil.NoSuchProcess):
                    if p.status() != psutil.STATUS_ZOMBIE:
                        live.append(p.pid)
            write_json(
                output / "process_cleanup.json",
                dict(
                    parent_returncode=process.returncode,
                    tracked_processes=len(tracked),
                    remaining_live_owned_pids=live,
                ),
            )
            if live:
                raise RuntimeError(f"Owned processes still alive after cleanup: {live}")


async def execute(root, recipe, *, dry_run=False):
    from robustops.launch.config import resolve_model_path, resolve_project_path
    from robustops.training.rl_resume import resolve_continuation
    from robustops.training.recovery_rl_data import require_current_task_contract

    root = Path(root).resolve()
    recipe = deepcopy(recipe)
    supported = {
        "schema_version",
        "interface",
        "pipeline",
        "experiment_id",
        "description",
        "hardware",
        "model",
        "data",
        "training",
        "output",
        "launcher",
        "runtime_hardware",
        "resume",
        "run_directory",
        "task_plan_source",
    }
    if set(recipe) - supported:
        raise ValueError(
            f"unsupported RL recipe fields: {sorted(set(recipe) - supported)}"
        )
    hardware = (
        recipe.get("runtime_hardware") or select_gpus(None, dry_run=dry_run).record()
    )
    config, saved, source_run = resolve_continuation(
        root, recipe, hardware["selected_gpu_count"]
    )
    validate_training_config(config)
    validate_gpu_count(config)
    if saved:
        require_current_task_contract(saved["data_directory"])
    model = resolve_model_path(root, recipe["model"]["path"])
    from robustops.models.checkpoint_validation import inspect_local_model

    provenance = inspect_local_model(
        root, model, lineage=recipe["model"].get("lineage"), allow_missing=dry_run
    )
    if config.get("verifier_graph_dir"):
        from robustops.training.recovery_verifier_graph import load_graphs

        config["verifier_graph_dir"] = str(
            resolve_project_path(root, config["verifier_graph_dir"])
        )
        load_graphs(config["verifier_graph_dir"])

    stamp = datetime.datetime.now(datetime.UTC).strftime("%Y%m%dT%H%M%S%fZ")
    continuation = recipe.get("resume", {})
    fork = bool(continuation.get("fork_output"))
    output_root = resolve_project_path(root, recipe["output"])
    if fork:
        run = resolve_project_path(root, continuation["fork_output"]).resolve()
        if run.is_relative_to(source_run):
            raise ValueError("fork_output must be outside the source run directory")
    elif saved:
        run = source_run
    elif recipe.get("run_directory"):
        run = resolve_project_path(root, recipe["run_directory"]).resolve()
    else:
        run = (
            output_root / f"train-{stamp}-{os.getpid()}"
        ).resolve()
    if (not saved or fork) and run.exists():
        raise ValueError(f"new run directory already exists: {run}")
    output = run / "trainer"
    config["focus_run_label"] = run.name
    config["metrics_start_step"] = (
        saved["step"] if fork else config.get("metrics_start_step", 0)
    )
    if fork:
        config["resume_history_output"] = str(source_run / "trainer")
    recipe["training"] = config
    recipe["data"] = config["data_sources"]
    recipe["runtime_hardware"] = hardware
    task_plan = recipe.get("task_plan_source")
    if task_plan and (saved or config.get("task_sampling") == "queue"):
        raise ValueError(
            "task-plan reuse requires a fresh run with frozen task sampling"
        )
    pairing = None
    if dry_run:
        if task_plan:
            import tempfile
            from robustops.training.rl_resume import reuse_task_plan

            with tempfile.TemporaryDirectory(prefix="rl-plan-") as tmp:
                pairing = reuse_task_plan(
                    resolve_project_path(root, task_plan),
                    tmp,
                    config,
                    include_initial_states=False,
                )
        print(
            json.dumps(
                dict(
                    model=str(model),
                    model_ready=provenance.get("ready", False),
                    data_sources=config["data_sources"],
                    output=str(run),
                    reward_normalization=config["reward_normalization"],
                    runtime_hardware=hardware,
                    resume_checkpoint=config.get("resume_checkpoint"),
                    initialization="full checkpoint including optimizer and queues"
                    if saved
                    else "SFT",
                    fork=fork,
                    schedule_pairing=pairing,
                    command=hydra_command(root, run, model, config, {}),
                    training_groups=dict(clean=16, fault=16),
                    gpu_tested=False,
                    actual_sequence_limit=config["max_context_tokens"],
                ),
                indent=2,
            )
        )
        return

    hardware = hardware_gate(hardware["selected_gpu_count"], min_host_memory_gib=config["min_host_memory_gib"])
    from robustops.training.recovery_rl_data import prepare
    from robustops.training.rl_resume import (
        extend_data,
        fork_data,
        preserve_uncommitted,
        reuse_task_plan,
    )
    from robustops.evaluation.runtime_services import launch, stop_processes, SERVERS

    run.mkdir(parents=True, exist_ok=bool(saved and not fork))
    if saved and not fork:
        preserve_uncommitted(run, saved["step"], stamp)
        completed = output / "completed.json"
        if completed.exists():
            write_json(
                output / "completions" / f"{saved['step']}.json",
                json.loads(completed.read_text()),
            )
            completed.unlink()
        for name in ("completed.json", "failed.json"):
            path = run / name
            if path.exists():
                target = output / "continuations" / stamp / name
                target.parent.mkdir(parents=True, exist_ok=True)
                path.rename(target)
    if fork:
        fork_data(saved, run)
        write_json(
            run / "fork_source.json",
            dict(
                checkpoint=config["resume_checkpoint"],
                step=saved["step"],
                source_run=str(source_run),
                reward_normalization=config["reward_normalization"],
                training_config=config,
            ),
        )
    if saved:
        extend_data(root, run / "data", config)
    elif task_plan:
        reuse_task_plan(resolve_project_path(root, task_plan), run, config)
    else:
        prepare(root, run / "data", config)
    persistent = {
        k: v
        for k, v in recipe.items()
        if k not in ("resume", "run_directory", "task_plan_source")
    }
    persistent["training"] = {
        k: v for k, v in config.items() if k != "resume_checkpoint"
    }
    write_json(run / "recipe.json", persistent)
    evidence = output / "continuations" / stamp if saved else run
    write_json(evidence / "hardware.json", hardware)
    write_json(evidence / "model_provenance.json", provenance)
    if saved:
        write_json(
            evidence / "continuation.json",
            dict(
                checkpoint=config["resume_checkpoint"],
                start_step=saved["step"],
                next_visit=saved["next_visit"],
                recipe=persistent,
            ),
        )
    git = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True
    )
    (evidence / "code_revision.txt").write_text(git.stdout)
    with (evidence / "runtime_code.patch").open("w") as stream:
        if git.returncode == 0:
            subprocess.run(
                ["git", "diff", "HEAD", "--", "src", "recipes", "scripts"],
                cwd=root,
                stdout=stream,
                check=True,
            )
    # Retain the exact training implementation, including newly created modules.
    for source in (root / "src/robustops/training").glob("*.py"):
        target = evidence / "code" / source.relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    if config.get("verifier_graph_dir"):
        shutil.copytree(config["verifier_graph_dir"], evidence / "verifier_graph")
    for relative in (
        "verl/verl/trainer/ppo/ray_trainer.py",
        "verl/verl/workers/utils/losses.py",
        "src/robustops/evaluation/tool_protocol.py",
    ):
        source = root / relative
        target = evidence / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    env = dict(
        os.environ,
        PYTHONPATH=os.pathsep.join([str(root / "src"), str(root / "verl")]),
        TOKENIZERS_PARALLELISM="false",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )
    processes = []
    try:
        urls = {
            name: f"http://127.0.0.1:{port + 2000}" for name, port in SERVERS.values()
        }
        command = hydra_command(root, run, model, config, urls)
        write_json(evidence / "preflight/command.json", command)
        await run_child(
            [
                sys.executable,
                "-m",
                "robustops.training.recovery_rl_preflight",
                "--command",
                str(evidence / "preflight/command.json"),
            ],
            evidence / "preflight",
            dict(
                config,
                run_timeout_seconds=600,
                no_output_timeout_seconds=600,
                save_checkpoint=False,
            ),
            env,
        )
        runtime = run / "mcp" / stamp
        runtime.mkdir(parents=True, exist_ok=False)
        processes, urls = await launch(
            {name for name, _ in SERVERS.values()}, runtime, port_offset=2000
        )
        write_json(evidence / "owned_mcp.json", [p.pid for p in processes])
        command = hydra_command(root, run, model, config, urls)
        write_json(output / "command.json", command)
        await run_child(
            command,
            output,
            config,
            dict(env, TENSORBOARD_DIR=str(output / "tensorboard")),
        )
        expected = config["steps"]
        checkpoint = None
        if config.get("save_checkpoint", True):
            directory = checkpoint_directory(run, config)
            step = int((directory / "latest_checkpointed_iteration.txt").read_text())
            if step != expected:
                raise RuntimeError("checkpoint does not match requested final step")
            checkpoint = directory / f"global_step_{step}"
            validate_resumable_checkpoint(
                checkpoint, expected_world_size=config["gpu_count"]
            )
            state = json.loads((checkpoint / "recovery_state.json").read_text())
            if (
                state["step"] != expected
                or state["next_visit"] != expected * config["batch_size"]
            ):
                raise RuntimeError(
                    "checkpoint optimizer step and sampler cursor disagree"
                )
            for step in config.get("hf_export_steps", []):
                if (saved["step"] if saved else 0) < step <= expected:
                    export = json.loads(
                        (output / "model_exports" / f"{step}.json").read_text()
                    )
                    validate_hf_export(export["path"])
        elif not re.search(
            rf"training/global_step:\s*{expected}\b", (output / "train.log").read_text()
        ):
            raise RuntimeError("trainer exited without final step metrics")
        completion = dict(
            steps=expected,
            checkpoint=str(checkpoint) if checkpoint else None,
            resumable=checkpoint is not None,
            reward_normalization=config["reward_normalization"],
        )
        write_json(output / "completed.json", completion)
        write_json(run / "completed.json", completion)
        print(f"RL COMPLETE: {run}", flush=True)
    except BaseException as exc:
        write_json(run / "failed.json", dict(type=type(exc).__name__, message=str(exc)))
        raise
    finally:
        stop_processes(processes)
        write_json(
            evidence / "resource_cleanup.json",
            dict(mcp_stopped=all(p.poll() is not None for p in processes)),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--recipe-json", help=argparse.SUPPRESS)
    source.add_argument("--config", type=Path)
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    parser.add_argument(
        "--resume-run", type=Path, help="run containing a full checkpoint"
    )
    parser.add_argument(
        "--resume-step", type=int, help="saved step; requires --fork-output"
    )
    parser.add_argument(
        "--fork-output",
        type=Path,
        help="new run directory for an independent continuation",
    )
    parser.add_argument(
        "--to-step", type=int, help="cumulative target step, not additional steps"
    )
    parser.add_argument("--gpus", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.recipe_json:
        recipe = json.loads(args.recipe_json)
    elif args.config:
        recipe = json.loads(args.config.read_text())
    elif args.resume_run:
        recipe = json.loads((args.resume_run / "recipe.json").read_text())
        # Saved recipes contain the resolved checkpoint from the previous launch.
        # Resolve the latest checkpoint again for this continuation.
        recipe["training"].pop("resume_checkpoint", None)
        # Older saved recipes may contain the removed permanent-retention setting.
        recipe["training"].pop("checkpoint_protected_steps", None)
    else:
        parser.error("provide --config or --resume-run")
    if args.resume_step is not None and not args.fork_output:
        parser.error("--resume-step requires --fork-output")
    if (args.resume_step is not None or args.fork_output) and not args.resume_run:
        parser.error("fork options require --resume-run")
    if args.resume_run:
        recipe["resume"] = dict(run=str(args.resume_run.resolve()))
        if args.resume_step is not None:
            recipe["resume"]["step"] = args.resume_step
        if args.fork_output:
            recipe["resume"]["fork_output"] = str(args.fork_output.resolve())
    if args.to_step is not None:
        recipe["training"]["steps"] = args.to_step
    if args.gpus is not None or not args.recipe_json:
        selected = select_gpus(args.gpus, dry_run=args.dry_run)
        recipe["runtime_hardware"] = selected.record()
        if not args.dry_run:
            os.environ["CUDA_VISIBLE_DEVICES"] = selected.cuda_visible_devices

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for sig in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, interrupted)
    asyncio.run(execute(args.root, recipe, dry_run=args.dry_run))


if __name__ == "__main__":
    main()
