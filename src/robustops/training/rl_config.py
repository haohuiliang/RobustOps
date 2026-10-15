"""Validate the supported GRPO recipe before launching services or allocating GPUs."""

import math

RUNTIME_DEFAULTS = {
    "checkpoint_max_to_keep": None,
    "min_host_memory_gib": 0,
    "ray_num_cpus": None,
    "ray_object_store_memory_gib": None,
    "max_turns": 50,
    "max_context_tokens": 32768,
    "max_generation_tokens": 2048,
    "rollout_gpu_memory_utilization": 0.5,
    "rollout_max_num_batched_tokens": 8192,
}

TRAINING_FIELDS = set(RUNTIME_DEFAULTS) | {
    "seed",
    "epochs",
    "steps",
    "batch_size",
    "group_size",
    "learning_rate",
    "rollout_concurrency_per_worker",
    "actor_token_batch_per_gpu",
    "episode_timeout_seconds",
    "no_output_timeout_seconds",
    "run_timeout_seconds",
    "save_checkpoint",
    "checkpoint_save_contents",
    "checkpoint_save_frequency",
    "checkpoint_max_to_keep",
    "kl_loss_coef",
    "verifier_graph_dir",
    "advantage_estimator",
    "reward_weights",
    "reward_normalization",
    "local_bad_eta",
    "sft_records",
    "data_sources",
    "hf_export_steps",
    "fault_point_policy",
    "rollout_temperature",
    "zero_pg_replay_delay",
    "zero_pg_replay_max_zero_groups",
    "sft_val_every_step",
    "q_temporal_credit",
    "local_read_repeat_from",
    "terminal_stop_audit",
    "terminal_stop_penalty",
    "exclude_all_success_groups",
    "verified_reference_repeats",
    "terminal_stop_advantage_cap",
    "task_sampling",
    "fault_strategy",
    "gpu_count",
    "resume_checkpoint",
    "resume_history_output",
    "focus_run_label",
    "metrics_start_step",
}

# These change run duration, storage or diagnostics, not the saved objective.
RESUME_FIELDS = {
    "min_host_memory_gib",
    "ray_num_cpus",
    "ray_object_store_memory_gib",
    "rollout_gpu_memory_utilization",
    "rollout_max_num_batched_tokens",
    "steps",
    "gpu_count",
    "episode_timeout_seconds",
    "no_output_timeout_seconds",
    "run_timeout_seconds",
    "checkpoint_save_frequency",
    "checkpoint_max_to_keep",
    "hf_export_steps",
    "rollout_concurrency_per_worker",
    "actor_token_batch_per_gpu",
}
# Full optimizer/dataloader state cannot be reinterpreted by a fork.
FORK_FIXED_FIELDS = {
    "data_sources",
    "seed",
    "batch_size",
    "group_size",
    "learning_rate",
    "epochs",
    "task_sampling",
    "sft_records",
    "advantage_estimator",
}


def validate_training_config(config):
    unknown = set(config) - TRAINING_FIELDS
    if unknown:
        raise ValueError(f"unsupported training settings: {sorted(unknown)}")
    runtime = dict(RUNTIME_DEFAULTS, **config)
    for name in ("max_turns", "max_context_tokens", "max_generation_tokens", "rollout_max_num_batched_tokens"):
        if type(runtime[name]) is not int or runtime[name] <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if runtime["max_generation_tokens"] >= runtime["max_context_tokens"]:
        raise ValueError("max_generation_tokens must be smaller than max_context_tokens")
    if runtime["ray_num_cpus"] is not None and (type(runtime["ray_num_cpus"]) is not int or runtime["ray_num_cpus"] <= 0):
        raise ValueError("ray_num_cpus must be null or a positive integer")
    for name in ("min_host_memory_gib", "ray_object_store_memory_gib", "rollout_gpu_memory_utilization"):
        value = runtime[name]
        if name == "ray_object_store_memory_gib" and value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
        if name == "ray_object_store_memory_gib" and value == 0:
            raise ValueError("ray_object_store_memory_gib must be null or positive")
    if not 0 < runtime["rollout_gpu_memory_utilization"] <= 1:
        raise ValueError("rollout_gpu_memory_utilization must be in (0, 1]")
    if "data_sources" in config:
        from robustops.training.rl_data_config import resolve_data_sources

        resolve_data_sources(".", config["data_sources"])
    if "sft_records" in config and (type(config["sft_records"]) is not int or config["sft_records"] <= 0):
        raise ValueError("sft_records must be a positive optional expected count")
    if config.get("advantage_estimator") != "grpo":
        raise ValueError("advantage_estimator must be grpo")
    if config.get("reward_normalization") not in ("aggregate", "component"):
        raise ValueError("reward_normalization must be aggregate or component")
    weights = config.get("reward_weights")
    if (
        not isinstance(weights, (list, tuple))
        or len(weights) != 3
        or any(
            type(w) not in (int, float) or not math.isfinite(w) or w < 0
            for w in weights
        )
        or not any(weights)
    ):
        raise ValueError(
            "reward_weights must contain three finite nonnegative values, with at least one positive"
        )
    for name in (
        "steps",
        "batch_size",
        "group_size",
        "checkpoint_save_frequency",
        "rollout_concurrency_per_worker",
        "episode_timeout_seconds",
        "no_output_timeout_seconds",
        "run_timeout_seconds",
    ):
        value = config.get(name)
        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    keep = config.get("checkpoint_max_to_keep")
    if keep is not None and (type(keep) is not int or keep <= 0):
        raise ValueError("checkpoint_max_to_keep must be null or a positive integer")
    if (config["batch_size"], config["group_size"]) != (32, 4):
        raise ValueError(
            "RL uses 16 clean and 16 fault groups with four samples each"
        )
    if config.get("task_sampling") not in ("queue", "frozen"):
        raise ValueError("task_sampling must be queue or frozen")
    if config.get("save_checkpoint", True) and set(
        config.get("checkpoint_save_contents", [])
    ) != {"model", "optimizer", "extra"}:
        raise ValueError(
            "resumable checkpoints require model, optimizer and extra state"
        )
    for name in (
        "local_bad_eta",
        "kl_loss_coef",
        "learning_rate",
        "rollout_temperature",
    ):
        value = config.get(name)
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if config["learning_rate"] == 0 or config["rollout_temperature"] == 0:
        raise ValueError("learning_rate and rollout_temperature must be positive")
    for name in ("hf_export_steps",):
        values = config.get(name, [])
        if (
            not isinstance(values, list)
            or any(type(v) is not int or v <= 0 for v in values)
            or len(set(values)) != len(values)
        ):
            raise ValueError(f"{name} must contain unique positive steps")
