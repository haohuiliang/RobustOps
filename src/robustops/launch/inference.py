from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
from typing import Any

from robustops.launch.config import (
    require_string_list as _string_list,
    LaunchConfigError,
    require_mapping,
    require_string,
    resolve_model_path,
    resolve_project_path,
)
DOMAINS = {"calendar", "csm", "drive", "email", "hr", "hybrid", "itsm", "teams"}
EVALUATION_MODES = ("oracle", "plus_5_tools", "plus_10_tools")
BACKEND_KIND_BY_MODE = {
    "oracle": "eval_oracle",
    "plus_5_tools": "eval_plus5",
    "plus_10_tools": "eval_plus10",
}
DEFAULT_MCP_SERVER_CONCURRENCY = {
    "gym-calendar": 2,
    "gym-email-mcp": 8,
    "gym-teams-mcp": 2,
    "sn-csm-server": 8,
    "sn-hr-internal": 4,
    "gym-itsm-mcp": 4,
    "gym-google-drive-mcp": 8,
}

PLAN_PRESERVING_EXECUTION = {
    "controller": "first_tool_only_plan_preserving_v1",
    "assistant_content_policy": "preserve_full",
    "tool_call_policy": "execute_first_only_then_reinfer",
}


def _positive_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise LaunchConfigError(f"{name} must be a positive integer")
    return value


def _validate_workload_scope(
    task_manifest: Path,
    workload: dict[str, Any],
    partitions: list[str],
    modes: list[str],
    expected: dict[str, Any],
) -> tuple[list[str], list[str] | None]:
    domains = _string_list(
        workload.get("domains", sorted(DOMAINS)),
        "workload.domains",
    )
    unsupported_domains = sorted(set(domains) - DOMAINS)
    if unsupported_domains:
        raise LaunchConfigError(f"unsupported workload domains: {unsupported_domains}")
    task_ids_value = workload.get("task_ids")
    task_ids = (
        _string_list(task_ids_value, "workload.task_ids")
        if task_ids_value is not None
        else None
    )
    if task_ids is not None and len(partitions) != 1:
        raise LaunchConfigError("an exact workload.task_ids subset requires exactly one partition")

    eligible: dict[str, dict[str, str]] = {partition: {} for partition in partitions}
    for line_number, line in enumerate(
        task_manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LaunchConfigError(
                f"invalid task manifest JSON at line {line_number}: {exc}"
            ) from exc
        partition = row.get("partition")
        domain = row.get("domain")
        task_id = row.get("task_id")
        runnable_modes = row.get("runnable_modes", row.get("available_modes", []))
        if (
            partition in eligible
            and domain in domains
            and isinstance(task_id, str)
            and set(modes) <= set(row.get("paired_modes", []))
            and set(modes) <= set(runnable_modes)
        ):
            if task_id in eligible[partition]:
                raise LaunchConfigError(f"duplicate paired task ID in manifest: {task_id}")
            eligible[partition][task_id] = domain

    for partition in partitions:
        expected_count = _positive_int(
            expected.get(partition),
            f"workload.expected_tasks_per_mode.{partition}",
        )
        selected_ids = task_ids if task_ids is not None else list(eligible[partition])
        missing = [task_id for task_id in selected_ids if task_id not in eligible[partition]]
        if missing:
            raise LaunchConfigError(
                f"exact task IDs are outside {partition}/{domains}: {missing}"
            )
        if len(selected_ids) != expected_count:
            raise LaunchConfigError(
                f"{partition} expected task count differs from manifest scope: "
                f"expected={expected_count}, actual={len(selected_ids)}"
            )
    return domains, task_ids


def _validate_available_workload_scope(
    root: Path,
    task_manifest: Path,
    workload: dict[str, Any],
    partitions: list[str],
    modes: list[str],
) -> tuple[list[str], dict[str, dict[str, set[str]]], dict[str, dict[str, int]]]:
    domains = _string_list(
        workload.get("domains", sorted(DOMAINS)),
        "workload.domains",
    )
    unsupported_domains = sorted(set(domains) - DOMAINS)
    if unsupported_domains:
        raise LaunchConfigError(f"unsupported workload domains: {unsupported_domains}")
    if workload.get("task_ids") is not None:
        raise LaunchConfigError(
            "mode evaluation does not accept a shared workload.task_ids subset"
        )
    selection = workload.get("selection")
    if selection not in {
        "all_available",
        "semantic_paired",
        "exact_by_mode",
        "exact_by_partition_and_mode",
    }:
        raise LaunchConfigError(
            "mode evaluation requires workload.selection='all_available' or "
            "'semantic_paired' or an exact per-mode selection"
        )
    manifest_mode_field = (
        "paired_modes" if selection == "semantic_paired" else "available_modes"
    )

    expected_value = workload.get("expected_tasks_by_mode")
    if not isinstance(expected_value, dict):
        raise LaunchConfigError("workload.expected_tasks_by_mode must be an object")
    if set(expected_value) != set(modes):
        raise LaunchConfigError(
            "workload.expected_tasks_by_mode keys differ from workload.modes: "
            f"expected={sorted(modes)}, actual={sorted(expected_value)}"
        )
    expected: dict[str, dict[str, int]] = {}
    for mode in modes:
        by_partition = expected_value.get(mode)
        if not isinstance(by_partition, dict):
            raise LaunchConfigError(
                f"workload.expected_tasks_by_mode.{mode} must be an object"
            )
        expected[mode] = {
            partition: _positive_int(
                by_partition.get(partition),
                f"workload.expected_tasks_by_mode.{mode}.{partition}",
            )
            for partition in partitions
        }

    eligible: dict[str, dict[str, set[str]]] = {
        partition: {mode: set() for mode in modes}
        for partition in partitions
    }
    seen: set[str] = set()
    for line_number, line in enumerate(
        task_manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LaunchConfigError(
                f"invalid task manifest JSON at line {line_number}: {exc}"
            ) from exc
        task_id = row.get("task_id")
        if not isinstance(task_id, str) or not task_id or task_id in seen:
            raise LaunchConfigError(f"invalid or duplicate task ID in manifest: {task_id!r}")
        seen.add(task_id)
        partition = row.get("partition")
        if partition not in eligible or row.get("domain") not in domains:
            continue
        selected_modes = row.get(manifest_mode_field)
        if not isinstance(selected_modes, list) or not all(
            isinstance(mode, str) for mode in selected_modes
        ):
            raise LaunchConfigError(
                f"invalid {manifest_mode_field} in task manifest at line {line_number}"
            )
        runnable_modes = row.get("runnable_modes", row.get("available_modes"))
        if not isinstance(runnable_modes, list) or not all(
            isinstance(mode, str) for mode in runnable_modes
        ):
            raise LaunchConfigError(
                f"invalid runnable_modes in task manifest at line {line_number}"
            )
        for mode in modes:
            if mode in selected_modes and mode in runnable_modes:
                eligible[partition][mode].add(task_id)

    # Exact delta recipes may intentionally select a +N cell whose Oracle cell
    # was already completed in an earlier invocation.  Preserve the manifest
    # eligibility before narrowing so the source-row check does not incorrectly
    # require Oracle to be regenerated in the same invocation.
    source_eligible = {
        partition: {
            mode: set(task_ids) for mode, task_ids in eligible_by_mode.items()
        }
        for partition, eligible_by_mode in eligible.items()
    }

    if selection == "exact_by_mode":
        if len(partitions) != 1:
            raise LaunchConfigError(
                "exact_by_mode inference requires exactly one partition"
            )
        configured_files = workload.get("task_id_files_by_mode")
        if not isinstance(configured_files, dict) or set(configured_files) != set(modes):
            raise LaunchConfigError(
                "workload.task_id_files_by_mode keys must equal workload.modes"
            )
        partition = partitions[0]
        selected: dict[str, set[str]] = {}
        for mode in modes:
            file_values = _string_list(
                configured_files.get(mode),
                f"workload.task_id_files_by_mode.{mode}",
            )
            requested: list[str] = []
            seen_requested: set[str] = set()
            for file_value in file_values:
                path = resolve_project_path(root, file_value)
                if not path.is_file():
                    raise LaunchConfigError(
                        f"exact task ID file does not exist for {mode}: {path}"
                    )
                for line_number, line in enumerate(
                    path.read_text(encoding="utf-8").splitlines(),
                    start=1,
                ):
                    task_id = line.strip()
                    if not task_id:
                        continue
                    if task_id in seen_requested:
                        raise LaunchConfigError(
                            f"duplicate exact task ID for {mode}: {task_id!r} "
                            f"at {path}:{line_number}"
                        )
                    seen_requested.add(task_id)
                    requested.append(task_id)
            if not requested:
                raise LaunchConfigError(
                    f"exact task selection is empty for {partition}/{mode}"
                )
            missing = sorted(set(requested) - eligible[partition][mode])
            if missing:
                raise LaunchConfigError(
                    f"exact task IDs are outside {partition}/{mode}: {missing}"
                )
            selected[mode] = set(requested)
        eligible[partition] = selected
    elif selection == "exact_by_partition_and_mode":
        configured = workload.get("task_ids_by_partition_and_mode")
        if not isinstance(configured, dict) or set(configured) != set(partitions):
            raise LaunchConfigError(
                "workload.task_ids_by_partition_and_mode keys must equal "
                "workload.partitions"
            )
        for partition in partitions:
            configured_by_mode = configured.get(partition)
            if (
                not isinstance(configured_by_mode, dict)
                or set(configured_by_mode) != set(modes)
            ):
                raise LaunchConfigError(
                    "workload.task_ids_by_partition_and_mode."
                    f"{partition} keys must equal workload.modes"
                )
            selected = {}
            for mode in modes:
                requested = _string_list(
                    configured_by_mode.get(mode),
                    "workload.task_ids_by_partition_and_mode."
                    f"{partition}.{mode}",
                )
                missing = sorted(set(requested) - eligible[partition][mode])
                if missing:
                    raise LaunchConfigError(
                        f"exact task IDs are outside {partition}/{mode}: {missing}"
                    )
                selected[mode] = set(requested)
            eligible[partition] = selected

    for partition in partitions:
        for mode in modes:
            actual = len(eligible[partition][mode])
            configured = expected[mode][partition]
            if actual != configured:
                raise LaunchConfigError(
                    f"{partition}/{mode} expected task count differs from manifest scope: "
                    f"expected={configured}, actual={actual}"
                )
        if "oracle" in modes and selection != "exact_by_mode":
            for mode in modes:
                if mode != "oracle" and not (
                    eligible[partition][mode] <= source_eligible[partition]["oracle"]
                ):
                    raise LaunchConfigError(
                        f"{partition} contains {mode} task IDs without an Oracle source row"
                    )
    return domains, eligible, expected


def _validate_plan_preserving_selection(
    root: Path,
    workload: dict[str, Any],
    partitions: list[str],
    modes: list[str],
    eligible: dict[str, dict[str, set[str]]],
    rollouts_per_task: int,
) -> tuple[dict[str, dict[str, list[str]]], dict[str, Any]]:
    # Source-branch regeneration is the only launch path that needs trajectory
    # parsing. Keep the heavy evaluator/Transformers import out of ordinary
    # config validation and exact-subset dry-runs.
    from robustops.evaluation.clean_baseline import _first_multi_tool_turn

    config = workload.get("plan_preserving")
    if not isinstance(config, dict):
        raise LaunchConfigError("workload.plan_preserving must be an object")
    required_policy = {
        "selection": "attempt_has_multi_tool_turn",
        "branch": "first_multi_tool_turn",
        "assistant_content_policy": "preserve_full",
        "tool_call_policy": "execute_first_only_then_reinfer",
    }
    for field, expected_value in required_policy.items():
        if config.get(field) != expected_value:
            raise LaunchConfigError(
                f"workload.plan_preserving.{field} must be {expected_value!r}"
            )
    sources = config.get("source_attempts_by_mode")
    expected_source_counts = config.get("expected_source_attempts_by_mode")
    expected_affected_attempts = config.get("expected_affected_attempts_by_mode")
    expected_affected_tasks = config.get("expected_affected_tasks_by_mode")
    for field, value in (
        ("source_attempts_by_mode", sources),
        ("expected_source_attempts_by_mode", expected_source_counts),
        ("expected_affected_attempts_by_mode", expected_affected_attempts),
        ("expected_affected_tasks_by_mode", expected_affected_tasks),
    ):
        if not isinstance(value, dict) or set(value) != set(modes):
            raise LaunchConfigError(
                f"workload.plan_preserving.{field} keys must equal workload.modes"
            )

    selected: dict[str, dict[str, list[str]]] = {
        partition: {} for partition in partitions
    }
    summary: dict[str, Any] = {
        "selection": required_policy["selection"],
        "branch": required_policy["branch"],
        "assistant_content_policy": required_policy["assistant_content_policy"],
        "tool_call_policy": required_policy["tool_call_policy"],
        "by_partition": {partition: {} for partition in partitions},
    }
    for mode in modes:
        configured_by_field: dict[str, dict[str, Any]] = {}
        for field, values in (
            ("source_attempts_by_mode", sources),
            ("expected_source_attempts_by_mode", expected_source_counts),
            ("expected_affected_attempts_by_mode", expected_affected_attempts),
            ("expected_affected_tasks_by_mode", expected_affected_tasks),
        ):
            by_partition = values.get(mode)
            if not isinstance(by_partition, dict) or set(by_partition) != set(partitions):
                raise LaunchConfigError(
                    f"workload.plan_preserving.{field}.{mode} keys must equal "
                    "workload.partitions"
                )
            configured_by_field[field] = by_partition

        for partition in partitions:
            source_value = configured_by_field["source_attempts_by_mode"][partition]
            if not isinstance(source_value, str) or not source_value:
                raise LaunchConfigError(
                    "workload.plan_preserving.source_attempts_by_mode."
                    f"{mode}.{partition} must be a path"
                )
            source_path = resolve_project_path(root, source_value)
            if not source_path.is_file():
                raise LaunchConfigError(
                    "plan-preserving source attempts do not exist for "
                    f"{partition}/{mode}: {source_path}"
                )
            source_count = 0
            seen_keys: set[tuple[str, int]] = set()
            affected_by_task: dict[str, list[int]] = {}
            with source_path.open(encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, start=1):
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise LaunchConfigError(
                            f"invalid plan-preserving source JSON at "
                            f"{source_path}:{line_number}"
                        ) from exc
                    task_id = row.get("task_id")
                    attempt_index = row.get("attempt")
                    cleanup = row.get("cleanup")
                    if (
                        row.get("mode") != mode
                        or not isinstance(task_id, str)
                        or task_id not in eligible[partition][mode]
                        or not isinstance(attempt_index, int)
                        or isinstance(attempt_index, bool)
                        or not 1 <= attempt_index <= rollouts_per_task
                        or row.get("valid") is not True
                        or row.get("status") not in {"success", "task_failure"}
                        or not isinstance(cleanup, dict)
                        or cleanup.get("succeeded") is not True
                    ):
                        raise LaunchConfigError(
                            "plan-preserving source attempt is outside the frozen valid "
                            f"{partition} scope: {source_path}:{line_number}"
                        )
                    key = (task_id, attempt_index)
                    if key in seen_keys:
                        raise LaunchConfigError(
                            f"duplicate plan-preserving source task/attempt key: {key}"
                        )
                    seen_keys.add(key)
                    source_count += 1
                    if _first_multi_tool_turn(row) is not None:
                        affected_by_task.setdefault(task_id, []).append(attempt_index)

            configured_source_count = _positive_int(
                configured_by_field["expected_source_attempts_by_mode"][partition],
                "workload.plan_preserving.expected_source_attempts_by_mode."
                f"{mode}.{partition}",
            )
            configured_affected_attempts = _positive_int(
                configured_by_field["expected_affected_attempts_by_mode"][partition],
                "workload.plan_preserving.expected_affected_attempts_by_mode."
                f"{mode}.{partition}",
            )
            configured_affected_tasks = _positive_int(
                configured_by_field["expected_affected_tasks_by_mode"][partition],
                "workload.plan_preserving.expected_affected_tasks_by_mode."
                f"{mode}.{partition}",
            )
            affected_attempt_count = sum(map(len, affected_by_task.values()))
            if (
                source_count != configured_source_count
                or affected_attempt_count != configured_affected_attempts
                or len(affected_by_task) != configured_affected_tasks
            ):
                raise LaunchConfigError(
                    f"plan-preserving source boundary changed for {partition}/{mode}: "
                    f"source={source_count}, affected_attempts={affected_attempt_count}, "
                    f"affected_tasks={len(affected_by_task)}"
                )
            attempt_indices_by_task = {
                task_id: sorted(indices)
                for task_id, indices in sorted(affected_by_task.items())
            }
            selected[partition][mode] = list(attempt_indices_by_task)
            summary["by_partition"][partition][mode] = {
                "source_attempts": str(source_path),
                "source_attempt_count": source_count,
                "affected_attempt_count": affected_attempt_count,
                "affected_task_count": len(affected_by_task),
                "attempt_indices_by_task": attempt_indices_by_task,
            }
    summary["affected_attempts_total"] = sum(
        mode_summary["affected_attempt_count"]
        for partition_summary in summary["by_partition"].values()
        for mode_summary in partition_summary.values()
    )
    return selected, summary


def _selected_domains_for_task_ids(
    task_manifest: Path,
    partition: str,
    configured_domains: list[str],
    task_ids: list[str],
) -> list[str]:
    """Return configured domains that contain at least one exact selected task."""

    requested = set(task_ids)
    if not requested:
        raise LaunchConfigError(
            f"exact task selection is empty for partition={partition!r}"
        )
    domain_by_task: dict[str, str] = {}
    configured = set(configured_domains)
    for line_number, line in enumerate(
        task_manifest.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise LaunchConfigError(
                f"invalid task manifest JSON at line {line_number}: {exc}"
            ) from exc
        task_id = row.get("task_id")
        if task_id not in requested:
            continue
        domain = row.get("domain")
        if row.get("partition") != partition or domain not in configured:
            continue
        if task_id in domain_by_task:
            raise LaunchConfigError(
                f"duplicate exact task ID in manifest: {task_id!r}"
            )
        domain_by_task[task_id] = domain
    missing = sorted(requested - set(domain_by_task))
    if missing:
        raise LaunchConfigError(
            f"exact task IDs have no domain in partition={partition!r}: {missing}"
        )
    selected_domain_set = set(domain_by_task.values())
    return [domain for domain in configured_domains if domain in selected_domain_set]


def _validate_continuation_selection(
    root: Path,
    recipe: dict[str, Any],
    workload: dict[str, Any],
    partitions: list[str],
    modes: list[str],
    eligible: dict[str, dict[str, set[str]]],
    rollouts_per_task: int,
) -> tuple[dict[str, dict[str, list[str]]] | None, dict[str, Any] | None]:
    continuation = workload.get("continuation")
    if continuation is None:
        return None, None
    if not isinstance(continuation, dict):
        raise LaunchConfigError("workload.continuation must be an object")
    if len(partitions) != 1:
        raise LaunchConfigError("continuation inference requires exactly one partition")
    if continuation.get("partial_task_policy") != "rerun_full_task":
        raise LaunchConfigError(
            "workload.continuation.partial_task_policy must be 'rerun_full_task'"
        )
    reuse_modes = _string_list(
        continuation.get("reuse_modes"),
        "workload.continuation.reuse_modes",
    )
    unsupported_reuse_modes = sorted(set(reuse_modes) - set(modes))
    if unsupported_reuse_modes:
        raise LaunchConfigError(
            f"continuation reuse modes are outside workload.modes: {unsupported_reuse_modes}"
        )
    expected_source = continuation.get("expected_source")
    if not isinstance(expected_source, dict):
        raise LaunchConfigError("workload.continuation.expected_source must be an object")
    expected_persisted = expected_source.get("persisted_attempts_by_mode")
    expected_complete = expected_source.get("complete_tasks_by_mode")
    expected_partial = expected_source.get("partial_task_attempts_by_mode")
    for name, value in (
        ("persisted_attempts_by_mode", expected_persisted),
        ("complete_tasks_by_mode", expected_complete),
        ("partial_task_attempts_by_mode", expected_partial),
    ):
        if not isinstance(value, dict) or set(value) != set(reuse_modes):
            raise LaunchConfigError(
                f"workload.continuation.expected_source.{name} keys must equal reuse_modes"
            )
    invalid_attempt_policy = continuation.get("invalid_attempt_policy")
    expected_invalid = expected_source.get("invalid_task_attempts_by_mode")
    if invalid_attempt_policy is not None:
        if invalid_attempt_policy != "rerun_full_task":
            raise LaunchConfigError(
                "workload.continuation.invalid_attempt_policy must be "
                "'rerun_full_task'"
            )
        if not isinstance(expected_invalid, dict) or set(expected_invalid) != set(
            reuse_modes
        ):
            raise LaunchConfigError(
                "workload.continuation.expected_source."
                "invalid_task_attempts_by_mode keys must equal reuse_modes"
            )
    elif expected_invalid is not None:
        raise LaunchConfigError(
            "invalid_task_attempts_by_mode requires invalid_attempt_policy"
        )

    source_root = resolve_project_path(
        root,
        require_string(continuation, "source_root"),
    )
    source_config_path = source_root / "resolved_launch_config.json"
    if not source_config_path.is_file():
        raise LaunchConfigError(
            f"continuation source resolved config does not exist: {source_config_path}"
        )
    try:
        source_config = json.loads(source_config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LaunchConfigError(
            f"invalid continuation source resolved config: {source_config_path}"
        ) from exc
    partition = partitions[0]
    model = require_mapping(recipe, "model")
    generation = require_mapping(recipe, "generation")
    source_contract = {
        "dataset_revision": source_config.get("dataset_revision"),
        "model_revision": source_config.get("resolved_model_revision"),
        "model_name": source_config.get("resolved_model_name"),
        "rollouts_per_task": source_config.get("resolved_rollouts_per_task"),
        "generation_common": (source_config.get("generation") or {}).get("common"),
        "partition_seed": ((source_config.get("generation") or {}).get("seed_by_partition") or {}).get(partition),
    }
    expected_contract = {
        "dataset_revision": recipe.get("dataset_revision"),
        "model_revision": model.get("revision"),
        "model_name": model.get("served_name"),
        "rollouts_per_task": rollouts_per_task,
        "generation_common": generation.get("common"),
        "partition_seed": (generation.get("seed_by_partition") or {}).get(partition),
    }
    if source_contract != expected_contract:
        differing = [
            key
            for key in expected_contract
            if source_contract.get(key) != expected_contract.get(key)
        ]
        raise LaunchConfigError(
            f"continuation source contract differs from the new run: {differing}"
        )
    source_partitions = source_config.get("resolved_partitions")
    source_modes = (source_config.get("workload") or {}).get("modes")
    if not isinstance(source_partitions, list) or partition not in source_partitions:
        raise LaunchConfigError(
            f"continuation source does not contain partition {partition!r}"
        )
    if not isinstance(source_modes, list) or not set(reuse_modes) <= set(source_modes):
        raise LaunchConfigError("continuation source does not contain all reuse modes")

    selected: dict[str, dict[str, list[str]]] = {
        partition: {mode: sorted(eligible[partition][mode]) for mode in modes}
    }
    expected_attempt_indices = set(range(1, rollouts_per_task + 1))
    mode_summaries: dict[str, Any] = {}
    for mode in reuse_modes:
        profile_root = (
            source_root / partition / mode
            if len(source_partitions) > 1
            else source_root / mode
        )
        attempt_paths = sorted(profile_root.glob("runs/all_domains_*/attempts.jsonl"))
        if not attempt_paths:
            raise LaunchConfigError(
                f"continuation source contains no attempt artifacts for {partition}/{mode}"
            )
        by_task: dict[str, set[int]] = {}
        seen_keys: set[tuple[str, int]] = set()
        artifact_hashes: dict[str, str] = {}
        persisted_attempts = 0
        invalid_task_attempts: dict[str, set[int]] = {}
        for path in attempt_paths:
            artifact_hashes[str(path.relative_to(source_root))] = hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            domain = path.parent.name.removeprefix("all_domains_")
            for line_number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LaunchConfigError(
                        f"invalid continuation attempt JSON at {path}:{line_number}"
                    ) from exc
                task_id = row.get("task_id")
                attempt = row.get("attempt")
                if (
                    not isinstance(task_id, str)
                    or task_id not in eligible[partition][mode]
                    or row.get("domain") != domain
                    or row.get("mode") != mode
                    or not isinstance(attempt, int)
                    or isinstance(attempt, bool)
                    or attempt not in expected_attempt_indices
                ):
                    raise LaunchConfigError(
                        f"continuation attempt identity is outside the frozen scope: "
                        f"{path}:{line_number}"
                    )
                key = (task_id, attempt)
                if key in seen_keys:
                    raise LaunchConfigError(f"duplicate continuation task/attempt key: {key}")
                seen_keys.add(key)
                cleanup = row.get("cleanup")
                reusable = (
                    row.get("valid") is True
                    and row.get("status") in {"success", "task_failure"}
                    and isinstance(cleanup, dict)
                    and cleanup.get("succeeded") is True
                )
                allowed_invalid = (
                    invalid_attempt_policy == "rerun_full_task"
                    and row.get("status") == "invalid"
                    and row.get("valid") is False
                    and isinstance(cleanup, dict)
                    and cleanup.get("succeeded") is True
                )
                if not reusable and not allowed_invalid:
                    raise LaunchConfigError(
                        f"continuation source attempt is not reusable: {path}:{line_number}"
                    )
                persisted_attempts += 1
                if reusable:
                    by_task.setdefault(task_id, set()).add(attempt)
                else:
                    invalid_task_attempts.setdefault(task_id, set()).add(attempt)

        complete_task_ids = sorted(
            task_id
            for task_id, attempts in by_task.items()
            if attempts == expected_attempt_indices
        )
        partial_task_attempts = {
            task_id: sorted(attempts)
            for task_id, attempts in sorted(by_task.items())
            if attempts != expected_attempt_indices
        }
        configured_persisted = _positive_int(
            expected_persisted.get(mode),
            f"workload.continuation.expected_source.persisted_attempts_by_mode.{mode}",
        )
        configured_complete = _positive_int(
            expected_complete.get(mode),
            f"workload.continuation.expected_source.complete_tasks_by_mode.{mode}",
        )
        configured_partial = expected_partial.get(mode)
        if not isinstance(configured_partial, dict):
            raise LaunchConfigError(
                f"workload.continuation.expected_source.partial_task_attempts_by_mode.{mode} "
                "must be an object"
            )
        configured_invalid = (
            expected_invalid.get(mode) if isinstance(expected_invalid, dict) else {}
        )
        if not isinstance(configured_invalid, dict):
            raise LaunchConfigError(
                "workload.continuation.expected_source.invalid_task_attempts_by_mode."
                f"{mode} must be an object"
            )
        normalized_invalid = {
            task_id: sorted(attempts)
            for task_id, attempts in sorted(invalid_task_attempts.items())
        }
        if (
            persisted_attempts != configured_persisted
            or len(complete_task_ids) != configured_complete
            or partial_task_attempts != configured_partial
            or normalized_invalid != configured_invalid
        ):
            raise LaunchConfigError(
                f"continuation source boundary changed for {mode}: "
                f"persisted={persisted_attempts}, complete={len(complete_task_ids)}, "
                f"partial={partial_task_attempts}, invalid={normalized_invalid}"
            )
        selected[partition][mode] = sorted(
            eligible[partition][mode] - set(complete_task_ids)
        )
        rerun_task_ids = set(partial_task_attempts) | set(invalid_task_attempts)
        if not rerun_task_ids <= set(selected[partition][mode]):
            raise LaunchConfigError(
                f"continuation superseded tasks are not selected for full rerun: {mode}"
            )
        partial_tasks_rerun_attempts = {
            task_id: sorted(expected_attempt_indices)
            for task_id in sorted(rerun_task_ids)
        }
        mode_summaries[mode] = {
            "persisted_attempts": persisted_attempts,
            "complete_tasks_reused": len(complete_task_ids),
            "attempts_reused": len(complete_task_ids) * rollouts_per_task,
            "partial_task_attempts_superseded": partial_task_attempts,
            "invalid_task_attempts_superseded": normalized_invalid,
            "partial_tasks_rerun_attempts": partial_tasks_rerun_attempts,
            "attempts_superseded": (
                sum(map(len, partial_task_attempts.values()))
                + sum(map(len, invalid_task_attempts.values()))
            ),
            "tasks_to_generate": len(selected[partition][mode]),
            "attempts_to_generate": (
                len(selected[partition][mode]) * rollouts_per_task
            ),
            "artifact_sha256": artifact_hashes,
        }

    summary = {
        "source_root": str(source_root),
        "source_resolved_config": str(source_config_path),
        "partition": partition,
        "reuse_modes": reuse_modes,
        "partial_task_policy": "rerun_full_task",
        "invalid_attempt_policy": invalid_attempt_policy,
        "by_mode": mode_summaries,
        "attempts_reused_total": sum(
            item["attempts_reused"] for item in mode_summaries.values()
        ),
        "partial_attempts_superseded_total": sum(
            sum(map(len, item["partial_task_attempts_superseded"].values()))
            for item in mode_summaries.values()
        ),
        "invalid_attempts_superseded_total": sum(
            sum(map(len, item["invalid_task_attempts_superseded"].values()))
            for item in mode_summaries.values()
        ),
        "attempts_superseded_total": sum(
            item["attempts_superseded"] for item in mode_summaries.values()
        ),
    }
    return selected, summary


def _validate_domain_recipes(
    root: Path,
    recipe: dict[str, Any],
) -> dict[str, dict[str, Path]]:
    workload = require_mapping(recipe, "workload")
    roots = require_mapping(workload, "recipe_roots")
    generation_config = require_mapping(recipe, "generation")
    generation_common = require_mapping(generation_config, "common")
    if "train" in workload.get("partitions", []) and "temperature" not in generation_common:
        raise LaunchConfigError("train generation.common.temperature must be explicit")
    generation_common.setdefault("temperature", 0.0)
    context = generation_common.get("max_context_tokens")
    if type(context) is not int or context <= 0:
        raise LaunchConfigError("generation.common.max_context_tokens must be a positive integer")
    seeds = require_mapping(generation_config, "seed_by_partition")
    model = require_mapping(recipe, "model")
    model_revision = model.get("base_revision", model.get("revision"))
    dataset_revision = recipe.get("dataset_revision")
    partitions = workload.get("partitions")
    modes = _string_list(workload.get("modes"), "workload.modes")
    if not isinstance(partitions, list) or not all(
        item in ("train", "validation", "test") for item in partitions
    ):
        raise LaunchConfigError("workload.partitions must contain train/validation/test values")
    unsupported_modes = sorted(set(modes) - set(EVALUATION_MODES))
    if unsupported_modes:
        raise LaunchConfigError(f"unsupported workload modes: {unsupported_modes}")
    resolved: dict[str, dict[str, Path]] = {}
    for partition in partitions:
        seed = seeds.get(partition)
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise LaunchConfigError(f"generation.seed_by_partition.{partition} must be an integer")
        generation = {**generation_common, "seed": seed}
        partition_roots = roots.get(partition)
        if not isinstance(partition_roots, dict):
            raise LaunchConfigError(f"workload.recipe_roots.{partition} must be an object")
        resolved[partition] = {}
        for mode in modes:
            root_value = partition_roots.get(mode)
            if not isinstance(root_value, str):
                raise LaunchConfigError(
                    f"workload.recipe_roots.{partition}.{mode} must be a path"
                )
            recipe_root = resolve_project_path(root, root_value)
            resolved[partition][mode] = recipe_root
            found = {path.stem for path in recipe_root.glob("*.json")}
            if found != DOMAINS:
                raise LaunchConfigError(
                    f"{partition}/{mode} recipe domains differ: "
                    f"expected={sorted(DOMAINS)}, actual={sorted(found)}"
                )
            for domain in sorted(DOMAINS):
                path = recipe_root / f"{domain}.json"
                payload = json.loads(path.read_text(encoding="utf-8"))
                if payload.get("domain") != domain or payload.get("mode") != mode:
                    raise LaunchConfigError(f"mode/domain mismatch in {path}")
                domain_generation = dict(payload.get("generation") or {})
                domain_generation.setdefault("temperature", 0.0)
                if domain_generation != generation:
                    raise LaunchConfigError(f"generation mismatch in {path}")
                if payload.get("model_revision") != model_revision:
                    raise LaunchConfigError(f"model revision mismatch in {path}")
                if payload.get("dataset_revision") != dataset_revision:
                    raise LaunchConfigError(f"dataset revision mismatch in {path}")
    return resolved


def _collect_profile(
    profile_dir: Path,
    expected_mode: str,
    expected_total: int,
    expected_domains: set[str],
    rollouts_per_task: int = 1,
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in sorted((profile_dir / "runs").glob("all_domains_*/attempts.jsonl")):
        attempts = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if not attempts:
            raise LaunchConfigError(f"empty trajectory artifact: {path}")
        domains = {attempt.get("domain") for attempt in attempts}
        modes = {attempt.get("mode") for attempt in attempts}
        if len(domains) != 1 or modes != {expected_mode}:
            raise LaunchConfigError(f"mode/domain mismatch in {path}: domains={domains}, modes={modes}")
        domain = domains.pop()
        task_ids = [attempt.get("task_id") for attempt in attempts]
        if not isinstance(domain, str) or any(not isinstance(task_id, str) for task_id in task_ids):
            raise LaunchConfigError(f"invalid task/domain identity in {path}")
        attempt_keys = [
            (attempt.get("task_id"), attempt.get("attempt")) for attempt in attempts
        ]
        if any(
            not isinstance(attempt_index, int)
            or isinstance(attempt_index, bool)
            or attempt_index <= 0
            for _, attempt_index in attempt_keys
        ):
            raise LaunchConfigError(f"invalid attempt index in {path}")
        if len(attempt_keys) != len(set(attempt_keys)):
            raise LaunchConfigError(f"duplicate task/attempt key in {path}")
        observed_by_task: dict[str, set[int]] = {}
        for task_id, attempt_index in attempt_keys:
            observed_by_task.setdefault(task_id, set()).add(attempt_index)
        expected_attempt_indices = set(range(1, rollouts_per_task + 1))
        incomplete = sorted(
            task_id
            for task_id, attempt_indices in observed_by_task.items()
            if attempt_indices != expected_attempt_indices
        )
        if incomplete:
            raise LaunchConfigError(
                f"tasks do not contain attempts 1..{rollouts_per_task} in {path}: "
                f"{incomplete}"
            )
        result[domain] = sorted(observed_by_task)
    if set(result) != expected_domains:
        raise LaunchConfigError(
            f"profile domains differ for {expected_mode}: "
            f"expected={sorted(expected_domains)}, actual={sorted(result)}"
        )
    if sum(map(len, result.values())) != expected_total:
        raise LaunchConfigError(f"unexpected {expected_mode} task total: {sum(map(len, result.values()))}")
    return result


def _paired_summary(
    oracle_dir: Path,
    plus10_dir: Path,
    partition: str,
    partition_root: Path,
    expected_total: int,
    expected_domains: set[str],
    experiment_id: str | None,
    rollouts_per_task: int = 1,
) -> None:
    oracle = _collect_profile(
        oracle_dir,
        "oracle",
        expected_total,
        expected_domains,
        rollouts_per_task,
    )
    plus10 = _collect_profile(
        plus10_dir,
        "plus_10_tools",
        expected_total,
        expected_domains,
        rollouts_per_task,
    )
    if oracle != plus10:
        mismatched = sorted(
            domain for domain in expected_domains if oracle[domain] != plus10[domain]
        )
        raise LaunchConfigError(f"Oracle/+10 task IDs differ for domains: {mismatched}")
    summary = {
        "experiment_id": experiment_id,
        "partition": partition,
        "paired_mode": "plus_10_tools",
        "tasks_per_mode": expected_total,
        "rollouts_per_task": rollouts_per_task,
        "attempts_per_mode": expected_total * rollouts_per_task,
        "task_counts_by_domain": {domain: len(oracle[domain]) for domain in sorted(oracle)},
        "task_ids_match": True,
        "oracle_profile": str(oracle_dir),
        "plus_10_tools_profile": str(plus10_dir),
    }
    (partition_root / "paired_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _available_mode_summary(
    oracle_dir: Path,
    plus10_dir: Path,
    partition: str,
    partition_root: Path,
    expected_oracle: int,
    expected_plus10: int,
    expected_domains: set[str],
    experiment_id: str | None,
    rollouts_per_task: int = 1,
) -> None:
    oracle = _collect_profile(
        oracle_dir,
        "oracle",
        expected_oracle,
        expected_domains,
        rollouts_per_task,
    )
    plus10 = _collect_profile(
        plus10_dir,
        "plus_10_tools",
        expected_plus10,
        expected_domains,
        rollouts_per_task,
    )
    oracle_ids = {task_id for ids in oracle.values() for task_id in ids}
    plus10_ids = {task_id for ids in plus10.values() for task_id in ids}
    if not plus10_ids <= oracle_ids:
        raise LaunchConfigError(
            f"{partition} +10 artifact contains task IDs outside the Oracle artifact"
        )
    oracle_only = sorted(oracle_ids - plus10_ids)
    summary = {
        "experiment_id": experiment_id,
        "partition": partition,
        "selection": "all_available",
        "rollouts_per_task": rollouts_per_task,
        "task_counts_by_mode": {
            "oracle": len(oracle_ids),
            "plus_10_tools": len(plus10_ids),
        },
        "attempt_counts_by_mode": {
            "oracle": len(oracle_ids) * rollouts_per_task,
            "plus_10_tools": len(plus10_ids) * rollouts_per_task,
        },
        "paired_overlap_tasks": len(plus10_ids),
        "oracle_only_tasks": len(oracle_only),
        "oracle_only_task_ids": oracle_only,
        "task_counts_by_domain_and_mode": {
            domain: {
                "oracle": len(oracle[domain]),
                "plus_10_tools": len(plus10[domain]),
            }
            for domain in sorted(expected_domains)
        },
        "oracle_profile": str(oracle_dir),
        "plus_10_tools_profile": str(plus10_dir),
    }
    (partition_root / "available_mode_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def launch(
    root: Path,
    recipe_path: Path,
    recipe: dict[str, Any],
    *,
    gpu_override: int | None,
    partition_override: str | None,
    dry_run: bool,
) -> int:
    if recipe.get("interface") != "inference":
        raise LaunchConfigError(f"inference interface received recipe for {recipe.get('interface')!r}")
    import sys
    pipeline = recipe.get("pipeline")
    if pipeline == "case_inference":
        from robustops.evaluation.cases import launch as launch_cases
        return launch_cases(root, recipe_path, recipe, gpu_override=gpu_override,
                            partition_override=partition_override, dry_run=dry_run)
    if pipeline == "teacher_fixed_test":
        from robustops.evaluation.teacher_evaluation import launch as launch_teacher
        return launch_teacher(root, recipe_path, recipe, gpu_override=gpu_override,
                              partition_override=partition_override, dry_run=dry_run)
    if pipeline == "fixed_checkpoint_test":
        from robustops.evaluation.checkpoint_evaluation import launch as launch_test
        return launch_test(root, recipe_path, recipe, gpu_override=gpu_override,
                           partition_override=partition_override, dry_run=dry_run)
    if pipeline == "fault_dataset_selection":
        if gpu_override not in (None,0) or partition_override not in (None,'all'):
            raise LaunchConfigError('full fault selector is API-only and covers all configured splits')
        command=[sys.executable,'-m','robustops.data.fault_dataset',
                 '--config',str(resolve_project_path(root,recipe['selector_config']))]
        if dry_run:command.append('--prepare-only')
        return subprocess.run(command,cwd=root,check=False).returncode
    if pipeline == "fault_candidate_selection":
        if gpu_override not in (None,0) or partition_override not in (None,recipe['partition'],'all'):
            raise LaunchConfigError('fault candidate selection is API-only for the configured partition')
        command=[sys.executable,'-m','robustops.data.fault_candidates',
                 '--config',str(resolve_project_path(root,recipe['selector_config'])),
                 '--phase','prepare' if dry_run else 'all']
        return subprocess.run(command,cwd=root,check=False).returncode
    if pipeline == "recovery_teacher_collection":
        if gpu_override not in (None, 0) or partition_override not in (None, recipe.get("partition", "train"), "all"):
            raise LaunchConfigError("Teacher collection is API-only; partition must match recipe")
        command = [sys.executable, "-m", "robustops.evaluation.teacher_rollout", "--config", str(recipe_path)]
        if dry_run:
            command.append("--dry-run")
        return subprocess.run(command, cwd=root, check=False).returncode
    if pipeline == "fault_assignment_evaluation":
        from robustops.launch.fault_assignment import launch_fault_assignment
        return launch_fault_assignment(root, recipe_path, recipe, gpu_override=gpu_override, partition_override=partition_override, dry_run=dry_run)
    if pipeline == "recovery_evaluation":
        from robustops.launch.recovery_evaluation import launch_recovery
        return launch_recovery(root, recipe_path, recipe, gpu_override=gpu_override, partition_override=partition_override, dry_run=dry_run)
    if pipeline not in {
        "paired_evaluation",
        "mode_evaluation",
        "trajectory_regeneration",
    }:
        raise LaunchConfigError(f"unsupported inference pipeline: {pipeline!r}")
    if "data_generation" in recipe:
        raise LaunchConfigError(
            "inference recipes cannot configure model-action interventions"
        )
    execution = recipe.get("execution", PLAN_PRESERVING_EXECUTION)
    if execution != PLAN_PRESERVING_EXECUTION:
        raise LaunchConfigError(
            "inference execution must use the first-tool-only plan-preserving "
            "controller with full assistant planning preserved"
        )
    fault = recipe.get("fault", {"enabled": False})
    fixed_fault = fault.get("enabled") is True and fault.get("kind") == "fixed_policy"
    if fault != {"enabled": False} and not fixed_fault:
        raise LaunchConfigError("inference requires disabled faults or a frozen fixed_policy")

    hardware = require_mapping(recipe, "hardware")
    topology = require_mapping(recipe, "topology")
    backend_config = recipe.get("backend", {"kind": "local_vllm"})
    if not isinstance(backend_config, dict):
        raise LaunchConfigError("backend must be an object")
    model_backend = require_string(backend_config, "kind")
    if model_backend not in {"local_vllm", "openai_compatible_api"}:
        raise LaunchConfigError(f"unsupported inference backend: {model_backend!r}")
    workload = require_mapping(recipe, "workload")
    modes = _string_list(workload.get("modes"), "workload.modes")
    unsupported_modes = sorted(set(modes) - set(EVALUATION_MODES))
    if unsupported_modes:
        raise LaunchConfigError(f"unsupported workload modes: {unsupported_modes}")
    paired_mode_sets = (["oracle", "plus_5_tools"], ["oracle", "plus_10_tools"])
    if pipeline == "paired_evaluation" and modes not in paired_mode_sets:
        raise LaunchConfigError(
            "paired evaluation modes must be ['oracle', 'plus_5_tools'] or "
            "['oracle', 'plus_10_tools']"
        )
    runtime = require_mapping(recipe, "runtime")
    output = require_mapping(recipe, "output")
    replicas = _positive_int(topology.get("replicas"), "topology.replicas")
    concurrency = _positive_int(topology.get("global_concurrency"), "topology.global_concurrency")
    requests_per_replica = _positive_int(
        topology.get("requests_per_replica"),
        "topology.requests_per_replica",
    )
    if concurrency != replicas * requests_per_replica:
        raise LaunchConfigError(
            "topology.global_concurrency must equal replicas * requests_per_replica"
        )
    mcp_server_concurrency_value = topology.get("mcp_server_concurrency", {})
    if not isinstance(mcp_server_concurrency_value, dict) or not all(
        isinstance(server_name, str)
        and server_name
        and isinstance(limit, int)
        and not isinstance(limit, bool)
        and limit > 0
        for server_name, limit in mcp_server_concurrency_value.items()
    ):
        raise LaunchConfigError(
            "topology.mcp_server_concurrency must map server names to positive "
            "integers"
        )
    mcp_server_concurrency = {
        **DEFAULT_MCP_SERVER_CONCURRENCY,
        **mcp_server_concurrency_value,
    }
    if model_backend == "local_vllm":
        gpus = gpu_override or _positive_int(hardware.get("gpus"), "hardware.gpus")
        if (
            replicas != gpus
            or topology.get("replicas_per_gpu") != 1
            or topology.get("tensor_parallel_size") != 1
        ):
            raise LaunchConfigError("local inference requires one TP=1 replica per GPU")
    else:
        if gpu_override is not None:
            raise LaunchConfigError("--gpus is not valid for an external API backend")
        if hardware.get("gpus") != 0:
            raise LaunchConfigError("external API inference requires hardware.gpus=0")
        if replicas != 1:
            raise LaunchConfigError("external API inference requires topology.replicas=1")
        gpus = 0

    recipe_roots = _validate_domain_recipes(root, recipe)
    configured_partitions = workload.get("partitions")
    if not isinstance(configured_partitions, list) or not all(
        item in ("train", "validation", "test") for item in configured_partitions
    ):
        raise LaunchConfigError("workload.partitions must contain train/validation/test values")
    partitions = configured_partitions
    if partition_override and partition_override != "all":
        if partition_override not in configured_partitions:
            raise LaunchConfigError(f"partition is not enabled by recipe: {partition_override}")
        partitions = [partition_override]

    if (
        workload.get("task_ids") is not None
        or workload.get("selection") == "exact_by_partition_and_mode"
    ) and partition_override not in {None, "all"}:
        if partitions != configured_partitions:
            raise LaunchConfigError("partition override cannot change an exact task subset")

    configured_rollouts_by_partition = workload.get("rollouts_per_task_by_partition")
    if configured_rollouts_by_partition is not None:
        if "rollouts_per_task" in workload:
            raise LaunchConfigError(
                "workload must configure either rollouts_per_task or "
                "rollouts_per_task_by_partition, not both"
            )
        if (
            not isinstance(configured_rollouts_by_partition, dict)
            or set(configured_rollouts_by_partition) != set(configured_partitions)
        ):
            raise LaunchConfigError(
                "workload.rollouts_per_task_by_partition keys must equal "
                "workload.partitions"
            )
        rollouts_by_partition = {
            partition: _positive_int(
                configured_rollouts_by_partition[partition],
                f"workload.rollouts_per_task_by_partition.{partition}",
            )
            for partition in partitions
        }
    else:
        rollouts_per_task = _positive_int(
            workload.get("rollouts_per_task", 1),
            "workload.rollouts_per_task",
        )
        rollouts_by_partition = {
            partition: rollouts_per_task for partition in partitions
        }
    unique_rollout_counts = set(rollouts_by_partition.values())
    uniform_rollouts_per_task = (
        next(iter(unique_rollout_counts)) if len(unique_rollout_counts) == 1 else None
    )

    expected_retry = {
        "max_retries_per_logical_attempt": 1,
        "mcp_transport_invalid": True,
        "model_protocol_error": model_backend == "openai_compatible_api",
        "superseded_attempts_preserved": True,
    }
    retry = recipe.get("retry", expected_retry)
    if retry != expected_retry:
        raise LaunchConfigError(
            "inference retry policy must preserve the bounded logical-attempt "
            "and superseded-attempt contract"
        )
    if (
        (workload.get("continuation") is not None or workload.get("plan_preserving") is not None)
        and uniform_rollouts_per_task is None
    ):
        raise LaunchConfigError(
            "continuation and source-branch regeneration require one uniform "
            "rollouts-per-task value"
        )

    model = require_mapping(recipe, "model")
    runtime_override_required = bool(model.get("runtime_override_required", False))
    runtime_model_path = os.environ.get("ROBUSTOPS_MODEL_PATH")
    runtime_model_name = os.environ.get("ROBUSTOPS_MODEL_NAME")
    runtime_model_revision = os.environ.get("ROBUSTOPS_MODEL_REVISION")
    if runtime_override_required and model_backend != "local_vllm":
        raise LaunchConfigError(
            "runtime candidate model overrides are only supported by local_vllm"
        )
    if runtime_override_required:
        missing_overrides = [
            name
            for name, value in (
                ("ROBUSTOPS_MODEL_PATH", runtime_model_path),
                ("ROBUSTOPS_MODEL_NAME", runtime_model_name),
                ("ROBUSTOPS_MODEL_REVISION", runtime_model_revision),
            )
            if not value
        ]
        if missing_overrides:
            raise LaunchConfigError(
                "runtime candidate evaluation requires: " + ", ".join(missing_overrides)
            )
    served_name = (
        runtime_model_name
        if runtime_override_required
        else require_string(model, "served_name")
    )
    model_revision = (
        runtime_model_revision
        if runtime_override_required
        else require_string(model, "revision")
    )
    model_path: Path | None = None
    api_provider: str | None = None
    api_base_url: str | None = None
    api_key_env: str | None = None
    api_token_counter: str | None = None
    api_tokenizer_path: Path | None = None
    api_trust_env = False
    if model_backend == "local_vllm":
        model_path = resolve_model_path(root, require_string(model, "path"))
        if not model_path.is_dir():
            raise LaunchConfigError(f"model path does not exist: {model_path}")
    else:
        if "api_key" in backend_config:
            raise LaunchConfigError(
                "backend.api_key is forbidden; use backend.api_key_env"
            )
        api_provider = require_string(backend_config, "provider")
        if api_provider != "deepseek":
            raise LaunchConfigError(
                "openai_compatible_api currently supports provider='deepseek'"
            )
        api_base_url = require_string(backend_config, "base_url").rstrip("/")
        if not api_base_url.startswith("https://"):
            raise LaunchConfigError("external API backend.base_url must use https://")
        api_key_env = require_string(backend_config, "api_key_env")
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", api_key_env):
            raise LaunchConfigError(
                "backend.api_key_env must be an uppercase environment-variable name"
            )
        api_token_counter = require_string(backend_config, "token_counter")
        if api_token_counter != "local_hf":
            raise LaunchConfigError(
                "DeepSeek API inference requires token_counter='local_hf' so the "
                "Qwen3 32K student boundary is the sole generation gate"
            )
        api_tokenizer_path = resolve_model_path(
            root,
            require_string(backend_config, "tokenizer_path"),
        )
        if not api_tokenizer_path.is_dir():
            raise LaunchConfigError(
                f"API student tokenizer path does not exist: {api_tokenizer_path}"
            )
        api_trust_env_value = backend_config.get("trust_env", False)
        if not isinstance(api_trust_env_value, bool):
            raise LaunchConfigError("backend.trust_env must be a boolean")
        api_trust_env = api_trust_env_value
        if not dry_run and not os.environ.get(api_key_env):
            raise LaunchConfigError(
                f"required API credential environment variable is unset: {api_key_env}"
            )
    runtime_root = Path(
        os.environ.get("ROBUSTOPS_MCP_RUNTIME_ROOT", require_string(runtime, "mcp_root"))
    ).expanduser()
    if not runtime_root.is_dir():
        raise LaunchConfigError(f"MCP runtime path does not exist: {runtime_root}")
    task_manifest = resolve_project_path(root, require_string(workload, "task_manifest"))
    if not task_manifest.is_file():
        raise LaunchConfigError(f"task manifest does not exist: {task_manifest}")
    selected_task_ids_by_mode: dict[str, dict[str, list[str]]] | None = None
    continuation_summary: dict[str, Any] | None = None
    plan_preserving_summary: dict[str, Any] | None = None
    if pipeline == "paired_evaluation":
        if workload.get("continuation") is not None:
            raise LaunchConfigError(
                "workload.continuation is only supported by mode_evaluation"
            )
        expected = workload.get("expected_tasks_per_mode")
        if not isinstance(expected, dict):
            raise LaunchConfigError("workload.expected_tasks_per_mode must be an object")
        domains, task_ids = _validate_workload_scope(
            task_manifest,
            workload,
            partitions,
            modes,
            expected,
        )
        expected_by_mode = {
            mode: {partition: int(expected[partition]) for partition in partitions}
            for mode in modes
        }
    else:
        domains, eligible_by_mode, expected_by_mode = (
            _validate_available_workload_scope(
                root,
                task_manifest,
                workload,
                partitions,
                modes,
            )
        )
        task_ids = None
        if workload.get("selection") in {
            "exact_by_mode",
            "exact_by_partition_and_mode",
        }:
            exact_task_ids_by_mode = {
                partition: {
                    mode: sorted(eligible_by_mode[partition][mode])
                    for mode in modes
                }
                for partition in partitions
            }
            selected_task_ids_by_mode, continuation_summary = (
                _validate_continuation_selection(
                    root,
                    recipe,
                    workload,
                    partitions,
                    modes,
                    eligible_by_mode,
                    uniform_rollouts_per_task or 1,
                )
            )
            if selected_task_ids_by_mode is None:
                selected_task_ids_by_mode = exact_task_ids_by_mode
        else:
            selected_task_ids_by_mode, continuation_summary = (
                _validate_continuation_selection(
                    root,
                    recipe,
                    workload,
                    partitions,
                    modes,
                    eligible_by_mode,
                    uniform_rollouts_per_task or 1,
                )
            )
        if workload.get("plan_preserving") is not None:
            if continuation_summary is not None:
                raise LaunchConfigError(
                    "workload.plan_preserving and workload.continuation are mutually exclusive"
                )
            selected_task_ids_by_mode, plan_preserving_summary = (
                _validate_plan_preserving_selection(
                    root,
                    workload,
                    partitions,
                    modes,
                    eligible_by_mode,
                    uniform_rollouts_per_task or 1,
                )
            )
    run_id = os.environ.get(
        "ROBUSTOPS_EVALUATION_ID",
        f"{output.get('run_id_prefix', 'inference')}-"
        f"{dt.datetime.now(dt.UTC).strftime('%Y%m%dT%H%M%SZ')}-{os.getpid()}",
    )
    output_override = os.environ.get("ROBUSTOPS_EVALUATION_OUTPUT_ROOT")
    output_root = (
        Path(output_override).expanduser()
        if output_override
        else resolve_project_path(root, require_string(output, "root")) / run_id
    )

    backend = root / "examples/enterpriseops/run_inference_profile.sh"
    env_base = os.environ.copy()
    env_base.update(
        {
            "ROBUSTOPS_GPU_COUNT": str(gpus),
            "ROBUSTOPS_MODEL_BACKEND": model_backend,
            "ROBUSTOPS_MODEL_NAME": served_name,
            "ROBUSTOPS_MODEL_REVISION": model_revision,
            "ROBUSTOPS_EXPERIMENT_ID": str(recipe.get("experiment_id", "")),
            "ROBUSTOPS_MCP_RUNTIME_ROOT": str(runtime_root),
            "ROBUSTOPS_TASK_MANIFEST": str(task_manifest),
            "ROBUSTOPS_EVAL_DOMAINS": ",".join(domains),
            "ROBUSTOPS_EVAL_SELECTION": (
                "paired"
                if pipeline == "paired_evaluation"
                or workload.get("selection") == "semantic_paired"
                else "available"
            ),
            "ROBUSTOPS_GLOBAL_CONCURRENCY": str(concurrency),
            "ROBUSTOPS_MCP_SERVER_CONCURRENCY_MAP": json.dumps(
                mcp_server_concurrency,
                sort_keys=True,
                separators=(",", ":"),
            ),
            "ROBUSTOPS_MAX_MODEL_LEN": str(
                require_mapping(require_mapping(recipe, "generation"), "common")["max_context_tokens"]
            ),
            "ROBUSTOPS_RETRY_MCP_TRANSPORT_INVALID": "1",
            "ROBUSTOPS_RETRY_MODEL_PROTOCOL_ERROR": (
                "1" if model_backend == "openai_compatible_api" else "0"
            ),
            "NO_PROXY": "localhost,127.0.0.1,::1",
            "no_proxy": "localhost,127.0.0.1,::1",
        }
    )
    if model_backend == "local_vllm":
        assert model_path is not None
        env_base.update(
            {
                "ROBUSTOPS_MODEL_PATH": str(model_path),
                "ROBUSTOPS_TP_SIZE": str(topology["tensor_parallel_size"]),
                "ROBUSTOPS_GPU_MEMORY_UTILIZATION": str(
                    topology["gpu_memory_utilization"]
                ),
                "ROBUSTOPS_TOKENIZER_MODE": require_string(
                    topology, "tokenizer_mode"
                ),
            }
        )
    else:
        assert api_provider is not None
        assert api_base_url is not None
        assert api_key_env is not None
        assert api_token_counter is not None
        assert api_tokenizer_path is not None
        env_base.update(
            {
                "ROBUSTOPS_MODEL_PROVIDER": api_provider,
                "ROBUSTOPS_MODEL_ENDPOINT": api_base_url,
                "ROBUSTOPS_API_KEY_ENV": api_key_env,
                "ROBUSTOPS_TOKEN_COUNTER": api_token_counter,
                "ROBUSTOPS_TOKENIZER_PATH": str(api_tokenizer_path),
                "ROBUSTOPS_MODEL_TRUST_ENV": "1" if api_trust_env else "0",
            }
        )

    exact_task_ids_file = output_root / "selected_task_ids.txt" if task_ids else None

    plan: list[dict[str, Any]] = []
    for partition in partitions:
        partition_root = output_root / partition if len(partitions) > 1 else output_root
        partition_rollouts = rollouts_by_partition[partition]
        for mode in modes:
            selected_ids = (
                task_ids
                if task_ids is not None
                else (
                    selected_task_ids_by_mode[partition][mode]
                    if selected_task_ids_by_mode is not None
                    else None
                )
            )
            task_ids_file = (
                exact_task_ids_file
                if task_ids is not None
                else (
                    output_root
                    / "selected_task_ids"
                    / f"{partition}__{mode}.txt"
                    if selected_ids is not None
                    else None
                )
            )
            expected_tasks = (
                len(selected_ids)
                if selected_ids is not None
                else expected_by_mode[mode][partition]
            )
            plan_preserving_mode = (
                plan_preserving_summary["by_partition"][partition][mode]
                if plan_preserving_summary is not None
                else None
            )
            stage_domains = (
                _selected_domains_for_task_ids(
                    task_manifest,
                    partition,
                    domains,
                    selected_ids,
                )
                if selected_ids is not None
                else domains
            )
            if pipeline == "paired_evaluation":
                runner_selection = "paired"
                runner_paired_mode: str | None = "plus_10_tools"
            elif workload.get("selection") == "semantic_paired" and mode != "oracle":
                runner_selection = "paired"
                runner_paired_mode = mode
            else:
                # Oracle has no distractor pair mode in the task runner. Exact
                # continuation IDs already freeze its selected subset.
                runner_selection = "available"
                runner_paired_mode = None
            plan.append({
                "partition": partition,
                "mode": mode,
                "expected_tasks": expected_tasks,
                "expected_attempts": (
                    plan_preserving_mode["affected_attempt_count"]
                    if plan_preserving_mode is not None
                    else expected_tasks * partition_rollouts
                ),
                "rollouts_per_task": partition_rollouts,
                "profile_dir": str(partition_root / mode),
                "recipe_root": str(recipe_roots[partition][mode]),
                "backend_kind": BACKEND_KIND_BY_MODE[mode],
                "domains": stage_domains,
                "runner_selection": runner_selection,
                "runner_paired_mode": runner_paired_mode,
                "task_ids": selected_ids,
                "task_ids_file": str(task_ids_file) if task_ids_file is not None else None,
                "plan_preserving_source_attempts": (
                    plan_preserving_mode["source_attempts"]
                    if plan_preserving_mode is not None
                    else None
                ),
                "planned_attempt_indices_by_task": (
                    plan_preserving_mode["attempt_indices_by_task"]
                    if plan_preserving_mode is not None
                    else None
                ),
            })
    resolved = {
        **recipe,
        "launch_recipe": str(recipe_path),
        "resolved_model_backend": model_backend,
        "resolved_model_path": str(model_path) if model_path is not None else None,
        "resolved_model_name": served_name,
        "resolved_model_revision": model_revision,
        "resolved_api_provider": api_provider,
        "resolved_api_base_url": api_base_url,
        "resolved_api_key_env": api_key_env,
        "resolved_api_token_counter": api_token_counter,
        "resolved_api_tokenizer_path": (
            str(api_tokenizer_path) if api_tokenizer_path is not None else None
        ),
        "resolved_retry_policy": retry,
        "resolved_execution_controller": {
            "name": "first_tool_only_plan_preserving_v1",
            "enabled_by_default": True,
            "assistant_content_policy": "preserve_full",
            "tool_call_policy": "execute_first_only_then_reinfer",
        },
        "resolved_mcp_root": str(runtime_root),
        "resolved_task_manifest": str(task_manifest),
        "resolved_output_root": str(output_root),
        "resolved_partitions": partitions,
        "resolved_domains": domains,
        "resolved_task_ids": task_ids,
        "resolved_task_ids_by_mode": selected_task_ids_by_mode,
        "resolved_rollouts_per_task": uniform_rollouts_per_task,
        "resolved_rollouts_per_task_by_partition": rollouts_by_partition,
        "resolved_continuation": continuation_summary,
        "resolved_plan_preserving": plan_preserving_summary,
        "plan": plan,
    }
    logical_attempts = sum(
        expected_by_mode[mode][partition] * rollouts_by_partition[partition]
        for partition in partitions
        for mode in modes
    )
    expected_attempts = sum(stage["expected_attempts"] for stage in plan)
    configured_attempts = workload.get("expected_attempts_total")
    if (
        configured_attempts is not None
        and partitions == configured_partitions
        and configured_attempts != expected_attempts
    ):
        raise LaunchConfigError(
            "workload.expected_attempts_total differs from selected scope: "
            f"expected={configured_attempts}, actual={expected_attempts}"
        )
    configured_logical_attempts = workload.get("expected_logical_attempts_total")
    if (
        configured_logical_attempts is not None
        and partitions == configured_partitions
        and configured_logical_attempts != logical_attempts
    ):
        raise LaunchConfigError(
            "workload.expected_logical_attempts_total differs from selected scope: "
            f"expected={configured_logical_attempts}, actual={logical_attempts}"
        )
    print(f"pipeline={pipeline}")
    print(f"model_backend={model_backend}")
    print(f"hardware_profile={hardware.get('profile')}")
    print(f"gpus={gpus}")
    print(f"replicas={replicas}")
    print(f"requests_per_replica={requests_per_replica}")
    print(f"global_concurrency={concurrency}")
    print("execution_controller=first_tool_only_plan_preserving_v1")
    print("tool_call_policy=execute_first_only_then_reinfer")
    print(
        "mcp_server_concurrency="
        + json.dumps(mcp_server_concurrency, sort_keys=True, separators=(",", ":"))
    )
    print("automatic_retry.mcp_transport_invalid=1")
    print(
        "automatic_retry.model_protocol_error="
        + ("1" if model_backend == "openai_compatible_api" else "0")
    )
    if uniform_rollouts_per_task is not None:
        print(f"rollouts_per_task={uniform_rollouts_per_task}")
    else:
        for partition in partitions:
            print(
                f"rollouts_per_task.{partition}="
                f"{rollouts_by_partition[partition]}"
            )
    print(f"modes={','.join(modes)}")
    print(f"selection={workload.get('selection')}")
    print(f"expected_attempts={expected_attempts}")
    if workload.get("selection") in {
        "exact_by_mode",
        "exact_by_partition_and_mode",
    }:
        for stage in plan:
            print(
                f"selected_tasks.{stage['partition']}.{stage['mode']}="
                f"{stage['expected_tasks']}"
            )
    if continuation_summary is not None:
        print(f"logical_attempts={logical_attempts}")
        print(
            "reused_attempts="
            f"{continuation_summary['attempts_reused_total']}"
        )
        print(
            "superseded_partial_attempts="
            f"{continuation_summary['partial_attempts_superseded_total']}"
        )
        print(
            "superseded_invalid_attempts="
            f"{continuation_summary['invalid_attempts_superseded_total']}"
        )
        for stage in plan:
            print(
                f"generated_tasks.{stage['partition']}.{stage['mode']}="
                f"{stage['expected_tasks']}"
            )
            print(
                f"generated_domains.{stage['partition']}.{stage['mode']}="
                f"{','.join(stage['domains'])}"
            )
            print(
                f"runner_selection.{stage['partition']}.{stage['mode']}="
                f"{stage['runner_selection']}"
            )
        for mode, mode_summary in continuation_summary["by_mode"].items():
            for task_id, attempt_indices in mode_summary[
                "partial_tasks_rerun_attempts"
            ].items():
                print(
                    f"rerun_full_task.{mode}.{task_id}="
                    f"{','.join(map(str, attempt_indices))}"
                )
    if plan_preserving_summary is not None:
        for partition, partition_summary in plan_preserving_summary[
            "by_partition"
        ].items():
            for mode, mode_summary in partition_summary.items():
                print(
                    f"plan_preserving.{partition}.{mode}.source_attempts="
                    f"{mode_summary['source_attempt_count']}"
                )
                print(
                    f"plan_preserving.{partition}.{mode}.affected_tasks="
                    f"{mode_summary['affected_task_count']}"
                )
                print(
                    f"plan_preserving.{partition}.{mode}.affected_attempts="
                    f"{mode_summary['affected_attempt_count']}"
                )
    print(f"temperature={recipe['generation']['common']['temperature']}")
    print(f"model_name={served_name}")
    print(f"model_revision={model_revision}")
    if model_path is not None:
        print(f"model_path={model_path}")
    else:
        print(f"api_provider={api_provider}")
        print(f"api_base_url={api_base_url}")
        print(f"api_key_env={api_key_env}")
        print(f"token_counter={api_token_counter}")
        print(f"tokenizer_path={api_tokenizer_path}")
    print(f"domains={','.join(domains)}")
    if task_ids:
        print(f"task_ids={','.join(task_ids)}")
    print(f"output_root={output_root}")
    if dry_run:
        for stage in plan:
            print(
                "command="
                + shlex.join(["bash", str(backend), stage["backend_kind"]])
                + f"  # {stage['partition']} {stage['mode']}"
            )
        return 0

    output_root.mkdir(parents=True, exist_ok=False)
    if exact_task_ids_file is not None:
        exact_task_ids_file.write_text("\n".join(task_ids or []) + "\n", encoding="utf-8")
    for stage in plan:
        if stage["task_ids_file"] is None or stage["task_ids"] is None:
            continue
        stage_task_ids_file = Path(stage["task_ids_file"])
        if exact_task_ids_file is not None and stage_task_ids_file == exact_task_ids_file:
            continue
        stage_task_ids_file.parent.mkdir(parents=True, exist_ok=True)
        stage_task_ids_file.write_text(
            "\n".join(stage["task_ids"]) + "\n",
            encoding="utf-8",
        )
    (output_root / "resolved_launch_config.json").write_text(
        json.dumps(resolved, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    for stage in plan:
        stage_env = env_base.copy()
        stage_env.pop("ROBUSTOPS_FIXED_FAULT_POLICY", None)
        if fixed_fault:
            stage_env["ROBUSTOPS_FIXED_FAULT_POLICY"] = str(root / fault["policies"][stage["partition"]])
        stage_env.update(
            {
                "ROBUSTOPS_TASK_PARTITION": stage["partition"],
                "ROBUSTOPS_EVAL_PROFILE_ID": f"{run_id}-{stage['partition']}-{stage['mode']}",
                "ROBUSTOPS_EVAL_PROFILE_DIR": stage["profile_dir"],
                "ROBUSTOPS_PROFILE_RECIPE_ROOT": stage["recipe_root"],
                "ROBUSTOPS_EVAL_DOMAINS": ",".join(stage["domains"]),
                "ROBUSTOPS_EVAL_SELECTION": stage["runner_selection"],
                "ROBUSTOPS_ATTEMPTS_PER_TASK": str(stage["rollouts_per_task"]),
            }
        )
        if stage["runner_paired_mode"] is not None:
            stage_env["ROBUSTOPS_EVAL_PAIRED_MODE"] = stage["runner_paired_mode"]
        else:
            stage_env.pop("ROBUSTOPS_EVAL_PAIRED_MODE", None)
        if stage["task_ids_file"] is not None:
            stage_env["ROBUSTOPS_EVAL_TASK_IDS_FILE"] = stage["task_ids_file"]
        if stage["plan_preserving_source_attempts"] is not None:
            stage_env["ROBUSTOPS_PLAN_PRESERVING_SOURCE_ATTEMPTS"] = stage[
                "plan_preserving_source_attempts"
            ]
        else:
            stage_env.pop("ROBUSTOPS_PLAN_PRESERVING_SOURCE_ATTEMPTS", None)
        result = subprocess.run(
            ["bash", str(backend), stage["backend_kind"]],
            cwd=root,
            env=stage_env,
            check=False,
        )
        if result.returncode:
            return result.returncode

        if stage["mode"] == "plus_10_tools" and modes == ["oracle", "plus_10_tools"]:
            partition_root = Path(stage["profile_dir"]).parent
            if pipeline == "paired_evaluation":
                _paired_summary(
                    partition_root / "oracle",
                    partition_root / "plus_10_tools",
                    stage["partition"],
                    partition_root,
                    expected_by_mode["oracle"][stage["partition"]],
                    set(domains),
                    recipe.get("experiment_id"),
                    stage["rollouts_per_task"],
                )
            else:
                _available_mode_summary(
                    partition_root / "oracle",
                    partition_root / "plus_10_tools",
                    stage["partition"],
                    partition_root,
                    expected_by_mode["oracle"][stage["partition"]],
                    expected_by_mode["plus_10_tools"][stage["partition"]],
                    set(domains),
                    recipe.get("experiment_id"),
                    stage["rollouts_per_task"],
                )

    generated_tasks_by_partition = {
        partition: {
            mode: next(
                stage["expected_tasks"]
                for stage in plan
                if stage["partition"] == partition and stage["mode"] == mode
            )
            for mode in modes
        }
        for partition in partitions
    }
    summary = {
        "experiment_id": recipe.get("experiment_id"),
        "hardware_profile": hardware.get("profile"),
        "model_backend": model_backend,
        "model_name": served_name,
        "model_revision": model_revision,
        "pipeline": pipeline,
        "partitions": {
            partition: {
                "tasks_by_mode": {
                    mode: generated_tasks_by_partition[partition][mode]
                    for mode in modes
                },
                "logical_tasks_by_mode": {
                    mode: expected_by_mode[mode][partition]
                    for mode in modes
                },
                "attempts": sum(
                    stage["expected_attempts"]
                    for stage in plan
                    if stage["partition"] == partition
                ),
                "rollouts_per_task": rollouts_by_partition[partition],
            }
            for partition in partitions
        },
        "modes": modes,
        "rollouts_per_task": uniform_rollouts_per_task,
        "rollouts_per_task_by_partition": rollouts_by_partition,
        "attempts_total": expected_attempts,
        "logical_attempts_total": logical_attempts,
        "continuation": continuation_summary,
        "plan_preserving": plan_preserving_summary,
        "selection": (
            "strict_semantic_pair"
            if pipeline == "paired_evaluation"
            else workload.get("selection")
        ),
    }
    (output_root / "evaluation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return 0
