"""Training outcome, reward, sampling and update diagnostics."""

import json
from pathlib import Path
import time
from statistics import mean
import numpy as np
from robustops.training.rl_training_metrics import FINAL_CARDS


def queue_metric_values(queues):
    if not queues:
        return {}
    streams = queues
    return {
        f"sampling/queue/{flow}/{key}": count
        for flow, value in streams.items()
        for key, count in value.items()
    }


def member_retry_metric_values(report):
    """Keep model-format retries distinct from infrastructure and zero-signal groups."""
    events = report.get("member_retry_events", [])
    rejected = report.get("signal_replacements", [])
    counts = {
        kind: sum(r.get("reason") == f"{kind}_member_retry_limit" for r in rejected)
        for kind in ("invalid", "model_protocol", "mixed")
    }
    return {
        "sampling/invalid_member_failures": sum(
            e.get("failure_kind", "infrastructure_invalid") == "infrastructure_invalid"
            for e in events
        ),
        "sampling/model_protocol_member_failures": sum(
            e.get("failure_kind") == "model_protocol_error" for e in events
        ),
        "sampling/member_resamples": sum(e["action"] == "retry_member" for e in events),
        "sampling/model_protocol_member_resamples": sum(
            e["action"] == "retry_member"
            and e.get("failure_kind") == "model_protocol_error"
            for e in events
        ),
        "sampling/rejected_invalid_groups": counts["invalid"],
        "sampling/rejected_model_protocol_groups": counts["model_protocol"],
        "sampling/rejected_mixed_retry_groups": counts["mixed"],
        "sampling/rejected_zero_pg_groups": len(rejected) - sum(counts.values()),
    }


_writers = {}


def json_metric_values(metrics):
    """Convert NumPy metric scalars to the built-in JSON number types."""
    return {
        key: value.item() if isinstance(value, np.generic) else value
        for key, value in metrics.items()
    }


def reward_metric_values(rows, flows, weights, initial=None):
    """Trajectory means of configured raw S/P/Q[/L] reward, before normalization.

    Local bad-token overrides and KL are loss terms, not extra scalar reward.
    """

    def average(selected):
        return sum(
            sum(
                w * r[k]
                for w, k in zip(
                    weights, ("reward_success", "reward_progress", "reward_quality")
                )
            )
            for r in selected
        ) / len(selected)

    result = {"reward/accepted/mean": average(rows)}
    for label, key in [
        ("S", "reward_success"),
        ("P", "reward_progress"),
        ("Q", "reward_quality"),
    ][: len(weights)]:
        result[f"reward/accepted/{label}_mean"] = sum(r[key] for r in rows) / len(rows)
    for flow in ("clean", "fault", "correction"):
        selected = [r for r, f in zip(rows, flows) if f == flow]
        if selected:
            result[f"reward/{flow}/mean"] = average(selected)
    if initial:
        result["reward/initial/mean"] = average([r["metrics"] for r in initial])
    return result


def record_metrics(batch, existing, step, cfg):
    from robustops.training.recovery_rl_core import write_json

    rows = list(batch.non_tensor_batch["recovery_metrics"])
    slots = list(batch.non_tensor_batch["recovery_joint_slot"])
    flows = list(batch.non_tensor_batch["recovery_flow"])
    root = Path(cfg.output)
    report = json.loads((root / "diagnostics" / f"{step}.json").read_text())
    clean_groups, fault_groups, _ = report["planned_group_counts"]
    fault_end = clean_groups + fault_groups
    metrics = {"reward/strict_mixed_groups": report["strict_mixed_groups"]}
    metrics["local_read_repeat/calls"] = sum(
        r.get("local_read_repeat_count", 0) for r in rows
    )
    metrics["terminal_stop/review_candidates"] = sum(
        r.get("terminal_stop_review_candidate", 0) for r in rows
    )
    if "initial_zero_pg_without_repeat" in report:
        metrics["local_read_repeat/initial_zero_to_nonzero_groups"] = len(
            set(report["initial_zero_pg_without_repeat"])
            - set(report["initial_zero_pg_slots"])
        )
    for window, value in report.get("replacement_epochs", {}).items():
        metrics[f"sampling/reserve_window_{window}/epoch"] = value["epoch"]
        metrics[f"sampling/reserve_window_{window}/remaining"] = len(value["pending"])
    metrics.update(queue_metric_values(report.get("task_queues")))
    for flow in sorted(set(flows)):
        members = [r for r, f in zip(rows, flows) if f == flow]
        metrics[f"local_read_repeat/{flow}/calls"] = sum(
            r.get("local_read_repeat_count", 0) for r in members
        )
        metrics[f"terminal_stop/{flow}/review_candidates"] = sum(
            r.get("terminal_stop_review_candidate", 0) for r in members
        )
    metrics.update(
        reward_metric_values(
            rows, flows, list(cfg.reward_weights), report.get("initial_batch_rows")
        )
    )
    metrics["reward/original_fault_success_pct"] = (
        100
        * sum(
            r["strict_success"]
            for i, r in enumerate(rows)
            if clean_groups <= slots[i] < fault_end
        )
        / (4 * fault_groups)
    )
    for flow in sorted(set(flows)):
        metrics[f"reward/allocation/{flow}"] = report["actual_allocation"].get(flow, 0)
    fault = [r for i, r in enumerate(rows) if flows[i] == "fault"]
    dose = sum(r["planned_dose"] for r in fault)
    if dose:
        metrics["reward/fault_dose_ratio"] = sum(r["actual_dose"] for r in fault) / dose
    metrics["reward/planned_fault_groups"] = report["target_allocation"]["fault"]
    metrics["reward/actual_fault_groups"] = len(fault) // 4
    metrics["reward/fault_fallback_groups"] = len(report["fallbacks"])
    for kind, count in report["fault_balance"]["labels"].items():
        metrics["reward/fault_kind_cumulative/" + kind] = count
    mask = batch.batch["response_mask"].bool()
    advantage = batch.batch["advantages"]
    zero_slots = {
        s
        for s in set(slots)
        if not advantage[[i for i, v in enumerate(slots) if v == s]].any()
    }
    from robustops.training.rl_credit import flow_signal_summary

    metrics.update(
        flow_signal_summary(
            rows,
            slots,
            flows,
            list(batch.non_tensor_batch["recovery_parent_visit"]),
            zero_slots,
            original_groups=32,
        )
    )
    donors = report["donor_rows"]
    metrics["recovery/planned_fault_donors"] = fault_groups
    metrics["preparation/actual_donors"] = len(donors)
    metrics["preparation/replacement_donors"] = len(donors) - fault_groups
    metrics["recovery/fault_replaced_slots"] = len(
        {r["group_slot"] for r in report["fault_replacements"]}
    )
    metrics["recovery/donors_without_opportunity"] = sum(
        not d["artifacts"]["opportunities"] for d in donors
    )
    metrics["recovery/donors_without_opportunity_failed"] = sum(
        not d["artifacts"]["opportunities"] and not d["metrics"]["strict_success"]
        for d in donors
    )
    metrics["recovery/donors_without_observed_opportunity"] = sum(
        not d["artifacts"]["observed_opportunity_kinds"] for d in donors
    )
    metrics["recovery/donors_with_budget_excluded_opportunity"] = sum(
        bool(d["artifacts"]["opportunity_exclusions"]) for d in donors
    )
    for key in (
        "generated_tokens",
        "tool_calls",
        "episode_seconds",
        "generation_request_seconds",
    ):
        metrics["preparation/donor_" + key] = sum(d["metrics"][key] for d in donors)
    discarded = report["discarded_rows"]
    metrics["sampling/initial_zero_pg_groups"] = len(report["initial_zero_pg_slots"])
    if "initial_zero_reward_slots" in report:
        metrics["sampling/initial_zero_reward_groups"] = len(
            report["initial_zero_reward_slots"]
        )
        metrics["sampling/initial_local_only_groups"] = len(
            set(report["initial_zero_reward_slots"])
            - set(report["initial_zero_pg_slots"])
        )
        metrics["sampling/rejected_zero_reward_groups"] = sum(
            r.get("reason") == "zero_reward_before_local_override"
            for r in report["signal_replacements"]
        )
    metrics.update(member_retry_metric_values(report))
    metrics["sampling/attempted_groups"] = 32 + len(report["signal_replacements"])
    metrics["sampling/accepted_group_fraction"] = (
        32 / metrics["sampling/attempted_groups"]
    )
    if "zero_pg_replay" in report:
        replay = report["zero_pg_replay"]
        attempts = replay["attempts"]
        metrics["replay/attempted_groups"] = len(attempts)
        metrics["replay/accepted_groups"] = sum(
            r["status"] == "accepted" for r in attempts
        )
        metrics["replay/zero_groups"] = sum(r["status"] == "zero_pg" for r in attempts)
        metrics["replay/generated_tokens"] = sum(
            r["generated_tokens"] for r in attempts
        )
        if attempts:
            metrics["replay/accepted_fraction"] = metrics[
                "replay/accepted_groups"
            ] / len(attempts)
        for flow in ("clean", "fault"):
            for key, value in replay["queue"][flow].items():
                metrics[f"replay/{flow}/{key}"] = value
            metrics[f"replay/{flow}/attempted_groups"] = sum(
                r["flow"] == flow for r in attempts
            )
            metrics[f"replay/{flow}/accepted_groups"] = sum(
                r["flow"] == flow and r["status"] == "accepted" for r in attempts
            )
    for key in (
        "generated_tokens",
        "tool_calls",
        "episode_seconds",
        "generation_request_seconds",
    ):
        metrics["sampling/discarded_" + key] = sum(d["metrics"][key] for d in discarded)
    initial = report["initial_batch_rows"]
    metrics["sampling/initial_valid_members"] = len(initial)
    if initial:
        metrics["sampling/initial_strict_success_pct"] = (
            100 * sum(d["metrics"]["strict_success"] for d in initial) / len(initial)
        )
    metrics["cost/all_generated_tokens"] = (
        metrics["preparation/donor_generated_tokens"]
        + metrics["sampling/discarded_generated_tokens"]
        + sum(r["generated_tokens"] for r in rows)
    )
    metrics["coverage/main_cells"] = len(
        {(r["task_id"], r["mode"]) for r in report["actual_task_plan"]}
    )
    successful_donors = {
        d["parent_slot"]
        for d in donors
        if d["accepted_for_training"] and d["metrics"]["strict_success"]
    }
    eligible = [
        r
        for i, r in enumerate(rows)
        if slots[i] in successful_donors and flows[i] == "fault" and r["actual_hit"]
    ]
    metrics["preparation/successful_donors"] = len(successful_donors)
    metrics["preparation/successful_donor_hit_members"] = len(eligible)
    metrics["preparation/successful_donor_hit_successes"] = sum(
        r["strict_success"] for r in eligible
    )
    # A training diagnostic, not the fixed-Val recovery rate.
    for key in ("domain", "mode", "planned_flow"):
        from collections import Counter

        for name, count in Counter(r[key] for r in report["actual_task_plan"]).items():
            metrics[f"coverage/{key}/{name}"] = count
    metrics["reward/zero_pg_generated_tokens"] = sum(
        r["generated_tokens"] for i, r in enumerate(rows) if slots[i] in zero_slots
    )
    metrics["reward/zero_pg_episode_seconds"] = sum(
        r["episode_seconds"] for i, r in enumerate(rows) if slots[i] in zero_slots
    )
    metrics["reward/zero_pg_token_fraction"] = metrics[
        "reward/zero_pg_generated_tokens"
    ] / int(mask.sum())
    rollout_seconds = existing.get("timing_s/gen", 0)
    if rollout_seconds:
        metrics["reward/nonzero_pg_groups_per_minute"] = (
            60 * (32 - len(zero_slots)) / rollout_seconds
        )
    from robustops.training.rl_credit import credit_accounting

    bad_tokens = batch.batch["recovery_bad_mask"].sum(-1).tolist()
    credit_metrics, credit_rows = credit_accounting(
        rows,
        slots,
        flows,
        bad_tokens,
        cfg,
        segments=batch.meta_info.get("recovery_credit_segments"),
    )
    metrics.update(credit_metrics)
    import math

    for category in ("mixed_success", "failed_progress", "success_quality", "other"):
        selected = []
        for slot in sorted(set(slots)):
            indices = [i for i, s in enumerate(slots) if s == slot]
            successes = sum(rows[i]["strict_success"] for i in indices)
            cat = (
                "mixed_success"
                if 0 < successes < 4
                else "failed_progress"
                if successes == 0
                and len({rows[i]["reward_progress"] for i in indices}) > 1
                else "success_quality"
                if successes == 4
                and len({rows[i]["reward_quality"] for i in indices}) > 1
                else "other"
            )
            if cat == category:
                selected.extend(indices)
        if selected:
            tokens = sum(rows[i]["generated_tokens"] for i in selected)
            square = sum(credit_rows[i]["final_square_sum"] for i in selected)
            metrics[f"normalization/{category}/final_token_rms"] = math.sqrt(
                square / tokens
            )
            metrics[f"normalization/{category}/final_coefficient_mass"] = sum(
                credit_rows[i]["coefficient_mass"]["final_abs"] for i in selected
            )
    metrics.update(outcome_metrics(rows, flows, report.get("initial_batch_rows")))
    write_json(
        root / "credit" / f"{step}.json",
        dict(
            step=step,
            rows=credit_rows,
            metrics=credit_metrics,
            scope="Flow-weighted token coefficients; not gradient attribution.",
        ),
    )
    final_evidence = None
    from robustops.training.rl_training_metrics import final_metric_values

    extra, final_evidence = final_metric_values(
        root,
        step,
        rows,
        slots,
        flows,
        list(batch.non_tensor_batch["recovery_kind"]),
        report,
    )
    metrics.update(extra)
    complete = dict(existing, **metrics)
    write_json(
        root / "metrics" / f"{step}.json",
        dict(
            step=step,
            metrics=json_metric_values(complete),
            scope="changing training tasks; success retention/generalization require fixed paired evaluation",
        ),
    )
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto import event_pb2, summary_pb2

    if (
        getattr(cfg, "metrics_start_step", None) is not None
        and step <= cfg.metrics_start_step
    ):
        raise ValueError("continuation training metric precedes its first new update")
    target = root / "focus"
    first_event_file = False
    if str(target) not in _writers:
        first_event_file = not any(target.glob("events.out.tfevents.*"))
        _writers[str(target)] = EventFileWriter(str(target))
        logdir = Path(cfg.focus_logdir)
        logdir.mkdir(parents=True, exist_ok=True)
        label = "grpo-" + cfg.reward_normalization
        if getattr(cfg, "focus_run_label", None):
            label += "-" + cfg.focus_run_label
        full_name = f"FULL-{label}"
        for name, path in [
            (f"FOCUS-{label}", target),
            (full_name, root / "tensorboard"),
        ]:
            link = logdir / name
            if link.is_symlink() and link.resolve() != path.resolve():
                if getattr(cfg, "metrics_start_step", None) is not None:
                    raise ValueError(
                        "isolated continuation TensorBoard label already points to another run"
                    )
                raise ValueError("TensorBoard label already belongs to another run")
            if not link.exists() and not link.is_symlink():
                link.symlink_to(path.resolve(), target_is_directory=True)
    cards = [(title, key, 1) for title, key in FINAL_CARDS]
    if first_event_file:
        start = getattr(cfg, "metrics_start_step", None)
        baseline = focus_baseline(root, start)
        if baseline:
            baseline_values = [
                summary_pb2.Summary.Value(
                    tag="focus/" + title, simple_value=float(baseline[key] * scale)
                )
                for title, key, scale in cards
                if key in baseline
            ]
            _writers[str(target)].add_event(
                event_pb2.Event(
                    wall_time=time.time(),
                    step=start,
                    summary=summary_pb2.Summary(value=baseline_values),
                )
            )
    values = [
        summary_pb2.Summary.Value(
            tag="focus/" + title, simple_value=float(complete[key] * scale)
        )
        for title, key, scale in cards
        if key in complete
    ]
    writer = _writers[str(target)]
    writer.add_event(
        event_pb2.Event(
            wall_time=time.time(), step=step, summary=summary_pb2.Summary(value=values)
        )
    )
    writer.flush()
    if final_evidence is not None:
        from robustops.training.rl_training_metrics import write_final_metrics

        write_final_metrics(
            root,
            step,
            json_metric_values(complete),
            final_evidence,
            cfg.focus_logdir,
            cfg.focus_run_label,
        )
    return metrics


def focus_baseline(root, start):
    """Only new-run validation measured at the restored checkpoint; no old curves."""
    if start is None:
        return {}
    path = Path(root) / "val_loss" / f"{start}.json"
    if not path.is_file():
        return {}
    row = json.loads(path.read_text())
    if row["step"] != start:
        raise ValueError("continuation validation baseline step mismatch")
    return {"val/sft_loss": row["loss"]}


def outcome_metrics(rows, flows, initial):
    """Explicit denominators; accepted and first attempts are separate series."""
    values = {}
    windows = dict(accepted=list(zip(rows, flows)))
    if initial:
        windows["initial"] = [(r["metrics"], r["flow"]) for r in initial]
    for window, pairs in windows.items():
        for flow in ("all", "clean", "fault"):
            subset = [r for r, f in pairs if flow == "all" or f == flow]
            if not subset:
                continue
            prefix = f"outcome/{window}/{flow}/"
            values[prefix + "members"] = len(subset)
            values[prefix + "successes"] = sum(r["strict_success"] for r in subset)
            values[prefix + "success_pct"] = (
                100 * values[prefix + "successes"] / len(subset)
            )
            values[prefix + "progress_mean"] = mean(
                r["reward_progress"] for r in subset
            )
            failed = [r for r in subset if not r["strict_success"]]
            values[prefix + "failed_members"] = len(failed)
            if failed:
                values[prefix + "failed_progress_mean"] = mean(
                    r["reward_progress"] for r in failed
                )
            for key in (
                "generated_tokens",
                "tool_calls",
                "bad_action_count",
                "budget_terminated",
            ):
                values[prefix + key + "_sum"] = sum(r[key] for r in subset)
            values[prefix + "generated_tokens_mean"] = values[
                prefix + "generated_tokens_sum"
            ] / len(subset)
            values[prefix + "budget_terminated_pct"] = (
                100 * values[prefix + "budget_terminated_sum"] / len(subset)
            )
            if values[prefix + "tool_calls_sum"]:
                values[prefix + "bad_call_pct"] = (
                    100
                    * values[prefix + "bad_action_count_sum"]
                    / values[prefix + "tool_calls_sum"]
                )
            # Retain the former per-trajectory average separately for comparison.
            values[prefix + "bad_ratio_trajectory_mean"] = mean(
                r["reward_bad_ratio"] for r in subset
            )
    return values
