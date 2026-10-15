"""Frozen final experiment, descriptive metrics, and a native live results panel.

No evaluation or sampling is triggered here. Train curves describe changing
tasks and selection; fixed paired Val/Test results are separate evidence.
"""

from collections import Counter
import csv
import json
from pathlib import Path
from statistics import mean, median
import time


def trajectory_metrics(trajectory, inherited_turns, injected_turns):
    """Only new actions; injected failures remain distinguishable from normal errors."""
    events = [e for e in trajectory["tool_events"] if e["turn"] > inherited_turns]
    calls = [
        c["function"]
        for t in trajectory["turns"]
        if t["turn"] > inherited_turns
        for c in t.get("tool_calls", [])
    ]
    repeats = Counter(
        (c["name"], json.dumps(c["arguments"], sort_keys=True)) for c in calls
    )
    maximum = max(repeats.values(), default=0)
    return dict(
        repeat_max=maximum,
        repeat_ge10=int(maximum >= 10),
        tool_error_calls=sum(bool(e["result"].get("is_error")) for e in events),
        noninjected_tool_error_without_status_calls=sum(
            bool(e["result"].get("is_error"))
            and e["result"].get("outcome", {}).get("status_code") is None
            for e in events
            if e["turn"] not in injected_turns
        ),
        noninjected_tool_error_calls=sum(
            bool(e["result"].get("is_error"))
            for e in events
            if e["turn"] not in injected_turns
        ),
        **{
            "termination_" + name: int(trajectory["termination"] == name)
            for name in (
                "model_stop",
                "max_tokens",
                "max_turns",
                "context_limit_pre_request",
            )
        },
    )


def sampling_cost(
    result, bridge, row, *, inherited_turns, flow, donor, member_retry, elapsed
):
    from robustops.evaluation.tool_protocol import model_protocol_retry_reason

    # Infrastructure-invalid results can explicitly contain trajectory=None.
    trajectory = result.get("trajectory") if result is not None else None
    events = trajectory["tool_events"] if trajectory is not None else None
    return dict(
        task_id=row["task_id"],
        mode=row["mode"],
        domain=row["domain"],
        training_source=row.get("training_source", "train"),
        flow=flow,
        donor=donor,
        member_retry=member_retry,
        generated_tokens=sum(bridge.mask),
        tool_calls=sum(e["turn"] > inherited_turns for e in events)
        if events is not None
        else None,
        episode_seconds=elapsed,
        generation_request_seconds=bridge.generated_seconds,
        training_valid=bool(
            result is not None
            and result["valid"]
            and result["cleanup"]["succeeded"]
            and not model_protocol_retry_reason(result)
        ),
        scope="New generation only; inherited prefix excluded; one record per physical attempt",
    )


def final_metric_values(root, step, rows, slots, flows, kinds, report):
    values = {}

    def summarize(prefix, selected):
        values[prefix + "/members"] = len(selected)
        values[prefix + "/successes"] = sum(r["strict_success"] for r in selected)
        if not selected:
            return
        values[prefix + "/success_pct"] = (
            100 * values[prefix + "/successes"] / len(selected)
        )
        for key in ("generated_tokens", "tool_calls", "episode_seconds"):
            sequence = sorted(r[key] for r in selected)
            values[prefix + "/" + key + "_p50"] = median(sequence)
            values[prefix + "/" + key + "_p90"] = sequence[
                max(0, (9 * len(sequence) + 9) // 10 - 1)
            ]
        for key in ("reward_progress_raw", "reward_progress", "final_fraction"):
            values[prefix + "/" + key + "_mean"] = mean(r[key] for r in selected)
        for key in (
            "repeat_ge10",
            "tool_error_calls",
            "noninjected_tool_error_calls",
            "noninjected_tool_error_without_status_calls",
            "bad_action_reference_rejection",
            "local_read_repeat_count",
            "regressed_checks",
            "termination_model_stop",
            "termination_max_tokens",
            "termination_max_turns",
            "termination_context_limit_pre_request",
        ):
            if all(key in r for r in selected):
                values[prefix + "/" + key] = sum(r[key] for r in selected)
        failed = [r for r in selected if not r["strict_success"]]
        values[prefix + "/failed_generated_tokens"] = sum(
            r["generated_tokens"] for r in failed
        )
        values[prefix + "/failed_tool_calls"] = sum(r["tool_calls"] for r in failed)

    windows = dict(
        accepted=list(zip(rows, flows)),
        initial=[(r["metrics"], r["flow"]) for r in report["initial_batch_rows"]],
    )
    for window, pairs in windows.items():
        for flow in ("all", "clean", "fault"):
            selected = [r for r, f in pairs if flow == "all" or f == flow]
            prefix = f"final/{window}/{flow}"
            summarize(prefix, selected)
            if flow == "fault":
                hit = [r for r in selected if r["actual_hit"]]
                summarize(prefix + "/hit", hit)
                values[prefix + "/planned_dose"] = sum(
                    r["planned_dose"] for r in selected
                )
                values[prefix + "/actual_dose"] = sum(
                    r["actual_dose"] for r in selected
                )
                values[prefix + "/full_dose_members"] = sum(
                    r["actual_hit"] and r["actual_dose"] >= r["planned_dose"]
                    for r in selected
                )
    plans = {r["slot"]: r for r in report["actual_task_plan"]}
    for dimension in ("domain", "mode"):
        for label in sorted({p[dimension] for p in plans.values()}):
            summarize(
                f"final/accepted/{dimension}/{label}",
                [r for r, slot in zip(rows, slots) if plans[slot][dimension] == label],
            )
    for kind in sorted({k for k in kinds if k}):
        summarize(
            f"final/accepted/fault_kind/{kind}",
            [r for r, k in zip(rows, kinds) if k == kind],
        )
    values.update(
        {
            "final/sampling/initial_valid_groups": len(report["initial_batch_rows"])
            // 4,
            "final/sampling/initial_invalid_groups": len(
                report["initial_invalid_group_slots"]
            ),
            "final/sampling/initial_all_success_groups": len(
                report["initial_all_success_slots"]
            ),
            "final/sampling/initial_filtered_groups": len(
                report["initial_filtered_slots"]
            ),
            "final/sampling/rejected_all_success_groups": sum(
                r.get("reason") == "all_success" for r in report["signal_replacements"]
            ),
            "final/sampling/rejected_zero_reward_groups": sum(
                r.get("reason") == "zero_reward_before_local_override"
                for r in report["signal_replacements"]
            ),
        }
    )
    # Costs written in the worker's finally block include failed model-format /
    # infrastructure attempts and valid siblings in subsequently dropped groups.
    root = Path(root)
    costs, missing = [], []
    for category in ("train", "donors"):
        for directory in (root / "episodes" / category / f"step_{step}").glob("*"):
            if not directory.is_dir():
                continue
            path = directory / "sampling_cost.json"
            if not path.exists():
                missing.append(str(directory))
                continue
            item = json.loads(path.read_text())
            costs.append(item)
    values["final/cost/recorded_attempts"] = len(costs)
    values["final/cost/missing_attempt_records"] = len(missing)
    values["final/cost/tool_calls_missing_attempts"] = sum(
        c["tool_calls"] is None for c in costs
    )
    for key in (
        "generated_tokens",
        "tool_calls",
        "episode_seconds",
        "generation_request_seconds",
    ):
        values["final/cost/recorded_" + key] = sum(
            c[key] for c in costs if c[key] is not None
        )
    values["final/cost/invalid_or_protocol_attempts"] = sum(
        not c["training_valid"] for c in costs
    )
    values["final/cost/invalid_or_protocol_generated_tokens"] = sum(
        c["generated_tokens"] for c in costs if not c["training_valid"]
    )
    values["final/coverage/attempted_cells"] = len(
        {(c["task_id"], c["mode"]) for c in costs}
    )
    values["final/coverage/attempted_tasks"] = len({c["task_id"] for c in costs})
    values["final/coverage/accepted_cells"] = len(
        {(p["task_id"], p["mode"]) for p in plans.values()}
    )
    return values, dict(
        cost_records=len(costs),
        missing_cost_records=missing,
        attempted_cells=sorted({(c["task_id"], c["mode"]) for c in costs}),
        accepted_cells=sorted({(p["task_id"], p["mode"]) for p in plans.values()}),
    )


# Curated display, while all scalar metrics remain in machine-readable exports.
FINAL_CARDS = [
    ("outcome/initial/clean/success_pct", "outcome/initial/clean/success_pct"),
    ("outcome/initial/fault/success_pct", "outcome/initial/fault/success_pct"),
    ("final/initial/fault/hit/success_pct", "final/initial/fault/hit/success_pct"),
    ("outcome/accepted/clean/success_pct", "outcome/accepted/clean/success_pct"),
    ("outcome/accepted/fault/success_pct", "outcome/accepted/fault/success_pct"),
    ("sampling/initial_valid_members", "sampling/initial_valid_members"),
    ("reward/initial/mean", "reward/initial/mean"),
    ("reward/accepted/mean", "reward/accepted/mean"),
    ("reward/accepted/S_mean", "reward/accepted/S_mean"),
    ("reward/accepted/P_mean", "reward/accepted/P_mean"),
    ("reward/accepted/Q_mean", "reward/accepted/Q_mean"),
    ("actor/loss", "actor/loss"),
    ("optimization/policy_gradient_loss", "actor/pg_loss"),
    ("actor/kl_loss", "actor/kl_loss"),
    ("optimization/gradient_norm", "actor/grad_norm"),
    ("actor/pg_clipfrac", "actor/pg_clipfrac"),
    ("optimization/entropy", "actor/entropy"),
    ("val/sft_loss", "val/sft_loss"),
    (
        "final/sampling/initial_all_success_groups",
        "final/sampling/initial_all_success_groups",
    ),
    (
        "final/sampling/initial_filtered_groups",
        "final/sampling/initial_filtered_groups",
    ),
    (
        "final/sampling/rejected_all_success_groups",
        "final/sampling/rejected_all_success_groups",
    ),
    (
        "final/sampling/rejected_zero_reward_groups",
        "final/sampling/rejected_zero_reward_groups",
    ),
    ("sampling/accepted_group_fraction", "sampling/accepted_group_fraction"),
    ("reward/strict_mixed_groups", "reward/strict_mixed_groups"),
    ("replay/clean/pending", "replay/clean/pending"),
    ("replay/fault/pending", "replay/fault/pending"),
    ("final/accepted/all/repeat_ge10", "final/accepted/all/repeat_ge10"),
    (
        "outcome/accepted/all/budget_terminated_pct",
        "outcome/accepted/all/budget_terminated_pct",
    ),
    ("outcome/accepted/all/bad_call_pct", "outcome/accepted/all/bad_call_pct"),
    (
        "final/accepted/all/bad_action_reference_rejection",
        "final/accepted/all/bad_action_reference_rejection",
    ),
    (
        "sampling/model_protocol_member_failures",
        "sampling/model_protocol_member_failures",
    ),
    ("sampling/invalid_member_failures", "sampling/invalid_member_failures"),
    (
        "final/accepted/all/noninjected_tool_error_without_status_calls",
        "final/accepted/all/noninjected_tool_error_without_status_calls",
    ),
    (
        "sampling/rejected_model_protocol_groups",
        "sampling/rejected_model_protocol_groups",
    ),
    ("sampling/rejected_invalid_groups", "sampling/rejected_invalid_groups"),
    ("sampling/rejected_mixed_retry_groups", "sampling/rejected_mixed_retry_groups"),
    ("final/cost/recorded_generated_tokens", "final/cost/recorded_generated_tokens"),
    (
        "final/cost/invalid_or_protocol_generated_tokens",
        "final/cost/invalid_or_protocol_generated_tokens",
    ),
    ("final/cost/recorded_tool_calls", "final/cost/recorded_tool_calls"),
    ("final/cost/missing_attempt_records", "final/cost/missing_attempt_records"),
    ("timing_s/step", "timing_s/step"),
    (
        "final/cumulative/recorded_generated_tokens",
        "final/cumulative/recorded_generated_tokens",
    ),
    ("final/cumulative/step_seconds", "final/cumulative/step_seconds"),
    ("actor/perf/max_memory_allocated_gb", "actor/perf/max_memory_allocated_gb"),
    ("final/cumulative/attempted_cells", "final/cumulative/attempted_cells"),
    ("final/cumulative/accepted_cells", "final/cumulative/accepted_cells"),
]
_writers = {}


def write_final_metrics(root, step, metrics, evidence, logdir, label):
    """Called by the trainer itself: no separate watcher or historical rescoring."""
    from robustops.training.recovery_rl_core import write_json
    from tensorboard.summary.writer.event_file_writer import EventFileWriter
    from tensorboard.compat.proto import event_pb2, summary_pb2

    root = Path(root)
    output = root / "final"
    target = output / "tensorboard"
    rows = [
        json.loads(p.read_text())
        for p in (output / "steps").glob("*.json")
        if int(p.stem) < step
    ]
    rows.sort(key=lambda r: r["step"])
    cumulative = dict(metrics)
    for title, key in [
        ("recorded_generated_tokens", "final/cost/recorded_generated_tokens"),
        ("recorded_tool_calls", "final/cost/recorded_tool_calls"),
        ("step_seconds", "timing_s/step"),
    ]:
        cumulative["final/cumulative/" + title] = sum(
            r["metrics"].get(key, 0) for r in rows
        ) + metrics.get(key, 0)
    for name in ("attempted_cells", "accepted_cells"):
        cells = {tuple(c) for r in rows for c in r["evidence"][name]}
        cells.update(tuple(c) for c in evidence[name])
        cumulative["final/cumulative/" + name] = len(cells)
    item = dict(
        step=step,
        metrics=cumulative,
        evidence=evidence,
        scope="Committed training steps, changing tasks; initial follows bounded validity retries. Not independent recovery or generalization.",
    )
    write_json(output / "steps" / f"{step}.json", item)
    rows.append(item)
    columns = sorted({k for r in rows for k in r["metrics"]})
    temporary = output / "metrics.csv.writing"
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["step", *columns])
        writer.writeheader()
        writer.writerows(dict(step=r["step"], **r["metrics"]) for r in rows)
    temporary.replace(output / "metrics.csv")
    link = Path(logdir) / ("FINAL-" + label)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink() and link.resolve() != target.resolve():
        raise ValueError("final TensorBoard label belongs to another run")
    target.mkdir(parents=True, exist_ok=True)
    if not link.exists() and not link.is_symlink():
        link.symlink_to(target.resolve(), target_is_directory=True)
    if str(target) not in _writers:
        first = not any(target.glob("events.out.tfevents.*"))
        _writers[str(target)] = EventFileWriter(str(target))
        baseline = root / "val_loss/0.json"
        if first and baseline.exists():
            loss = json.loads(baseline.read_text())["loss"]
            _writers[str(target)].add_event(
                event_pb2.Event(
                    wall_time=time.time(),
                    step=0,
                    summary=summary_pb2.Summary(
                        value=[
                            summary_pb2.Summary.Value(
                                tag="final/optimization/SFT_validation_loss",
                                simple_value=loss,
                            )
                        ]
                    ),
                )
            )
    values = [
        summary_pb2.Summary.Value(
            tag="final/" + title, simple_value=float(cumulative[key])
        )
        for title, key in FINAL_CARDS
        if key in cumulative
    ]
    _writers[str(target)].add_event(
        event_pb2.Event(
            wall_time=time.time(), step=step, summary=summary_pb2.Summary(value=values)
        )
    )
    _writers[str(target)].flush()
