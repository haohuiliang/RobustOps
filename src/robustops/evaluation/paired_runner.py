"""Execute fixed clean/fault groups for local models and API teachers."""

import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
import os

from robustops.evaluation.invalid_resample import resolve_attempt
from robustops.evaluation.fixed_fault_nohit_resample import (
    ROOT,
    SEED_STRIDE,
    read,
    save,
)
from robustops.evaluation.clean_baseline import (
    TokenCounter,
    _tool_content,
    run_task_attempt,
)
from robustops.evaluation.runtime_services import FaultReplica, Model, SERVERS
from robustops.evaluation.selected_fault import SelectedFaultController
from robustops.envs.enterpriseops.task import load_tasks


def validate(a, row):
    if (
        (a["task_id"], a["mode"]) != (row["task_id"], row["mode"])
        or not a["valid"]
        or not a["cleanup"]["succeeded"]
    ):
        raise ValueError(f"Invalid result for cell {row['cell']}")


def metric(a, index, source, temperature):
    from robustops.training.rl_action_feedback import bad_action_cost

    t = a["trajectory"]
    calls = [c["function"] for turn in t["turns"] for c in turn.get("tool_calls", [])]
    repeat = Counter(
        (c["name"], json.dumps(c["arguments"], sort_keys=True)) for c in calls
    )
    return dict(
        index=index,
        source=source,
        success=a["success"],
        temperature=temperature,
        hits=a["fault"].get("injected_calls", 0),
        tools=t["tool_call_count"],
        errors=t["failed_tool_call_count"],
        bad_actions=bad_action_cost(t.get("tool_events", []))["bad_action_count"],
        repeat_max=max(repeat.values(), default=0),
        termination=t["termination"],
        seconds=a.get("duration_seconds"),
        completion_tokens=t.get("token_usage", {}).get(
            "reported_completion_tokens_sum"
        ),
        passed=sum(v["passed"] for v in a["verifiers"]),
        verifiers=len(a["verifiers"]),
    )


def summarize(groups, samples=4):
    result = {}
    for condition in ["clean", "fault"]:
        selected = [g for g in groups.values() if g["condition"] == condition]
        done = [g for g in selected if len(g["attempts"]) == samples]
        all_counts = Counter(sum(a["success"] for a in g["attempts"]) for g in done)
        attempts = [a for g in selected for a in g["attempts"] if a["index"] > 0]
        result[condition] = dict(
            assigned=len(selected),
            completed_groups=len(done),
            skipped_groups=sum(bool(g.get("skipped_samples")) for g in selected),
            resolved_groups=sum(
                len(g["attempts"]) + len(g.get("skipped_samples", [])) == samples
                for g in selected
            ),
            original_t0_success_reference=sum(
                g.get("original", {}).get("success", False) for g in selected
            ),
            **{
                f"pass{samples}_success": sum(n for k, n in all_counts.items() if k > 0)
            },
            success_count_distribution={
                str(i): all_counts[i] for i in range(samples + 1)
            },
            mixed_groups=sum(all_counts[i] for i in range(1, samples)),
            new_attempts=len(attempts),
            new_success=sum(a["success"] for a in attempts),
            actual_hit_attempts=sum(a["hits"] > 0 for a in attempts),
            actual_hit_success=sum(a["hits"] > 0 and a["success"] for a in attempts),
            injected_calls=sum(a["hits"] for a in attempts),
            tools=sum(a["tools"] for a in attempts),
            errors=sum(a["errors"] for a in attempts),
            bad_actions=sum(a["bad_actions"] for a in attempts),
            repeat10=sum(a["repeat_max"] >= 10 for a in attempts),
            budget_terminations=sum(
                a["termination"].startswith(("max_", "context_limit")) for a in attempts
            ),
            attempt_seconds=sum(a["seconds"] or 0 for a in attempts),
            completion_tokens=sum(a["completion_tokens"] or 0 for a in attempts),
        )
        if condition == "fault":
            result[condition].update(
                no_hit_attempts=sum(a["hits"] == 0 for a in attempts),
                no_hit_success=sum(a["hits"] == 0 and a["success"] for a in attempts),
                planned_injected_calls=sum(
                    g["target_fault"]["budget"] * len(g["attempts"]) for g in selected
                ),
                full_dose_attempts=sum(
                    a["hits"] >= g["target_fault"]["budget"]
                    for g in selected
                    for a in g["attempts"]
                ),
                hit_count_distribution={
                    str(i): sum(
                        sum(a["hits"] > 0 for a in g["attempts"]) == i for g in done
                    )
                    for i in range(samples + 1)
                },
            )
    return dict(
        metrics=result,
        groups=list(groups.values()),
        interpretation=(
            "Four fresh temperature-0.6 trajectories per condition; original temperature-0 results excluded from Pass@4. "
            if samples == 4
            else "One temperature-0 trajectory per condition. "
        )
        + "Fault samples are independent full trajectories under fixed Base-kind policy, not shared-prefix recovery suffixes. Actual-hit success is not a clean-solvable recovery rate.",
    )


class PairedModelEvaluation:
    """A model's paired results, independent of which GPU executes each group."""

    def __init__(
        self,
        key,
        originals,
        rows,
        specs,
        policy,
        output,
        temperature,
        sample_seed_indices,
        experiment_id,
        invalid_retries,
        *,
        model_config=None,
        api_key=None,
    ):
        samples = len(sample_seed_indices)
        groups = {}
        for row in rows:
            for condition in ["clean", "fault"]:
                group = dict(
                    cell=row["cell"],
                    task_id=row["task_id"],
                    domain=row["domain"],
                    mode=row["mode"],
                    condition=condition,
                    target_fault=specs[row["cell"]] if condition == "fault" else None,
                    attempts=[],
                )
                if originals is not None:
                    a, source = originals[row["cell"], condition]
                    group["original"] = metric(a, 0, source, 0)
                    save(
                        output
                        / condition
                        / f"cell_{row['cell']:02}"
                        / "original_source.json",
                        group["original"],
                    )
                groups[row["cell"], condition] = group
        tasks = {}
        for row in rows:
            recipe = read(
                ROOT
                / f"recipes/enterpriseops/paired/{'oracle' if row['mode'] == 'oracle' else 'plus5'}/{row['domain']}.json"
            )
            task = next(
                t
                for t in load_tasks(
                    ROOT / recipe["data_path"],
                    expected_domain=row["domain"],
                    expected_count=recipe["expected_task_count"],
                )
                if t.task_id == row["task_id"]
            )
            if (
                task.system_prompt != row["payload"]["task"]["system_policy"]
                or task.user_prompt != row["payload"]["task"]["user_request"]
                or list(task.verifiers) != row["payload"]["verifiers"]
            ):
                raise ValueError(f"Task contract changed: {row['cell']}")
            tasks[row["cell"]] = task
        counter = TokenCounter(ROOT / policy["tokenizer"], enable_thinking=False)

        self.key, self.rows, self.specs, self.policy = key, rows, specs, policy
        self.output, self.temperature = output, temperature
        self.sample_seed_indices, self.experiment_id = (
            sample_seed_indices,
            experiment_id,
        )
        self.invalid_retries = invalid_retries
        self.samples, self.groups, self.tasks, self.counter = (
            samples,
            groups,
            tasks,
            counter,
        )
        self.model_config, self.api_key = model_config, api_key

    async def run_group(self, row, condition, replica):
        key, specs, policy, output = self.key, self.specs, self.policy, self.output
        groups, tasks, counter = self.groups, self.tasks, self.counter
        temperature, sample_seed_indices = self.temperature, self.sample_seed_indices
        samples, experiment_id, invalid_retries = (
            self.samples,
            self.experiment_id,
            self.invalid_retries,
        )
        cell = row["cell"]
        spec = specs[cell]
        for index, seed_index in enumerate(sample_seed_indices, 1):
            # Separate from earlier no-hit resamples; paired across models and conditions.
            seed = row["seed"] + seed_index * SEED_STRIDE
            folder = output / condition / f"cell_{cell:02}" / f"attempt_{index:02}"

            async def generate(folder, seed):
                path = folder / "attempt.json"
                (folder / "calls").mkdir(parents=True)
                controller = (
                    SelectedFaultController(
                        row["payload"],
                        row["selection"],
                        spec["kind"],
                        spec["budget"],
                        "eligible_calls",
                        lambda v: counter.text(_tool_content(v)),
                    )
                    if condition == "fault"
                    else None
                )
                model_config = dict(
                    self.model_config or policy["base_model"],
                    model=replica.served_name,
                    temperature=temperature,
                    endpoint=replica.endpoint,
                )
                model = (
                    Model(model_config, folder / "calls", seed)
                    if self.model_config is None
                    else Model(model_config, folder / "calls", seed, self.api_key)
                )
                try:
                    a = await run_task_attempt(
                        task=tasks[cell],
                        model=model,
                        token_counter=counter,
                        archive_path=ROOT / "enterpriseops_gym/gym_dbs.zip",
                        mcp_url_overrides=replica.urls,
                        max_turns=policy["max_turns"],
                        max_context_tokens=policy["max_context_tokens"],
                        max_generation_tokens=policy["max_generation_tokens"],
                        experiment_id=experiment_id,
                        run_id=output.parent.name,
                        mode=row["mode"],
                        model_name=replica.served_name,
                        model_endpoint=replica.endpoint,
                        seed=seed,
                        attempt_index=index,
                        plan_preserving=True,
                        tool_intervention=controller,
                    )
                    if condition == "fault":
                        hits = a["fault"]["injected_calls"]
                        a["fault"].update(
                            target_mechanism=spec["kind"],
                            selected_mechanism=spec["kind"] if hits else None,
                            no_hit=hits == 0,
                            base_injected_calls=spec["base_injected_calls"],
                        )
                        save(folder / "tool_audit.json", controller.calls)
                    a["sampling"] = dict(
                        seed=seed,
                        temperature=temperature,
                        supplemental_attempt_index=index,
                    )
                    save(path, a)
                finally:
                    await model.client.close()
                return a

            if invalid_retries:
                a, path = await resolve_attempt(
                    folder,
                    seed,
                    row,
                    dict(temperature=temperature, supplemental_attempt_index=index),
                    generate,
                    invalid_retries,
                )
                if a is None:
                    groups[cell, condition].setdefault("skipped_samples", []).append(
                        index
                    )
                    save(output / "summary.json", summarize(groups, samples))
                    print(
                        f"{key} {condition} cell={cell} sample={index} skipped_invalid",
                        flush=True,
                    )
                    continue
            else:
                path = folder / "attempt.json"
                if not path.exists() and folder.exists():
                    folder.rename(
                        folder.with_name(
                            folder.name
                            + f"_incomplete_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}"
                        )
                    )
                a = read(path) if path.exists() else await generate(folder, seed)
                validate(a, row)
                if a["sampling"] != dict(
                    seed=seed, temperature=temperature, supplemental_attempt_index=index
                ):
                    raise ValueError(f"Resume sampling mismatch: {path}")
            groups[cell, condition]["attempts"].append(
                metric(a, index, str(path), temperature)
            )
            save(output / "summary.json", summarize(groups, samples))
            print(
                f"{key} {condition} cell={cell} sample={index}/{samples} success={a['success']} hits={a['fault'].get('injected_calls', 0)}",
                flush=True,
            )

    def finish(self):
        result = summarize(self.groups, self.samples)
        assert all(
            v["resolved_groups"] == len(self.rows) for v in result["metrics"].values()
        )
        save(self.output / "summary.json", result)
        save(self.output / "completed.json", result["metrics"])
        return result["metrics"]


async def run_model(
    key,
    model_path,
    originals,
    rows,
    specs,
    policy,
    output,
    temperature,
    gpu=0,
    sample_seed_indices=(11, 12, 13, 14),
    experiment_id="SFT-EPOCH23-30CELL-SUPPLEMENT",
    invalid_retries=0,
):
    evaluation = PairedModelEvaluation(
        key,
        originals,
        rows,
        specs,
        policy,
        output,
        temperature,
        sample_seed_indices,
        experiment_id,
        invalid_retries,
    )
    runtime = (
        output / f"runtime_{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}-{os.getpid()}"
    )
    replica = FaultReplica(
        gpu, model_path, ROOT / policy["tokenizer"], f"SFT-{key}-supplement", runtime,
        max_context_tokens=policy["max_context_tokens"]
    )
    gate = asyncio.Semaphore(2)

    async def one_group(row, condition):
        async with gate:
            await evaluation.run_group(row, condition, replica)

    try:
        await replica.start({server for server, _ in SERVERS.values()})
        async with asyncio.TaskGroup() as tg:
            for row in rows:
                for condition in ["clean", "fault"]:
                    tg.create_task(one_group(row, condition))
    finally:
        replica.close()
    return evaluation.finish()
