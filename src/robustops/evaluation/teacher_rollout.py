"""Collect guided teacher trajectories under the shared clean/fault protocol."""

import argparse, asyncio, json, subprocess, sys
from datetime import datetime, timezone
from collections import Counter
from contextlib import AsyncExitStack
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from robustops.providers.credentials import credential
from robustops.data.recovery_sampling import (
    teacher_guidance,
    StudentContextCounter,
    require_sft_member,
)
from robustops.data.teacher_sft import (
    student_record,
    student_token_proof,
    student_export_allowed,
)
from robustops.evaluation.runtime_services import Model, launch, stop_processes
from robustops.evaluation.fault_schedule import (
    FaultScheduler,
    load_schedule,
    observation_budgets,
    select_clean_branches,
)
from robustops.evaluation.fault_scheduling_controller import (
    RuntimePriorityFault,
    run_with_nohit_replay,
)
from robustops.evaluation.clean_fault_branch import run_clean_branch, first_fault_state
from robustops.evaluation.selected_fault import SelectedFaultController
from robustops.evaluation.clean_baseline import (
    TokenCounter,
    run_task_attempt,
    _tool_content,
)
from robustops.envs.enterpriseops.task import load_tasks

ROOT = Path(__file__).resolve().parents[3]


def read(p):
    return json.loads(Path(p).read_text())


def save(p, x):
    Path(p).write_text(json.dumps(x, ensure_ascii=False, indent=2) + "\n")


def isolated_infrastructure_failure(a):
    return (
        not a["valid"]
        and a["cleanup"]["succeeded"]
        and (a.get("error") or {}).get("type") == "InfrastructureError"
    )


def retry_balance(rows, seed, source_summary, restored):
    """Keep successful T0 fault cases; count actual hits in the current retry run."""
    balance = FaultScheduler(rows, seed)
    for prior in source_summary:
        if prior["condition"] == "fault" and prior["success"] and prior["hits"] > 0:
            balance.claim(prior["mechanism"], str(prior["cell"]))
    for entry in restored:
        if entry["condition"] == "fault" and entry["hits"] > 0:
            balance.claim(entry["mechanism"], str(entry["cell"]))
    return balance


async def run(args):
    c = read(args.config)
    if c["pipeline"] != "recovery_teacher_collection":
        raise ValueError("Expected recovery_teacher_collection pipeline")
    if not isinstance(c.get("provider"), dict):
        raise ValueError("teacher collection requires a provider object")
    rows = read(ROOT / c["tasks"])
    pool = read(ROOT / c["pool"])
    allowed = set(c["train_faults"])
    assert c["temperature"] in (0, 0.6)
    if c["temperature"] != 0:
        assert c.get("focused_conditions") and c.get("source_run")
    assert len(rows) == c["expected_task_modes"] and len(
        {r["task_id"] for r in rows}
    ) == c["expected_tasks"]
    assert len({(r["task_id"], r["mode"]) for r in rows}) == len(rows)
    validation = c.get("partition", "train") == "validation"
    assert c.get("partition", "train") in ("train", "validation")
    assert not allowed & set(c["heldout_faults"])
    if not validation:
        assert "ignored_filter" not in allowed
    else:
        assert c.get("focused_conditions") or c.get("selector_policy"), (
            "Validation requires fixed conditions or frozen policy"
        )
        split_rows = [
            json.loads(line)
            for line in (ROOT / "data/task_sets/enterpriseops.jsonl")
            .read_text()
            .splitlines()
        ]
        val_pairs = {
            (t["task_id"], m)
            for t in split_rows
            if t["partition"] == "validation"
            for m in t["runnable_modes"]
        }
        assert {(r["task_id"], r["mode"]) for r in rows} <= val_pairs
    if not validation:
        expected = {
            (t["task_id"], m)
            for t in pool["tasks"]
            if t["pool"] == "sft"
            for m in t["modes"]
            if m in c["modes"]
        }
        assert {(r["task_id"], r["mode"]) for r in rows} == expected
    for r in rows:
        if not validation:
            require_sft_member(pool, r["task_id"], r["mode"])
        assert all(set(t["faults"]) <= allowed for t in r["selection"]["tools"])
    conditions = c.get("conditions", ["clean", "fault"])
    assert (
        conditions
        and set(conditions) <= {"clean", "fault"}
        and len(conditions) == len(set(conditions))
    )
    if not c.get("focused_conditions") and conditions != ["clean", "fault"]:
        raise ValueError(
            "Paired collection must finish all clean cases before fault branches"
        )
    if validation and not c.get("focused_conditions"):
        policy = read(ROOT / c["selector_policy"])
        frozen_schedule = load_schedule(ROOT / c["schedule"])
        policy_rows = {r["cell"]: r for r in policy["rows"]}
        assert all(r == policy_rows[r["cell"]] for r in rows)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    cells=len(rows),
                    episodes=len(c["focused_conditions"])
                    if c.get("focused_conditions")
                    else len(conditions) * len(rows),
                    mcp_port_offset=c.get("mcp_port_offset", 1000),
                    domains=dict(Counter(r["domain"] for r in rows)),
                    modes=dict(Counter(r["mode"] for r in rows)),
                    allowed=sorted(allowed),
                ),
                indent=2,
            )
        )
        return
    from robustops.launch.hardware import visible_gpus

    devices = [gpu.__dict__ for gpu in visible_gpus()]
    from robustops.launch.resources import check_host_memory
    memory = check_host_memory(c.get("hardware", {}).get("min_host_memory_gib", 0))
    out = ROOT / c["output"]
    resuming = c.get("resume", False)
    restored = []
    completed = set()
    terminal_invalid = []
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    if resuming:
        previous = read(out / "config.json")
        assert {k: v for k, v in c.items() if k != "resume"} == {
            k: v for k, v in previous.items() if k != "resume"
        }, "resume config differs"
        assert read(out / "tasks.json") == rows, "resume tasks differ"
        restored = read(out / "summary.json")
        for entry in restored:
            a = read(
                out / entry["condition"] / f"cell_{entry['cell']:02d}" / "attempt.json"
            )
            assert a["valid"] and a["cleanup"]["succeeded"], (
                "summary contains invalid result"
            )
            completed.add((entry["condition"], entry["cell"]))
        # Finished infrastructure-invalid conditions are terminal, not model failures.
        for row in rows:
            for condition in ["clean", "fault"]:
                path = out / condition / f"cell_{row['cell']:02d}" / "attempt.json"
                if (condition, row["cell"]) in completed or not path.exists():
                    continue
                a = read(path)
                if isolated_infrastructure_failure(a):
                    terminal_invalid.append(
                        dict(cell=row["cell"], condition=condition, error=a["error"])
                    )
                    completed.add((condition, row["cell"]))
        save(out / "infrastructure_invalid.json", terminal_invalid)
        evidence = out / "continuations" / stamp
        evidence.mkdir(parents=True)
        save(evidence / "config.json", c)
    else:
        out.mkdir(parents=True, exist_ok=False)
        save(out / "config.json", c)
        save(out / "tasks.json", rows)
        evidence = out

    guidance = teacher_guidance(c["guidance_variant"])
    if resuming:
        assert (out / "teacher_prompt.txt").read_text() == guidance, (
            "resume prompt differs"
        )
    (evidence / "teacher_prompt.txt").write_text(guidance)
    save(
        evidence / "environment.json",
        dict(
            devices=devices,
            host_memory_bytes=memory,
            command=sys.argv,
            code_revision=subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, capture_output=True
            ).stdout.strip()
            or "source-bundle",
            model_alias=c["model"],
            model_immutable_revision="not supplied by provider",
            dataset_manifest=read(ROOT / "data/task_sets/enterpriseops.json"),
        ),
    )
    (evidence / "code.patch").write_text(
        subprocess.run(["git", "diff"], cwd=ROOT, text=True, capture_output=True).stdout
    )
    for source in [
        Path(__file__),
        ROOT / "src/robustops/evaluation/runtime_services.py",
        ROOT / "src/robustops/evaluation/fault_scheduling_controller.py",
        ROOT / "src/robustops/evaluation/selected_fault.py",
        ROOT / "src/robustops/data/recovery_sampling.py",
        ROOT / "src/robustops/data/teacher_sft.py",
    ]:
        dst = evidence / "code" / source.relative_to(ROOT)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(source.read_bytes())
    counter = TokenCounter(ROOT / c["tokenizer"], enable_thinking=False)
    tasks = {}
    cache = {}
    for r in rows:
        key = (r["domain"], r["mode"])
        if key not in cache:
            rc = read(
                ROOT
                / f"recipes/enterpriseops/paired/{'oracle' if r['mode'] == 'oracle' else 'plus5'}/{r['domain']}.json"
            )
            cache[key] = {
                t.task_id: t
                for t in load_tasks(
                    ROOT / rc["data_path"],
                    expected_domain=r["domain"],
                    expected_count=rc["expected_task_count"],
                )
            }
        t = cache[key][r["task_id"]]
        assert (
            t.system_prompt == r["payload"]["task"]["system_policy"]
            and t.user_prompt == r["payload"]["task"]["user_request"]
            and list(t.verifiers) == r["payload"]["verifiers"]
        )
        tasks[r["cell"]] = t
    key = credential(c["provider"])
    schedule = load_schedule(ROOT / c["schedule"])
    save(out / "schedule.json", schedule)
    limits = (
        c.get("mcp_server_concurrency")
        or read(ROOT / c["topology"])["topology"]["mcp_server_concurrency"]
    )
    gate = asyncio.Semaphore(c["concurrency"])
    caps = {k: asyncio.Semaphore(v) for k, v in limits.items()}
    branch_plans = {}
    results = restored.copy()
    clean = {
        x["cell"]: {"success": x["success"]}
        for x in restored
        if x["condition"] == "clean"
    }
    procs = []
    runtime = evidence / "runtime"
    runtime.mkdir()
    balance = None
    budgets = None
    invalid = terminal_invalid.copy()
    task_locks = {r["task_id"]: asyncio.Lock() for r in rows}

    async def episode(r, condition):
        if (condition, r["cell"]) in completed:
            return
        async with task_locks[r["task_id"]], gate, AsyncExitStack() as stack:
            original = tasks[r["cell"]]
            for name in sorted({s["mcp_server_name"] for s in original.gym_servers}):
                await stack.enter_async_context(caps[name])
            path = out / condition / f"cell_{r['cell']:02}"
            if path.exists():
                assert resuming
                archived = evidence / "interrupted" / condition
                archived.mkdir(parents=True, exist_ok=True)
                path.rename(archived / path.name)
            path.mkdir(parents=True)
            (path / "calls").mkdir()
            m = Model(dict(c, provider="deepseek"), path / "calls", None, key)
            task = replace(
                original,
                system_prompt=original.system_prompt
                + ("\n\n" + guidance if guidance else ""),
            )
            if c.get("initial_states"):
                state = deepcopy(read(ROOT / c["initial_states"][str(r["cell"])]))
            else:
                state = (
                    deepcopy(
                        read(
                            ROOT
                            / c["source_run"]
                            / f"clean/cell_{r['cell']:02d}/initial_state.json"
                        )
                    )
                    if c.get("focused_conditions")
                    else (
                        {}
                        if condition == "clean"
                        else read(
                            out / f"clean/cell_{r['cell']:02d}/initial_state.json"
                        )
                    )
                )
            kw = dict(
                task=task,
                token_counter=StudentContextCounter(
                    counter, original.system_prompt, guidance
                ),
                archive_path=ROOT / "enterpriseops_gym/gym_dbs.zip",
                mcp_url_overrides=urls,
                max_turns=c["max_turns"],
                max_context_tokens=c["max_context_tokens"],
                max_generation_tokens=c["reserved_generation_tokens"],
                experiment_id=c.get("experiment_id", "teacher_train"),
                run_id=out.name,
                mode=r["mode"],
                model_name=c["model"],
                model_endpoint=c["endpoint"],
                seed=None,
                plan_preserving=True,
                initial_state=state,
            )
            try:
                controllers = []
                if condition == "clean":
                    controller = None
                    if not c.get("focused_conditions"):
                        controller = RuntimePriorityFault(
                            r["payload"],
                            r["selection"],
                            observation_budgets(r["selection"]),
                            SelectedFaultController,
                            lambda x: counter.text(_tool_content(x)),
                            schedule=dict(schedule, observe_only=True),
                        )
                    a = await run_task_attempt(
                        model=m, tool_intervention=controller, **kw
                    )
                    clean[r["cell"]] = {"success": a["success"]}
                    save(path / "initial_state.json", state)
                    if controller is not None:
                        save(path / "checkpoints.json", controller.checkpoints)
                elif c.get("focused_conditions"):

                    def factory(target):
                        controller = RuntimePriorityFault(
                            r["payload"],
                            r["selection"],
                            observation_budgets(r["selection"]),
                            SelectedFaultController,
                            lambda x: counter.text(_tool_content(x)),
                            target,
                            None,
                            str(r["cell"]),
                            dict(
                                schedule,
                                reservation_mode="eligible_now",
                                draw_on_selection=True,
                                capture_first_fault=True,
                            ),
                            balance,
                        )
                        controllers.append(controller)
                        return controller

                    outcome = await run_with_nohit_replay(
                        controller_factory=factory,
                        model=m,
                        attempt_kwargs=kw,
                        save_attempt=lambda name, a: save(
                            path / (name + ".attempt.json"), a
                        ),
                    )
                    a = outcome.pop("final")
                    save(path / "branch_selection.json", outcome)
                    controller = controllers[-1]
                    checkpoint = first_fault_state(controller, a)
                    if checkpoint is not None:
                        save(path / "first_fault_checkpoint.json", checkpoint)
                else:
                    spec = branch_plans[str(r["cell"])]
                    donor_path = out / "clean" / f"cell_{r['cell']:02d}"
                    donor = read(donor_path / "attempt.json")
                    if spec["kind"] is None:
                        a = deepcopy(donor)
                        save(
                            path / "branch_selection.json",
                            dict(
                                scored_branch="clean_no_opportunity",
                                reason=spec["reason"],
                                replay_attempted=False,
                            ),
                        )
                    else:
                        controller = RuntimePriorityFault(
                            r["payload"],
                            r["selection"],
                            {spec["kind"]: spec["budget"]},
                            SelectedFaultController,
                            lambda x: counter.text(_tool_content(x)),
                            spec["target"],
                            schedule=dict(schedule, capture_first_fault=True),
                        )
                        checkpoint = read(donor_path / "checkpoints.json")[spec["kind"]]
                        a = await run_clean_branch(
                            donor=donor,
                            checkpoint=checkpoint,
                            target=spec["target"],
                            controller=controller,
                            model=m,
                            attempt_kwargs=kw,
                        )
                        save(
                            path / "branch_selection.json",
                            dict(
                                scored_branch="clean_checkpoint",
                                target=spec["target"],
                                budget=spec["budget"],
                                replay_attempted=False,
                            ),
                        )
                        checkpoint = first_fault_state(controller, a)
                        if checkpoint is not None:
                            save(path / "first_fault_checkpoint.json", checkpoint)
                save(path / "attempt.json", a)
                if not a["cleanup"]["succeeded"]:
                    raise RuntimeError(f"cleanup failed: {condition}/{r['cell']}")
                if not a["valid"]:
                    error = a.get("error") or {}
                    if isolated_infrastructure_failure(a):
                        invalid.append(
                            dict(cell=r["cell"], condition=condition, error=error)
                        )
                        save(out / "infrastructure_invalid.json", invalid)
                        print(
                            "INVALID_INFRASTRUCTURE",
                            condition,
                            r["cell"],
                            error,
                            flush=True,
                        )
                        return
                    raise RuntimeError(f"invalid: {condition}/{r['cell']}: {error}")
                events = (a.get("fault") or {}).get("events", [])
                assert all(e.get("selected_fault") in allowed for e in events), (
                    "OOD event in teacher collection"
                )
                export = False
                if (
                    a["success"]
                    and student_export_allowed(a)
                    and (
                        condition == "clean"
                        or (a.get("fault") or {}).get("injected_calls", 0) > 0
                    )
                ):
                    last = read(path / "calls" / f"{m.n:03}.request.json")
                    record = student_record(
                        original,
                        a,
                        last["tools"],
                        last["messages"],
                        f"{out.name}-{condition}-{r['cell']:02}",
                        guidance=guidance,
                    )
                    record["partition"] = c.get("partition", "train")
                    proof = student_token_proof(record, counter.tokenizer)
                    save(path / "student_token_check.json", proof)
                    if proof["within_limit"]:
                        record["admission"] = "verifier_passed_automatic_checks"
                        with (out / f"{condition}_sft_records.jsonl").open("a") as sink:
                            sink.write(json.dumps(record, ensure_ascii=False) + "\n")
                        save(path / "student_candidate.json", record)
                        export = True
                summary = dict(
                    cell=r["cell"],
                    condition=condition,
                    task_id=r["task_id"],
                    domain=r["domain"],
                    mode=r["mode"],
                    success=a["success"],
                    hits=(a.get("fault") or {}).get("injected_calls", 0),
                    mechanism=(a.get("fault") or {}).get("selected_mechanism"),
                    turns=a["trajectory"]["turn_count"],
                    termination=a["trajectory"]["termination"],
                    student_candidate=export,
                )
                results.append(summary)
                save(out / "summary.json", results)
                print("DONE", summary, flush=True)
            finally:
                await m.client.close()

    try:
        procs, urls = await launch(
            {s["mcp_server_name"] for t in tasks.values() for s in t.gym_servers},
            runtime,
            port_offset=c.get("mcp_port_offset", 1000),
        )
        if c.get("focused_conditions"):
            balance = retry_balance(
                rows,
                schedule["seed"],
                read(ROOT / c["source_run"] / "summary.json"),
                restored,
            )
            # Separate phases preserve same-task DB isolation.
            for condition in ["clean", "fault"]:
                async with asyncio.TaskGroup() as group:
                    for spec in c["focused_conditions"]:
                        if spec["condition"] == condition:
                            group.create_task(
                                episode(
                                    next(r for r in rows if r["cell"] == spec["cell"]),
                                    condition,
                                )
                            )
            return
        for condition in conditions:
            if condition == "fault":
                donors = {
                    str(r["cell"]): dict(
                        attempt=read(
                            out / "clean" / f"cell_{r['cell']:02d}/attempt.json"
                        ),
                        checkpoints=read(
                            out / "clean" / f"cell_{r['cell']:02d}/checkpoints.json"
                        ),
                    )
                    for r in rows
                }
                branch_plans, balance_report = select_clean_branches(
                    rows, donors, schedule
                )
                if (out / "branch_plans.json").exists() and read(
                    out / "branch_plans.json"
                ) != branch_plans:
                    raise ValueError("Resumed clean branch assignments changed")
                save(out / "branch_plans.json", branch_plans)
                save(out / "balance.json", balance_report)
                budgets = {
                    key: ({spec["kind"]: spec["budget"]} if spec["kind"] else {})
                    for key, spec in branch_plans.items()
                }
                save(out / "candidate_budgets.json", budgets)
            async with asyncio.TaskGroup() as group:
                for r in rows:
                    group.create_task(episode(r, condition))
            print("PHASE COMPLETE", condition, flush=True)
    finally:
        if c.get("focused_conditions") and balance is not None:
            save(out / "balance.json", balance.report())
        stop_processes(procs)
        save(
            out / "cleanup.json",
            dict(
                completed=len(results),
                invalid_infrastructure=len(invalid),
                mcp_stopped=all(p.poll() is not None for p in procs),
            ),
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--dry-run", action="store_true")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
