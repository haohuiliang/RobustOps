"""Independent 16 clean / 16 fault G4 groups; no correction pool."""

import asyncio
from robustops.evaluation.fault_schedule import (
    observation_budgets,
    shared_strategy,
    STRATEGY_PATH,
)

from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from time import perf_counter

from verl.utils.ray_utils import auto_await
from robustops.training.rl_workers import RolloutManagerBase, ReadyQueue
from robustops.training.recovery_rl_core import write_json


class RecoveryManager(RolloutManagerBase):
    @auto_await
    async def generate_sequences(self, prompts):
        from robustops.training.recovery_rl_replay import ZeroSignalReplay

        cfg = self.config.recovery
        exclude_success = cfg.get("exclude_all_success_groups", False)
        step = int(prompts.meta_info["global_steps"])
        if (
            prompts.meta_info.get("validate")
            or len(prompts) != 128
            or self.rollout_config.n != 4
        ):
            raise ValueError("RL requires 32 training G4 groups")
        group_rows = [
            dict(x["recovery"]) for x in prompts.non_tensor_batch["extra_info"][::4]
        ]
        if (
            [r["joint_slot"] for r in group_rows] != list(range(32))
            or group_rows[0]["visit"] != self.next_visit
            or self.next_visit != (step - 1) * 32
        ):
            raise ValueError("Independent dataloader cursor/slot mismatch")
        queue_mode = getattr(cfg, "task_sampling", "frozen") == "queue"
        if queue_mode:
            if not hasattr(self, "task_queues"):
                from robustops.training.recovery_task_queue import restore_task_queues

                resume = getattr(cfg, "resume_checkpoint", None)
                saved = (
                    json.loads((Path(resume) / "recovery_state.json").read_text())
                    if resume
                    else {}
                )
                self.task_queues = restore_task_queues(
                    Path(cfg.task_schedule_file).parent,
                    cfg.seed,
                    saved,
                    getattr(cfg, "resume_history_output", cfg.output),
                )
            used = set()
            for slot in range(32):
                row = self.task_queues.take(
                    "clean" if slot < 16 else "fault",
                    used,
                    self.task_queues.primary_source(step, slot),
                )
                row.setdefault("training_source", "train")
                group_rows[slot].update(
                    row,
                    planned_flow="clean" if slot < 16 else "fault",
                    joint_fault_parents=list(range(16, 32)),
                )
                used.add((row["task_id"], row["mode"]))
        parents = deepcopy(group_rows)
        main_rows = group_rows
        if len({(r["task_id"], r["mode"]) for r in main_rows}) != len(main_rows):
            raise ValueError("task plan contains duplicate cells")
        changed = asyncio.Event()
        ready = ReadyQueue(
            self.agent_loop_workers, cfg.concurrency_per_worker, on_change=changed.set
        )
        output_futures, plans = [None] * 128, {}
        clean, donors, selected, fallbacks = {}, {}, [], {}
        all_donors, replacements = [], []
        signal_replacements, discarded = [], []
        member_retry_events = []
        retry_log = Path(cfg.output) / "member_retries" / f"{step}.jsonl"
        monitors, monitor_flows, first_results = {}, {}, {}
        selector_lock = asyncio.Lock()
        donor_jobs = []
        fault_jobs = []
        generations = [0] * 32
        active_rows = {}
        active_context = {}
        replay_attempts, replay_enqueued = [], []
        replay_zero_limit = getattr(cfg, "zero_pg_replay_max_zero_groups", 5)
        replay_zero_count, replay_inflight = 0, 0
        used_cells = {(r["task_id"], r["mode"]) for r in main_rows}
        window = "queue" if queue_mode else str((step - 1) // 30)
        reserve = (
            []
            if queue_mode
            else json.loads(
                Path(cfg.task_schedule_file)
                .with_name("fault_reserves.json")
                .read_text()
            )[window]["rows"]
        )
        if not hasattr(self, "fault_reserve_cursors"):
            resume = (
                cfg.get("resume_checkpoint")
                if hasattr(cfg, "get")
                else getattr(cfg, "resume_checkpoint", None)
            )
            saved = (
                json.loads((Path(resume) / "recovery_state.json").read_text())
                if resume
                else {}
            )
            self.fault_reserve_cursors = saved.get("fault_reserve_cursors", {})
        if not queue_mode:
            self.fault_reserve_cursors.setdefault(window, 0)
        if not hasattr(self, "sampling_history"):
            from robustops.training.recovery_sampling_history import SamplingHistory

            resume = getattr(cfg, "resume_checkpoint", None)
            saved = (
                json.loads((Path(resume) / "recovery_state.json").read_text())
                if resume
                else {}
            )
            self.sampling_history = SamplingHistory.restore(
                saved, getattr(cfg, "resume_history_output", cfg.output)
            )
        for row in main_rows:
            self.sampling_history.mark(row)
        if not hasattr(self, "zero_signal_replay"):
            resume = getattr(cfg, "resume_checkpoint", None)
            saved = (
                json.loads((Path(resume) / "recovery_state.json").read_text())
                if resume
                else {}
            )
            self.zero_signal_replay = ZeroSignalReplay(
                getattr(cfg, "zero_pg_replay_delay", 0), saved.get("zero_signal_replay")
            )
        replay = self.zero_signal_replay
        started = perf_counter()
        seed_seconds = 0.0
        jobs = []
        completed_batch = False
        sampling = dict(
            temperature=self.rollout_config.temperature,
            top_p=self.rollout_config.top_p,
            top_k=self.rollout_config.top_k,
            repetition_penalty=1.0,
            logprobs=True,
        )

        async def seed(row):
            nonlocal seed_seconds
            from robustops.training.recovery_rl_data import capture_initial_state

            target = (
                Path(cfg.shared_plans)
                / "initial_states"
                / f"{row['task_id']}_{row['mode']}.json"
            )
            if not target.exists():
                t = perf_counter()
                async with asyncio.timeout(cfg.episode_timeout_seconds):
                    await capture_initial_state(
                        cfg.root, row, dict(cfg.mcp_urls), target
                    )
                seed_seconds += perf_counter() - t

        def launch(
            group,
            parent,
            flow,
            state=None,
            selected_plan=None,
            supplement=False,
            source_row=None,
            replay_entry=None,
            retry_budget=None,
        ):
            row = source_row if source_row is not None else parents[parent]
            row.setdefault("training_source", "train")
            active_rows[group] = row
            plan = dict(
                visit=group_rows[group]["visit"],
                task_id=row["task_id"],
                mode=row["mode"],
                epoch=row["epoch"],
                fault_enabled=flow == "fault",
                kind=None,
                budget=0,
                seed=cfg.seed + group_rows[group]["visit"] * 1009,
                candidate_doses=observation_budgets(row["selection"]),
            )
            if selected_plan:
                plan.update(selected_plan)
                plan["visit"] = group_rows[group]["visit"]
            if replay_entry:
                # Keep the environment/prefix and fault contract, not the old RNG.
                plan["seed"] = (
                    cfg.seed
                    + group_rows[group]["visit"] * 1009
                    + generations[group] * 10000019
                )
            plans[group] = plan
            active_context[group] = dict(state_path=state, replay_entry=replay_entry)
            used_cells.add((row["task_id"], row["mode"]))
            from robustops.training.recovery_rl_member_retry import MemberRetryBudget

            if retry_budget is None:
                retry_budget = MemberRetryBudget(
                    member_retry_events,
                    retry_log,
                    dict(
                        step=step,
                        group_slot=group,
                        generation=generations[group],
                        flow=flow,
                        task_id=row["task_id"],
                        mode=row["mode"],
                        source_state=state,
                    ),
                )
            else:
                retry_budget.context.update(source_state=state, phase="member")
            active_context[group]["retry_budget"] = retry_budget
            futures = []
            for sample in range(4):
                i = group * 4 + sample
                kwargs = {
                    key: value[i] for key, value in prompts.non_tensor_batch.items()
                }
                kwargs.update(
                    agent_name="recovery_agent",
                    extra_info=dict(index=plan["visit"], recovery=row),
                    recovery_plan=plan,
                    recovery_sample=sample,
                    recovery_step=step,
                    joint_flow=flow,
                    joint_slot=group,
                    joint_state_path=state,
                    joint_donor=False,
                    joint_supplement=supplement,
                    joint_generation=generations[group],
                )
                priority = (
                    3
                    if supplement
                    else 0
                    if flow == "clean" and sample == 0
                    else 1
                    if state
                    else 2
                )

                def submit(retry, kwargs=kwargs, sample=sample, priority=priority):
                    return ready.submit(
                        priority,
                        "run_member",
                        sampling,
                        dict(
                            step=step,
                            sample_index=plan["visit"],
                            rollout_n=sample,
                            validate=False,
                        ),
                        dict(kwargs, joint_member_retry=retry),
                    )

                async def sample_member(
                    submit=submit,
                    sample=sample,
                    budget=retry_budget,
                    source=row["training_source"],
                ):
                    member = await budget.run(submit, sample)
                    if member is not None:
                        member.extra_fields["recovery_metrics"]["training_source"] = (
                            source
                        )
                    return member

                future = asyncio.create_task(sample_member())
                futures.append(future)
                output_futures[i] = future
            if group not in monitors:
                monitor_flows[group] = flow
                monitors[group] = asyncio.create_task(ensure_signal(group, futures))
                monitors[group].add_done_callback(lambda _: changed.set())
            return futures

        async def replace_task(parent, flow, reason):
            cursor = (
                self.task_queues.serial
                if queue_mode
                else self.fault_reserve_cursors[window]
            )
            try:
                row = (
                    self.task_queues.take(
                        flow,
                        used_cells,
                        parents[parent].get("training_source", "train"),
                    )
                    if queue_mode
                    else self.sampling_history.take(window, reserve, cursor, used_cells)
                )
            except RuntimeError:
                write_json(
                    Path(cfg.output)
                    / "diagnostics"
                    / f"{step}_replacement_exhausted.json",
                    dict(
                        step=step,
                        window=window,
                        reason=reason,
                        fault_replacements=replacements,
                        signal_replacements=signal_replacements,
                        discarded=discarded,
                        reserve_size=len(reserve),
                    ),
                )
                raise
            if not queue_mode:
                self.fault_reserve_cursors[window] = cursor + 1
            visit = 10000000 + (0 if queue_mode else int(window) * 100000) + cursor
            row.update(
                visit=visit,
                parent_visit=visit,
                joint_slot=parent,
                joint_parent_slot=parent,
                joint_fault_parents=[parent] if flow == "fault" else [],
                planned_flow=flow,
                independent_tasks=True,
            )
            parents[parent] = row
            used_cells.add((row["task_id"], row["mode"]))
            await seed(row)
            return dict(
                task_id=row["task_id"],
                mode=row["mode"],
                parent_visit=visit,
                training_source=row.get("training_source", "train"),
            )

        async def replace_zero(group, flow, reason, rejected):
            nonlocal replay_inflight
            # Each in-flight retry can still fail: reserve one failure slot before
            # dispatch so parallel completions cannot exceed the per-step bound.
            source = active_rows[group].get("training_source", "train")
            entry = (
                replay.claim(flow, step, used_cells)
                if replay_zero_count + replay_inflight < replay_zero_limit
                else None
            )
            if entry:
                replay_inflight += 1
                row = deepcopy(entry["row"])
                row.setdefault("training_source", "train")
                visit = (
                    100000000 + group_rows[group]["visit"] * 10000 + generations[group]
                )
                row.update(
                    visit=visit,
                    parent_visit=visit,
                    joint_slot=group,
                    joint_parent_slot=group,
                    joint_fault_parents=[group] if flow == "fault" else [],
                    planned_flow=flow,
                )
                parents[group] = row
                used_cells.add((row["task_id"], row["mode"]))
                state_path = entry["state_path"]
                if flow == "fault" and not Path(state_path).is_file():
                    raise RuntimeError(
                        f"missing delayed fault replay snapshot: {state_path}"
                    )
                if flow == "clean":
                    await seed(row)
                attempt = dict(
                    group_slot=group,
                    generation=generations[group],
                    flow=flow,
                    training_source=row["training_source"],
                    requested_source=source,
                    task_id=row["task_id"],
                    mode=row["mode"],
                    source_state=state_path,
                    first_zero_step=entry["first_zero_step"],
                    last_zero_step=entry["last_zero_step"],
                    eligible_step=entry["eligible_step"],
                    zero_count=entry["zero_count"],
                    replay_step=step,
                    policy_update_gap=step - entry["last_zero_step"],
                    status="running",
                )
                replay_attempts.append(attempt)
                if flow == "fault":
                    async with selector_lock:
                        before = deepcopy(self.planner.state_dict())
                        self.planner.claim(row, entry["plan"]["kind"])
                        self.sampling_history.claim_fault(row, entry["plan"]["kind"])
                        selected.append(
                            dict(
                                parent_slot=group,
                                group_slot=group,
                                plan=entry["plan"],
                                balance_before=before,
                                balance_after=self.planner.state_dict(),
                                anchor=dict(
                                    state_path=state_path,
                                    actual_kind=entry["plan"]["kind"],
                                ),
                                supplement=True,
                                replay=True,
                                accepted_for_training=True,
                            )
                        )
                rejected.update(
                    replacement_source="delayed_zero_pg",
                    replacement=dict(
                        task_id=row["task_id"],
                        mode=row["mode"],
                        parent_visit=visit,
                        training_source=row["training_source"],
                        source_state=state_path,
                        last_zero_step=entry["last_zero_step"],
                    ),
                )
                return launch(
                    group,
                    group,
                    flow,
                    state_path,
                    entry["plan"],
                    supplement=True,
                    replay_entry=entry,
                )
            rejected["replacement_source"] = (
                "continuous_task_queue" if queue_mode else "outside_30_step_reserve"
            )
            if replay_zero_count + replay_inflight >= replay_zero_limit:
                rejected["replay_skip_reason"] = (
                    "zero_limit_reached_or_reserved_by_inflight"
                )
            rejected["replacement"] = await replace_task(group, flow, reason)
            if flow == "fault":
                donors[group] = start_donor(group)
                rejected["replacement_submitted_s"] = perf_counter() - started
                await fault_group(group, group, supplement=True)
                return output_futures[group * 4 : group * 4 + 4]
            return launch(group, group, "clean", supplement=True)

        def finish_replay(group, members, status):
            nonlocal replay_zero_count, replay_inflight
            context = active_context[group]
            if context["replay_entry"] is None:
                return
            attempt = next(
                x
                for x in reversed(replay_attempts)
                if x["group_slot"] == group and x["generation"] == generations[group]
            )
            if attempt["status"] == "running":
                replay_inflight -= 1
            if status == "zero_pg" and attempt["status"] != "zero_pg":
                replay_zero_count += 1
            attempt.update(
                status=status,
                successes=sum(
                    o.extra_fields["recovery_metrics"]["strict_success"]
                    for o in members
                ),
                episodes=[o.extra_fields["recovery_episode"] for o in members],
            )
            for key in (
                "generated_tokens",
                "tool_calls",
                "episode_seconds",
                "generation_request_seconds",
            ):
                attempt[key] = sum(
                    o.extra_fields["recovery_metrics"][key] for o in members
                )

        async def fault_group(group, parent, *, supplement=False):
            while True:
                donor = await donors[parent]
                available = donor.extra_fields["recovery_artifacts"]["opportunities"]
                reason = "no_eligible_clean_opportunity"
                if available:
                    async with selector_lock:
                        before = deepcopy(self.planner.state_dict())
                        choice = self.planner.reserve(
                            parents[parent],
                            parents[parent]["parent_visit"],
                            forced=True,
                            available=self.sampling_history.prefer_faults(
                                parents[parent], available
                            ),
                            claim=False,
                            repeat_config=shared_strategy(
                                cfg.get("fault_strategy", STRATEGY_PATH), root=cfg.root
                            ),
                        )
                        choice["cell_fault_counts_before"] = dict(
                            self.sampling_history.faults.get(
                                json.dumps(
                                    [
                                        parents[parent]["task_id"],
                                        parents[parent]["mode"],
                                    ],
                                    separators=(",", ":"),
                                ),
                                {},
                            )
                        )
                        opportunity_path = available[choice["kind"]][
                            str(choice["budget"])
                        ]
                        # One ready anchor unlocks four suffixes. Dispatch it at
                        # the next free slot, ahead of not-yet-started rollouts.
                        from robustops.training.recovery_rl_member_retry import (
                            MemberRetryBudget,
                        )

                        retry_budget = MemberRetryBudget(
                            member_retry_events,
                            retry_log,
                            dict(
                                step=step,
                                group_slot=group,
                                generation=generations[group],
                                flow="fault",
                                phase="anchor",
                                task_id=parents[parent]["task_id"],
                                mode=parents[parent]["mode"],
                                source_state=opportunity_path,
                            ),
                        )
                        anchor_path = (
                            Path(cfg.output)
                            / "anchors"
                            / f"step_{step}"
                            / f"parent_{parent}_visit_{parents[parent]['parent_visit']}"
                        )

                        def submit_anchor(retry):
                            path = str(anchor_path) + (
                                f"_retry_{retry}" if retry else ""
                            )
                            return ready.submit(
                                -1,
                                "prepare_anchor",
                                parents[parent],
                                choice,
                                opportunity_path,
                                path,
                            )

                        anchor = await retry_budget.run(submit_anchor, -1)
                        if anchor is not None:
                            self.planner.claim(parents[parent], anchor["actual_kind"])
                            self.sampling_history.claim_fault(
                                parents[parent], anchor["actual_kind"]
                            )
                        else:
                            anchor = dict(
                                state_path=None,
                                fallback_reason=retry_budget.discard_reason("anchor"),
                            )
                        selected.append(
                            dict(
                                parent_slot=parent,
                                group_slot=group,
                                plan=choice,
                                balance_before=before,
                                balance_after=self.planner.state_dict(),
                                anchor=anchor,
                                supplement=supplement,
                                accepted_for_training=bool(anchor["state_path"]),
                            )
                        )
                    if anchor["state_path"]:
                        launch(
                            group,
                            parent,
                            "fault",
                            anchor["state_path"],
                            choice,
                            supplement=supplement,
                            retry_budget=retry_budget,
                        )
                        return
                    reason = anchor["fallback_reason"]
                cursor = (
                    self.task_queues.serial
                    if queue_mode
                    else self.fault_reserve_cursors[window]
                )
                rejected = dict(
                    group_slot=group,
                    task_id=parents[parent]["task_id"],
                    mode=parents[parent]["mode"],
                    parent_visit=parents[parent]["parent_visit"],
                    reason=reason,
                    reserve_cursor=cursor,
                )
                replacements.append(rejected)
                rejected["replacement"] = await replace_task(parent, "fault", reason)
                donors[parent] = start_donor(parent)

        async def originals():
            try:
                # Keep initial-state capture serial, but overlap it with work
                # whose state is already ready. No whole-batch setup barrier.
                from robustops.training.recovery_rl_emergency import check_stop_request

                for parent, row in enumerate(parents):
                    check_stop_request(cfg.output)
                    await seed(row)
                    if parent < 16:
                        clean[parent] = launch(parent, parent, "clean")
                    else:
                        donors[parent] = start_donor(parent)
                        task = asyncio.create_task(fault_group(parent, parent))
                        fault_jobs.append(task)
                        task.add_done_callback(lambda _: changed.set())
                await asyncio.gather(*fault_jobs)
            finally:
                # gather propagates the original error, but does not cancel its
                # siblings. Stop all parent continuations before global cleanup.
                for task in fault_jobs:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*fault_jobs, return_exceptions=True)
                changed.set()

        def launch_donor(parent):
            row = parents[parent]
            plan = dict(
                visit=row["visit"],
                task_id=row["task_id"],
                mode=row["mode"],
                epoch=row["epoch"],
                fault_enabled=False,
                kind=None,
                budget=0,
                seed=cfg.seed + row["visit"] * 1009 + 70000001,
                candidate_doses=observation_budgets(row["selection"]),
            )
            i = parent * 4
            kwargs = {key: value[i] for key, value in prompts.non_tensor_batch.items()}
            kwargs.update(
                agent_name="recovery_agent",
                extra_info=dict(index=plan["visit"], recovery=row),
                recovery_plan=plan,
                recovery_sample=0,
                recovery_step=step,
                joint_flow="donor",
                joint_slot=parent,
                joint_state_path=None,
                joint_donor=True,
                joint_supplement=False,
                preparation_only=True,
            )
            from robustops.training.recovery_rl_member_retry import MemberRetryBudget

            retry_budget = MemberRetryBudget(
                member_retry_events,
                retry_log,
                dict(
                    step=step,
                    group_slot=parent,
                    parent_visit=row["parent_visit"],
                    flow="donor",
                    task_id=row["task_id"],
                    mode=row["mode"],
                ),
            )

            def submit(retry):
                return ready.submit(
                    0,
                    "run_member",
                    sampling,
                    dict(
                        step=step,
                        sample_index=plan["visit"],
                        rollout_n=0,
                        validate=False,
                    ),
                    dict(kwargs, joint_member_retry=retry),
                )

            future = asyncio.create_task(retry_budget.run(submit, 0))
            all_donors.append((parent, row["parent_visit"], future))
            return future, retry_budget

        def start_donor(parent):
            # Launch now; its own continuation replaces an ineligible donor without
            # waiting for earlier slots' fault selection or the remaining batch.
            first, first_budget = launch_donor(parent)

            async def eligible():
                future, retry_budget = first, first_budget
                while True:
                    out = await future
                    if (
                        out is not None
                        and out.extra_fields["recovery_artifacts"]["opportunities"]
                    ):
                        return out
                    rejected = dict(
                        group_slot=parent,
                        task_id=parents[parent]["task_id"],
                        mode=parents[parent]["mode"],
                        parent_visit=parents[parent]["parent_visit"],
                        reason=retry_budget.discard_reason("donor")
                        if out is None
                        else "no_eligible_clean_opportunity",
                        detected_s=perf_counter() - started,
                        reserve_cursor=self.task_queues.serial
                        if queue_mode
                        else self.fault_reserve_cursors[window],
                    )
                    replacements.append(rejected)
                    rejected["replacement"] = await replace_task(
                        parent, "fault", rejected["reason"]
                    )
                    future, retry_budget = launch_donor(parent)
                    rejected["replacement_submitted_s"] = perf_counter() - started

            task = asyncio.create_task(eligible())
            donor_jobs.append(task)
            task.add_done_callback(lambda _: changed.set())
            return task

        async def ensure_signal(group, futures, force=False):
            from robustops.training.rl_advantage import (
                training_filter_slots,
                all_success_slots,
            )

            members = await asyncio.gather(*futures)

            first_results.setdefault(
                group, members if all(o is not None for o in members) else []
            )
            # Every group is centered before batch whitening. Its zero support can
            # be checked as soon as G4 finishes; actual final-batch support is also
            # checked below, covering floating-point threshold boundary cases.
            while (
                force
                or any(o is None for o in members)
                or training_filter_slots(
                    members,
                    list(cfg.reward_weights),
                    cfg.reward_normalization,
                    group_count=1,
                    exclude_all_success_groups=exclude_success,
                )
            ):
                force = False
                invalid = any(o is None for o in members)
                solved = (
                    not invalid and exclude_success and bool(all_success_slots(members))
                )
                members = [o for o in members if o is not None]
                source_flow = monitor_flows[group]
                context = active_context[group]
                retry_budget = context["retry_budget"]
                protocol_rejected = invalid and bool(
                    retry_budget.failure_counts.get("model_protocol_error")
                )
                rejected = dict(
                    group_slot=group,
                    generation=generations[group],
                    source_flow=source_flow,
                    training_source=active_rows[group].get("training_source", "train"),
                    reason=retry_budget.discard_reason()
                    if invalid
                    else "all_success"
                    if solved
                    else "zero_reward_before_local_override",
                    detected_s=perf_counter() - started,
                    episodes=[o.extra_fields["recovery_episode"] for o in members],
                )
                if invalid:
                    rejected.update(
                        member_retries_used=retry_budget.used,
                        failure_counts=dict(retry_budget.failure_counts),
                    )
                signal_replacements.append(rejected)
                finish_replay(
                    group,
                    members,
                    "all_success"
                    if solved
                    else "model_protocol_error"
                    if protocol_rejected
                    else "zero_pg",
                )
                if solved:
                    replay.resolve(
                        source_flow, active_rows[group], context["state_path"]
                    )
                # A format-rejected group has no usable G4 advantage. Do not
                # enqueue it as evidence of a valid zero-signal business path.
                entry = (
                    None
                    if protocol_rejected or solved
                    else replay.reject(
                        source_flow,
                        active_rows[group],
                        plans[group],
                        context["state_path"],
                        step,
                        previous=context["replay_entry"],
                    )
                )
                if entry:
                    replay_enqueued.append(
                        dict(
                            key=entry["key"],
                            last_zero_step=step,
                            eligible_step=entry["eligible_step"],
                            zero_count=entry["zero_count"],
                        )
                    )
                discarded.extend(
                    dict(
                        group_slot=group,
                        flow=source_flow,
                        kind=o.extra_fields.get("recovery_kind"),
                        metrics=o.extra_fields["recovery_metrics"],
                        episode=o.extra_fields["recovery_episode"],
                    )
                    for o in members
                )
                generations[group] += 1
                rejected["replacement_flow"] = source_flow
                futures = await replace_zero(
                    group, source_flow, rejected["reason"], rejected
                )
                rejected.setdefault("replacement_submitted_s", perf_counter() - started)
                members = await asyncio.gather(*futures)

            finish_replay(group, members, "accepted")
            replay.resolve(
                members[0].extra_fields["recovery_flow"],
                active_rows[group],
                active_context[group]["state_path"],
            )
            return members

        async def supervise():
            from robustops.training.recovery_rl_emergency import check_stop_request

            while True:
                check_stop_request(cfg.output)
                changed.clear()
                for task in [*jobs, *donor_jobs, *fault_jobs, *monitors.values()]:
                    if task.done() and not task.cancelled():
                        task.result()
                if (
                    all(t.done() for t in jobs)
                    and len(monitors) == 32
                    and all(t.done() for t in monitors.values())
                ):
                    return
                try:
                    await asyncio.wait_for(changed.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

        try:
            jobs = [asyncio.create_task(originals())]
            for job in jobs:
                job.add_done_callback(lambda _: changed.set())
            await supervise()
            target = dict(clean=16, fault=16)
            resolved = await asyncio.gather(*output_futures)
            from robustops.training.rl_advantage import (
                zero_pg_slots,
                zero_reward_advantage_slots,
                training_filter_slots,
                all_success_slots,
            )

            initial_outputs = [o for group in range(32) for o in first_results[group]]
            initial_batch = [
                dict(
                    group_slot=o.extra_fields["recovery_joint_slot"],
                    flow=o.extra_fields["recovery_flow"],
                    metrics=o.extra_fields["recovery_metrics"],
                    episode=o.extra_fields["recovery_episode"],
                )
                for o in initial_outputs
            ]
            initial_zero = sorted(
                g
                for g, members in first_results.items()
                if members
                and zero_pg_slots(
                    members,
                    list(cfg.reward_weights),
                    cfg.reward_normalization,
                    cfg.local_bad_eta,
                    q_temporal_credit=cfg.get("q_temporal_credit", False),
                    terminal_stop_cap=cfg.get("terminal_stop_advantage_cap"),
                    group_count=1,
                )
            )
            initial_zero_reward = sorted(
                g
                for g, members in first_results.items()
                if members
                and zero_reward_advantage_slots(
                    members,
                    list(cfg.reward_weights),
                    cfg.reward_normalization,
                    group_count=1,
                )
            )
            zeros = training_filter_slots(
                resolved,
                list(cfg.reward_weights),
                cfg.reward_normalization,
                exclude_all_success_groups=exclude_success,
            )
            while zeros:
                # Only a numerical boundary missed by the per-group check can
                # reach here; ordinary zero groups have already refilled online.
                for group in zeros:
                    monitors[group] = asyncio.create_task(
                        ensure_signal(
                            group, output_futures[group * 4 : group * 4 + 4], force=True
                        )
                    )
                    monitors[group].add_done_callback(lambda _: changed.set())
                await supervise()
                resolved = await asyncio.gather(*output_futures)
                zeros = training_filter_slots(
                    resolved,
                    list(cfg.reward_weights),
                    cfg.reward_normalization,
                    exclude_all_success_groups=exclude_success,
                )
            # Recompute final batch whitening in prepare_advantages after packing.
            queue = ready.summary()
            output = await self.agent_loop_workers[0].pack_members.remote(
                resolved, prompts.non_tensor_batch
            )
            timing = self._performance_metrics(
                [output.meta_info.pop("metrics")], output
            )
            timing.update(
                joint_seed_setup=seed_seconds,
                joint_queue_wall=queue["wall_s"],
                joint_busy_slot_sum=queue["busy_slot_s"],
                joint_idle_slot_sum=queue["idle_slot_s"],
                joint_queue_wait_sum=queue["queue_wait_sum_s"],
                joint_anchor_service_sum=queue["anchor_service_sum_s"],
            )
            output.meta_info["timing"] = timing
            self.next_visit = group_rows[-1]["visit"] + 1
            self.save_joint_state(step)
            streams = (
                self.task_queues.summary()
                if queue_mode
                else json.loads(Path(cfg.task_schedule_file).read_text())["batches"][
                    step - 1
                ]["streams"]
            )
            state_path = Path(cfg.output) / "recovery_states" / f"{step}.json"
            state = json.loads(state_path.read_text())
            state.update(
                schedule="independent_16_16",
                task_streams=streams,
                task_plan_step=step,
                reward_normalization=cfg.reward_normalization,
                fault_reserve_cursors=self.fault_reserve_cursors,
                zero_signal_replay=replay.state_dict(),
                sampling_history=self.sampling_history.state_dict(),
            )
            if queue_mode:
                state.update(
                    task_sampling="queue", task_queues=self.task_queues.state_dict()
                )
            write_json(state_path, state)
            self.report_joint(step, output, selected, fallbacks, ready.peak)
            report_path = Path(cfg.output) / "diagnostics" / f"{step}.json"
            report = json.loads(report_path.read_text())
            from robustops.training.rl_advantage import fault_exposure_summary

            accepted_rows = [
                dict(
                    flow=o.extra_fields["recovery_flow"],
                    kind=o.extra_fields.get("recovery_kind"),
                    metrics=o.extra_fields["recovery_metrics"],
                )
                for o in resolved
            ]
            report.update(
                member_retry_limit=4,
                member_retry_events=member_retry_events,
                primary_source_groups=dict(
                    Counter(r.get("training_source", "train") for r in main_rows)
                ),
                accepted_source_groups=dict(
                    Counter(
                        r.get("training_source", "train") for r in active_rows.values()
                    )
                ),
                initial_invalid_group_slots=[
                    g for g, members in first_results.items() if not members
                ],
                fault_point_policy=cfg.get("fault_point_policy")
                if hasattr(cfg, "get")
                else getattr(cfg, "fault_point_policy", None),
                fault_exposure=dict(
                    attempted=fault_exposure_summary([*discarded, *accepted_rows]),
                    accepted=fault_exposure_summary(accepted_rows),
                ),
                reward_normalization=cfg.reward_normalization,
                parent_tasks=32,
                planned_fault_parents=16,
                planned_fault_parent_fraction=0.5,
                planned_group_counts=[16, 16, 0],
                task_plan=[
                    dict(
                        task_id=r["task_id"],
                        mode=r["mode"],
                        domain=r["domain"],
                        epoch=r["epoch"],
                        slot=r["joint_slot"],
                        planned_flow=r["planned_flow"],
                        training_source=r.get("training_source", "train"),
                    )
                    for r in main_rows
                ],
                actual_task_plan=[
                    dict(
                        task_id=r["task_id"],
                        mode=r["mode"],
                        domain=r["domain"],
                        slot=g,
                        planned_flow=resolved[g * 4].extra_fields["recovery_flow"],
                        training_source=r.get("training_source", "train"),
                    )
                    for g, r in sorted(active_rows.items())
                ],
                fault_replacements=replacements,
                fault_reserve_cursors=dict(self.fault_reserve_cursors),
                signal_replacements=signal_replacements,
                discarded_rows=discarded,
                zero_pg_replay=dict(
                    delay=replay.delay,
                    max_zero_groups=replay_zero_limit,
                    zero_groups=replay_zero_count,
                    queue=replay.summary(step),
                    enqueued=replay_enqueued,
                    attempts=replay_attempts,
                ),
                group_filter="reward_exclude_all_success_before_local_override"
                if exclude_success
                else "reward_before_local_override",
                initial_all_success_slots=all_success_slots(initial_outputs),
                initial_filtered_slots=sorted(
                    set(initial_zero_reward)
                    | (
                        set(all_success_slots(initial_outputs))
                        if exclude_success
                        else set()
                    )
                ),
                initial_zero_reward_slots=initial_zero_reward,
                initial_batch_rows=initial_batch,
                initial_zero_pg_slots=initial_zero,
                initial_zero_pg_without_repeat=sorted(
                    g
                    for g, members in first_results.items()
                    if members
                    and zero_pg_slots(
                        members,
                        list(cfg.reward_weights),
                        cfg.reward_normalization,
                        cfg.local_bad_eta,
                        q_temporal_credit=cfg.get("q_temporal_credit", False),
                        terminal_stop_cap=cfg.get("terminal_stop_advantage_cap"),
                        group_count=1,
                        ignore_rules=("repeated_no_progress_read",),
                    )
                ),
                task_sampling="queue" if queue_mode else "frozen",
                fault_dispatch="donor_ready_serial_actual_hit_selection",
                initial_state_dispatch="seed_then_submit_each_parent",
                anchor_ready_priority=-1,
                task_queues=self.task_queues.summary() if queue_mode else None,
                replacement_epochs={}
                if queue_mode
                else deepcopy(self.sampling_history.reserves),
                donor_rows=[
                    dict(
                        parent_slot=p,
                        parent_visit=v,
                        accepted_for_training=v == parents[p]["parent_visit"],
                        metrics=f.result().extra_fields["recovery_metrics"],
                        artifacts=f.result().extra_fields["recovery_artifacts"],
                    )
                    for p, v, f in all_donors
                    if f.result() is not None
                ],
                target_allocation=target,
                actual_allocation={
                    k: report["flow_rows"].get(k, 0) // 4 for k in ("clean", "fault")
                },
            )
            write_json(report_path, report)
            print("RECOVERY_NORMALIZATION_BATCH " + json.dumps(report), flush=True)
            write_json(
                Path(cfg.output) / "diagnostics" / f"{step}_schedule.json",
                dict(
                    step=step,
                    queue=queue,
                    events=ready.events,
                    seed_setup_s=seed_seconds,
                ),
            )
            completed_batch = True
            return output
        finally:
            cleanup_jobs = [*jobs, *donor_jobs, *fault_jobs, *monitors.values()]
            for job in cleanup_jobs:
                if not job.done():
                    job.cancel()
            await asyncio.gather(*cleanup_jobs, return_exceptions=True)
            await ready.close()
            if not completed_batch:
                write_json(
                    Path(cfg.output) / "diagnostics" / f"{step}_schedule.json",
                    dict(
                        step=step,
                        status="aborted",
                        events=ready.events,
                        fault_replacements=replacements,
                        fault_reserve_cursors=dict(self.fault_reserve_cursors),
                        signal_replacements=signal_replacements,
                        discarded_rows=discarded,
                        zero_pg_replay=dict(
                            queue=replay.summary(step),
                            enqueued=replay_enqueued,
                            attempts=replay_attempts,
                        ),
                        seed_setup_s=seed_seconds,
                    ),
                )
