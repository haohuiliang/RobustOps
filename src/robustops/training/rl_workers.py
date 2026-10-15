"""Bound rollout concurrency, dispatch ready work, and persist sampler state."""

import asyncio
from collections import Counter
import itertools
import json
from pathlib import Path
from time import perf_counter

import ray
from verl.experimental.agent_loop.agent_loop import AgentLoopManager, AgentLoopWorker
from robustops.training.recovery_rl_core import GroupPlanner, write_json
from robustops.training.rl_action_feedback import group_signal_diagnostics


class RecoveryWorker(AgentLoopWorker):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gate = asyncio.Semaphore(self.config.recovery.concurrency_per_worker)

    async def _run_agent_loop(self, *args, **kwargs):
        async with self.gate:
            return await super()._run_agent_loop(*args, **kwargs)


class JointWorker(RecoveryWorker):
    async def run_member(self, sampling, trajectory, kwargs):
        return await self._run_agent_loop(sampling, trajectory, trace=False, **kwargs)

    async def prepare_anchor(self, row, plan, opportunity_path, output):
        from robustops.training.rl_agent import prepare_anchor

        async with self.gate:
            return await prepare_anchor(
                self.config.recovery,
                self.tokenizer,
                row,
                plan,
                opportunity_path,
                output,
            )

    def pack_members(self, outputs, non_tensor_batch):
        return self._postprocess(
            outputs, input_non_tensor_batch=non_tensor_batch, validate=False
        )


class ReadyQueue:
    """Only ready work occupies a worker slot; dependent placeholders consume none."""

    def __init__(self, workers, per_worker, on_change=None):
        self.on_change = on_change
        self.queue = asyncio.PriorityQueue()
        self.sequence = itertools.count()
        self.active, self.peak = 0, 0
        self.started = perf_counter()
        self.capacity = len(workers) * per_worker
        self.events = []
        self.remote_calls = []
        self.tasks = [
            asyncio.create_task(self.consume(worker))
            for worker in workers
            for _ in range(per_worker)
        ]

    def submit(self, priority, method, *args):
        future = asyncio.get_running_loop().create_future()
        job = next(self.sequence)
        event = dict(
            job=job,
            method=method,
            priority=priority,
            submitted_s=perf_counter() - self.started,
        )
        if method == "run_member":
            event.update(
                flow=args[-1]["joint_flow"],
                slot=args[-1]["joint_slot"],
                sample=args[-1]["recovery_sample"],
            )
        else:
            event.update(parent=args[0]["joint_parent_slot"], kind=args[1]["kind"])
        self.events.append(event)
        self.queue.put_nowait((priority, job, method, args, future))
        if self.on_change:
            self.on_change()
        return future

    async def consume(self, worker):
        while True:
            _, job, method, args, future = await self.queue.get()
            event = self.events[job]
            if future.cancelled():
                self.queue.task_done()
                continue
            event["started_s"] = perf_counter() - self.started
            self.active += 1
            self.peak = max(self.peak, self.active)
            if self.on_change:
                self.on_change()
            try:
                remote_call = asyncio.ensure_future(
                    getattr(worker, method).remote(*args)
                )
                self.remote_calls.append(remote_call)
                value = await asyncio.shield(remote_call)
                if not future.done():
                    future.set_result(value)
            except Exception as exc:
                if not future.done():
                    future.set_exception(exc)
            finally:
                event["finished_s"] = perf_counter() - self.started
                self.active -= 1
                self.queue.task_done()
                if self.on_change:
                    self.on_change()

    def summary(self):
        wall = perf_counter() - self.started
        busy = sum(e["finished_s"] - e["started_s"] for e in self.events)
        wait = sum(e["started_s"] - e["submitted_s"] for e in self.events)
        anchors = [e for e in self.events if e["method"] == "prepare_anchor"]
        return dict(
            wall_s=wall,
            capacity=self.capacity,
            busy_slot_s=busy,
            idle_slot_s=max(0.0, wall * self.capacity - busy),
            queue_wait_sum_s=wait,
            anchor_service_sum_s=sum(e["finished_s"] - e["started_s"] for e in anchors),
            slot_occupancy=busy / (wall * self.capacity),
            jobs=len(self.events),
        )

    async def close(self):
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
        # Cancelling a local await does not stop a remote Ray episode. Drain
        # already dispatched calls before the trainer can save actor state.
        await asyncio.gather(*self.remote_calls, return_exceptions=True)


class RolloutManagerBase(AgentLoopManager):
    def __init__(self, *args, **kwargs):
        self.agent_loop_workers_class = ray.remote(JointWorker)
        super().__init__(*args, **kwargs)
        self.planner = GroupPlanner(self.config.recovery.seed)
        self.validation_call = 0
        self.next_visit = 0
        resume = getattr(self.config.recovery, "resume_checkpoint", None)
        if resume:
            state = json.loads((Path(resume) / "recovery_state.json").read_text())
            self.planner.load_state_dict(state["planner"])
            self.validation_call = state["validation_call"]
            self.next_visit = state["next_visit"]

    def save_joint_state(self, step):
        cfg = self.config.recovery
        state = dict(
            step=step,
            next_visit=self.next_visit,
            validation_call=self.validation_call,
            planner=self.planner.state_dict(),
            schedule="independent_16_16",
        )
        write_json(Path(cfg.output) / "recovery_states" / f"{step}.json", state)

    def report_joint(self, step, output, selected, fallbacks, peak):
        metrics = list(output.non_tensor_batch["recovery_metrics"])
        flows = list(output.non_tensor_batch["recovery_flow"])
        report = dict(
            step=step,
            update_step=step,
            validation=False,
            episodes=32 * 4,
            groups=32,
            parent_tasks=32,
            planned_fault_parents=16,
            planned_fault_parent_fraction=0.5,
            enabled=sum(m["fault_enabled"] for m in metrics),
            hit=sum(m["actual_hit"] for m in metrics),
            strict_successes=sum(m["strict_success"] for m in metrics),
            strict_mixed_groups=sum(
                len({m["strict_success"] for m in metrics[i : i + 4]}) > 1
                for i in range(0, 32 * 4, 4)
            ),
            shaped_mixed_groups=sum(
                len({m["reward"] for m in metrics[i : i + 4]}) > 1
                for i in range(0, 32 * 4, 4)
            ),
            flow_rows=dict(Counter(flows)),
            fallbacks=fallbacks,
            selections=selected,
            active_episode_peak=peak,
            fault_balance=self.planner.state_dict(),
        )
        report["signal_by_flow"] = {
            flow: value
            for flow, value in group_signal_diagnostics(
                metrics, flows, [i // 4 for i in range(len(metrics))]
            ).items()
            if flow in flows
        }
        for condition in (False, True):
            local = [m for m in metrics if bool(m["actual_hit"]) == condition]
            report["fault" if condition else "clean"] = dict(
                total=len(local), successes=sum(m["strict_success"] for m in local)
            )
        write_json(
            Path(self.config.recovery.output) / "diagnostics" / f"{step}.json", report
        )
