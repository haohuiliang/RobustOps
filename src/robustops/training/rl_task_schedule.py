"""Deterministic task schedules and reserve pools for paired RL runs."""

from collections import defaultdict
import random

CLEAN, FAULT, GROUPS = 16, 16, 32


class TaskStream:
    """Shuffled strata with proportional interleaving; collisions defer, never drop."""

    def __init__(self, rows, seed):
        self.rows, self.seed, self.epoch, self.pending, self.consumed = (
            rows,
            seed,
            -1,
            [],
            0,
        )

    def refill(self):
        self.epoch += 1
        rng = random.Random(f"{self.seed}/epoch/{self.epoch}")
        strata = defaultdict(list)
        for row in self.rows:
            strata[(row["domain"], row["mode"])].append(row)
        merged = []
        for key in sorted(strata):
            values = list(strata[key])
            rng.shuffle(values)
            phase = rng.random()
            merged.extend(
                ((i + phase) / len(values), key, dict(row, epoch=self.epoch))
                for i, row in enumerate(values)
            )
        self.pending.extend(
            row for _, _, row in sorted(merged, key=lambda x: (x[0], x[1]))
        )

    def take(self, used):
        if not self.pending:
            self.refill()
        used_tasks = {task for task, _ in used}
        for require_distinct_task in (True, False):
            for i, row in enumerate(self.pending):
                cell = (row["task_id"], row["mode"])
                if cell not in used and (
                    not require_distinct_task or cell[0] not in used_tasks
                ):
                    self.consumed += 1
                    return self.pending.pop(i)
        # A cross-stream collision at the tail can defer a few old-epoch cells.
        self.refill()
        return self.take(used)

    def state(self):
        # The frozen ordered task schedule + step is the replayable cursor; queue
        # internals can be regenerated from the seed without serializing them per step.
        return dict(epoch=self.epoch, consumed=self.consumed, pending=len(self.pending))


def task_schedule(rows, seed, steps):
    if len({r["task_id"] for r in rows}) < GROUPS:
        raise ValueError("independent schedule needs at least 32 base tasks")
    streams = {name: TaskStream(rows, f"{seed}/{name}") for name in ("clean", "fault")}
    batches = []
    for step in range(1, steps + 1):
        used, selected = set(), {}
        for flow, count in (("clean", CLEAN), ("fault", FAULT)):
            selected[flow] = []
            for _ in range(count):
                row = streams[flow].take(used)
                used.add((row["task_id"], row["mode"]))
                selected[flow].append(row)
        groups = []
        for flow in ("clean", "fault"):
            for source in selected[flow]:
                slot = len(groups)
                visit = (step - 1) * GROUPS + slot
                groups.append(
                    dict(
                        source,
                        visit=visit,
                        parent_visit=visit,
                        joint_slot=slot,
                        joint_parent_slot=slot,
                        joint_fault_parents=list(range(CLEAN, CLEAN + FAULT)),
                        planned_flow=flow,
                        independent_tasks=True,
                    )
                )
        batches.append(
            dict(
                step=step,
                groups=groups,
                streams={k: s.state() for k, s in streams.items()},
            )
        )
    return batches


def fault_reserve_schedule(rows, seed, steps):
    """Finite run-local reserves outside each complete 30-step planned window."""
    windows = (steps + 29) // 30
    batches = task_schedule(rows, seed, windows * 30)
    result = {}
    for window in range(windows):
        excluded = {
            (r["task_id"], r["mode"])
            for b in batches[window * 30 : (window + 1) * 30]
            for r in b["groups"]
        }
        stream = TaskStream(rows, f"{seed}/fault_reserve/{window}")
        stream.refill()
        reserve = [
            r for r in stream.pending if (r["task_id"], r["mode"]) not in excluded
        ]
        result[str(window)] = dict(
            start_step=window * 30 + 1,
            end_step=(window + 1) * 30,
            excluded_cells=sorted(excluded),
            rows=reserve,
        )
    return result
