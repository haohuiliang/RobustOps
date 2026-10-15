"""Continuous training-task queues.

Training length is not a sampling boundary. Clean and fault streams retain
their cursors across resume; delayed replay has its own queue.
"""

from copy import deepcopy
import json
from pathlib import Path

from robustops.training.rl_task_schedule import TaskStream


def key(row):
    return (row["task_id"], row["mode"])


def load_pool(directory):
    directory = Path(directory)
    path = directory / "task_pool.json"
    if path.exists():
        return json.loads(path.read_text())["rows"]
    raise ValueError("saved task_pool.json is required to restore continuous queues")


class ContinuousTaskQueues:
    def __init__(self, rows, seed, *, state=None):
        self.rows = {key(r): deepcopy(r) for r in rows}
        if len(self.rows) != len(rows):
            raise ValueError("duplicate task×mode in queue pool")
        self.seed = seed
        self.serial = 0
        self.streams = {}
        if state is not None:
            if state["seed"] != seed or {tuple(k) for k in state["pool"]} != set(
                self.rows
            ):
                raise ValueError("continuous task queue seed or pool changed on resume")
            self.serial = state["serial"]
            self.streams = deepcopy(state["streams"])
            for stream in self.streams.values():
                stream.setdefault("deferred", [])
        else:
            for flow in ("clean", "fault"):
                order = self.order(flow, 0)
                self.streams[flow] = dict(
                    epoch=0, consumed=0, pending=[list(k) for k in order], deferred=[]
                )

    def order(self, flow, epoch):
        stream = TaskStream(
            [self.rows[k] for k in sorted(self.rows)], f"{self.seed}/queue/{flow}"
        )
        stream.epoch = epoch - 1
        stream.refill()
        return [key(r) for r in stream.pending]

    @classmethod
    def restore(cls, directory, seed, saved, output):
        rows = load_pool(directory)
        if "task_queues" in saved:
            return cls(rows, seed, state=saved["task_queues"])
        if saved.get("step", 0):
            raise ValueError("checkpoint is missing continuous task queue state")
        return cls(rows, seed)

    def take(self, flow, used, source="train"):
        if source != "train":
            raise ValueError("Train-only queue cannot sample Val")
        stream = self.streams[flow]
        if not any(k not in used for k in self.rows):
            raise RuntimeError("all queue tasks already used in current step")
        while True:
            # Always drain eligible older tails before the newest epoch. Keep
            # each tail's original epoch and counts, including across resume.
            for cohort in [*stream["deferred"], stream]:
                for i, item in enumerate(cohort["pending"]):
                    if tuple(item) not in used:
                        cohort["pending"].pop(i)
                        cohort["consumed"] += 1
                        self.serial += 1
                        if cohort is not stream and not cohort["pending"]:
                            stream["deferred"].remove(cohort)
                        return dict(self.rows[tuple(item)], epoch=cohort["epoch"])
            # Remaining entries all conflict with this step. Preserve them,
            # then open the next epoch instead of stopping a valid rollout.
            if stream["pending"]:
                stream["deferred"].append(
                    dict(
                        epoch=stream["epoch"],
                        consumed=stream["consumed"],
                        pending=stream["pending"],
                    )
                )
            stream["epoch"] += 1
            stream["consumed"] = 0
            stream["pending"] = [list(k) for k in self.order(flow, stream["epoch"])]

    def summary(self):
        return {
            flow: dict(
                epoch=s["epoch"],
                consumed=s["consumed"],
                pending=len(s["pending"]),
                deferred_epochs=len(s["deferred"]),
                deferred_pending=sum(len(c["pending"]) for c in s["deferred"]),
            )
            for flow, s in self.streams.items()
        }

    def state_dict(self):
        return dict(
            seed=self.seed,
            pool=[list(k) for k in sorted(self.rows)],
            serial=self.serial,
            streams=deepcopy(self.streams),
        )

    def primary_source(self, step, slot):
        return "train"


def restore_task_queues(directory, seed, saved, output):
    return ContinuousTaskQueues.restore(directory, seed, saved, output)


def carrier_batches(rows, start, stop):
    """Dataloader carries visit/slot positions; the runtime queue chooses tasks."""
    templates = sorted(rows, key=key)[:32]
    if len(templates) != 32:
        raise ValueError("32 queue carrier slots required")
    return [
        dict(
            step=step,
            streams={},
            groups=[
                dict(
                    row,
                    epoch=0,
                    visit=(step - 1) * 32 + slot,
                    parent_visit=(step - 1) * 32 + slot,
                    joint_slot=slot,
                    joint_parent_slot=slot,
                    joint_fault_parents=list(range(16, 32)),
                    planned_flow="clean" if slot < 16 else "fault",
                    independent_tasks=True,
                )
                for slot, row in enumerate(templates)
            ],
        )
        for step in range(start, stop + 1)
    ]
