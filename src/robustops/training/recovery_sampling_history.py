"""Committed replacement epochs and per-task fault diversity across resumes."""

from copy import deepcopy
import json
from pathlib import Path


def cell_key(row):
    return json.dumps([row["task_id"], row["mode"]], separators=(",", ":"))


class SamplingHistory:
    def __init__(self, state=None):
        state = deepcopy(state or {})
        self.seen = set(state.get("seen", []))
        self.faults = state.get("faults", {})
        self.reserves = state.get("reserves", {})

    @classmethod
    def restore(cls, saved, output):
        if "sampling_history" in saved:
            return cls(saved["sampling_history"])
        if saved.get("step", 0):
            raise ValueError("checkpoint is missing sampling history")
        return cls()

    def mark(self, row):
        self.seen.add(cell_key(row))

    def claim_fault(self, row, kind):
        counts = self.faults.setdefault(cell_key(row), {})
        counts[kind] = counts.get(kind, 0) + 1
        self.mark(row)

    def prefer_faults(self, row, available):
        counts = self.faults.get(cell_key(row), {})
        minimum = min(counts.get(kind, 0) for kind in available)
        return {
            kind: path
            for kind, path in available.items()
            if counts.get(kind, 0) == minimum
        }

    def take(self, window, rows, cursor, used):
        if not rows:
            raise RuntimeError(
                "empty replacement pool outside the complete 30-step plan"
            )
        state = self.reserves.setdefault(
            window, dict(epoch=0, pending=list(range(cursor, len(rows))))
        )
        if not state["pending"]:
            state.update(epoch=state["epoch"] + 1, pending=list(range(len(rows))))
        candidates = [
            i
            for i in state["pending"]
            if (rows[i]["task_id"], rows[i]["mode"]) not in used
        ]
        if not candidates:
            # Never start another epoch while this epoch still has unconsumed
            # cells. A pathological all-zero step remains a diagnosed failure.
            raise RuntimeError(
                "replacement reserve exhausted for this step: remaining epoch cells already sampled"
            )
        index = min(candidates, key=lambda i: (cell_key(rows[i]) in self.seen, i))
        state["pending"].remove(index)
        row = dict(rows[index], epoch=state["epoch"])
        self.mark(row)
        return row

    def state_dict(self):
        return dict(
            seen=sorted(self.seen),
            faults=deepcopy(self.faults),
            reserves=deepcopy(self.reserves),
        )
