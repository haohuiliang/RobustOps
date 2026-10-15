"""Fault planning, verifier metrics, and JSON helpers for recovery training."""

from collections import Counter, defaultdict
import json
import random
from pathlib import Path

from robustops.evaluation.fault_schedule import (
    GROUPS,
    shared_strategy,
    draw_repeat_limit,
)
from robustops.evaluation.fault_scheduling_controller import PRIORITY

KINDS = tuple(k for k in PRIORITY if k != "ignored_filter")


def verifier_metrics(initial, final):
    if not initial or len(initial) != len(final):
        raise ValueError("initial/final verifier coverage mismatch")
    for a, b in zip(initial, final, strict=True):
        if (a["index"], a["name"]) != (b["index"], b["name"]):
            raise ValueError("verifier identity changed")
        if type(a["passed"]) is not bool or type(b["passed"]) is not bool:
            raise ValueError("invalid verifier outcome is not a negative reward")
    q0 = sum(x["passed"] for x in initial) / len(initial)
    qt = sum(x["passed"] for x in final) / len(final)
    success = int(all(x["passed"] for x in final))
    gain = max(0.0, qt - q0) / (1 - q0) if q0 < 1 else 0.0
    return dict(
        strict_success=success,
        initial_fraction=q0,
        final_fraction=qt,
        net_progress=gain,
        regressed_checks=sum(
            a["passed"] and not b["passed"] for a, b in zip(initial, final)
        ),
        reward=float(success),
    )


class GroupPlanner:
    """Actual-hit balance across batches; provisional reservations within a batch.

    Like FaultScheduler, priority is lexicographic, never a probability weight.
    The seen tie-break follows balance counts and precedes random tie breaks.
    """

    def __init__(self, seed):
        self.seed = seed
        self.groups = Counter()
        self.labels = Counter()
        self.local_groups = defaultdict(Counter)
        self.local_labels = defaultdict(Counter)

    def claim(self, row, kind):
        if kind not in (PRIORITY if row.get("validation") else KINDS):
            raise ValueError("held-out/unknown fault in training")
        local = (row["domain"], row["mode"])
        self.groups[GROUPS[kind]] += 1
        self.labels[kind] += 1
        self.local_groups[local][GROUPS[kind]] += 1
        self.local_labels[local][kind] += 1

    def state_dict(self):
        return dict(
            seed=self.seed,
            groups=dict(self.groups),
            labels=dict(self.labels),
            local_groups=[[list(k), dict(v)] for k, v in self.local_groups.items()],
            local_labels=[[list(k), dict(v)] for k, v in self.local_labels.items()],
        )

    def load_state_dict(self, state):
        if state["seed"] != self.seed:
            raise ValueError("fault scheduler seed changed on resume")
        self.groups = Counter(state["groups"])
        self.labels = Counter(state["labels"])
        self.local_groups = defaultdict(
            Counter, {tuple(k): Counter(v) for k, v in state["local_groups"]}
        )
        self.local_labels = defaultdict(
            Counter, {tuple(k): Counter(v) for k, v in state["local_labels"]}
        )

    def reserve(
        self, row, visit, *, forced=None, available=None, claim=True, repeat_config=None
    ):
        rng = random.Random(f"{self.seed}/visit/{visit}")
        enabled = rng.random() < 0.5 if forced is None else forced
        allowed = PRIORITY if row.get("validation") else KINDS
        kinds = [
            k
            for k in allowed
            if any(k in t["faults"] for t in row["selection"]["tools"])
        ]
        if available is not None:
            kinds = [k for k in kinds if k in available]
        if enabled and not kinds:
            raise ValueError(
                f"no train fault candidates: {row['task_id']} {row['mode']}"
            )
        ties = list(allowed)
        rng.shuffle(ties)
        local = (row["domain"], row["mode"])
        seen = set(row.get("sft_seen", [])) if row["epoch"] == 0 else set()
        groups = sorted(set(GROUPS[k] for k in allowed))
        rng.shuffle(groups)
        ranked = sorted(
            kinds,
            key=lambda k: (
                self.groups[GROUPS[k]],
                int(
                    GROUPS[k] not in {"unknown_commit", "reply_mixup", "ignored_filter"}
                ),
                self.local_groups[local][GROUPS[k]],
                groups.index(GROUPS[k]),
                self.labels[k],
                self.local_labels[local][k],
                int(k in seen),
                ties.index(k),
            ),
        )
        kind = ranked[0] if enabled else None
        budget = (
            row["candidate_budgets"][kind]
            if enabled and row.get("validation")
            else draw_repeat_limit(
                kind, repeat_config or shared_strategy(), f"rl/{self.seed}/{visit}"
            )
            if enabled
            else 0
        )
        plan = dict(
            visit=visit,
            task_id=row["task_id"],
            mode=row["mode"],
            epoch=row["epoch"],
            fault_enabled=enabled,
            kind=kind,
            budget=budget,
            ranked_candidates=ranked,
            seed=rng.randrange(2**30),
        )
        if enabled and row.get("validation"):
            plan["budget"] = row["candidate_budgets"][kind]
        if kind and claim:
            self.claim(row, kind)
        return plan


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def load_jsonl(path):
    return [
        json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()
    ]
