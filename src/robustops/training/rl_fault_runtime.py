"""Real SQL/controller/token snapshots for clean donors and shared recovery states."""

from copy import deepcopy
import random
from robustops.evaluation.fault_scheduling_controller import RuntimePriorityFault
from robustops.envs.enterpriseops.runtime import InfrastructureError

REPLY_FIELDS = (
    "previous",
    "phase",
    "matches",
    "bound",
    "pending",
    "read_restart",
    "enumeration_recovery",
    "calls",
)


class JointRuntime(RuntimePriorityFault):
    def __init__(
        self, *args, token_state, anchor_only=False, point_seed=None, limits=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.point_seed = point_seed
        self.point_priorities = {}
        self.point_records = {}
        self.token_state = token_state
        from robustops.training.rl_config import RUNTIME_DEFAULTS
        limits = limits or RUNTIME_DEFAULTS
        self.max_turns = limits["max_turns"]
        self.max_context_tokens = limits["max_context_tokens"]
        self.max_generation_tokens = limits["max_generation_tokens"]
        self.anchor_only = anchor_only
        self.stop_after_capture = False
        self.anchor_checkpoint = None
        self.joint_restore = None

    def point_priority(self, kind, index):
        return random.Random(f"{self.point_seed}/fault-point/{kind}/{index}").random()

    def point_has_suffix_room(self):
        tokens = self.token_state()
        return (
            tokens["turns"] < self.max_turns
            and len(tokens["ids"]) + self.max_generation_tokens
            <= self.max_context_tokens
        )

    def checkpoint_candidates(self, name, index):
        if self.point_seed is None:
            return super().checkpoint_candidates(name, index)
        if not self.point_has_suffix_room():
            return set()
        # Independent random priorities give every qualifying position the same
        # selection probability. Capture only when a priority can beat the saved
        # one; qualification itself must still succeed before replacing it.
        return {
            kind
            for kind, h in self.handlers.items()
            if name in h.targets
            and self.point_priority(kind, index) < self.point_priorities.get(kind, 1.0)
        }

    def remember_opportunity(self, opportunity, checkpoint):
        if self.point_seed is None:
            return super().remember_opportunity(opportunity, checkpoint)
        opportunity = dict(opportunity, suffix_eligible=self.point_has_suffix_room())
        self.opportunities.append(opportunity)
        if checkpoint is not None:
            kind, index = opportunity["kind"], opportunity["call_index"]
            self.checkpoints[kind] = checkpoint
            self.point_priorities[kind] = self.point_priority(kind, index)
            self.point_records.setdefault(kind, []).append(checkpoint)

    def finalize_points(self):
        if self.point_seed is None:
            return
        for kind, records in self.point_records.items():
            points = [
                p
                for p in self.opportunities
                if p["kind"] == kind and p["suffix_eligible"]
            ]
            # 1: whole eligible list; 2: first half; 4: first quarter. Ceil
            # retains at least one point for short trajectories. Record minima
            # suffice to select uniformly in any prefix without saving all states.
            budget = self.budgets[kind]
            count = (len(points) + budget - 1) // budget
            last = points[count - 1]["call_index"]
            self.checkpoints[kind] = next(
                p for p in reversed(records) if p["call_index"] <= last
            )

    def point_options(self, kind, levels):
        """Prepare each possible cap's window without drawing the selected cap."""
        if self.point_seed is None:
            return {cap: self.checkpoints[kind] for cap in levels}
        points = [
            p for p in self.opportunities if p["kind"] == kind and p["suffix_eligible"]
        ]
        records = self.point_records[kind]
        return {
            cap: next(
                p
                for p in reversed(records)
                if p["call_index"]
                <= points[(len(points) + cap - 1) // cap - 1]["call_index"]
            )
            for cap in levels
        }

    def point_metadata(self, kind, total_calls, *, budget=None, snapshot=None):
        budget = self.budgets[kind] if budget is None else budget
        chosen = (snapshot or self.checkpoints[kind])["call_index"]
        points = [
            p
            for p in self.opportunities
            if p["kind"] == kind and p.get("suffix_eligible", True)
        ]
        rank = next(i + 1 for i, p in enumerate(points) if p["call_index"] == chosen)
        window = (
            (len(points) + budget - 1) // budget if self.point_seed is not None else 1
        )
        return dict(
            policy="dose_window_uniform"
            if self.point_seed is not None
            else "earliest_strictly_eligible",
            planned_dose=budget,
            eligible_window_count=window,
            prefix_snapshot_count=len(self.point_records.get(kind, [])),
            seed=self.point_seed,
            call_index=chosen,
            donor_total_calls=total_calls,
            call_fraction=chosen / total_calls,
            eligible_rank=rank,
            eligible_count=len(points),
            donor_later_opportunities=len(points) - rank,
        )

    def bind(self, caller, tools):
        super().bind(caller, tools)
        self.tools = deepcopy(tools)
        if self.joint_restore is not None:
            state = self.joint_restore
            if state["tools"] != tools or state["routes"] != dict(self.routes):
                raise InfrastructureError("joint snapshot tool schema/routes changed")
            runtime = state["runtime"]
            self.calls, self.events = (
                deepcopy(runtime["calls"]),
                deepcopy(runtime["events"]),
            )
            self.call_offset, self.bound = runtime["call_offset"], runtime["bound"]
            for kind, h in self.handlers.items():
                saved = runtime["handlers"][kind]
                for field in ("calls", "events", "bound"):
                    setattr(h, field, deepcopy(saved[field]))
                if kind == "reply_mixup":
                    for field in REPLY_FIELDS:
                        setattr(h.reply, field, deepcopy(saved["reply"][field]))
        return self

    async def capture_checkpoint(self, index):
        databases = {
            name: await env.export_seed_state()
            for name, env in self.caller.environments.items()
        }
        handlers = {}
        for kind, handler in self.handlers.items():
            state = {
                field: deepcopy(getattr(handler, field))
                for field in ("calls", "events", "bound")
            }
            if kind == "reply_mixup":
                state["reply"] = {
                    field: deepcopy(getattr(handler.reply, field))
                    for field in REPLY_FIELDS
                }
            handlers[kind] = state
        return dict(
            call_index=index,
            databases=databases,
            tools=deepcopy(self.tools),
            routes=dict(self.routes),
            runtime=dict(
                calls=deepcopy(self.calls),
                events=deepcopy(self.events),
                bound=self.bound,
                call_offset=self.call_offset,
                handlers=handlers,
            ),
            tokens=deepcopy(self.token_state()),
            budgets=dict(self.budgets),
            observe_only=self.observe_only,
        )

    async def call_tool(self, name, arguments):
        value = await super().call_tool(name, arguments)
        if self.anchor_only:
            if len(self.events) != 1 or not self.calls[-1]["injected"]:
                raise InfrastructureError(
                    "joint replay anchor must inject exactly once"
                )
            self.anchor_checkpoint = await self.capture_checkpoint(
                self.call_offset + len(self.calls) + 1
            )
            self.stop_after_capture = True
        return value

    def report(self):
        report = super().report()
        if self.point_seed is not None:
            report["point_policy"] = "dose_window_uniform"
        elif self.joint_restore and self.joint_restore.get("injection_point"):
            report["point_policy"] = self.joint_restore["injection_point"]["policy"]
        report.update(
            enabled=not self.observe_only and bool(self.kinds),
            observe_only=self.observe_only,
            checkpoint_restored=self.joint_restore is not None,
            inherited_call_count=(
                len(self.joint_restore["runtime"]["calls"])
                + self.joint_restore["runtime"]["call_offset"]
            )
            if self.joint_restore
            else 0,
        )
        return report


def attach_prefix(snapshot, trajectory, *, before=False):
    """Only executed history through the boundary is inherited; later events vanish."""
    state = deepcopy(snapshot)
    turn = state["tokens"]["turns"] - int(before)
    state["prefix"] = dict(
        turns=[t for t in trajectory["turns"] if t["turn"] <= turn],
        tool_events=[e for e in trajectory["tool_events"] if e["turn"] <= turn],
    )
    return state
