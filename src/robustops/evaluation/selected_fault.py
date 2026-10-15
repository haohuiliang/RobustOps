"""Single selected mechanism, finite event budget, existing per-call injectors.
Strict runtime controller; does not alter frozen historical policies.
"""

from copy import deepcopy
from robustops.evaluation.tool_names import is_read_name
from collections import Counter
from robustops.evaluation.tool_fault import ToolFault
from robustops.evaluation.qualified_commit import make_selected_write_fault
from robustops.evaluation.filter_fault import ObservedFilterFault, READ, CONTROL


class SelectedFaultController:
    def __init__(
        self,
        payload,
        selection,
        kind,
        budget=3,
        binding="eligible_calls",
        footprint=None,
    ):
        if type(budget) is not int or budget not in (0, 1, 2, 3, 4):
            raise ValueError("finite 0/1/2/3/4 budget required")
        if kind not in (
            "transient_unexecuted",
            "truncated_read",
            "partial_read",
            "unknown_commit",
            "unknown_commit_create",
            "ignored_filter",
            "reply_mixup",
        ):
            raise ValueError("unsupported/disabled mechanism")
        if binding not in ("same_operation", "eligible_calls"):
            raise ValueError("invalid binding")
        self.payload = payload
        self.selection = (
            selection["tools"] if isinstance(selection, dict) else selection
        )
        self.kind = kind
        self.budget = budget
        self.binding = binding
        self.footprint = footprint
        self.calls = []
        self.events = []
        self.bound = None
        self.caller = None
        self.targets = {t["tool"] for t in self.selection if kind in t["faults"]}
        # The burst arm is specifically a failed write, retaining a pending operation.
        if kind == "transient_unexecuted":
            declared_reads = {
                t["tool"]
                for t in self.selection
                if t.get("role") not in (None, "write")
            }
            self.targets = {
                n
                for n in self.targets
                if n not in declared_reads and not is_read_name(n)
            }

    def bind(self, caller, tools):
        if self.caller is not None:
            raise ValueError("new controller per episode")
        self.caller = caller
        self.routes = caller.routes
        self.tools = tools
        self.payload = dict(self.payload, tools=tools, tool_servers=dict(caller.routes))
        self.schemas = {
            t["function"]["name"]: t["function"]["parameters"] for t in tools
        }
        self.rejected_targets = {
            n: "unexposed selected target"
            for n in self.targets
            if n not in self.schemas
        }
        if self.kind in (
            "truncated_read",
            "partial_read",
            "ignored_filter",
            "reply_mixup",
        ):
            self.rejected_targets.update(
                {
                    n: "unsupported pure-read name boundary"
                    for n in self.targets
                    if n in self.schemas and not is_read_name(n)
                }
            )
        self.targets -= self.rejected_targets.keys()
        if self.kind == "reply_mixup":
            if self.footprint is None:
                raise ValueError("reply token counter required")
            from robustops.evaluation.reply_fault import ScopedReplyFault

            self.reply = ScopedReplyFault(self.targets, self.footprint).bind(
                caller, tools
            )
        return self

    @property
    def fault_call_audit(self):
        return self.calls

    async def call_tool(self, name, args):
        available = len(self.events) < self.budget
        matches = name in self.targets and (
            self.bound is None
            or self.binding == "eligible_calls"
            or (name, args) == self.bound
        )
        eligible = available and matches
        if self.kind == "reply_mixup":
            self.reply.phase = "initial" if eligible else "history_only"
            result = await self.reply.call_tool(name, args)
            entry = deepcopy(self.reply.calls[-1])
        elif not eligible:
            result = await self.caller.call_tool(name, args)
            entry = dict(
                tool=name,
                arguments=deepcopy(args),
                actual_result=deepcopy(result),
                visible_result=deepcopy(result),
                injected=False,
                backend_dispatch_count=1,
            )
            entry["controller_skip"] = (
                "budget_exhausted"
                if not available
                else "outside_bound_operation"
                if name in self.targets
                else "not_selected"
            )
        else:
            if self.kind.startswith("unknown_commit"):
                f = make_selected_write_fault(
                    self.payload, self.selection, name, self.kind
                ).bind(self.caller, self.tools)
            elif self.kind == "ignored_filter":
                schema = self.schemas[name]
                params = sorted(
                    set(args)
                    & set(schema.get("properties", {}))
                    - set(schema.get("required", []))
                    - CONTROL
                )
                if params:
                    f = ObservedFilterFault(name, params[0]).bind(
                        self.caller, self.tools
                    )
                else:
                    f = ToolFault({"kind": "observe"}).bind(self.caller, self.tools)
            else:
                k = {
                    "transient_unexecuted": "pre_dispatch",
                    "truncated_read": "truncated_read",
                    "partial_read": "partial_read",
                }[self.kind]
                f = ToolFault(
                    dict(
                        kind=k,
                        tool=name,
                        server=self.routes[name],
                        count=1,
                        **({} if k == "pre_dispatch" else {"read_only": True}),
                    )
                ).bind(self.caller, self.tools)
            result = await f.call_tool(name, args)
            entry = deepcopy(f.calls[-1])
        if name in self.rejected_targets:
            entry["eligibility_skip"] = self.rejected_targets[name]
        entry.update(call_index=len(self.calls) + 1, selected_fault=self.kind)
        entry.setdefault("backend_dispatch_count", int(entry.get("dispatched", True)))
        if entry.get("injected"):
            if self.bound is None:
                self.bound = (name, deepcopy(args))
            entry["injection_index"] = len(self.events) + 1
            self.events.append(deepcopy(entry))
        self.calls.append(entry)
        return result

    def report(self):
        hits = len(self.events)
        return dict(
            enabled=self.budget > 0,
            triggered=hits > 0,
            config=dict(kind=self.kind, budget=self.budget, binding=self.binding),
            planned_injections=self.budget,
            injected_calls=hits,
            actual_injections=hits,
            quota_fully_triggered=self.budget > 0 and hits == self.budget,
            repeated_events_triggered=hits >= 2,
            budget_exhausted=hits >= self.budget,
            rollout_stop_required=False,
            remaining_budget=self.budget - hits,
            qualification_read_calls=sum(
                len(c.get("probes", [])) + int("probe_result" in c) for c in self.calls
            ),
            skip_reasons=dict(
                Counter(
                    c.get("eligibility_skip") or c.get("controller_skip")
                    for c in self.calls
                    if c.get("eligibility_skip") or c.get("controller_skip")
                )
            ),
            events=deepcopy(self.events),
            call_audit=deepcopy(self.calls),
        )
