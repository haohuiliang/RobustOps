"""Schedule runtime fault candidates with one outcome-blind no-hit replay."""

from copy import deepcopy
import json
from robustops.envs.enterpriseops.runtime import InfrastructureError
from robustops.evaluation.tool_names import is_read_name

PRIORITY = (
    "unknown_commit",
    "unknown_commit_create",
    "ignored_filter",
    "partial_read",
    "truncated_read",
    "reply_mixup",
    "transient_unexecuted",
)
RARE = frozenset(PRIORITY[:3])


class CallFrame:
    """One real original call; memoized read-only qualification probes per boundary."""

    def __init__(self, caller):
        self.caller = caller
        self.routes = caller.routes

    def begin(self, name, arguments):
        self.original = (name, deepcopy(arguments))
        self.cache = {}
        self.dispatches = []

    async def call_tool(self, name, arguments):
        if (name, arguments) != self.original and not is_read_name(name):
            raise InfrastructureError("qualification attempted an additional write")
        key = (name, json.dumps(arguments, sort_keys=True))
        if key not in self.cache:
            # A before-write witness must never be reused as an after-write read.
            if (name, arguments) == self.original and not is_read_name(name):
                self.cache.clear()
            result = await self.caller.call_tool(name, arguments)
            self.cache[key] = deepcopy(result)
            self.dispatches.append(
                dict(
                    tool=name,
                    arguments=deepcopy(arguments),
                    original=(name, arguments) == self.original,
                )
            )
        return deepcopy(self.cache[key])


class RuntimePriorityFault:
    """Wait for rare candidates; retain earliest qualified fallback opportunities.

    Factory is the existing SelectedFaultController, shared by historical studies.
    Shadow checks never alter the model-visible reply or consume injection budget.
    """

    def __init__(
        self,
        payload,
        selection,
        budgets,
        factory,
        footprint,
        replay_target=None,
        task_caps=None,
        task_key=None,
        schedule=None,
        balance=None,
    ):
        self.payload, self.selection = payload, selection
        self.task_caps, self.task_key = task_caps, task_key
        self.balance = balance
        self.observe_only = bool((schedule or {}).get("observe_only", False))
        self.draw_on_selection = bool((schedule or {}).get("draw_on_selection", False))
        self.schedule = schedule or {}
        self.capture_first_fault = bool(self.schedule.get("capture_first_fault", False))
        self.first_fault_checkpoint = None
        self.collect_opportunities = bool(
            (schedule or {}).get("collect_opportunities", True)
        )
        self.wait_for_priority = (schedule or {}).get(
            "reservation_mode"
        ) == "wait_for_priority_label"
        self.reservation_balance = balance.report() if balance is not None else None
        self.priority = tuple((schedule or {}).get("priority", PRIORITY))
        self.preferred = set((schedule or {}).get("preferred_mechanisms", RARE))
        if set(self.priority) != set(PRIORITY) or len(self.priority) != len(PRIORITY):
            raise ValueError("priority must contain each enabled label once")
        self.budgets, self.factory, self.footprint = budgets, factory, footprint
        entries = selection["tools"] if isinstance(selection, dict) else selection
        labels = {f for t in entries for f in t["faults"]}
        self.kinds = [k for k in self.priority if k in labels and budgets.get(k, 0) > 0]
        self.live = set(self.kinds) & self.preferred
        if not self.live and self.kinds:
            self.live = {self.kinds[0]}
        if self.balance is not None:
            if self.wait_for_priority:
                self.priority = tuple(
                    self.balance.rank(self.priority, self.task_key, self.preferred)
                )
                self.kinds = [k for k in self.priority if k in self.kinds]
                self.live = {self.kinds[0]} if self.kinds else set()
            else:
                self.live = set(self.kinds)
        self.replay_target = deepcopy(replay_target)
        self.bound = None
        self.calls, self.events, self.opportunities = [], [], []
        self.handlers = {}
        self.caller = None
        self.checkpoints = {}
        self.restore_checkpoint = None
        self.call_offset = 0

    def bind(self, caller, tools):
        if self.caller is not None:
            raise ValueError("new scheduler required per branch")
        self.caller, self.routes = caller, caller.routes
        self.frame = CallFrame(caller)
        for kind in self.kinds:
            self.handlers[kind] = self.factory(
                self.payload,
                self.selection,
                kind,
                self.budgets[kind],
                "eligible_calls",
                self.footprint,
            ).bind(self.frame, tools)
        if self.restore_checkpoint is not None:
            checkpoint = self.restore_checkpoint
            if (
                tools != checkpoint["tools"]
                or dict(self.routes) != checkpoint["routes"]
            ):
                raise InfrastructureError(
                    "checkpoint exposed tool schema/routes changed"
                )
            self.call_offset = checkpoint["call_index"] - 1
            state = checkpoint["controller"]
            if self.wait_for_priority:
                self.priority = tuple(state["priority"])
                self.kinds = [k for k in self.priority if k in self.kinds]
                self.live = set(state["live"])
                self.reservation_balance = deepcopy(state["reservation_balance"])
            for kind, h in self.handlers.items():
                h.calls = [dict(restored_prefix=True)] * state["handler_call_counts"][
                    kind
                ]
            if "reply_mixup" in self.handlers:
                reply = self.handlers["reply_mixup"].reply
                for key, value in state["reply_state"].items():
                    setattr(reply, key, deepcopy(value))
                reply.calls = [dict(restored_prefix=True)] * state["reply_call_count"]
            if "after_turn" in checkpoint:
                runtime = checkpoint["runtime"]
                self.calls, self.events = (
                    deepcopy(runtime["calls"]),
                    deepcopy(runtime["events"]),
                )
                self.call_offset, self.bound = runtime["call_offset"], runtime["bound"]
                for kind, h in self.handlers.items():
                    for field in ("calls", "events", "bound"):
                        setattr(h, field, deepcopy(runtime["handlers"][kind][field]))
        return self

    async def capture_checkpoint(self, index):
        if not hasattr(self.caller, "environments"):
            raise InfrastructureError(
                "checkpoint backend does not expose snapshot environments"
            )
        databases = {}
        for server, environment in self.caller.environments.items():
            databases[server] = await environment.export_seed_state()
        state = dict(
            handler_call_counts={k: len(h.calls) for k, h in self.handlers.items()}
        )
        if self.wait_for_priority:
            state.update(
                priority=list(self.priority),
                live=sorted(self.live),
                reservation_balance=deepcopy(self.reservation_balance),
            )
        if "reply_mixup" in self.handlers:
            reply = self.handlers["reply_mixup"].reply
            state["reply_state"] = {
                k: deepcopy(getattr(reply, k))
                for k in (
                    "previous",
                    "phase",
                    "matches",
                    "bound",
                    "pending",
                    "read_restart",
                    "enumeration_recovery",
                )
            }
            state["reply_call_count"] = len(reply.calls)
        tools = next(iter(self.handlers.values())).tools
        return dict(
            call_index=index,
            databases=databases,
            controller=state,
            tools=deepcopy(tools),
            routes=dict(self.routes),
            budgets=dict(self.budgets),
            runtime=dict(
                calls=deepcopy(self.calls),
                events=deepcopy(self.events),
                call_offset=self.call_offset,
                bound=self.bound,
                handlers={
                    kind: {
                        field: deepcopy(getattr(h, field))
                        for field in ("calls", "events", "bound")
                    }
                    for kind, h in self.handlers.items()
                },
            ),
        )

    async def after_tool_event(self, event):
        if (
            self.capture_first_fault
            and self.first_fault_checkpoint is None
            and len(self.events) == 1
            and not event.get("agent_protocol_error")
            and self.calls[-1]["injected"]
        ):
            self.first_fault_checkpoint = await self.capture_checkpoint(
                self.call_offset + len(self.calls) + 1
            )
            self.first_fault_checkpoint["after_turn"] = event["turn"]

    @property
    def fault_call_audit(self):
        return self.calls

    def checkpoint_candidates(self, name, index):
        return {
            kind
            for kind, h in self.handlers.items()
            if name in h.targets and kind not in self.checkpoints
        }

    def remember_opportunity(self, opportunity, checkpoint):
        kind = opportunity["kind"]
        if kind not in self.checkpoints and checkpoint is not None:
            self.checkpoints[kind] = checkpoint
        self.opportunities.append(opportunity)

    async def call_tool(self, name, arguments):
        index = self.call_offset + len(self.calls) + 1
        checkpoint = None
        checkpoint_kinds = set()
        if self.collect_opportunities and not self.replay_target and not self.events:
            checkpoint_kinds = self.checkpoint_candidates(name, index)
            if checkpoint_kinds:
                checkpoint = await self.capture_checkpoint(index)
        self.frame.begin(name, arguments)
        target = self.replay_target
        available_kinds = [
            k
            for k in self.kinds
            if self.task_caps is None or self.task_caps.available(k, self.task_key)
        ]
        active = {self.bound} if self.bound else set(available_kinds) & self.preferred
        if not active and available_kinds:
            active = {available_kinds[0]}
        if self.draw_on_selection and not self.wait_for_priority and not self.bound:
            active = set(available_kinds)
        if target and not self.bound:
            active = {target["kind"]} if index == target["call_index"] else set()
        balance_before = self.balance.report() if self.balance else None
        if self.balance is not None:
            active = (
                {self.bound}
                if self.bound
                else (set(self.live) if self.wait_for_priority else set(self.kinds))
            )
        if target and not self.bound:
            active = {target["kind"]} if index == target["call_index"] else set()
        qualifications, chosen, visible = [], None, None
        if self.observe_only:
            active = set()
        # An actual unexecuted failure must win BEFORE any backend dispatch.
        order = (
            self.balance.rank(self.kinds, self.task_key)
            if self.balance is not None and not self.wait_for_priority
            else list(self.kinds)
        )
        if (
            self.balance is None or self.wait_for_priority
        ) and "transient_unexecuted" in active:
            order.remove("transient_unexecuted")
            order.insert(0, "transient_unexecuted")
        if target and not self.bound and index == target["call_index"]:
            order.remove(target["kind"])
            order.insert(0, target["kind"])
        for kind in order:
            if self.bound and kind != self.bound:
                continue
            h = self.handlers[kind]
            original_already_executed = kind == "transient_unexecuted" and any(
                d["original"] for d in self.frame.dispatches
            )
            if self.balance is not None:
                if name not in h.targets:
                    continue
            old_events, old_bound = deepcopy(h.events), deepcopy(h.bound)
            result = await h.call_tool(name, arguments)
            entry = deepcopy(h.calls[-1])
            qualifies = bool(entry.get("injected"))
            qualifications.append(
                dict(
                    kind=kind,
                    eligible=qualifies,
                    reason=entry.get("eligibility_skip")
                    or entry.get("controller_skip"),
                    evidence=entry,
                )
            )
            if (
                qualifies
                and kind in active
                and not original_already_executed
                and (
                    self.task_caps is None or self.task_caps.claim(kind, self.task_key)
                )
            ):
                if self.bound is None and self.draw_on_selection:
                    from robustops.evaluation.fault_schedule import draw_repeat_limit

                    cap = draw_repeat_limit(kind, self.schedule, self.task_key)
                    self.budgets[kind] = cap
                    h.budget = cap
                chosen, visible = kind, result
                self.bound = kind
                if self.balance is not None:
                    self.balance.claim(kind, self.task_key)
                event = dict(
                    entry,
                    call_index=index,
                    selected_fault=kind,
                    injection_index=len(self.events) + 1,
                )
                self.events.append(event)
                self.checkpoints.clear()
                break
            # Keep actual reply history for reply_mixup, but no shadow dose/binding.
            h.events, h.bound = old_events, old_bound
            if qualifies:
                self.remember_opportunity(
                    dict(
                        kind=kind,
                        call_index=index,
                        tool=name,
                        arguments=deepcopy(arguments),
                    ),
                    checkpoint if kind in checkpoint_kinds else None,
                )
        if chosen is None:
            visible = await self.frame.call_tool(name, arguments)
        audit = dict(
            call_index=index,
            tool=name,
            arguments=deepcopy(arguments),
            injected=chosen is not None,
            selected_fault=chosen,
            visible_result=deepcopy(visible),
            qualifications=qualifications,
            backend_dispatch_count=len(self.frame.dispatches),
            backend_dispatches=deepcopy(self.frame.dispatches),
        )
        if self.balance is not None:
            audit.update(
                balance_before=balance_before,
                ranked_candidates=order,
                active_candidates=sorted(active),
            )
        original_key = (name, json.dumps(arguments, sort_keys=True))
        if original_key in self.frame.cache:
            audit["actual_result"] = deepcopy(self.frame.cache[original_key])
        self.calls.append(audit)
        if target and index == target["call_index"] and not chosen:
            raise InfrastructureError("replay target no longer strictly injectable")
        return visible

    def fallback_target(self):
        if self.events or self.replay_target or not self.opportunities:
            return None
        # Same mechanism priority and earliest qualifying point, independent of score.
        available = [
            p
            for p in self.opportunities
            if self.task_caps is None
            or self.task_caps.available(p["kind"], self.task_key)
        ]
        if not available:
            return None
        priority = (
            self.balance.rank(
                {p["kind"] for p in available}, self.task_key, self.preferred
            )
            if self.balance is not None and not self.wait_for_priority
            else self.priority
        )
        return deepcopy(
            min(available, key=lambda p: (priority.index(p["kind"]), p["call_index"]))
        )

    def report(self):
        hits = len(self.events)
        budget = self.budgets.get(self.bound, 0)
        return dict(
            enabled=not self.observe_only and bool(self.kinds),
            observe_only=self.observe_only,
            triggered=bool(hits),
            selected_mechanism=self.bound,
            injected_calls=hits,
            planned_injections=budget if self.bound else None,
            candidate_budgets=deepcopy(self.budgets),
            rejected_targets={
                k: deepcopy(getattr(h, "rejected_targets", {}))
                for k, h in self.handlers.items()
                if getattr(h, "rejected_targets", {})
            },
            qualification_read_calls=sum(
                not d["original"] for c in self.calls for d in c["backend_dispatches"]
            ),
            priority=list(self.priority),
            live_priority_candidates=sorted(self.live),
            reservation_mode="wait_for_priority_label"
            if self.wait_for_priority
            else "eligible_now",
            reservation_balance=deepcopy(self.reservation_balance),
            reserved_label=next(iter(self.live))
            if self.wait_for_priority and self.live
            else None,
            fallback_excluded_no_opportunity=[
                k
                for k in self.kinds
                if k not in {p["kind"] for p in self.opportunities}
            ]
            if not self.events
            else [],
            batch_task_caps=self.task_caps.report() if self.task_caps else None,
            point_policy="earliest_strictly_eligible",
            checkpoint_restored=self.restore_checkpoint is not None,
            inherited_call_count=self.call_offset,
            balance=self.balance.report() if self.balance is not None else None,
            opportunities=deepcopy(self.opportunities),
            replay_target=self.replay_target,
            events=deepcopy(self.events),
            call_audit=deepcopy(self.calls),
            rollout_stop_required=False,
        )


async def run_with_nohit_replay(
    *,
    controller_factory,
    model,
    attempt_kwargs,
    save_attempt,
    run_attempt=None,
    resume_episode=None,
):
    """One scored branch; retain exploration and all invalid replay evidence."""
    from robustops.evaluation.clean_baseline import run_task_attempt
    from robustops.evaluation.fault_checkpoint import (
        CheckpointBranchModel,
        prefix_messages,
    )

    run_attempt = run_attempt or run_task_attempt
    attempt_kwargs = dict(attempt_kwargs)
    attempt_kwargs.setdefault("initial_state", {})
    first_controller = controller_factory(None)
    if resume_episode is None:
        first = await run_attempt(
            model=model, tool_intervention=first_controller, **attempt_kwargs
        )
        target = first_controller.fallback_target()
    else:
        first = deepcopy(resume_episode["initial"])
        checkpoint = deepcopy(resume_episode["checkpoint"])
        if (
            not first["valid"]
            or not first["cleanup"]["succeeded"]
            or first["fault"]["injected_calls"]
        ):
            raise InfrastructureError(
                "only a completed zero-hit initial branch can resume at a checkpoint"
            )
        target = checkpoint["target"]
        first_controller.checkpoints[target["kind"]] = checkpoint
    save_attempt("initial", first)
    if attempt_kwargs["initial_state"]:
        save_attempt("initial_state", dict(attempt_kwargs["initial_state"]))
    if not first["valid"] or not first["cleanup"]["succeeded"] or target is None:
        return dict(
            final=first, scored_branch="initial", replay_attempted=False, target=None
        )
    second_controller = controller_factory(target)
    try:
        checkpoint = first_controller.checkpoints[target["kind"]]
        if checkpoint["call_index"] != target["call_index"]:
            raise InfrastructureError(
                "checkpoint does not match selected injection point"
            )
        trajectory = first["trajectory"]
        exposed = [
            e for e in trajectory["tool_events"] if not e.get("agent_protocol_error")
        ]
        event = exposed[target["call_index"] - 1]
        if (event["name"], event["arguments"]) != (target["tool"], target["arguments"]):
            raise InfrastructureError("checkpoint audit/trajectory boundary mismatch")
        prefix = dict(
            turns=[t for t in trajectory["turns"] if t["turn"] < event["turn"]],
            tool_events=[
                e for e in trajectory["tool_events"] if e["turn"] < event["turn"]
            ],
        )
        target_turn = next(t for t in trajectory["turns"] if t["turn"] == event["turn"])
        replay_model = CheckpointBranchModel(
            model, prefix_messages(attempt_kwargs["task"], prefix), target_turn
        )
        second_controller.restore_checkpoint = checkpoint
        attempt_kwargs["initial_state"] = deepcopy(checkpoint["databases"])
        attempt_kwargs["resume_trajectory"] = prefix
        save_attempt("checkpoint", dict(checkpoint, target=target, prefix=prefix))
    except (ValueError, KeyError, IndexError, InfrastructureError) as exc:
        invalid = dict(
            first,
            valid=False,
            success=False,
            status="invalid",
            error=dict(
                stage="checkpoint_setup", type=type(exc).__name__, message=str(exc)
            ),
        )
        save_attempt("replay_setup_invalid", invalid)
        return dict(
            final=invalid,
            scored_branch=None,
            replay_attempted=False,
            target=target,
            replay_setup_failed=True,
        )
    second = await run_attempt(
        model=replay_model, tool_intervention=second_controller, **attempt_kwargs
    )
    save_attempt("replay", second)
    # Never fall back to the better score when replay is invalid or unsuccessful.
    return dict(
        final=second,
        scored_branch="replay",
        replay_attempted=True,
        target=target,
        replay=replay_model.report(),
        initial_branch_reused=resume_episode is not None,
    )
