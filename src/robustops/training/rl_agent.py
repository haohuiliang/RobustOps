"""Clean G4, one forced fault anchor, and independent current-policy suffixes."""

import asyncio
from time import perf_counter
from copy import deepcopy
import json
from pathlib import Path

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput
from robustops.evaluation.clean_baseline import run_task_attempt, _tool_content
from robustops.evaluation.fault_checkpoint import CheckpointBranchModel, prefix_messages
from robustops.evaluation.selected_fault import SelectedFaultController
from robustops.training.rl_verifier import cached_task, InitialVerifier
from robustops.training.recovery_rl_bridge import TokenBridge
from robustops.training.recovery_rl_core import verifier_metrics, write_json
from robustops.training.recovery_rl_member_retry import (
    InvalidJointRollout,
    ModelProtocolRollout,
)
from robustops.evaluation.tool_protocol import model_protocol_retry_reason
from robustops.training.rl_action_feedback import bad_action_cost
from robustops.evaluation.fault_schedule import shared_strategy, STRATEGY_PATH
from robustops.training.rl_fault_runtime import JointRuntime, attach_prefix


class BoundaryVerifier(InitialVerifier):
    """Private verifier measurement at the actual generation boundary."""

    def __init__(self, task, controller):
        super().__init__(task, controller)
        self.boundary = None

    async def measure(self):
        if self.boundary is None:
            self.boundary = [
                await self.caller.environments[v["gym_name"]].run_verifier(v, i)
                for i, v in enumerate(self.task.verifiers, 1)
            ]
            if self.initial is None:
                self.initial = deepcopy(self.boundary)


def controller_for(task, row, budgets, tokenizer, token_state, **kwargs):
    payload = dict(
        task_id=task.task_id,
        task=dict(system_policy=task.system_prompt, user_request=task.user_prompt),
        verifiers=list(task.verifiers),
    )
    return JointRuntime(
        payload,
        row["selection"],
        budgets,
        SelectedFaultController,
        lambda x: len(tokenizer.encode(_tool_content(x), add_special_tokens=False)),
        token_state=token_state,
        **kwargs,
    )


def attempt_args(cfg, row, task, intervention, state, prefix=None):
    return dict(
        task=task,
        archive_path=Path(cfg.root) / "enterpriseops_gym/gym_dbs.zip",
        mcp_url_overrides=dict(cfg.mcp_urls),
        max_turns=cfg.max_turns,
        max_context_tokens=cfg.max_context_tokens,
        max_generation_tokens=cfg.max_generation_tokens,
        experiment_id="RECOVERY-RL-JOINT",
        run_id=Path(cfg.output).name,
        mode=row["mode"],
        model_name="RobustOps-RL-4B",
        model_endpoint="verl-token-api",
        plan_preserving=True,
        tool_intervention=intervention,
        initial_state=state,
        resume_trajectory=prefix,
    )


def suffix_ready(state, task, tokenizer, cfg):
    if state["tokens"]["turns"] >= cfg.max_turns:
        return False
    bridge = TokenBridge(tokenizer, None, {}, None, cfg.max_generation_tokens, 0)
    bridge.restore_prefix(state["tokens"]["ids"], state["tokens"]["turns"])
    context = bridge.context(prefix_messages(task, state["prefix"]), [])
    return context + cfg.max_generation_tokens <= cfg.max_context_tokens


def save_state(path, state, task, tokenizer, cfg, initial):
    state["initial_verifiers"] = deepcopy(initial)
    state["suffix_ready"] = suffix_ready(state, task, tokenizer, cfg)
    write_json(path, state)
    return str(path) if state["suffix_ready"] else None


async def prepare_anchor(cfg, tokenizer, row, plan, opportunity_path, output):
    """Execute the forced target once, then clone its actual post-call state."""
    source = json.loads(Path(opportunity_path).read_text())
    task = cached_task(cfg.root, row["domain"], row["mode"], row["task_id"])
    kind = plan["kind"]
    controller = controller_for(
        task,
        row,
        {kind: plan["budget"]},
        tokenizer,
        lambda: source["tokens"],
        replay_target=source["target"],
        anchor_only=True,
        limits=cfg,
    )
    controller.joint_restore = source
    intervention = InitialVerifier(task, controller)
    intervention.initial = source["initial_verifiers"]
    model = CheckpointBranchModel(
        None, prefix_messages(task, source["prefix"]), source["target_turn"]
    )

    class ForcedCounter:
        def context(self, messages, tools):
            return len(source["tokens"]["ids"])

        def text(self, content):
            return len(tokenizer.encode(content, add_special_tokens=False))

    args = attempt_args(
        cfg, row, task, intervention, source["databases"], source["prefix"]
    )
    args["max_generation_tokens"] = 0  # Forced action has no policy generation cost.
    path = Path(output)
    path.mkdir(parents=True, exist_ok=False)
    write_json(path / "plan.json", plan)
    async with asyncio.timeout(cfg.episode_timeout_seconds):
        result = await run_task_attempt(
            model=model, token_counter=ForcedCounter(), seed=plan["seed"], **args
        )
    write_json(path / "attempt.json", result)
    if not result["cleanup"]["succeeded"]:
        raise RuntimeError(
            f"unclean shared fault anchor: {path}; {result.get('error')}"
        )
    if not result["valid"]:
        raise InvalidJointRollout(
            f"invalid shared fault anchor: {path}; {result.get('error')}"
        )
    if controller.anchor_checkpoint is None:
        raise RuntimeError(
            f"invalid shared fault anchor: {path}; {result.get('error')}"
        )
    state = attach_prefix(controller.anchor_checkpoint, result["trajectory"])
    if "injection_point" in source:
        state["injection_point"] = source["injection_point"]
    saved = save_state(
        path / "state.json", state, task, tokenizer, cfg, intervention.initial
    )
    return dict(
        state_path=saved,
        actual_kind=kind,
        actual_dose=len(controller.events),
        injection_point=source.get("injection_point"),
        fallback_reason=None if saved else "anchor_no_suffix_budget",
        artifact=str(path),
    )


class RecoveryAgentLoop(AgentLoopBase):
    async def run(self, sampling_params, **kwargs):
        started = perf_counter()
        cfg = self.config.recovery
        row, plan = (
            dict(kwargs["extra_info"]["recovery"]),
            dict(kwargs["recovery_plan"]),
        )
        sample, step = int(kwargs["recovery_sample"]), int(kwargs["recovery_step"])
        flow, donor = kwargs["joint_flow"], bool(kwargs.get("joint_donor", False))
        task = cached_task(cfg.root, row["domain"], row["mode"], row["task_id"])
        graph = None
        if cfg.get("verifier_graph_dir"):
            from robustops.training.recovery_verifier_graph import task_graph

            graph = task_graph(cfg.verifier_graph_dir, task, row["mode"])
        state_path = kwargs.get("joint_state_path")
        state = json.loads(Path(state_path).read_text()) if state_path else None
        bridge = None
        controller = None
        if donor or state:
            budgets = state["budgets"] if state else plan["candidate_doses"]
            controller = controller_for(
                task,
                row,
                budgets,
                self.tokenizer,
                lambda: dict(ids=list(bridge.ids), turns=bridge.turns),
                limits=cfg,
                schedule=dict(
                    observe_only=state["observe_only"] if state else True,
                    collect_opportunities=donor and not state,
                ),
                point_seed=plan["seed"]
                if donor
                and not state
                and cfg.get("fault_point_policy") == "dose_window_uniform"
                else None,
            )
            controller.joint_restore = state
        intervention = BoundaryVerifier(task, controller)
        if state:
            intervention.initial = state["initial_verifiers"]
        member_retry = int(kwargs.get("joint_member_retry", 0))
        member_seed = plan["seed"] + sample * 1000003 + member_retry * 100000007
        bridge = TokenBridge(
            self.tokenizer,
            self.server_manager.generate,
            sampling_params,
            intervention,
            cfg.max_generation_tokens,
            member_seed,
        )
        if state:
            bridge.restore_prefix(state["tokens"]["ids"], state["tokens"]["turns"])
        inherited_turns = state["tokens"]["turns"] if state else 0
        initial_state = (
            state["databases"]
            if state
            else json.loads(
                (
                    Path(cfg.shared_plans)
                    / "initial_states"
                    / f"{row['task_id']}_{row['mode']}.json"
                ).read_text()
            )
        )
        path = (
            Path(cfg.output)
            / (
                "episodes/donors"
                if kwargs.get("preparation_only")
                else "episodes/train"
            )
            / f"step_{step}"
            / f"visit_{plan['visit']}_sample_{sample}"
        )
        if kwargs.get("joint_generation", 0):
            path = path.with_name(
                path.name + f"_replacement_{kwargs['joint_generation']}"
            )
        if member_retry:
            path = path.with_name(path.name + f"_retry_{member_retry}")
        path.mkdir(parents=True, exist_ok=False)
        write_json(
            path / "plan.json",
            dict(
                plan,
                flow=flow,
                source_state=state_path,
                parent_visit=row["parent_visit"],
                member_retry=member_retry,
                member_seed=member_seed,
            ),
        )
        result = None
        try:
            async with asyncio.timeout(cfg.episode_timeout_seconds):
                result = await run_task_attempt(
                    model=bridge,
                    token_counter=bridge,
                    seed=member_seed,
                    **attempt_args(
                        cfg,
                        row,
                        task,
                        intervention,
                        initial_state,
                        state["prefix"] if state else None,
                    ),
                )
            write_json(path / "attempt.json", result)
            if not result["cleanup"]["succeeded"]:
                raise RuntimeError(
                    f"unclean joint rollout: {path}; {result.get('error')}"
                )
            if not result["valid"]:
                raise InvalidJointRollout(
                    f"invalid joint rollout: {path}; {result.get('error')}"
                )
            if model_protocol_retry_reason(result):
                write_json(
                    path / "tokens.json",
                    dict(
                        prompt_ids=bridge.prompt_ids,
                        response_ids=bridge.ids[len(bridge.prompt_ids) :],
                        response_mask=bridge.mask,
                        response_logprobs=bridge.logprobs,
                    ),
                )
                raise ModelProtocolRollout(
                    f"model protocol rollout: {path}; {result['trajectory'].get('protocol_error')}"
                )
            trajectory = result["trajectory"]
            components = verifier_metrics(intervention.initial, result["verifiers"])
            if bool(components["strict_success"]) != bool(result["success"]):
                components.update(
                    strict_success=int(result["success"]),
                    reward=float(result["success"]),
                )
            cost_events = trajectory["tool_events"]
            exposed = [e for e in cost_events if not e.get("agent_protocol_error")]
            injected_turns = set()
            for audit in result["fault"].get("call_audit", []):
                if audit["injected"]:
                    event = exposed[audit["call_index"] - 1]
                    if (event["name"], event["arguments"]) != (
                        audit["tool"],
                        audit["arguments"],
                    ):
                        raise ValueError("fault audit/action identity mismatch")
                    injected_turns.add(event["turn"])
            cost_events = [
                dict(e, result=dict(is_error=False), schema_violations=[])
                if e["turn"] in injected_turns
                else e
                for e in cost_events
            ]
            cost = bad_action_cost(
                cost_events,
                inherited_turns=inherited_turns,
                verified_reference_repeats=cfg.get("verified_reference_repeats", False),
                tools=intervention.tools,
            )
            components.update(
                {k: cost[k] for k in ("bad_action_count", "bad_action_cost")}
            )
            components["bad_action_invalid_arguments"] = cost["rule_counts"].get(
                "repeated_invalid_arguments", 0
            )
            components["bad_action_unexposed_tool"] = cost["rule_counts"].get(
                "repeated_unexposed_tool", 0
            )
            components["bad_action_reference_rejection"] = cost["rule_counts"].get(
                "repeated_reference_rejection", 0
            )
            hits = result["fault"].get("injected_calls", 0)
            prefix_hits = len(state["runtime"]["events"]) if state else 0
            components.update(
                fault_enabled=int(bool(state and not state["observe_only"])),
                actual_hit=int(hits > 0),
                actual_dose=hits,
                inherited_dose=prefix_hits,
                new_dose=hits - prefix_hits,
                planned_dose=plan.get("budget", 0),
                generated_tokens=sum(bridge.mask),
                context_tokens=len(bridge.ids),
                tool_calls=sum(
                    e["turn"] > inherited_turns for e in trajectory["tool_events"]
                ),
                inherited_turns=inherited_turns,
                budget_terminated=int(
                    trajectory["termination"]
                    in ("max_tokens", "max_turns", "context_limit_pre_request")
                ),
            )
            from robustops.training.rl_training_metrics import trajectory_metrics

            components.update(
                trajectory_metrics(trajectory, inherited_turns, injected_turns)
            )
            if state and state.get("injection_point") and flow == "fault":
                point = state["injection_point"]
                components.update(
                    fault_first_call_index=point["call_index"],
                    fault_first_call_fraction=point["call_fraction"],
                    fault_eligible_rank=point["eligible_rank"],
                    fault_eligible_count=point["eligible_count"],
                    fault_eligible_window_count=point["eligible_window_count"],
                    fault_donor_later_opportunities=point["donor_later_opportunities"],
                )
            bad_spans = []
            repeat_evidence, stop_audit = [], None
            from robustops.training.rl_credit import (
                components as reward_components,
                exact_bad_spans,
            )

            components.update(
                reward_components(
                    intervention.boundary,
                    result["verifiers"],
                    components["strict_success"],
                    cost["bad_action_count"],
                    components["tool_calls"],
                    dependencies=graph["dependencies"] if graph is not None else None,
                )
            )
            # L requires completed group peers; the manager finalizes score before packing.
            components["reward"] = sum(
                w * components[k]
                for w, k in zip(
                    cfg.reward_weights,
                    ("reward_success", "reward_progress", "reward_quality"),
                )
            )
            bad_spans = exact_bad_spans(bridge, trajectory, cost["evidence"])
            if cfg.get("local_read_repeat_from", 0):
                from robustops.training.recovery_local_credit import (
                    repeated_read_evidence,
                )

                repeat_evidence = repeated_read_evidence(
                    trajectory["tool_events"],
                    intervention.tools,
                    inherited_turns=inherited_turns,
                    injected_turns=injected_turns,
                    threshold=cfg.local_read_repeat_from,
                )
                repeat_spans = exact_bad_spans(bridge, trajectory, repeat_evidence)
                bad_spans.extend(repeat_spans)
                components.update(
                    local_read_repeat_count=len(repeat_evidence),
                    local_read_repeat_tokens=sum(
                        s["end"] - s["start"] for s in repeat_spans
                    ),
                )
            if cfg.get("terminal_stop_audit", False) or cfg.get(
                "terminal_stop_penalty", False
            ):
                from robustops.training.recovery_local_credit import terminal_stop_audit

                stop_audit = terminal_stop_audit(
                    trajectory,
                    success=components["strict_success"],
                    inherited_turns=inherited_turns,
                )
                components["terminal_stop_review_candidate"] = int(
                    stop_audit is not None
                )
                if cfg.get("terminal_stop_penalty", False):
                    from robustops.training.recovery_local_credit import (
                        exact_terminal_stop_span,
                    )

                    stop_span = exact_terminal_stop_span(bridge, stop_audit)
                    components["terminal_stop_penalized"] = int(stop_span is not None)
                    if stop_span:
                        bad_spans.append(stop_span)
                        stop_audit.update(
                            penalty_applied=True,
                            rule=stop_span["rule"],
                            requires_semantic_review=False,
                            classification="all_valid_failed_native_stops",
                        )
            if not any(bridge.mask) or len(bridge.ids) > cfg.max_context_tokens:
                raise ValueError(
                    f"joint trajectory has no new tokens or exceeds context: {path}"
                )
            artifacts = dict(
                opportunities={},
                observed_opportunity_kinds=[],
                opportunity_exclusions={},
                error=None,
                attempt=str(path / "attempt.json"),
                task_id=row["task_id"],
                mode=row["mode"],
                parent_visit=row["parent_visit"],
            )
            if donor:
                strategy = shared_strategy(
                    cfg.get("fault_strategy", STRATEGY_PATH), root=cfg.root
                )
                artifacts["observed_opportunity_kinds"] = sorted(controller.checkpoints)
                # Save possible windows; draw the cap only after selecting a kind.
                for kind in controller.checkpoints:
                    paths = {}
                    for cap, snapshot in controller.point_options(
                        kind, map(int, strategy["repeat_probabilities"][kind])
                    ).items():
                        turn = snapshot["tokens"]["turns"]
                        if turn >= cfg.max_turns:
                            artifacts["opportunity_exclusions"][kind] = (
                                "no_remaining_turn"
                            )
                            continue
                        opportunity = next(
                            p
                            for p in controller.opportunities
                            if p["kind"] == kind
                            and p["call_index"] == snapshot["call_index"]
                        )
                        saved = attach_prefix(snapshot, trajectory, before=True)
                        saved["budgets"][kind] = cap
                        saved.update(
                            target=opportunity,
                            target_turn=next(
                                t for t in trajectory["turns"] if t["turn"] == turn
                            ),
                            initial_verifiers=intervention.initial,
                            injection_point=controller.point_metadata(
                                kind,
                                len(controller.calls),
                                budget=cap,
                                snapshot=snapshot,
                            ),
                        )
                        target = path / f"opportunity_{kind}_cap{cap}.json"
                        write_json(target, saved)
                        paths[str(cap)] = str(target)
                    if paths:
                        artifacts["opportunities"][kind] = paths
            artifacts["bad_spans"] = bad_spans
            artifacts["local_read_repeat_evidence"] = repeat_evidence
            artifacts["terminal_stop_audit"] = stop_audit
            graph_source = (
                dict(
                    directory=cfg.verifier_graph_dir,
                    graph_id=graph["row"]["graph_id"],
                    task_id=task.task_id,
                    mode=row["mode"],
                )
                if graph is not None
                else None
            )
            if graph_source:
                artifacts["verifier_graph"] = graph_source
            components.update(
                episode_seconds=perf_counter() - started,
                generation_request_seconds=bridge.generated_seconds,
            )
            write_json(
                path / "reward.json",
                dict(
                    components,
                    initial_verifiers=intervention.initial,
                    boundary_verifiers=intervention.boundary,
                    bad_action_evidence=cost["evidence"],
                    local_read_repeat_evidence=repeat_evidence,
                    terminal_stop_audit=stop_audit,
                    verifier_graph=graph_source,
                ),
            )
            write_json(
                path / "tokens.json",
                dict(
                    prompt_ids=bridge.prompt_ids,
                    response_ids=bridge.ids[len(bridge.prompt_ids) :],
                    response_mask=bridge.mask,
                    response_logprobs=bridge.logprobs,
                ),
            )
            write_json(path / "artifacts.json", artifacts)
            return AgentLoopOutput(
                prompt_ids=bridge.prompt_ids,
                response_ids=bridge.ids[len(bridge.prompt_ids) :],
                response_mask=bridge.mask,
                response_logprobs=bridge.logprobs,
                reward_score=components["reward"],
                num_turns=2 * bridge.turns,
                metrics=dict(generate_sequences=bridge.generated_seconds),
                extra_fields=dict(
                    reward_extra_info={
                        k: v for k, v in components.items() if k != "reward"
                    },
                    recovery_metrics=components,
                    recovery_bad_spans=bad_spans,
                    recovery_visit=plan["visit"],
                    recovery_kind=plan["kind"],
                    recovery_flow=flow,
                    recovery_task_id=row["task_id"],
                    recovery_mode=row["mode"],
                    recovery_parent_visit=row["parent_visit"],
                    recovery_source_state=state_path,
                    recovery_joint_slot=int(kwargs["joint_slot"]),
                    recovery_sample=sample,
                    recovery_planned_fault_parent=int(
                        row["joint_parent_slot"] in row["joint_fault_parents"]
                    ),
                    recovery_artifacts=artifacts,
                    recovery_episode=str(path),
                ),
            )
        except BaseException as exc:
            write_json(
                path / "error.json", dict(type=type(exc).__name__, message=str(exc))
            )
            raise
        finally:
            from robustops.training.rl_training_metrics import sampling_cost

            write_json(
                path / "sampling_cost.json",
                sampling_cost(
                    result,
                    bridge,
                    row,
                    inherited_turns=inherited_turns,
                    flow=flow,
                    donor=donor,
                    member_retry=member_retry,
                    elapsed=perf_counter() - started,
                ),
            )
