"""Local read-loop, failed native-stop and success-length credit."""

import json
import re


# Local read-loop coverage only; these are not terminal-state proof contracts.
# HR profile/report managers issue list/count queries; CSM installed-product
# managers issue SELECT/COUNT queries (fixed backend interfaces).
ADDITIONAL_READS = {
    "fetch_hr_profiles": ("hr.profile", "hr_profile_id"),
    "count_case_for_assignment_group": ("hr.case_count", "group_id"),
    "retrieve_installed_products": ("csm.installed_product", "installed_product_id"),
    "installed_product_warranties_ended_count": ("csm.warranty_count", "ended_count"),
}


def read_contracts(tools):
    from robustops.data.terminal_domain_state import DOMAIN_CONTRACTS
    from robustops.data.terminal_tool_catalog import resolve_contracts
    from robustops.data.terminal_state import ToolContract

    contracts = resolve_contracts(tools, DOMAIN_CONTRACTS)
    contracts.update(
        {
            name: ToolContract(resource, identity, "read")
            for name, (resource, identity) in ADDITIONAL_READS.items()
        }
    )
    result = {}
    for item in tools:
        schema = item.get("function", item)
        name = schema["name"]
        contract = contracts.get(name)
        # A name prefix alone never establishes read-only semantics. Dynamic
        # operations remain outside this first rule, even if a contract exists.
        description = name + " " + schema.get("description", "")
        if (
            contract
            and contract.effect == "read"
            and not re.search(
                r"\b(async\w*|poll\w*|pending|long.running|operation status|job status|wait\w*)\b",
                description.replace("_", " "),
                re.I,
            )
        ):
            result[name] = contract
    return result


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _incomplete_response(value):
    """Visible uncertainty, not an injector label; pagination is conservative too."""
    if isinstance(value, list):
        return any(_incomplete_response(v) for v in value)
    if not isinstance(value, dict):
        return False
    if any(
        value.get(k)
        for k in (
            "nextPageToken",
            "nextLink",
            "@odata.nextLink",
            "next_cursor",
            "has_more",
            "hasMore",
        )
    ):
        return True
    counts = (
        "resultSizeEstimate",
        "total_count",
        "totalCount",
        "count",
        "@odata.count",
        "total",
    )
    lists = [v for v in value.values() if isinstance(v, list)]
    if len(lists) == 1 and any(
        type(value.get(k)) in (int, float) and value[k] > len(lists[0]) for k in counts
    ):
        return True
    return any(_incomplete_response(v) for v in value.values())


def effective_response(event):
    result = event.get("result", {})
    if (
        result.get("is_error")
        or result.get("outcome", {}).get("application_ok") is False
    ):
        return None
    payload = result.get("result", {})
    if payload.get("isError"):
        return None
    content = []
    for part in payload.get("content", []):
        if part.get("type") != "text":
            return None
        text = part["text"]
        try:
            text = json.loads(text)
        except json.JSONDecodeError:
            # Truncated JSON is visible evidence to retry, even without an
            # injected-turn exemption or an outer error flag.
            if text.lstrip().startswith(("{", "[")):
                return None
        content.append(text)
    structured = payload.get("structuredContent")
    if not content and structured is None:
        return None
    value = dict(content=content, structured=structured)
    if _incomplete_response(value):
        return None
    # Do not punish waiting for an asynchronous transition, including endpoints
    # whose schema description does not explicitly advertise polling.
    if re.search(
        r"\b(pending|queued|in_progress|running|processing)\b", _canonical(value), re.I
    ):
        return None
    return _canonical(value)


def repeated_read_evidence(
    events, tools, *, inherited_turns=0, injected_turns=(), threshold=3
):
    """Allow two unchanged reads since the last new observation, including revisits.

    A/B queries no longer erase each other. A new query or changed response
    grants a new verification allowance for every query. Unknown/write/error,
    incomplete or injected responses clear all evidence, in clean and fault.
    """
    if threshold < 3:
        raise ValueError("read repetition threshold must allow two verification reads")
    contracts = read_contracts(tools)
    observations, counts, evidence = {}, {}, []
    for event in events:
        name = event["name"]
        response = effective_response(event)
        if (
            name not in contracts
            or event["turn"] in injected_turns
            or event.get("schema_violations")
            or event.get("agent_protocol_error")
            or response is None
        ):
            observations.clear()
            counts.clear()
            continue
        key = (event.get("gym_name"), name, _canonical(event["arguments"]))
        if observations.get(key) != response:
            counts.clear()
        observations[key] = response
        count = counts.get(key, 0) + 1
        counts[key] = count
        if count >= threshold and event["turn"] > inherited_turns:
            evidence.append(
                dict(
                    turn=event["turn"],
                    name=name,
                    arguments=event["arguments"],
                    rule="repeated_no_progress_read",
                    occurrence=count,
                    resource=contracts[name].resource,
                )
            )
    return evidence


def terminal_stop_audit(trajectory, *, success, inherited_turns):
    """Collect failed native stops for semantic review, never create a loss mask."""
    if success or trajectory["termination"] != "model_stop":
        return None
    turns = trajectory["turns"]
    if not turns or turns[-1]["turn"] <= inherited_turns:
        return None
    turn = turns[-1]
    if any(e["turn"] == turn["turn"] for e in trajectory["tool_events"]):
        return None
    return dict(
        rule="failed_native_stop_review",
        turn=turn["turn"],
        penalty_applied=False,
        requires_semantic_review=True,
        assistant_content=turn.get("assistant_content", ""),
        reported_usage=turn.get("reported_usage"),
    )


def exact_terminal_stop_span(bridge, audit):
    """Penalize every valid failed native stop, without judging its wording.

    Caller obtains audit only after trajectory validity/success checks. The
    audit excludes inherited turns, tool execution and budget terminations.
    Apply to the final generated turn, including its sampled ending tokens;
    controller-added separators and earlier actions remain outside the span.
    """
    if audit is None:
        return None
    usage = audit["reported_usage"]
    left = usage["prompt_tokens"] - len(bridge.prompt_ids)
    right = left + usage["completion_tokens"]
    if not 0 <= left < right <= len(bridge.mask) or not all(bridge.mask[left:right]):
        raise ValueError("terminal stop is outside exact newly generated tokens")
    return dict(start=left, end=right, turn=audit["turn"], rule="failed_native_stop")


QUALITY_BAD_RULES = frozenset(
    (
        "repeated_invalid_arguments",
        "repeated_unexposed_tool",
        "repeated_reference_rejection",
    )
)


def credit_segments(
    mask,
    spans,
    scalar,
    quality,
    eta,
    *,
    temporal=False,
    bad_count=None,
    terminal_stop_cap=None,
):
    """Exact piecewise token credit shared by selection, actor and accounting.

    Q retains its normalized sign and initial magnitude. Each counted bad call
    consumes one equal share at its exclusive end. No Q events means unchanged
    relative Q credit. Local repeat/stop labels never enter the Q count.
    """
    import math

    if not math.isfinite(eta) or eta < 0:
        raise ValueError("local_bad_eta must be finite and nonnegative")
    if terminal_stop_cap is not None and (
        not math.isfinite(terminal_stop_cap) or terminal_stop_cap > 0
    ):
        raise ValueError("terminal_stop_cap must be finite and nonpositive")
    if hasattr(mask, "ndim"):
        if mask.ndim == 2 and mask.shape[0] == 1:
            mask = mask[0]
        if mask.ndim != 1:
            raise ValueError("response mask must be one row")
        mask = mask.tolist()
    if not any(mask):
        raise ValueError(
            "empty generated-token mask is invalid, not a zero-signal replacement"
        )
    boundaries = {0, len(mask)}
    for span in spans:
        left, right = span["start"], span["end"]
        if not 0 <= left < right <= len(mask) or not all(mask[left:right]):
            raise ValueError("bad span outside generated tokens")
        boundaries.update((left, right))
    events = [s for s in spans if s.get("rule") in QUALITY_BAD_RULES]
    if temporal and (bad_count is None or len(events) != bad_count):
        raise ValueError("Q bad-action count differs from exact event spans")
    segments = []
    edges = sorted(boundaries)
    for left, right in zip(edges, edges[1:]):
        tokens = sum(mask[left:right])
        if not tokens:
            continue
        fraction = (
            sum(s["end"] > left for s in events) / len(events)
            if temporal and events
            else 1.0
        )
        q = quality * fraction
        before = scalar if fraction == 1 else scalar - quality + q
        bad = any(s["start"] <= left < s["end"] for s in spans)
        rules = sorted(
            {
                s.get("rule", "legacy_bad")
                for s in spans
                if s["start"] <= left < s["end"]
            }
        )
        after = before
        if any(rule != "failed_native_stop" for rule in rules) and eta:
            after = min(after, -eta)
        if "failed_native_stop" in rules:
            cap = (
                terminal_stop_cap
                if terminal_stop_cap is not None
                else -eta
                if eta
                else None
            )
            if cap is not None:
                after = min(after, cap)
        segments.append(
            dict(
                start=left,
                end=right,
                tokens=tokens,
                q_fraction=fraction,
                quality=q,
                bad=bad,
                rules=rules,
                before=before,
                after=after,
            )
        )
    return segments
