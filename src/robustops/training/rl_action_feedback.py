"""Evidence-based bad-action labels and flow-balanced loss weights."""

from collections import Counter
import json
import re


# Only argument-validation categories, not generic value/business/auth errors.
PARAMETER_ERROR_TYPES = frozenset(
    {
        "missing",
        "string_type",
        "int_type",
        "int_parsing",
        "float_type",
        "float_parsing",
        "bool_type",
        "bool_parsing",
        "list_type",
        "dict_type",
        "literal_error",
        "enum",
        "greater_than",
        "greater_than_equal",
        "less_than",
        "less_than_equal",
        "string_too_short",
        "string_too_long",
        "string_pattern_mismatch",
        "date_parsing",
        "datetime_parsing",
        "uuid_parsing",
    }
)


def explicit_parameter_feedback(event):
    """Read model-visible error fields; never infer from HTTP status or hidden outcome."""
    result = event.get("result", {})
    if not result.get("is_error"):
        return None

    def inspect(value):
        if isinstance(value, dict):
            location = value.get("loc", [])
            if (
                isinstance(location, list)
                and location
                and location[0] in ("body", "query", "path")
                and value.get("type") in PARAMETER_ERROR_TYPES
            ):
                # A reported missing field that was supplied is a contract conflict.
                if value["type"] == "missing":
                    node = event["arguments"]
                    try:
                        for key in location[1:]:
                            node = node[key]
                    except (KeyError, IndexError, TypeError):
                        pass
                    else:
                        return None
                return dict(
                    type=value["type"], location=location, message=value.get("msg")
                )
            for key in (
                "error",
                "detail",
                "details",
                "message",
                "msg",
                "content",
                "text",
                "structuredContent",
                "result",
            ):
                if key in value:
                    found = inspect(value[key])
                    if found:
                        return found
        elif isinstance(value, list):
            for item in value:
                found = inspect(item)
                if found:
                    return found
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                return inspect(parsed)
            if re.search(r"invalid tool arguments\s*:", value, re.I):
                return dict(message=value)
            fields = set(event.get("schema_property_arguments", [])) | set(
                event["arguments"]
            )
            mentions_field = any(
                re.search(r"(?<!\w)" + re.escape(field) + r"(?!\w)", value, re.I)
                for field in fields
            )
            format_error = re.search(
                r"\binvalid\b[^\n:]{0,60}\bformat\b|\bmust be (?:a |an )?(?:string|integer|boolean|number|valid (?:uuid|date|datetime))\b|\bvalid (?:values|roles|options)\s*:",
                value,
                re.I,
            )
            if mentions_field and format_error:
                return dict(message=value)
        return None

    # outcome is deliberately absent: the model never sees it.
    return inspect({key: value for key, value in result.items() if key != "outcome"})


def explicit_reference_feedback(event):
    """Only visible field-specific missing references, never a generic 404/5xx."""
    result = event.get("result", {})
    if not result.get("is_error") or event.get("agent_protocol_error"):
        return None
    args = event["arguments"]

    def inspect(value):
        if isinstance(value, dict):
            field, message = None, str(value.get("msg", value.get("message", "")))
            loc = value.get("loc")
            if (
                value.get("type") == "value_error.not_found"
                and isinstance(loc, list)
                and len(loc) == 2
                and loc[0] == "body"
            ):
                field = loc[1]
            elif (
                value.get("code") in ("VALIDATION_ERROR", "RESOURCE_NOT_FOUND")
                and isinstance(value.get("field"), str)
                and re.search(r"\bnot found\b|\bdoes not exist\b", message, re.I)
            ):
                field = value["field"]
            if isinstance(field, str) and field in args:
                return dict(field=field, value=args[field], message=message)
            for key in (
                "error",
                "detail",
                "details",
                "message",
                "msg",
                "content",
                "text",
                "structuredContent",
                "result",
            ):
                if key in value:
                    found = inspect(value[key])
                    if found:
                        return found
        elif isinstance(value, list):
            for item in value:
                found = inspect(item)
                if found:
                    return found
        elif isinstance(value, str):
            try:
                parsed = json.loads(value)
            except (ValueError, TypeError):
                parsed = None
            if isinstance(parsed, (dict, list)):
                return inspect(parsed)
        return None

    return inspect({k: v for k, v in result.items() if k != "outcome"})


def bad_action_cost(
    events,
    *,
    inherited_turns=0,
    saturation=4,
    verified_reference_repeats=False,
    tools=(),
):
    """Charge verified repeated argument failures or reusing a known unexposed tool.

    Remember each exact argument failure across alternate failed arguments and
    successful calls. Unknown/error/hidden feedback breaks parameter evidence;
    a successful retry clears that query. Unexposed tools remain unavailable.
    Inherited history supplies evidence but incurs no new charge itself.
    """
    rejected, unavailable, evidence = {}, {}, []
    references, read_observations = {}, {}
    if verified_reference_repeats:
        from robustops.training.recovery_local_credit import (
            read_contracts,
            effective_response,
        )

        reads = read_contracts(tools)
    for event in events:
        name = event["name"]
        result = event.get("result", {})
        expected = f"tool {name!r} is not exposed for this task"
        unexposed = (
            result.get("is_error")
            and event.get("agent_protocol_error") == expected
            and result.get("error") == expected
        )
        feedback = (
            dict(schema_violations=event["schema_violations"])
            if result.get("is_error") and event.get("schema_violations")
            else explicit_parameter_feedback(event)
        )
        key = (
            event.get("gym_name"),
            name,
            json.dumps(event["arguments"], sort_keys=True),
        )
        feedback_key = json.dumps(feedback, sort_keys=True) if feedback else None
        reference = (
            explicit_reference_feedback(event) if verified_reference_repeats else None
        )
        reference_key = json.dumps(reference, sort_keys=True) if reference else None
        rule, prior = None, None
        if unexposed and name in unavailable:
            rule, prior = "repeated_unexposed_tool", unavailable[name]
        elif (
            feedback
            and key in rejected
            and rejected[key]["feedback_key"] == feedback_key
        ):
            rule = "repeated_invalid_arguments"
            prior = rejected[key]["evidence"]
        elif (
            reference
            and key in references
            and references[key]["feedback_key"] == reference_key
        ):
            rule, prior = "repeated_reference_rejection", references[key]["evidence"]
        if rule and event["turn"] > inherited_turns:
            evidence.append(
                dict(
                    turn=event["turn"],
                    name=name,
                    arguments=event["arguments"],
                    rule=rule,
                    previous=prior,
                    feedback=dict(error=expected)
                    if unexposed
                    else reference
                    if rule == "repeated_reference_rejection"
                    else feedback,
                )
            )
        if verified_reference_repeats:
            if reference:
                references[key] = dict(
                    feedback_key=reference_key,
                    evidence=dict(
                        turn=event["turn"],
                        name=name,
                        arguments=event["arguments"],
                        feedback=reference,
                    ),
                )
            else:
                response = effective_response(event) if name in reads else None
                # A new observation grants another try. Writes, faults, unknown
                # results and incomplete reads invalidate all previous proof.
                if response is None or read_observations.get(key) != response:
                    references.clear()
                if response is None:
                    read_observations.clear()
                else:
                    read_observations[key] = response
        current = dict(
            turn=event["turn"],
            name=name,
            arguments=event["arguments"],
            feedback=dict(error=expected) if unexposed else feedback,
        )
        if unexposed:
            unavailable[name] = current
        if feedback:
            rejected[key] = dict(feedback_key=feedback_key, evidence=current)
        elif (
            result.get("is_error")
            or result.get("outcome", {}).get("application_ok") is not True
        ):
            # Includes the deliberately blanked feedback of injected calls.
            rejected.clear()
        else:
            rejected.pop(key, None)
    count = len(evidence)
    return dict(
        bad_action_count=count,
        bad_action_cost=count / (count + saturation),
        rule_counts=dict(Counter(e["rule"] for e in evidence)),
        evidence=evidence,
    )


def group_signal_diagnostics(metrics, flows, slots):
    """Keep zero-reward-range groups; distinguish solved and failed zero signal."""
    groups = {}
    for metric, flow, slot in zip(metrics, flows, slots, strict=True):
        groups.setdefault((flow, slot), []).append(metric)
    report = {
        flow: dict(
            groups=0,
            zero_signal_groups=0,
            zero_all_success=0,
            zero_all_failed=0,
            zero_with_bad_actions=0,
        )
        for flow in ("clean", "fault")
    }
    for (flow, _), group in groups.items():
        if len(group) != 4:
            raise ValueError("signal diagnostics require complete G4")
        row = report[flow]
        row["groups"] += 1
        if len({m["reward"] for m in group}) == 1:
            row["zero_signal_groups"] += 1
            row["zero_all_success"] += int(all(m["strict_success"] for m in group))
            row["zero_all_failed"] += int(not any(m["strict_success"] for m in group))
            row["zero_with_bad_actions"] += int(
                any(m.get("bad_action_count", 0) for m in group)
            )
    return report


def flow_multipliers(flows, token_counts):
    """w_s * global token count / flow token count, for one global token-mean loss."""
    rows, tokens = Counter(flows), Counter()
    for flow, count in zip(flows, token_counts, strict=True):
        tokens[flow] += count
    total = sum(token_counts)
    if not total or set(rows) - {"clean", "fault", "correction"}:
        raise ValueError("invalid joint flow/token allocation")
    if any(tokens[s] <= 0 for s in rows):
        raise ValueError("joint flow has no generated tokens")
    weights = {s: n / len(flows) for s, n in rows.items()}
    return [weights[s] * total / tokens[s] for s in flows], weights
