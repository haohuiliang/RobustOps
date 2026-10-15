"""Reconstruct student-visible messages from executed turns and tool receipts."""

from __future__ import annotations
from collections.abc import Mapping
from copy import deepcopy
from typing import Any
from robustops.envs.enterpriseops.task import EnterpriseOpsTask
from robustops.evaluation.clean_baseline import _tool_content


class TrajectoryMessageError(ValueError):
    """Executed turns and tool events do not form a consistent conversation."""


def build_messages(
    task: EnterpriseOpsTask,
    trajectory: Mapping[str, Any],
    *,
    allow_tool_ending: bool = False,
) -> tuple[list[dict[str, Any]], list[int]]:
    turns = trajectory["turns"]
    events = trajectory["tool_events"]
    events_by_id: dict[str, Mapping[str, Any]] = {}
    for event in events:
        call_id = event.get("tool_call_id")
        if not isinstance(call_id, str) or not call_id or call_id in events_by_id:
            raise TrajectoryMessageError(
                f"{task.task_id}: invalid or duplicate tool_call_id"
            )
        events_by_id[call_id] = event

    messages: list[dict[str, Any]] = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": task.user_prompt},
    ]
    assistant_indices: list[int] = []
    consumed: set[str] = set()
    for turn in turns:
        calls = turn.get("tool_calls") or []
        if not isinstance(calls, list):
            raise TrajectoryMessageError(f"{task.task_id}: tool_calls is not a list")
        canonical_calls: list[dict[str, Any]] = []
        ordered_events: list[Mapping[str, Any]] = []
        for call in calls:
            function = call.get("function") if isinstance(call, Mapping) else None
            call_id = call.get("id") if isinstance(call, Mapping) else None
            event = events_by_id.get(call_id)
            if (
                not isinstance(call_id, str)
                or not isinstance(function, Mapping)
                or event is None
                or function.get("name") != event.get("name")
            ):
                raise TrajectoryMessageError(
                    f"{task.task_id}: turn/event identity mismatch"
                )
            canonical_calls.append(
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": event["name"],
                        "arguments": deepcopy(event["arguments"]),
                    },
                }
            )
            ordered_events.append(event)
            consumed.add(call_id)
        assistant: dict[str, Any] = {
            "role": "assistant",
            "content": str(turn.get("assistant_content") or ""),
        }
        if canonical_calls:
            assistant["tool_calls"] = canonical_calls
        assistant_indices.append(len(messages))
        messages.append(assistant)
        for event in ordered_events:
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": event["tool_call_id"],
                    "name": event["name"],
                    "content": _tool_content(event["result"]),
                }
            )
    if consumed != set(events_by_id):
        raise TrajectoryMessageError(
            f"{task.task_id}: trajectory contains unmatched tool events"
        )
    if not allow_tool_ending and (
        messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls")
    ):
        raise TrajectoryMessageError(
            f"{task.task_id}: final message is not a terminal assistant turn"
        )
    return messages, assistant_indices
