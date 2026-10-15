"""Token-exact selective supervision for Qwen tool trajectories."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence


LOSS_SCOPE = "selective_assistant_spans_per_trajectory_normalized"
DECISION_BALANCED_LOSS_SCOPE = "selective_assistant_events_source_balanced"
TARGET_FULL = "full"
TARGET_TOOL_CALL_ONLY = "tool_call_only"
TARGET_MASKED = "masked"
TARGETS = frozenset({TARGET_FULL, TARGET_TOOL_CALL_ONLY, TARGET_MASKED})


class SelectiveSupervisionError(ValueError):
    """A row cannot satisfy the selective token-loss contract."""


def validate_supervision_policy(
    messages: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
) -> dict[int, str]:
    """Require exactly one explicit target policy for every assistant turn."""
    if not messages or not all(isinstance(message, Mapping) for message in messages):
        raise SelectiveSupervisionError("messages must be a non-empty list of objects")
    if not isinstance(policies, list) or not policies:
        raise SelectiveSupervisionError("assistant supervision must be a non-empty list")

    by_index: dict[int, str] = {}
    for policy in policies:
        if not isinstance(policy, Mapping):
            raise SelectiveSupervisionError("assistant supervision entries must be objects")
        index = policy.get("message_index")
        target = policy.get("target")
        if not isinstance(index, int) or isinstance(index, bool) or index in by_index:
            raise SelectiveSupervisionError(
                "assistant supervision indices must be unique integers"
            )
        if target not in TARGETS:
            raise SelectiveSupervisionError(
                f"assistant supervision target must be one of {sorted(TARGETS)}"
            )
        if index < 0 or index >= len(messages) or messages[index].get("role") != "assistant":
            raise SelectiveSupervisionError(
                f"assistant supervision index {index} is not an assistant message"
            )
        if target == TARGET_TOOL_CALL_ONLY and not messages[index].get("tool_calls"):
            raise SelectiveSupervisionError(
                f"tool_call_only target {index} does not contain a tool call"
            )
        by_index[index] = str(target)

    assistant_indices = [
        index for index, message in enumerate(messages) if message.get("role") == "assistant"
    ]
    if sorted(by_index) != assistant_indices:
        raise SelectiveSupervisionError(
            "every and only assistant message must have one supervision policy"
        )
    if messages[-1].get("role") == "tool":
        # A completed action/receipt suffix needs no synthetic assistant terminal.
        last_assistant = assistant_indices[-1]
        calls = messages[last_assistant].get("tool_calls") or []
        receipts = messages[last_assistant + 1:]
        call_ids = [call.get("id") for call in calls]
        if (not calls or any(not i for i in call_ids)
                or len(set(call_ids)) != len(call_ids)
                or any(m.get("role") != "tool" for m in receipts)
                or [m.get("tool_call_id") for m in receipts] != call_ids):
            raise SelectiveSupervisionError("tool-ending row requires complete matching receipts")
    else:
        if messages[-1].get("role") != "assistant" or messages[-1].get("tool_calls"):
            raise SelectiveSupervisionError("final message must be a terminal assistant turn")
        if not str(messages[-1].get("content") or "").strip():
            raise SelectiveSupervisionError("terminal assistant content must be non-empty")
    if all(target == TARGET_MASKED for target in by_index.values()):
        raise SelectiveSupervisionError("each row must retain at least one supervised turn")
    return by_index


def _as_token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        if len(value) != 1:
            raise SelectiveSupervisionError("chat template returned more than one sequence")
        value = value[0]
    if not isinstance(value, list) or any(
        not isinstance(token, int) or isinstance(token, bool) for token in value
    ):
        raise SelectiveSupervisionError("chat template did not return a token-id sequence")
    return value


def _render_ids(
    renderer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    *,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any],
) -> list[int]:
    value = renderer.apply_chat_template(
        list(messages),
        tools=list(tools),
        add_generation_prompt=add_generation_prompt,
        tokenize=True,
        return_dict=True,
        **dict(template_kwargs),
    )
    return _as_token_ids(value)


def _build_selective_token_spans(
    *,
    renderer: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[tuple[int, int, int]]]:
    """Render one conversation and locate every positive assistant event.

    ``tool_call_only`` starts at Qwen's actual ``<tool_call>`` special token
    and ends after the assistant ``<|im_end|>``.  Planning remains in
    ``input_ids`` as causal context but contributes no target loss.
    """
    by_index = validate_supervision_policy(messages, policies)
    if not isinstance(tools, Sequence) or isinstance(tools, (str, bytes)) or not tools:
        raise SelectiveSupervisionError("tools must be a non-empty list")
    kwargs = dict(template_kwargs or {})
    full_ids = _render_ids(
        renderer,
        messages,
        tools,
        add_generation_prompt=False,
        template_kwargs=kwargs,
    )
    spans: list[tuple[int, int, int]] = []

    open_id = tokenizer.convert_tokens_to_ids("<tool_call>")
    close_id = tokenizer.convert_tokens_to_ids("</tool_call>")
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if (
        not isinstance(open_id, int)
        or not isinstance(close_id, int)
        or open_id == unk_id
        or close_id == unk_id
        or open_id == close_id
    ):
        raise SelectiveSupervisionError(
            "tokenizer must expose distinct <tool_call> and </tool_call> special tokens"
        )

    previous_end = -1
    for assistant_index in sorted(by_index):
        generation_prefix = _render_ids(
            renderer,
            messages[:assistant_index],
            tools,
            add_generation_prompt=True,
            template_kwargs=kwargs,
        )
        assistant_prefix = _render_ids(
            renderer,
            messages[: assistant_index + 1],
            tools,
            add_generation_prompt=False,
            template_kwargs=kwargs,
        )
        start, end = len(generation_prefix), len(assistant_prefix)
        if start >= end or start < previous_end:
            raise SelectiveSupervisionError(
                f"assistant target {assistant_index} has an unstable token boundary"
            )
        if full_ids[:start] != generation_prefix or full_ids[:end] != assistant_prefix:
            raise SelectiveSupervisionError(
                f"assistant target {assistant_index} is not a stable chat-template prefix"
            )

        target = by_index[assistant_index]
        target_start = start
        if target == TARGET_TOOL_CALL_ONLY:
            opens = [position for position in range(start, end) if full_ids[position] == open_id]
            closes = [position for position in range(start, end) if full_ids[position] == close_id]
            if len(opens) != 1 or len(closes) != 1 or opens[0] >= closes[0]:
                raise SelectiveSupervisionError(
                    f"assistant target {assistant_index} does not contain exactly one "
                    "well-ordered structured tool call"
                )
            target_start = opens[0]
        if target != TARGET_MASKED:
            if spans and target_start < spans[-1][2]:
                raise SelectiveSupervisionError("assistant supervision spans overlap")
            spans.append((assistant_index, target_start, end))
        previous_end = end

    if not spans:
        raise SelectiveSupervisionError("selective token mask contains no target tokens")
    return full_ids, spans


def build_selective_token_mask(
    *,
    renderer: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Render one full conversation and return its exact binary loss mask."""
    full_ids, spans = _build_selective_token_spans(
        renderer=renderer,
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        policies=policies,
        template_kwargs=template_kwargs,
    )
    loss_mask = [0] * len(full_ids)
    for _, start, end in spans:
        loss_mask[start:end] = [1] * (end - start)
    return full_ids, loss_mask


def build_selective_event_token_weights(
    *,
    renderer: Any,
    tokenizer: Any,
    messages: Sequence[Mapping[str, Any]],
    tools: Sequence[Mapping[str, Any]],
    policies: Sequence[Mapping[str, Any]],
    trajectory_mass: float,
    template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[float]]:
    """Give each positive assistant decision equal mass within a trajectory.

    The resulting token weights sum to ``trajectory_mass``.  Each positive
    assistant event first receives an equal share of that mass, then divides
    its share uniformly over its supervised tokens.  Masked failed actions and
    context-only planning therefore contribute exactly zero target loss.
    """
    if not isinstance(trajectory_mass, (int, float)) or isinstance(trajectory_mass, bool):
        raise SelectiveSupervisionError("trajectory_mass must be a finite positive number")
    trajectory_mass = float(trajectory_mass)
    if not math.isfinite(trajectory_mass) or trajectory_mass <= 0.0:
        raise SelectiveSupervisionError("trajectory_mass must be a finite positive number")

    full_ids, spans = _build_selective_token_spans(
        renderer=renderer,
        tokenizer=tokenizer,
        messages=messages,
        tools=tools,
        policies=policies,
        template_kwargs=template_kwargs,
    )
    token_weights = [0.0] * len(full_ids)
    event_mass = trajectory_mass / len(spans)
    for _, start, end in spans:
        token_mass = event_mass / (end - start)
        token_weights[start:end] = [token_mass] * (end - start)
    return full_ids, token_weights
