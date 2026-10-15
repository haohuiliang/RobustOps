"""One Hermes action boundary for local evaluation and exact-token training."""

import json
from collections.abc import Mapping


TOOL_CALL_END = '</tool_call>'


def model_protocol_retry_reason(attempt: Mapping) -> str | None:
    """A failed model action is retryable without relabeling it as server invalid."""
    trajectory = attempt.get('trajectory')
    if isinstance(trajectory, Mapping) and trajectory.get('termination') == 'model_protocol_error':
        return 'model_protocol_error'
    return None


def parse_tool_response(text: str, *, finish_reason: str, call_id: str) -> dict:
    """Read the first action without repairing or executing generated text.

    Generation stops at the first closing tag in both callers. A truncated
    unfinished call remains a budget termination; a completed bad action is a
    model protocol error. Keep the original text for trajectory diagnostics.
    """
    message = dict(role='assistant', content=text, tool_calls=[], raw_model_content=text)
    body = text.removesuffix('<|im_end|>')
    start = body.find('<tool_call>')
    if start < 0:
        if '<tool_call' not in body and TOOL_CALL_END not in body:
            message['content'] = body
            return message
        error = 'incomplete or unmatched tool-call tag'
        if finish_reason == 'length':
            return message
    else:
        end = body.find(TOOL_CALL_END, start)
        if end < 0:
            if finish_reason == 'length':
                return message
            error = 'tool call is missing its closing tag'
        else:
            raw = body[start + len('<tool_call>'):end]
            try:
                call = json.loads(raw)
                if (not isinstance(call, dict) or not isinstance(call.get('name'), str)
                        or not call['name'] or not isinstance(call.get('arguments'), dict)):
                    raise ValueError('tool call requires a name and object arguments')
                message['content'] = body[:start]
                message['tool_calls'] = [dict(id=call_id, type='function', function=dict(
                    name=call['name'], arguments=json.dumps(call['arguments'], ensure_ascii=False)))]
                return message
            except (ValueError, TypeError) as exc:
                error = f'invalid tool call: {exc}'
    message['model_protocol_error'] = dict(error=error, mcp_called=False)
    return message
