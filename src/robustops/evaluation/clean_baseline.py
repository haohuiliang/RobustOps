"""No-fault MidTool evaluation on configured EnterpriseOps recipes."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Mapping, Protocol

import httpx

from robustops.evaluation.tool_protocol import TOOL_CALL_END, parse_tool_response, model_protocol_retry_reason

from robustops.data.enterpriseops_split import (
    SplitContractError,
    load_partition_task_ids,
)
from robustops.envs.enterpriseops.runtime import (
    EnterpriseOpsEnvironment,
    InfrastructureError,
)
from robustops.envs.enterpriseops.task import (
    EnterpriseOpsTask,
    TaskContractError,
    load_task_scope,
    load_tasks,
    validate_seed_members,
)


class ModelClient(Protocol):
    async def complete(
        self, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]: ...


class TokenCounterLike(Protocol):
    def text(self, content: str) -> int: ...

    def context(
        self, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]]
    ) -> int: ...


class ToolCaller(Protocol):
    async def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]: ...


class RoutedToolCaller:
    """Route an exposed tool to the one MCP server that advertised it."""

    def __init__(
        self,
        environments: Mapping[str, EnterpriseOpsEnvironment],
        routes: Mapping[str, str],
    ) -> None:
        self.environments = dict(environments)
        self.routes = dict(routes)

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        server_name = self.routes.get(name)
        if server_name is None or server_name not in self.environments:
            raise InfrastructureError(f"tool {name!r} has no MCP route")
        return await self.environments[server_name].call_tool(name, arguments)


def _normalize_deepseek_messages(
    messages: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    normalized = [dict(message) for message in messages]
    for message in normalized:
        if message.get("role") == "tool":
            message.pop("name", None)
    return normalized


class OpenAIChatClient:
    """Single-attempt OpenAI-compatible client; deliberately has no retry layer."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_name: str,
        temperature: float,
        max_tokens: int,
        seed: int | None,
        enable_thinking: bool,
        provider: str = "local_vllm",
        api_key: str | None = None,
        trust_env: bool = False,
    ) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.seed = seed
        self.enable_thinking = enable_thinking
        self.provider = provider
        self.api_key = api_key
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),
            trust_env=trust_env,
        )

    def _safe_error_detail(self, value: Any) -> str:
        detail = str(value)
        if self.api_key:
            detail = detail.replace(self.api_key, "<redacted-api-credential>")
        detail = re.sub(
            r"(?i)sk-[A-Za-z0-9_-]{8,}",
            "<redacted-api-key>",
            detail,
        )
        return detail[:2000]

    async def complete(
        self, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]]
    ) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
        request_messages = (
            _normalize_deepseek_messages(messages)
            if self.provider == "deepseek"
            else [dict(message) for message in messages]
        )
        payload: dict[str, Any] = {
            "model": self.model_name,
            "messages": request_messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.provider == "local_vllm":
            if not tools:
                payload.pop("tools")
            payload["seed"] = self.seed
            payload["chat_template_kwargs"] = {"enable_thinking": self.enable_thinking}
            # Keep tools in the template, but bypass the server's lossy parser.
            # vLLM's exclude_tools_when_tool_choice_none must remain false.
            payload.update(tool_choice="none", skip_special_tokens=False,
                           stop=[TOOL_CALL_END], include_stop_str_in_output=True)
        elif self.provider == "deepseek":
            # DeepSeek uses a top-level thinking object.
            payload["thinking"] = {
                "type": "enabled" if self.enable_thinking else "disabled"
            }
        else:
            raise InfrastructureError(f"unsupported model provider: {self.provider!r}")
        authorization = self.api_key if self.api_key is not None else "not-needed"
        try:
            response = await self._client.post(
                f"{self.endpoint}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {authorization}"},
            )
            response.raise_for_status()
            body = response.json()
        except httpx.HTTPStatusError as exc:
            try:
                response_body = exc.response.json()
            except ValueError:
                response_body = exc.response.text.strip()[:2000]
            if isinstance(response_body, Mapping):
                error = response_body.get("error")
                if isinstance(error, Mapping):
                    response_body = {
                        key: error[key]
                        for key in ("message", "type", "param", "code")
                        if key in error
                    }
            detail = self._safe_error_detail(
                json.dumps(response_body, ensure_ascii=False, default=str)
            )
            raise InfrastructureError(
                f"model request failed: HTTP {exc.response.status_code}: {detail}"
            ) from exc
        except (httpx.HTTPError, ValueError) as exc:
            raise InfrastructureError(
                f"model request failed: {self._safe_error_detail(exc)}"
            ) from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        if (
            not isinstance(choices, list)
            or not choices
            or not isinstance(choices[0], dict)
        ):
            raise InfrastructureError("model response lacks choices[0]")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise InfrastructureError("model response lacks choices[0].message")
        usage = body.get("usage") or {}
        if not isinstance(usage, dict):
            raise InfrastructureError("model response usage must be an object")
        usage = dict(usage)
        usage["finish_reason"] = choices[0].get("finish_reason")
        usage["response_id"] = body.get("id")
        usage["response_model"] = body.get("model")
        usage["system_fingerprint"] = body.get("system_fingerprint")
        if self.provider == "local_vllm":
            if message.get("tool_calls") or not isinstance(message.get("content"), str):
                raise InfrastructureError("local model must return raw text for shared tool parsing")
            message = parse_tool_response(
                message["content"], finish_reason=usage["finish_reason"],
                call_id=f'call_{body.get("id") or "local"}',
            )
        return message, usage

    async def close(self) -> None:
        await self._client.aclose()


class TokenCounter:
    def __init__(self, model_path: Path, *, enable_thinking: bool) -> None:
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            local_files_only=True,
            fix_mistral_regex=True,
        )
        self.enable_thinking = enable_thinking

    def text(self, content: str) -> int:
        return len(self.tokenizer.encode(content, add_special_tokens=False))

    def context(
        self, messages: list[Mapping[str, Any]], tools: list[Mapping[str, Any]]
    ) -> int:
        template_messages = _normalize_vllm_chat_template_messages(messages)
        try:
            tokens = self.tokenizer.apply_chat_template(
                template_messages,
                tools=tools,
                tokenize=True,
                return_dict=False,
                add_generation_prompt=True,
                enable_thinking=self.enable_thinking,
            )
        except Exception as exc:
            raise InfrastructureError(
                f"local chat-template tokenization failed: {exc}"
            ) from exc
        if not isinstance(tokens, list) or any(
            not isinstance(token, int) for token in tokens
        ):
            raise InfrastructureError(
                "local chat-template tokenization did not return a flat token ID list"
            )
        return len(tokens)


def _normalize_vllm_chat_template_messages(
    messages: list[Mapping[str, Any]],
) -> list[Mapping[str, Any]]:
    """Mirror vLLM 0.24 preprocessing before applying a tool-use chat template."""

    normalized = deepcopy(messages)
    for message in normalized:
        if message.get("role") != "assistant" or "tool_calls" not in message:
            continue
        tool_calls = message.get("tool_calls")
        if not isinstance(tool_calls, list):
            raise InfrastructureError("assistant message.tool_calls must be a list")
        if not tool_calls:
            message.pop("tool_calls", None)
            continue
        for tool_call in tool_calls:
            if not isinstance(tool_call, dict):
                raise InfrastructureError(
                    "assistant tool_calls entries must be objects"
                )
            function = tool_call.get("function")
            if tool_call.get("type", "function") != "function" or not isinstance(
                function, dict
            ):
                raise InfrastructureError(
                    "assistant tool_calls only support function entries"
                )
            arguments = function.get("arguments")
            if arguments:
                if not isinstance(arguments, (dict, list)):
                    try:
                        arguments = json.loads(arguments)
                    except (TypeError, json.JSONDecodeError) as exc:
                        raise InfrastructureError(
                            "assistant tool-call arguments are not valid JSON"
                        ) from exc
                function["arguments"] = arguments if arguments is not None else {}
            else:
                function["arguments"] = {}
    return normalized


def _openai_tools(
    discovered_by_server: Mapping[str, list[Mapping[str, Any]]],
    task: EnterpriseOpsTask,
) -> tuple[
    list[Mapping[str, Any]],
    Mapping[str, Mapping[str, Any]],
    Mapping[str, str],
    tuple[str, ...],
]:
    owners_by_name: dict[str, list[tuple[str, Mapping[str, Any]]]] = {}
    for server_name, discovered in discovered_by_server.items():
        names_on_server: set[str] = set()
        for tool in discovered:
            name = tool.get("name")
            if not isinstance(name, str) or not name:
                raise InfrastructureError(
                    f"MCP server {server_name!r} returned a tool without a valid name"
                )
            if name in names_on_server:
                raise InfrastructureError(
                    f"MCP server {server_name!r} returned duplicate tool name {name!r}"
                )
            names_on_server.add(name)
            owners_by_name.setdefault(name, []).append((server_name, tool))

    selected_names = [
        name
        for name in task.unique_selected_tools
        if name not in set(task.restricted_tools)
    ]
    missing = [name for name in selected_names if name not in owners_by_name]
    allowed_missing = {
        correction.tool_name for correction in task.missing_tool_corrections
    }
    unexpected_missing = [name for name in missing if name not in allowed_missing]
    if unexpected_missing:
        raise InfrastructureError(
            f"declared tools missing from MCP server: {unexpected_missing}"
        )
    omitted_missing = tuple(name for name in selected_names if name in set(missing))
    exposed_names = [name for name in selected_names if name not in set(missing)]

    tools: list[Mapping[str, Any]] = []
    selected: dict[str, Mapping[str, Any]] = {}
    routes: dict[str, str] = {}
    corrections = {
        correction.tool_name: correction for correction in task.tool_owner_corrections
    }
    for name in exposed_names:
        owners = owners_by_name[name]
        correction = corrections.get(name)
        if correction is not None:
            observed_server_names = tuple(
                sorted(server_name for server_name, _ in owners)
            )
            expected_server_names = tuple(sorted(correction.expected_server_names))
            if observed_server_names != expected_server_names:
                raise InfrastructureError(
                    f"selected tool {name!r} no longer matches its pinned owner set: "
                    f"expected={list(expected_server_names)}, "
                    f"observed={list(observed_server_names)}"
                )
            matching_owners = [
                owner for owner in owners if owner[0] == correction.target_server_name
            ]
            if len(matching_owners) != 1:
                raise InfrastructureError(
                    f"selected tool {name!r} pinned target "
                    f"{correction.target_server_name!r} is not uniquely available"
                )
            server_name, tool = matching_owners[0]
        elif len(owners) != 1:
            raise InfrastructureError(
                f"selected tool {name!r} is advertised by multiple MCP servers: "
                f"{sorted(server_name for server_name, _ in owners)}"
            )
        else:
            server_name, tool = owners[0]
        schema = tool.get("inputSchema", {"type": "object", "properties": {}})
        if not isinstance(schema, dict):
            raise InfrastructureError(f"tool {name!r} has a non-object inputSchema")
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": tool.get("description", ""),
                    "parameters": schema,
                },
            }
        )
        selected[name] = tool
        routes[name] = server_name
    if len(tools) != len(exposed_names):
        raise InfrastructureError(
            "tool exposure count does not match declared unique tools"
        )
    return tools, selected, routes, omitted_missing


def _effective_mode(mode: str, omitted_missing_tools: tuple[str, ...]) -> str:
    """Describe the actual distractor count after a registered source omission."""

    match = re.fullmatch(r"plus_(\d+)_tools", mode)
    if not match or not omitted_missing_tools:
        return mode
    return f"plus_{int(match.group(1)) - len(omitted_missing_tools)}_tools"


def _tool_content(value: Mapping[str, Any]) -> str:
    observation = {key: item for key, item in value.items() if key != "outcome"}
    return json.dumps(
        observation, ensure_ascii=False, default=str, separators=(",", ":")
    )


def _argument_schema_diagnostics(
    tool: Mapping[str, Any] | None,
    arguments: Mapping[str, Any],
) -> Mapping[str, Any]:
    schema = tool.get("inputSchema", {}) if tool is not None else {}
    if not isinstance(schema, Mapping):
        raise InfrastructureError("selected tool has a non-object inputSchema")
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if not isinstance(required, list) or not all(
        isinstance(field, str) for field in required
    ):
        raise InfrastructureError("tool inputSchema.required must be a string list")
    if not isinstance(properties, Mapping):
        raise InfrastructureError("tool inputSchema.properties must be an object")
    property_names = [field for field in properties if isinstance(field, str)]
    violations: list[dict[str, Any]] = []

    def add(path: str, kind: str, **details: Any) -> None:
        violations.append({"path": path, "kind": kind, **details})

    def matches_type(value: Any, expected: str) -> bool:
        return {
            "object": isinstance(value, Mapping),
            "array": isinstance(value, list),
            "string": isinstance(value, str),
            "integer": isinstance(value, int) and not isinstance(value, bool),
            "number": isinstance(value, (int, float)) and not isinstance(value, bool),
            "boolean": isinstance(value, bool),
            "null": value is None,
        }.get(expected, True)

    def visit(value: Any, node: Any, path: str) -> None:
        if not isinstance(node, Mapping):
            return
        expected = node.get("type")
        if isinstance(expected, str) and not matches_type(value, expected):
            add(path, "type_mismatch", expected=expected, actual=type(value).__name__)
            return
        if isinstance(value, Mapping):
            nested_required = node.get("required", [])
            nested_properties = node.get("properties", {})
            if not isinstance(nested_required, list) or not all(
                isinstance(field, str) for field in nested_required
            ):
                return
            if not isinstance(nested_properties, Mapping):
                return
            for field in sorted(set(nested_required)):
                if field not in value:
                    add(f"{path}.{field}" if path else field, "missing_required")
            for field in sorted(value, key=str):
                item = value[field]
                item_path = f"{path}.{field}" if path else str(field)
                if field in nested_properties:
                    visit(item, nested_properties[field], item_path)
                elif node.get("additionalProperties") is False:
                    add(item_path, "additional_property")
        elif isinstance(value, list):
            item_schema = node.get("items")
            if isinstance(item_schema, Mapping):
                for index, item in enumerate(value):
                    visit(item, item_schema, f"{path}[{index}]")

    visit(arguments, schema, "")
    return {
        "schema_required_arguments": sorted(set(required)),
        "schema_property_arguments": sorted(set(property_names)),
        "missing_required_arguments": sorted(
            field for field in set(required) if field not in arguments
        ),
        "arguments_not_in_schema_properties": sorted(
            field for field in arguments if field not in properties
        ),
        "schema_violations": violations,
    }


PLAN_PRESERVING_CONTROLLER = "first_tool_only_plan_preserving_v1"


def _first_multi_tool_turn(attempt: Mapping[str, Any]) -> int | None:
    """Return the recorded turn number of the first batched tool call."""

    trajectory = attempt.get("trajectory")
    if not isinstance(trajectory, Mapping):
        return None
    turns = trajectory.get("turns")
    if not isinstance(turns, list):
        return None
    for fallback_index, turn in enumerate(turns, start=1):
        if not isinstance(turn, Mapping):
            continue
        calls = turn.get("tool_calls")
        if isinstance(calls, list) and len(calls) > 1:
            turn_number = turn.get("turn", fallback_index)
            if not isinstance(turn_number, int) or isinstance(turn_number, bool):
                raise TaskContractError(
                    "plan-preserving source has a non-integer turn number"
                )
            return turn_number
    return None


def _load_plan_preserving_attempts(
    path: Path,
    *,
    domain: str,
    mode: str,
    selected_task_ids: set[str],
    attempts_per_task: int,
) -> dict[tuple[str, int], Mapping[str, Any]]:
    """Load only source attempts that actually contain a multi-tool turn."""

    if not path.is_file():
        raise TaskContractError(f"plan-preserving source attempts do not exist: {path}")
    selected: dict[tuple[str, int], Mapping[str, Any]] = {}
    seen_matching_keys: set[tuple[str, int]] = set()
    try:
        stream = path.open("r", encoding="utf-8")
    except OSError as exc:
        raise TaskContractError(
            f"cannot read plan-preserving source attempts {path}: {exc}"
        ) from exc
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                attempt = json.loads(line)
            except json.JSONDecodeError as exc:
                raise TaskContractError(
                    f"invalid plan-preserving source JSON at {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(attempt, Mapping):
                raise TaskContractError(
                    f"plan-preserving source row is not an object: {path}:{line_number}"
                )
            if attempt.get("domain") != domain or attempt.get("mode") != mode:
                continue
            task_id = attempt.get("task_id")
            attempt_index = attempt.get("attempt")
            if isinstance(task_id, str) and task_id not in selected_task_ids:
                continue
            if (
                not isinstance(task_id, str)
                or not isinstance(attempt_index, int)
                or isinstance(attempt_index, bool)
                or not 1 <= attempt_index <= attempts_per_task
            ):
                raise TaskContractError(
                    "plan-preserving source attempt is outside the selected task/attempt "
                    f"scope: {path}:{line_number}"
                )
            key = (task_id, attempt_index)
            if key in seen_matching_keys:
                raise TaskContractError(
                    f"duplicate plan-preserving source task/attempt key {key}: "
                    f"{path}:{line_number}"
                )
            seen_matching_keys.add(key)
            cleanup = attempt.get("cleanup")
            if (
                attempt.get("valid") is not True
                or attempt.get("status") not in {"success", "task_failure"}
                or not isinstance(cleanup, Mapping)
                or cleanup.get("succeeded") is not True
            ):
                raise TaskContractError(
                    f"plan-preserving source attempt is not a valid completed rollout: "
                    f"{path}:{line_number}"
                )
            first_multi_turn = _first_multi_tool_turn(attempt)
            if first_multi_turn is None:
                continue
            selected[key] = {
                "attempt": attempt,
                "source_path": str(path),
                "source_line": line_number,
                "first_multi_tool_turn": first_multi_turn,
            }
    if not selected:
        raise TaskContractError(
            f"plan-preserving source contains no affected attempts for {domain}/{mode}"
        )
    return selected


def _normalized_assistant_tool_calls(
    raw_tool_calls: Any,
) -> list[Mapping[str, Any]]:
    if raw_tool_calls is None:
        return []
    if not isinstance(raw_tool_calls, list):
        raise InfrastructureError("model message.tool_calls must be a list")
    assistant_tool_calls: list[Mapping[str, Any]] = []
    for call in raw_tool_calls:
        if not isinstance(call, dict):
            raise InfrastructureError("model tool call must be an object")
        call_id = call.get("id")
        function = call.get("function")
        if (
            not isinstance(call_id, str)
            or not call_id
            or not isinstance(function, dict)
        ):
            raise InfrastructureError("model tool call lacks id/function")
        name = function.get("name")
        arguments = function.get("arguments")
        if not isinstance(name, str) or not name or not isinstance(arguments, str):
            raise InfrastructureError("model tool call lacks string name/arguments")
        assistant_tool_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    return assistant_tool_calls


async def run_agent_loop(
    *,
    task: EnterpriseOpsTask,
    model: ModelClient,
    environment: ToolCaller,
    tools: list[Mapping[str, Any]],
    selected_tools: Mapping[str, Mapping[str, Any]],
    token_counter: TokenCounterLike,
    max_turns: int,
    max_context_tokens: int,
    max_generation_tokens: int,
    tool_routes: Mapping[str, str] | None = None,
    plan_preserving: bool = True,
    plan_preserving_source: Mapping[str, Any] | None = None,
    resume_trajectory: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    messages: list[Mapping[str, Any]] = [
        {"role": "system", "content": task.system_prompt},
        {"role": "user", "content": task.user_prompt},
    ]
    turns: list[Mapping[str, Any]] = []
    tool_events: list[Mapping[str, Any]] = []
    context_tokens: list[int] = []
    context_checks: list[int] = []
    reported_prompt_tokens = 0
    reported_completion_tokens = 0
    observation_tokens = 0
    termination = "max_turns"
    context_limit: Mapping[str, Any] | None = None
    protocol_error: Mapping[str, Any] | None = None
    static_prompt_tokens = token_counter.text(task.system_prompt) + token_counter.text(
        task.user_prompt
    )
    initial_counter = getattr(token_counter, 'initial_context', token_counter.context)
    initial_context_without_tools = initial_counter(messages, [])
    initial_context_with_tools = initial_counter(messages, tools)
    if plan_preserving_source is not None and not plan_preserving:
        raise TaskContractError(
            "a plan-preserving source cannot be used when the controller is disabled"
        )
    controller: Mapping[str, Any] | None = (
        {
            "name": PLAN_PRESERVING_CONTROLLER,
            "activation": "live_default",
            "assistant_content_policy": "preserve_full",
            "tool_call_policy": "execute_first_only_then_reinfer",
        }
        if plan_preserving
        else None
    )

    async def execute_tool_calls(
        *,
        turn_index: int,
        content: str,
        calls: list[Mapping[str, Any]],
        origin: str,
    ) -> bool:
        nonlocal observation_tokens, protocol_error
        parsed_calls: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
        for call in calls:
            function = call["function"]
            arguments_text = function["arguments"]
            try:
                arguments = json.loads(arguments_text)
                if not isinstance(arguments, dict):
                    raise ValueError("arguments must decode to an object")
            except (json.JSONDecodeError, ValueError) as exc:
                protocol_error = {
                    "turn": turn_index,
                    "tool_call_id": call["id"],
                    "name": function["name"],
                    "raw_arguments": arguments_text,
                    "error": f"invalid tool arguments: {exc}",
                    "mcp_called": False,
                }
                return False
            parsed_calls.append((call, arguments))

        messages.append(
            {
                "role": "assistant",
                "content": content,
                "tool_calls": deepcopy(calls),
            }
        )
        for call, arguments in parsed_calls:
            function = call["function"]
            name = function["name"]
            tool_error: str | None = None
            if name not in selected_tools:
                tool_error = f"tool {name!r} is not exposed for this task"
            schema_diagnostics = _argument_schema_diagnostics(
                selected_tools.get(name), arguments
            )
            if tool_error is None:
                result = await environment.call_tool(name, arguments)
            else:
                result = {"is_error": True, "error": tool_error}
            observation = _tool_content(result)
            observation_tokens += token_counter.text(observation)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "name": name,
                    "content": observation,
                }
            )
            tool_events.append(
                {
                    "turn": turn_index,
                    "tool_call_id": call["id"],
                    "name": name,
                    "gym_name": tool_routes.get(name)
                    if tool_routes is not None
                    else None,
                    "arguments": arguments,
                    **schema_diagnostics,
                    "result": result,
                    "agent_protocol_error": tool_error,
                    **({"origin": origin} if plan_preserving else {}),
                }
            )
            hook = getattr(environment, 'after_tool_event', None)
            if hook is not None:
                await hook(tool_events[-1])
        return True

    next_turn_index = 1
    if plan_preserving_source is not None:
        source_attempt = plan_preserving_source.get("attempt")
        if not isinstance(source_attempt, Mapping):
            raise TaskContractError("plan-preserving source envelope lacks an attempt")
        source_trajectory = source_attempt.get("trajectory")
        source_turns = (
            source_trajectory.get("turns")
            if isinstance(source_trajectory, Mapping)
            else None
        )
        first_multi_turn = plan_preserving_source.get("first_multi_tool_turn")
        if not isinstance(source_turns, list) or not isinstance(first_multi_turn, int):
            raise TaskContractError(
                "plan-preserving source lacks a trajectory/first multi-tool turn"
            )
        controller = {
            "name": PLAN_PRESERVING_CONTROLLER,
            "activation": "source_branch",
            "source_path": plan_preserving_source.get("source_path"),
            "source_line": plan_preserving_source.get("source_line"),
            "source_task_id": source_attempt.get("task_id"),
            "source_attempt": source_attempt.get("attempt"),
            "first_multi_tool_turn": first_multi_turn,
            "assistant_content_policy": "preserve_full",
            "tool_call_policy": "execute_first_only_then_reinfer",
        }
        for fallback_index, source_turn in enumerate(source_turns, start=1):
            if not isinstance(source_turn, Mapping):
                raise TaskContractError("plan-preserving source turn is not an object")
            source_turn_index = source_turn.get("turn", fallback_index)
            if source_turn_index != fallback_index:
                raise TaskContractError(
                    "plan-preserving source turns must be contiguous and one-indexed"
                )
            raw_calls = _normalized_assistant_tool_calls(
                source_turn.get("tool_calls") or []
            )
            if source_turn_index < first_multi_turn and len(raw_calls) > 1:
                raise TaskContractError(
                    "plan-preserving source first multi-tool turn is inconsistent"
                )
            if source_turn_index > first_multi_turn:
                break
            if source_turn_index < first_multi_turn and not raw_calls:
                raise TaskContractError(
                    "plan-preserving source stops before its first multi-tool turn"
                )
            if source_turn_index == first_multi_turn and len(raw_calls) <= 1:
                raise TaskContractError(
                    "plan-preserving branch turn does not contain multiple tool calls"
                )
            content = source_turn.get("assistant_content")
            if content is None:
                content = ""
            if not isinstance(content, str):
                content = json.dumps(content, ensure_ascii=False, default=str)
            effective_calls = raw_calls[:1] if len(raw_calls) > 1 else raw_calls
            local_context_tokens = token_counter.context(messages, tools)
            context_checks.append(local_context_tokens)
            turns.append(
                {
                    "turn": source_turn_index,
                    "origin": (
                        "source_branch"
                        if source_turn_index == first_multi_turn
                        else "source_replay"
                    ),
                    "finish_reason": source_turn.get("finish_reason"),
                    "local_context_tokens": local_context_tokens,
                    "reported_usage": {
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "total_tokens": 0,
                        "response_id": None,
                        "response_model": None,
                        "system_fingerprint": None,
                    },
                    "source_reported_usage": deepcopy(
                        source_turn.get("reported_usage") or {}
                    ),
                    "assistant_content": content,
                    "raw_tool_calls": deepcopy(raw_calls),
                    "tool_calls": deepcopy(effective_calls),
                    "discarded_tool_calls": deepcopy(raw_calls[1:]),
                }
            )
            if not await execute_tool_calls(
                turn_index=source_turn_index,
                content=content,
                calls=effective_calls,
                origin=(
                    "source_branch"
                    if source_turn_index == first_multi_turn
                    else "source_replay"
                ),
            ):
                termination = "model_protocol_error"
                break
            next_turn_index = source_turn_index + 1
            if source_turn_index == first_multi_turn:
                break

    if resume_trajectory is not None:
        if plan_preserving_source is not None or not plan_preserving:
            raise TaskContractError("checkpoint prefix cannot be mixed with source replay")
        from robustops.evaluation.fault_checkpoint import prefix_messages
        messages = prefix_messages(task, resume_trajectory)
        turns = deepcopy(resume_trajectory['turns'])
        tool_events = deepcopy(resume_trajectory['tool_events'])
        for expected_index, turn in enumerate(turns, 1):
            if turn['turn'] != expected_index:
                raise TaskContractError("checkpoint prefix turns are not contiguous")
            turn['origin'] = 'checkpoint_restored'
            turn['source_reported_usage'] = deepcopy(turn.get('reported_usage') or {})
            turn['reported_usage'] = dict(prompt_tokens=0, completion_tokens=0, total_tokens=0)
        for event in tool_events:
            event['origin'] = 'checkpoint_restored'
        observation_tokens = sum(token_counter.text(m['content']) for m in messages if m['role'] == 'tool')
        next_turn_index = len(turns) + 1
        if controller is not None:
            controller = dict(controller, activation='checkpoint_restored',
                              restored_tool_calls=len(tool_events), prefix_reexecuted_calls=0)

    for turn_index in range(next_turn_index, max_turns + 1):
        if protocol_error is not None:
            break
        if getattr(environment, "exhausted", False):
            termination = "fault_budget_exhausted"
            break
        local_context_tokens = token_counter.context(messages, tools)
        context_checks.append(local_context_tokens)
        if local_context_tokens + max_generation_tokens > max_context_tokens:
            termination = "context_limit_pre_request"
            context_limit = {
                "turn": turn_index,
                "prompt_tokens": local_context_tokens,
                "reserved_generation_tokens": max_generation_tokens,
                "max_context_tokens": max_context_tokens,
            }
            break
        context_tokens.append(local_context_tokens)
        message, usage = await model.complete(messages, tools)

        reported_prompt_tokens += int(usage.get("prompt_tokens") or 0)
        reported_completion_tokens += int(usage.get("completion_tokens") or 0)
        content = message.get("content")
        if content is None:
            content = ""
        if not isinstance(content, str):
            content = json.dumps(content, ensure_ascii=False, default=str)
        raw_tool_calls = _normalized_assistant_tool_calls(
            message.get("tool_calls") or []
        )
        assistant_tool_calls = (
            raw_tool_calls[:1]
            if plan_preserving and len(raw_tool_calls) > 1
            else raw_tool_calls
        )

        turn_record = {
            "turn": turn_index,
            "finish_reason": usage.get("finish_reason"),
            "local_context_tokens": local_context_tokens,
            "reported_usage": {
                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                "completion_tokens": int(usage.get("completion_tokens") or 0),
                "total_tokens": int(usage.get("total_tokens") or 0),
                "response_id": usage.get("response_id"),
                "response_model": usage.get("response_model"),
                "system_fingerprint": usage.get("system_fingerprint"),
            },
            "assistant_content": content,
            "tool_calls": deepcopy(assistant_tool_calls),
        }
        if plan_preserving:
            turn_record.update(
                {
                    "origin": (
                        "model_reinfer"
                        if plan_preserving_source is not None
                        else "model_live"
                    ),
                    "raw_tool_calls": deepcopy(raw_tool_calls),
                    "discarded_tool_calls": deepcopy(raw_tool_calls[1:]),
                }
            )
        turns.append(turn_record)

        if "raw_model_content" in message:
            turn_record["raw_model_content"] = message["raw_model_content"]
        if message.get("model_protocol_error") is not None:
            protocol_error = dict(message["model_protocol_error"], turn=turn_index,
                                  raw_output=message.get("raw_model_content", content))
            turn_record["model_protocol_error"] = deepcopy(protocol_error)
            messages.append({"role": "assistant", "content": content})
            termination = "model_protocol_error"
            break

        if not assistant_tool_calls:
            messages.append({"role": "assistant", "content": content})
            termination = (
                "max_tokens" if usage.get("finish_reason") == "length" else "model_stop"
            )
            break

        if not await execute_tool_calls(
            turn_index=turn_index,
            content=content,
            calls=assistant_tool_calls,
            origin=(
                "model_reinfer" if plan_preserving_source is not None else "model_live"
            ),
        ):
            termination = "model_protocol_error"
            break

        if getattr(environment, "stop_after_capture", False):
            termination = "state_capture"
            break

    if getattr(environment, "exhausted", False):
        termination = "fault_budget_exhausted"
    trajectory = {
        "termination": termination,
        "context_limit": context_limit,
        "controller": controller,
        "protocol_valid": protocol_error is None,
        "protocol_error": protocol_error,
        "turn_count": len(turns),
        "tool_call_count": len(tool_events),
        "raw_tool_call_count": sum(
            len(turn.get("raw_tool_calls", turn.get("tool_calls", [])))
            for turn in turns
        ),
        "discarded_tool_call_count": sum(
            len(turn.get("discarded_tool_calls", [])) for turn in turns
        ),
        "failed_tool_call_count": sum(
            bool(event["agent_protocol_error"] or event["result"].get("is_error"))
            for event in tool_events
        ),
        "truncated_turn_count": sum(
            turn.get("finish_reason") == "length" for turn in turns
        ),
        "turns": turns,
        "tool_events": tool_events,
        "token_usage": {
            "local_static_system_user_tokens": static_prompt_tokens,
            "local_initial_context_without_tools": initial_context_without_tools,
            "local_initial_context_with_tools": initial_context_with_tools,
            "local_tool_schema_increment_tokens": (
                initial_context_with_tools - initial_context_without_tools
            ),
            "local_context_tokens_per_turn": context_tokens,
            "local_context_tokens_checked": context_checks,
            "local_peak_context_tokens": max(context_checks, default=0),
            "max_context_tokens": max_context_tokens,
            "reserved_generation_tokens": max_generation_tokens,
            "local_tool_observation_tokens": observation_tokens,
            "reported_prompt_tokens_sum": reported_prompt_tokens,
            "reported_completion_tokens_sum": reported_completion_tokens,
        },
    }
    return trajectory


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


async def run_task_attempt(
    *,
    task: EnterpriseOpsTask,
    model: ModelClient,
    token_counter: TokenCounterLike,
    archive_path: Path,
    mcp_url_overrides: Mapping[str, str],
    max_turns: int,
    max_context_tokens: int,
    max_generation_tokens: int,
    experiment_id: str,
    run_id: str,
    mode: str,
    model_name: str,
    model_endpoint: str,
    seed: int | None,
    attempt_index: int = 1,
    plan_preserving: bool = True,
    plan_preserving_source: Mapping[str, Any] | None = None,
    tool_intervention: Any | None = None,
    initial_state: dict[str, str] | None = None,
    resume_trajectory: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    started_at = _utc_now()
    started = time.monotonic()
    stage = "task_contract"
    environments: dict[str, EnterpriseOpsEnvironment] = {}
    trajectory: Mapping[str, Any] | None = None
    verifier_results: list[Mapping[str, Any]] = []
    status = "invalid"
    error: Mapping[str, Any] | None = None
    cleanup_succeeded = True
    cleanup_error: str | None = None
    cleanup_results: dict[str, Mapping[str, Any]] = {}
    exposed_tool_count = 0
    seeded_server_count = 0
    connected_server_count = 0
    applied_tool_owner_corrections: set[str] = set()
    omitted_missing_tools: tuple[str, ...] = ()

    try:
        for server in task.gym_servers:
            server_name = server["mcp_server_name"]
            environments[server_name] = EnterpriseOpsEnvironment(
                server_name=server_name,
                base_url=mcp_url_overrides.get(server_name, server["mcp_server_url"]),
                mcp_endpoint=task.mcp_endpoint,
                seed_database_file=server["seed_database_file"],
                context=server["context"],
                database_archive=archive_path,
            )

        for server_name, environment in environments.items():
            stage = f"database_seed:{server_name}"
            if initial_state is None:
                await environment.seed()
            else:
                await environment.seed(initial_state.get(server_name))
                captured = await environment.export_seed_state()
                if server_name not in initial_state:
                    initial_state[server_name] = captured
                else:
                    from robustops.evaluation.seed_state import state_contents
                    if state_contents(initial_state[server_name]) != state_contents(captured):
                        raise InfrastructureError("restored initial database state differs")
            seeded_server_count += 1
        for server_name, environment in environments.items():
            stage = f"mcp_initialize:{server_name}"
            await environment.connect()
            connected_server_count += 1
        stage = "tool_discovery"
        discovered_by_server = {
            server_name: await environment.list_tools()
            for server_name, environment in environments.items()
        }
        tools, selected_tools, tool_routes, omitted_missing_tools = _openai_tools(
            discovered_by_server, task
        )
        if omitted_missing_tools:
            print(
                f"warning: {task.task_id}: omitting registered unavailable distractor(s) "
                f"{list(omitted_missing_tools)}; effective mode is "
                f"{_effective_mode(mode, omitted_missing_tools)}",
                file=sys.stderr,
            )
        applied_tool_owner_corrections = {
            correction.tool_name
            for correction in task.tool_owner_corrections
            if tool_routes.get(correction.tool_name) == correction.target_server_name
        }
        exposed_tool_count = len(tools)
        tool_caller = RoutedToolCaller(environments, tool_routes)
        if tool_intervention is not None:
            tool_caller = tool_intervention.bind(tool_caller, tools)

        stage = "agent_loop"
        trajectory = await run_agent_loop(
            task=task,
            model=model,
            environment=tool_caller,
            tools=tools,
            selected_tools=selected_tools,
            token_counter=token_counter,
            max_turns=max_turns,
            max_context_tokens=max_context_tokens,
            max_generation_tokens=max_generation_tokens,
            tool_routes=tool_routes,
            plan_preserving=plan_preserving,
            plan_preserving_source=plan_preserving_source,
            resume_trajectory=resume_trajectory,
        )

        stage = "verifier"
        for index, verifier in enumerate(task.verifiers, start=1):
            server_name = verifier["gym_name"]
            environment = environments.get(server_name)
            if environment is None:
                raise InfrastructureError(
                    f"verifier {index} targets unavailable MCP server {server_name!r}"
                )
            verifier_results.append(await environment.run_verifier(verifier, index))
        declared_verifier_count = len(task.verifiers)
        if len(verifier_results) != declared_verifier_count:
            raise InfrastructureError(
                f"executed {len(verifier_results)}/{declared_verifier_count} verifiers"
            )
        status = (
            "task_failure"
            if trajectory.get("termination") in {"model_protocol_error", "fault_budget_exhausted"}
            else (
                "success"
                if all(result["passed"] for result in verifier_results)
                else "task_failure"
            )
        )
    except (InfrastructureError, TaskContractError) as exc:
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        status = "invalid"
    except Exception as exc:
        error = {"stage": stage, "type": type(exc).__name__, "message": str(exc)}
        status = "invalid"
    finally:
        cleanup_errors: list[str] = []
        for server_name, environment in environments.items():
            succeeded, server_error = await environment.cleanup()
            cleanup_results[server_name] = {
                "succeeded": succeeded,
                "error": server_error,
            }
            if not succeeded:
                cleanup_errors.append(f"{server_name}: {server_error}")
        cleanup_succeeded = not cleanup_errors
        cleanup_error = "; ".join(cleanup_errors) if cleanup_errors else None

    return {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "task_id": task.task_id,
        "domain": task.domain,
        "mode": mode,
        "attempt": attempt_index,
        "status": status,
        "success": status == "success",
        "valid": status in {"success", "task_failure"},
        "skip": None,
        "context_limit_terminated": (
            trajectory is not None
            and trajectory.get("termination") == "context_limit_pre_request"
        ),
        "fault": (tool_intervention.report() if tool_intervention is not None
                  else {"enabled": False, "triggered": False}),
        "plan_preserving": (
            {
                "controller": PLAN_PRESERVING_CONTROLLER,
                "activation": (
                    "source_branch"
                    if plan_preserving_source is not None
                    else "live_default"
                ),
                "source_path": (
                    plan_preserving_source.get("source_path")
                    if plan_preserving_source is not None
                    else None
                ),
                "source_line": (
                    plan_preserving_source.get("source_line")
                    if plan_preserving_source is not None
                    else None
                ),
                "first_multi_tool_turn": (
                    plan_preserving_source.get("first_multi_tool_turn")
                    if plan_preserving_source is not None
                    else None
                ),
            }
            if plan_preserving
            else None
        ),
        "model": {"name": model_name, "endpoint": model_endpoint, "seed": seed},
        "task_contract": {
            "declared_server_count": len(task.gym_servers),
            "seeded_server_count": seeded_server_count,
            "connected_server_count": connected_server_count,
            "source_declared_tool_count": (
                task.selected_tool_addition_corrections[0].source_declared_tool_count
                if task.selected_tool_addition_corrections
                else len(task.selected_tools)
            ),
            "source_unique_declared_tool_count": (
                task.selected_tool_addition_corrections[
                    0
                ].source_unique_declared_tool_count
                if task.selected_tool_addition_corrections
                else len(task.unique_selected_tools)
            ),
            "declared_tool_count": len(task.selected_tools),
            "unique_declared_tool_count": len(task.unique_selected_tools),
            "duplicate_declared_tools": list(task.duplicate_selected_tools),
            "exposed_tool_count": exposed_tool_count,
            "effective_mode": _effective_mode(mode, omitted_missing_tools),
            "omitted_missing_tools": list(omitted_missing_tools),
            "missing_tool_corrections": [
                {
                    "tool_name": correction.tool_name,
                    "server_name": correction.server_name,
                    "rationale": correction.rationale,
                    "applied": correction.tool_name in omitted_missing_tools,
                }
                for correction in task.missing_tool_corrections
            ],
            "selected_tool_addition_corrections": [
                {
                    "tool_name": correction.tool_name,
                    "server_name": correction.server_name,
                    "source_variant": correction.source_variant,
                    "source_declared_tool_count": (
                        correction.source_declared_tool_count
                    ),
                    "source_unique_declared_tool_count": (
                        correction.source_unique_declared_tool_count
                    ),
                    "rationale": correction.rationale,
                    "applied": correction.tool_name in task.unique_selected_tools,
                }
                for correction in task.selected_tool_addition_corrections
            ],
            "context_key_corrections": [
                {
                    "server_name": correction.server_name,
                    "source_key": correction.source_key,
                    "target_key": correction.target_key,
                    "rationale": correction.rationale,
                    "applied": True,
                }
                for correction in task.context_key_corrections
            ],
            "context_value_corrections": [
                {
                    "server_name": correction.server_name,
                    "key": correction.key,
                    "target_identity": correction.target_identity,
                    "rationale": correction.rationale,
                    "applied": True,
                }
                for correction in task.context_value_corrections
            ],
            "seed_database_file_corrections": [
                {
                    "server_name": correction.server_name,
                    "source_seed_database_file": correction.source_seed_database_file,
                    "target_seed_database_file": correction.target_seed_database_file,
                    "rationale": correction.rationale,
                    "applied": True,
                }
                for correction in task.seed_database_file_corrections
            ],
            "tool_owner_corrections": [
                {
                    "tool_name": correction.tool_name,
                    "expected_server_names": list(correction.expected_server_names),
                    "target_server_name": correction.target_server_name,
                    "rationale": correction.rationale,
                    "applied": correction.tool_name in applied_tool_owner_corrections,
                }
                for correction in task.tool_owner_corrections
            ],
            "official_revised_overlays": [
                {
                    "task_id": overlay.task_id,
                    "domain": overlay.domain,
                    "source_mode": overlay.source_mode,
                    "relative_path": overlay.relative_path,
                    "revised_commit": overlay.revised_commit,
                    "changed_fields": list(overlay.changed_fields),
                    "source_verifier_count": overlay.source_verifier_count,
                    "revised_verifier_count": overlay.revised_verifier_count,
                    "already_revised_fields": list(overlay.already_revised_fields),
                    "applied": True,
                }
                for overlay in task.official_revised_overlays
            ],
            "verifier_corrections": [
                {
                    "index": correction.index,
                    "source_gym_name": correction.source_gym_name,
                    "target_gym_name": correction.target_gym_name,
                    "query_changed": correction.query_changed,
                    "rationale": correction.rationale,
                    "applied": True,
                }
                for correction in task.verifier_corrections
            ],
            "verifier_expected_value_corrections": [
                {
                    "index": correction.index,
                    "source_value": correction.source_value,
                    "target_value": correction.target_value,
                    "rationale": correction.rationale,
                    "applied": True,
                }
                for correction in task.verifier_expected_value_corrections
            ],
            "verifier_contract_corrections": [
                {
                    "source_verifier_count": correction.source_verifier_count,
                    "source_unique_verifier_count": (
                        correction.source_unique_verifier_count
                    ),
                    "removed_verifier_count": correction.removed_verifier_count,
                    "replaced_verifier_count": correction.replaced_verifier_count,
                    "added_verifier_count": correction.added_verifier_count,
                    "rationale": correction.rationale,
                    "applied": True,
                }
                for correction in task.verifier_contract_corrections
            ],
            "verifier_set_normalization": (
                {
                    "source_count": task.verifier_set_normalization.source_count,
                    "effective_count": task.verifier_set_normalization.effective_count,
                    "exact_duplicate_count": (
                        task.verifier_set_normalization.exact_duplicate_count
                    ),
                    "name_corrections": [
                        {
                            "index": correction.index,
                            "source_name": correction.source_name,
                            "target_name": correction.target_name,
                        }
                        for correction in task.verifier_set_normalization.name_corrections
                    ],
                }
                if task.verifier_set_normalization is not None
                else None
            ),
            "source_declared_database_verifier_count": (
                task.official_revised_overlays[0].source_verifier_count
                if task.official_revised_overlays
                and "verifiers" in task.official_revised_overlays[0].changed_fields
                else task.verifier_set_normalization.source_count
                if task.verifier_set_normalization is not None
                else len(task.verifiers)
            ),
            "source_declared_verifier_count": (
                task.official_revised_overlays[0].source_verifier_count
                if task.official_revised_overlays
                and "verifiers" in task.official_revised_overlays[0].changed_fields
                else task.verifier_set_normalization.source_count
                if task.verifier_set_normalization is not None
                else len(task.verifiers)
            ),
            "declared_database_verifier_count": len(task.verifiers),
            "declared_verifier_count": len(task.verifiers),
            "executed_verifier_count": len(verifier_results),
        },
        "started_at": started_at,
        "finished_at": _utc_now(),
        "duration_seconds": round(time.monotonic() - started, 3),
        "trajectory": trajectory,
        "verifiers": verifier_results,
        "error": error,
        "cleanup": {
            "succeeded": cleanup_succeeded,
            "error": cleanup_error,
            "servers": cleanup_results,
        },
    }


def _percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    low = int(index)
    high = min(low + 1, len(ordered) - 1)
    weight = index - low
    return ordered[low] * (1 - weight) + ordered[high] * weight


def summarize_attempts(
    attempts: list[Mapping[str, Any]],
    *,
    scheduled: int,
    wall_duration_seconds: float | None = None,
) -> Mapping[str, Any]:
    counts = Counter(attempt["status"] for attempt in attempts)
    valid_count = counts["success"] + counts["task_failure"]
    peaks = [
        float(attempt["trajectory"]["token_usage"]["local_peak_context_tokens"])
        for attempt in attempts
        if attempt.get("trajectory")
    ]
    durations = [float(attempt["duration_seconds"]) for attempt in attempts]
    trajectories = [
        attempt["trajectory"] for attempt in attempts if attempt.get("trajectory")
    ]
    context_limit_attempts = [
        attempt
        for attempt in attempts
        if (attempt.get("trajectory") or {}).get("termination")
        == "context_limit_pre_request"
    ]
    protocol_error_attempts = [
        attempt
        for attempt in attempts
        if (attempt.get("trajectory") or {}).get("termination")
        == "model_protocol_error"
    ]

    def trajectory_values(field: str) -> list[float]:
        return [float(trajectory[field]) for trajectory in trajectories]

    def distribution(values: list[float]) -> Mapping[str, Any]:
        return {
            "sum": sum(values),
            "median": statistics.median(values) if values else None,
            "p95": _percentile(values, 0.95),
            "max": max(values, default=None),
        }

    turns = trajectory_values("turn_count")
    tool_calls = trajectory_values("tool_call_count")
    failed_tool_calls = trajectory_values("failed_tool_call_count")
    truncated_turns = trajectory_values("truncated_turn_count")
    generated_tokens = [
        float(trajectory["token_usage"]["reported_completion_tokens_sum"])
        for trajectory in trajectories
    ]
    observation_tokens = [
        float(trajectory["token_usage"]["local_tool_observation_tokens"])
        for trajectory in trajectories
    ]
    if len(attempts) != scheduled:
        run_status = "incomplete"
    elif counts["invalid"]:
        run_status = "completed_with_invalid"
    else:
        run_status = "completed"
    summary = {
        "run_status": run_status,
        "task_failure_policy": "record_invalid_cleanup_and_continue",
        "scheduled": scheduled,
        "completed": len(attempts),
        "success": counts["success"],
        "task_failure": counts["task_failure"],
        "context_limit_terminated": len(context_limit_attempts),
        "context_limit_success": sum(
            attempt.get("status") == "success" for attempt in context_limit_attempts
        ),
        "context_limit_task_failure": sum(
            attempt.get("status") == "task_failure"
            for attempt in context_limit_attempts
        ),
        "model_protocol_error": len(protocol_error_attempts),
        "invalid": counts["invalid"],
        "invalid_attempts": [
            {
                "task_id": attempt.get("task_id"),
                "domain": attempt.get("domain"),
                "model_endpoint": (attempt.get("model") or {}).get("endpoint"),
                "error": attempt.get("error"),
                "cleanup_succeeded": (attempt.get("cleanup") or {}).get("succeeded"),
            }
            for attempt in attempts
            if attempt.get("status") == "invalid"
        ],
        "clean_tsr": counts["success"] / scheduled if scheduled else None,
        "valid_task_success_rate": counts["success"] / valid_count
        if valid_count
        else None,
        "duration_seconds": {
            "sum": round(sum(durations), 3),
            "median": statistics.median(durations) if durations else None,
            "p95": _percentile(durations, 0.95),
        },
        "peak_context_tokens": {
            "max": max(peaks, default=None),
            "p95": _percentile(peaks, 0.95),
        },
        "turns": distribution(turns),
        "tool_calls": distribution(tool_calls),
        "failed_tool_calls": distribution(failed_tool_calls),
        "truncated_turns": int(sum(truncated_turns)),
        "generated_tokens": distribution(generated_tokens),
        "tool_observation_tokens": distribution(observation_tokens),
        "cleanup_failures": sum(
            not attempt["cleanup"]["succeeded"] for attempt in attempts
        ),
    }
    if wall_duration_seconds is not None:
        summary["wall_duration_seconds"] = round(wall_duration_seconds, 3)
        summary["throughput_tasks_per_second"] = (
            len(attempts) / wall_duration_seconds if wall_duration_seconds > 0 else None
        )
    return summary


def _model_endpoints(args: argparse.Namespace) -> list[str]:
    raw = args.model_endpoints or args.model_endpoint
    endpoints = [value.strip().rstrip("/") for value in raw.split(",") if value.strip()]
    if not endpoints:
        raise TaskContractError("at least one model endpoint is required")
    return endpoints


def _load_recipe(path: Path) -> Mapping[str, Any]:
    try:
        recipe = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TaskContractError(f"cannot load recipe {path}: {exc}") from exc
    required = {
        "experiment_id",
        "dataset_revision",
        "model_revision",
        "domain",
        "mode",
        "expected_task_count",
        "data_path",
        "database_archive",
        "served_model_name",
        "generation",
        "scopes",
    }
    missing = sorted(required - set(recipe))
    if missing:
        raise TaskContractError(f"recipe missing fields: {missing}")
    if not isinstance(recipe["experiment_id"], str) or not recipe["experiment_id"].strip():
        raise TaskContractError("recipe experiment_id must be a nonempty string")
    domains = {"calendar", "csm", "drive", "email", "hr", "hybrid", "itsm", "teams"}
    if recipe["domain"] not in domains:
        raise TaskContractError(f"clean recipe domain must be one of {sorted(domains)}")
    modes = {"oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools"}
    if recipe["mode"] not in modes:
        raise TaskContractError(f"clean recipe mode must be one of {sorted(modes)}")
    generation = recipe["generation"]
    if not isinstance(generation, Mapping):
        raise TaskContractError("recipe generation must be an object")
    for field in ("max_tokens", "max_turns", "max_context_tokens"):
        value = generation.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value < 1:
            raise TaskContractError(
                f"recipe generation.{field} must be a positive integer"
            )
    if generation["max_tokens"] >= generation["max_context_tokens"]:
        raise TaskContractError(
            "recipe generation.max_tokens must be below max_context_tokens"
        )
    if generation["max_context_tokens"] != 32_768:
        raise TaskContractError(
            "recipe generation.max_context_tokens must be 32768; the Qwen3 "
            "student context boundary is the sole inference gate"
        )
    return recipe


def _git_state(project_root: Path) -> Mapping[str, Any]:
    def run(*args: str) -> tuple[int, str]:
        completed = subprocess.run(
            ["git", *args],
            cwd=project_root,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        return completed.returncode, completed.stdout.strip()

    try:
        code, revision = run("rev-parse", "HEAD")
        status_code, status = run("status", "--short")
    except FileNotFoundError as exc:
        return {
            "revision": "unavailable",
            "dirty": None,
            "status_short": [],
            "error": f"git executable unavailable: {exc}",
        }
    return {
        "revision": revision if code == 0 else "unborn",
        "dirty": bool(status) if status_code == 0 else None,
        "status_short": status.splitlines() if status_code == 0 else [],
    }


def _runtime_metadata() -> Mapping[str, Any]:
    packages = {}
    for name in ("torch", "vllm", "transformers", "pyarrow", "httpx", "verl"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    hardware: dict[str, Any] = {"cuda_available": False, "devices": []}
    try:
        import torch

        hardware["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            hardware["devices"] = [
                torch.cuda.get_device_name(index)
                for index in range(torch.cuda.device_count())
            ]
    except Exception as exc:
        hardware["probe_error"] = str(exc)
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "packages": packages,
        "hardware": hardware,
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def _resolve_project_path(project_root: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else project_root / path


def _parse_string_mapping(raw: str | None, *, option: str) -> dict[str, str]:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskContractError(f"{option} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str) and key and isinstance(item, str) and item
        for key, item in value.items()
    ):
        raise TaskContractError(f"{option} must map non-empty server names to strings")
    return value


def _parse_positive_int_mapping(raw: str | None, *, option: str) -> dict[str, int]:
    if raw is None:
        return {}
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise TaskContractError(f"{option} must be a JSON object: {exc}") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and key
        and isinstance(item, int)
        and not isinstance(item, bool)
        and item > 0
        for key, item in value.items()
    ):
        raise TaskContractError(
            f"{option} must map non-empty server names to positive integers"
        )
    return value


def _mcp_transport_retry_reason(attempt: Mapping[str, Any]) -> str | None:
    if attempt.get("status") != "invalid":
        return None
    cleanup = attempt.get("cleanup")
    error = attempt.get("error")
    if (
        not isinstance(cleanup, Mapping)
        or cleanup.get("succeeded") is not True
        or not isinstance(error, Mapping)
        or error.get("type") != "InfrastructureError"
        or not isinstance(error.get("message"), str)
    ):
        return None
    if re.match(r"^MCP .+ transport failed:", error["message"]):
        return "mcp_transport_invalid"
    return None


def _model_protocol_retry_reason(attempt: Mapping[str, Any]) -> str | None:
    return model_protocol_retry_reason(attempt)


def _validate_run_id(run_id: str) -> None:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", run_id):
        raise TaskContractError(
            "run ID must start with an alphanumeric character and contain only A-Z, a-z, 0-9, ., _, -"
        )


def _load_requested_task_ids(path: Path) -> list[str]:
    try:
        task_ids = [
            line.strip() for line in path.read_text(encoding="utf-8").splitlines()
        ]
    except OSError as exc:
        raise TaskContractError(f"cannot read task ID file {path}: {exc}") from exc
    task_ids = [task_id for task_id in task_ids if task_id]
    if not task_ids:
        raise TaskContractError(f"task ID file is empty: {path}")
    if len(task_ids) != len(set(task_ids)):
        raise TaskContractError(f"task ID file contains duplicates: {path}")
    return task_ids


async def _run(args: argparse.Namespace) -> int:
    project_root = args.project_root.resolve()
    recipe = _load_recipe(args.recipe.resolve())
    experiment_id = args.experiment_id or recipe["experiment_id"]
    data_path = _resolve_project_path(project_root, recipe["data_path"])
    archive_path = _resolve_project_path(project_root, recipe["database_archive"])
    if args.model_provider not in {"local_vllm", "deepseek"}:
        raise TaskContractError(f"unsupported model provider: {args.model_provider!r}")
    api_key: str | None = None
    if args.model_provider == "deepseek":
        if not re.fullmatch(r"[A-Z][A-Z0-9_]*", args.api_key_env or ""):
            raise TaskContractError(
                "deepseek requires --api-key-env naming an uppercase environment variable"
            )
        api_key = os.environ.get(args.api_key_env)
        if not args.dry_run and not api_key:
            raise TaskContractError(
                f"required API credential environment variable is unset: {args.api_key_env}"
            )
    if args.token_counter != "local_hf":
        raise TaskContractError(f"unsupported token counter: {args.token_counter!r}")
    configured_tokenizer_path = (
        args.tokenizer_path
        if args.tokenizer_path is not None
        else args.model_path
        if args.model_path is not None
        else recipe.get("model_path")
    )
    if configured_tokenizer_path is None:
        raise TaskContractError(
            "local_hf token counting requires --tokenizer-path, --model-path, "
            "or recipe model_path"
        )
    tokenizer_path = _resolve_project_path(
        project_root,
        configured_tokenizer_path,
    )
    token_counter: TokenCounterLike = TokenCounter(
        tokenizer_path,
        enable_thinking=bool(recipe["generation"]["enable_thinking"]),
    )
    tasks = load_tasks(
        data_path,
        expected_domain=recipe["domain"],
        expected_count=int(recipe["expected_task_count"]),
    )
    selected = load_task_scope(tasks, scope=args.scope, scopes=recipe["scopes"])
    if bool(args.task_manifest) != bool(args.partition):
        raise TaskContractError(
            "--task-manifest and --partition must be provided together"
        )
    if args.paired_mode and args.available_mode:
        raise TaskContractError(
            "--paired-mode and --available-mode are mutually exclusive"
        )
    if (args.paired_mode or args.available_mode) and not args.task_manifest:
        raise TaskContractError(
            "--paired-mode/--available-mode requires --task-manifest and --partition"
        )
    if args.task_ids_file and not args.task_manifest:
        raise TaskContractError(
            "--task-ids-file requires --task-manifest and --partition"
        )
    if args.paired_mode and recipe["mode"] not in {"oracle", args.paired_mode}:
        raise TaskContractError(
            f"recipe mode {recipe['mode']!r} cannot use paired mode {args.paired_mode!r}"
        )
    if args.available_mode and recipe["mode"] != args.available_mode:
        raise TaskContractError(
            f"recipe mode {recipe['mode']!r} cannot use available mode "
            f"{args.available_mode!r}"
        )
    if args.task_manifest:
        manifest_path = args.task_manifest.resolve()
        try:
            partition_ids = load_partition_task_ids(
                manifest_path,
                domain=recipe["domain"],
                partition=args.partition,
                paired_mode=args.paired_mode,
                available_mode=args.available_mode,
                runnable_mode=recipe["mode"],
            )
        except SplitContractError as exc:
            raise TaskContractError(str(exc)) from exc
        selected_by_id = {task.task_id: task for task in selected}
        missing = [
            task_id for task_id in partition_ids if task_id not in selected_by_id
        ]
        if missing:
            raise TaskContractError(
                f"task manifest partition contains IDs outside recipe scope: {missing}"
            )
        selected = [selected_by_id[task_id] for task_id in partition_ids]
    if args.task_ids_file:
        requested_ids = _load_requested_task_ids(args.task_ids_file.resolve())
        selected_by_id = {task.task_id: task for task in selected}
        missing = [
            task_id for task_id in requested_ids if task_id not in selected_by_id
        ]
        if missing:
            raise TaskContractError(
                f"requested task IDs are outside the selected domain/partition: {missing}"
            )
        selected = [selected_by_id[task_id] for task_id in requested_ids]
    if args.attempts_per_task < 1:
        raise TaskContractError("attempts per task must be at least 1")
    plan_preserving_enabled = not args.disable_plan_preserving
    plan_preserving_path: Path | None = None
    plan_preserving_attempts: dict[tuple[str, int], Mapping[str, Any]] = {}
    if args.plan_preserving_source_attempts is not None:
        if not plan_preserving_enabled:
            raise TaskContractError(
                "--plan-preserving-source-attempts cannot be combined with "
                "--disable-plan-preserving"
            )
        if (
            args.partition not in {"train", "validation", "test"}
            or args.task_manifest is None
        ):
            raise TaskContractError(
                "--plan-preserving-source-attempts requires a frozen train, validation, "
                "or test partition"
            )
        plan_preserving_path = _resolve_project_path(
            project_root, args.plan_preserving_source_attempts
        ).resolve()
        plan_preserving_attempts = _load_plan_preserving_attempts(
            plan_preserving_path,
            domain=recipe["domain"],
            mode=recipe["mode"],
            selected_task_ids={task.task_id for task in selected},
            attempts_per_task=args.attempts_per_task,
        )
        affected_task_ids = {key[0] for key in plan_preserving_attempts}
        selected = [task for task in selected if task.task_id in affected_task_ids]
    fixed_fault_rows = {}
    fixed_fault_path = os.environ.get("ROBUSTOPS_FIXED_FAULT_POLICY")
    if fixed_fault_path:
        for line in Path(fixed_fault_path).read_text().splitlines():
            row = json.loads(line)
            if row['partition'] != args.partition:
                raise TaskContractError('fault policy partition mismatch')
            key = (row['task_id'], row['mode'])
            if key in fixed_fault_rows:
                raise TaskContractError('duplicate fault policy case')
            fixed_fault_rows[key] = row
        if any((t.task_id, recipe['mode']) not in fixed_fault_rows for t in selected):
            raise TaskContractError('missing fixed fault policy')
        from robustops.evaluation.fixed_fault import FixedPolicyFault
    planned_attempts_by_task = {
        task.task_id: (
            sorted(
                attempt_index
                for task_id, attempt_index in plan_preserving_attempts
                if task_id == task.task_id
            )
            if plan_preserving_path is not None
            else list(range(1, args.attempts_per_task + 1))
        )
        for task in selected
    }
    scheduled_attempts = sum(map(len, planned_attempts_by_task.values()))
    validate_seed_members(selected, archive_path)
    required_server_names = sorted(
        {server["mcp_server_name"] for task in selected for server in task.gym_servers}
    )
    mcp_server_concurrency = _parse_positive_int_mapping(
        args.mcp_server_concurrency_map,
        option="--mcp-server-concurrency-map",
    )
    server_semaphores = {
        server_name: asyncio.Semaphore(limit)
        for server_name, limit in mcp_server_concurrency.items()
        if server_name in required_server_names
    }
    if args.mcp_url and args.mcp_url_map:
        raise TaskContractError("--mcp-url and --mcp-url-map cannot be used together")
    mcp_url_overrides = _parse_string_mapping(
        args.mcp_url_map,
        option="--mcp-url-map",
    )
    if args.mcp_url:
        if len(required_server_names) != 1:
            raise TaskContractError(
                "--mcp-url only supports a single Gym server; use --mcp-url-map"
            )
        mcp_url_overrides = {required_server_names[0]: args.mcp_url}
    if mcp_url_overrides:
        missing_urls = sorted(set(required_server_names) - set(mcp_url_overrides))
        if missing_urls:
            raise TaskContractError(
                f"--mcp-url-map lacks selected Gym servers: {missing_urls}"
            )

    if args.mcp_image_digest and args.mcp_image_digest_map:
        raise TaskContractError(
            "--mcp-image-digest and --mcp-image-digest-map cannot be used together"
        )
    mcp_image_digests = _parse_string_mapping(
        args.mcp_image_digest_map,
        option="--mcp-image-digest-map",
    )
    if args.mcp_image_digest:
        if len(required_server_names) != 1:
            raise TaskContractError(
                "--mcp-image-digest only supports a single Gym server; "
                "use --mcp-image-digest-map"
            )
        mcp_image_digests = {required_server_names[0]: args.mcp_image_digest}
    if mcp_image_digests:
        missing_digests = sorted(set(required_server_names) - set(mcp_image_digests))
        if missing_digests:
            raise TaskContractError(
                f"--mcp-image-digest-map lacks selected Gym servers: {missing_digests}"
            )
    generation = dict(recipe["generation"])
    if args.partition == "train" and "temperature" not in generation:
        raise TaskContractError("train generation.temperature must be explicit")
    generation.setdefault("temperature", 0.0)
    if args.max_context_tokens is not None:
        generation["max_context_tokens"] = args.max_context_tokens
    if generation["max_context_tokens"] != 32_768:
        raise TaskContractError(
            "max context tokens must be 32768; the Qwen3 student context "
            "boundary is the sole inference gate"
        )
    if generation["max_context_tokens"] <= generation["max_tokens"]:
        raise TaskContractError("max context tokens must exceed max generation tokens")
    endpoints = _model_endpoints(args)
    if args.concurrency < 1:
        raise TaskContractError("concurrency must be at least 1")
    base_seed = int(generation["seed"])
    attempt_seeds = [base_seed + index for index in range(args.attempts_per_task)]
    provider_seed_sent = args.model_provider == "local_vllm"

    if args.dry_run:
        prompt_tokens = [
            token_counter.text(task.system_prompt)
            + token_counter.text(task.user_prompt)
            for task in selected
        ]
        print(
            json.dumps(
                {
                    "dry_run": True,
                    "experiment_id": experiment_id,
                    "domain": recipe["domain"],
                    "mode": recipe["mode"],
                    "scope": args.scope,
                    "partition": args.partition,
                    "paired_mode": args.paired_mode,
                    "available_mode": args.available_mode,
                    "task_manifest": str(args.task_manifest.resolve())
                    if args.task_manifest
                    else None,
                    "task_count": len(selected),
                    "attempts_per_task": args.attempts_per_task,
                    "scheduled_attempts": scheduled_attempts,
                    "planned_attempt_indices_by_task": planned_attempts_by_task,
                    "plan_preserving": (
                        {
                            "controller": PLAN_PRESERVING_CONTROLLER,
                            "activation": (
                                "source_branch"
                                if plan_preserving_path is not None
                                else "live_default"
                            ),
                            "source_attempts": (
                                str(plan_preserving_path)
                                if plan_preserving_path is not None
                                else None
                            ),
                            "assistant_content_policy": "preserve_full",
                            "tool_call_policy": "execute_first_only_then_reinfer",
                        }
                        if plan_preserving_enabled
                        else None
                    ),
                    "attempt_seeds": attempt_seeds if provider_seed_sent else None,
                    "provider_seed_sent": provider_seed_sent,
                    "task_ids": [task.task_id for task in selected],
                    "gym_servers": required_server_names,
                    "mcp_url_overrides": mcp_url_overrides,
                    "mcp_image_digests": mcp_image_digests,
                    "mcp_server_concurrency": mcp_server_concurrency,
                    "model_provider": args.model_provider,
                    "tokenizer_path": (
                        str(tokenizer_path) if tokenizer_path is not None else None
                    ),
                    "api_key_env": args.api_key_env,
                    "token_counter": args.token_counter,
                    "model_endpoints": endpoints,
                    "concurrency": args.concurrency,
                    "fault_enabled": False,
                    "max_context_tokens": generation["max_context_tokens"],
                    "reserved_generation_tokens": generation["max_tokens"],
                    "system_user_tokens": {
                        "min": min(prompt_tokens),
                        "median": statistics.median(prompt_tokens),
                        "max": max(prompt_tokens),
                    },
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return 0

    if args.scope == "full" and not mcp_image_digests:
        raise TaskContractError(
            "full scope requires --mcp-image-digest or --mcp-image-digest-map"
        )

    run_id = (
        args.run_id
        or f"{args.scope}-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    )
    _validate_run_id(run_id)
    output_root = (
        args.output_root.resolve()
        if args.output_root
        else project_root / "runs/phase0_clean"
    )
    run_dir = output_root / run_id
    try:
        run_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError as exc:
        raise TaskContractError(f"run directory already exists: {run_dir}") from exc

    model_name = args.model_name or recipe["served_model_name"]
    resolved_config = {
        "experiment_id": experiment_id,
        "run_id": run_id,
        "recipe": str(args.recipe.resolve()),
        "domain": recipe["domain"],
        "mode": recipe["mode"],
        "scope": args.scope,
        "partition": args.partition,
        "paired_mode": args.paired_mode,
        "available_mode": args.available_mode,
        "task_manifest": str(args.task_manifest.resolve())
        if args.task_manifest
        else None,
        "task_ids": [task.task_id for task in selected],
        "dataset_revision": recipe["dataset_revision"],
        "model_revision": args.model_revision or recipe["model_revision"],
        "model_provider": args.model_provider,
        "tokenizer_path": str(tokenizer_path) if tokenizer_path is not None else None,
        "api_key_env": args.api_key_env,
        "token_counter": args.token_counter,
        "model_trust_env": args.model_trust_env,
        "model_name": model_name,
        "model_endpoint": endpoints[0],
        "model_endpoints": endpoints,
        "concurrency": args.concurrency,
        "mcp_url_override": args.mcp_url,
        "mcp_url_overrides": mcp_url_overrides,
        "mcp_image_digest": args.mcp_image_digest,
        "mcp_image_digests": mcp_image_digests,
        "mcp_server_concurrency": mcp_server_concurrency,
        "generation": generation,
        "fault": {"enabled": False},
        "attempts_per_task": args.attempts_per_task,
        "planned_attempt_indices_by_task": planned_attempts_by_task,
        "plan_preserving": (
            {
                "controller": PLAN_PRESERVING_CONTROLLER,
                "activation": (
                    "source_branch"
                    if plan_preserving_path is not None
                    else "live_default"
                ),
                "source_attempts": (
                    str(plan_preserving_path)
                    if plan_preserving_path is not None
                    else None
                ),
                "selection": (
                    "attempt_has_multi_tool_turn"
                    if plan_preserving_path is not None
                    else None
                ),
                "branch": (
                    "first_multi_tool_turn"
                    if plan_preserving_path is not None
                    else None
                ),
                "assistant_content_policy": "preserve_full",
                "tool_call_policy": "execute_first_only_then_reinfer",
            }
            if plan_preserving_enabled
            else None
        ),
        "attempt_seeds": attempt_seeds if provider_seed_sent else None,
        "provider_seed_sent": provider_seed_sent,
        "automatic_retry": {
            "max_retries_per_logical_attempt": 1,
            "mcp_transport_invalid": args.retry_mcp_transport_invalid,
            "model_protocol_error": args.retry_model_protocol_error,
            "superseded_artifact": str(run_dir / "superseded_attempts.jsonl"),
        },
        "artifacts": {
            "trajectories": str(run_dir / "attempts.jsonl"),
            "summary": str(run_dir / "summary.json"),
        },
        "git": _git_state(project_root),
        "runtime": _runtime_metadata(),
        "command": [sys.executable, *sys.argv],
        "created_at": _utc_now(),
    }
    _write_json(run_dir / "resolved_config.json", resolved_config)

    models = [
        [
            OpenAIChatClient(
                endpoint=endpoint,
                model_name=model_name,
                temperature=float(generation["temperature"]),
                max_tokens=int(generation["max_tokens"]),
                seed=attempt_seed if provider_seed_sent else None,
                enable_thinking=bool(generation["enable_thinking"]),
                provider=args.model_provider,
                api_key=api_key,
                trust_env=args.model_trust_env,
            )
            for attempt_seed in attempt_seeds
        ]
        for endpoint in endpoints
    ]
    attempts: list[Mapping[str, Any]] = []
    superseded_attempts: list[Mapping[str, Any]] = []
    attempts_path = run_dir / "attempts.jsonl"
    superseded_path = run_dir / "superseded_attempts.jsonl"
    wall_started = time.monotonic()
    try:
        with (
            attempts_path.open("a", encoding="utf-8") as stream,
            superseded_path.open("a", encoding="utf-8") as superseded_stream,
        ):
            queue: asyncio.Queue[EnterpriseOpsTask] = asyncio.Queue()
            for task in selected:
                queue.put_nowait(task)
            write_lock = asyncio.Lock()

            async def worker(worker_index: int) -> None:
                model_index = worker_index % len(models)
                endpoint = endpoints[model_index]
                while True:
                    try:
                        task = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    # Repeated rollouts for one task stay sequential so a second seed/reset
                    # cannot overwrite the same task database while the first is active.
                    for attempt_index in planned_attempts_by_task[task.task_id]:
                        attempt_seed = attempt_seeds[attempt_index - 1]
                        limited_servers = sorted(
                            {
                                server["mcp_server_name"]
                                for server in task.gym_servers
                                if server["mcp_server_name"] in server_semaphores
                            }
                        )
                        acquired: list[asyncio.Semaphore] = []
                        try:
                            for server_name in limited_servers:
                                semaphore = server_semaphores[server_name]
                                await semaphore.acquire()
                                acquired.append(semaphore)
                            retry_reason: str | None = None
                            attempt: Mapping[str, Any]
                            for retry_index in range(2):
                                attempt = await run_task_attempt(
                                    task=task,
                                    tool_intervention=(FixedPolicyFault(fixed_fault_rows[(task.task_id, recipe['mode'])]) if fixed_fault_path else None),
                                    model=models[model_index][attempt_index - 1],
                                    token_counter=token_counter,
                                    archive_path=archive_path,
                                    mcp_url_overrides=mcp_url_overrides,
                                    max_turns=int(generation["max_turns"]),
                                    max_context_tokens=int(
                                        generation["max_context_tokens"]
                                    ),
                                    max_generation_tokens=int(generation["max_tokens"]),
                                    experiment_id=experiment_id,
                                    run_id=run_id,
                                    mode=recipe["mode"],
                                    model_name=model_name,
                                    model_endpoint=endpoint,
                                    seed=(attempt_seed if provider_seed_sent else None),
                                    attempt_index=attempt_index,
                                    plan_preserving=plan_preserving_enabled,
                                    plan_preserving_source=plan_preserving_attempts.get(
                                        (task.task_id, attempt_index)
                                    ),
                                )
                                attempt = dict(attempt)
                                attempt["automatic_retry"] = {
                                    "retry_index": retry_index,
                                    "triggered_by": retry_reason,
                                }
                                next_reason = (
                                    _mcp_transport_retry_reason(attempt)
                                    if args.retry_mcp_transport_invalid
                                    else None
                                )
                                if (
                                    next_reason is None
                                    and args.retry_model_protocol_error
                                ):
                                    next_reason = _model_protocol_retry_reason(attempt)
                                if retry_index == 1 or next_reason is None:
                                    break
                                superseded = dict(attempt)
                                superseded["superseded_by_retry"] = {
                                    "retry_index": retry_index + 1,
                                    "reason": next_reason,
                                }
                                async with write_lock:
                                    superseded_attempts.append(superseded)
                                    superseded_stream.write(
                                        json.dumps(
                                            superseded,
                                            ensure_ascii=False,
                                            default=str,
                                        )
                                        + "\n"
                                    )
                                    superseded_stream.flush()
                                retry_reason = next_reason
                        finally:
                            for semaphore in reversed(acquired):
                                semaphore.release()
                        async with write_lock:
                            attempts.append(attempt)
                            stream.write(
                                json.dumps(attempt, ensure_ascii=False, default=str)
                                + "\n"
                            )
                            stream.flush()
                            seed_label = (
                                f"seed={attempt_seed}"
                                if provider_seed_sent
                                else "provider_seed=unsupported"
                            )
                            print(
                                f"{task.task_id} attempt {attempt_index}/"
                                f"{args.attempts_per_task}: {attempt['status']} "
                                f"({attempt['duration_seconds']:.3f}s, endpoint={endpoint}, "
                                f"{seed_label})",
                                flush=True,
                            )
                    queue.task_done()

            worker_count = min(args.concurrency, len(selected))
            await asyncio.gather(*(worker(index) for index in range(worker_count)))
    finally:
        await asyncio.gather(
            *(model.close() for endpoint_models in models for model in endpoint_models)
        )

    wall_duration_seconds = time.monotonic() - wall_started

    summary = dict(
        summarize_attempts(
            attempts,
            scheduled=scheduled_attempts,
            wall_duration_seconds=wall_duration_seconds,
        )
    )
    summary["task_count"] = len(selected)
    summary["attempts_per_task"] = args.attempts_per_task
    summary["planned_attempt_indices_by_task"] = planned_attempts_by_task
    summary["plan_preserving"] = resolved_config["plan_preserving"]
    summary["attempt_seeds"] = attempt_seeds if provider_seed_sent else None
    summary["provider_seed_sent"] = provider_seed_sent
    summary["trajectory_artifact"] = str(attempts_path)
    summary["automatic_retry"] = {
        "physical_attempts": len(attempts) + len(superseded_attempts),
        "superseded_attempts": len(superseded_attempts),
        "by_reason": dict(
            sorted(
                Counter(
                    (attempt.get("superseded_by_retry") or {}).get("reason")
                    for attempt in superseded_attempts
                ).items()
            )
        ),
        "superseded_artifact": str(superseded_path),
    }
    _write_json(run_dir / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a configured MidTool EnterpriseOps clean evaluation without fault injection."
    )
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument(
        "--experiment-id",
        help="Optional outer launch experiment ID recorded in resolved config and attempts.",
    )
    parser.add_argument("--scope", choices=("full",), required=True)
    parser.add_argument(
        "--task-manifest",
        type=Path,
        help="Frozen JSONL task manifest; must be used with --partition.",
    )
    parser.add_argument(
        "--partition",
        choices=("train", "validation", "test"),
        help="Select one frozen task partition; must be used with --task-manifest.",
    )
    parser.add_argument(
        "--paired-mode",
        choices=("plus_5_tools", "plus_10_tools", "plus_15_tools"),
        help=(
            "Restrict the partition to task IDs semantically paired with this mode; "
            "requires --task-manifest and --partition."
        ),
    )
    parser.add_argument(
        "--available-mode",
        choices=("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools"),
        help=(
            "Restrict the partition to task IDs whose source row exists for this mode; "
            "requires --task-manifest and --partition."
        ),
    )
    parser.add_argument(
        "--task-ids-file",
        type=Path,
        help=(
            "Optional newline-delimited exact task subset after manifest/partition filtering; "
            "requires --task-manifest and --partition."
        ),
    )
    parser.add_argument(
        "--model-endpoint",
        default=os.environ.get("ROBUSTOPS_MODEL_ENDPOINT", "http://127.0.0.1:8000/v1"),
        help="OpenAI-compatible vLLM /v1 endpoint.",
    )
    parser.add_argument(
        "--model-endpoints",
        default=os.environ.get("ROBUSTOPS_MODEL_ENDPOINTS"),
        help="Comma-separated OpenAI-compatible endpoints; workers are assigned round-robin.",
    )
    parser.add_argument(
        "--model-provider",
        choices=("local_vllm", "deepseek"),
        default=os.environ.get("ROBUSTOPS_MODEL_PROVIDER", "local_vllm"),
        help="Request payload/authentication contract (default: local_vllm).",
    )
    parser.add_argument(
        "--api-key-env",
        default=os.environ.get("ROBUSTOPS_API_KEY_ENV"),
        help="Credential environment-variable name; the secret is never persisted.",
    )
    parser.add_argument(
        "--token-counter",
        choices=("local_hf",),
        default=os.environ.get("ROBUSTOPS_TOKEN_COUNTER", "local_hf"),
        help="Qwen-family local context-bound estimator used before every request.",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=Path,
        default=os.environ.get("ROBUSTOPS_TOKENIZER_PATH"),
        help="Local tokenizer path used by the context gate independently of the served model.",
    )
    parser.add_argument(
        "--model-trust-env",
        action="store_true",
        default=os.environ.get("ROBUSTOPS_MODEL_TRUST_ENV", "0") == "1",
        help="Allow HTTP proxy variables for model requests (disabled by default).",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=int(os.environ.get("ROBUSTOPS_CONCURRENCY", "1")),
        help="Maximum number of tasks executed concurrently (default: 1).",
    )
    parser.add_argument(
        "--attempts-per-task",
        type=int,
        default=1,
        help=(
            "Independent sampled rollouts per selected task. Attempt N uses the domain "
            "recipe seed plus N-1; repeated rollouts of one task execute sequentially."
        ),
    )
    parser.add_argument(
        "--plan-preserving-source-attempts",
        type=Path,
        help=(
            "Controlled trajectory regeneration: select source attempts containing a "
            "multi-tool assistant turn, replay the prefix through the first such turn, "
            "preserve its assistant content, execute only its first call, and re-infer."
        ),
    )
    parser.add_argument(
        "--disable-plan-preserving",
        action="store_true",
        help=(
            "Reproduce legacy batch execution. By default every inference keeps the "
            "full assistant planning text but executes only the first tool call in a "
            "turn, then asks the model to re-infer from the real observation."
        ),
    )
    parser.add_argument(
        "--model-name",
        default=os.environ.get("ROBUSTOPS_MODEL_NAME"),
        help="Served model name; defaults to the absolute local model path.",
    )
    parser.add_argument(
        "--model-revision",
        default=os.environ.get("ROBUSTOPS_MODEL_REVISION"),
        help="Artifact revision recorded in provenance; defaults to the domain recipe revision.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=os.environ.get("ROBUSTOPS_MODEL_PATH"),
        help="Local tokenizer/model path; defaults to the recipe model_path.",
    )
    parser.add_argument(
        "--mcp-url",
        default=os.environ.get(
            "ROBUSTOPS_MCP_URL", os.environ.get("ROBUSTOPS_EMAIL_MCP_URL")
        ),
        help="Optional MCP base URL override; otherwise use the dataset URL.",
    )
    parser.add_argument(
        "--mcp-url-map",
        help=(
            "JSON object mapping Gym server names to MCP base URLs; required for "
            "multi-server tasks when local/external URLs differ from the dataset."
        ),
    )
    parser.add_argument(
        "--mcp-image-digest",
        default=os.environ.get(
            "ROBUSTOPS_MCP_IMAGE_DIGEST",
            os.environ.get("ROBUSTOPS_EMAIL_MCP_IMAGE_DIGEST"),
        ),
        help="Exact MCP container image digest; required for a full run.",
    )
    parser.add_argument(
        "--mcp-image-digest-map",
        help="JSON object mapping Gym server names to exact container image digests.",
    )
    parser.add_argument(
        "--mcp-server-concurrency-map",
        default=os.environ.get("ROBUSTOPS_MCP_SERVER_CONCURRENCY_MAP"),
        help=(
            "Optional JSON object limiting concurrent complete attempts that use each "
            "named MCP server."
        ),
    )
    parser.add_argument(
        "--retry-mcp-transport-invalid",
        action="store_true",
        default=os.environ.get("ROBUSTOPS_RETRY_MCP_TRANSPORT_INVALID", "0") == "1",
        help="Retry one complete logical attempt after an MCP transport invalid.",
    )
    parser.add_argument(
        "--retry-model-protocol-error",
        action="store_true",
        default=os.environ.get("ROBUSTOPS_RETRY_MODEL_PROTOCOL_ERROR", "0") == "1",
        help="Retry one complete logical attempt after invalid model tool-call JSON.",
    )
    parser.add_argument("--run-id")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument(
        "--max-context-tokens",
        type=int,
        help="Effective vLLM model context limit; checked locally before every request.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate local task/seed/token contracts without model, MCP, Docker, or output writes.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    try:
        raise SystemExit(asyncio.run(_run(args)))
    except (TaskContractError, InfrastructureError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
