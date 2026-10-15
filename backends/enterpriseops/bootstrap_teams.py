#!/usr/bin/env python3
"""Bootstrap the pinned Teams MCP with its corrected public contract."""

from __future__ import annotations

import inspect
import os
import re
from copy import deepcopy
from functools import wraps
from typing import Any

RESTORED_TOOL_NAME = "remove_channel_member"
RESTORED_REQUIRED_ARGUMENTS = {"teamId", "channelId", "membershipId"}
RESTORED_ENDPOINT = (
    "DELETE /teams/{team-id}/channels/{channel-id}/members/{membership-id}"
)
ALIASED_TOOL_NAME = "list_team_apps"
ALIASED_SOURCE_NAME = "list_teams_apps"
ALIASED_REQUIRED_ARGUMENTS: set[str] = set()
ALIASED_ENDPOINT = "GET /appCatalogs/teamsApps"
CHANNEL_MESSAGE_TOOL_NAME = "send_channel_message"
CHANNEL_MESSAGE_ENDPOINT = "POST /teams/{team-id}/channels/{channel-id}/messages"
CHANNEL_MESSAGE_OPTIONAL_DESCRIPTION = (
    "- Only the body property is mandatory; other properties are optional"
)
CHANNEL_MESSAGE_REQUIRED_DESCRIPTION = (
    "- The body property is mandatory and must explicitly contain both "
    "contentType and content; other top-level properties are optional"
)
CHANNEL_MESSAGE_BODY_DESCRIPTION = (
    "The message body content (MANDATORY property - only required field per "
    "Microsoft Graph API)"
)
CHANNEL_MESSAGE_BODY_REQUIRED_DESCRIPTION = (
    "The mandatory message body; this pinned backend requires explicit "
    "contentType and content fields"
)
TOWNHALL_CREATE_ARGUMENTS = (
    "self",
    "org_id",
    "creator_user_id",
    "request_data",
)
CHANNEL_FILTER_ARGUMENTS = ("self", "filter_expr")
_STARTSWITH_FILTER = re.compile(
    r"startswith\(\s*(displayName|description)\s*,\s*'([^']+)'\s*\)",
    re.IGNORECASE,
)


def patch_list_channels_filter_contract(
    manager_class: Any,
    channel_model: Any,
) -> None:
    """Fix the pinned startswith helper and reject unsupported filters."""

    original = manager_class._build_channel_filter_condition
    if getattr(original, "__robustops_channel_filter_compat__", False):
        return
    if tuple(inspect.signature(original).parameters) != CHANNEL_FILTER_ARGUMENTS:
        raise RuntimeError("pinned Teams channel filter signature no longer matches")

    @wraps(original)
    def build_channel_filter_condition(self: Any, filter_expr: str) -> Any:
        expression = filter_expr.strip()
        startswith_match = _STARTSWITH_FILTER.fullmatch(expression)
        if startswith_match:
            field_name, value = startswith_match.groups()
            column = (
                channel_model.display_name
                if field_name.lower() == "displayname"
                else channel_model.description
            )
            return column.ilike(f"{value}%")

        condition = original(self, expression)
        if condition is None:
            raise ValueError(f"Unsupported OData filter expression: {filter_expr}")
        return condition

    build_channel_filter_condition.__robustops_channel_filter_compat__ = True
    manager_class._build_channel_filter_condition = build_channel_filter_condition
    print(
        "Teams MCP channel-filter compatibility applied: repair startswith and "
        "reject unsupported expressions",
        flush=True,
    )


def _validate_tool(
    tool: Any,
    *,
    tool_name: str,
    required_arguments: set[str],
    endpoint: str,
) -> None:
    schema = tool.get("inputSchema") if isinstance(tool, dict) else None
    required = set(schema.get("required", ())) if isinstance(schema, dict) else set()
    description = tool.get("description") if isinstance(tool, dict) else None
    if required != required_arguments or not isinstance(description, str):
        raise RuntimeError(f"pinned Teams {tool_name!r} schema no longer matches")
    if endpoint not in description:
        raise RuntimeError(f"pinned Teams {tool_name!r} endpoint no longer matches")


def apply_dataset_catalog_compatibility(teams_mcp: Any) -> None:
    """Restore exact image functionality declared by the fixed official dataset."""

    all_tools = teams_mcp.ALL_TOOLS
    exposed_tools = teams_mcp.MCP_TOOLS
    approved_tools = teams_mcp.PRODUCTION_APPROVED_TOOLS
    matches = [tool for tool in all_tools if tool.get("name") == RESTORED_TOOL_NAME]
    if len(matches) != 1:
        raise RuntimeError(
            f"pinned Teams image must contain exactly one {RESTORED_TOOL_NAME!r} "
            "built-in tool"
        )
    if any(tool.get("name") == RESTORED_TOOL_NAME for tool in exposed_tools):
        raise RuntimeError(
            f"pinned Teams image unexpectedly already exposes {RESTORED_TOOL_NAME!r}"
        )
    if RESTORED_TOOL_NAME in approved_tools:
        raise RuntimeError(
            f"pinned Teams image unexpectedly already approves {RESTORED_TOOL_NAME!r}"
        )

    tool = matches[0]
    _validate_tool(
        tool,
        tool_name=RESTORED_TOOL_NAME,
        required_arguments=RESTORED_REQUIRED_ARGUMENTS,
        endpoint=RESTORED_ENDPOINT,
    )
    approved_tools.add(RESTORED_TOOL_NAME)
    exposed_tools.append(tool)

    alias_sources = [
        tool for tool in exposed_tools if tool.get("name") == ALIASED_SOURCE_NAME
    ]
    if len(alias_sources) != 1:
        raise RuntimeError(
            f"pinned Teams image must expose exactly one {ALIASED_SOURCE_NAME!r} tool"
        )
    if any(tool.get("name") == ALIASED_TOOL_NAME for tool in all_tools):
        raise RuntimeError(
            f"pinned Teams image unexpectedly contains {ALIASED_TOOL_NAME!r}"
        )
    if ALIASED_TOOL_NAME in approved_tools:
        raise RuntimeError(
            f"pinned Teams image unexpectedly already approves {ALIASED_TOOL_NAME!r}"
        )
    alias_source = alias_sources[0]
    _validate_tool(
        alias_source,
        tool_name=ALIASED_SOURCE_NAME,
        required_arguments=ALIASED_REQUIRED_ARGUMENTS,
        endpoint=ALIASED_ENDPOINT,
    )
    alias = deepcopy(alias_source)
    alias["name"] = ALIASED_TOOL_NAME
    approved_tools.add(ALIASED_TOOL_NAME)
    exposed_tools.append(alias)
    print(
        "Teams MCP catalog compatibility applied: restored built-in "
        f"{RESTORED_TOOL_NAME} and aliased {ALIASED_TOOL_NAME} to "
        f"{ALIASED_SOURCE_NAME}",
        flush=True,
    )


def register_handler_alias(mcp_handler: Any) -> None:
    handlers = mcp_handler.TOOL_HANDLERS
    if ALIASED_SOURCE_NAME not in handlers:
        raise RuntimeError(
            f"pinned Teams image no longer handles {ALIASED_SOURCE_NAME!r}"
        )
    source_handler = handlers[ALIASED_SOURCE_NAME]
    alias_handler = handlers.get(ALIASED_TOOL_NAME)
    if alias_handler is not None and alias_handler is not source_handler:
        raise RuntimeError(
            f"pinned Teams image registered an unexpected {ALIASED_TOOL_NAME!r} handler"
        )

    async def execute_alias(
        tool_name: str,
        arguments: dict[str, Any],
        database_id: str,
        user_id: str,
        access_token: str | None = None,
    ) -> Any:
        if tool_name != ALIASED_TOOL_NAME:
            raise RuntimeError(f"unexpected aliased tool name {tool_name!r}")
        return await source_handler(
            ALIASED_SOURCE_NAME,
            arguments,
            database_id,
            user_id,
            access_token,
        )

    handlers[ALIASED_TOOL_NAME] = execute_alias


def patch_send_channel_message_schema(teams_mcp: Any) -> None:
    """Expose the pinned backend's real nested body requirements to agents."""

    all_matches = [
        tool
        for tool in teams_mcp.ALL_TOOLS
        if tool.get("name") == CHANNEL_MESSAGE_TOOL_NAME
    ]
    exposed_matches = [
        tool
        for tool in teams_mcp.MCP_TOOLS
        if tool.get("name") == CHANNEL_MESSAGE_TOOL_NAME
    ]
    if len(all_matches) != 1 or len(exposed_matches) != 1:
        raise RuntimeError(
            "pinned Teams image must contain and expose exactly one "
            f"{CHANNEL_MESSAGE_TOOL_NAME!r} tool"
        )
    tool = all_matches[0]
    if exposed_matches[0] is not tool:
        raise RuntimeError(
            f"pinned Teams {CHANNEL_MESSAGE_TOOL_NAME!r} catalog identity changed"
        )
    description = tool.get("description")
    schema = tool.get("inputSchema")
    properties = schema.get("properties") if isinstance(schema, dict) else None
    body = properties.get("body") if isinstance(properties, dict) else None
    body_properties = body.get("properties") if isinstance(body, dict) else None
    content_type = (
        body_properties.get("contentType")
        if isinstance(body_properties, dict)
        else None
    )
    if (
        not isinstance(description, str)
        or CHANNEL_MESSAGE_ENDPOINT not in description
        or description.count(CHANNEL_MESSAGE_OPTIONAL_DESCRIPTION) != 1
        or not isinstance(body, dict)
        or body.get("description") != CHANNEL_MESSAGE_BODY_DESCRIPTION
        or body.get("required") != ["content"]
        or not isinstance(content_type, dict)
        or content_type.get("default") != "text"
        or content_type.get("enum") != ["text", "html"]
    ):
        raise RuntimeError(
            f"pinned Teams {CHANNEL_MESSAGE_TOOL_NAME!r} schema no longer matches"
        )

    body["required"] = ["contentType", "content"]
    body["description"] = CHANNEL_MESSAGE_BODY_REQUIRED_DESCRIPTION
    content_type.pop("default")
    tool["description"] = description.replace(
        CHANNEL_MESSAGE_OPTIONAL_DESCRIPTION,
        CHANNEL_MESSAGE_REQUIRED_DESCRIPTION,
    )
    print(
        "Teams MCP channel-message schema compatibility applied: require explicit "
        "body.contentType and body.content",
        flush=True,
    )


def patch_virtual_event_townhall_return(
    manager_class: Any,
    *,
    townhall_model: Any | None = None,
    coorganizer_model: Any | None = None,
    invited_attendee_model: Any | None = None,
) -> None:
    """Return the row committed by the pinned image's create implementation."""

    original = manager_class.create_townhall
    if getattr(original, "__robustops_townhall_return_compat__", False):
        return
    if tuple(inspect.signature(original).parameters) != TOWNHALL_CREATE_ARGUMENTS:
        raise RuntimeError("pinned Teams townhall create signature no longer matches")
    model = (
        townhall_model
        if townhall_model is not None
        else original.__globals__.get("VirtualEventTownhall")
    )
    if model is None:
        raise RuntimeError("pinned Teams townhall model is unavailable")
    coorganizer = (
        coorganizer_model
        if coorganizer_model is not None
        else original.__globals__.get("VirtualEventTownhallCoOrganizer")
    )
    invited_attendee = (
        invited_attendee_model
        if invited_attendee_model is not None
        else original.__globals__.get("VirtualEventTownhallInvitedAttendee")
    )
    if coorganizer is None or invited_attendee is None:
        raise RuntimeError("pinned Teams townhall relation models are unavailable")

    @wraps(original)
    def create_townhall(
        self: Any,
        org_id: str,
        creator_user_id: str,
        request_data: Any,
    ) -> Any:
        before_session = self.session_manager.get_session(self.database_id)
        try:
            before_ids = {
                row.id
                for row in before_session.query(model)
                .filter(
                    model.org_id == org_id,
                    model.created_by_user_id == creator_user_id,
                )
                .all()
            }
        finally:
            before_session.close()

        result = original(self, org_id, creator_user_id, request_data)
        if result is not None:
            return result

        after_session = self.session_manager.get_session(self.database_id)
        try:
            created = [
                row
                for row in after_session.query(model)
                .filter(
                    model.org_id == org_id,
                    model.created_by_user_id == creator_user_id,
                )
                .all()
                if row.id not in before_ids
            ]
            if len(created) != 1:
                raise RuntimeError(
                    "pinned Teams townhall create did not commit exactly one new row"
                )
            created_townhall = created[0]
            coorganizers = (
                after_session.query(coorganizer)
                .filter(
                    coorganizer.org_id == org_id,
                    coorganizer.townhall_id == created_townhall.id,
                )
                .all()
            )
            invited_attendees = (
                after_session.query(invited_attendee)
                .filter(
                    invited_attendee.org_id == org_id,
                    invited_attendee.townhall_id == created_townhall.id,
                )
                .all()
            )
            return self._townhall_to_dict(
                after_session,
                created_townhall,
                coorganizers,
                invited_attendees,
            )
        finally:
            after_session.close()

    create_townhall.__robustops_townhall_return_compat__ = True
    manager_class.create_townhall = create_townhall
    print(
        "Teams MCP townhall compatibility applied: return the single row committed "
        "by create_townhall",
        flush=True,
    )


def main() -> None:
    import teams_mcp

    apply_dataset_catalog_compatibility(teams_mcp)
    patch_send_channel_message_schema(teams_mcp)

    from database.managers.channel_manager import ChannelManager
    from database.managers.virtual_event_townhall_manager import (
        VirtualEventTownhallManager,
    )
    from database.models.channel import Channel

    patch_list_channels_filter_contract(ChannelManager, Channel)
    patch_virtual_event_townhall_return(VirtualEventTownhallManager)

    import main as teams_main
    from handlers import mcp_handler

    register_handler_alias(mcp_handler)

    import uvicorn

    uvicorn.run(
        teams_main.app,
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", "8005")),
    )


if __name__ == "__main__":
    main()
