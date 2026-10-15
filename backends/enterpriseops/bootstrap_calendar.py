#!/usr/bin/env python3
"""Bootstrap the pinned Calendar MCP with stable errors and ACL invariants."""

from __future__ import annotations

import inspect
import json
import os
import re
from collections.abc import Mapping
from contextvars import ContextVar
from datetime import datetime, timezone
from functools import wraps
from typing import Any

_INTERNAL_NOT_FOUND = re.compile(
    r"^(?:❌\s*)?Internal server error:\s*(?P<details>.+\bnot found\b.*)$",
    re.IGNORECASE,
)
_FINAL_OWNER_DELETE = "Cannot delete the final owner ACL rule"
_IMMOVABLE_EVENT = re.compile(
    r"^Cannot move event of type '[^']+'\. Only default events can be moved\. "
    r"Events of type 'birthday', 'focusTime', 'fromGmail', 'outOfOffice', "
    r"and 'workingLocation' cannot be moved\.$"
)
_EVENT_RESPONSE_VALIDATION = re.compile(
    r"^(?:❌\s*)?(?:Internal server error:\s*)?"
    r"(?P<details>\d+ validation errors? for (?:Event|EventSchema)\b.*)$",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_HANDLER_RESULT: ContextVar[Mapping[str, Any] | None] = ContextVar(
    "calendar_tool_handler_result",
    default=None,
)


def _event_time_kind(value: Any) -> str:
    from fastapi import HTTPException

    has_date = getattr(value, "date", None) is not None
    has_date_time = getattr(value, "dateTime", None) is not None
    if has_date == has_date_time:
        raise HTTPException(
            status_code=400,
            detail="Event time must specify exactly one of date or dateTime",
        )
    return "date" if has_date else "dateTime"


def _date_time_has_offset(value: Any) -> bool | None:
    raw = getattr(value, "dateTime", None)
    if not isinstance(raw, str):
        return None
    normalized = f"{raw[:-1]}+00:00" if raw.endswith(("Z", "z")) else raw
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    return parsed.utcoffset() is not None


def validate_event_time_pair(start: Any, end: Any) -> None:
    """Reject time pairs that the pinned manager cannot compare safely."""

    if start is None or end is None:
        return
    start_kind = _event_time_kind(start)
    end_kind = _event_time_kind(end)
    if start_kind != end_kind:
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail="Event start and end must both use date or both use dateTime",
        )
    if start_kind != "dateTime":
        return
    start_has_offset = _date_time_has_offset(start)
    end_has_offset = _date_time_has_offset(end)
    if (
        start_has_offset is not None
        and end_has_offset is not None
        and start_has_offset != end_has_offset
    ):
        from fastapi import HTTPException

        raise HTTPException(
            status_code=400,
            detail=(
                "Event start and end dateTime values must both include timezone "
                "offsets or both omit them"
            ),
        )


def patch_event_time_validation(manager_class: type[Any]) -> None:
    """Turn the pinned manager's naive/aware comparison crash into HTTP 400."""

    original = manager_class.update_event
    if tuple(inspect.signature(original).parameters) != (
        "self",
        "user_id",
        "calendar_id",
        "event_id",
        "event_request",
        "is_patch",
        "query_params",
    ):
        raise RuntimeError("pinned Calendar update_event signature drifted")

    @wraps(original)
    def update_event(
        self: Any,
        user_id: str,
        calendar_id: str,
        event_id: str,
        event_request: Any,
        is_patch: bool = True,
        query_params: Mapping[str, Any] | None = None,
    ) -> Any:
        validate_event_time_pair(
            getattr(event_request, "start", None),
            getattr(event_request, "end", None),
        )
        try:
            return original(
                self,
                user_id,
                calendar_id,
                event_id,
                event_request,
                is_patch,
                query_params,
            )
        except TypeError as exc:
            if "can't compare offset-naive and offset-aware datetimes" not in str(exc):
                raise
            # The original manager has already rolled back its transaction in
            # this path.  Translate only this deterministic request error.
            from fastapi import HTTPException

            raise HTTPException(
                status_code=400,
                detail="Event start and end must use compatible time zone information",
            ) from exc

    manager_class.update_event = update_event


def patch_optional_event_timezone(manager_class: type[Any]) -> None:
    """Accept the same optional timezone in dict and schema-object requests."""
    original = manager_class._parse_datetime_from_api_format
    if tuple(inspect.signature(original).parameters) != ('self', 'datetime_obj'):
        raise RuntimeError('pinned Calendar datetime parser signature drifted')

    @wraps(original)
    def parse(self, datetime_obj):
        if isinstance(datetime_obj, dict):
            value = dict(datetime_obj)
            value.setdefault('timeZone', None)
            zone = value['timeZone']
        else:
            value = datetime_obj
            zone = value.timeZone
        parsed, stored_zone = original(self, value)
        if not zone and parsed.utcoffset() is not None:
            # SQLite drops tzinfo. Match the native UTC fallback only after
            # converting the instant, so 09:00-04:00 becomes 13:00 UTC.
            parsed = parsed.astimezone(timezone.utc)
        return parsed, stored_zone

    manager_class._parse_datetime_from_api_format = parse


def patch_materialized_event_listing(manager_class: type[Any], recurring_model: type[Any]) -> None:
    """Bridge a stale lazy-expansion hook to the pinned eager storage model.

    create_event already writes up to 365 concrete instances. RecurringEvent
    only stores an id and original rule, not a template capable of generating
    more events. Reads must use the stored instances, including their edits and
    cancellations, rather than recreating them from an arbitrary sibling.
    """
    if hasattr(manager_class, 'expand_recurring_events') or set(recurring_model.__table__.columns.keys()) != {
        'recurring_event_id', 'original_recurrence'
    }:
        raise RuntimeError('pinned Calendar materialized recurrence contract drifted')

    def expand_recurring_events(self, user_id, calendar_id, start_date, end_date):
        # Native list_events performs permission checks before this hook and
        # applies the time window to the materialized rows immediately after it.
        return None

    manager_class.expand_recurring_events = expand_recurring_events


def recurring_masters(rows, known_roots=None):
    """Identify roots in the pinned materialized layout, not by title/time alone.

    Native create writes the root first, then a duplicate first occurrence with
    the same series/original-start key. Seed series without that root are already
    concrete instances. Ambiguous duplicate roots must not be silently dropped.
    """
    from collections import defaultdict
    grouped = defaultdict(list)
    for row in rows:
        if row.recurring_event_id and row.recurring_event_id != 'NULL' and row.recurrence:
            grouped[row.recurring_event_id].append(row)
    roots = {}
    for series, members in grouped.items():
        known = (known_roots or {}).get(series)
        if known is not None:
            match = next((r for r in members if r.event_id == known), None)
            if match is not None:
                roots[series] = match
            continue
        explicit = [r for r in members if r.event_id == series or
                    getattr(r, 'iCalUID', None) == r.event_id + '@calendar.google.com' or
                    r.originalStartTime_dateTime is None and r.originalStartTime_date is None]
        if len(explicit) == 1:
            roots[series] = explicit[0]
            continue
        slots = defaultdict(list)
        for row in members:
            slots[(row.originalStartTime_dateTime, row.originalStartTime_date)].append(row)
        duplicates = [group for group in slots.values() if len(group) > 1]
        if not duplicates:
            continue
        # Native root creation strictly precedes its generated children. Require
        # both the duplicate first slot and a unique earliest creation timestamp.
        earliest = min((r.created_at for r in members if r.created_at is not None), default=None)
        candidates = [r for r in members if r.created_at == earliest]
        if (len(explicit) == 0 and earliest is not None and len(candidates) == 1
                and len(duplicates) == 1 and candidates[0] in duplicates[0]):
            roots[series] = candidates[0]
        else:
            raise RuntimeError(f'Ambiguous recurring event root for series {series}')
    return roots


def recurring_exception(item, root):
    """Compare the occurrence's visible content with its series template."""
    def time_value(value):
        from zoneinfo import ZoneInfo
        raw = value.dateTime or value.date
        parsed = datetime.fromisoformat(str(raw).replace('Z', '+00:00'))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=ZoneInfo(value.timeZone) if value.timeZone else timezone.utc)
        return parsed.astimezone(timezone.utc)

    if item.status == 'cancelled':
        return True
    if item.originalStartTime and time_value(item.start) != time_value(item.originalStartTime):
        return True
    if time_value(item.end) - time_value(item.start) != time_value(root.end) - time_value(root.start):
        return True
    ignored = {'id', 'etag', 'htmlLink', 'created', 'updated', 'start', 'end',
               'originalStartTime', 'recurringEventId', 'recurrence'}
    return item.model_dump(exclude=ignored) != root.model_dump(exclude=ignored)


def patch_recurring_queries(module: Any) -> None:
    """Filter materialized parents/instances before applying public pagination."""
    from fastapi import HTTPException
    from sqlalchemy import text
    manager_class = module.EventManager

    original_create = manager_class.create_event
    create_signature = inspect.signature(original_create)

    @wraps(original_create)
    def create_event(self, *args, **kwargs):
        result = original_create(self, *args, **kwargs)
        bound = create_signature.bind(self, *args, **kwargs)
        request = bound.arguments['event_request']
        if request.recurrence:
            # Persist the actual returned root id, including custom iCalUIDs.
            # This identity survives deletion/rescheduling of the first child
            # and is carried by the ordinary SQLite snapshots and cleanup.
            session = module.get_session(self.database_id)
            try:
                row = session.query(module.Event).filter(module.Event.event_id == result.id).one()
                session.execute(text('CREATE TABLE IF NOT EXISTS robustops_recurring_roots '
                                     '(series_id TEXT PRIMARY KEY, event_id TEXT NOT NULL UNIQUE)'))
                session.execute(text('INSERT INTO robustops_recurring_roots VALUES (:series, :event)'),
                                {'series': row.recurring_event_id, 'event': row.event_id})
                session.commit()
            finally:
                session.close()
        return result

    manager_class.create_event = create_event

    def inventory(self, calendar_id):
        session = module.get_session(self.database_id)
        try:
            rows = session.query(module.Event).filter(module.Event.calendar_id == calendar_id).all()
            has_roots = session.execute(text("SELECT 1 FROM sqlite_master WHERE type='table' AND name='robustops_recurring_roots'")).scalar()
            known = dict(session.execute(text('SELECT series_id, event_id FROM robustops_recurring_roots')).all()) if has_roots else {}
            roots = recurring_masters(rows, known)
            by_id = {r.event_id: r.recurring_event_id for r in rows}
            schemas = {s: self._convert_db_event_to_schema(r) for s, r in roots.items()}
            calendar = session.query(module.Calendar).filter(module.Calendar.calendar_id == calendar_id).first()
            return by_id, schemas, calendar
        finally:
            session.close()

    def page(self, response, items, arguments):
        limit = arguments.get('max_results')
        limit = 250 if limit is None else min(limit, 2500)
        if limit < 1:
            raise HTTPException(400, 'maxResults must be positive')
        token = arguments.get('page_token')
        try:
            offset = int(token) if token else 0
            if offset < 0:
                raise ValueError('negative offset')
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, 'Invalid page token') from exc
        selected = items[offset:offset + limit]
        maximum = arguments.get('max_attendees')
        if maximum is not None:
            for item in selected:
                if item.attendees and len(item.attendees) > maximum:
                    item.attendees = item.attendees[:maximum]
                    item.attendeesOmitted = True
        response.items = selected
        # The pinned HTTP route requires numeric tokens, despite the manager's
        # unused encoder producing base64. Return tokens the route can consume.
        response.nextPageToken = str(offset + limit) if offset + limit < len(items) else None
        return response

    def wrap(original, instances):
        signature = inspect.signature(original)
        required = {'self', 'calendar_id', 'user_id', 'max_results', 'page_token', 'max_attendees'}
        if not required <= set(signature.parameters) or (not instances and 'single_events' not in signature.parameters):
            raise RuntimeError('pinned Calendar event query signature drifted')

        @wraps(original)
        def query(self, *args, **kwargs):
            bound = signature.bind(self, *args, **kwargs)
            bound.apply_defaults()
            arguments = dict(bound.arguments)
            by_id, roots, calendar = inventory(self, arguments['calendar_id'])
            expanded = instances or bool(arguments.get('single_events'))
            if calendar is not None and self._get_user_calendar_role(arguments['user_id'], calendar) == 'freeBusyReader':
                expanded = True
            if instances:
                requested = arguments['event_id']
                # The public create response exposes the root event id, while
                # this image's native instances query expects an internal id.
                if requested in by_id and by_id[requested] in roots and roots[by_id[requested]].id == requested:
                    bound.arguments['event_id'] = by_id[requested]
            bound.arguments.update(max_results=None, page_token=None, max_attendees=None)
            if not instances:
                bound.arguments['show_deleted'] = False
            response = original(*bound.args, **bound.kwargs)
            if not instances and (not expanded or arguments.get('show_deleted')):
                # This image interprets show_deleted=True as cancelled-only,
                # rather than include-cancelled. Merge both disjoint sets before
                # root/exception filtering and pagination.
                bound.arguments['show_deleted'] = True
                response.items.extend(original(*bound.args, **bound.kwargs).items)
            if arguments.get('order_by') == 'updated':
                response.items.sort(key=lambda item: (str(item.updated), item.id), reverse=True)
            else:
                response.items.sort(key=lambda item: (str(item.start.dateTime or item.start.date), item.id))
            selected = []
            for item in response.items:
                root = roots.get(by_id.get(item.id))
                is_root = root is not None and item.id == root.id
                if expanded and is_root:
                    continue
                if not expanded:
                    if root is not None and not is_root:
                        if not recurring_exception(item, root):
                            continue
                    elif item.status == 'cancelled' and not arguments.get('show_deleted'):
                        continue
                if root is not None:
                    if is_root:
                        item.recurringEventId = None
                        item.originalStartTime = None
                    else:
                        item.recurringEventId = root.id
                        item.recurrence = None
                selected.append(item)
            return page(self, response, selected, arguments)

        return query

    manager_class.list_events = wrap(manager_class.list_events, False)
    manager_class.get_event_instances = wrap(manager_class.get_event_instances, True)


def patch_tool_handler_result_capture(module: Any) -> None:
    """Capture the status that the pinned MCP response builder drops."""

    handlers = getattr(module, "TOOL_HANDLERS", None)
    if not isinstance(handlers, dict) or not handlers:
        raise RuntimeError("pinned Calendar TOOL_HANDLERS registry is unavailable")
    expected = ("tool_name", "arguments", "database_id", "user_id")
    for name, original in list(handlers.items()):
        if tuple(inspect.signature(original).parameters) != expected:
            raise RuntimeError(
                f"pinned Calendar tool handler signature drifted for {name}"
            )

        @wraps(original)
        async def execute_tool(
            tool_name: str,
            arguments: dict[str, Any],
            database_id: str,
            user_id: str,
            _original: Any = original,
        ) -> Any:
            tool_result = await _original(
                tool_name,
                arguments,
                database_id,
                user_id,
            )
            if isinstance(tool_result, Mapping):
                _TOOL_HANDLER_RESULT.set(tool_result)
            return tool_result

        handlers[name] = execute_tool


def _restore_captured_error_contract(
    result: dict[str, Any], tool_result: Mapping[str, Any] | None
) -> None:
    if not isinstance(tool_result, Mapping):
        return
    raw_is_error = tool_result.get("isError") is True
    raw_status = tool_result.get("status_code")
    status_code = (
        raw_status
        if isinstance(raw_status, int) and not isinstance(raw_status, bool)
        else None
    )
    if not raw_is_error and not (
        status_code is not None and not 200 <= status_code <= 299
    ):
        return
    details = tool_result.get("text")
    structured: dict[str, Any] = {
        "success": False,
        "error": {"message": str(details or "Calendar tool request failed")},
    }
    if status_code is not None:
        structured["status_code"] = status_code
        structured["error"]["code"] = status_code
    result["isError"] = True
    result["structuredContent"] = structured


def normalize_focus_time_properties(value: Any) -> Any:
    """Normalize only the two legacy shapes emitted by the pinned seed/backend."""

    if not isinstance(value, Mapping):
        return value
    normalized = dict(value)
    chat_status = normalized.get("chatStatus")
    if isinstance(chat_status, Mapping):
        if set(chat_status) == {"type"} and chat_status.get("type") in {
            "available",
            "doNotDisturb",
        }:
            normalized["chatStatus"] = chat_status["type"]
    elif chat_status is None or chat_status == "":
        normalized["chatStatus"] = None

    # Both fields are Optional in the public schema, but the pinned response
    # model uses the invalid default "".  Missing internal values are null;
    # explicit invalid request values still fail request validation upstream.
    auto_decline_mode = normalized.get("autoDeclineMode")
    if auto_decline_mode is None or auto_decline_mode == "":
        normalized["autoDeclineMode"] = None
    return normalized


def patch_nullable_conference_key(manager_class: type[Any]) -> None:
    """Make the dict path match native schema-object handling of optional keys."""
    original = manager_class._process_create_conference_data
    if tuple(inspect.signature(original).parameters) != ('self', 'db_event', 'conference_data', 'session'):
        raise RuntimeError('pinned Calendar conference handler signature drifted')

    @wraps(original)
    def process(self, db_event, conference_data, session):
        if isinstance(conference_data, dict):
            create = conference_data.get('createRequest')
            if isinstance(create, dict) and 'conferenceSolutionKey' in create and create['conferenceSolutionKey'] is None:
                # Native dict.get(default) does not handle an explicit None.
                # Omit this internal nullable member; never invent a solution.
                create = dict(create)
                del create['conferenceSolutionKey']
                conference_data = {**conference_data, 'createRequest': create}
        return original(self, db_event, conference_data, session)
    manager_class._process_create_conference_data = process


def patch_acl_request_sessions(module: Any) -> None:
    """Release the sessions owned by each native ACL route, also on rejection."""
    original_factory = module.get_acl_manager_instance
    if tuple(inspect.signature(original_factory).parameters) != ('database_id', 'user_id'):
        raise RuntimeError('pinned Calendar ACL manager factory differs')
    owned: ContextVar[Any] = ContextVar('calendar_acl_request_sessions', default=None)

    @wraps(original_factory)
    def factory(*args, **kwargs):
        manager = original_factory(*args, **kwargs)
        sessions = owned.get()
        if sessions is not None:
            sessions.append(manager.db)
        return manager
    module.get_acl_manager_instance = factory

    def release(sessions, token):
        try:
            for session in sessions:
                # Session.close rolls back pending writes and releases locks.
                # It cannot undo commits already made by the native manager.
                session.close()
        finally:
            owned.reset(token)

    # FastAPI serializes ORM responses after the endpoint returns. Own the
    # session around the complete route handler, including that serialization.
    from fastapi.routing import APIRoute
    from starlette.routing import request_response

    class ACLSessionRoute(APIRoute):
        def get_route_handler(self):
            original_handler = super().get_route_handler()

            async def handle(request):
                sessions = []
                token = owned.set(sessions)
                try:
                    return await original_handler(request)
                finally:
                    release(sessions, token)
            return handle

    module.router.route_class = ACLSessionRoute
    for route in module.router.routes:
        if isinstance(route, APIRoute):
            # include_router preserves type(route) in its route_class_override.
            route.__class__ = ACLSessionRoute
            route.app = request_response(route.get_route_handler())


def patch_event_response_conversion(manager_class: type[Any]) -> None:
    original = manager_class._convert_db_event_to_schema
    if tuple(inspect.signature(original).parameters) != ("self", "db_event"):
        raise RuntimeError(
            "pinned Calendar _convert_db_event_to_schema signature drifted"
        )

    class EventView:
        def __init__(self, event: Any, focus_time_properties: Any, out_of_office_properties: Any):
            self._event = event
            self.focusTimeProperties = focus_time_properties
            self.outOfOfficeProperties = out_of_office_properties

        def __getattr__(self, name: str) -> Any:
            return getattr(self._event, name)

    @wraps(original)
    def convert(self: Any, db_event: Any) -> Any:
        current = getattr(db_event, "focusTimeProperties", None)
        normalized = normalize_focus_time_properties(current)
        office = getattr(db_event, "outOfOfficeProperties", None)
        office_normalized = office
        # OutOfOfficeProperties has the same invalid Optional enum default as
        # FocusTimeProperties.  A valid create with only declineMessage commits
        # the row, then fails response validation on the internal empty default.
        # Preserve the missing value as null; do not choose a decline policy.
        if isinstance(office, Mapping) and office.get("autoDeclineMode") in (None, ""):
            office_normalized = {**office, "autoDeclineMode": None}
        if normalized is current and office_normalized is office:
            return original(self, db_event)
        return original(self, EventView(db_event, normalized, office_normalized))

    manager_class._convert_db_event_to_schema = convert


def patch_not_found_results(module: Any) -> None:
    original = module.handle_tools_call
    if tuple(inspect.signature(original).parameters) != ("request", "fastapi_request"):
        raise RuntimeError("pinned Calendar handle_tools_call signature drifted")

    @wraps(original)
    async def handle_tools_call(request: Any, fastapi_request: Any) -> Any:
        token = _TOOL_HANDLER_RESULT.set(None)
        try:
            response = await original(request, fastapi_request)
            tool_result = _TOOL_HANDLER_RESULT.get()
        finally:
            _TOOL_HANDLER_RESULT.reset(token)
        result = getattr(response, "result", None)
        if not isinstance(result, dict):
            return response
        _restore_captured_error_contract(result, tool_result)
        if isinstance(tool_result, Mapping) and isinstance(tool_result.get('status_code'), int) and tool_result['status_code'] >= 400:
            # Typed HTTP status takes precedence over words in the message.
            return response
        for item in result.get("content", []):
            if not isinstance(item, dict) or not isinstance(item.get("text"), str):
                continue
            stripped = item["text"].strip()
            if stripped == _FINAL_OWNER_DELETE:
                # The pinned MCP handler flattens the manager's HTTP 400 into a
                # plain text result.  Preserve that exact model-visible text,
                # but restore the MCP application-error bit.
                result["isError"] = True
                break
            if _IMMOVABLE_EVENT.fullmatch(stripped):
                # The manager correctly rejects immutable event types, but
                # the pinned MCP handler drops the application-error flag.
                result["isError"] = True
                break
            if stripped.lower().startswith(("validation error:", "validation errors:")):
                # These are model/request 4xx failures.  Keep the exact text
                # and execution semantics; only restore the missing MCP flag.
                result["isError"] = True
                break
            validation_match = _EVENT_RESPONSE_VALIDATION.match(stripped)
            if validation_match is not None:
                structured = {
                    "success": False,
                    "status_code": 500,
                    "error": {
                        "code": 500,
                        "message": "EVENT_RESPONSE_VALIDATION_FAILED",
                        "details": validation_match.group("details"),
                    },
                }
                item["text"] = json.dumps(structured)
                result["isError"] = True
                result["structuredContent"] = structured
                break
            match = _INTERNAL_NOT_FOUND.match(stripped)
            if match is None:
                continue
            structured = {
                "success": False,
                "status_code": 500,
                "error": {
                    "code": 500,
                    "message": "UNCLASSIFIED_BACKEND_ERROR",
                    "details": match.group("details"),
                },
            }
            item["text"] = json.dumps(structured)
            result["isError"] = True
            result["structuredContent"] = structured
            break
        return response

    module.handle_tools_call = handle_tools_call


def patch_last_owner_deletion(
    manager_class: type[Any], acl_model: Any, calendar_model: Any, owner_role: Any
) -> None:
    original = manager_class.delete_rule
    if tuple(inspect.signature(original).parameters) != (
        "self",
        "calendar_id",
        "rule_id",
    ):
        raise RuntimeError("pinned Calendar ACL delete_rule signature drifted")

    @wraps(original)
    def delete_rule(self: Any, calendar_id: str, rule_id: str) -> bool:
        from fastapi import HTTPException

        session = self.db
        try:
            rule = (
                session.query(acl_model)
                .join(calendar_model)
                .filter(
                    acl_model.id == rule_id,
                    acl_model.calendar_id == calendar_id,
                    calendar_model.user_id == self.user_id,
                )
                .first()
            )
            if not rule:
                return False
            if rule.role == owner_role:
                owner_count = (
                    session.query(acl_model)
                    .filter(
                        acl_model.calendar_id == calendar_id,
                        acl_model.role == owner_role,
                    )
                    .count()
                )
                if owner_count <= 1:
                    raise HTTPException(
                        status_code=400,
                        detail="Cannot delete the final owner ACL rule",
                    )
            rule_data = {
                "id": rule.id,
                "calendar_id": rule.calendar_id,
                "user_id": rule.user_id,
                "role": rule.role.value,
                "scope": rule.scope.as_dict() if rule.scope else {},
                "etag": rule.etag,
            }
            session.delete(rule)
            session.commit()
            self._send_acl_notification(calendar_id, "delete", rule_data)
            return True
        except HTTPException:
            session.rollback()
            raise
        except Exception:
            session.rollback()
            raise

    manager_class.delete_rule = delete_rule


def patch_acl_scope_validation(module: Any) -> None:
    """Return parameter validation errors for missing non-default scope values."""
    scope = module.ScopeInput
    original = scope.validate_scope_value

    @wraps(original)
    def validate_scope_value(self: Any) -> Any:
        if self.type != module.ScopeType.default and self.value is None:
            raise ValueError("scope.value is required for non-default scope types")
        return original(self)

    scope.validate_scope_value = validate_scope_value
    scope.__pydantic_decorators__.model_validators["validate_scope_value"].func = validate_scope_value
    # These request/response models embed ScopeInput's compiled validator.
    for model in (scope, module.ACLRuleInput, module.PatchACLRuleInput, module.InsertACLRule):
        model.model_rebuild(force=True)


def patch_extended_property_json(model: type[Any]) -> None:
    """Support the pinned manager's .astext on its SQLite JSON property column.

    Keep the original query, scope, pagination and permissions. Only this
    column gets the legacy accessor; do not patch SQLAlchemy's global JSON type.
    """
    from sqlalchemy import JSON

    class PropertyJSON(JSON):
        class Comparator(JSON.Comparator):
            @property
            def astext(self) -> Any:
                return self.as_string()

        comparator_factory = Comparator

    column = model.__table__.c.properties
    if type(column.type) is not JSON:
        raise RuntimeError("Calendar ExtendedProperty.properties must use generic JSON")
    column.type = PropertyJSON(none_as_null=column.type.none_as_null)


def patch_empty_event_recurrence(model: type[Any]) -> None:
    """Store an explicitly empty recurrence list as JSON, like nonempty lists.

    The pinned update manager's truthiness check skips json.dumps for [],
    then assigns the list directly to its TEXT column. Keep omitted/None and
    already serialized recurrence values unchanged.
    """
    from sqlalchemy import Text, event

    if not isinstance(model.__table__.c.recurrence.type, Text):
        raise RuntimeError("pinned Calendar recurrence must use a TEXT column")

    def serialize_empty(target: Any, value: Any, oldvalue: Any, initiator: Any) -> Any:
        if isinstance(value, list) and not value:
            return json.dumps(value)
        return value

    event.listen(model.recurrence, "set", serialize_empty, retval=True)


def patch_utc_event_storage(model: type[Any]) -> None:
    """Preserve explicit UTC provenance before SQLite drops datetime tzinfo.

    The pinned quick_add path supplies aware UTC datetimes but no zone columns.
    Never infer a zone for an already naive datetime or replace an explicit zone.
    """
    from sqlalchemy import event

    def preserve_utc(mapper, connection, target):
        for prefix in ('start', 'end'):
            value = getattr(target, prefix + '_datetime')
            if (value is not None and value.tzinfo is timezone.utc
                    and getattr(target, prefix + '_timezone') is None):
                setattr(target, prefix + '_timezone', 'UTC')

    event.listen(model, 'before_insert', preserve_utc)


def patch_get_event_timezone(manager_class: type[Any]) -> None:
    """Apply the requested response zone without filtering calendar ownership."""
    from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
    from fastapi import HTTPException

    original = manager_class.get_event
    if tuple(inspect.signature(original).parameters) != (
        'self', 'user_id', 'calendar_id', 'event_id', 'timeZone', 'maxAttendees'
    ):
        raise RuntimeError('pinned Calendar get_event signature drifted')

    @wraps(original)
    def get_event(self, user_id, calendar_id, event_id, timeZone=None, maxAttendees=None):
        if timeZone is None or not timeZone.strip():
            return original(self, user_id, calendar_id, event_id, timeZone, maxAttendees)
        try:
            zone = timezone.utc if timeZone in ("UTC", "Etc/UTC") else ZoneInfo(timeZone)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise HTTPException(status_code=400, detail=f'Invalid timeZone: {timeZone}') from exc
        result = original(self, user_id, calendar_id, event_id, None, maxAttendees)
        if result is None:
            return None
        result = result.model_copy(deep=True)
        for name in ('start', 'end', 'originalStartTime'):
            value = getattr(result, name, None)
            if value is None or value.dateTime is None:
                continue  # All-day dates must not shift across calendar days.
            parsed = datetime.fromisoformat(value.dateTime.replace('Z', '+00:00'))
            if parsed.tzinfo is None:
                if not value.timeZone:
                    raise ValueError('Stored event dateTime lacks offset and timeZone')
                source_zone = timezone.utc if value.timeZone in ("UTC", "Etc/UTC") else ZoneInfo(value.timeZone)
                parsed = parsed.replace(tzinfo=source_zone)
            value.dateTime = parsed.astimezone(zone).isoformat()
            value.timeZone = timeZone
        return result

    manager_class.get_event = get_event


def patch_nullable_ical_uid(module: Any) -> None:
    """Honor the request schema's nullable UID without changing string validation."""
    def wrap(original):
        @wraps(original.__func__)
        def validate(cls, v):
            return None if v is None else original(v)
        return classmethod(validate)

    for model in (module.EventCreateRequest, module.EventUpdateRequest):
        original = model._validate_iCalUID
        if tuple(inspect.signature(original).parameters) != ('v',):
            raise RuntimeError('pinned Calendar iCalUID validator signature drifted')
        model._validate_iCalUID = wrap(original)
        model.__pydantic_decorators__.field_validators['_validate_iCalUID'].func = model._validate_iCalUID
        model.model_rebuild(force=True)


def patch_calendar_list_http_errors(module: Any) -> None:
    """Preserve request errors swallowed by this pinned route's outer except."""
    from fastapi import HTTPException

    original = module.get_calendar_list
    routes = [r for r in module.router.routes if r.endpoint is original]
    if len(routes) != 1 or routes[0].path != "/users/me/calendarList":
        raise RuntimeError("pinned Calendar list route drifted")

    @wraps(original)
    async def get_calendar_list(*args: Any, **kwargs: Any) -> Any:
        try:
            return await original(*args, **kwargs)
        except HTTPException as exc:
            cause = exc.__context__
            if (
                exc.status_code == 500
                and isinstance(cause, HTTPException)
                and cause.status_code in (400, 410)
                and exc.detail == f"Internal server error: {cause}"
            ):
                raise cause from None
            raise

    module.get_calendar_list = get_calendar_list
    routes[0].endpoint = get_calendar_list
    routes[0].dependant.call = get_calendar_list


def patch_watch_channel_conflicts(manager_class: type[Any], method_name: str) -> None:
    """Keep domain meaning local; share exception propagation and response mapping."""
    if __package__:
        from .backend_errors import wrap_backend_call
    else:
        from backend_errors import wrap_backend_call
    original = getattr(manager_class, method_name)
    setattr(manager_class, method_name, wrap_backend_call(
        original, conflict=lambda exc: isinstance(exc, ValueError)
        and re.fullmatch(r"Channel with Id .+ already exists", str(exc)) is not None))


def install_snapshot_restore(module: Any) -> None:
    """Restore isolated snapshots without seed-time primary/ACL repair.

    Native seed-database is still used for first initialization. Restored
    databases are disposable branches, not new entries in the reset seed store.
    """
    import sqlite3
    from pathlib import Path
    from fastapi import HTTPException

    def restore_database(body: dict) -> dict:
        database_id = body.get("database_id")
        sql = body.get("sql_content")
        if not isinstance(database_id, str) or not re.fullmatch(r"robustops_[0-9a-f]{16}", database_id):
            raise HTTPException(400, "Expected an isolated RobustOps database_id")
        if not isinstance(sql, str) or not sql.strip():
            raise HTTPException(400, "sql_content is required")
        path = Path(module.calendar_session_manager.get_db_path(database_id))
        # A restore owns a fresh branch; never overwrite an active database.
        try:
            with path.open("xb"):
                pass
        except FileExistsError:
            raise HTTPException(409, "Restore database already exists") from None
        connection = sqlite3.connect(path)
        try:
            connection.executescript(sql)
            if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                raise sqlite3.DatabaseError("Restored snapshot failed integrity_check")
            connection.commit()
        finally:
            connection.close()
        # No create_all or business repairs: schema and rows come from snapshot.
        module.calendar_session_manager.init_database(database_id, create_tables=False)
        return {"success": True, "database_id": database_id}

    module.router.add_api_route("/restore-database", restore_database, methods=["POST"])


def main() -> None:
    import importlib

    patch_acl_scope_validation(importlib.import_module("schemas.acl"))
    patch_nullable_ical_uid(importlib.import_module("schemas.event"))
    acl_manager = importlib.import_module("database.managers.acl_manager")
    acl_models = importlib.import_module("database.models.acl")
    calendar_models = importlib.import_module("database.models.calendar")
    patch_last_owner_deletion(
        acl_manager.ACLManager,
        acl_models.ACLs,
        calendar_models.Calendar,
        acl_models.AclRole.owner,
    )
    patch_watch_channel_conflicts(acl_manager.ACLManager, "watch_acl")
    calendar_list_manager = importlib.import_module("database.managers.calendar_list_manager")
    patch_watch_channel_conflicts(calendar_list_manager.CalendarListManager, "watch_calendar_list")
    event_manager = importlib.import_module("database.managers.event_manager")
    event_models = importlib.import_module("database.models.event")
    patch_empty_event_recurrence(event_models.Event)
    patch_utc_event_storage(event_models.Event)
    patch_extended_property_json(event_models.ExtendedProperty)
    patch_event_time_validation(event_manager.EventManager)
    patch_optional_event_timezone(event_manager.EventManager)
    patch_materialized_event_listing(event_manager.EventManager, event_models.RecurringEvent)
    patch_event_response_conversion(event_manager.EventManager)
    patch_nullable_conference_key(event_manager.EventManager)
    patch_get_event_timezone(event_manager.EventManager)
    patch_recurring_queries(event_manager)

    handler = importlib.import_module("handlers.mcp_handler")
    patch_tool_handler_result_capture(handler)
    patch_not_found_results(handler)
    patch_calendar_list_http_errors(importlib.import_module("apis.calendarList.router"))
    patch_acl_request_sessions(importlib.import_module('apis.acl.router'))
    install_snapshot_restore(importlib.import_module("apis.database_router"))
    calendar_main = importlib.import_module("main")

    import uvicorn

    uvicorn.run(
        calendar_main.app,
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", "8003")),
    )


if __name__ == "__main__":
    main()
