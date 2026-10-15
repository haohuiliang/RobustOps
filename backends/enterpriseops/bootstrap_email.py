#!/usr/bin/env python3
"""Bootstrap the pinned Email MCP with narrow backend-contract repairs."""

from __future__ import annotations

import inspect
import os
import re
import secrets
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime, timedelta
from functools import wraps
from typing import Any


_WRAPPED_CREATE_DRAFT_4XX = re.compile(
    r"^Error creating draft: (?P<status>4\d\d): .+",
    re.DOTALL,
)
_PRECREATED_HISTORY_INFO_KEY = "robustops_precreated_email_history"


def _replace_endpoint(routes: Iterable[Any], name: str, replacement: Any) -> int:
    """Replace both Starlette's endpoint and FastAPI's dependency call."""

    replaced = 0
    for route in routes:
        endpoint = getattr(route, "endpoint", None)
        if getattr(endpoint, "__name__", None) != name:
            continue
        route.endpoint = replacement
        dependant = getattr(route, "dependant", None)
        if dependant is not None:
            dependant.call = replacement
        replaced += 1
    return replaced


def build_create_send_as_alias(module: Any) -> Any:
    original = module.create_send_as_alias
    if tuple(inspect.signature(original).parameters) != ("user", "send_as_data", "db"):
        raise RuntimeError("pinned Email create_send_as_alias signature drifted")

    @wraps(original)
    async def create_send_as_alias(user: Any, send_as_data: dict, db: Any) -> dict:
        if "sendAsEmail" not in send_as_data:
            raise module.HTTPException(
                status_code=400, detail="sendAsEmail is required"
            )
        send_as_email = send_as_data["sendAsEmail"]
        reply_to_email = send_as_data.get("replyToAddress")
        if not module.validate_email(send_as_email):
            raise module.HTTPException(
                status_code=400,
                detail="sendAsEmail must be a valid email address",
            )
        if reply_to_email not in (None, "") and not module.validate_email(
            reply_to_email
        ):
            raise module.HTTPException(
                status_code=400,
                detail="replyToAddress must be a valid email address",
            )
        if module.check_duplicate_send_as(db, user.id, send_as_email):
            raise module.HTTPException(
                status_code=409,
                detail=f"Send-as alias {send_as_email} already exists",
            )
        if "smtpMsa" in send_as_data:
            smtp_error = module.validate_smtp_msa(send_as_data["smtpMsa"])
            if smtp_error:
                raise module.HTTPException(
                    status_code=400, detail=f"Invalid smtpMsa: {smtp_error}"
                )
        for field in ("isPrimary", "isDefault", "treatAsAlias"):
            if field in send_as_data and not isinstance(send_as_data[field], bool):
                raise module.HTTPException(
                    status_code=400, detail=f"{field} must be a boolean"
                )
        if send_as_data.get("isPrimary", False):
            raise module.HTTPException(
                status_code=400,
                detail="Cannot create primary send-as alias via API",
            )

        signature = send_as_data.get("signature", "")
        if signature:
            signature = module.sanitize_html_signature(signature)
        verification_token = None
        verification_expires = None
        verification_status = "accepted"
        if send_as_email != user.primary_email:
            verification_token = secrets.token_urlsafe(32)
            verification_expires = datetime.now(UTC).replace(tzinfo=None) + timedelta(
                hours=24
            )
            verification_status = "pending"

        new_send_as = module.SendAsAlias(
            id=str(uuid.uuid4()),
            user_id=user.id,
            send_as_email=send_as_email,
            display_name=send_as_data.get("displayName", ""),
            reply_to_address=send_as_data.get("replyToAddress", ""),
            signature=signature,
            is_primary=(send_as_email == user.primary_email),
            is_default=send_as_data.get("isDefault", False),
            treat_as_alias=send_as_data.get("treatAsAlias", True),
            smtp_msa=send_as_data.get("smtpMsa"),
            verification_status=verification_status,
            verification_token=verification_token,
            verification_token_expires=verification_expires,
        )
        db.add(new_send_as)
        if new_send_as.is_default:
            module.ensure_single_default(db, user.id, new_send_as.id)
        db.commit()
        db.refresh(new_send_as)
        return module.format_send_as_response(new_send_as)

    return create_send_as_alias


def build_create_draft_status_passthrough(module: Any) -> Any:
    """Restore a swallowed draft 4xx without changing its model-visible detail."""

    original = module.create_draft
    expected = ("request", "http_request", "db", "user")
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("pinned Email create_draft signature drifted")

    @wraps(original)
    async def create_draft(request: Any, http_request: Any, db: Any, user: Any) -> Any:
        try:
            return await original(request, http_request, db, user)
        except module.HTTPException as exc:
            detail = getattr(exc, "detail", None)
            match = (
                _WRAPPED_CREATE_DRAFT_4XX.fullmatch(detail)
                if exc.status_code == 500 and isinstance(detail, str)
                else None
            )
            if match is None:
                raise
            raise module.HTTPException(
                status_code=int(match.group("status")),
                detail=detail,
            ) from exc

    return create_draft


def build_history_record(module: Any) -> Any:
    """Create the history parent before flushing rows that reference it."""

    original = module.create_history_record
    expected = (
        "db",
        "user_id",
        "message_id",
        "action",
        "label_changes",
        "history_id",
    )
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("pinned Email create_history_record signature drifted")

    def _loaded_message(db: Any, message_id: str) -> Any | None:
        values = list(getattr(db, "new", ()))
        identity_map = getattr(db, "identity_map", None)
        if identity_map is not None:
            values.extend(identity_map.values())
        return next(
            (
                value
                for value in values
                if isinstance(value, module.Message)
                and getattr(value, "id", None) == message_id
            ),
            None,
        )

    def _changed_history_id(message: Any) -> str | None:
        # modify_message in the pinned image writes a non-mapped `history_id`
        # attribute by mistake.  Preserve the generated value while assigning
        # it to the real history_id_last column.
        typo_value = getattr(message, "__dict__", {}).get("history_id")
        if isinstance(typo_value, str) and typo_value:
            return typo_value
        try:
            from sqlalchemy import inspect as sqlalchemy_inspect

            state = sqlalchemy_inspect(message)
            attribute = state.attrs.history_id_last
            if attribute.history.has_changes():
                value = attribute.value
                if isinstance(value, str) and value:
                    return value
        except (AttributeError, TypeError):
            pass
        return None

    @wraps(original)
    def create_history_record(
        db: Any,
        user_id: str,
        message_id: str,
        action: str,
        label_changes: dict | None = None,
        history_id: str | None = None,
    ) -> Any:
        message = _loaded_message(db, message_id)
        if history_id is None and action == "messageAdded" and message is not None:
            message_history_id = getattr(message, "history_id_last", None)
            precreated = getattr(db, "info", {}).get(
                _PRECREATED_HISTORY_INFO_KEY, {}
            )
            history_event = (
                precreated.get(message_history_id)
                if isinstance(precreated, dict)
                and isinstance(message_history_id, str)
                else None
            )
            if history_event is not None:
                history_event.event_type = action
                history_event.event_data = {"message_id": message_id}
                db.flush([history_event])
                return history_event
        if history_id is None and action == "labelsChanged" and message is not None:
            history_id = _changed_history_id(message)
            if history_id is not None:
                # messageAdded may already have inserted this history parent
                # in the same transaction. labelsChanged is a distinct event.
                # Avoid autoflush: the pending Message may reference a parent
                # that this helper has not inserted yet.
                with db.no_autoflush:
                    exists = db.query(module.HistoryEvent).filter(
                        module.HistoryEvent.history_id == history_id
                    ).first() is not None
                if exists:
                    history_id = None
        history_id = history_id or str(module.time.time_ns() // 1_000)

        event_data: dict[str, Any] = {}
        if action in {"messageAdded", "messageDeleted"}:
            event_data = {"message_id": message_id}
        elif action == "labelsChanged" and label_changes:
            event_data = {
                "message_id": message_id,
                "labels_added": label_changes.get("added", []),
                "labels_removed": label_changes.get("removed", []),
            }

        history_event = module.HistoryEvent(
            user_id=user_id,
            history_id=history_id,
            event_type=action,
            event_timestamp=module.datetime.now(),
            event_data=event_data,
        )
        db.add(history_event)
        # Flush only the parent.  The endpoint's final commit remains the
        # single atomic commit for Thread/Message/labels/history.
        db.flush([history_event])
        if message is not None and action != "messageDeleted":
            message.history_id_last = history_id
        return history_event

    return create_history_record


def build_ordered_message_create_endpoint(
    endpoint: Any,
    module: Any,
    *,
    nested_message: bool,
) -> Any:
    """Create a new history parent before the image can flush message children."""

    signature = inspect.signature(endpoint)
    if "request" not in signature.parameters or "db" not in signature.parameters:
        raise RuntimeError(
            f"pinned Email {endpoint.__name__} endpoint signature drifted"
        )

    @wraps(endpoint)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        request = bound.arguments.get("request")
        db = bound.arguments.get("db")
        user = bound.arguments.get("user")
        if request is None or db is None or user is None:
            raise RuntimeError(
                f"Email {endpoint.__name__} did not receive request/db/user"
            )
        bind = db.get_bind()
        if getattr(getattr(bind, "dialect", None), "name", None) != "sqlite":
            raise RuntimeError("pinned Email backend is no longer SQLite")

        message_request = getattr(request, "message", None) if nested_message else request
        if (
            message_request is None
            or not hasattr(message_request, "historyId")
            or not hasattr(message_request, "threadId")
        ):
            raise RuntimeError(
                f"pinned Email {endpoint.__name__} request history/thread ID drifted"
            )

        original_history_id = message_request.historyId
        original_thread_id = message_request.threadId
        original_autoflush = db.autoflush
        precreated = getattr(db, "info", {}).setdefault(
            _PRECREATED_HISTORY_INFO_KEY, {}
        )
        if not isinstance(precreated, dict):
            raise RuntimeError("Email session precreated-history state drifted")

        injected_history_id: str | None = None
        injected_thread_id: str | None = None
        try:
            if not original_history_id:
                injected_history_id = str(module.time.time_ns() // 1_000)
                history_event = module.HistoryEvent(
                    user_id=user.id,
                    history_id=injected_history_id,
                    event_type="messageAdded",
                    event_timestamp=module.datetime.now(),
                    event_data={},
                )
                db.add(history_event)
                db.flush([history_event])
                precreated[injected_history_id] = history_event
                message_request.historyId = injected_history_id

            if not original_thread_id:
                history_event_exists = injected_history_id is not None
                if not history_event_exists:
                    history_event_exists = (
                        db.query(module.HistoryEvent)
                        .filter(
                            module.HistoryEvent.history_id
                            == message_request.historyId,
                            module.HistoryEvent.user_id == user.id,
                        )
                        .first()
                        is not None
                    )
                # Let the original endpoint return its existing 400 for an
                # invalid caller-supplied history ID.
                if history_event_exists:
                    injected_thread_id = str(module.uuid.uuid4())
                    thread = module.Thread(
                        id=injected_thread_id,
                        user_id=user.id,
                        history_id_last=message_request.historyId,
                        snippet=getattr(message_request, "snippet", None) or "",
                    )
                    db.add(thread)
                    db.flush([thread])
                    message_request.threadId = injected_thread_id

            # The pinned session factory disables autoflush.  These two image
            # endpoints issue Core association INSERTs after ORM adds, so their
            # intervening validation/label queries must flush Thread/Message.
            db.autoflush = True
            return await endpoint(*args, **kwargs)
        except BaseException:
            db.rollback()
            raise
        finally:
            db.autoflush = original_autoflush
            message_request.historyId = original_history_id
            message_request.threadId = original_thread_id
            if injected_history_id is not None:
                precreated.pop(injected_history_id, None)
            if not precreated:
                getattr(db, "info", {}).pop(_PRECREATED_HISTORY_INFO_KEY, None)

    return wrapped


def build_send_draft_preserving_message(endpoint: Any, draft_model: Any) -> Any:
    """Sending removes the draft wrapper, not its sent Message via ORM cascade."""
    signature = inspect.signature(endpoint)
    @wraps(endpoint)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        bound = signature.bind_partial(*args, **kwargs)
        db = bound.arguments["db"]
        request = bound.arguments["request"]
        original_delete = db.delete
        def delete(instance):
            if isinstance(instance, draft_model) and instance.id == request.id:
                # Bulk DELETE bypasses the pinned Draft.message delete cascade.
                # Keep the endpoint's original transaction/commit/rollback.
                db.query(draft_model).filter(draft_model.id == instance.id).delete(synchronize_session=False)
            else:
                original_delete(instance)
        db.delete = delete
        try:
            return await endpoint(*args, **kwargs)
        finally:
            db.delete = original_delete
    return wrapped


def build_deferred_fk_endpoint(endpoint: Any) -> Any:
    """Defer SQLite FK checks within one endpoint transaction, never globally."""

    signature = inspect.signature(endpoint)
    if "db" not in signature.parameters:
        raise RuntimeError(
            f"pinned Email {endpoint.__name__} endpoint no longer receives db"
        )

    @wraps(endpoint)
    async def wrapped(*args: Any, **kwargs: Any) -> Any:
        from sqlalchemy import text

        bound = signature.bind_partial(*args, **kwargs)
        db = bound.arguments.get("db")
        if db is None:
            raise RuntimeError(f"Email {endpoint.__name__} did not receive db")
        bind = db.get_bind()
        if getattr(getattr(bind, "dialect", None), "name", None) != "sqlite":
            raise RuntimeError("pinned Email backend is no longer SQLite")
        db.execute(text("PRAGMA defer_foreign_keys = ON"))
        try:
            return await endpoint(*args, **kwargs)
        except BaseException:
            db.rollback()
            raise

    return wrapped


def apply_label_filters(
    query: Any, label_ids: list[str], message_model: Any, table: Any
) -> Any:
    """Apply Gmail's all-label semantics without sharing a FROM with q filters."""

    from sqlalchemy import and_, exists

    for index, label_id in enumerate(dict.fromkeys(label_ids)):
        association = table.alias(f"list_messages_label_{index}")
        query = query.filter(
            exists().where(
                and_(
                    association.c.message_id == message_model.id,
                    association.c.label_id == label_id,
                )
            )
        )
    return query


def build_list_messages(module: Any) -> Any:
    original = module.list_messages
    expected = (
        "request",
        "maxResults",
        "pageToken",
        "q",
        "labelIds",
        "includeSpamTrash",
        "db",
        "user",
    )
    if tuple(inspect.signature(original).parameters) != expected:
        raise RuntimeError("pinned Email list_messages signature drifted")

    @wraps(original)
    async def list_messages(
        request: Any,
        maxResults: int = 100,
        pageToken: str | None = None,
        q: str | None = None,
        labelIds: list[str] | None = None,
        includeSpamTrash: bool = False,
        db: Any = None,
        user: Any = None,
    ) -> dict:
        try:
            labels = labelIds or []
            query = db.query(module.Message).filter(module.Message.user_id == user.id)
            query = apply_label_filters(
                query, labels, module.Message, module.message_labels
            )
            if not includeSpamTrash:
                spam_trash = (
                    db.query(module.message_labels.c.message_id)
                    .filter(module.message_labels.c.label_id.in_(["SPAM", "TRASH"]))
                    .subquery()
                )
                query = query.filter(
                    ~module.Message.id.in_(db.query(spam_trash.c.message_id))
                )
            if q:
                query_filter = module.parse_email_query(q, db, user.id)
                if query_filter is not None:
                    query = query.filter(query_filter)
            query = query.order_by(module.Message.internal_date_ms.desc())
            offset = 0
            if pageToken:
                try:
                    offset = int(pageToken)
                except ValueError as exc:
                    raise module.HTTPException(
                        status_code=400, detail="Invalid pageToken"
                    ) from exc
            total_count = query.count()
            messages = query.offset(offset).limit(maxResults).all()
            response = {
                "messages": [
                    module.format_message_response(message, db) for message in messages
                ],
                "resultSizeEstimate": total_count,
            }
            if offset + maxResults < total_count:
                response["nextPageToken"] = str(offset + maxResults)
            return response
        except module.HTTPException:
            raise
        except Exception as exc:
            raise module.HTTPException(
                status_code=500, detail=f"Error listing messages: {exc}"
            ) from exc

    return list_messages


def patch_mcp_error_flags(module: Any) -> None:
    original = module.handle_tools_call
    if getattr(original, "__robustops_email_error_flags__", False):
        return

    @wraps(original)
    async def handle_tools_call(request: Any, fastapi_request: Any) -> Any:
        response = await original(request, fastapi_request)
        result = getattr(response, "result", None)
        if isinstance(result, dict):
            texts = [
                item.get("text", "")
                for item in result.get("content", [])
                if isinstance(item, dict)
            ]
            if any(
                text.startswith(("❌ Unknown tool:", "❌ Tool execution failed:"))
                for text in texts
            ):
                result["isError"] = True
        return response

    handle_tools_call.__robustops_email_error_flags__ = True
    module.handle_tools_call = handle_tools_call


def patch_cse_pagination(app: Any, module: Any) -> None:
    if __package__:
        from .backend_errors import validate_offset_token
    else:
        from backend_errors import validate_offset_token
    import inspect
    original = module.email_list_cse_keypairs
    signature = inspect.signature(original)
    if 'pageToken' not in signature.parameters:
        raise RuntimeError('pinned Email CSE pagination signature drifted')

    @wraps(original)
    async def listing(*args, **kwargs):
        values = signature.bind(*args, **kwargs)
        validate_offset_token(values.arguments.get('pageToken'))
        return await original(*args, **kwargs)
    if _replace_endpoint(app.routes, 'email_list_cse_keypairs', listing) != 1:
        raise RuntimeError('pinned Email CSE listing route differs')
    module.email_list_cse_keypairs = listing


def build_modify_message_validation(original: Any) -> Any:
    """Reject duplicate additions before the native junction-table writes."""
    signature = inspect.signature(original)

    @wraps(original)
    async def modify(*args, **kwargs):
        from fastapi import HTTPException
        request = signature.bind(*args, **kwargs).arguments['request']
        additions = request.addLabelIds or []
        if len(additions) != len(set(additions)):
            raise HTTPException(400, 'addLabelIds must not contain duplicate label IDs')
        return await original(*args, **kwargs)
    return modify


def main() -> None:
    import importlib

    email_main = importlib.import_module("main")
    drafts = importlib.import_module("app.api.drafts.router")
    send_as = importlib.import_module("app.api.send_as.router")
    messages = importlib.import_module("app.api.messages.router")
    mcp_handler = importlib.import_module("app.mcp.handlers.mcp_handler")
    send_as_tools = importlib.import_module("app.mcp.tools.send_as_tools")

    tools = [
        tool
        for tool in send_as_tools.send_as_tools
        if tool.name == "create_send_as_alias"
    ]
    if len(tools) != 1 or "replyToAddress" in tools[0].inputSchema.get("required", []):
        raise RuntimeError("pinned Email create_send_as_alias schema drifted")

    create_replacement = build_create_send_as_alias(send_as)
    list_replacement = build_list_messages(messages)
    history_replacement = build_history_record(messages)
    messages.create_history_record = history_replacement
    drafts.create_history_record = history_replacement

    draft_replacement = build_ordered_message_create_endpoint(
        build_create_draft_status_passthrough(drafts),
        messages,
        nested_message=True,
    )
    create_count = _replace_endpoint(
        email_main.app.routes, "create_send_as_alias", create_replacement
    )
    list_count = _replace_endpoint(
        email_main.app.routes, "list_messages", list_replacement
    )
    transaction_endpoints = {
        "create_draft": draft_replacement,
        "update_draft": build_deferred_fk_endpoint(drafts.update_draft),
        "delete_draft": build_deferred_fk_endpoint(drafts.delete_draft),
        "send_draft": build_deferred_fk_endpoint(build_send_draft_preserving_message(drafts.send_draft, drafts.Draft)),
        "batch_delete_messages": build_deferred_fk_endpoint(
            messages.batch_delete_messages
        ),
        "batch_modify_messages": build_deferred_fk_endpoint(
            messages.batch_modify_messages
        ),
        "delete_message": build_deferred_fk_endpoint(messages.delete_message),
        "import_message": build_deferred_fk_endpoint(messages.import_message),
        "insert_message": build_deferred_fk_endpoint(messages.insert_message),
        "modify_message": build_deferred_fk_endpoint(build_modify_message_validation(messages.modify_message)),
        "send_message": build_ordered_message_create_endpoint(
            messages.send_message,
            messages,
            nested_message=False,
        ),
        "trash_message": build_deferred_fk_endpoint(messages.trash_message),
        "untrash_message": build_deferred_fk_endpoint(messages.untrash_message),
    }
    transaction_counts = {
        name: _replace_endpoint(email_main.app.routes, name, replacement)
        for name, replacement in transaction_endpoints.items()
    }
    if (
        create_count != 1
        or list_count != 1
        or any(count != 1 for count in transaction_counts.values())
    ):
        raise RuntimeError(
            "pinned Email root routes did not contain unique contract endpoints"
        )
    for name in ("create_draft", "update_draft", "delete_draft", "send_draft"):
        setattr(drafts, name, transaction_endpoints[name])
    for name in transaction_endpoints.keys() - {
        "create_draft",
        "update_draft",
        "delete_draft",
        "send_draft",
    }:
        setattr(messages, name, transaction_endpoints[name])
    send_as.create_send_as_alias = create_replacement
    messages.list_messages = list_replacement
    patch_mcp_error_flags(mcp_handler)
    if __package__:
        from .backend_errors import patch_mcp_result_capture
    else:
        from backend_errors import patch_mcp_result_capture
    patch_mcp_result_capture(mcp_handler)
    patch_cse_pagination(email_main.app, importlib.import_module('app.api.cse_keypairs.router'))

    import uvicorn

    uvicorn.run(
        email_main.app,
        host="0.0.0.0",
        port=int(os.environ.get("API_PORT", "8005")),
    )


if __name__ == "__main__":
    main()
