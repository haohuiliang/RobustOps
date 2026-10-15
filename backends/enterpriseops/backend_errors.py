"""Shared error propagation for pinned backends, with request-owned DB scope.

The caller supplies the actual ORM table, not task IDs or error-message lists.
An internal/generated key or another table's write is never a request rejection.
Transactions remain owned by the original manager; only a failed SQLAlchemy
session explicitly supplied by that manager is rolled back here.
"""
import inspect
import json
import re
import sqlite3
from functools import wraps
from contextvars import ContextVar
from collections.abc import Mapping


SQLITE_INT_MIN = -(2 ** 63)
SQLITE_INT_MAX = 2 ** 63 - 1


def validate_offset_token(token):
    """Validate an existing numeric-offset API without rewriting its argument."""
    from fastapi import HTTPException
    if token in (None, ''):
        return
    try:
        offset = int(token)
    except (ValueError, TypeError):
        offset = -1
    if offset < 0 or offset > SQLITE_INT_MAX:
        raise HTTPException(400, {
            'code': 'INVALID_PAGE_TOKEN', 'status_code': 400,
            'message': 'Invalid pageToken. Use nextPageToken returned by this listing.',
        })


def restore_tool_error(result, tool_result):
    """Restore typed handler metadata discarded by the native MCP formatter."""
    if not isinstance(result, dict) or not isinstance(tool_result, Mapping):
        return
    status = tool_result.get('status_code')
    if not isinstance(status, int) or isinstance(status, bool) or status < 400:
        return
    error = tool_result.get('error')
    if not isinstance(error, Mapping):
        error = {'code': status, 'message': str(tool_result.get('text') or 'Tool request failed')}
    result['isError'] = True
    result['structuredContent'] = {
        'success': False, 'status_code': status, 'error': dict(error),
        'source': 'backend_http',
    }


def patch_mcp_result_capture(module):
    """Scope handler metadata to one call, including concurrent MCP requests."""
    captured = ContextVar('backend_tool_result', default=None)
    handlers = module.TOOL_HANDLERS
    for name, original in list(handlers.items()):
        def wrap_handler(function):
            @wraps(function)
            async def call(*args, **kwargs):
                value = await function(*args, **kwargs)
                captured.set(value)
                return value
            return call
        handlers[name] = wrap_handler(original)
    original_call = module.handle_tools_call

    @wraps(original_call)
    async def call(*args, **kwargs):
        token = captured.set(None)
        try:
            response = await original_call(*args, **kwargs)
            restore_tool_error(getattr(response, 'result', None), captured.get())
            return response
        finally:
            captured.reset(token)
    module.handle_tools_call = call


def json_safe_validation_errors(errors):
    def safe(value):
        try:
            json.dumps(value)
        except (TypeError, ValueError):
            if isinstance(value, dict):
                return {str(key): safe(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [safe(item) for item in value]
            return str(value)
        return value
    return [{**{key: safe(value) for key, value in error.items()},
             'loc': list(error.get('loc', ()))} for error in errors]


def _causes(exc):
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        yield exc
        exc = exc.__cause__ or exc.__context__


def _request_values(function, args, kwargs):
    values = {}
    for name, value in inspect.signature(function).bind(*args, **kwargs).arguments.items():
        if name == 'self':
            continue
        if hasattr(value, 'model_dump'):
            values.update(value.model_dump(exclude_unset=True))
        elif isinstance(value, dict):
            values.update(value)
        elif isinstance(value, (str, int, float, bool)) or value is None:
            values[name] = value
    return values


def _constraint_detail(exc, table, values, context_fields):
    from sqlalchemy import CheckConstraint
    from sqlalchemy.exc import IntegrityError

    if not isinstance(exc, IntegrityError) or not isinstance(exc.orig, sqlite3.IntegrityError):
        return None
    # ORM statements are trusted SQL; scope the rejection to this operation's
    # declared table so a failure in an internal helper is not mislabeled.
    statement = re.match(r'^\s*(?:INSERT\s+INTO|UPDATE|DELETE\s+FROM)\s+(?:"([^\"]+)"|(\w+))(?=\s|\()',
                         exc.statement or '', re.I)
    if statement is None or (statement[1] or statement[2]) != table.name:
        return None
    match = re.fullmatch(r'(UNIQUE|NOT NULL|CHECK|FOREIGN KEY) constraint failed(?:: (.*))?', str(exc.orig))
    if match is None:
        return None
    kind, detail = match.groups()
    provided = set(values) & set(table.c.keys())
    allowed = provided | set(context_fields)
    if kind in ('UNIQUE', 'NOT NULL'):
        columns = (detail or '').split(', ')
        if any(not column.startswith(table.name + '.') for column in columns):
            return None
        fields = {column.split('.', 1)[1] for column in columns}
        if not fields <= allowed or not fields & provided:
            return None
        if kind == 'NOT NULL' and any(values.get(f) not in (None, '') for f in fields):
            return None  # A nonempty request value lost by the implementation.
    elif kind == 'FOREIGN KEY':
        foreign = {fk.parent.name for fk in table.foreign_keys}
        # Only fields actually supplied by this request can justify rejection;
        # an internally generated history/parent ID is an implementation error.
        required = {f for f in foreign if not table.c[f].nullable}
        if not required <= allowed or not foreign & provided:
            return None
        fields = foreign & provided
    else:
        checks = [c for c in table.constraints if isinstance(c, CheckConstraint)
                  and (c.name == detail or str(c.sqltext) == detail)]
        if len(checks) != 1:
            return None
        expression = str(checks[0].sqltext)
        fields = {column.name for column in table.c
                  if re.search(r'\b' + re.escape(column.name) + r'\b', expression)}
        if not fields & provided:
            return None
    if not fields or not fields <= set(table.c.keys()):
        return None
    internal_pk = {column.name for column in table.primary_key if column.name not in provided}
    if fields & internal_pk:
        return None
    message = {
        'UNIQUE': 'A record already exists for these values.',
        'FOREIGN KEY': 'A referenced record does not exist.',
        'NOT NULL': 'Required fields cannot be empty.',
        'CHECK': 'The requested values conflict with the record constraints.',
    }[kind]
    status = 422 if kind == 'NOT NULL' else 409
    return status, dict(code='CONSTRAINT_REJECTED', status_code=status,
                        constraint_kind=kind, fields=sorted(fields), message=message)


def mapped_error(exc, *, table=None, values=None, context_fields=(), conflict=None):
    """Preserve typed 4xx through native wrappers; never classify arbitrary 500 text."""
    from fastapi import HTTPException
    from sqlalchemy.exc import DBAPIError, IntegrityError
    chain = list(_causes(exc))
    if any(isinstance(c, HTTPException) and c.status_code >= 501
           or isinstance(c, DBAPIError) and not isinstance(c, IntegrityError) for c in chain):
        return None
    if isinstance(exc, HTTPException) and exc.status_code < 500:
        return exc
    for cause in chain:
        if isinstance(cause, HTTPException) and 400 <= cause.status_code < 500:
            return cause
    for cause in chain:
        if conflict is not None and conflict(cause):
            return HTTPException(409, str(cause))
        if table is not None and all(isinstance(c, (HTTPException, IntegrityError, sqlite3.IntegrityError))
                                     for c in chain):
            result = _constraint_detail(cause, table, values or {}, context_fields)
            if result is not None:
                status, detail = result
                return HTTPException(status, detail)
    return None


def wrap_backend_call(function, *, table=None, context_fields=(), conflict=None, manager_session=False):
    def handle(exc, args, kwargs, values):
        if manager_session:
            session = getattr(args[0], 'session', None)
            # Failed flush/commit poisons a Session. Roll back before exposing
            # the rejection. Never claim to undo an already committed write.
            if session is not None and not session.is_active:
                session.rollback()
        mapped = mapped_error(exc, table=table, values=values,
                              context_fields=context_fields, conflict=conflict)
        if mapped is not None and mapped is not exc:
            raise mapped from exc

    if inspect.iscoroutinefunction(function):
        @wraps(function)
        async def call(*args, **kwargs):
            values = _request_values(function, args, kwargs)
            try:
                return await function(*args, **kwargs)
            except Exception as exc:
                handle(exc, args, kwargs, values)
                raise
    else:
        @wraps(function)
        def call(*args, **kwargs):
            values = _request_values(function, args, kwargs)
            try:
                return function(*args, **kwargs)
            except Exception as exc:
                handle(exc, args, kwargs, values)
                raise
    return call


def patch_route_errors(module, endpoint_name, *, table, context_fields=()):
    original = getattr(module, endpoint_name)
    routes = [route for route in module.router.routes if route.endpoint is original]
    if len(routes) != 1:
        raise RuntimeError(f'pinned route differs: {endpoint_name}')
    replacement = wrap_backend_call(original, table=table, context_fields=context_fields)
    setattr(module, endpoint_name, replacement)
    routes[0].endpoint = replacement
    routes[0].dependant.call = replacement
