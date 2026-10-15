"""Strict HTTP boundary for the clean EnterpriseOps evaluation path."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
from typing import Any, Mapping
from uuid import uuid4
from zipfile import ZipFile

import httpx


class InfrastructureError(RuntimeError):
    """An external service or response contract made the attempt invalid."""


_CALENDAR_PLAIN_ERROR = re.compile(
    r"(?:\bnot found\b|\bdoes not exist\b|\bis required\b|"
    r"^validation errors?:|"
    r"^cannot move event of type '[^']+'\. only default events can be moved\.|"
    r"^cannot delete primary calendar\.?$|"
    r"^can only clear primary calendars$|"
    r"^event end time must be after start time$|"
    r"^eventtype once set can not be modified\.|"
    r"^both 'start' and 'end' fields are required for put operations$|"
    r"^recurrence validation failed:|"
    r"^to set backgroundcolor/foregroundcolor you must set colorrgbformat=true$|"
    r"^use outofofficeproperties only when eventtype is set to 'outofoffice'$|"
    r"^value for type under conferencesolutionkey\b|"
    r"^no fields provided\b|"
    r"^(?:error|failed|invalid|bad request|forbidden|unauthorized|internal server error)\b)",
    re.IGNORECASE,
)
_CALENDAR_LEGACY_5XX_APPLICATION_ERROR = re.compile(
    r"^(?:internal server error: )?cannot delete primary calendar\.?$",
    re.IGNORECASE,
)
_DRIVE_HANDLER_ERROR = re.compile(r"^Error executing tool [A-Za-z0-9_]+:")
_BACKEND_INTEGRITY_ERROR = re.compile(
    r"\((?:sqlite3\.)?IntegrityError\)\s+"
    r"(?:FOREIGN KEY|UNIQUE|NOT NULL|CHECK) constraint failed",
    re.IGNORECASE,
)


def _content_texts(result: Mapping[str, Any]) -> list[str]:
    content = result.get("content")
    if not isinstance(content, list):
        return []
    return [
        item["text"]
        for item in content
        if isinstance(item, Mapping) and isinstance(item.get("text"), str)
    ]


def _error_message(error: Any) -> str | None:
    if isinstance(error, Mapping):
        raw_message = error.get("message") or error.get("error")
    else:
        raw_message = error
    return str(raw_message) if raw_message is not None else None


def _known_calendar_application_error(
    server_name: str, message: str | None
) -> bool:
    return (
        server_name == "gym-calendar"
        and message is not None
        and _CALENDAR_LEGACY_5XX_APPLICATION_ERROR.fullmatch(message.strip()) is not None
    )


def _database_constraint_outcome(result):
    """Classify explicit SQL constraint rejection, not arbitrary result data.

    Constraint failures are unsuccessful tool operations. They do not establish
    whether the caller or backend is at fault, or whether earlier writes occurred.
    Operational/corruption errors and unavailable-service codes remain invalid.
    """
    messages=[]; codes=[]; constraint_details=[]
    def visit(value, failure=False):
        if isinstance(value, Mapping):
            failure = failure or value.get("success") is False or value.get("error") is True or bool(value.get("error_code"))
            if (failure and value.get('code') == 'CONSTRAINT_REJECTED'
                    and value.get('constraint_kind') in ('UNIQUE', 'FOREIGN KEY', 'NOT NULL', 'CHECK')
                    and value.get('status_code') in (409, 422)
                    and isinstance(value.get('message'), str)):
                constraint_details.append(value)
            for key, item in value.items():
                if key in ("status_code", "code") and isinstance(item, int) and not isinstance(item, bool):
                    codes.append(item)
                visit(item, failure or key == "error")
        elif isinstance(value, list):
            for item in value:visit(item, failure)
        elif isinstance(value, str) and failure:
            messages.append(value)
    marked=result.get("isError") is True
    visit(result.get("structuredContent"),marked)
    for text in _content_texts(result):
        try:payload=json.loads(text)
        except json.JSONDecodeError:payload=None
        if isinstance(payload, (Mapping,list)):
            visit(payload,marked)
        elif marked or re.match(r"^(?:❌\s*)?(?:Error|Failed)\b",text.strip(),re.IGNORECASE):
            messages.append(text)
    matches=[m for m in messages if _BACKEND_INTEGRITY_ERROR.search(m)]
    fatal=re.compile(r"\b(?:sqlite3\.)?(?:OperationalError|DatabaseError)\b|"
                     r"database disk image is malformed|database corruption|disk I/O error|"
                     r"database is locked|unable to open database file|no such (?:table|column)",re.IGNORECASE)
    if any(fatal.search(m) for m in messages):
        raise InfrastructureError("MCP returned a database operational or corruption failure")
    if not matches and not constraint_details:return None
    if any(500<=c<=599 for c in codes):
        raise InfrastructureError("MCP database constraint response also contains an infrastructure failure")
    if not constraint_details:
        if any(400 <= c <= 499 for c in codes):
            return None  # Preserve an explicit rejection, without guessing responsibility.
        raise InfrastructureError("MCP returned an unclassified database constraint failure")
    detail = constraint_details[0] if constraint_details else None
    return {"is_error":True,"outcome":{"transport_ok":True,"application_ok":False,
            "source":"database_constraint_rejection","mcp_is_error":marked,
            "status_code":next((c for c in codes if c>=400),None),
            "message":detail['message'] if detail is not None else matches[0],
            **({'constraint_kind':detail['constraint_kind']} if detail is not None else {})},"result":result}


def normalize_mcp_tool_result(
    server_name: str,
    result: Mapping[str, Any],
    *,
    tool_name: str | None = None,
    arguments: Mapping[str, Any] | None = None,
) -> Mapping[str, Any]:
    """Normalize explicit application failures without changing the raw MCP result."""
    mcp_is_error = result.get("isError", False)
    if not isinstance(mcp_is_error, bool):
        raise InfrastructureError("MCP tools/call result.isError must be a boolean")

    is_error = mcp_is_error
    source = "mcp_is_error" if mcp_is_error else None
    status_code: int | None = None
    message: str | None = None

    structured_content = result.get("structuredContent")
    if structured_content is not None and not isinstance(structured_content, Mapping):
        raise InfrastructureError(
            "MCP tools/call result.structuredContent must be an object"
        )
    constraint = _database_constraint_outcome(result)
    if constraint is not None:
        return constraint
    if isinstance(structured_content, Mapping):
        payload_status = structured_content.get("status_code")
        error = structured_content.get("error")
        error_message = _error_message(error)
        error_code = error.get("code") if isinstance(error, Mapping) else None
        error_status = (
            error.get("status_code") if isinstance(error, Mapping) else None
        )
        candidate_codes = [
            candidate
            for candidate in (payload_status, error_code, error_status)
            if isinstance(candidate, int) and not isinstance(candidate, bool)
        ]
        backend_status = next(
            (candidate for candidate in candidate_codes if 500 <= candidate <= 599),
            None,
        )
        known_application_5xx = backend_status is not None and _known_calendar_application_error(
            server_name, error_message
        )
        if backend_status is not None and not known_application_5xx:
            raise InfrastructureError(
                f"MCP {server_name} returned backend HTTP {backend_status}"
            )
        application_status = next(
            (candidate for candidate in candidate_codes if 400 <= candidate <= 499),
            None,
        )
        status_code = backend_status if known_application_5xx else application_status or (
            candidate_codes[0] if candidate_codes else None
        )
        explicit_failure = structured_content.get("success") is False and (
            status_code is not None or error is not None
        )
        if explicit_failure or (
            status_code is not None and 400 <= status_code <= 499
        ):
            is_error = True
            if not mcp_is_error:
                source = "structured_application_result"
            if error_message is not None:
                message = error_message

    for text in _content_texts(result):
        stripped = text.strip()
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            payload = None

        if isinstance(payload, Mapping):
            payload_status = payload.get("status_code")
            error = payload.get("error")
            error_message = _error_message(error)
            error_code = error.get("code") if isinstance(error, Mapping) else None
            error_status = (
                error.get("status_code") if isinstance(error, Mapping) else None
            )
            candidate_codes = [
                candidate
                for candidate in (payload_status, error_code, error_status)
                if isinstance(candidate, int) and not isinstance(candidate, bool)
            ]
            backend_status = next(
                (candidate for candidate in candidate_codes if 500 <= candidate <= 599),
                None,
            )
            known_application_5xx = backend_status is not None and _known_calendar_application_error(
                server_name, error_message
            )
            if backend_status is not None and not known_application_5xx:
                raise InfrastructureError(
                    f"MCP {server_name} returned backend HTTP {backend_status}"
                )
            application_status = next(
                (candidate for candidate in candidate_codes if 400 <= candidate <= 499),
                None,
            )
            status_code = backend_status if known_application_5xx else application_status or (
                candidate_codes[0] if candidate_codes else None
            )
            explicit_failure = payload.get("success") is False and (
                status_code is not None or error is not None
            )
            if explicit_failure or (
                status_code is not None and 400 <= status_code <= 499
            ):
                is_error = True
                if not mcp_is_error:
                    source = "structured_application_result"
                if error_message is not None:
                    message = error_message
                break
        elif server_name == "gym-google-drive-mcp" and _DRIVE_HANDLER_ERROR.match(
            stripped
        ):
            is_error = True
            if not mcp_is_error:
                source = "drive_handler_error"
            message = stripped
            break
        elif server_name == "gym-calendar" and _CALENDAR_PLAIN_ERROR.search(stripped):
            is_error = True
            if not mcp_is_error:
                source = "calendar_plain_error"
            message = stripped
            break

    return {
        "is_error": is_error,
        "outcome": {
            "transport_ok": True,
            "application_ok": not is_error,
            "source": source,
            "mcp_is_error": mcp_is_error,
            "status_code": status_code,
            "message": message,
        },
        "result": result,
    }


def _context_headers(context: Mapping[str, Any]) -> dict[str, str]:
    headers: dict[str, str] = {}
    for key, value in context.items():
        header = (
            key
            if key.lower().startswith("x-")
            else f"x-{key.lower().replace('_', '-')}"
        )
        headers[header] = str(value)
    return headers


def extract_sql_value(api_result: Any) -> Any:
    """Match the released EnterpriseOps database-state value extraction."""
    if not isinstance(api_result, dict):
        return api_result
    if "data" in api_result:
        rows = api_result["data"]
    elif "rows" in api_result:
        rows = api_result["rows"]
    elif "result" in api_result:
        return api_result["result"]
    elif "content" in api_result:
        content = api_result["content"]
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    return item.get("text")
        return content
    else:
        return api_result

    if not isinstance(rows, list) or len(rows) != 1:
        return rows
    row = rows[0]
    if isinstance(row, dict) and len(row) == 1:
        return next(iter(row.values()))
    if isinstance(row, list) and len(row) == 1:
        return row[0]
    return row


def compare_verifier_value(actual: Any, expected: Any, comparison_type: str) -> bool:
    if comparison_type == "equals":
        return actual == expected
    if comparison_type == "greater_than":
        return actual > expected
    if comparison_type == "less_than":
        return actual < expected
    if comparison_type == "contains":
        return str(expected) in str(actual)
    raise InfrastructureError(f"unsupported verifier comparison: {comparison_type}")


@dataclass
class EnterpriseOpsEnvironment:
    server_name: str
    base_url: str
    mcp_endpoint: str
    seed_database_file: str
    context: Mapping[str, Any]
    database_archive: Path

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")
        self.database_id: str | None = None
        self.mcp_session_id: str | None = None
        self._request_id = 0
        # Pinned MCP servers can reset a reused HTTP/1.1 connection after
        # concurrent blocking seed requests. A real 8-way Teams probe reproduced
        # ECONNRESET on pooled connections; fresh connections preserved all states.
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(30.0), trust_env=False,
            limits=httpx.Limits(max_keepalive_connections=0),
        )

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self.mcp_session_id:
            headers["mcp-session-id"] = self.mcp_session_id
        if self.database_id:
            headers["x-database-id"] = self.database_id
        headers.update(_context_headers(self.context))
        return headers

    async def seed(self, sql_content: str | None = None) -> str:
        restoring = sql_content is not None
        if sql_content is None:
            try:
                with ZipFile(self.database_archive) as archive:
                    sql_content = archive.read(self.seed_database_file).decode("utf-8")
            except (KeyError, OSError, UnicodeDecodeError) as exc:
                raise InfrastructureError(
                    f"cannot read database seed {self.seed_database_file!r}: {exc}"
                ) from exc

        else:
            from robustops.envs.enterpriseops.sqlite_snapshot import transport_sql
            sql_content = transport_sql(sql_content)

        database_id = f"robustops_{uuid4().hex[:16]}"
        # The server can allocate the database before rejecting seed SQL.
        # Keep ownership early so finally/cleanup can delete that partial DB.
        self.database_id = database_id
        payload = {
            "database_id": database_id,
            "name": f"RobustOps clean {datetime.now(timezone.utc).isoformat()}",
            "description": f"EXP-0001 seed from {Path(self.seed_database_file).name}",
            "sql_content": sql_content,
        }
        try:
            endpoint = "restore-database" if restoring and self.server_name == "gym-calendar" else "seed-database"
            response = await self._client.post(
                f"{self.base_url}/api/{endpoint}",
                json=payload,
                headers={"Content-Type": "application/json"},
                timeout=httpx.Timeout(1200.0),
            )
            response.raise_for_status()
        except (httpx.HTTPError, ValueError) as exc:
            raise InfrastructureError(f"database seed failed on {self.server_name}: {type(exc).__name__}: {exc!r}") from exc
        self.database_id = database_id
        return database_id

    async def export_seed_state(self) -> str:
        """Capture the actual initialized DB, including evaluated seed timestamps.

        Called before model execution on this isolated database. Checkpoint first
        so the native download includes committed WAL contents.
        """
        import sqlite3
        import tempfile
        async def download():
            for attempt in range(2):
                try:
                    return await self._client.get(
                        f"{self.base_url}/api/download-db-file", headers=self._headers())
                except httpx.TransportError as exc:
                    logging.getLogger(__name__).warning(
                        "Snapshot download failure server=%s database=%s attempt=%s: %s %r",
                        self.server_name, self.database_id, attempt + 1, type(exc).__name__, exc)
                    if attempt == 1:
                        raise
        try:
            response = await download()
            response.raise_for_status()
            if not response.content.startswith(b"SQLite format 3\x00"):
                raise InfrastructureError("seed snapshot endpoint did not return SQLite")
            # Rollback-journal databases already have committed contents in this
            # file. Only WAL databases need a checkpoint before the download.
            if response.content[18:20] == b"\x02\x02":
                checkpoint = await self._client.post(
                    f"{self.base_url}/api/sql-runner",
                    json={"query": "PRAGMA wal_checkpoint(TRUNCATE)", "database_id": self.database_id},
                    headers=self._headers())
                checkpoint.raise_for_status()
                response = await download()
                response.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".db") as saved:
                saved.write(response.content)
                saved.flush()
                connection = sqlite3.connect(saved.name)
                try:
                    if connection.execute("PRAGMA integrity_check").fetchone() != ("ok",):
                        raise InfrastructureError("invalid SQLite seed snapshot")
                    objects = connection.execute("SELECT type,name FROM sqlite_master WHERE type IN ('view','table') AND name NOT LIKE 'sqlite_%' ORDER BY type DESC,name").fetchall()
                    drops = ['DROP ' + kind.upper() + ' IF EXISTS "' + name.replace('"', '""') + '";' for kind, name in objects]
                    return "PRAGMA foreign_keys=OFF;\n" + "\n".join(drops) + "\n" + "\n".join(connection.iterdump())
                finally:
                    connection.close()
        except (httpx.HTTPError, sqlite3.Error, OSError) as exc:
            raise InfrastructureError(
                f"seed snapshot failed on {self.server_name} database={self.database_id}: "
                f"{type(exc).__name__}: {exc!r}") from exc

    async def _mcp_request(
        self, method: str, params: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        self._request_id += 1
        payload = {
            "jsonrpc": "2.0",
            "id": self._request_id,
            "method": method,
            "params": dict(params),
        }
        try:
            response = await self._client.post(
                f"{self.base_url}{self.mcp_endpoint}",
                json=payload,
                headers=self._headers(),
            )
            response.raise_for_status()
            if session_id := response.headers.get("mcp-session-id"):
                self.mcp_session_id = session_id
            body = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise InfrastructureError(f"MCP {method} transport failed: {exc}") from exc
        if not isinstance(body, dict) or body.get("jsonrpc") != "2.0":
            raise InfrastructureError(
                f"MCP {method} returned an invalid JSON-RPC response"
            )
        return body

    async def connect(self) -> None:
        body = await self._mcp_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {"tools": {}},
                "clientInfo": {"name": "robustops", "version": "0.1.0"},
            },
        )
        if "error" in body or "result" not in body:
            raise InfrastructureError(f"MCP initialize failed: {body.get('error')!r}")
        notification = {
            "jsonrpc": "2.0",
            "method": "notifications/initialized",
            "params": {},
        }
        try:
            response = await self._client.post(
                f"{self.base_url}{self.mcp_endpoint}",
                json=notification,
                headers=self._headers(),
            )
            if response.status_code not in {200, 204}:
                raise InfrastructureError(
                    f"MCP initialized notification failed: HTTP {response.status_code}"
                )
        except httpx.HTTPError as exc:
            raise InfrastructureError(
                f"MCP initialized notification failed: {exc}"
            ) from exc

    async def list_tools(self) -> list[Mapping[str, Any]]:
        body = await self._mcp_request("tools/list", {})
        if "error" in body:
            raise InfrastructureError(f"MCP tools/list failed: {body['error']!r}")
        result = body.get("result")
        tools = result.get("tools") if isinstance(result, dict) else None
        if not isinstance(tools, list) or not all(
            isinstance(tool, dict) for tool in tools
        ):
            raise InfrastructureError("MCP tools/list response lacks a tool list")
        return tools

    async def call_tool(
        self, name: str, arguments: Mapping[str, Any]
    ) -> Mapping[str, Any]:
        body = await self._mcp_request(
            "tools/call", {"name": name, "arguments": dict(arguments)}
        )
        if "error" in body:
            return {
                "is_error": True,
                "outcome": {
                    "transport_ok": True,
                    "application_ok": False,
                    "source": "json_rpc_error",
                    "mcp_is_error": None,
                    "status_code": None,
                    "message": str(body["error"]),
                },
                "error": body["error"],
            }
        if "result" not in body:
            raise InfrastructureError("MCP tools/call response lacks result/error")
        result = body["result"]
        if not isinstance(result, Mapping):
            raise InfrastructureError("MCP tools/call result must be an object")
        return normalize_mcp_tool_result(self.server_name, result, tool_name=name, arguments=arguments)

    async def run_verifier(
        self, verifier: Mapping[str, Any], index: int
    ) -> Mapping[str, Any]:
        if verifier.get("gym_name") != self.server_name:
            raise InfrastructureError(
                f"verifier {index} targets {verifier.get('gym_name')!r}, expected {self.server_name!r}"
            )
        validation = verifier["validation_config"]
        payload = {"query": validation["query"], "database_id": self.database_id}
        request_failures = []
        # Verifier SELECTs read the same terminal state; retry only transport
        # failures, never tool mutations, SQL errors or malformed responses.
        attempts = 2 if validation['query'].lstrip().upper().startswith('SELECT') else 1
        for attempt in range(attempts):
            try:
                response = await self._client.post(
                    f"{self.base_url}/api/sql-runner",
                    json=payload,
                    headers=self._headers(),
                )
                response.raise_for_status()
                api_result = response.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                details = dict(type=type(exc).__name__, message=repr(exc), attempt=attempt + 1)
                request_failures.append(details)
                logging.getLogger(__name__).warning(
                    "Verifier request failure server=%s database=%s verifier=%s: %s",
                    self.server_name, self.database_id, index, details,
                )
                if isinstance(exc, httpx.TransportError) and attempt + 1 < attempts:
                    continue
                raise InfrastructureError(
                    f"verifier {index} SQL request failed on {self.server_name} "
                    f"database={self.database_id}: {request_failures!r}"
                ) from exc
        actual = extract_sql_value(api_result)
        comparison = validation.get("comparison_type", "equals")
        try:
            passed = compare_verifier_value(
                actual, validation["expected_value"], comparison
            )
        except (TypeError, ValueError) as exc:
            raise InfrastructureError(
                f"verifier {index} comparison failed: {exc}"
            ) from exc
        return {
            "index": index,
            "name": verifier.get("name") or f"verifier_{index}",
            "request_failures": request_failures,
            "gym_name": self.server_name,
            "passed": passed,
            "comparison_type": comparison,
            "expected": validation["expected_value"],
            "actual": actual,
        }

    async def cleanup(self) -> tuple[bool, str | None]:
        if not self.database_id:
            await self._client.aclose()
            return True, None
        try:
            response = await self._client.request(
                "DELETE",
                f"{self.base_url}/api/delete-database",
                json={"database_id": self.database_id},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            return True, None
        except httpx.HTTPError as exc:
            return False, str(exc)
        finally:
            await self._client.aclose()
