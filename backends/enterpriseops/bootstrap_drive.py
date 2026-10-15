#!/usr/bin/env python3
"""Bootstrap the pinned Drive MCP with strict schemas and atomic updates."""

from __future__ import annotations

import json
import os
import re
from datetime import UTC, date, datetime
from functools import wraps
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
    root_validator,
)

_EMAIL_VALUE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_INTEGER_VALUE = re.compile(r"^[+-]?\d+$")

_PUBLIC_UPDATE_FILES_FIELDS = frozenset(
    {
        "fileId",
        "name",
        "description",
        "mimeType",
        "starred",
        "trashed",
        "properties",
        "appProperties",
        "addParents",
        "removeParents",
        "supportsAllDrives",
        "includePermissionsForView",
        "includeLabels",
        # Added below by this compatibility layer to make the image's existing
        # validated ContentHint type reachable through the advertised catalog.
        "contentHints",
    }
)
_UPDATE_FILES_BODY_FIELDS = (
    "name",
    "description",
    "mimeType",
    "starred",
    "trashed",
    "properties",
    "appProperties",
)
_UPDATE_FILES_PARENT_FIELDS = ("addParents", "removeParents")


class StrictFieldModification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    fieldId: str = Field(min_length=1)
    setDateValues: list[str] | None = Field(None, min_length=1, max_length=1)
    setTextValues: list[str] | None = Field(None, min_length=1, max_length=1)
    setSelectionValues: list[str] | None = Field(None, min_length=1, max_length=1)
    setIntegerValues: list[str] | None = Field(None, min_length=1, max_length=1)
    setUserValues: list[str] | None = Field(None, min_length=1, max_length=1)
    unsetValues: bool | None = None
    kind: Literal["drive#labelFieldModification"] = "drive#labelFieldModification"

    @field_validator(
        "setDateValues",
        "setTextValues",
        "setSelectionValues",
        "setIntegerValues",
        "setUserValues",
    )
    @classmethod
    def values_must_be_nonempty(cls, values: list[str] | None) -> list[str] | None:
        if values is not None and any(not value.strip() for value in values):
            raise ValueError("setter values must be non-empty strings")
        return values

    @field_validator("setDateValues")
    @classmethod
    def dates_are_full_dates(cls, values: list[str] | None) -> list[str] | None:
        if values is not None:
            for value in values:
                if date.fromisoformat(value).isoformat() != value:
                    raise ValueError("date values must use YYYY-MM-DD")
        return values

    @field_validator("setIntegerValues")
    @classmethod
    def numbers_are_numeric(cls, values: list[str] | None) -> list[str] | None:
        if values is not None:
            for value in values:
                if _INTEGER_VALUE.fullmatch(value) is None:
                    raise ValueError("integer values must be decimal integer strings")
                parsed = int(value)
                if not -(2**63) <= parsed < 2**63:
                    raise ValueError("integer values must fit signed int64")
        return values

    @field_validator("setUserValues")
    @classmethod
    def users_are_emails(cls, values: list[str] | None) -> list[str] | None:
        if values is not None:
            for value in values:
                if _EMAIL_VALUE.fullmatch(value) is None:
                    raise ValueError("user values must be valid email addresses")
        return values

    @model_validator(mode="after")
    def exactly_one_operation(self) -> StrictFieldModification:
        operations = (
            self.setDateValues,
            self.setTextValues,
            self.setSelectionValues,
            self.setIntegerValues,
            self.setUserValues,
        )
        count = sum(value is not None for value in operations) + (
            self.unsetValues is True
        )
        if count != 1:
            raise ValueError("exactly one setter or unsetValues=true is required")
        return self


class StrictLabelModification(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labelId: str = Field(min_length=1)
    fieldModifications: list[StrictFieldModification] | None = Field(
        None, min_length=1, max_length=1
    )
    removeLabel: bool | None = None
    kind: Literal["drive#labelModification"] = "drive#labelModification"

    @model_validator(mode="after")
    def remove_or_modify(self) -> StrictLabelModification:
        if (self.removeLabel is True) == (self.fieldModifications is not None):
            raise ValueError(
                "choose exactly one of removeLabel=true or fieldModifications"
            )
        return self


class StrictModifyLabelsRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    labelModifications: list[StrictLabelModification] = Field(min_length=1)

    @model_validator(mode="after")
    def unique_labels(self) -> StrictModifyLabelsRequest:
        ids = [item.labelId for item in self.labelModifications]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate labelId values are not allowed")
        return self


class StrictModifyFilesLabelsArgs(StrictModifyLabelsRequest):
    fileId: str = Field(min_length=1)
    database_id: str | None = None
    auth_token: str | None = None


def strict_model_subclasses(
    drive_module: Any, files_module: Any, watch_module: Any
) -> tuple[type[Any], type[Any], type[Any]]:
    """Build strict subclasses while retaining the image's existing validators."""

    class StrictBackgroundImageFile(drive_module.BackgroundImageFile):
        model_config = ConfigDict(extra="forbid")

    class StrictRestrictions(drive_module.Restrictions):
        model_config = ConfigDict(extra="forbid")

    class StrictUpdateDriveArgs(drive_module.UpdateDriveArgs):
        model_config = ConfigDict(extra="forbid")
        backgroundImageFile: StrictBackgroundImageFile | None = None
        restrictions: StrictRestrictions | None = None

    class StrictContentHint(files_module.ContentHint):
        model_config = ConfigDict(extra="forbid")

    class StrictUpdateFilesArgs(files_module.UpdateFilesArgs):
        model_config = ConfigDict(extra="forbid")
        contentHints: StrictContentHint | None = None

        @model_validator(mode="before")
        @classmethod
        def reject_unadvertised_inherited_fields(cls, value: Any) -> Any:
            # The image's internal model declares fields that its public MCP
            # schema does not advertise. Inheriting that model would otherwise
            # silently make those hidden fields callable.
            if isinstance(value, dict):
                unexpected = sorted(set(value) - _PUBLIC_UPDATE_FILES_FIELDS)
                if unexpected:
                    raise ValueError(
                        "update_files arguments are not exposed by the public "
                        f"tool contract: {', '.join(unexpected)}"
                    )
            return value

        @root_validator(pre=False, skip_on_failure=True)
        def check_at_least_one_update_field(cls, values: dict[str, Any]):
            # Override the image validator of the same name: Google Drive moves
            # are valid PATCH requests with only addParents/removeParents in the
            # query and an empty metadata body.
            has_body_update = any(
                values.get(field) is not None
                for field in (*_UPDATE_FILES_BODY_FIELDS, "contentHints")
            )
            has_parent_update = any(
                isinstance(values.get(field), str) and values[field].strip()
                for field in _UPDATE_FILES_PARENT_FIELDS
            )
            if not (has_body_update or has_parent_update):
                raise ValueError(
                    "At least one public metadata field or non-empty "
                    "addParents/removeParents value must be provided"
                )
            return values

    class StrictWatchFilesArgs(watch_module.WatchFilesArgs):
        model_config = ConfigDict(extra="forbid")
        id: str = Field(min_length=1)
        address: str = Field(min_length=1)
        type: Literal["web_hook", "webhook"] = "web_hook"

        @field_validator("expiration")
        @classmethod
        def expiration_is_future_ms(cls, value: str | None) -> str | None:
            if value is None:
                return value
            if not value.isdigit() or int(value) <= 0:
                raise ValueError(
                    "expiration must be a positive Unix millisecond string"
                )
            if int(value) <= int(datetime.now(UTC).timestamp() * 1000):
                raise ValueError("expiration must be in the future")
            return value

    return StrictUpdateDriveArgs, StrictUpdateFilesArgs, StrictWatchFilesArgs


if __package__:
    from .backend_errors import json_safe_validation_errors
else:
    from backend_errors import json_safe_validation_errors


def build_validation_exception_handler() -> Any:
    from fastapi.encoders import jsonable_encoder
    from fastapi.responses import JSONResponse

    async def validation_exception_handler(request: Any, exc: Any) -> Any:
        errors = json_safe_validation_errors(exc.errors())
        empty_labels = any(
            "labelmodifications" in str(error.get("loc", [])).lower()
            and (
                "empty" in str(error.get("msg", "")).lower()
                or error.get("type") == "too_short"
            )
            for error in errors
        )
        status_code = 400 if empty_labels else 422
        if empty_labels:
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": 400,
                        "message": "labelModifications array cannot be empty",
                    }
                },
            )
        try:
            body = jsonable_encoder(exc.body)
        except Exception:  # noqa: BLE001 - malformed request bodies are untrusted objects
            body = str(exc.body)
        return JSONResponse(
            status_code=status_code,
            content={"detail": errors, "body": body, "message": "Validation failed"},
        )

    return validation_exception_handler


def patch_tool_catalog(tools: list[Any]) -> None:
    by_name = {tool.name: tool for tool in tools}
    required = {"update_files", "modify_files_labels", "watch_files"}
    if not required.issubset(by_name):
        raise RuntimeError("pinned Drive tool catalog is missing contract tools")

    update_schema = by_name["update_files"].inputSchema
    if "contentHints" in update_schema["properties"]:
        raise RuntimeError("pinned Drive update_files schema unexpectedly changed")
    update_schema["properties"]["contentHints"] = {
        "type": "object",
        "properties": {
            "indexableText": {"type": "string"},
            "thumbnail": {"type": "object"},
        },
        "additionalProperties": False,
    }

    watch_schema = by_name["watch_files"].inputSchema
    if set(watch_schema.get("required", ())) != {
        "fileId",
        "id",
        "resourceId",
        "resourceUri",
        "address",
    }:
        raise RuntimeError("pinned Drive watch_files schema unexpectedly changed")
    watch_schema["required"] = ["fileId", "id", "address"]
    watch_schema["properties"]["expiration"]["description"] = (
        "Future Unix timestamp in milliseconds, encoded as a decimal string (optional)"
    )
    watch_schema["properties"]["expiration"]["examples"] = ["1893456000000"]

    labels = by_name["modify_files_labels"].inputSchema
    labels["properties"]["labelModifications"]["minItems"] = 1
    label = labels["properties"]["labelModifications"]["items"]
    label["properties"]["fieldModifications"]["minItems"] = 1
    label["properties"]["fieldModifications"]["maxItems"] = 1
    field = label["properties"]["fieldModifications"]["items"]
    for setter in (
        "setDateValues",
        "setTextValues",
        "setSelectionValues",
        "setIntegerValues",
        "setUserValues",
    ):
        field["properties"][setter]["minItems"] = 1
        field["properties"][setter]["maxItems"] = 1


def build_update_files_handler(module: Any) -> Any:
    original = module._handle_update_files

    @wraps(original)
    async def update_files(
        arguments: dict[str, Any], database_id: str, auth_token: str
    ) -> Any:
        # These two fields are injected by the MCP transport, not part of the
        # public update_files contract guarded by extra="forbid".
        public_arguments = {
            key: value
            for key, value in arguments.items()
            if key not in {"database_id", "auth_token"}
        }
        validated, error = module.validate_arguments(
            public_arguments, module.UpdateFilesArgs
        )
        if error:
            return error
        url = f"{module.settings.FASTAPI_BASE_URL}/api/v1/files/{validated.fileId}"
        headers = {"X-Database-ID": database_id, "X-Auth-Token": auth_token}
        params = {}
        for field in (
            "addParents",
            "removeParents",
            "includePermissionsForView",
            "includeLabels",
        ):
            value = getattr(validated, field)
            if value:
                params[field] = value
        if validated.supportsAllDrives is not None:
            params["supportsAllDrives"] = validated.supportsAllDrives
        body = {
            field: getattr(validated, field)
            for field in _UPDATE_FILES_BODY_FIELDS
            if getattr(validated, field) is not None
        }
        if validated.contentHints is not None:
            body["contentHints"] = validated.contentHints.model_dump(
                mode="json", exclude_none=True
            )
        response = await module.make_api_request(
            "PATCH", url, headers=headers, params=params, json_data=body
        )
        if response.status_code == 200:
            payload = {
                "success": True,
                "data": response.json(),
                "status": "success",
                "message": "File updated successfully",
            }
        else:
            payload = {
                "success": False,
                "error": await module.handle_http_error(response),
                "status": "error",
                "message": "Error updating file",
            }
        return [module.TextContent(type="text", text=json.dumps(payload, indent=2))]

    return update_files


def build_parent_aware_files_update_request(module: Any) -> type[Any]:
    """Allow the official API route's metadata body to be empty for moves."""

    class ParentAwareFilesUpdateRequest(module.FilesUpdateRequest):
        def __init__(self, **data: Any) -> None:
            # The image's custom __init__ rejects {}, although Drive's official
            # addParents/removeParents move request has no metadata body. Keep
            # normal Pydantic field validation while bypassing only that check.
            BaseModel.__init__(self, **data)

    ParentAwareFilesUpdateRequest.__name__ = "FilesUpdateRequest"
    ParentAwareFilesUpdateRequest.__qualname__ = "FilesUpdateRequest"
    return ParentAwareFilesUpdateRequest


def build_parent_aware_update_endpoint(module: Any) -> Any:
    """Permit parent-only moves but still reject a genuinely empty update."""

    original = module.update_files

    @wraps(original)
    def update_files(*args: Any, **kwargs: Any) -> Any:
        update_data = kwargs.get("update_data")
        fields_set = getattr(update_data, "model_fields_set", set())
        has_metadata_update = any(
            getattr(update_data, field, None) is not None for field in fields_set
        )
        has_parent_update = any(
            isinstance(kwargs.get(field), str) and kwargs[field].strip()
            for field in _UPDATE_FILES_PARENT_FIELDS
        )
        if not (has_metadata_update or has_parent_update):
            raise module.HTTPException(
                status_code=400,
                detail=(
                    "At least one metadata field or non-empty "
                    "addParents/removeParents value must be provided"
                ),
            )
        return original(*args, **kwargs)

    return update_files


def build_modify_labels_handler(module: Any) -> Any:
    original = module.handle_files_labels_tools

    @wraps(original)
    async def modify_labels(name: str, arguments: dict[str, Any]) -> Any:
        if name != "modify_files_labels":
            return await original(name, arguments)
        database_id = arguments.get("database_id")
        auth_token = arguments.get("auth_token")
        if not database_id:
            database_id = await module.get_or_create_default_session()
        validated, error = module.validate_arguments(
            arguments, module.ModifyFilesLabelsArgs
        )
        if error:
            return error
        url = (
            f"{module.settings.FASTAPI_BASE_URL}/api/v1/files/"
            f"{validated.fileId}/modifyLabels"
        )
        headers = {
            "X-Database-ID": database_id,
            "X-Auth-Token": auth_token,
            "Content-Type": "application/json",
        }
        body = validated.model_dump(mode="json", exclude_none=True)
        response = await module.http_client.post(
            url,
            headers=headers,
            json={"labelModifications": body["labelModifications"]},
        )
        if response.status_code == 200:
            payload = {
                "success": True,
                "data": response.json(),
                "status": "success",
                "message": "File labels modified successfully",
            }
        else:
            payload = {
                "success": False,
                "error": await module.handle_http_error(response),
                "status": "error",
                "message": "Error modifying file labels",
            }
        return [module.TextContent(type="text", text=json.dumps(payload, indent=2))]

    return modify_labels


def _label_response(label: Any) -> dict[str, Any]:
    result = {
        "id": label.id,
        "key": label.key,
        "name": label.name,
        "type": label.type.value if label.type else None,
        "required": label.required,
        "created_time": label.created_time.isoformat() if label.created_time else None,
    }
    if label.options:
        result["options"] = label.options
    return result


def build_atomic_modify_labels_endpoint(module: Any) -> Any:
    """Validate the full request before making a single transactional change."""

    def invalid(message: str) -> None:
        raise module.HTTPException(status_code=400, detail=message)

    def modify_files_labels(
        fileId: str,
        request: StrictModifyLabelsRequest,
        current_user: Any,
        db: Any,
    ) -> dict[str, Any]:
        try:
            file_id = fileId.strip()
            if not file_id:
                invalid("File ID is required")
            file = db.query(module.File).filter(module.File.id == file_id).first()
            if not file:
                raise module.HTTPException(
                    status_code=404, detail=f"File '{file_id}' not found"
                )
            if not module.get_user_file_permission(current_user, file, db):
                raise module.HTTPException(status_code=403, detail="Access denied")

            prepared = []
            for change in request.labelModifications:
                label = (
                    db.query(module.Label)
                    .filter(module.Label.id == change.labelId)
                    .first()
                )
                if not label:
                    invalid(f"Label '{change.labelId}' does not exist")
                existing = (
                    db.query(module.FileLabel)
                    .filter(
                        module.and_(
                            module.FileLabel.file_id == file_id,
                            module.FileLabel.label_id == change.labelId,
                        )
                    )
                    .first()
                )
                if change.removeLabel is True:
                    if existing is None:
                        invalid(f"Label '{change.labelId}' is not associated with file")
                    prepared.append((label, existing, "remove", None))
                    continue
                field = change.fieldModifications[0]
                if field.fieldId != label.key:
                    invalid(
                        f"fieldId '{field.fieldId}' does not match label field "
                        f"'{label.key}'"
                    )
                if field.unsetValues is True:
                    if existing is None:
                        invalid(f"Label '{change.labelId}' is not associated with file")
                    prepared.append((label, existing, "remove", None))
                    continue
                operation_map = {
                    "setTextValues": ("string", "value_string", field.setTextValues),
                    "setIntegerValues": (
                        "number",
                        "value_number",
                        field.setIntegerValues,
                    ),
                    "setDateValues": ("date", "value_date", field.setDateValues),
                    "setSelectionValues": (
                        "enum",
                        "value_enum",
                        field.setSelectionValues,
                    ),
                    "setUserValues": ("string", "value_string", field.setUserValues),
                }
                selected = next(
                    (item for item in operation_map.items() if item[1][2] is not None),
                    None,
                )
                if selected is None:
                    invalid("field modification has no operation")
                _, (expected_type, column, values) = selected
                if label.type.value != expected_type:
                    invalid(
                        f"setter does not match label '{change.labelId}' type {label.type.value}"
                    )
                value: Any = values[0]
                if column == "value_number":
                    value = int(value)
                elif column == "value_date":
                    value = datetime.combine(
                        date.fromisoformat(value), datetime.min.time()
                    )
                if (
                    column == "value_enum"
                    and label.options
                    and value not in label.options
                ):
                    invalid(
                        f"selection '{value}' is not valid for label '{change.labelId}'"
                    )
                prepared.append((label, existing, column, value))

            modified = []
            columns = (
                "value_string",
                "value_number",
                "value_boolean",
                "value_date",
                "value_enum",
            )
            for label, row, operation, value in prepared:
                if operation == "remove":
                    db.delete(row)
                else:
                    if row is None:
                        row = module.FileLabel(
                            id=module.generate_uuid(),
                            file_id=file_id,
                            label_id=label.id,
                        )
                        db.add(row)
                    for column in columns:
                        setattr(row, column, value if column == operation else None)
                modified.append(_label_response(label))
            db.commit()
            return {
                "success": True,
                "data": {
                    "modifiedLabels": modified,
                    "kind": "drive#modifyLabelsResponse",
                },
                "message": f"Modified {len(modified)} labels for file '{file_id}' successfully",
            }
        except module.HTTPException:
            db.rollback()
            raise
        except Exception as exc:
            db.rollback()
            raise module.HTTPException(
                status_code=500, detail=f"Failed to modify file labels: {exc}"
            ) from exc

    return modify_files_labels


def _replace_named_endpoint(routes: list[Any], name: str, replacement: Any) -> int:
    count = 0
    for route in routes:
        if getattr(getattr(route, "endpoint", None), "__name__", None) != name:
            continue
        route.endpoint = replacement
        route.dependant.call = replacement
        count += 1
    return count


def build_shortcut_copy_endpoint(module: Any) -> Any:
    """Preserve the source shortcut flag in the pinned native copy constructor."""
    import ast
    import inspect
    import textwrap

    original = module.copy_file
    tree = ast.parse(textwrap.dedent(inspect.getsource(original)))
    function = tree.body[0]
    function.decorator_list = []
    constructors = [node for node in ast.walk(function)
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                    and node.func.id == 'File']
    if len(constructors) != 1:
        raise RuntimeError('pinned Drive copy File constructor drifted')
    constructor = constructors[0]
    keys = {kw.arg for kw in constructor.keywords}
    if 'is_shortcut' in keys or not {'shortcut_target_id', 'shortcut_target_mime'} <= keys:
        raise RuntimeError('pinned Drive copy shortcut fields drifted')
    constructor.keywords.append(ast.keyword(
        arg='is_shortcut', value=ast.Attribute(
            value=ast.Name(id='source_file', ctx=ast.Load()),
            attr='is_shortcut', ctx=ast.Load())))
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, inspect.getfile(original), 'exec'), original.__globals__, namespace)
    return wraps(original)(namespace[original.__name__])


def main() -> None:
    import importlib

    from fastapi.exceptions import RequestValidationError

    drive_validation = importlib.import_module("app.schemas.tools.drive_validation")
    files_validation = importlib.import_module("app.schemas.tools.files_validation")
    watch_validation = importlib.import_module("app.schemas.tools.file_validation")
    labels_validation = importlib.import_module(
        "app.schemas.tools.files_labels_validation"
    )
    api_labels_schema = importlib.import_module("app.schemas.files_labels_schema")
    api_files_schema = importlib.import_module("app.schemas.files_schema")
    strict_drive, strict_files, strict_watch = strict_model_subclasses(
        drive_validation, files_validation, watch_validation
    )
    drive_validation.UpdateDriveArgs = strict_drive
    files_validation.UpdateFilesArgs = strict_files
    watch_validation.WatchFilesArgs = strict_watch
    labels_validation.ModifyFilesLabelsArgs = StrictModifyFilesLabelsArgs
    api_labels_schema.ModifyLabelsRequest = StrictModifyLabelsRequest
    api_files_schema.FilesUpdateRequest = build_parent_aware_files_update_request(
        api_files_schema
    )

    drive_main = importlib.import_module("app.main")
    mcp_server = importlib.import_module("app.mcp_server")
    drive_handlers = importlib.import_module("app.mcp_tool_handler.drives_handlers")
    files_handlers = importlib.import_module("app.mcp_tool_handler.files_handlers")
    watch_handlers = importlib.import_module("app.mcp_tool_handler.file_handlers")
    labels_handlers = importlib.import_module(
        "app.mcp_tool_handler.files_labels_handlers"
    )
    labels_router = importlib.import_module("app.api.files_labels_router")
    files_router = importlib.import_module("app.api.files_router")
    models = importlib.import_module("app.models.google_drive_model")

    drive_handlers.UpdateDriveArgs = strict_drive
    files_handlers.UpdateFilesArgs = strict_files
    watch_handlers.WatchFilesArgs = strict_watch
    labels_handlers.ModifyFilesLabelsArgs = StrictModifyFilesLabelsArgs
    files_handlers._handle_update_files = build_update_files_handler(files_handlers)
    labels_handlers.handle_files_labels_tools = build_modify_labels_handler(
        labels_handlers
    )
    # The root dispatcher imported this symbol by value.
    importlib.import_module(
        "app.mcp_tool_handler.handler"
    ).handle_files_labels_tools = labels_handlers.handle_files_labels_tools

    original_get_tools = mcp_server.get_tools_list

    @wraps(original_get_tools)
    def get_tools_list() -> list[Any]:
        tools = original_get_tools()
        patch_tool_catalog(tools)
        return tools

    mcp_server.get_tools_list = get_tools_list
    drive_main.app.exception_handlers[RequestValidationError] = (
        build_validation_exception_handler()
    )

    labels_router.generate_uuid = models.generate_uuid
    replacement = build_atomic_modify_labels_endpoint(labels_router)
    if (
        _replace_named_endpoint(
            drive_main.app.routes, "modify_files_labels", replacement
        )
        != 1
    ):
        raise RuntimeError("pinned Drive modify_files_labels route drifted")
    labels_router.modify_files_labels = replacement

    parent_aware_update = build_parent_aware_update_endpoint(files_router)
    if (
        _replace_named_endpoint(
            drive_main.app.routes, "update_files", parent_aware_update
        )
        != 1
    ):
        raise RuntimeError("pinned Drive update_files route drifted")
    files_router.update_files = parent_aware_update

    shortcut_copy = build_shortcut_copy_endpoint(files_router)
    if _replace_named_endpoint(drive_main.app.routes, 'copy_file', shortcut_copy) != 1:
        raise RuntimeError('pinned Drive copy_file route drifted')
    files_router.copy_file = shortcut_copy

    import uvicorn

    uvicorn.run(
        drive_main.app,
        host="0.0.0.0",
        port=int(os.environ.get("MCP_SERVER_PORT", "8005")),
    )


if __name__ == "__main__":
    main()
