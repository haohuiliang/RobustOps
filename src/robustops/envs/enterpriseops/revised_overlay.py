"""Apply the pinned upstream EnterpriseOps revised-task overlay.

The upstream repository publishes complete Oracle task JSON files under
``data/revised`` but its evaluator does not load them.  This module projects
only fields that actually changed relative to the fixed public Oracle row onto
all matching tool modes.  Mode-specific ``selected_tools`` therefore remain
untouched.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq


OFFICIAL_CODE_REVISION = "271f2c357f763376997dfd16807fcde2474ae41b"
OFFICIAL_REVISED_COMMIT = "724dd34c09fa2bb6bdb76f8b8d5975350d648732"

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_SOURCE_ROOT = PROJECT_ROOT / "data/raw/EnterpriseOps-Gym"
DEFAULT_REVISED_ROOT = PROJECT_ROOT / "enterpriseops_gym/data/revised"

OFFICIAL_REVISED_TASKS: Mapping[str, str] = {
    "task_20251205_153330_906_a8eea1c0_8c7a6205": "csm",
    "task_20251205_154853_044_d4a463c3_fce76046": "csm",
    "task_20251209_132736_542_99ba2325_6705caf4": "csm",
    "task_20251210_002938_761_897c3e0b_f075229a": "csm",
    "task_20251210_113549_827_d4a463c3_47b48a98": "csm",
    "task_20251231_162628_797_a8eea1c0_0a53db6c": "csm",
    "task_20260101_043715_660_a8eea1c0_7304e48e": "csm",
    "task_20260101_122222_300_ad5a67e3_3757206c": "csm",
    "task_20260102_155148_014_accab84d_910b176a": "csm",
    "task_20260102_161510_799_ad5a67e3_7221f3c8": "csm",
    "task_20260102_172945_725_ad5a67e3_5480e26f": "csm",
    "task_20260104_202644_765_00364ece_a86667c0": "csm",
    "task_20251212_172511_458_e6427839_47076c15": "itsm",
}

# Complete upstream revised tasks that are absent from every mode of the fixed
# 649-row public Parquet snapshot and remain executable from published assets.
# They are retained as an official supplement rather than copied into or
# presented as modifications of the raw Parquet.
OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS: Mapping[str, Mapping[str, Any]] = {
    "task_20260102_161510_799_ad5a67e3_7221f3c8": {
        "domain": "csm",
        "runnable": True,
        "blocked_reason": None,
    },
    "task_20260104_202644_765_00364ece_a86667c0": {
        "domain": "csm",
        "runnable": True,
        "blocked_reason": None,
    },
}

# Published revised-only definitions that cannot enter the effective benchmark
# because a required official asset is absent.  Keep them in the source audit,
# but never load them as tasks and never assign them a benchmark partition.
OFFICIAL_REVISED_EXCLUDED_TASKS: Mapping[str, Mapping[str, str]] = {
    "task_20251209_132736_542_99ba2325_6705caf4": {
        "domain": "csm",
        "reason": (
            "official seed db_1765219280033_d7pqrz32b.sql is absent from the "
            "published gym_dbs.zip"
        ),
    },
}

OFFICIAL_REVISED_FIELDS = (
    "mcp_endpoint",
    "number_of_runs",
    "reset_database_between_runs",
    "gym_servers_config",
    "system_prompt",
    "user_prompt",
    "selected_tools",
    "restricted_tools",
    "verifiers",
)
_TASK_FIELDS = OFFICIAL_REVISED_FIELDS
_MODE_INVARIANT_FIELDS = (
    "mcp_endpoint",
    "number_of_runs",
    "reset_database_between_runs",
    "gym_servers_config",
    "system_prompt",
    "user_prompt",
    "restricted_tools",
    "verifiers",
)
_JSON_FIELDS = {"gym_servers_config", "verifiers"}
_SOURCE_MODES = ("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools")


class OfficialRevisedOverlayError(ValueError):
    """The local source or upstream revised file drifted from the pinned overlay."""


@dataclass(frozen=True)
class OfficialRevisedOverlay:
    """Provenance for one upstream revised task applied at runtime."""

    task_id: str
    domain: str
    source_mode: str
    relative_path: str
    revised_commit: str
    changed_fields: tuple[str, ...]
    source_verifier_count: int
    revised_verifier_count: int
    already_revised_fields: tuple[str, ...] = ()


def _decoded(value: Any, field: str, *, where: str) -> Any:
    if field not in _JSON_FIELDS or not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise OfficialRevisedOverlayError(
            f"{where}: {field} is not valid JSON: {exc}"
        ) from exc


@lru_cache(maxsize=None)
def _source_mode_by_id(
    mode: str, domain: str, source_root_text: str
) -> Mapping[str, Mapping[str, Any]]:
    path = Path(source_root_text) / mode / f"{domain}-00000-of-00001.parquet"
    if not path.is_file():
        raise OfficialRevisedOverlayError(
            f"pinned source Oracle parquet is missing: {path}"
        )
    rows = pq.read_table(path).to_pylist()
    result = {str(row.get("task_id")): row for row in rows}
    if len(result) != len(rows):
        raise OfficialRevisedOverlayError(f"duplicate task_id in {path}")
    return result


def _matching_source_mode(
    row: Mapping[str, Any], *, task_id: str, domain: str, source_root: Path
) -> tuple[str, Mapping[str, Any]]:
    """Identify the fixed source mode from its mode-defining tool selection."""

    observed_tools = row.get("selected_tools")
    matches: list[tuple[str, Mapping[str, Any]]] = []
    for mode in _SOURCE_MODES:
        source = _source_mode_by_id(
            mode, domain, str(source_root.resolve())
        ).get(task_id)
        if source is not None and source.get("selected_tools") == observed_tools:
            matches.append((mode, source))
    if len(matches) != 1:
        raise OfficialRevisedOverlayError(
            f"{task_id}: selected_tools do not identify exactly one fixed source "
            f"mode; matches={[mode for mode, _ in matches]}"
        )
    return matches[0]


@lru_cache(maxsize=None)
def _revised_task(
    task_id: str, domain: str, revised_root_text: str
) -> Mapping[str, Any]:
    path = Path(revised_root_text) / domain / f"{task_id}.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise OfficialRevisedOverlayError(
            f"cannot load pinned official revised task {path}: {exc}"
        ) from exc
    if not isinstance(value, dict) or set(value) != set(_TASK_FIELDS):
        raise OfficialRevisedOverlayError(
            f"{path}: expected official revised fields {list(_TASK_FIELDS)}, "
            f"found {sorted(value) if isinstance(value, dict) else type(value).__name__}"
        )
    return value


def apply_official_revised_overlay(
    row: Mapping[str, Any],
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    revised_root: Path = DEFAULT_REVISED_ROOT,
) -> tuple[Mapping[str, Any], tuple[OfficialRevisedOverlay, ...]]:
    """Apply a complete upstream revision as a mode-safe field projection.

    The revised files are Oracle-shaped.  A changed mode-invariant field is
    applied to Oracle and distractor variants only when that variant still
    equals the pinned public Oracle source (or already equals the revised
    value).  ``selected_tools`` is intentionally not projected because it is
    the mode-defining field and none of the published Oracle revisions changes
    it relative to the fixed Oracle source.
    """

    task_id = row.get("task_id")
    if not isinstance(task_id, str) or task_id not in OFFICIAL_REVISED_TASKS:
        return row, ()
    domain = OFFICIAL_REVISED_TASKS[task_id]
    if row.get("domain") != domain:
        raise OfficialRevisedOverlayError(
            f"{task_id}: domain no longer matches official revised path {domain!r}"
        )

    oracle_source = _source_mode_by_id(
        "oracle", domain, str(source_root.resolve())
    ).get(task_id)
    if oracle_source is None:
        if task_id not in OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS:
            return row, ()
        revised = _revised_task(task_id, domain, str(revised_root.resolve()))
        mismatches = [
            field
            for field in _TASK_FIELDS
            if _decoded(row.get(field), field, where=f"{task_id}:supplement")
            != _decoded(revised[field], field, where=f"{task_id}:revised")
        ]
        if mismatches:
            raise OfficialRevisedOverlayError(
                f"{task_id}: official revised supplement drifted in {mismatches}"
            )
        revised_verifiers = _decoded(
            revised["verifiers"], "verifiers", where=f"{task_id}:revised"
        )
        return row, (
            OfficialRevisedOverlay(
                task_id=task_id,
                domain=domain,
                source_mode="official_revised_supplement",
                relative_path=f"data/revised/{domain}/{task_id}.json",
                revised_commit=OFFICIAL_REVISED_COMMIT,
                changed_fields=(),
                source_verifier_count=0,
                revised_verifier_count=len(revised_verifiers),
            ),
        )
    revised = _revised_task(task_id, domain, str(revised_root.resolve()))

    changed_fields = tuple(
        field
        for field in _TASK_FIELDS
        if _decoded(oracle_source[field], field, where=f"{task_id}:source")
        != _decoded(revised[field], field, where=f"{task_id}:revised")
    )
    unsupported = set(changed_fields) - set(_MODE_INVARIANT_FIELDS)
    if unsupported:
        raise OfficialRevisedOverlayError(
            f"{task_id}: official revision changes mode-specific field(s) "
            f"{sorted(unsupported)}; explicit mode projection is required"
        )

    source_mode, source = _matching_source_mode(
        row, task_id=task_id, domain=domain, source_root=source_root
    )
    # Compare unchanged fields with the exact fixed source mode.  Some source
    # modes reorder semantically identical verifiers, so the Oracle row itself
    # cannot be used as the byte-for-byte source for every variant.
    for field in _MODE_INVARIANT_FIELDS:
        if field in changed_fields:
            continue
        observed = _decoded(row[field], field, where=f"{task_id}:input")
        expected = _decoded(source[field], field, where=f"{task_id}:source")
        if observed != expected:
            raise OfficialRevisedOverlayError(
                f"{task_id}: unchanged field {field!r} no longer matches the "
                "pinned official source"
            )

    result = dict(row)
    already_revised: list[str] = []
    for field in changed_fields:
        observed = _decoded(row[field], field, where=f"{task_id}:input")
        source_value = _decoded(source[field], field, where=f"{task_id}:source")
        revised_value = _decoded(revised[field], field, where=f"{task_id}:revised")
        if observed == revised_value:
            already_revised.append(field)
        elif observed != source_value:
            raise OfficialRevisedOverlayError(
                f"{task_id}: changed field {field!r} matches neither the pinned "
                "source nor official revised value"
            )
        result[field] = deepcopy(revised_value)

    source_verifiers = _decoded(source["verifiers"], "verifiers", where=task_id)
    revised_verifiers = _decoded(revised["verifiers"], "verifiers", where=task_id)
    relative_path = f"data/revised/{domain}/{task_id}.json"
    return result, (
        OfficialRevisedOverlay(
            task_id=task_id,
            domain=domain,
            source_mode=source_mode,
            relative_path=relative_path,
            revised_commit=OFFICIAL_REVISED_COMMIT,
            changed_fields=changed_fields,
            source_verifier_count=len(source_verifiers),
            revised_verifier_count=len(revised_verifiers),
            already_revised_fields=tuple(already_revised),
        ),
    )


def official_revised_inventory(
    *,
    source_root: Path = DEFAULT_SOURCE_ROOT,
    revised_root: Path = DEFAULT_REVISED_ROOT,
) -> Mapping[str, Any]:
    """Return a deterministic inventory for audit/report generation."""

    expected_paths = {
        revised_root / domain / f"{task_id}.json"
        for task_id, domain in OFFICIAL_REVISED_TASKS.items()
    }
    observed_paths = set(revised_root.glob("*/*.json"))
    if observed_paths != expected_paths:
        raise OfficialRevisedOverlayError(
            "official revised inventory drifted: "
            f"missing={sorted(str(path) for path in expected_paths - observed_paths)}, "
            f"extra={sorted(str(path) for path in observed_paths - expected_paths)}"
        )

    records: list[Mapping[str, Any]] = []
    public_count = 0
    for task_id, domain in sorted(OFFICIAL_REVISED_TASKS.items()):
        source = _source_mode_by_id(
            "oracle", domain, str(source_root.resolve())
        ).get(task_id)
        revised = _revised_task(task_id, domain, str(revised_root.resolve()))
        changed_fields = []
        if source is not None:
            public_count += 1
            changed_fields = [
                field
                for field in _TASK_FIELDS
                if _decoded(source[field], field, where=f"{task_id}:source")
                != _decoded(revised[field], field, where=f"{task_id}:revised")
            ]
        records.append(
            {
                "task_id": task_id,
                "domain": domain,
                "relative_path": f"data/revised/{domain}/{task_id}.json",
                "present_in_fixed_public_oracle": source is not None,
                "changed_fields": changed_fields,
            }
        )
    return {
        "code_revision": OFFICIAL_CODE_REVISION,
        "revised_commit": OFFICIAL_REVISED_COMMIT,
        "published_file_count": len(records),
        "fixed_public_match_count": public_count,
        "fixed_public_missing_count": len(records) - public_count,
        "records": records,
    }


def official_revised_supplement_rows(
    *, domain: str, mode: str = "oracle", revised_root: Path = DEFAULT_REVISED_ROOT
) -> list[Mapping[str, Any]]:
    """Load executable train-only official tasks without altering raw data."""

    if mode != "oracle":
        return []
    rows: list[Mapping[str, Any]] = []
    for task_id, contract in sorted(OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS.items()):
        if contract["domain"] != domain:
            continue
        revised = _revised_task(task_id, domain, str(revised_root.resolve()))
        rows.append({"task_id": task_id, "domain": domain, **deepcopy(revised)})
    return rows


def official_revised_payload(task: Any) -> Mapping[str, Any]:
    """Serialize one effective task into the upstream complete-task JSON shape.

    Upstream identifies ``task_id`` and ``domain`` by the file path, so neither
    belongs in the JSON object.  Runtime-only contracts such as tool routing and
    trajectory verifiers are deliberately not embedded in this payload.
    """

    payload = {
        "mcp_endpoint": task.mcp_endpoint,
        "number_of_runs": task.number_of_runs,
        "reset_database_between_runs": task.reset_database_between_runs,
        "gym_servers_config": deepcopy(list(task.gym_servers)),
        "system_prompt": task.system_prompt,
        "user_prompt": task.user_prompt,
        "selected_tools": list(task.selected_tools),
        "restricted_tools": list(task.restricted_tools),
        "verifiers": deepcopy(list(task.verifiers)),
    }
    if tuple(payload) != OFFICIAL_REVISED_FIELDS:
        raise OfficialRevisedOverlayError(
            "effective task serializer drifted from the official revised field order"
        )
    return payload
