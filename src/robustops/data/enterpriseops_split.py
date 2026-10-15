"""Build the frozen EnterpriseOps task split used by RobustOps.

The public benchmark release has domain splits and tool-exposure modes, but it
does not provide a training/validation/test partition or task-family labels.
This module anchors every variant to its Oracle ``task_id`` and groups tasks by
an auditable outcome-schema proxy before assigning whole groups to a partition.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Mapping

import pyarrow.parquet as pq

from robustops.envs.enterpriseops.revised_overlay import (
    OFFICIAL_REVISED_COMMIT,
    OFFICIAL_REVISED_EXCLUDED_TASKS,
    OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS,
    official_revised_supplement_rows,
)


SOURCE_REVISION = "c8e538eae8a6205294f0a86675fefdc1fac408f6"
SPLIT_ID = "enterpriseops"
ARTIFACT_STEM = "enterpriseops"
MODES = ("oracle", "plus_5_tools", "plus_10_tools", "plus_15_tools")
SINGLE_DOMAINS = ("calendar", "csm", "drive", "email", "hr", "itsm", "teams")
COMPOSITION_DOMAINS = ("hybrid",)
ALL_DOMAINS = SINGLE_DOMAINS + COMPOSITION_DOMAINS
SPLIT_SEED = 20260830
RATIOS = {"train": 0.80, "validation": 0.10, "test": 0.10}

# Invalid source contracts are excluded from every effective partition.
# Raw rows and exclusion reasons remain available for historical provenance.
EFFECTIVE_TASK_EXCLUSIONS: Mapping[str, str] = {
    "task_20260107_180857_873_ae7b45e2_47771a8c": (
        "Drive prompt requests Family Photo.jpg (893KB), but all three verifiers "
        "target Company Policies and Compliance Review document content; "
        "the task and verifier describe different outcomes"
    ),
    "task_20251127_070938_325_b393a327_a40b1f4e": (
        "CSM request claims Sheri while the authenticated Jose principal is hidden "
        "from model messages and no selected tool exposes the current principal; "
        "the policy decision is therefore not model-observable"
    ),
    "task_20251204_115902_743_13762b97_19892db1": (
        "Drive prompt targets Technical Spec while every verifier targets "
        "Marketing Presentation; neither side can be uniquely repaired"
    ),
}
EXPECTED_COUNTS = {
    "oracle": {
        "calendar": 61,
        "csm": 103,
        "drive": 64,
        "email": 67,
        "hr": 102,
        "hybrid": 88,
        "itsm": 103,
        "teams": 61,
    },
    "plus_5_tools": {
        "calendar": 59,
        "csm": 102,
        "drive": 64,
        "email": 67,
        "hr": 102,
        "hybrid": 88,
        "itsm": 103,
        "teams": 52,
    },
    "plus_10_tools": {
        "calendar": 59,
        "csm": 102,
        "drive": 64,
        "email": 67,
        "hr": 102,
        "hybrid": 88,
        "itsm": 103,
        "teams": 52,
    },
    "plus_15_tools": {
        "calendar": 59,
        "csm": 102,
        "drive": 64,
        "email": 67,
        "hr": 102,
        "hybrid": 88,
        "itsm": 103,
        "teams": 52,
    },
}

_CTE_PATTERN = re.compile(r"(?:\bwith|,)\s*([a-z_][a-z0-9_]*)\s+as\s*\(")
_TABLE_PATTERN = re.compile(r"\b(?:from|join)\s+[`\"]?([a-z_][a-z0-9_]*)")
_SEMANTIC_FIELDS = (
    "domain",
    "system_prompt",
    "user_prompt",
    "restricted_tools",
    "mcp_endpoint",
    "number_of_runs",
    "reset_database_between_runs",
    "gym_servers_config",
    "verifiers",
)


class SplitContractError(ValueError):
    """The local dataset cannot produce the registered split contract."""


def _parse_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return value
    return value


def semantic_signature(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Return mode-invariant task fields in a semantic representation.

    Final-state verifiers form an order-independent conjunction.  At the fixed
    public revision, some mode rows only differ because the verifier list was
    reordered, exact verifier objects were duplicated, or a numeric expected
    value was serialized as a canonical decimal string in Oracle and an integer
    in distractor modes.  Those representation differences do not define a new
    task and are normalized here; all other fields remain exact.
    """

    return tuple(
        (
            _canonical_verifiers(row[field])
            if field == "verifiers"
            else json.dumps(_parse_json(row[field]), ensure_ascii=False, sort_keys=True)
        )
        for field in _SEMANTIC_FIELDS
    )


def _canonical_integer(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    if value == "0" or (value.isdigit() and not value.startswith("0")):
        return int(value)
    return value


def _canonical_verifiers(value: Any) -> str:
    verifiers = _parse_json(value)
    if not isinstance(verifiers, list) or not all(
        isinstance(verifier, Mapping) for verifier in verifiers
    ):
        return json.dumps(verifiers, ensure_ascii=False, sort_keys=True)

    canonical: set[str] = set()
    for source in verifiers:
        verifier = dict(source)
        validation = verifier.get("validation_config")
        if isinstance(validation, Mapping) and "expected_value" in validation:
            normalized_validation = dict(validation)
            normalized_validation["expected_value"] = _canonical_integer(
                validation["expected_value"]
            )
            verifier["validation_config"] = normalized_validation
        canonical.add(json.dumps(verifier, ensure_ascii=False, sort_keys=True))
    return json.dumps(sorted(canonical), ensure_ascii=False)


def verifier_tables(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Extract physical tables referenced by final-state verifier SQL."""

    verifiers = _parse_json(row["verifiers"])
    if not isinstance(verifiers, list) or not verifiers:
        raise SplitContractError(f"{row.get('task_id')}: verifiers must be a non-empty list")
    tables: set[str] = set()
    for verifier in verifiers:
        query = verifier.get("validation_config", {}).get("query")
        if not isinstance(query, str) or not query.strip():
            raise SplitContractError(f"{row.get('task_id')}: verifier query is missing")
        lowered = query.lower()
        cte_names = set(_CTE_PATTERN.findall(lowered))
        tables.update(name for name in _TABLE_PATTERN.findall(lowered) if name not in cte_names)
    if not tables:
        raise SplitContractError(f"{row.get('task_id')}: no verifier tables could be extracted")
    return tuple(sorted(tables))


def family_proxy(row: Mapping[str, Any]) -> tuple[str, tuple[str, ...], int]:
    """Group tasks by domain, outcome tables, and verifier cardinality.

    This deliberately avoids opaque task-id fragments, prompt similarity, and
    verifier literals. It is a documented proxy because the public release has
    no authoritative task-family or template field.
    """

    verifiers = _parse_json(row["verifiers"])
    return str(row["domain"]), verifier_tables(row), len(verifiers)


def family_name(key: tuple[str, tuple[str, ...], int]) -> str:
    domain, tables, verifier_count = key
    return f"{domain}::tables={'+'.join(tables)}::verifiers={verifier_count}"


def _read_revision(data_root: Path) -> str:
    revision_path = data_root / "REVISION"
    if not revision_path.is_file():
        raise SplitContractError(f"missing dataset revision marker: {revision_path}")
    revision = revision_path.read_text(encoding="utf-8").strip()
    if revision != SOURCE_REVISION:
        raise SplitContractError(
            f"dataset revision must equal {SOURCE_REVISION}, found {revision or 'empty'}"
        )
    return SOURCE_REVISION


def load_rows(data_root: Path) -> dict[str, dict[str, dict[str, Mapping[str, Any]]]]:
    """Load the fixed public modes plus the official Oracle-only supplement."""

    _read_revision(data_root)
    result: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for mode in MODES:
        result[mode] = {}
        for domain in ALL_DOMAINS:
            path = data_root / mode / f"{domain}-00000-of-00001.parquet"
            if not path.is_file():
                raise SplitContractError(f"missing dataset file: {path}")
            rows = pq.read_table(path).to_pylist()
            expected = EXPECTED_COUNTS[mode][domain]
            if len(rows) != expected:
                raise SplitContractError(f"{path}: expected {expected} rows, found {len(rows)}")
            by_id = {row["task_id"]: row for row in rows}
            if len(by_id) != len(rows):
                raise SplitContractError(f"{path}: duplicate task_id")
            wrong = sorted({row["domain"] for row in rows if row["domain"] != domain})
            if wrong:
                raise SplitContractError(f"{path}: unexpected domains {wrong}")
            result[mode][domain] = by_id
    for row in official_revised_supplement_rows(domain="csm", mode="oracle"):
        task_id = str(row["task_id"])
        if task_id in result["oracle"]["csm"]:
            raise SplitContractError(f"official supplement duplicates public task {task_id}")
        result["oracle"]["csm"][task_id] = row
    return result


def _domain_targets(total: int) -> dict[str, int]:
    targets = {
        "train": round(total * RATIOS["train"]),
        "validation": round(total * RATIOS["validation"]),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    return targets


def assign_domain_families(
    rows: Iterable[Mapping[str, Any]], *, domain_index: int
) -> dict[str, str]:
    """Assign whole family-proxy groups while preserving domain ratios."""

    groups: dict[tuple[str, tuple[str, ...], int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[family_proxy(row)].append(row)

    total = sum(len(group) for group in groups.values())
    targets = _domain_targets(total)
    counts = {partition: 0 for partition in RATIOS}
    assignment: dict[str, str] = {}

    ordered = sorted(groups.items(), key=lambda item: family_name(item[0]))
    random.Random(SPLIT_SEED + domain_index).shuffle(ordered)
    ordered.sort(key=lambda item: len(item[1]), reverse=True)

    for key, group in ordered:
        size = len(group)
        # A single common family should not consume most of a held-out domain.
        # Keeping it in train preserves more family diversity in validation/test.
        if size > min(targets["validation"], targets["test"]) / 2:
            partition = "train"
        else:
            def cost(candidate: str) -> float:
                return sum(
                    (
                        (
                            counts[name]
                            + (size if name == candidate else 0)
                            - targets[name]
                        )
                        / targets[name]
                    )
                    ** 2
                    for name in RATIOS
                )

            partition = min(RATIOS, key=lambda name: (cost(name), tuple(RATIOS).index(name)))

        for row in group:
            assignment[row["task_id"]] = partition
        counts[partition] += size

    if counts != targets:
        raise SplitContractError(
            f"family allocation did not meet domain targets: expected {targets}, found {counts}"
        )
    return assignment


def build_manifest(
    rows_by_mode: Mapping[str, Mapping[str, Mapping[str, Mapping[str, Any]]]]
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    oracle = rows_by_mode["oracle"]
    effective_oracle = {
        domain: {
            task_id: row
            for task_id, row in oracle[domain].items()
            if task_id not in EFFECTIVE_TASK_EXCLUSIONS
        }
        for domain in ALL_DOMAINS
    }
    partitions: dict[str, str] = {}
    supplement_ids = set(OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS)
    for domain_index, domain in enumerate(ALL_DOMAINS):
        # Reconstruct the frozen source allocation without reshuffling the
        # remaining tasks. Excluded rows never receive an effective partition.
        source_allocation = assign_domain_families(
            (
                row
                for task_id, row in oracle[domain].items()
                if task_id not in supplement_ids
            ),
            domain_index=domain_index,
        )
        partitions.update(
            (task_id, partition)
            for task_id, partition in source_allocation.items()
            if task_id in effective_oracle[domain]
        )
    partitions.update({task_id: "train" for task_id in supplement_ids})

    entries: list[dict[str, Any]] = []
    for domain in ALL_DOMAINS:
        for task_id, oracle_row in sorted(effective_oracle[domain].items()):
            key = family_proxy(oracle_row)
            available_modes = [mode for mode in MODES if task_id in rows_by_mode[mode][domain]]
            oracle_signature = semantic_signature(oracle_row)
            paired_modes = [
                mode
                for mode in available_modes
                if semantic_signature(rows_by_mode[mode][domain][task_id]) == oracle_signature
            ]
            supplement = OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS.get(task_id)
            if supplement is not None:
                runnable_modes = ["oracle"]
            else:
                runnable_modes = list(available_modes)
            entries.append(
                {
                    "task_id": task_id,
                    "domain": domain,
                    "partition": partitions[task_id],
                    "family_proxy": family_name(key),
                    "verifier_tables": list(key[1]),
                    "verifier_count": key[2],
                    "available_modes": available_modes,
                    "paired_modes": paired_modes,
                    "runnable_modes": runnable_modes,
                    "source_kind": (
                        "official_revised_supplement"
                        if supplement is not None
                        else "fixed_public_parquet"
                    ),
                    "execution_blocked_reason": None,
                }
            )

    _validate_manifest(entries)
    summary = _summarize(entries)
    metadata = {
        "schema_version": 4,
        "split_id": SPLIT_ID,
        "source": {
            "dataset": "ServiceNow-AI/EnterpriseOps-Gym",
            "revision": SOURCE_REVISION,
            "base_mode": "oracle",
            "public_oracle_tasks": sum(len(tasks) for tasks in oracle.values()) - len(supplement_ids),
            "official_revised_commit": OFFICIAL_REVISED_COMMIT,
            "official_revised_train_supplement_tasks": len(supplement_ids),
            "official_revised_excluded_tasks": len(OFFICIAL_REVISED_EXCLUDED_TASKS),
            "effective_oracle_tasks": len(entries),
            "excluded_fixed_public_tasks": len(EFFECTIVE_TASK_EXCLUSIONS),
            "excluded_task_records": [
                {"task_id": task_id, "domain": domain, "reason": reason}
                for task_id, reason in sorted(EFFECTIVE_TASK_EXCLUSIONS.items())
                for domain in ALL_DOMAINS
                if task_id in oracle[domain]
            ],
        },
        "policy": {
            "domains": list(ALL_DOMAINS),
            "single_domains": list(SINGLE_DOMAINS),
            "composition_domains": list(COMPOSITION_DOMAINS),
            "ratios": RATIOS,
            "seed": SPLIT_SEED,
            "family_proxy": "domain + physical verifier table set + verifier count",
            "family_proxy_limit": (
                "The public release has no authoritative task-family/template label; "
                "this outcome-schema grouping is an auditable conservative proxy."
            ),
            "variant_rule": (
                "All mode, fault, state, rollout, SFT, and RL variants inherit the "
                "partition of the Oracle base task_id."
            ),
            "mode_rule": (
                "Oracle is the learning mode. Distractor modes are evaluation-only and "
                "are paired only when task semantics equal the Oracle row."
            ),
            "pairing_normalization": (
                "Verifier order is ignored, exact duplicate verifier objects collapse, "
                "and canonical decimal integer expected values compare equal to their "
                "integer representation; every other mode-invariant field remains exact."
            ),
            "ood_rule": (
                "No domain or composition is held out from the main split. Unseen fault "
                "families are the primary OOD condition; domain and composition remain "
                "reported evaluation slices."
            ),
            "control_plane_rule": (
                "family_proxy, verifier_tables, and verifier_count are split-audit "
                "metadata only and must never be exposed to the model at runtime."
            ),
            "large_family_rule": (
                "A family larger than half the smaller held-out target stays in train "
                "to preserve validation/test family diversity."
            ),
            "official_revised_supplement_rule": (
                "The two executable official revised-only CSM tasks are train-only "
                "and do not acquire fabricated distractor variants. A revised-only "
                "task whose official seed is absent is excluded before partitioning."
            ),
            "source_contract_exclusion_rule": (
                "A fixed-public task whose prompt/verifiers cannot be uniquely reconciled "
                "or whose required policy precondition is not observable by the model "
                "is excluded from every effective partition and recorded only in source "
                "exclusion metadata. Remaining task memberships retain the frozen source "
                "allocation without resampling."
            ),
        },
        "summary": summary,
    }
    return metadata, entries


def _validate_manifest(entries: list[Mapping[str, Any]]) -> None:
    ids = [entry["task_id"] for entry in entries]
    if set(ids) & EFFECTIVE_TASK_EXCLUSIONS.keys():
        raise SplitContractError("excluded source tasks must not enter an effective partition")
    if any(not entry["runnable_modes"] or entry.get("execution_blocked_reason") for entry in entries):
        raise SplitContractError("effective partitions must contain runnable tasks only")
    expected = 649 + len(OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS) - len(EFFECTIVE_TASK_EXCLUSIONS)
    if len(entries) != expected or len(set(ids)) != len(ids):
        raise SplitContractError(
            f"manifest must contain {expected} unique effective Oracle task IDs"
        )
    supplements = {
        entry["task_id"]: entry
        for entry in entries
        if entry["task_id"] in OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS
    }
    if set(supplements) != set(OFFICIAL_REVISED_TRAIN_SUPPLEMENT_TASKS) or any(
        entry["partition"] != "train" for entry in supplements.values()
    ):
        raise SplitContractError("official revised supplements must all be train-only")
    family_partitions: dict[str, set[str]] = defaultdict(set)
    for entry in entries:
        family_partitions[entry["family_proxy"]].add(entry["partition"])
        if "oracle" not in entry["paired_modes"]:
            raise SplitContractError(f"{entry['task_id']}: Oracle must pair with itself")
    leaked = sorted(name for name, parts in family_partitions.items() if len(parts) != 1)
    if leaked:
        raise SplitContractError(f"family proxies cross partitions: {leaked}")


def _summarize(entries: list[Mapping[str, Any]]) -> dict[str, Any]:
    partitions = Counter(entry["partition"] for entry in entries)
    by_domain: dict[str, dict[str, int]] = {}
    for domain in ALL_DOMAINS:
        by_domain[domain] = dict(
            sorted(Counter(
                entry["partition"] for entry in entries if entry["domain"] == domain
            ).items())
        )
    paired_by_mode = {
        mode: sum(mode in entry["paired_modes"] for entry in entries) for mode in MODES
    }
    runnable_by_mode = {
        mode: sum(mode in entry["runnable_modes"] for entry in entries)
        for mode in MODES
    }
    family_counts = Counter(entry["partition"] for entry in {
        (entry["family_proxy"], entry["partition"]): entry for entry in entries
    }.values())
    return {
        "tasks_by_partition": dict(sorted(partitions.items())),
        "families_by_partition": dict(sorted(family_counts.items())),
        "tasks_by_domain_and_partition": by_domain,
        "semantically_paired_tasks_by_mode": paired_by_mode,
        "runnable_tasks_by_mode": runnable_by_mode,
    }


def write_artifacts(output_dir: Path, metadata: Mapping[str, Any], entries: list[Mapping[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / f"{ARTIFACT_STEM}.json"
    manifest_path = output_dir / f"{ARTIFACT_STEM}.jsonl"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    manifest_path.write_text(
        "".join(json.dumps(entry, ensure_ascii=False, sort_keys=True) + "\n" for entry in entries),
        encoding="utf-8",
    )


def load_partition_task_ids(
    manifest_path: Path,
    *,
    domain: str,
    partition: str,
    paired_mode: str | None = None,
    available_mode: str | None = None,
    runnable_mode: str | None = None,
) -> list[str]:
    """Load one domain/partition with an optional pair or availability filter."""

    if not manifest_path.is_file():
        raise SplitContractError(f"task manifest not found: {manifest_path}")
    if paired_mode is not None and available_mode is not None:
        raise SplitContractError("paired_mode and available_mode are mutually exclusive")
    if paired_mode is not None and paired_mode not in MODES:
        raise SplitContractError(f"unsupported paired mode: {paired_mode!r}")
    if available_mode is not None and available_mode not in MODES:
        raise SplitContractError(f"unsupported available mode: {available_mode!r}")
    if runnable_mode is not None and runnable_mode not in MODES:
        raise SplitContractError(f"unsupported runnable mode: {runnable_mode!r}")
    selected: list[str] = []
    seen: set[str] = set()
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        try:
            entry = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SplitContractError(
                f"{manifest_path}:{line_number}: invalid JSON: {exc}"
            ) from exc
        task_id = entry.get("task_id")
        if not isinstance(task_id, str) or not task_id:
            raise SplitContractError(f"{manifest_path}:{line_number}: invalid task_id")
        if task_id in seen:
            raise SplitContractError(f"{manifest_path}: duplicate task_id {task_id}")
        seen.add(task_id)
        if entry.get("domain") != domain or entry.get("partition") != partition:
            continue
        if paired_mode is not None:
            paired_modes = entry.get("paired_modes")
            if not isinstance(paired_modes, list) or not all(
                isinstance(mode, str) for mode in paired_modes
            ):
                raise SplitContractError(
                    f"{manifest_path}:{line_number}: invalid paired_modes"
                )
            if paired_mode not in paired_modes:
                continue
        if available_mode is not None:
            available_modes = entry.get("available_modes")
            if not isinstance(available_modes, list) or not all(
                isinstance(mode, str) for mode in available_modes
            ):
                raise SplitContractError(
                    f"{manifest_path}:{line_number}: invalid available_modes"
                )
            if available_mode not in available_modes:
                continue
        if runnable_mode is not None:
            runnable_modes = entry.get("runnable_modes", entry.get("available_modes"))
            if not isinstance(runnable_modes, list) or not all(
                isinstance(mode, str) for mode in runnable_modes
            ):
                raise SplitContractError(
                    f"{manifest_path}:{line_number}: invalid runnable_modes"
                )
            if runnable_mode not in runnable_modes:
                continue
        selected.append(task_id)
    if not selected:
        raise SplitContractError(
            f"task manifest has no tasks for domain={domain!r}, "
            f"partition={partition!r}, paired_mode={paired_mode!r}, "
            f"available_mode={available_mode!r}"
        )
    return sorted(selected)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    rows = load_rows(args.data_root.resolve())
    metadata, entries = build_manifest(rows)
    write_artifacts(args.output_dir.resolve(), metadata, entries)
    print(json.dumps(metadata["summary"], ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
