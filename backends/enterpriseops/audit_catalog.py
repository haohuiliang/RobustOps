#!/usr/bin/env python3
"""Audit every allowed dataset tool against the pinned MCP catalogs."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
from typing import Mapping

import pyarrow.parquet as pq

from robustops.envs.enterpriseops.runtime import EnterpriseOpsEnvironment
from robustops.envs.enterpriseops.task import load_tasks


DEFAULT_SERVERS = {
    "calendar": ("gym-calendar", "http://127.0.0.1:8003"),
    "csm": ("sn-csm-server", "http://127.0.0.1:8001"),
    "drive": ("gym-google-drive-mcp", "http://127.0.0.1:8009"),
    "email": ("gym-email-mcp", "http://127.0.0.1:8004"),
    "hr": ("sn-hr-internal", "http://127.0.0.1:8008"),
    "itsm": ("gym-itsm-mcp", "http://127.0.0.1:8006"),
    "teams": ("gym-teams-mcp", "http://127.0.0.1:8002"),
}


async def _discover_catalog(
    project_root: Path,
    server_name: str,
    base_url: str,
) -> tuple[str, set[str]]:
    environment = EnterpriseOpsEnvironment(
        server_name=server_name,
        base_url=base_url,
        mcp_endpoint="/mcp",
        seed_database_file="unused-for-tools-list",
        context={},
        database_archive=project_root / "enterpriseops_gym/gym_dbs.zip",
    )
    try:
        await environment.connect()
        tools = await environment.list_tools()
    finally:
        await environment.cleanup()
    return server_name, {tool["name"] for tool in tools}


def audit_catalogs(
    project_root: Path,
    catalogs: Mapping[str, set[str]],
) -> dict[str, object]:
    missing: list[dict[str, object]] = []
    allowed_missing: list[dict[str, object]] = []
    row_count = 0
    reference_count = 0
    data_root = project_root / "data/raw/EnterpriseOps-Gym"
    parquet_paths = sorted(data_root.glob("*/*.parquet"))

    for path in parquet_paths:
        mode = path.parent.name
        domain = path.name.split("-", 1)[0]
        expected_count = pq.ParquetFile(path).metadata.num_rows
        tasks = load_tasks(
            path,
            expected_domain=domain,
            expected_count=expected_count,
        )
        row_count += len(tasks)
        for task in tasks:
            restricted = set(task.restricted_tools)
            allowed = [
                name for name in task.unique_selected_tools if name not in restricted
            ]
            reference_count += len(allowed)
            server_names = tuple(
                server["mcp_server_name"] for server in task.gym_servers
            )
            try:
                available = set().union(*(catalogs[name] for name in server_names))
            except KeyError as exc:
                raise ValueError(f"missing live catalog for {exc.args[0]!r}") from exc
            for name in allowed:
                if name in available:
                    continue
                record = {
                    "mode": mode,
                    "domain": domain,
                    "task_id": task.task_id,
                    "tool": name,
                    "servers": server_names,
                }
                registered_omissions = {
                    correction.tool_name
                    for correction in task.missing_tool_corrections
                }
                if name in registered_omissions:
                    allowed_missing.append(record)
                else:
                    missing.append(record)

    by_tool = Counter(record["tool"] for record in missing)
    by_mode = Counter((record["mode"], record["tool"]) for record in missing)
    return {
        "parquet_files": len(parquet_paths),
        "task_rows_scanned": row_count,
        "allowed_tool_references_scanned": reference_count,
        "missing_occurrences": len(missing),
        "missing_by_tool": dict(sorted(by_tool.items())),
        "missing_by_mode_and_tool": {
            f"{mode}:{tool}": count
            for (mode, tool), count in sorted(by_mode.items())
        },
        "missing_records": missing,
        "allowed_missing_occurrences": len(allowed_missing),
        "allowed_missing_records": allowed_missing,
    }


async def audit_live(project_root: Path) -> dict[str, object]:
    discovered = await asyncio.gather(
        *(
            _discover_catalog(project_root, server_name, base_url)
            for server_name, base_url in DEFAULT_SERVERS.values()
        )
    )
    catalogs = dict(discovered)
    result = audit_catalogs(project_root, catalogs)
    result["live_tool_counts"] = {
        domain: len(catalogs[server_name])
        for domain, (server_name, _) in DEFAULT_SERVERS.items()
    }
    return result


def _load_catalog_dir(catalog_dir: Path) -> dict[str, set[str]]:
    catalogs: dict[str, set[str]] = {}
    for domain, (expected_server_name, _) in DEFAULT_SERVERS.items():
        path = catalog_dir / f"{domain}.json"
        payload = json.loads(path.read_text())
        server_name = payload.get("server_name")
        tools = payload.get("tools")
        if server_name != expected_server_name:
            raise ValueError(
                f"{path}: expected server {expected_server_name!r}, found {server_name!r}"
            )
        if not isinstance(tools, list) or not all(
            isinstance(name, str) and name for name in tools
        ):
            raise ValueError(f"{path}: tools must be a list of non-empty strings")
        catalogs[server_name] = set(tools)
    return catalogs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--catalog-dir", type=Path)
    parser.add_argument("--collect-domain", choices=tuple(DEFAULT_SERVERS))
    parser.add_argument("--base-url")
    args = parser.parse_args()
    project_root = args.project_root.resolve()
    if args.collect_domain:
        if not args.base_url:
            parser.error("--collect-domain requires --base-url")
        server_name = DEFAULT_SERVERS[args.collect_domain][0]
        discovered_name, tools = asyncio.run(
            _discover_catalog(project_root, server_name, args.base_url)
        )
        print(
            json.dumps(
                {"server_name": discovered_name, "tools": sorted(tools)},
                ensure_ascii=False,
                indent=2,
            )
        )
        return
    if args.base_url:
        parser.error("--base-url is only valid with --collect-domain")
    if args.catalog_dir:
        catalogs = _load_catalog_dir(args.catalog_dir.resolve())
        result = audit_catalogs(project_root, catalogs)
        result["live_tool_counts"] = {
            domain: len(catalogs[server_name])
            for domain, (server_name, _) in DEFAULT_SERVERS.items()
        }
    else:
        result = asyncio.run(audit_live(project_root))
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["missing_occurrences"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
