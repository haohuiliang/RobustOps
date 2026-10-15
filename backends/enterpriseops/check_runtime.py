#!/usr/bin/env python3
"""Check one real EnterpriseOps task against a running single-domain MCP."""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from robustops.envs.enterpriseops.runtime import EnterpriseOpsEnvironment
from robustops.envs.enterpriseops.task import load_tasks, validate_seed_members


DOMAIN_TASK_COUNTS = {
    "calendar": 61,
    "csm": 103,
    "drive": 64,
    "email": 67,
    "hr": 102,
    "itsm": 103,
    "teams": 61,
}


async def check(project_root: Path, domain: str, base_url: str) -> dict[str, object]:
    tasks = load_tasks(
        project_root / f"data/raw/EnterpriseOps-Gym/oracle/{domain}-00000-of-00001.parquet",
        expected_domain=domain,
        expected_count=DOMAIN_TASK_COUNTS[domain],
    )
    task = tasks[0]
    if len(task.gym_servers) != 1:
        raise ValueError(f"{task.task_id}: expected one MCP server, found {len(task.gym_servers)}")

    archive = project_root / "enterpriseops_gym/gym_dbs.zip"
    validate_seed_members([task], archive)
    server = task.gym_servers[0]
    environment = EnterpriseOpsEnvironment(
        server_name=server["mcp_server_name"],
        base_url=base_url,
        mcp_endpoint=task.mcp_endpoint,
        seed_database_file=server["seed_database_file"],
        context=server["context"],
        database_archive=archive,
    )

    cleanup_ok = False
    cleanup_error: str | None = None
    try:
        database_id = await environment.seed()
        await environment.connect()
        tools = await environment.list_tools()
        verifier_results = [
            await environment.run_verifier(verifier, index)
            for index, verifier in enumerate(task.verifiers)
        ]
    finally:
        cleanup_ok, cleanup_error = await environment.cleanup()

    if not cleanup_ok:
        raise RuntimeError(f"database cleanup failed: {cleanup_error}")
    return {
        "domain": domain,
        "task_id": task.task_id,
        "database_seeded": bool(database_id),
        "tool_count": len(tools),
        "verifier_queries_executed": len(verifier_results),
        "verifiers_passing_before_agent_action": sum(
            bool(result["passed"]) for result in verifier_results
        ),
        "database_cleanup": "passed",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("domain", choices=tuple(DOMAIN_TASK_COUNTS))
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(check(args.project_root.resolve(), args.domain, args.base_url))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
