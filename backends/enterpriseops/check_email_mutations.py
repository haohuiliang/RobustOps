#!/usr/bin/env python3
"""Smoke-test parent-first Email mutations against the real pinned MCP."""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
from typing import Any

from robustops.envs.enterpriseops.runtime import EnterpriseOpsEnvironment
from robustops.envs.enterpriseops.task import load_tasks, validate_seed_members


TASK_ID = "task_20251125_150631_436_e5bf1251_77dd7cbb"


async def check(project_root: Path, base_url: str) -> dict[str, Any]:
    tasks = load_tasks(
        project_root
        / "data/raw/EnterpriseOps-Gym/oracle/email-00000-of-00001.parquet",
        expected_domain="email",
        expected_count=67,
    )
    task = next(task for task in tasks if task.task_id == TASK_ID)
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

    async def run_sql(query: str) -> list[dict[str, Any]]:
        response = await environment._client.post(  # noqa: SLF001
            f"{environment.base_url}/api/sql-runner",
            json={"query": query, "database_id": environment.database_id},
            headers=environment._headers(),  # noqa: SLF001
        )
        response.raise_for_status()
        body = response.json()
        rows = body.get("data", body.get("rows", body.get("result")))
        if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
            raise RuntimeError("Email SQL runner returned an unexpected row shape")
        return rows

    cleanup_ok = False
    cleanup_error: str | None = None
    try:
        await environment.seed()
        await environment.connect()
        before_fk = await run_sql("PRAGMA foreign_key_check")
        before_counts = (
            await run_sql(
                "SELECT "
                "(SELECT COUNT(*) FROM history_events) AS histories, "
                "(SELECT COUNT(*) FROM threads) AS threads, "
                "(SELECT COUNT(*) FROM messages) AS messages"
            )
        )[0]

        draft = await environment.call_tool(
            "create_draft",
            {
                "userId": "me",
                "message": {
                    "payload": {
                        "partId": "0",
                        "mimeType": "text/plain",
                        "filename": "smoke.txt",
                        "headers": [
                            {"name": "To", "value": "qa@example.com"},
                            {"name": "Subject", "value": "FK attachment smoke"},
                        ],
                        "body": {"data": "c21va2U=", "size": 5},
                    }
                },
            },
        )
        sent = await environment.call_tool(
            "send_message",
            {
                "userId": "me",
                "raw": (
                    "From: me\nTo: qa@example.com\nSubject: FK send smoke"
                    "\n\nSend body"
                ),
            },
        )
        rejected = await environment.call_tool(
            "create_draft",
            {
                "userId": "me",
                "message": {
                    "payload": {
                        "headers": [{"name": "Subject", "value": "invalid"}],
                        "body": {"data": "eA==", "size": 1},
                    }
                },
            },
        )

        after_fk = await run_sql("PRAGMA foreign_key_check")
        after_counts = (
            await run_sql(
                "SELECT "
                "(SELECT COUNT(*) FROM history_events) AS histories, "
                "(SELECT COUNT(*) FROM threads) AS threads, "
                "(SELECT COUNT(*) FROM messages) AS messages"
            )
        )[0]
        before_counter = Counter(
            json.dumps(row, sort_keys=True) for row in before_fk
        )
        after_counter = Counter(json.dumps(row, sort_keys=True) for row in after_fk)
        new_fk = after_counter - before_counter
        removed_fk = before_counter - after_counter
        deltas = {
            key: int(after_counts[key]) - int(before_counts[key])
            for key in before_counts
        }
        passed = (
            draft["outcome"]["application_ok"] is True
            and sent["outcome"]["application_ok"] is True
            and rejected["outcome"]["application_ok"] is False
            and not new_fk
            and not removed_fk
            and deltas == {"histories": 2, "threads": 2, "messages": 2}
        )
        result = {
            "passed": passed,
            "task_id": task.task_id,
            "seed_database_file": server["seed_database_file"],
            "draft_with_attachment_application_ok": draft["outcome"][
                "application_ok"
            ],
            "send_message_application_ok": sent["outcome"]["application_ok"],
            "invalid_request_application_ok": rejected["outcome"][
                "application_ok"
            ],
            "preexisting_fk_violations": len(before_fk),
            "new_fk_violations": sum(new_fk.values()),
            "removed_fk_violations": sum(removed_fk.values()),
            "row_count_deltas": deltas,
        }
        if not passed:
            raise RuntimeError(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return result
    finally:
        cleanup_ok, cleanup_error = await environment.cleanup()
        if not cleanup_ok:
            raise RuntimeError(f"database cleanup failed: {cleanup_error}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    args = parser.parse_args()
    result = asyncio.run(
        check(args.project_root.resolve(), args.base_url.rstrip("/"))
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
