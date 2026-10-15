"""Materialize the full train task×mode pool and a fixed validation panel."""

from collections import defaultdict
import json
from pathlib import Path
import random

from robustops.training.recovery_rl_core import KINDS, load_jsonl, write_json
from robustops.evaluation.fault_schedule import load_schedule


TASK_CONTRACT_REVISION = "20260923-photo-itsm-hr"


def require_current_task_contract(data_directory):
    summary = json.loads((Path(data_directory) / "data_summary.json").read_text())
    if summary.get("task_contract_revision") != TASK_CONTRACT_REVISION:
        raise ValueError(
            "Saved RL data predates the task/verifier/schema repairs. Start a fresh "
            "suite from SFT; old data/plans/checkpoints remain historical and cannot "
            "be resumed under the changed task contract."
        )


def load_task(root, domain, mode, task_id):
    from robustops.envs.enterpriseops.task import load_tasks
    from robustops.data.enterpriseops_split import EFFECTIVE_TASK_EXCLUSIONS

    if task_id in EFFECTIVE_TASK_EXCLUSIONS:
        raise ValueError(
            f"Excluded RL task {task_id}: {EFFECTIVE_TASK_EXCLUSIONS[task_id]}"
        )
    directory = "oracle" if mode == "oracle" else "plus5"
    config = json.loads(
        (
            Path(root) / f"recipes/enterpriseops/paired/{directory}/{domain}.json"
        ).read_text()
    )
    return next(
        t
        for t in load_tasks(
            Path(root) / config["data_path"],
            expected_domain=domain,
            expected_count=config["expected_task_count"],
        )
        if t.task_id == task_id
    )


def teacher_seen(root, expected_records=None, *, sources=None, input_files=None):
    from robustops.training.rl_data_config import resolve_data_sources

    sources = resolve_data_sources(root, sources)
    records = load_jsonl(Path(sources["sft_records"]))
    if not records or len({r["record_id"] for r in records}) != len(records):
        raise ValueError("SFT records must be nonempty with unique record IDs")
    if expected_records is not None and len(records) != expected_records:
        raise ValueError(f"SFT record count differs: expected {expected_records}, got {len(records)}")
    exposure = Path(sources["sft_fault_exposure"]) if sources["sft_fault_exposure"] else None
    if exposure is not None and exposure.is_file():
        frozen = json.loads(exposure.read_text())
        if (
            frozen["teacher_records"] != len(records)
        ):
            raise ValueError(
                "SFT record count differs from the frozen teacher exposure manifest"
            )
        known = {(r["task_id"], r["mode"]) for r in records}
        seen = defaultdict(set)
        for row in frozen["rows"]:
            key = row["task_id"], row["mode"]
            if key not in known or key in seen:
                raise ValueError("Unknown or repeated teacher exposure cell")
            seen[key].update(k for k in row["kinds"] if k in KINDS)
        return seen
    if sources["sft_provenance"] is None:
        raise ValueError("Provide sft_fault_exposure or sft_provenance for teacher exposure")
    provenance_rows = load_jsonl(Path(sources["sft_provenance"]))
    provenance = {r["record_id"]: r for r in provenance_rows}
    if len(provenance) != len(provenance_rows) or not {r["record_id"] for r in records} <= set(provenance):
        raise ValueError("Missing or duplicate SFT provenance records")
    seen = defaultdict(set)
    for row in records:
        source = provenance[row["record_id"]]
        if source["condition"] != "fault":
            continue
        attempt_path = Path(root) / source["source"] / "attempt.json"
        if input_files is not None:
            input_files.append(attempt_path)
        attempt = json.loads(attempt_path.read_text())
        if (
            not attempt["valid"]
            or not attempt["success"]
            or attempt["task_id"] != row["task_id"]
            or attempt.get("mode", row["mode"]) != row["mode"]
        ):
            raise ValueError("SFT actual-attempt lineage mismatch")
        for event in attempt["fault"].get("events", []):
            kind = event.get("selected_fault") or attempt["fault"].get(
                "selected_mechanism"
            )
            if kind in KINDS:
                seen[(row["task_id"], row["mode"])].add(kind)
    return seen


async def capture_initial_state(root, row, urls, target):
    """Materialize evaluated seed SQL once; all group members restore this state."""
    from robustops.envs.enterpriseops.runtime import EnterpriseOpsEnvironment

    task = load_task(root, row["domain"], row["mode"], row["task_id"])
    state = {}
    for server in task.gym_servers:
        name = server["mcp_server_name"]
        env = EnterpriseOpsEnvironment(
            server_name=name,
            base_url=urls[name],
            mcp_endpoint=task.mcp_endpoint,
            seed_database_file=server["seed_database_file"],
            context=server["context"],
            database_archive=Path(root) / "enterpriseops_gym/gym_dbs.zip",
        )
        try:
            await env.seed()
            state[name] = await env.export_seed_state()
        finally:
            succeeded, error = await env.cleanup()
            if not succeeded:
                raise RuntimeError(f"initial-state database cleanup failed: {error}")
    write_json(target, state)


def prepare(root, output, config):
    import pyarrow as pa
    import pyarrow.parquet as pq

    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    from robustops.training.rl_data_config import resolve_data_sources, file_fingerprints

    sources = resolve_data_sources(root, config.get("data_sources"))
    input_files = list(sources.values())
    pool = json.loads(Path(sources["train_pool"]).read_text())["tasks"]
    manifest_rows = load_jsonl(Path(sources["task_manifest"]))
    manifest = {r["task_id"]: r for r in manifest_rows}
    if len(manifest) != len(manifest_rows):
        raise ValueError("Duplicate task IDs in task manifest")
    if not pool or len({r["task_id"] for r in pool}) != len(pool):
        raise ValueError("Training pool must be nonempty with unique task IDs")
    expected = {
        (r["task_id"], mode)
        for r in manifest.values() if r["partition"] == "train"
        for mode in ("oracle", "plus_5_tools") if mode in r["runnable_modes"]
    }
    selector_rows = load_jsonl(Path(sources["fault_selector"]))
    selectors = {(r["task_id"], r["mode"]): r for r in selector_rows}
    if len(selectors) != len(selector_rows) or not expected <= set(selectors):
        raise ValueError("Fault selectors must uniquely cover the Train manifest task×mode cells")
    records = load_jsonl(Path(sources["sft_records"]))
    if any((r["task_id"], r["mode"]) not in expected for r in records):
        raise ValueError("SFT records contain tasks outside the Train manifest")
    seen = teacher_seen(root, config.get("sft_records"), sources=sources, input_files=input_files)
    rows = []
    for task in sorted(pool, key=lambda r: r["task_id"]):
        original = manifest.get(task["task_id"])
        if original is None or original["partition"] != "train" or task["domain"] != original["domain"]:
            raise ValueError("Training pool task/domain must match the Train manifest")
        modes = task["modes"]
        if len(modes) != len(set(modes)):
            raise ValueError("Duplicate tool modes in training pool")
        for mode in ("oracle", "plus_5_tools"):
            if mode not in modes:
                continue
            selection = selectors.get((task["task_id"], mode))
            if selection is None or selection["partition"] != "train" or selection["domain"] != task["domain"]:
                raise ValueError("Fault selector task/domain/partition mismatch")
            rows.append(dict(task_id=task["task_id"], domain=task["domain"], mode=mode,
                selection=dict(tools=[dict(t, faults=[k for k in t["faults"] if k in KINDS])
                                      for t in selection["tools"]]),
                sft_seen=sorted(seen[(task["task_id"], mode)])))
    if {(r["task_id"], r["mode"]) for r in rows} != expected:
        raise ValueError("Training pool must cover all Train manifest task×mode cells")
    if len(rows) < 32:
        raise ValueError("Training pool needs at least 32 distinct task×mode cells for the configured groups")
    rng = random.Random(config["seed"])
    rng.shuffle(rows)
    from robustops.training.rl_task_schedule import (
        task_schedule,
        fault_reserve_schedule,
    )

    queue_mode = config.get("task_sampling") == "queue"
    if queue_mode:
        from robustops.training.recovery_task_queue import carrier_batches

        schedule = carrier_batches(rows, 1, config["steps"])
        write_json(output / "task_pool.json", dict(rows=rows, seed=config["seed"]))
    else:
        schedule = task_schedule(rows, config["seed"], config["steps"])
    train = [
        dict(
            prompt=[
                dict(
                    role="user",
                    content="Task loaded by the isolated EnterpriseOps agent loop.",
                )
            ],
            data_source="enterpriseops_train",
            agent_name="recovery_agent",
            reward_model=dict(style="rule", ground_truth=""),
            extra_info=dict(index=row["visit"], recovery=row),
        )
        for batch in schedule
        for row in batch["groups"]
    ]
    write_json(
        output / "task_schedule.json",
        dict(
            seed=config["seed"],
            batches=schedule,
            pairing="visit_slots_only; runtime continuous queues"
            if queue_mode
            else "task_x_mode_only; faults remain run-local",
        ),
    )
    if not queue_mode:
        write_json(
            output / "fault_reserves.json",
            fault_reserve_schedule(rows, config["seed"], config["steps"]),
        )
    pq.write_table(pa.Table.from_pylist(train), output / "train.parquet")
    inference = json.loads(Path(sources["validation_recipe"]).read_text())
    policy_path = Path(sources["validation_policy"]) if sources["validation_policy"] else root / inference["selector_policy"]
    input_files.append(policy_path)
    val_policy = json.loads(policy_path.read_text())
    val_rows = sorted(val_policy["rows"], key=lambda r: (r["domain"], r["mode"], r["task_id"]))
    val_keys = [(r["task_id"], r["mode"]) for r in val_rows]
    if not val_rows or len(set(val_keys)) != len(val_keys) or len({r["cell"] for r in val_rows}) != len(val_rows):
        raise ValueError("Validation policy must contain unique task×mode cells and cell IDs")
    for row in val_rows:
        original = manifest.get(row["task_id"])
        if (row["partition"] != "validation" or original is None
                or original["partition"] != "validation" or original["domain"] != row["domain"]
                or row["mode"] not in original["runnable_modes"]):
            raise ValueError("Train/Val contamination or validation task identity mismatch")
    validation = []
    schedule_path = root / inference["schedule"]
    input_files.append(schedule_path)
    binding = json.loads(schedule_path.read_text())
    if binding.get("strategy"):
        input_files.append(root / binding["strategy"])
    val_schedule = load_schedule(schedule_path, root=root)
    budgets_path = root / val_schedule["candidate_budgets"]
    input_files.append(budgets_path)
    val_budgets = json.loads(budgets_path.read_text())
    for row in val_rows:
        for fault in (False, True):
            visit = len(validation)
            recovery = dict(
                task_id=row["task_id"],
                domain=row["domain"],
                mode=row["mode"],
                selection=row["selection"],
                candidate_budgets=val_budgets[str(row["cell"])],
                sft_seen=[],
                epoch=1,
                visit=visit,
                validation=True,
                force_fault=fault,
            )
            validation.append(
                dict(
                    prompt=[dict(role="user", content="Fixed validation task.")],
                    data_source="enterpriseops_val_fault"
                    if fault
                    else "enterpriseops_val_clean",
                    agent_name="recovery_agent",
                    reward_model=dict(style="rule", ground_truth=""),
                    extra_info=dict(index=visit, recovery=recovery),
                )
            )
    pq.write_table(pa.Table.from_pylist(validation), output / "validation.parquet")
    write_json(
        output / "data_summary.json",
        dict(
            train_cells=len(rows),
            train_tasks=len({r["task_id"] for r in rows}),
            task_contract_revision=TASK_CONTRACT_REVISION,
            train_rows=len(train),
            joint_rollout=True,
            joint_group_counts=[16, 16, 0],
            sft_records=len(records),
            data_sources=sources,
            input_fingerprints=file_fingerprints(input_files),
            independent_tasks=True,
            validation_rows=len(validation),
            sft_seen_cells=sum(bool(v) for v in seen.values()),
            sampling_unit="task_x_mode",
            test_used=False,
        ),
    )
    return rows
