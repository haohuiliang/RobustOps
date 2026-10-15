from robustops.evaluation.invalid_resample import resolve_attempt, resolved_path
"""Supplement only Base-kind Val cells whose original fault did not inject."""

import asyncio
import json
from pathlib import Path

from robustops.evaluation.clean_baseline import TokenCounter, _tool_content, run_task_attempt
from robustops.evaluation.runtime_services import Model
from robustops.evaluation.selected_fault import SelectedFaultController
from robustops.envs.enterpriseops.task import load_tasks

ROOT = Path(__file__).resolve().parents[3]
POLICY = ROOT / "artifacts/val_fault_policy/policy.json"
MANIFEST = ROOT / "artifacts/base_val_fixed_mechanisms.json"
SEED_STRIDE = 1_000_003


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def targets_from_source(source, source_kind):
    """Return the exact original zero-hit cells with frozen kind and dose."""
    source = Path(source)
    if source_kind != 'fixed':
        raise ValueError('Seed supplementation is only for an original fixed-fault evaluation')
    policy = read(POLICY)
    manifest = read(MANIFEST)
    rows = {r["cell"]: r for r in policy["rows"]}
    specs = {r["cell"]: r for r in manifest["rows"]}
    if set(rows) != set(range(1, 128)) or set(specs) != set(rows):
        raise ValueError("Val policy and Base-kind manifest must cover all 127 cells")
    if source_kind == "base":
        canonical = {}
        for mode in ("oracle", "plus5"):
            path = source / mode / "val/temperature_0/data/attempts.jsonl"
            for line in path.read_text().splitlines():
                attempt = json.loads(line)
                key = (attempt["task_id"], attempt["mode"])
                if key in canonical or not attempt["valid"] or not attempt["cleanup"]["succeeded"]:
                    raise ValueError(f"Invalid or duplicate Base source: {key}")
                canonical[key] = attempt
        if len(canonical) != 127:
            raise ValueError("Historical Base source must contain 127 valid cells")
    elif source_kind == "fixed":
        if len(read(source / "completion.json")) != 127:
            raise ValueError("Fixed-fault source must be complete: 127 cells")
    else:
        raise ValueError(f"Unknown source kind: {source_kind}")

    targets = []
    for cell in sorted(rows):
        row, spec = rows[cell], specs[cell]
        if any(row[key] != spec[key] for key in ("task_id", "mode", "domain")):
            raise ValueError(f"Manifest identity mismatch: cell {cell}")
        if spec["kind"] not in {kind for tool in row["selection"]["tools"] for kind in tool["faults"]}:
            raise ValueError(f"Disabled Base kind: cell {cell}")
        if source_kind == "base":
            attempt = canonical[(row["task_id"], row["mode"])]
            hits = attempt["fault"]["injected_calls"]
            if hits != spec["base_injected_calls"]:
                raise ValueError(f"Base source and manifest disagree: cell {cell}")
            source_mode = "oracle" if row["mode"] == "oracle" else "plus5"
            source_path = f"{source}/{source_mode}/val/temperature_0/data/attempts.jsonl"
        else:
            source_path = source / "base_fixed_kind" / f"cell_{cell:02}" / "attempt.json"
            attempt = read(source_path)
            if (attempt["task_id"], attempt["mode"]) != (row["task_id"], row["mode"]):
                raise ValueError(f"Fixed source identity mismatch: cell {cell}")
            if not attempt["valid"] or not attempt["cleanup"]["succeeded"]:
                raise ValueError(f"Invalid original fixed result: cell {cell}")
            fault = attempt["fault"]
            if fault.get("target_mechanism") != spec["kind"]:
                raise ValueError(f"Fixed source changed fault kind: cell {cell}")
            hits = fault["injected_calls"]
            if bool(fault.get("no_hit")) != (hits == 0):
                raise ValueError(f"Fixed source hit marker mismatch: cell {cell}")
        if hits == 0:
            require_fixed_zero_hit(attempt, row, spec)
            targets.append(dict(row=row, spec=spec, original_attempt=str(source_path)))
    return targets


def require_fixed_zero_hit(attempt, row, spec):
    """Business failure, partial cap use and invalid collection are not no-hit."""
    fault = attempt.get('fault') or {}
    config = fault.get('config') or {}
    if (not attempt.get('valid') or not attempt.get('cleanup', {}).get('succeeded')
            or not attempt.get('trajectory', {}).get('protocol_valid', True)
            or (attempt.get('task_id'), attempt.get('mode')) != (row['task_id'], row['mode'])
            or fault.get('injected_calls') != 0
            or (fault.get('target_mechanism') or config.get('kind')) != spec['kind']
            or config.get('kind', spec['kind']) != spec['kind']
            or config.get('budget') != spec['budget']
            or type(spec['budget']) is not int or spec['budget'] <= 0):
        raise ValueError('Seed supplementation requires a valid zero-hit attempt under the same fixed kind and cap')


def summarize(output, targets, max_attempts):
    cells = []
    for target in targets:
        cell = target["row"]["cell"]
        attempts = []
        skipped = False
        for index in range(1, max_attempts + 1):
            path = resolved_path(output / f"cell_{cell:02}" / f"attempt_{index:02}")
            if path is None:
                skipped = True
                break
            if not path.exists():
                break
            attempt = read(path)
            if not attempt["valid"] or not attempt["cleanup"]["succeeded"]:
                break
            attempts.append(dict(index=index, seed=attempt["sampling"]["seed"],
                                 injected_calls=attempt["fault"]["injected_calls"],
                                 success=attempt["success"], valid=attempt["valid"], path=str(path)))
            if attempt["fault"]["injected_calls"] > 0:
                break
        cells.append(dict(cell=cell, task_id=target["row"]["task_id"], mode=target["row"]["mode"],
                          target_mechanism=target["spec"]["kind"], budget=target["spec"]["budget"],
                          original_attempt=target["original_attempt"], attempts=attempts, skipped=skipped,
                          hit=bool(attempts and attempts[-1]["injected_calls"] > 0),
                          exhausted=len(attempts) == max_attempts and (not attempts or attempts[-1]["injected_calls"] == 0)))
    result = dict(assigned=len(cells), hit=sum(c["hit"] for c in cells),
                  exhausted=sum(c["exhausted"] for c in cells),
                  skipped=sum(c["skipped"] for c in cells),
                  pending=sum(not c["hit"] and not c["exhausted"] and not c["skipped"] for c in cells),
                  supplemental_attempts=sum(len(c["attempts"]) for c in cells), cells=cells)
    save(output / "summary.json", result)
    return result


def ensure_resample_plan(*, source, source_kind, output, model_path, served_name, targets,
                         max_attempts=4, policy=None):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    expected_plan = dict(source=str(Path(source).resolve()), source_kind=source_kind,
                         model_path=str(Path(model_path).resolve()), served_name=served_name,
                         temperature=0, max_supplemental_attempts=max_attempts,
                         seed_stride=SEED_STRIDE, target_cells=[t["row"]["cell"] for t in targets])
    if policy is not None:
        expected_plan['policy'] = {k: v for k, v in policy.items() if k != 'rows'}
    plan_path = output / "plan.json"
    if plan_path.exists():
        if read(plan_path) != expected_plan:
            raise ValueError(f"Cannot resume with a different resample plan: {output}")
    else:
        save(plan_path, expected_plan)


async def resample_cell(target, *, output, replica, served_name, policy, counter, task,
                        max_attempts=4, invalid_retries=0, progress=None, model_config=None, api_key=None,
                        temperature=0):
    """Finish one cell's bounded seed chain on an already loaded model replica."""
    output = Path(output)
    if progress is None:
        progress = lambda: summarize(output, [target], max_attempts)
    row, spec = target["row"], target["spec"]
    original = read(target['original_attempt'])
    require_fixed_zero_hit(original, row, spec)
    if original.get('sampling', {}).get('temperature', 0) != temperature:
        raise ValueError('No-hit supplementation must retain the original temperature')
    for index in range(1, max_attempts + 1):
        path = output / f"cell_{row['cell']:02}" / f"attempt_{index:02}"
        seed = row["seed"] + index * SEED_STRIDE
        async def generate(path, seed):
            result_path = path / "attempt.json"
            (path / "calls").mkdir(parents=True)
            controller = SelectedFaultController(row["payload"], row["selection"],
                spec["kind"], spec["budget"], "eligible_calls",
                lambda value: counter.text(_tool_content(value)))
            config = dict(model_config or policy["base_model"], model=getattr(replica, "served_name", served_name),
                          temperature=temperature, endpoint=replica.endpoint)
            model = (Model(config, path / "calls", seed) if model_config is None
                     else Model(config, path / "calls", seed, api_key))
            try:
                attempt = await run_task_attempt(
                    task=task, model=model, token_counter=counter,
                    archive_path=ROOT / "enterpriseops_gym/gym_dbs.zip",
                    mcp_url_overrides=replica.urls, max_turns=policy["max_turns"],
                    max_context_tokens=policy["max_context_tokens"],
                    max_generation_tokens=policy["max_generation_tokens"],
                    experiment_id="BASE-KIND-NOHIT-RESAMPLE", run_id=output.parent.name,
                    mode=row["mode"], model_name=served_name,
                    model_endpoint=replica.endpoint, seed=seed, attempt_index=index,
                    plan_preserving=True, tool_intervention=controller)
                hits = attempt["fault"]["injected_calls"]
                attempt["fault"].update(target_mechanism=spec["kind"],
                    selected_mechanism=spec["kind"] if hits else None,
                    no_hit=hits == 0, base_injected_calls=spec["base_injected_calls"])
                attempt["sampling"] = dict(temperature=temperature, seed=seed,
                    supplemental_attempt_index=index)
                save(result_path, attempt)
                save(path / "tool_audit.json", controller.calls)
            finally:
                await model.client.close()
            return attempt
        if invalid_retries:
            attempt, result_path = await resolve_attempt(path, seed, row,
                dict(temperature=temperature, supplemental_attempt_index=index), generate, invalid_retries)
            if attempt is None:
                progress()
                print(f"RESAMPLE cell={row['cell']} skipped_invalid", flush=True)
                break
        else:
            result_path = path / "attempt.json"
            attempt = read(result_path) if result_path.exists() else None
            if attempt is not None and (attempt['sampling']['seed'] != seed or attempt['fault']['target_mechanism'] != spec['kind']):
                raise ValueError(f"Existing supplemental attempt changed seed or fault: {result_path}")
            if attempt is None or not attempt['valid'] or not attempt['cleanup']['succeeded']:
                if path.exists():
                    retry = 1
                    while path.with_name(f'{path.name}_invalid_{retry}').exists():
                        retry += 1
                    path.rename(path.with_name(f'{path.name}_invalid_{retry}'))
                attempt = await generate(path, seed)
            if (attempt['sampling']['seed'] != seed or attempt['fault']['target_mechanism'] != spec['kind']):
                raise ValueError(f"Existing supplemental attempt changed seed or fault: {result_path}")
            if not attempt['valid'] or not attempt['cleanup']['succeeded']:
                raise RuntimeError(f"Invalid supplemental attempt: {result_path}")
        summary = progress()
        print(f"RESAMPLE {output.name} cell={row['cell']} attempt={index} seed={seed} "
              f"hits={attempt['fault']['injected_calls']} progress={summary['hit']}/{summary['assigned']}",
              flush=True)
        if attempt["fault"]["injected_calls"] > 0:
            break


async def run_resamples(*, source, source_kind, output, replica, model_path,
                        served_name, max_attempts=4, targets=None, invalid_retries=0,
                        policy=None):
    output = Path(output)
    targets = targets if targets is not None else targets_from_source(source, source_kind)
    if max_attempts != 4:
        raise ValueError("This batch is defined as exactly four supplemental seeds")
    output.mkdir(parents=True, exist_ok=True)
    ensure_resample_plan(source=source, source_kind=source_kind, output=output,
        model_path=model_path, served_name=served_name, targets=targets,
        max_attempts=max_attempts, policy=policy)
    policy = policy if policy is not None else read(POLICY)
    counter = TokenCounter(ROOT / policy["tokenizer"], enable_thinking=False)
    tasks = {}
    for target in targets:
        row = target["row"]
        recipe = read(ROOT / f"recipes/enterpriseops/paired/{'oracle' if row['mode'] == 'oracle' else 'plus5'}/{row['domain']}.json")
        tasks[row["cell"]] = next(t for t in load_tasks(ROOT / recipe["data_path"],
            expected_domain=row["domain"], expected_count=recipe["expected_task_count"])
            if t.task_id == row["task_id"])
        task = tasks[row["cell"]]
        if (task.system_prompt != row["payload"]["task"]["system_policy"] or
            task.user_prompt != row["payload"]["task"]["user_request"] or
            list(task.verifiers) != row["payload"]["verifiers"]):
            raise ValueError(f"Task definition changed: cell {row['cell']}")

    gate = asyncio.Semaphore(2)

    async def one_cell(target):
        async with gate:
            await resample_cell(target, output=output, replica=replica, served_name=served_name,
                policy=policy, counter=counter, task=tasks[target['row']['cell']],
                max_attempts=max_attempts, invalid_retries=invalid_retries,
                progress=lambda: summarize(output, targets, max_attempts))

    async with asyncio.TaskGroup() as group:
        for target in targets:
            group.create_task(one_cell(target))
    summary = summarize(output, targets, max_attempts)
    if summary["pending"]:
        raise RuntimeError(f"Supplemental batch incomplete: {output}")
    save(output / "completed.json", dict(summary=summary, source=str(source)))
    return summary
