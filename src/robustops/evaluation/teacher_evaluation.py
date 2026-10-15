"""Unguided API teacher reference under the adopted fixed Val/Test contract."""
import asyncio
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import subprocess
import sys
import time
from types import SimpleNamespace

from robustops.providers.credentials import credential
from robustops.evaluation import paired_runner as runner
from robustops.evaluation.checkpoint_evaluation import fixed_rows, freeze_backend, nohit_targets, preflight
from robustops.evaluation.fixed_fault_nohit_resample import resample_cell, summarize
from robustops.evaluation.runtime_services import launch as launch_mcp, stop_processes
from robustops.launch.config import LaunchConfigError, resolve_project_path
from robustops.launch.resources import host_memory_bytes
from robustops.launch.inference import PLAN_PRESERVING_EXECUTION


def build_plan(root, recipe):
    if (recipe.get('pipeline') != 'teacher_fixed_test' or recipe.get('guidance_variant') != 'none'
            or recipe.get('thinking') != 'disabled' or recipe.get('execution') != PLAN_PRESERVING_EXECUTION):
        raise LaunchConfigError('Teacher evaluation requires no added guidance, nonthinking and plan-preserving execution')
    for name, expected in dict(temperature=0, samples_per_condition=1, max_turns=50,
                               max_context_tokens=32768, max_generation_tokens=2048,
                               max_nohit_resamples=4, invalid_retries=4).items():
        if recipe.get(name) != expected:
            raise LaunchConfigError(f'Teacher evaluation requires {name}={expected}')
    backend = recipe['backend']
    if backend.get('kind') != 'deepseek_api' or not re.fullmatch(r'[A-Z][A-Z0-9_]*', backend['api_key_env']):
        raise LaunchConfigError('Teacher evaluation requires DeepSeek API with a named credential environment variable')
    if 'api_key' in backend:
        raise LaunchConfigError('Do not store API credentials in recipes; use api_key_env')
    if type(recipe['concurrency']) is not int or recipe['concurrency'] < 1:
        raise LaunchConfigError('concurrency must be a positive integer')
    policy, rows, specs = fixed_rows(root, recipe)
    for name in ('max_turns', 'max_context_tokens', 'max_generation_tokens'):
        if policy[name] != recipe[name]:
            raise LaunchConfigError(f'Teacher budget differs from frozen Base policy: {name}')
    policy = dict(policy, tokenizer=str(resolve_project_path(root, recipe['tokenizer'])))
    selected = recipe.get('selected_cells', [r['cell'] for r in rows])
    if (not selected or len(set(selected)) != len(selected)
            or any(type(cell) is not int or cell not in specs for cell in selected)):
        raise LaunchConfigError('selected_cells must be unique cells from the configured fixed manifest')
    rows = [r for r in rows if r['cell'] in selected]
    plan = dict(model=backend['model'], endpoint=backend['endpoint'], api_key_env=backend['api_key_env'],
                backend_revision=recipe['backend_revision'], rows=rows,
                fixed_fault_specs=[specs[r['cell']] for r in rows],
                policy={k:v for k,v in policy.items() if k != 'rows'}, execution=recipe['execution'],
                guidance_variant='none', thinking='disabled', original_slots=2*len(rows),
                max_nohit_resamples=4, invalid_retries=4, provider_seed_supported=False,
                provider_immutable_revision='not supplied by provider',
                context_counter='Base tokenizer and chat template; full history and tools; reserve 2048 tokens')
    return policy, rows, specs, plan


async def evaluate(root, recipe_path, recipe, *, dry_run):
    policy, rows, specs, plan = build_plan(root, recipe)
    output = resolve_project_path(root, recipe['output']).resolve()
    print(json.dumps(dict(output=str(output), partition=recipe['partition'], cells=len(rows), original_attempts=plan['original_slots'],
        model=plan['model'], api_key_env=plan['api_key_env'], guidance_variant='none',
        execution=recipe['execution'], max_context_tokens=32768, max_generation_tokens=2048,
        max_turns=50, temperature=0, concurrency=recipe['concurrency'],
        provider_seed_supported=False, modes=dict(Counter(r['mode'] for r in rows))), indent=2), flush=True)
    if dry_run:
        return
    backend = dict(recipe['backend'])
    if backend.get('local_env_file'):
        backend['local_env_file'] = str(resolve_project_path(root, backend['local_env_file']))
    try:
        key = credential(backend)
    except ValueError as exc:
        raise LaunchConfigError(f"Set {backend['api_key_env']} or fill the configured local environment file") from exc
    if not (Path(policy['tokenizer']) / 'tokenizer.json').is_file():
        raise LaunchConfigError('Bind a local copy of the Base tokenizer before teacher evaluation')
    if not (root / 'enterpriseops_gym/gym_dbs.zip').is_file():
        raise LaunchConfigError('The pinned EnterpriseOps database archive is required')
    preflight(root, 0, min_host_memory_gib=recipe.get("hardware", {}).get("min_host_memory_gib", 0))
    resuming = output.exists() and recipe['resume']
    if resuming:
        if runner.read(output / 'plan.json') != plan:
            raise LaunchConfigError('Teacher resume plan changed; use a new output directory')
    else:
        output.mkdir(parents=True, exist_ok=False)
        runner.save(output / 'plan.json', plan)
        runner.save(output / 'resolved_inference_config.json', recipe)
        runner.save(output / 'code_state.json', dict(
            created_at=datetime.now(timezone.utc).isoformat(),
            revision=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True).stdout.strip() or 'source-bundle',
            dirty=subprocess.check_output(['git', 'status', '--short'], cwd=root, text=True),
            command=sys.argv, recipe=str(recipe_path)))
        runner.save(output / 'environment.json', dict(
            python=sys.version, executable=sys.executable,
            devices=[],
            host_memory_bytes=host_memory_bytes()))
    freeze_backend(root, output, resume=resuming)
    if ((output / 'completed.json').is_file()
            and runner.read(output / 'completed.json').get('guidance_variant') == 'none'):
        print(f'Already complete: {output}', flush=True)
        return
    model_config = dict(provider='deepseek', model=plan['model'], endpoint=plan['endpoint'],
                        temperature=0, max_tokens=2048)
    evaluation = runner.PairedModelEvaluation('teacher', None, rows, specs, policy, output,
        0, (0,), recipe['experiment_id'], 4, model_config=model_config, api_key=key)
    supplement = output / 'nohit_resamples'
    replica = SimpleNamespace(endpoint=plan['endpoint'], served_name=plan['model'], urls={})
    gate = asyncio.Semaphore(recipe['concurrency'])
    locks = {r['task_id']:asyncio.Lock() for r in rows}
    processes = []
    runtime = output / f'runtime_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}'
    runtime.mkdir()

    async def run_cell(row):
        async with gate, locks[row['task_id']]:
            for condition in ('clean', 'fault'):
                await evaluation.run_group(row, condition, replica)
            if not evaluation.groups[row['cell'], 'fault']['attempts']:
                return
            targets = nohit_targets(output, [row], specs)
            if targets:
                await resample_cell(targets[0], output=supplement, replica=replica,
                    served_name=replica.served_name, policy=policy, counter=evaluation.counter,
                    task=evaluation.tasks[row['cell']], max_attempts=4, invalid_retries=4,
                    model_config=model_config, api_key=key)

    try:
        processes, replica.urls = await launch_mcp({server for server,_ in runner.SERVERS.values()},
                                                  runtime, port_offset=recipe['mcp_port_offset'])
        # First paired cell profiles the real API/MCP path within the final plan.
        started = time.monotonic()
        await run_cell(rows[0])
        print(f'FIRST_CELL_COMPLETE seconds={time.monotonic() - started:.2f}', flush=True)
        async with asyncio.TaskGroup() as group:
            for row in rows[1:]:
                group.create_task(run_cell(row))
        metrics = evaluation.finish()
        valid_fault_rows = [r for r in rows if evaluation.groups[r['cell'], 'fault']['attempts']]
        nohit = summarize(supplement, nohit_targets(output, valid_fault_rows, specs), 4)
        if nohit['pending']:
            raise RuntimeError('Teacher no-hit supplementation is incomplete')
        runner.save(output / 'completed.json', dict(original=metrics, nohit_resampling=nohit,
                    guidance_variant='none', context_counter=plan['context_counter']))
        print(f'COMPLETE {output}', flush=True)
    finally:
        stop_processes(processes)
        runner.save(runtime / 'resource_cleanup.json', dict(mcp_stopped=all(p.poll() is not None for p in processes)))


def launch(root, recipe_path, recipe, *, gpu_override, partition_override, dry_run):
    if gpu_override not in (None, 0) or partition_override not in (None, 'all', recipe['partition']):
        raise LaunchConfigError('Teacher evaluation is API-only and uses its configured Val/Test partition')
    asyncio.run(evaluate(Path(root), recipe_path, recipe, dry_run=dry_run))
    return 0
