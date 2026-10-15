"""Config-driven fixed Val/Test evaluation for local model checkpoints."""
import asyncio
from collections import Counter, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from robustops.evaluation import paired_runner as runner
from robustops.evaluation.fixed_fault_nohit_resample import (
    ensure_resample_plan, resample_cell, summarize, require_fixed_zero_hit,
)
from robustops.evaluation.invalid_resample import resolved_path
from robustops.models.checkpoint_validation import sft_checkpoint_path
from robustops.launch.config import LaunchConfigError, resolve_project_path
from robustops.training.recovery_rl_storage import validate_hf_export


def fixed_rows(root, recipe):
    policy = runner.read(resolve_project_path(root, recipe['selector_policy']))
    manifest = runner.read(resolve_project_path(root, recipe['base_fixed_mechanisms']))
    rows = policy['rows']
    specs = {r['cell']: r for r in manifest['rows']}
    partition = recipe['partition']
    expected = {'test': 130, 'validation': 127}.get(partition)
    if (expected is None or recipe['expected_task_modes'] != expected
            or manifest['partition'] != partition
            or (partition == 'test' and manifest['budget_revision'] != 'unknown_commit_cap2')
            or len(rows) != expected or len(manifest['rows']) != expected
            or {r['cell'] for r in rows} != set(range(1, expected + 1)) or set(specs) != set(range(1, expected + 1))
            or len({(r['task_id'], r['mode']) for r in rows}) != expected):
        raise LaunchConfigError('Expected the full fixed Val/Test policy and matching Base manifest')
    budgets = runner.read(resolve_project_path(root, manifest['candidate_budgets']))
    for row in rows:
        spec = specs[row['cell']]
        kinds = {k for tool in row['selection']['tools'] for k in tool['faults']}
        if (row['partition'] != partition or row['mode'] not in ('oracle', 'plus_5_tools')
                or any(row[k] != spec[k] for k in ('task_id', 'mode', 'domain'))
                or spec['kind'] not in kinds or spec['budget'] != budgets[str(row['cell'])][spec['kind']]):
            raise LaunchConfigError(f'Fixed Test fault/identity mismatch: {row["cell"]}')
    return policy, rows, specs


def model_exports(root, recipe):
    models = {}
    for spec in recipe['models']:
        key = spec['key']
        if not key.isidentifier() or key in models:
            raise LaunchConfigError(f'Invalid or duplicate model key: {key}')
        if 'path' in spec:
            path = resolve_project_path(root, spec['path']).resolve()
            dtype = spec.get('weight_dtype', 'float32')
            if dtype not in ('float32', 'bfloat16'):
                raise LaunchConfigError(f'Unsupported preserved weight dtype: {dtype}')
            details = validate_hf_export(path, allowed_dtypes=('F32',) if dtype == 'float32' else ('BF16',)) if list(path.glob('*.safetensors')) else None
            models[key] = dict(path=str(path), ready=details is not None,
                               checkpoint=spec.get('checkpoint', path.name), details=details)
        elif 'sft_train_run' in spec:
            train = resolve_project_path(root, spec['sft_train_run'])
            path, step = sft_checkpoint_path(train, spec.get('step'))
            details = validate_hf_export(path, allowed_dtypes=('F32', 'F16', 'BF16')) if list(path.glob('*.safetensors')) else None
            models[key] = dict(path=str(path.resolve()), ready=details is not None,
                               checkpoint=f'global_step_{step}', details=details)
        else:
            marker = resolve_project_path(root, spec['export_record'])
            record = runner.read(marker) if marker.is_file() else None
            if record is not None and record['step'] != spec['step']:
                raise LaunchConfigError(f'Unexpected HF export step: {marker}')
            path = resolve_project_path(root, record['path']).resolve() if record else None
            details = validate_hf_export(path) if path is not None else None
            models[key] = dict(path=str(path) if path else None, ready=details is not None,
                export_record=str(marker), checkpoint=f'global_step_{spec["step"]}', details=details)
    if not models:
        raise LaunchConfigError('At least one model is required')
    return models


def nohit_targets(output, rows, specs):
    targets = []
    for row in rows:
        path = resolved_path(output / 'fault' / f'cell_{row["cell"]:02}' / 'attempt_01')
        if path is None:  # Shared invalid/protocol retry budget exhausted.
            continue
        attempt = runner.read(path)
        runner.validate(attempt, row)
        fault, spec = attempt['fault'], specs[row['cell']]
        if (fault['target_mechanism'] != spec['kind'] or fault['config']['budget'] != spec['budget']
                or fault['no_hit'] != (fault['injected_calls'] == 0)):
            raise ValueError(f'Original Test fixed fault differs: {path}')
        if fault['injected_calls'] == 0:
            require_fixed_zero_hit(attempt, row, spec)
            targets.append(dict(row=row, spec=spec, original_attempt=str(path)))
    return targets


class ModelWork:
    """Shared pending groups; a GPU keeps its model through original and no-hit work."""

    def __init__(self, key, model, output, rows, specs, policy, experiment_id, *,
                 temperature=0.0, seed_indices=(0,), max_nohit_resamples=4, invalid_retries=4):
        self.key, self.model, self.output = key, model, output
        self.rows, self.specs, self.policy = rows, specs, policy
        self.evaluation = runner.PairedModelEvaluation(key, None, rows, specs, policy,
            output, temperature, seed_indices, experiment_id, invalid_retries)
        self.samples = len(seed_indices)
        self.max_nohit_resamples = max_nohit_resamples
        self.pending, self.active_gpus, self.task_seconds = deque(), set(), []
        self.nohit_output = output / 'nohit_resamples'
        self.nohit_name = f'Recovery-{key}-test-nohit'
        self.completed = False

    def targets(self):
        resolved = [r for r in self.rows
                    if self.evaluation.groups[r['cell'], 'fault']['attempts']]
        return nohit_targets(self.output, resolved, self.specs)

    def resample_plan(self):
        ensure_resample_plan(source=self.output, source_kind='paired_test_t0',
            output=self.nohit_output, model_path=self.model, served_name=self.nohit_name,
            targets=self.targets(), policy=self.policy)

    async def prepare(self):
        # Reconstruct completed groups before loading a model, even after a partial run.
        for row in self.rows:
            for condition in ('clean', 'fault'):
                markers = [self.output / condition / f'cell_{row["cell"]:02}' / f'attempt_{i:02}' / 'resolution.json'
                           for i in range(1, self.samples + 1)]
                statuses = [runner.read(p) if p.is_file() else {} for p in markers]
                if all(s.get('selected') or s.get('skipped') for s in statuses):
                    await self.evaluation.run_group(row, condition, None)
                else:
                    self.pending.append((row, condition))
        if not self.max_nohit_resamples:
            return
        if (self.nohit_output / 'plan.json').is_file():
            self.resample_plan()
        previous = summarize(self.nohit_output, self.targets(), 4)
        by_cell = {c['cell']: c for c in previous['cells']}
        for row in self.rows:
            cell = by_cell.get(row['cell'])
            if cell and not (cell['hit'] or cell['exhausted'] or cell['skipped']):
                self.pending.append((row, 'nohit'))

    async def run(self, row, condition, replica):
        if condition != 'nohit':
            await self.evaluation.run_group(row, condition, replica)
        if self.max_nohit_resamples and condition in ('fault', 'nohit'):
            targets = nohit_targets(self.output, [row], self.specs)
            if targets:
                await resample_cell(targets[0], output=self.nohit_output, replica=replica,
                    served_name=self.nohit_name, policy=self.policy,
                    counter=self.evaluation.counter, task=self.evaluation.tasks[row['cell']],
                    invalid_retries=4,
                    progress=lambda: summarize(self.nohit_output, self.targets(), 4))

    def finish(self):
        if self.completed:
            return
        self.evaluation.finish()
        if not self.max_nohit_resamples:
            self.completed = True
            return
        self.resample_plan()
        summary = summarize(self.nohit_output, self.targets(), 4)
        if summary['pending']:
            raise RuntimeError(f'Supplemental batch incomplete: {self.nohit_output}')
        runner.save(self.nohit_output / 'completed.json', dict(summary=summary, source=str(self.output)))
        self.completed = True


def choose_work(works, switch_seconds, concurrency_per_gpu=2, model_switch_seconds=None):
    pending = [w for w in works if w.pending]
    # Start untouched models while the other GPU finishes its current model.
    free = next((w for w in pending if not w.active_gpus), None)
    if free is not None:
        return free
    observed = [t for w in works for t in w.task_seconds]
    def queued_seconds_per_replica(work):
        samples = work.task_seconds[-20:] or observed[-20:]
        mean_seconds = sum(samples) / len(samples) if samples else 1.0
        return len(work.pending) * mean_seconds / len(work.active_gpus)

    # Balance replica counts before timings exist; then favor the longer queue in time.
    for work in sorted(pending, key=queued_seconds_per_replica, reverse=True):
        samples = work.task_seconds[-20:] or observed[-20:]
        if not samples and len(work.pending) <= concurrency_per_gpu:
            continue
        remaining = (len(work.pending) * sum(samples) / len(samples)
                     / (concurrency_per_gpu * len(work.active_gpus))) if samples else float('inf')
        # Loading a second copy helps only if the current GPU will still have queued work.
        cost = (model_switch_seconds or {}).get(getattr(work, 'key', None), switch_seconds)
        if remaining > cost:
            return work
    return None


async def schedule_models(models, output, rows, specs, policy, experiment_id, gpus, *, concurrency_per_gpu=2,
                          temperature=0.0, seed_indices=(0,), max_nohit_resamples=4, invalid_retries=4):
    works = []
    for key, model in models.items():
        model_output = output / key
        if ((model_output / 'completed.json').is_file()
                and (not max_nohit_resamples or (model_output / 'nohit_resamples/completed.json').is_file())):
            continue
        work = ModelWork(key, Path(model['path']), model_output, rows, specs, policy, experiment_id,
            temperature=temperature, seed_indices=seed_indices,
            max_nohit_resamples=max_nohit_resamples, invalid_retries=invalid_retries)
        await work.prepare()
        if not work.pending:
            work.finish()
        works.append(work)
    execution = dict(gpus=gpus, scheduling='shared_task_queues', concurrency_per_gpu=concurrency_per_gpu,
                     gpu_assignment={w.key: [] for w in works}, model_loads=[], deferred_joins=[])

    def record():
        runner.save(output / 'execution.json', execution)

    async def worker(gpu):
        switch_seconds, unload_seconds = 0.0, 0.0
        while True:
            model_switch_seconds = {}
            for candidate in works:
                loads = [item['load_seconds'] for item in execution['model_loads']
                         if item['model'] == candidate.key and 'load_seconds' in item]
                if loads:
                    model_switch_seconds[candidate.key] = unload_seconds + sum(loads) / len(loads)
            work = choose_work(works, switch_seconds, concurrency_per_gpu, model_switch_seconds)
            if work is None:
                if any(w.pending for w in works):
                    execution['deferred_joins'].append(dict(gpu=gpu, switch_seconds=switch_seconds,
                        pending={w.key: len(w.pending) for w in works if w.pending}))
                    record()
                return
            work.active_gpus.add(gpu)
            execution['gpu_assignment'][work.key].append(gpu)
            runtime = work.output / f'runtime_{datetime.now(timezone.utc):%Y%m%dT%H%M%S%f}-{os.getpid()}-gpu{gpu}'
            replica = runner.FaultReplica(gpu, work.model, runner.ROOT / policy['tokenizer'],
                f'SFT-{work.key}-supplement', runtime, max_context_tokens=policy['max_context_tokens'])
            started = time.monotonic()
            load = dict(model=work.key, gpu=gpu, pending_at_start=len(work.pending))
            execution['model_loads'].append(load)
            record()
            try:
                await replica.start({server for server, _ in runner.SERVERS.values()})
                load['load_seconds'] = time.monotonic() - started
                switch_seconds = unload_seconds + load['load_seconds']
                record()

                async def consume():
                    while work.pending:
                        row, condition = work.pending.popleft()
                        started = time.monotonic()
                        await work.run(row, condition, replica)
                        work.task_seconds.append(time.monotonic() - started)

                async with asyncio.TaskGroup() as group:
                    for _ in range(concurrency_per_gpu):
                        group.create_task(consume())
            finally:
                started = time.monotonic()
                await asyncio.to_thread(replica.close)
                unload_seconds = time.monotonic() - started
                switch_seconds = unload_seconds + load.get('load_seconds', 0.0)
                load.update(unload_seconds=unload_seconds, switch_seconds=switch_seconds)
                work.active_gpus.remove(gpu)
            if not work.pending and not work.active_gpus:
                work.finish()

    async with asyncio.TaskGroup() as group:
        for gpu in range(gpus):
            group.create_task(worker(gpu))
    for work in works:
        work.finish()
    record()


def backend_files(root):
    compat = Path('backends/enterpriseops')
    return ([compat / f'bootstrap_{domain}.py' for domain in runner.SERVERS]
            + [compat / 'backend_errors.py', compat / 'launch_udocker.sh',
               Path('src/robustops/envs/enterpriseops/runtime.py'),
               Path('src/robustops/evaluation/runtime_services.py')])


def freeze_backend(root, batch, *, resume):
    """Record actual current sources; prohibit mixing backend versions on resume."""
    for relative in backend_files(root):
        source = root / relative
        target = batch / 'code' / relative
        if resume:
            if source.read_bytes() != target.read_bytes():
                raise LaunchConfigError(f'Backend changed since this Test batch; use a new output: {relative}')
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)


def preflight(root, gpus, *, min_host_memory_gib=0):
    from robustops.launch.hardware import device_for_worker
    from robustops.launch.resources import check_host_memory
    check_host_memory(min_host_memory_gib)
    if gpus:
        subprocess.check_output(['nvidia-smi', '-L'], text=True)
    for gpu in range(gpus):
        subprocess.run(['nvidia-smi', '-i', device_for_worker(gpu).uuid,
                        '--query-gpu=name', '--format=csv,noheader'], check=True)


async def evaluate(root, recipe_path, recipe, *, gpus, dry_run):
    temperature, seed_indices, nohit = sampling_config(recipe)
    samples = len(seed_indices)
    concurrency = recipe.get('concurrency_per_gpu', 2)
    if type(concurrency) is not int or concurrency not in (1, 2):
        raise LaunchConfigError('concurrency_per_gpu must be 1 or 2')
    policy, rows, specs = fixed_rows(root, recipe)
    models = model_exports(root, recipe)
    output = resolve_project_path(root, recipe['output']).resolve()
    plan = dict(models=models, rows=rows, fixed_fault_specs=[specs[r['cell']] for r in rows],
        policy={k: v for k, v in policy.items() if k != 'rows'},
        experiment_id=recipe['experiment_id'], backend_revision=recipe['backend_revision'],
        temperature=temperature, samples_per_condition=samples, seed_indices=list(seed_indices),
        conditions=['clean', 'fault'], seed_stride=runner.SEED_STRIDE,
        max_nohit_resamples=nohit, stop_supplement_on_first_hit=bool(nohit), invalid_retries=4,
        concurrency_per_gpu=concurrency, planned_original_slots=2 * samples * len(rows) * len(models))
    plan = json.loads(json.dumps(plan))
    missing = [key for key, model in models.items() if not model['ready']]
    print(json.dumps(dict(models=models, output=str(output), partition=recipe['partition'],
        runtime_hardware=recipe.get('runtime_hardware'),
        backend_revision=recipe['backend_revision'], backend_launcher='current workspace launch_udocker.sh',
        cells_per_model=len(rows), original_attempts_per_model=2 * samples * len(rows),
        original_attempts=plan['planned_original_slots'], temperature=temperature, gpus=gpus,
        samples_per_condition=samples, seed_indices=list(seed_indices),
        models_parallel=min(gpus, len(models)), scheduling='shared_task_queues', concurrency_per_gpu=concurrency,
        reuse_replica_for_nohit=bool(nohit), join_after_loading_cost_check=True, nohit_resampling=bool(nohit),
        max_nohit_resamples=nohit, missing_models=missing, resume=recipe['resume'],
        modes=dict(Counter(r['mode'] for r in rows))), ensure_ascii=False, indent=2), flush=True)
    if dry_run:
        return
    if missing:
        raise LaunchConfigError(f'HF exports not ready: {", ".join(missing)}; finish export before Test evaluation')
    resume = output.exists() and recipe['resume']
    if resume:
        if runner.read(output / 'plan.json') != plan:
            raise LaunchConfigError('Resume plan changed: models, Test tasks, fixed faults or sampling differ')
        freeze_backend(root, output, resume=True)
    preflight(root, gpus, min_host_memory_gib=recipe.get("hardware", {}).get("min_host_memory_gib", 0))
    if not resume:
        output.mkdir(parents=True, exist_ok=False)
        freeze_backend(root, output, resume=False)
        runner.save(output / 'plan.json', plan)
        runner.save(output / 'resolved_inference_config.json', recipe)
        runner.save(output / 'code_state.json', dict(created_at=datetime.now(timezone.utc).isoformat(),
            revision=subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, text=True, capture_output=True).stdout.strip() or 'source-bundle',
            dirty=subprocess.run(['git', 'status', '--short'], cwd=root, text=True, capture_output=True).stdout.strip(),
            command=sys.argv, recipe=str(recipe_path), backend_files=[str(p) for p in backend_files(root)]))
    runner.save(output / 'runtime_hardware.json', recipe.get('runtime_hardware'))
    await schedule_models(models, output, rows, specs, policy, recipe['experiment_id'], gpus,
                          concurrency_per_gpu=concurrency, temperature=temperature,
                          seed_indices=seed_indices, max_nohit_resamples=nohit, invalid_retries=4)
    summaries = {key: runner.read(output / key / 'summary.json') for key in models}
    for key, summary in summaries.items():
        groups = summary['groups']
        if (len(groups) != 2 * len(rows) or len({(g['cell'], g['condition']) for g in groups}) != 2 * len(rows)
                or any(m['resolved_groups'] != len(rows) for m in summary['metrics'].values())):
            raise ValueError(f'Incomplete full Test: {key}')
    nohit = {key: runner.read(output / key / 'nohit_resamples/summary.json') for key in models} if nohit else {}
    histories = [runner.read(p) for key in models for p in (output / key).rglob('resolution.json')]
    invalid = sum(h['invalid_attempts'] for h in histories)
    protocol = sum(h.get('model_protocol_attempts', 0) for h in histories)
    actual = sum(1 for key in models for _ in (output / key).rglob('attempt.json'))
    runner.save(output / 'summary.json', dict(original={key: s['metrics'] for key, s in summaries.items()},
        nohit_resampling=nohit, invalid_attempts=invalid, model_protocol_attempts=protocol,
        backend_revision=recipe['backend_revision']))
    runner.save(output / 'completed.json', dict(models=list(models), planned_original_slots=plan['planned_original_slots'],
        valid_original_attempts=sum(m['new_attempts'] for s in summaries.values() for m in s['metrics'].values()),
        supplemental_attempts=sum(s['supplemental_attempts'] for s in nohit.values()),
        invalid_attempts=invalid, model_protocol_attempts=protocol, total_attempts=actual))
    print(f'COMPLETE {output}', flush=True)


def sampling_config(recipe):
    """Keep the fixed T0 panel and independent Val Pass@4 protocols explicit."""
    if recipe['invalid_retries'] != 4:
        raise LaunchConfigError('Fixed evaluation uses four shared invalid/protocol retries')
    if recipe['temperature'] == 0 and recipe['samples_per_condition'] == 1 and recipe['max_nohit_resamples'] == 4:
        return 0.0, (0,), 4
    if (recipe['partition'] == 'validation' and recipe['temperature'] == .6
            and recipe['samples_per_condition'] == 4 and recipe['max_nohit_resamples'] == 0):
        return .6, (11, 12, 13, 14), 0
    raise LaunchConfigError('Use fixed T0 Pass@1 or validation T0.6 Pass@4 sampling')


def launch(root, recipe_path, recipe, *, gpu_override, partition_override, dry_run):
    from robustops.launch.hardware import select_gpus
    configured = recipe.get('hardware', {}).get('gpus')
    selected = select_gpus(gpu_override if gpu_override is not None else configured, dry_run=dry_run)
    gpus = selected.count
    hardware_record = selected.record()
    if gpu_override is None and configured is not None:
        hardware_record['selection'] = 'recipe'
    recipe = dict(recipe, runtime_hardware=hardware_record)
    if partition_override not in (None, 'all', recipe['partition']):
        raise LaunchConfigError('Fixed evaluation partition must match the recipe')
    sampling_config(recipe)
    asyncio.run(evaluate(root, recipe_path, recipe, gpus=gpus, dry_run=dry_run))
    return 0
