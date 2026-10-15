"""Shared input preparation and API batching for the current fault selector."""
import asyncio
from collections import Counter
import json
from pathlib import Path

from robustops.providers.credentials import credential
from robustops.envs.enterpriseops.task import load_tasks
from robustops.evaluation.clean_baseline import OpenAIChatClient, _openai_tools


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(config, output, *, include_runtime=False):
    """Load only runnable effective tasks, through the project's overlay loader."""
    manifest = [json.loads(line) for line in Path(config['manifest']).read_text().splitlines() if line.strip()]
    partition = [r for r in manifest if r['partition'] == config['partition']]
    targets = [r for r in partition if not r.get('execution_blocked_reason') and r['runnable_modes']]
    if not targets:
        raise ValueError('no runnable tasks in configured partition')
    catalog = {k: v['tools'] for k, v in json.loads(Path(config['schema_catalog']).read_text())['servers'].items()}
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode, directory in config['modes'].items():
        for domain in sorted({r['domain'] for r in targets}):
            selected = sorted((r for r in targets if r['domain'] == domain and mode in r['runnable_modes']), key=lambda r: r['task_id'])
            if not selected:
                continue
            recipe = json.loads((Path(config['task_recipes']) / directory / f'{domain}.json').read_text())
            tasks = {t.task_id: t for t in load_tasks(Path(recipe['data_path']), expected_domain=domain, expected_count=recipe['expected_task_count'])}
            for row in selected:
                task = tasks[row['task_id']]
                names = {s['mcp_server_name'] for s in task.gym_servers}
                tools, _, routes, omitted = _openai_tools({n: catalog[n] for n in names}, task)
                cell = len(rows) + 1
                payload = {'task_id': task.task_id, 'mode': mode, 'task': {'system_policy': task.system_prompt, 'user_request': task.user_prompt}, 'tools': tools}
                if include_runtime:
                    payload.update(tool_servers=routes, verifiers=list(task.verifiers))
                path = inputs / f'cell_{cell:04}.json'
                if path.exists() and json.loads(path.read_text()) != payload:
                    raise ValueError(f'prepared input changed: {path}; use a new output directory')
                write_json(path, payload)
                rows.append({'cell': cell, 'task_id': task.task_id, 'domain': domain, 'mode': mode, 'input': str(path), 'source': recipe['data_path'], 'tool_count': len(tools), 'registered_omissions': list(omitted)})
            del tasks
    if not rows:
        raise ValueError('no task/mode pairs')
    write_json(output / 'cells.json', rows)
    write_json(output / 'preparation.json', {'partition': config['partition'], 'manifest_records': len(partition), 'excluded_tasks': [r['task_id'] for r in partition if r not in targets], 'unique_tasks': len({r['task_id'] for r in rows}), 'pairs': len(rows), 'modes': dict(Counter(r['mode'] for r in rows))})
    return rows


def successful_attempt(directory):
    for attempt in sorted(directory.glob('attempt_*'), reverse=True):
        validation = attempt / 'validation.json'
        if validation.exists() and json.loads(validation.read_text()).get('structural_valid'):
            return attempt
    return None


def summarize(output, rows):
    valid = 0
    coverage = {}
    usage = Counter()
    rejected = 0
    with (output / 'proposals.jsonl').open('w') as stream:
        for row in rows:
            directory = output / f"cell_{row['cell']:04}"
            for path in directory.glob('attempt_*/*response.json'):
                u = json.loads(path.read_text()).get('usage') or {}
                usage.update({k: v for k, v in u.items() if isinstance(v, (int, float))})
            attempt = successful_attempt(directory)
            if attempt is None:
                continue
            valid += 1
            result = json.loads((attempt / 'proposal.json').read_text())
            stream.write(json.dumps(result, ensure_ascii=False) + '\n')
            rejected += len(json.loads((attempt / 'validation.json').read_text())['rejected_paths'])
            for fault in {f for t in result['tools'] for f in t['faults'] + t['ood_faults']}:
                group = coverage.setdefault(fault, {'pairs': 0, 'tasks': set()})
                group['pairs'] += 1
                group['tasks'].add(row['task_id'])
    result = {'total_pairs': len(rows), 'valid_pairs': valid, 'pending_or_failed_pairs': len(rows) - valid, 'rejected_paths': rejected, 'usage_all_attempts': dict(usage), 'conditional_coverage': {k: {'pairs': v['pairs'], 'tasks': len(v['tasks'])} for k, v in coverage.items()}}
    write_json(output / 'summary.json', result)
    return result


async def run(config, output, rows, *, selector):
    provider = config['provider']
    client = OpenAIChatClient(endpoint=provider['base_url'], model_name=provider['model'], temperature=0, max_tokens=config['max_tokens'], seed=None, enable_thinking=False, provider='deepseek', api_key=credential(provider))
    semaphore = asyncio.Semaphore(config['concurrency'])

    async def one(row):
        async with semaphore:
            directory = output / f"cell_{row['cell']:04}"
            if successful_attempt(directory):
                return
            directory.mkdir(exist_ok=True)
            numbers = [int(p.name.split('_')[1]) for p in directory.glob('attempt_*')]
            attempt = directory / f'attempt_{max(numbers, default=0) + 1:04}'
            attempt.mkdir()
            try:
                await selector(json.loads(Path(row['input']).read_text()), attempt, client, extended_paths=config.get('extended_creation_paths', False))
                print(row['cell'], row['task_id'], row['mode'], 'valid', flush=True)
            except Exception as error:
                write_json(attempt / 'error.json', {'error': str(error)})
                print(row['cell'], 'failed; see', attempt / 'error.json', flush=True)
    try:
        await asyncio.gather(*(one(row) for row in rows))
    finally:
        await client.close()
        print(json.dumps(summarize(output, rows), ensure_ascii=False), flush=True)
