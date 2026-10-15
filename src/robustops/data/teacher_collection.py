"""Collect and export fresh teacher supervision from a selected Train collection manifest."""
import argparse
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys

from robustops.data.collection_pool import load_collection_pool

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value, *, frozen=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if frozen and path.exists() and read(path) != value:
        raise ValueError(f'Frozen teacher input changed: {path}')
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def prepare(root, config):
    from robustops.envs.enterpriseops.task import load_tasks
    from robustops.evaluation.clean_baseline import _openai_tools
    manifest = {r['task_id']: r for r in map(json.loads, (root / config['manifest']).read_text().splitlines())}
    pool = load_collection_pool(root / config['pool'], manifest.values())
    selectors = {(r['task_id'], r['mode']): r for r in map(json.loads, (root / config['selector']).read_text().splitlines())}
    catalog = {key: value['tools'] for key, value in read(root / config['schema_catalog'])['servers'].items()}
    teacher = config['teacher']
    allowed = set(teacher['train_faults'])
    if allowed & set(teacher['heldout_faults']) or 'ignored_filter' in allowed:
        raise ValueError('Held-out mechanisms must not enter teacher collection')
    cache, rows = {}, []
    for member in pool['tasks']:
        meta = manifest[member['task_id']]
        if meta['partition'] != 'train' or meta.get('execution_blocked_reason'):
            raise ValueError('Teacher collection must use runnable Train tasks only')
        for mode in teacher['modes']:
            if mode not in member['modes']:
                continue
            if mode not in meta['runnable_modes']:
                raise ValueError('Frozen SFT pool contains an unrunnable mode')
            domain = member['domain']
            key = domain, mode
            if key not in cache:
                directory = 'oracle' if mode == 'oracle' else 'plus5'
                recipe = read(root / f'recipes/enterpriseops/paired/{directory}/{domain}.json')
                cache[key] = {task.task_id: task for task in load_tasks(root / recipe['data_path'],
                    expected_domain=domain, expected_count=recipe['expected_task_count'])}
            task = cache[key][member['task_id']]
            servers = {s['mcp_server_name'] for s in task.gym_servers}
            tools, _, routes, _ = _openai_tools({s: catalog[s] for s in servers}, task)
            selection = deepcopy(selectors[task.task_id, mode]['tools'])
            for tool in selection:
                tool['faults'] = [kind for kind in tool['faults'] if kind in allowed]
            payload = dict(task_id=task.task_id, mode=mode,
                task=dict(system_policy=task.system_prompt, user_request=task.user_prompt),
                tools=tools, tool_servers=routes, verifiers=list(task.verifiers))
            rows.append(dict(cell=len(rows)+1, task_id=task.task_id, domain=domain, mode=mode,
                             partition='train', payload=payload, selection=dict(tools=selection)))
    folder = root / config['output'] / 'inputs'
    write(folder / 'tasks.json', rows, frozen=True)
    write(folder / 'pool.json', pool, frozen=True)
    run = dict(teacher, tasks=str(folder / 'tasks.json'), pool=str(folder / 'pool.json'),
               output=str(root / config['output'] / 'temperature_0'),
               expected_tasks=len(pool['tasks']), expected_task_modes=len(rows),
               experiment_id='teacher_train')
    write(folder / 'round1.json', run, frozen=True)
    return run, rows


def collect(root, config):
    run, rows = prepare(root, config)
    folder = root / config['output'] / 'inputs'
    def launch(spec, name):
        spec = dict(spec, resume=Path(spec['output']).exists())
        path = folder / f'{name}_launch.json'
        write(path, spec)
        subprocess.run([sys.executable, '-m', 'robustops.evaluation.teacher_rollout',
                        '--config', str(path)], cwd=root, check=True)
    launch(run, 'round1')
    source = Path(run['output'])
    summary = read(source / 'summary.json')
    expected = {(r['cell'], condition) for r in rows for condition in ['clean', 'fault']}
    if {(r['cell'], r['condition']) for r in summary} != expected or len(summary) != len(expected):
        raise ValueError('Resolve incomplete or invalid collection before retrying business failures')
    if not read(source / 'cleanup.json')['mcp_stopped']:
        raise ValueError('Teacher services were not cleaned up')
    focused = []
    for row in sorted(summary, key=lambda r: (r['condition'], r['cell'])):
        attempt = read(source / row['condition'] / f'cell_{row["cell"]:02}' / 'attempt.json')
        if not attempt['valid'] or not attempt['cleanup']['succeeded']:
            raise ValueError('Invalid collection is distinct from business failure')
        if not row['success']:
            focused.append(dict(cell=row['cell'], condition=row['condition']))
    retry = dict(run, temperature=.6, output=str(root / config['output'] / 'temperature_0.6'),
                 source_run=str(source),
                 focused_conditions=focused, experiment_id='teacher_train_failed_resample')
    write(folder / 'round2.json', retry, frozen=True)
    if focused:
        launch(retry, 'round2')
    write(root / config['output'] / 'completed.json', dict(original_conditions=len(expected),
          failed_conditions_resampled=len(focused), note='Fresh API outputs do not reproduce exactly 692 accepted records.'))


def export(root, config):
    from robustops.data.teacher_sft import run
    read(root / config['output'] / 'completed.json')
    return run(root, dict(source=config['output'], output=config['export'],
        manifest=config['manifest'], partition='train', tokenizer=config['teacher']['tokenizer'],
        template_parquet=config['template_parquet'], validation_data=config['validation_data'],
        reviewed_annotations=config.get('reviewed_annotations')))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--pool', type=Path, help='Override the Train teacher-collection manifest')
    parser.add_argument('--stage', choices=['prepare', 'collect', 'export', 'all'], default='all')
    args = parser.parse_args(); config = read(args.config)
    if args.pool is not None:
        config['pool'] = str(args.pool)
    if args.stage == 'prepare':
        _, rows = prepare(ROOT, config); print(json.dumps(dict(cells=len(rows), tasks=len({r['task_id'] for r in rows}))))
    if args.stage in ['collect', 'all']:
        collect(ROOT, config)
    if args.stage in ['export', 'all']:
        print(json.dumps(export(ROOT, config)))


if __name__ == '__main__':
    main()
