"""Select a Train-only teacher collection pool without changing dataset splits."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import random

ROOT = Path(__file__).resolve().parents[3]


def load_collection_pool(path, manifest):
    """Read a collection manifest; mixed historical pools use their SFT subset."""
    source = json.loads(Path(path).read_text())
    canonical = {row['task_id']: row for row in manifest}
    selected, seen = [], set()
    for row in source['tasks']:
        if row.get('pool', 'sft') != 'sft':
            continue
        task_id = row['task_id']
        meta = canonical.get(task_id)
        if (meta is None or meta['partition'] != 'train'
                or meta.get('execution_blocked_reason') or not meta['runnable_modes']):
            raise ValueError(f'Collection task must be runnable Train: {task_id}')
        modes = row.get('modes', meta['runnable_modes'])
        if (task_id in seen or not modes or len(set(modes)) != len(modes)
                or not set(modes) <= set(meta['runnable_modes'])
                or row.get('domain', meta['domain']) != meta['domain']):
            raise ValueError(f'Collection task identity/modes differ: {task_id}')
        seen.add(task_id)
        selected.append(dict(row, domain=meta['domain'], modes=modes, pool='sft'))
    if not selected:
        raise ValueError('Collection manifest has no SFT tasks')
    return dict(source, tasks=selected)


def build_collection_pool(manifest, fraction, seed):
    if not math.isfinite(fraction) or not 0 < fraction <= 1:
        raise ValueError('Collection fraction must be in (0, 1]')
    groups = defaultdict(list)
    seen = set()
    for row in manifest:
        if row['task_id'] in seen:
            raise ValueError(f'Duplicate task in source manifest: {row["task_id"]}')
        seen.add(row['task_id'])
        if (row['partition'] == 'train' and not row.get('execution_blocked_reason')
                and row['runnable_modes']):
            groups[row['domain'], tuple(sorted(row['runnable_modes']))].append(row)
    total = sum(map(len, groups.values()))
    if not total:
        raise ValueError('Source manifest has no runnable Train tasks')
    target = max(1, round(total * fraction))
    quotas = {key: math.floor(len(rows) * fraction) for key, rows in groups.items()}
    order = sorted(groups, key=lambda key: (-(len(groups[key]) * fraction - quotas[key]), key))
    for key in order[:target - sum(quotas.values())]:
        quotas[key] += 1
    selected = []
    for key in sorted(groups):
        rows = sorted(groups[key], key=lambda row: row['task_id'])
        random.Random(f'{seed}/{key}').shuffle(rows)
        selected.extend(dict(task_id=row['task_id'], domain=row['domain'],
                             modes=row['runnable_modes'], pool='sft') for row in rows[:quotas[key]])
    return dict(schema_version=1, partition='train', fraction=fraction, seed=seed,
                unit='base_task_id', method='domain x runnable modes; largest remainder; seeded shuffle',
                counts=dict(runnable_train_tasks=total, selected_tasks=len(selected)),
                tasks=sorted(selected, key=lambda row: row['task_id']))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'data/task_sets/enterpriseops.jsonl')
    parser.add_argument('--fraction', type=float, required=True)
    parser.add_argument('--seed', type=int, default=20260916)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines() if line.strip()]
    result = build_collection_pool(rows, args.fraction, args.seed)
    result['source'] = str(args.manifest)
    if args.output.exists():
        raise FileExistsError(f'Collection manifest already exists: {args.output}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(output=str(args.output), fraction=args.fraction, **result['counts'])))


if __name__ == '__main__':
    main()
