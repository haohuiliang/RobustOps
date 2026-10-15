"""Freeze faults from a local Base run, or summarize fresh fixed evaluations."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def fraction(n, d):
    return dict(n=n, d=d, percent=round(100*n/d, 2) if d else None)


def first_hit(original, supplemental):
    """Business success must never influence which valid hit is retained."""
    if original is not None and original['hits'] > 0:
        return original
    for attempt in sorted(supplemental, key=lambda r: r['index']):
        if attempt['valid'] and attempt['injected_calls'] > 0:
            return dict(success=attempt['success'], hits=attempt['injected_calls'], source=attempt['path'])
    return None


def load_model(folder, rows, specs):
    groups = read(folder / 'summary.json')['groups']
    if len(groups) != 2*len(rows) or len({(g['cell'],g['condition']) for g in groups}) != len(groups):
        raise ValueError(f'Incomplete fixed evaluation: {folder}')
    supplement = {g['cell']:g for g in read(folder / 'nohit_resamples/summary.json')['cells']}
    if any(not (g['hit'] or g['exhausted'] or g['skipped']) for g in supplement.values()):
        raise ValueError('Finish no-hit supplementation before reporting results')
    indexed = {(g['cell'],g['condition']):g for g in groups}
    output = {}
    for row in rows:
        cell = row['cell']; selected = {}
        for condition in ['clean','fault']:
            group = indexed[cell,condition]
            if (group['task_id'],group['mode']) != (row['task_id'],row['mode']):
                raise ValueError('Task/mode mismatch across runs')
            # Summary attempts contain only resolved valid attempts.
            attempts = group['attempts']
            if len(attempts) > 1:
                raise ValueError('Expected Pass@1, not a Pass@k summary')
            attempt = attempts[0] if attempts else None
            if attempt:
                raw = read(attempt['source'])
                if not raw['valid'] or not raw['cleanup']['succeeded'] or not raw['trajectory'].get('protocol_valid',True):
                    raise ValueError('Summary references a collection/protocol failure')
                if condition == 'fault':
                    spec = specs[cell]; fault = raw['fault']
                    if fault['target_mechanism'] != spec['kind'] or fault['config']['budget'] != spec['budget']:
                        raise ValueError('Attempt differs from its frozen fault manifest')
            selected[condition] = attempt
        selected['first_hit'] = first_hit(selected['fault'],supplement.get(cell,{}).get('attempts',[]))
        if selected['first_hit']:
            hit = selected['first_hit']; raw = read(hit['source']); spec = specs[cell]
            if (not raw['valid'] or not raw['cleanup']['succeeded']
                    or not raw['trajectory'].get('protocol_valid',True)
                    or (raw['task_id'],raw['mode'])!=(row['task_id'],row['mode'])
                    or raw['fault']['injected_calls']<=0
                    or raw['success']!=hit['success']
                    or raw['fault']['target_mechanism']!=spec['kind']
                    or raw['fault']['config']['budget']!=spec['budget']):
                raise ValueError('Selected hit differs from its source or fixed manifest')
        output[cell] = selected
    return output


def summarize(root, config):
    models = {}; reference = None
    for binding in config['runs']:
        folder = root / binding['path']; plan = read(folder / 'plan.json')
        rows = plan['rows']; specs = {r['cell']:r for r in plan['fixed_fault_specs']}
        identity = [(r['cell'],r['task_id'],r['mode'],r['partition'],specs[r['cell']]['kind'],specs[r['cell']]['budget']) for r in rows]
        if reference is not None and identity != reference:
            raise ValueError('Compare only runs with identical task/mode/fault/budget manifests')
        reference = identity
        expected = {'test':130,'validation':127}.get(rows[0]['partition'])
        if len(rows) != expected:
            raise ValueError('Expected the complete fixed Val or Test panel')
        names = list(plan.get('models',{})) or [binding.get('model_key','teacher')]
        for name in names:
            if name in models:
                raise ValueError(f'Duplicate model key: {name}')
            models[name] = load_model(folder/name if 'models' in plan else folder, rows, specs)
    if not models:
        raise ValueError('At least one evaluation is required')
    hits = {name:{cell for cell,result in data.items() if result['first_hit'] is not None} for name,data in models.items()}
    common = set.intersection(*hits.values())
    solvable = {r['cell'] for r in rows if all(data[r['cell']]['clean'] and data[r['cell']]['clean']['success'] for data in models.values())} & common
    ood = {r['cell'] for r in rows if specs[r['cell']]['kind']=='ignored_filter'}
    totals = {}
    for name,data in models.items():
        totals[name] = dict(clean_success=fraction(sum(bool(r['clean'] and r['clean']['success']) for r in data.values()),len(rows)),
            valid_clean=sum(r['clean'] is not None for r in data.values()),
            final_injection_coverage=fraction(len(hits[name]),len(rows)),
            actual_hit_task_success=fraction(sum(data[c]['first_hit']['success'] for c in hits[name]),len(hits[name])),
            common_hit_success=fraction(sum(data[c]['first_hit']['success'] for c in common),len(common)),
            common_clean_solvable_recovery=fraction(sum(data[c]['first_hit']['success'] for c in solvable),len(solvable)),
            ood_common_hit_success=fraction(sum(data[c]['first_hit']['success'] for c in common & ood),len(common & ood)))
    report = dict(partition=rows[0]['partition'],models=totals,common_hit_cells=sorted(common),
        common_clean_solvable_hit_cells=sorted(solvable),ood_common_hit_cells=sorted(common & ood),
        selection='Original valid actual hit, otherwise first valid supplemental hit irrespective of success.',
        skipped='Invalid/exhausted protocol slots remain missing; injection coverage uses all planned cells.')
    save(root/config['output'],report)
    return report


def freeze(root, config):
    policy_path = root/config['selector_policy']; rows = read(policy_path)['rows']
    base = root/config['base_run']; partition = config['partition']
    budget_path = base/'candidate_budgets.json'
    if not budget_path.exists():
        budget_path = root/config['candidate_budgets']
    budgets = read(budget_path)
    expected = {'test':130,'validation':127}.get(partition)
    if len(rows)!=expected or any(r['partition']!=partition for r in rows):
        raise ValueError('Full fixed Val/Test selector coverage is required')
    # Runtime-priority Base chooses the mechanism along its own actual tool path.
    # A completed fixed Base run can also be converted without reassignment.
    plan = read(base/'plan.json') if (base/'plan.json').exists() else None
    reused = read(base/'reused_results.json') if (base/'reused_results.json').exists() else {}
    groups = None
    if plan:
        model = config.get('base_model_key','base')
        groups = {(g['cell'],g['condition']):g for g in read(base/model/'summary.json')['groups']}
    output = []
    for row in rows:
        cell = row['cell']
        if groups is not None:
            attempts = groups[cell,'fault']['attempts']
            if len(attempts)!=1: raise ValueError('Base needs a valid original fault attempt per cell')
            path = Path(attempts[0]['source'])
        else:
            path = base/'runtime_priority'/f'cell_{cell:02}'/'attempt.json'
            if not path.exists() and str(cell) in reused:
                path = Path(reused[str(cell)])
        attempt = read(path); fault = attempt['fault']
        if not attempt['valid'] or not attempt['cleanup']['succeeded'] or (attempt['task_id'],attempt['mode'])!=(row['task_id'],row['mode']):
            raise ValueError(f'Invalid or mismatched Base source: {path}')
        kind = fault.get('target_mechanism') or fault.get('selected_mechanism') or fault.get('reserved_label')
        eligible = {k for t in row['selection']['tools'] for k in t['faults']}
        if kind not in eligible:
            raise ValueError(f"Base cell {row['cell']} has no eligible selected/reserved mechanism")
        budget = budgets[str(cell)][kind]
        actual_budget = fault.get('candidate_budgets',{}).get(kind)
        if actual_budget is None:
            actual_budget = fault.get('config',{}).get('budget')
        if actual_budget!=budget:
            raise ValueError('Base source did not use the shared candidate dose table')
        if kind.startswith('unknown_commit') and budget>2:
            raise ValueError('Uncertain commits must use the shared cap2 budget table')
        output.append(dict(cell=cell,task_id=row['task_id'],mode=row['mode'],domain=row['domain'],
            kind=kind,budget=budget,base_injected_calls=fault['injected_calls'],
            kind_source='local_base_selected_or_reserved',source_attempt=str(path)))
    manifest = dict(schema_version=1,partition=partition,budget_revision='unknown_commit_cap2',
        source=str(base),selector_policy=config['selector_policy'],candidate_budgets=str(budget_path.resolve()),rows=output)
    manifest['sha256'] = hashlib.sha256(json.dumps(output,sort_keys=True).encode()).hexdigest()
    target = root/config['output']
    if target.exists() and read(target)!=manifest:
        raise ValueError('Frozen custom manifest exists; choose a new output for a different Base run')
    save(target,manifest)
    return dict(output=str(target),cells=len(output),sha256=manifest['sha256'])


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--action',choices=['freeze','summarize'],required=True)
    parser.add_argument('--base-run',help='override the Base assignment directory for freezing')
    parser.add_argument('--output',help='override the output JSON path')
    args=parser.parse_args(argv)
    config=read(args.config)
    if args.base_run is not None:
        if args.action!='freeze':
            parser.error('--base-run is only valid with --action freeze')
        config['base_run']=args.base_run
    if args.output is not None:
        config['output']=args.output
    print(json.dumps((freeze if args.action=='freeze' else summarize)(ROOT,config),indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
