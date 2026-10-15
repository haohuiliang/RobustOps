"""Offline preparation, bounded API generation and export of verifier DAGs.

The model proposes only dependencies. Author SQL/comparisons stay in the input
contract and are copied unchanged to the export. This module never starts MCP,
a student model, or training, and does not import historical analysis scripts.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import time
from threading import Event
import zipfile

import httpx
import pyarrow.parquet as pq
from robustops.providers.credentials import credential
from robustops.data.verifier_graph import (
    boundary_check, progress, reindex_dependencies, validate_graph,
)
from robustops.envs.enterpriseops.task import load_tasks
from robustops.envs.enterpriseops.runtime import compare_verifier_value, extract_sql_value

ROOT = Path(__file__).resolve().parents[3]

def read(path):
    return json.loads(Path(path).read_text())

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)

def key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False)

def verifier_key(v):
    return key({k:x for k,x in v.items() if k != 'index'})

def signature(payload):
    value = {k:v for k,v in payload.items() if k not in ('verifiers','databases')}
    value['verifiers'] = sorted(verifier_key(v) for v in payload['verifiers'])
    value['databases'] = sorted(payload['databases'], key=lambda d:d['gym_name'])
    return key(value)

def validate_original(payload, states):
    checks = []
    for gym, sql in states.items():
        db = sqlite3.connect(':memory:')
        try:
            db.executescript(sql)
            db.execute('PRAGMA query_only=ON')
            for v in payload['verifiers']:
                if v['gym_name'] != gym:
                    continue
                c = v['validation_config']
                rows = [list(row) for row in db.execute(c['query']).fetchall()]
                actual = extract_sql_value({'rows':rows})
                checks.append(dict(id=f"v{v['index']}", initial_value=int(compare_verifier_value(
                    actual, c['expected_value'], c.get('comparison_type','equals')))))
        finally:
            db.close()
    if len(checks) != len(payload['verifiers']):
        raise ValueError('Some original checks have no database state')
    return checks

def paths(config):
    return ROOT/config['run_dir'], ROOT/config['output']

def prepare(config, *, reuse_from=None):
    run_dir, output = paths(config)
    gen = run_dir/'generation'
    if gen.exists():
        raise ValueError('Prepared run already exists; use phase run/export to resume it.')
    meta = [json.loads(line) for line in (ROOT/config['manifest']).read_text().splitlines() if line.strip()]
    targets = [m for m in meta if m['partition'] in config['partitions'] and not m.get('execution_blocked_reason')]
    if not targets or not set(config['partitions']) <= {'train','validation'}:
        raise ValueError('This recipe supports nonempty Train/Val preparation only.')
    recipe = read(ROOT/config['generation_config'])
    prompt = (ROOT/recipe['prompt_source']).read_text()
    templates = read(ROOT/config['database_schema'])['servers']
    old_inputs = {}
    old_results = {}
    if reuse_from is not None:
        reuse_from = Path(reuse_from).resolve()
        if (reuse_from/'prompt.txt').read_text() != prompt:
            raise ValueError('Reuse prompt differs from the frozen generation prompt')
        old_inputs = {signature(r['payload']):r for r in read(reuse_from/'inputs.json')}
        old_results = {r['label']:r for r in read(reuse_from/'results.json') if not r['error']}
    tasks = {}
    for mode in config['modes']:
        for domain in sorted({r['domain'] for r in targets}):
            p = ROOT/config['task_data']/mode/(domain+'-00000-of-00001.parquet')
            tasks[mode,domain] = {t.task_id:t for t in load_tasks(p, expected_domain=domain,
                expected_count=pq.ParquetFile(p).metadata.num_rows)}
    inputs, selection, mappings, reused, checks, results = [], [], [], [], [], []
    by_signature, state_cache, schema_cache = {}, {}, {}
    run_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ROOT/config['database_archive']) as archive:
        for m in sorted(targets,key=lambda r:(r['partition'],r['task_id'])):
            for mode in config['modes']:
                if mode not in m['runnable_modes']:
                    continue
                t = tasks[mode,m['domain']][m['task_id']]
                states, databases = {}, []
                for server in t.gym_servers:
                    gym, seed = server['mcp_server_name'], server['seed_database_file']
                    sk = gym, seed
                    if sk not in state_cache:
                        sql = archive.read(seed).decode('utf8')
                        if not re.search(r'CREATE\s+TABLE',sql,re.I):
                            sql = templates[gym]['ddl']+sql
                        state_cache[sk] = sql
                        db = sqlite3.connect(':memory:')
                        db.executescript(sql)
                        tables = dict(db.execute("SELECT name,sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"))
                        fks = {name:[v[2] for v in db.execute('PRAGMA foreign_key_list("'+name.replace('"','""')+'")')] for name in tables}
                        db.close()
                        schema_cache[sk] = tables, fks
                    states[gym] = state_cache[sk]
                    tables, fks = schema_cache[sk]
                    queries = '\n'.join(v['validation_config']['query'] for v in t.verifiers if v['gym_name']==gym)
                    selected = {name for name in tables if re.search(r'\b'+re.escape(name)+r'\b',queries,re.I)}
                    for name in list(selected):
                        selected.update(v for v in fks[name] if v in tables)
                    databases.append(dict(gym_name=gym,schema={name:tables[name] for name in sorted(selected)}))
                payload = dict(task_id=t.task_id,domain=t.domain,user_prompt=t.user_prompt,
                    system_prompt=t.system_prompt,verifiers=[dict(index=i,**v) for i,v in enumerate(t.verifiers,1)],databases=databases)
                sig = signature(payload)
                if sig not in by_signature:
                    label = f"graph_{len(inputs)+1:04d}_{m['partition']}_{t.domain}"
                    by_signature[sig] = len(inputs)
                    old = old_inputs.get(sig)
                    if old and old['label'] in old_results:
                        payload = old['payload']
                        proposal = read(reuse_from/(old['label']+'.proposal.json'))
                        check = boundary_check(proposal,payload)
                        save(gen/(label+'.proposal.json'),proposal)
                        source = dict(folder=str(reuse_from),label=old['label'])
                        results.append(dict(old_results[old['label']],label=label,boundary_check=check,reused_from=source))
                        reused.append(dict(label=label,source_label=old['label'],source_folder=str(reuse_from)))
                    statepath = run_dir/'initial_states'/(label+'.json')
                    save(statepath,states)
                    checks.append(dict(label=label,checks=validate_original(payload,states)))
                    inputs.append(dict(label=label,payload=payload))
                    selection.append(dict(label=label,task_id=t.task_id,domain=t.domain,
                        partition=m['partition'],mode=mode,initial_states=str(statepath)))
                canonical = inputs[by_signature[sig]]
                indices = {verifier_key(v):v['index'] for v in canonical['payload']['verifiers']}
                if len(indices) != len(t.verifiers):
                    raise ValueError('Duplicate original checks require explicit source-contract handling')
                mapping = [indices[verifier_key(v)] for v in t.verifiers]
                mappings.append(dict(task_id=t.task_id,domain=t.domain,partition=m['partition'],mode=mode,
                    graph_id=canonical['label'],original_verifiers=list(t.verifiers),mode_to_graph_indices=mapping))
    gen.mkdir(exist_ok=True)
    output.mkdir(parents=True,exist_ok=True)
    provenance = dict(recipe,requests=len(inputs)-len(reused),reused_graphs=len(reused),unique_graphs=len(inputs),
        revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True),no_mcp=True,no_gpu=True,no_training=True)
    for name,value in [('inputs',inputs),('selection',selection),('mode_mappings',mappings),
                       ('reused_sources',reused),('results',results),('provenance',provenance)]:
        save(gen/(name+'.json'),value)
    (gen/'prompt.txt').write_text(prompt)
    (output/'verifier_graph.md').write_text(prompt)
    save(run_dir/'original_sql_preflight.json',dict(graph_inputs=len(checks),checks=checks))
    coverage = dict(base_tasks=len({r['task_id'] for r in mappings}),
        partition_base_tasks=dict(Counter(r['partition'] for r in targets)),mode_pairs=len(mappings),
        partition_mode_pairs=dict(Counter(r['partition'] for r in mappings)),mode_counts=dict(Counter(r['mode'] for r in mappings)),
        unique_graphs=len(inputs),reused_graphs=len(reused),new_requests=len(inputs)-len(reused),
        reordered_modes=sum(r['mode_to_graph_indices']!=list(range(1,len(r['original_verifiers'])+1)) for r in mappings),
        excluded_blocked_tasks=[r['task_id'] for r in meta if r['partition'] in config['partitions'] and r.get('execution_blocked_reason')],test_included=False)
    save(run_dir/'preparation.json',coverage)
    print(json.dumps(coverage,ensure_ascii=False),flush=True)
    return coverage

def generate_one(item, folder, settings, provider, api_key, stop_event=None):
    if stop_event is not None and stop_event.is_set():
        return None
    label, payload = item['label'], item['payload']
    if (folder/(label+'.request.json')).exists():
        raise ValueError('Refusing to resend an already recorded attempt')
    request = dict(model=settings['model'],thinking={'type':settings['thinking']},
        max_tokens=settings['max_tokens'],response_format={'type':'json_object'},stream=False,
        messages=[dict(role='system',content=(folder/'prompt.txt').read_text()),
                  dict(role='user',content=json.dumps(payload,ensure_ascii=False))])
    if settings['thinking']=='enabled':
        request['reasoning_effort']=settings['reasoning_effort']
    else:
        request['temperature']=settings['temperature']
    started=datetime.now(timezone.utc).isoformat()
    save(folder/(label+'.request.json'),dict(started=started,body=request))
    t0=time.monotonic()
    row=dict(label=label,task_id=payload['task_id'],error=None,usage=None)
    try:
        with httpx.Client(timeout=settings['request_timeout_seconds'],trust_env=False) as client:
            response=client.post(provider['base_url'].rstrip('/')+'/chat/completions',
                headers={'Authorization':'Bearer '+api_key},json=request)
            response.raise_for_status()
            raw=response.json()
        save(folder/(label+'.response.json'),dict(started=started,seconds=time.monotonic()-t0,response=raw))
        row.update(model=raw.get('model'),usage=raw.get('usage'))
        choice=raw['choices'][0]
        if choice['finish_reason']!='stop':
            raise ValueError('finish_reason='+str(choice['finish_reason']))
        proposal=json.loads(choice['message']['content'])
        save(folder/(label+'.proposal.json'),proposal)
        row['boundary_check']=boundary_check(proposal,payload)
        row['edges']=[dict(index=n['index'],depends_on=n['depends_on']) for n in proposal['nodes']]
    except httpx.HTTPError as exc:
        row['error']=type(exc).__name__
        status=exc.response.status_code if isinstance(exc,httpx.HTTPStatusError) else None
        row['http_status']=status
        row['stop_batch']=status in (401,402,403)
        if row['stop_batch'] and stop_event is not None:
            stop_event.set()
        save(folder/(label+'.failure.json'),dict(type=type(exc).__name__,http_status=status,usage_unknown=True))
    except (ValueError,KeyError,TypeError,AssertionError) as exc:
        row['error']=type(exc).__name__+': '+str(exc)
    row['seconds']=time.monotonic()-t0
    save(folder/(label+'.result.json'),[row])
    return row

def resolve_sources(run_dir):
    gen=run_dir/'generation'
    sources={r['label']:(gen,r) for r in read(gen/'results.json') if not r['error']}
    for folder in [gen]+sorted(run_dir.glob('attempt_*')):
        if not folder.is_dir():
            continue
        for path in folder.glob('*.result.json'):
            r=read(path)[0]
            if not r['error']:
                sources[r['label']]=(folder,r)
    return sources

def generate(config):
    run_dir,_=paths(config)
    gen=run_dir/'generation'
    inputs=read(gen/'inputs.json')
    settings=read(gen/'provenance.json')
    if (gen/'prompt.txt').read_text()!=(ROOT/settings['prompt_source']).read_text():
        raise ValueError('Frozen prompt differs from current prompt; do not mix generations')
    if not (run_dir/'original_sql_preflight.json').exists():
        raise ValueError('Original SQL preflight missing; complete preparation first')
    max_attempts=config['max_attempts']
    if not 1 <= max_attempts <= 3:
        raise ValueError('Expected one original and at most two failed-single retries')
    provider=dict(config['provider'])
    if provider.get('local_env_file'):
        provider['local_env_file']=str(ROOT/provider['local_env_file'])
    api_key=None
    for attempt in range(1,max_attempts+1):
        sources=resolve_sources(run_dir)
        pending=[r for r in inputs if r['label'] not in sources]
        if not pending:
            break
        folder=gen if attempt==1 else run_dir/f'attempt_{attempt}'
        folder.mkdir(exist_ok=True)
        if folder!=gen and not (folder/'prompt.txt').exists():
            (folder/'prompt.txt').write_text((gen/'prompt.txt').read_text())
            save(folder/'inputs.json',pending)
            save(folder/'provenance.json',dict(settings,attempt=attempt))
        unsent=[]
        for item in pending:
            label=item['label']
            if (folder/(label+'.result.json')).exists() or (folder/(label+'.failure.json')).exists():
                continue
            if (folder/(label+'.request.json')).exists():
                raise ValueError(f'Recorded request has unknown completion: {folder}/{label}; inspect before resuming')
            unsent.append(item)
        if not unsent:
            continue
        if api_key is None:
            api_key=credential(provider)
        results={r['label']:r for r in read(folder/'results.json')} if (folder/'results.json').exists() else {}
        stop_event=Event()
        with ThreadPoolExecutor(max_workers=settings['total_concurrency']) as pool:
            futures={pool.submit(generate_one,item,folder,settings,provider,api_key,stop_event):item['label'] for item in unsent}
            for future in as_completed(futures):
                row=future.result()
                if row is None:
                    continue
                results[row['label']]=row
                save(folder/'results.json',list(results.values()))
                print(json.dumps(dict(attempt=attempt,label=row['label'],error=row['error'],http_status=row.get('http_status'))),flush=True)
                save(run_dir/'status.json',dict(status='running',valid=len(resolve_sources(run_dir)),total=len(inputs),attempt=attempt))
        if stop_event.is_set():
            save(run_dir/'status.json',dict(status='account_blocked',valid=len(resolve_sources(run_dir)),total=len(inputs),attempt=attempt))
            raise RuntimeError('Provider returned HTTP 401/402/403; queued calls and further retries stopped. Resolve the account condition before resuming.')
    missing=[r['label'] for r in inputs if r['label'] not in resolve_sources(run_dir)]
    if missing:
        raise ValueError(f'Incomplete after bounded attempts: {missing}')

def export(config):
    run_dir,output=paths(config)
    gen=run_dir/'generation'
    inputs={r['label']:r['payload'] for r in read(gen/'inputs.json')}
    selection={r['label']:r for r in read(gen/'selection.json')}
    mappings=read(gen/'mode_mappings.json')
    sources=resolve_sources(run_dir)
    if set(sources)!=set(inputs):
        raise ValueError('Graph coverage incomplete or contains unexpected IDs')
    graph_rows=[]
    graphs={}
    counts=Counter()
    for label,payload in inputs.items():
        folder,result=sources[label]
        proposal=read(folder/(label+'.proposal.json'))
        deps=validate_graph(proposal,payload)
        check=result['boundary_check']
        if not check['full_success_equivalent'] or check['boolean_vectors_checked']!=2**len(deps):
            raise ValueError('Missing exhaustive boundary verification')
        counts.update(graphs=1,nodes=len(deps),edges=sum(map(len,deps.values())),empty_graphs=int(not any(deps.values())),boolean_vectors=check['boolean_vectors_checked'])
        graphs[label]=deps
        graph_rows.append(dict(graph_id=label,task_id=payload['task_id'],domain=payload['domain'],partition=selection[label]['partition'],
            original_verifiers=payload['verifiers'],proposal=proposal,source=result.get('reused_from') or dict(folder=str(folder.relative_to(ROOT)),label=label),boundary_check=check))
    exports={'train':[],'validation':[]}
    for m in mappings:
        label=m['graph_id'];mapping=m['mode_to_graph_indices']
        canonical=inputs[label]['verifiers']
        if not all(verifier_key(v)==verifier_key(canonical[j-1]) for v,j in zip(m['original_verifiers'],mapping)):
            raise ValueError('Mode original SQL/metadata differs from its canonical check')
        deps=reindex_dependencies(graphs[label],mapping)
        proposal=dict(task_id=m['task_id'],final_success='original_verifiers_unchanged',nodes=[dict(index=i,depends_on=v) for i,v in deps.items()])
        validate_graph(proposal,dict(task_id=m['task_id'],verifiers=m['original_verifiers']))
        inverse={g:i for i,g in enumerate(mapping,1)}
        vectors=[[1]*len(deps),[0]*len(deps)]+[[int(j!=i) for j in range(len(deps))] for i in range(len(deps))]
        for bits in vectors:
            original=progress([bits[inverse[i]-1] for i in range(1,len(deps)+1)],graphs[label])
            remapped=progress(bits,deps)
            if remapped!=[original[j-1] for j in mapping] or all(remapped)!=all(bits):
                raise ValueError('Mode reindexing changed graph outcomes')
        exports[m['partition']].append(dict(task_id=m['task_id'],domain=m['domain'],partition=m['partition'],mode=m['mode'],
            graph_id=label,mode_to_graph_indices=mapping,verifiers=[dict(index=i,**v,depends_on=deps[i]) for i,v in enumerate(m['original_verifiers'],1)],final_success='original_verifiers_unchanged'))
    manifest=[json.loads(line) for line in (ROOT/config['manifest']).read_text().splitlines() if line.strip()]
    expected={(r['partition'],r['task_id'],mode) for r in manifest if r['partition'] in config['partitions'] and not r.get('execution_blocked_reason') for mode in config['modes'] if mode in r['runnable_modes']}
    actual=[(r['partition'],r['task_id'],r['mode']) for rows in exports.values() for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:
        raise ValueError('Export differs from current Train/Val manifest coverage')
    if {r['task_id'] for r in exports['train']}&{r['task_id'] for r in exports['validation']}:
        raise ValueError('Train/Val overlap')
    output.mkdir(parents=True,exist_ok=True)
    frozen_prompt=(gen/'prompt.txt').read_text()
    if frozen_prompt!=(ROOT/read(gen/'provenance.json')['prompt_source']).read_text():
        raise ValueError('Current prompt differs from the frozen generation')
    (output/'verifier_graph.md').write_text(frozen_prompt)
    for name,rows in [('graphs',graph_rows),('train',exports['train']),('val',exports['validation'])]:
        path=output/(name+'.jsonl');tmp=path.with_suffix('.jsonl.tmp')
        tmp.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows));tmp.replace(path)
    history=[]
    for folder in [gen]+sorted(run_dir.glob('attempt_*')):
        if not folder.is_dir():
            continue
        for request in sorted(folder.glob('*.request.json')):
            label=request.name.removesuffix('.request.json')
            result_path=folder/(label+'.result.json')
            result=read(result_path)[0] if result_path.exists() else None
            history.append(dict(label=label,folder=str(folder.relative_to(ROOT)),
                attempt=1 if folder==gen else int(folder.name.split('_')[1]),
                error=result['error'] if result else 'No completed result',
                http_status=result.get('http_status') if result else None))
    save(run_dir/'attempt_history.json',history)
    summary=read(output/'summary.json') if (output/'summary.json').exists() else {}
    summary.update(status='complete',prompt_status='final',source_run=str(run_dir.relative_to(ROOT)),coverage=read(run_dir/'preparation.json'),
        config=read(gen/'provenance.json'),new_api_attempts=len(history),new_first_attempt_valid=sum(r['attempt']==1 and not r['error'] for r in history),
        retry_attempts=sum(r['attempt']>1 for r in history),retained_failed_attempts=[r for r in history if r['error']],
        graph_summary=dict(counts),all_full_success_equivalent=True,original_sql_unchanged=True,mode_original_order_preserved=True,
        train_validation_disjoint=True,test_included=False,training_integration=summary.get('training_integration',False),
        publication_entry='scripts/build_verifier_graphs.py',proof='Every original node is retained and AND-gated only by original nodes in a finite DAG: all original true iff all gated true. Each unique graph has exhaustive Boolean verification; every mode has exact original-contract mapping.')
    save(output/'summary.json',summary)
    save(run_dir/'status.json',dict(status='complete',valid=len(graph_rows),total=len(inputs),output=str(output)))
    print(json.dumps(dict(status='complete',graph_summary=dict(counts),output=str(output)),ensure_ascii=False),flush=True)
    return summary

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--phase',choices=['prepare','run','export','all'],default='all')
    parser.add_argument('--reuse-from',type=Path,help='Optional completed generation folder with the exact same prompt')
    parser.add_argument('--run-dir',type=Path,help='Override run folder for a new independent generation')
    parser.add_argument('--output',type=Path,help='Override final dataset folder')
    args=parser.parse_args()
    config=read(args.config)
    if args.run_dir:config['run_dir']=str(args.run_dir)
    if args.output:config['output']=str(args.output)
    if args.phase in ('prepare','all'):prepare(config,reuse_from=args.reuse_from)
    if args.phase in ('run','all'):generate(config)
    if args.phase in ('export','all'):export(config)

if __name__=='__main__':
    main()
