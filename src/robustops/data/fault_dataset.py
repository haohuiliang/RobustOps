"""Refresh all Oracle/+5 fault labels; publish only after all splits complete."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path
import shutil

from robustops.data import fault_candidates as candidates
from robustops.data import fault_selector_batch as batch
from robustops.data.fault_candidate_prompt import PROMPT

async def run(config, prepare_only=False):
    all_rows = {}; configs = {}
    for split, path in config['splits'].items():
        cfg = candidates.read(path); output = Path(cfg['output'])
        rows = candidates.prepare(cfg, output)
        reused = candidates.reuse(cfg, output, rows)
        all_rows[split] = rows; configs[split] = cfg
        print(json.dumps(dict(split=split, task_modes=len(rows), reused=reused)),flush=True)
    if prepare_only:
        return
    for split, rows in all_rows.items():
        cfg = configs[split]; observations = candidates.read(cfg['response_shapes'])
        async def select(*args, **kwargs):
            return await candidates.select(*args, observations=observations, **kwargs)
        if cfg.get('reuse_only'):
            if any(batch.successful_attempt(Path(cfg['output'])/f"cell_{r['cell']:04}") is None for r in rows):
                raise ValueError('reuse-only split incomplete; API calls forbidden')
            batch.summarize(Path(cfg['output']), rows)
        else:
            await batch.run(cfg, Path(cfg['output']), rows, selector=select)
        # Freeze Val evaluation as soon as its selector is complete.
        if cfg.get('policy_output'):
            candidates.freeze(cfg,Path(cfg['output']),rows)
    publish(config,configs,all_rows)

def publish(config, configs, all_rows):
    output = Path(config['output']); staging=Path(config['staging'])
    staging.mkdir(parents=True,exist_ok=True)
    summary = dict(schema='conditional_candidates',complete=True,scope=['oracle','plus_5_tools'],splits={})
    seen_tasks=set()
    for split, rows in all_rows.items():
        cfg=configs[split]; records=[];reused=0
        for r in rows:
            attempt=batch.successful_attempt(Path(cfg['output'])/f"cell_{r['cell']:04}")
            if attempt is None:raise ValueError(f'incomplete {split} cell {r["cell"]}; old dataset retained')
            payload=candidates.read(r['input']);selection=candidates.read(attempt/'selection.json')
            candidates.validate_selection(selection,payload)
            records.append(dict(task_id=r['task_id'],mode=r['mode'],domain=r['domain'],partition=cfg['partition'],tools=selection['tools'],source_attempt=str(attempt)))
            reused+=int((attempt/'reused_from.json').exists())
        keys={(r['task_id'],r['mode']) for r in records};tasks={r['task_id'] for r in records}
        if len(keys)!=cfg['expected_task_modes'] or len(keys)!=len(records) or tasks & seen_tasks:
            raise ValueError('dataset identity/partition coverage mismatch')
        seen_tasks.update(tasks)
        (staging/f'{split}.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records))
        summary['splits'][split]=dict(tasks=len(tasks),pairs=len(records),reused=reused,modes=dict(Counter(r['mode'] for r in records)),conditional_coverage=dict(Counter(k for r in records for k in {k for t in r['tools'] for k in t['faults']})))
    summary.update(total_tasks=len(seen_tasks),total_pairs=sum(s['pairs'] for s in summary['splits'].values()))
    batch.write_json(staging/'summary.json',summary)
    batch.write_json(staging/'prompts.json',dict(system=PROMPT))
    batch.write_json(staging/'export_config.json',config)
    shutil.copy2(next(iter(configs.values()))['response_shapes'],staging/'response_shape_observations.json')
    (staging/'README.md').write_text('''# Fault candidate labels

Candidate labels cover effective Oracle/+5 task-mode cells. Selector outputs are retained without manual relabeling. Seven labels form five mechanism groups; eligibility is checked by the runtime injector on actual calls.

Per-tool reasons, recovery tools, replay basis, and faults are candidate evidence, not execution guarantees. Source references and prompt statistics are recorded with the data. Frozen Test snapshots remain independent of dataset refreshes.
''')
    backup=Path(config['backup'])
    if backup.exists():
        if all((output/p.name).exists() and (output/p.name).read_bytes()==p.read_bytes() for p in staging.iterdir()):
            print(json.dumps(summary,ensure_ascii=False),flush=True);return
        raise ValueError('backup exists and publication differs; refusing overwrite')
    backup.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():output.rename(backup)
    try:staging.rename(output)
    except BaseException:
        if backup.exists():backup.rename(output)
        raise
    print(json.dumps(summary,ensure_ascii=False),flush=True)

def export_main():
    p=argparse.ArgumentParser(description='Publish complete current selector results without API calls')
    p.add_argument('--config',required=True);a=p.parse_args();config=candidates.read(a.config)
    configs={split:candidates.read(path) for split,path in config['splits'].items()}
    rows={split:candidates.read(Path(cfg['output'])/'cells.json') for split,cfg in configs.items()}
    publish(config,configs,rows)

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True);p.add_argument('--prepare-only',action='store_true');a=p.parse_args()
    asyncio.run(run(candidates.read(a.config),a.prepare_only))

if __name__=='__main__':main()
