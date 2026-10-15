"""Run frozen normal/fault conditions through the existing local inference launcher."""
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from robustops.launch.config import LaunchConfigError, resolve_project_path


def launch_recovery(root, recipe_path, recipe, *, gpu_override, partition_override, dry_run):
    from robustops.launch.inference import launch
    partitions=recipe['workload']['partitions']
    if partition_override not in (None,'all'):
        if partition_override not in partitions:raise LaunchConfigError('partition outside recovery recipe')
        partitions=[partition_override]
    run_id=os.environ.get('ROBUSTOPS_EVALUATION_ID') or recipe['output']['run_id_prefix']+'-'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+str(os.getpid())
    out=Path(os.environ['ROBUSTOPS_EVALUATION_OUTPUT_ROOT']) if os.environ.get('ROBUSTOPS_EVALUATION_OUTPUT_ROOT') else resolve_project_path(root,recipe['output']['root'])/run_id
    if recipe['model'].get('runtime_override_required') and not os.environ.get('ROBUSTOPS_MODEL_PATH'):
        raise LaunchConfigError('Set ROBUSTOPS_MODEL_PATH to the selected SFT final huggingface directory')
    if recipe['model'].get('runtime_override_required'):
        recipe=deepcopy(recipe);model_path=Path(os.environ['ROBUSTOPS_MODEL_PATH']).expanduser()
        recipe['model'].update(runtime_override_required=False,path=str(model_path),revision=str(model_path),served_name='RobustOps-SFT-4B')
    manifest={r['task_id']:r for r in map(json.loads,(root/recipe['workload']['task_manifest']).read_text().splitlines())}
    for part in partitions:
        path=root/recipe['fault']['policies'][part]
        rows=[json.loads(l) for l in path.read_text().splitlines()]
        keys={(r['task_id'],r['mode']) for r in rows}
        expected={(r['task_id'],mode) for r in manifest.values() if r['partition']==part and not r['execution_blocked_reason'] for mode in recipe['workload']['modes'] if mode in r['runnable_modes']}
        if keys!=expected or len(keys)!=len(rows) or any(r['partition']!=part for r in rows):raise LaunchConfigError('frozen policy coverage/partition mismatch')
    if not dry_run:
        out.mkdir(parents=True,exist_ok=False)
        (out/'recipe.json').write_text(json.dumps(recipe,indent=2)+'\n')
        (out/'policies').mkdir()
        for part in partitions:
            (out/'policies'/f'{part}.jsonl').write_bytes((root/recipe['fault']['policies'][part]).read_bytes())
    print(f'recovery_evaluation output={out} partitions={partitions} conditions=normal,faulted',flush=True)
    saved={k:os.environ.get(k) for k in ['ROBUSTOPS_EVALUATION_ID','ROBUSTOPS_EVALUATION_OUTPUT_ROOT']}
    try:
        for part in partitions:
            for condition in ['normal','faulted']:
                child=deepcopy(recipe);child['pipeline']='mode_evaluation';w=child['workload'];w['partitions']=[part]
                for key in ['expected_attempts_total','expected_logical_attempts_total']:
                    w[key]=sum(w['expected_tasks_by_mode'][m][part] for m in w['modes'])
                child['fault']={'enabled':False} if condition=='normal' else dict(enabled=True,kind='fixed_policy',policies={part:str(root/recipe['fault']['policies'][part]) if dry_run else str(out/'policies'/f'{part}.jsonl')})
                os.environ['ROBUSTOPS_EVALUATION_ID']=f'{run_id}-{part}-{condition}'
                os.environ['ROBUSTOPS_EVALUATION_OUTPUT_ROOT']=str(out/part/condition)
                result=launch(root,recipe_path,child,gpu_override=gpu_override,partition_override=None,dry_run=dry_run)
                if result:return result
    finally:
        for k,v in saved.items():
            if v is None:os.environ.pop(k,None)
            else:os.environ[k]=v
    return 0
