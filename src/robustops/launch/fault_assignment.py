"""Inference adapter for the existing tool-fault evaluation runner."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from robustops.launch.config import LaunchConfigError, resolve_project_path, resolve_model_path
from robustops.evaluation.fault_schedule import assign_budgets, load_schedule
from robustops.launch.resources import check_host_memory


def launch_fault_assignment(root, recipe_path, recipe, *, gpu_override, partition_override, dry_run):
    partition=recipe.get('partition')
    from robustops.launch.hardware import select_gpus
    selection=select_gpus(gpu_override if gpu_override is not None else recipe.get('hardware', {}).get('gpus'), dry_run=dry_run)
    gpus=min(selection.count, recipe.get('samples_per_task',1), 2)
    if gpus not in (1,2):raise LaunchConfigError('Supported GPU count: 1 or 2')
    if partition_override not in (None,partition,'all'):
        raise LaunchConfigError('Partition override must match the configured split')
    temperature=recipe.get('temperature')
    samples=recipe.get('samples_per_task',1)
    if temperature not in (0,0.6) or samples not in (1,4) or partition not in ('test','validation'):
        raise LaunchConfigError('Supported: Val/Test, temperature 0 or 0.6, 1 or 4 samples')
    if samples==4 and (temperature!=0.6 or recipe.get('resume_from') or recipe.get('cells')):
        raise LaunchConfigError('Pass@4 requires temperature 0.6, full split, no reused attempts')
    if gpus==2 and samples!=4:raise LaunchConfigError('Two GPUs are supported for independent Pass@4 rounds')
    sample_resumes=recipe.get('sample_resume_from',{})
    if set(sample_resumes)-{str(i) for i in range(1,samples+1)}:raise LaunchConfigError('Invalid sample resume index')
    policy_path=resolve_project_path(root,recipe['selector_policy'])
    schedule_path=resolve_project_path(root,recipe['schedule'])
    if not policy_path.exists():raise LaunchConfigError('Run the configured fault selector before fault evaluation: '+str(recipe.get('selector_config',policy_path)))
    policy=json.loads(policy_path.read_text());schedule=load_schedule(schedule_path, root=root)
    rows=policy['rows']
    expected=recipe['expected_task_modes']
    if len(rows)!=expected or len({(r['task_id'],r['mode']) for r in rows})!=expected or any(r['partition']!=partition for r in rows):
        raise LaunchConfigError('unexpected held-out task-mode coverage')
    if policy['temperature']!=0 or policy['base_model']['temperature']!=0 or policy['concurrency']!=2 or policy['replicas']!=1:
        raise LaunchConfigError('source policy must use temperature 0, one replica and concurrency 2')
    budgets={} if recipe.get('clean_first') else assign_budgets(rows,schedule)
    if recipe.get('sampling_layout')=='taskwise':
        if gpus!=2 or samples!=4 or temperature!=0.6:raise LaunchConfigError('Taskwise Pass4 requires two GPUs, four samples, temperature 0.6')
        from robustops.evaluation.taskwise_fault_pass4 import reusable_attempts
        reused_counts={}
        for sample,path in sample_resumes.items():
            reused_counts[sample]=len(reusable_attempts(resolve_project_path(root,path),rows,schedule,budgets,temperature,(int(sample)-1)*1000003,recipe['model_name']))
        command=[sys.executable,'-m','robustops.evaluation.taskwise_fault_pass4',
                 '--config',str(recipe_path.resolve()),'--model-path',str(resolve_model_path(root,recipe['model_path']))]
        print(json.dumps(dict(command=command,gpus=2,task_modes=len(rows),samples_per_task=4,total_attempts=len(rows)*4,
            reused_valid=reused_counts,new_attempts=len(rows)*4-sum(reused_counts.values()),temperature=temperature,
            layout='taskwise shared queue: earliest ready task; per-sample ordered; one rollout per GPU',
            model_ports=[8010,8011],mcp_port_offsets=[1000,1100],dry_run=dry_run),indent=2),flush=True)
        if dry_run:return 0
        check_host_memory(recipe.get("hardware", {}).get("min_host_memory_gib", 0))
        return subprocess.run(command,cwd=root,check=False).returncode
    command=[sys.executable,'-m','robustops.evaluation.fault_evaluation','--runtime-priority',
             '--policy',str(policy_path),'--schedule',str(schedule_path),'--inference-config',str(recipe_path.resolve()),
             '--model-path',str(resolve_model_path(root,recipe['model_path'])),
             '--served-name',recipe['model_name'],'--output',str(resolve_project_path(root,recipe['output']))]
    command+=['--temperature',str(temperature)]
    if recipe.get('cells'):command+=['--cells',*map(str,recipe['cells'])]
    if recipe.get('resume_from'):command+=['--resume-from',str(resolve_project_path(root,recipe['resume_from']))]
    commands=[]
    for index in range(samples):
        item=list(command)
        if samples>1:
            item[item.index('--output')+1]=str(resolve_project_path(root,recipe['output'])/f'sample_{index+1}')
            item+=['--sampling-seed-offset',str(index*1000003)]
        worker=index%gpus
        item+=['--gpu-id',str(worker),'--model-port',str(8010+worker),'--mcp-port-offset',str(1000+100*worker)]
        if str(index+1) in sample_resumes:
            item+=['--resume-from',str(resolve_project_path(root,sample_resumes[str(index+1)]))]
        commands.append(item)
    histogram={k:dict(Counter(b[k] for b in budgets.values() if k in b)) for k in schedule['groups']}
    print(json.dumps(dict(pipeline='fault_assignment_evaluation',task_modes=len(rows),gpus=gpus,partition=partition,temperature=temperature,samples_per_task=samples,total_attempts=len(rows)*samples,
        budget_unit='cumulative_eligible_calls',candidate_dose_counts=histogram,commands=commands,
        clean_first=recipe.get('clean_first',False),clean_attempts=len(rows) if recipe.get('clean_first') else 0,
        cap_selection='after_mechanism_selection' if recipe.get('clean_first') else 'legacy_candidate_table',
        dry_run=dry_run),ensure_ascii=False,indent=2),flush=True)
    if dry_run:return 0
    check_host_memory(recipe.get("hardware", {}).get("min_host_memory_gib", 0))
    def lane(worker):
        for command in commands[worker::gpus]:
            if gpus==1:
                code=subprocess.run(command,cwd=root,check=False).returncode
            else:
                log_path=Path(command[command.index('--output')+1]).with_suffix('.log')
                log_path.parent.mkdir(parents=True,exist_ok=True)
                with log_path.open('w') as log:
                    code=subprocess.run(command,cwd=root,stdout=log,stderr=subprocess.STDOUT,check=False).returncode
            if code:return code
        return 0
    if gpus==1:
        code=lane(0)
        if code:return code
    else:
        with ThreadPoolExecutor(max_workers=gpus) as pool:
            codes=list(pool.map(lane,range(gpus)))
        if any(codes):return next(c for c in codes if c)
    if samples>1:
        output=resolve_project_path(root,recipe['output'])
        trials=[json.loads((output/f'sample_{i+1}'/'completion.json').read_text()) for i in range(samples)]
        report=summarize_pass_at_k(rows,trials)
        (output/'pass_at_4.json').write_text(json.dumps(report,indent=2)+'\n')
        print(json.dumps(report,indent=2),flush=True)
    return 0


def summarize_pass_at_k(rows,trials):
    expected={r['cell'] for r in rows}
    by_trial=[]
    for trial in trials:
        indexed={r['cell']:r for r in trial}
        if len(trial)!=len(expected) or set(indexed)!=expected or any(not r['valid'] for r in trial):
            raise LaunchConfigError('Pass@k requires all expected valid attempts in every sample')
        by_trial.append(indexed)
    details=[dict(task_id=r['task_id'],mode=r['mode'],cell=r['cell'],
                  successes=sum(bool(t[r['cell']]['success']) for t in by_trial),
                  fault_hit_samples=sum((t[r['cell']].get('hits') or 0)>0 for t in by_trial)) for r in rows]
    passed=sum(r['successes']>0 for r in details)
    return dict(k=len(trials),task_modes=len(rows),attempts=len(rows)*len(trials),
                passed=passed,pass_at_k=passed/len(rows),
                pass_at_1=sum(bool(r['success']) for r in by_trial[0].values())/len(rows),
                pass_at_1_successes=sum(bool(r['success']) for r in by_trial[0].values()),
                pass_at_1_definition='First sample only, temperature unchanged',
                per_sample_successes=[sum(bool(r['success']) for r in t.values()) for t in by_trial],
                interpretation='Any of k independently sampled trajectories passes the task verifier; fault exposure may vary by trajectory.',
                rows=details)
