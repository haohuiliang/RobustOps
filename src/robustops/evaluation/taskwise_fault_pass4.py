"""Four independent rollouts per task, a work queue feeding two isolated GPU/MCP replicas."""
import argparse
import asyncio
from copy import deepcopy
import json
from pathlib import Path
import subprocess

from robustops.evaluation.runtime_services import FaultReplica, Model, dump
from robustops.evaluation.fault_schedule import FaultScheduler, assign_budgets, load_schedule
from robustops.evaluation.fault_scheduling_controller import RuntimePriorityFault, run_with_nohit_replay
from robustops.evaluation.selected_fault import SelectedFaultController
from robustops.evaluation.clean_baseline import TokenCounter, run_task_attempt, _tool_content
from robustops.envs.enterpriseops.task import load_tasks
from robustops.launch.fault_assignment import summarize_pass_at_k

ROOT=Path(__file__).resolve().parents[3]
def read(path):return json.loads(Path(path).read_text())
def result_row(row, attempt, source=None):
    result=dict(cell=row['cell'],condition='runtime_priority',success=attempt['success'],valid=attempt['valid'],
                hits=attempt['fault'].get('injected_calls',0),termination=(attempt.get('trajectory') or {}).get('termination'))
    if source:result['reused_from']=str(source)
    return result


def reusable_attempts(source, rows, schedule, budgets, temperature, offset, model_name):
    """Reuse valid completed results only; invalid/cancelled attempts start fresh."""
    prior=read(source/'policy_used.json')
    if (prior['temperature']!=temperature or prior.get('sampling_seed_offset',0)!=offset
            or prior['base_model']['model']!=model_name or prior['runtime_schedule']!=schedule):
        raise ValueError('Resume sampling/model/schedule mismatch')
    if read(source/'candidate_budgets.json')!=budgets:raise ValueError('Resume budget mismatch')
    old={r['cell']:r for r in prior['rows']}
    if set(old)!={r['cell'] for r in rows}:raise ValueError('Resume task set mismatch')
    inherited=read(source/'reused_results.json') if (source/'reused_results.json').exists() else {}
    found={}
    for row in rows:
        if any(row[k]!=old[row['cell']][k] for k in ('task_id','mode','payload','selection','seed')):
            raise ValueError('Resume task changed')
        path=source/'runtime_priority'/f"cell_{row['cell']:02}"/'attempt.json'
        if not path.exists() and str(row['cell']) in inherited:path=Path(inherited[str(row['cell'])])
        if path.exists():
            attempt=read(path)
            if attempt['valid'] and attempt['cleanup']['succeeded']:
                if attempt.get('sampling')!=dict(temperature=temperature,seed=row['seed']+offset):
                    raise ValueError('Resume attempt seed mismatch')
                found[row['cell']]=(path,attempt)
    return found


async def run_rollout_queue(rows, operation, *, workers=2, samples=4):
    """Feed idle GPUs from four ordered sample streams, without task barriers.

    One in-flight attempt per sample preserves that sample's balance history.
    Prefer the earliest unfinished task, so four attempts stay close together.
    """
    next_row=[0]*samples
    active=set()
    condition=asyncio.Condition()
    failed=False
    async def worker(gpu):
        nonlocal failed
        while True:
            async with condition:
                while True:
                    if failed:return
                    ready=[i for i in range(samples) if i not in active and next_row[i]<len(rows)]
                    if ready:break
                    if all(n==len(rows) for n in next_row):return
                    await condition.wait()
                sample=min(ready,key=lambda i:(next_row[i],i))
                index=next_row[sample]
                active.add(sample)
            try:
                await operation(rows[index],sample,gpu)
            except BaseException:
                async with condition:
                    failed=True
                    active.remove(sample)
                    condition.notify_all()
                raise
            async with condition:
                next_row[sample]+=1
                active.remove(sample)
                condition.notify_all()
            # A reused result may complete synchronously. Let the other GPU
            # claim work rather than consuming its stream in this worker.
            await asyncio.sleep(0)
    async with asyncio.TaskGroup() as group:
        for gpu in range(workers):group.create_task(worker(gpu))


async def run(args):
    recipe=read(args.config); source=read(ROOT/recipe['selector_policy']); rows=source['rows']
    schedule=load_schedule(ROOT/recipe['schedule']); budgets=assign_budgets(rows,schedule)
    output=ROOT/recipe['output']; output.mkdir(parents=True,exist_ok=False)
    dump(output/'resolved_inference_config.json',recipe)
    dump(output/'code_state.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)))
    for rel in ['src/robustops/evaluation/taskwise_fault_pass4.py','src/robustops/evaluation/runtime_services.py','src/robustops/evaluation/fault_scheduling_controller.py','src/robustops/evaluation/fault_schedule.py','src/robustops/envs/enterpriseops/runtime.py','src/robustops/launch/fault_assignment.py']:
        target=output/'code'/rel;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/rel).read_bytes())
    dump(output/'model_used.json',dict(path=str(args.model_path.resolve()),name=recipe['model_name']))
    (output/'runtime_code.patch').write_text(subprocess.check_output(['git','diff','--','src/robustops/evaluation','src/robustops/envs/enterpriseops/runtime.py','src/robustops/launch/fault_assignment.py'],cwd=ROOT,text=True))
    tasks={}; by_domain={}
    for row in rows:
        key=(row['domain'],row['mode'])
        if key not in by_domain:
            cfg=read(ROOT/f"recipes/enterpriseops/paired/{'oracle' if row['mode']=='oracle' else 'plus5'}/{row['domain']}.json")
            by_domain[key]={t.task_id:t for t in load_tasks(ROOT/cfg['data_path'],expected_domain=row['domain'],expected_count=cfg['expected_task_count'])}
        task=by_domain[key][row['task_id']];tasks[row['cell']]=task
        assert task.system_prompt==row['payload']['task']['system_policy'] and task.user_prompt==row['payload']['task']['user_request'] and list(task.verifiers)==row['payload']['verifiers']
    counter=TokenCounter(ROOT/source['tokenizer'],enable_thinking=False)
    balances=[FaultScheduler(rows,schedule['seed']) for _ in range(4)]
    results=[{} for _ in range(4)]; reused=[{} for _ in range(4)]
    for i in range(4):
        dest=output/f'sample_{i+1}';dest.mkdir()
        plan=deepcopy(source);plan.update(temperature=0.6,sampling_seed_offset=i*1000003,runtime_schedule=schedule)
        plan['base_model'].update(model=recipe['model_name'],temperature=0.6)
        dump(dest/'policy_used.json',plan);dump(dest/'candidate_budgets.json',budgets)
        resume=recipe.get('sample_resume_from',{}).get(str(i+1))
        if resume:
            reused[i]=reusable_attempts(ROOT/resume,rows,schedule,budgets,0.6,i*1000003,recipe['model_name'])
            for cell,(path,attempt) in reused[i].items():
                row=next(r for r in rows if r['cell']==cell)
                results[i][cell]=result_row(row,attempt,path)
            # Claims are replayed at their task position, not all preloaded:
            # future reused tasks must not alter the priority of earlier tasks.
        dump(dest/'reused_results.json',{str(c):str(p) for c,(p,a) in reused[i].items()})
    replicas=[FaultReplica(i,args.model_path,ROOT/source['tokenizer'],recipe['model_name'],output/f'replica_{i}', max_context_tokens=source['max_context_tokens']) for i in range(2)]
    required={g['mcp_server_name'] for t in tasks.values() for g in t.gym_servers}
    async def episode(row,i,gpu):
        cell=row['cell']; dest=output/f'sample_{i+1}'
        if cell in reused[i]:
            fault=reused[i][cell][1]['fault']
            if fault['injected_calls']:balances[i].claim(fault['selected_mechanism'],str(cell))
            return
        replica=replicas[gpu]
        path=dest/'runtime_priority'/f'cell_{cell:02}';(path/'calls').mkdir(parents=True,exist_ok=False)
        seed=row['seed']+i*1000003
        cfg=dict(source['base_model'],model=recipe['model_name'],temperature=0.6,endpoint=replica.endpoint)
        model=Model(cfg,path/'calls',seed)
        factory=lambda target:RuntimePriorityFault(row['payload'],row['selection'],budgets[str(cell)],SelectedFaultController,
            lambda value:counter.text(_tool_content(value)),target,None,str(cell),schedule,balances[i])
        kwargs=dict(task=tasks[cell],token_counter=counter,archive_path=ROOT/'enterpriseops_gym/gym_dbs.zip',
            mcp_url_overrides=replica.urls,max_turns=source['max_turns'],max_context_tokens=source['max_context_tokens'],
            max_generation_tokens=source['max_generation_tokens'],experiment_id='TASKWISE-FAULT-PASS4',run_id=dest.name,
            mode=row['mode'],model_name=recipe['model_name'],model_endpoint=replica.endpoint,seed=row['seed'],plan_preserving=True)
        try:
            if schedule['max_zero_hit_replays']:
                outcome=await run_with_nohit_replay(controller_factory=factory,model=model,attempt_kwargs=kwargs,
                    save_attempt=lambda branch,a:dump(path/(branch+'.attempt.json'),a))
                attempt=outcome.pop('final');dump(path/'branch_selection.json',outcome)
            else:
                attempt=await run_task_attempt(model=model,tool_intervention=factory(None),**kwargs)
            attempt['sampling']=dict(temperature=0.6,seed=seed)
            dump(path/'attempt.json',attempt);dump(dest/'balance.json',balances[i].report())
            results[i][cell]=result_row(row,attempt)
            dump(output/'progress.json',dict(completed=[len(r) for r in results],last_cell=cell,last_sample=i+1,expected_per_sample=len(rows)))
            print('DONE',cell,'sample',i+1,'success',attempt['success'],'valid',attempt['valid'],flush=True)
            if not attempt['valid'] or not attempt['cleanup']['succeeded']:raise RuntimeError('Invalid attempt or cleanup failed; stop batch')
        finally:await model.client.close()
    try:
        # Start sequentially: if the second replica fails, the first is still owned and cleaned.
        for replica in replicas:await replica.start(required)
        await run_rollout_queue(rows,episode)
        trials=[list(r.values()) for r in results]
        for i,trial in enumerate(trials):dump(output/f'sample_{i+1}'/'completion.json',trial)
        report=summarize_pass_at_k(rows,trials);dump(output/'pass_at_4.json',report)
        print('PASS@1',report['pass_at_1'],'PASS@4',report['pass_at_k'],flush=True)
    finally:
        for replica in reversed(replicas):replica.close()
        dump(output/'resource_cleanup.json',dict(completed=[len(r) for r in results],reused=[len(r) for r in reused],
            gpu_stopped=all(r.gpu is None or r.gpu.poll() is not None for r in replicas),
            mcp_stopped=all(p.poll() is not None for r in replicas for p in r.mcp)))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True);parser.add_argument('--model-path',type=Path,required=True)
    args=parser.parse_args();asyncio.run(run(args))
if __name__=='__main__':main()
