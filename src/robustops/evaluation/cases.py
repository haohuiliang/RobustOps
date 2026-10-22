"""One fixed-case inference script for local models and API teacher modes."""
import asyncio
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from robustops.providers.credentials import credential
from robustops.data.collection_pool import load_collection_pool
from robustops.data.recovery_sampling import StudentContextCounter, teacher_guidance
from robustops.data.teacher_sft import student_export_allowed, student_record, student_token_proof
from robustops.evaluation.clean_baseline import TokenCounter, _tool_content, run_task_attempt
from robustops.evaluation.runtime_services import FaultReplica, Model, launch as launch_mcp, stop_processes
from robustops.evaluation.selected_fault import SelectedFaultController
from robustops.launch.config import LaunchConfigError, resolve_project_path


def read(path):
    return json.loads(Path(path).read_text())


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')


class CaseFaultController(SelectedFaultController):
    """Optional fixed requested-call indices still require normal eligibility."""
    def __init__(self, payload, selection, spec, footprint):
        super().__init__(payload, selection, spec['kind'], spec['budget'], spec['binding'], footprint)
        self.spec = spec

    async def call_tool(self, name, args):
        index = len(self.calls)+1
        outside = self.spec['point_policy']=='fixed_call_indices' and index not in self.spec['call_indices']
        budget = self.budget
        try:
            if outside:
                self.budget=len(self.events)
            result=await super().call_tool(name,args)
        finally:
            self.budget=budget
        if outside:
            self.calls[-1]['controller_skip']='outside_fixed_call_indices'
        return result

    def report(self):
        result=super().report()
        result.update(target_mechanism=self.kind,case_point_policy=self.spec['point_policy'],
                      fixed_call_indices=self.spec.get('call_indices'))
        return result


def build_plan(root, recipe):
    from robustops.training.recovery_rl_data import load_task
    if recipe['mode'] not in ('evaluation','teacher_data'):
        raise LaunchConfigError('Case mode must be evaluation or teacher_data')
    backend=recipe['backend']
    if backend['kind'] not in ('local','deepseek_api') or 'api_key' in backend:
        raise LaunchConfigError('Use local or deepseek_api with credentials outside the recipe')
    if recipe['mode']=='teacher_data' and backend['kind']!='deepseek_api':
        raise LaunchConfigError('teacher_data mode requires the teacher API')
    limits=recipe['limits']
    if (limits['max_context_tokens']!=32768 or limits['max_generation_tokens']!=2048
            or limits['max_turns']!=50 or limits['temperature']!=0):
        raise LaunchConfigError('Case inference keeps the fixed 32K/2K/50-turn temperature-0 contract')
    cases=read(root/recipe['cases'])['cases']
    if not cases or len({c['case_id'] for c in cases})!=len(cases):
        raise LaunchConfigError('Case IDs must be nonempty and unique')
    if recipe.get('case_ids'):
        requested=set(recipe['case_ids'])
        unknown=requested-{c['case_id'] for c in cases}
        if unknown:
            raise LaunchConfigError(f'Unknown case IDs: {sorted(unknown)}')
        cases=[c for c in cases if c['case_id'] in requested]
    manifest={r['task_id']:r for r in map(json.loads,(root/'data/task_sets/enterpriseops.jsonl').read_text().splitlines())}
    pool_path=recipe.get('teacher',{}).get('pool','data/task_sets/sft_collection_pool.json')
    if recipe['mode']=='teacher_data':
        try:
            pool=load_collection_pool(root/pool_path,manifest.values())
        except ValueError as exc:
            raise LaunchConfigError(str(exc)) from exc
        training={(r['task_id'],m) for r in pool['tasks'] for m in r['modes']}
    resolved=[]
    for case in cases:
        if not case['case_id'].isidentifier() or type(case['seed']) is not int:
            raise LaunchConfigError('Case IDs must be identifiers; seed must be fixed')
        meta=manifest[case['task_id']]
        if (meta['partition']!=case['partition'] or case['mode'] not in meta['runnable_modes']
                or meta.get('execution_blocked_reason')):
            raise LaunchConfigError('Case split/mode differs from the fixed runnable task manifest')
        if recipe['mode']=='teacher_data' and (case['partition']!='train' or (case['task_id'],case['mode']) not in training):
            raise LaunchConfigError('Teacher case training data must come from the selected Train collection manifest')
        conditions=case['conditions']
        if not conditions or len(set(conditions))!=len(conditions) or set(conditions)-{'clean','fault'}:
            raise LaunchConfigError('Case conditions must be distinct clean/fault values')
        fault=case.get('fault')
        if 'fault' in conditions:
            eligible={k for t in case['selection']['tools'] for k in t['faults']}
            if (not fault or fault['kind'] not in eligible or type(fault['budget']) is not int
                    or fault['budget'] not in (1,2,4) or fault['binding']!='eligible_calls'
                    or fault['kind'].startswith('unknown_commit') and fault['budget']>2):
                raise LaunchConfigError('Fault must fix an eligible mechanism and finite shared dose')
            if recipe['mode']=='teacher_data' and fault['kind']=='ignored_filter':
                raise LaunchConfigError('Held-out mechanism cannot enter teacher training data')
            if fault['point_policy'] not in ('earliest_strictly_eligible','fixed_call_indices'):
                raise LaunchConfigError('Unsupported case injection point policy')
            if fault['point_policy']=='fixed_call_indices':
                indices=fault.get('call_indices',[])
                if (not indices or len(set(indices))!=len(indices) or len(indices)<fault['budget']
                        or any(type(i) is not int or i<1 for i in indices)):
                    raise LaunchConfigError('Fixed call indices must be distinct positive integers covering the dose')
        task=load_task(root,case['domain'],case['mode'],case['task_id'])
        if case['initial_state']!='seeded_task':
            snapshot=read(root/case['initial_state'])
            case=dict(case,initial_state_sha256=hashlib.sha256(json.dumps(snapshot,sort_keys=True).encode()).hexdigest())
        fixed=dict(case,task=dict(system_policy=task.system_prompt,user_request=task.user_prompt),
                   verifiers=list(task.verifiers),servers=task.gym_servers)
        resolved.append(fixed)
    guidance=teacher_guidance(recipe['teacher']['guidance_variant']) if recipe['mode']=='teacher_data' else ''
    if recipe['mode']=='teacher_data' and not guidance:
        raise LaunchConfigError('Teacher-data mode requires explicit guidance')
    plan=dict(mode=recipe['mode'],backend=backend,cases=resolved,limits=limits,
              guidance=guidance,tokenizer=recipe['tokenizer'],controller='first_tool_only_plan_preserving_v1',
              provider_seed_supported=backend['kind']=='local',
              selection='One fixed attempt per condition; no success-based replacement or no-hit resampling.')
    if recipe['mode']=='teacher_data':
        plan['collection_pool']=pool_path
    # Canonical JSON types make a saved plan equal to a freshly rebuilt plan.
    plan=json.loads(json.dumps(plan))
    plan['sha256']=hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()
    return plan


async def execute_case(root, recipe, case, task, urls, endpoint, counter, guidance, folder, key, initial_state):
    """Shared execution and optional student export; mode changes guidance only."""
    limits=recipe['limits'];backend=recipe['backend']
    model_config=dict(endpoint=endpoint,model=backend['model'],temperature=limits['temperature'],
                      max_tokens=limits['max_generation_tokens'],provider='deepseek' if backend['kind']=='deepseek_api' else 'local_vllm')
    results=[]
    for condition in case['conditions']:
        target=folder/condition;path=target/'attempt.json'
        if path.exists():
            attempt=read(path)
            if attempt['valid'] and attempt['cleanup']['succeeded']:
                results.append(dict(condition=condition,success=attempt['success'],valid=True,
                    injected_calls=attempt['fault'].get('injected_calls',0),source=str(path)))
                continue
            raise LaunchConfigError('Invalid case attempt exists; preserve it and choose a new output')
        (target/'calls').mkdir(parents=True,exist_ok=False)
        model=Model(model_config,target/'calls',case['seed'] if backend['kind']=='local' else None,key)
        guided=replace(task,system_prompt=task.system_prompt+('\n\n'+guidance if guidance else ''))
        visible_counter=StudentContextCounter(counter,task.system_prompt,guidance) if guidance else counter
        spec=case.get('fault')
        controller=(CaseFaultController(case,case['selection'],spec,lambda x:counter.text(_tool_content(x)))
                    if condition=='fault' else None)
        try:
            attempt=await run_task_attempt(task=guided,model=model,token_counter=visible_counter,
                archive_path=root/'enterpriseops_gym/gym_dbs.zip',mcp_url_overrides=urls,
                max_turns=limits['max_turns'],max_context_tokens=limits['max_context_tokens'],
                max_generation_tokens=limits['max_generation_tokens'],seed=case['seed'],
                mode=case['mode'],experiment_id='fixed_cases',run_id=folder.parent.name,
                model_name=backend['model'],model_endpoint=endpoint,plan_preserving=True,
                initial_state=initial_state,tool_intervention=controller)
            save(path,attempt);save(folder/'initial_state.json',initial_state)
            result=dict(condition=condition,success=attempt['success'],valid=attempt['valid'],
                        injected_calls=attempt['fault'].get('injected_calls',0),source=str(path))
            if (recipe['mode']=='teacher_data' and attempt['valid'] and attempt['success']
                    and attempt['cleanup']['succeeded'] and student_export_allowed(attempt)
                    and (condition=='clean' or result['injected_calls']>0)):
                request=read(sorted((target/'calls').glob('*.request.json'))[-1])
                record=student_record(task,attempt,request['tools'],request['messages'],
                    f"case-{case['case_id']}-{condition}",guidance)
                proof=student_token_proof(record,counter.tokenizer)
                save(target/'student_token_check.json',proof)
                if proof['within_limit']:
                    save(target/'student_candidate.json',record)
                    result['student_candidate']=True
            results.append(result)
            if not attempt['cleanup']['succeeded']:
                raise RuntimeError('Case database cleanup failed')
        finally:
            await model.client.close()
    return results


async def run(root, recipe, plan, gpu_override):
    from robustops.launch.hardware import select_gpus
    from robustops.training.recovery_rl_data import load_task
    from robustops.training.recovery_rl_storage import validate_hf_export
    output=resolve_project_path(root,recipe['output'])
    if output.exists():
        if not recipe['resume'] or read(output/'plan.json')!=plan:
            raise LaunchConfigError('Case plan changed; preserve the old output and choose a new directory')
    else:
        output.mkdir(parents=True);save(output/'plan.json',plan)
    from datetime import datetime, timezone
    runtime_root=output/'runtime'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    runtime_root.mkdir(parents=True)
    counter=TokenCounter(resolve_project_path(root,recipe['tokenizer']),enable_thinking=False)
    required={s['mcp_server_name'] for c in plan['cases'] for s in c['servers']}
    backend=recipe['backend']; replicas=[];processes=[];results={}
    try:
        if backend['kind']=='local':
            model_path=resolve_project_path(root,backend['path'])
            validate_hf_export(model_path,allowed_dtypes=('BF16',) if backend.get('weight_dtype')=='bfloat16' else ('F32',))
            selection=select_gpus(gpu_override,dry_run=False)
            for worker in range(min(selection.count,len(plan['cases']))):
                replica=FaultReplica(worker,model_path,resolve_project_path(root,recipe['tokenizer']),backend['model'],
                                     runtime_root/f'worker_{worker}', max_context_tokens=recipe['limits']['max_context_tokens'])
                replicas.append(replica);await replica.start(required)
        else:
            key=credential(dict(backend,local_env_file=str(root/backend['local_env_file']) if backend.get('local_env_file') else None))
            processes,urls=await launch_mcp(required,runtime_root,port_offset=recipe['mcp_port_offset'])
        # Each worker keeps its model while consuming the next pending fixed case.
        queue=asyncio.Queue()
        for case in plan['cases']:queue.put_nowait(case)
        async def lane(replica=None):
            while not queue.empty():
                case=queue.get_nowait();folder=output/case['case_id']
                task=load_task(root,case['domain'],case['mode'],case['task_id'])
                state_path=folder/'initial_state.json'
                state=read(state_path) if state_path.exists() else ({} if case['initial_state']=='seeded_task' else read(root/case['initial_state']))
                results[case['case_id']]=await execute_case(root,recipe,case,task,
                    replica.urls if replica else urls,replica.endpoint if replica else backend['endpoint'],
                    counter,plan['guidance'],folder,None if replica else key,deepcopy(state))
                save(output/'summary.json',results)
        await asyncio.gather(*(lane(replica) for replica in replicas)) if replicas else await lane()
    finally:
        for replica in replicas:replica.close()
        stop_processes(processes)
        save(output/'cleanup.json',dict(mcp_stopped=all(p.poll() is not None for p in processes)))


def launch(root, recipe_path, recipe, *, gpu_override, partition_override, dry_run):
    if recipe['backend']['kind']=='deepseek_api' and gpu_override not in (None,0):
        raise LaunchConfigError('API case inference does not select GPUs')
    plan=build_plan(root,recipe)
    if partition_override not in (None,'all') and any(c['partition']!=partition_override for c in plan['cases']):
        raise LaunchConfigError('Case partition override differs from its fixed cases')
    print(json.dumps(dict(mode=plan['mode'],backend=plan['backend']['kind'],
        cases=[dict(case_id=c['case_id'],task_id=c['task_id'],mode=c['mode'],conditions=c['conditions'],fault=c.get('fault')) for c in plan['cases']],
        plan_sha256=plan['sha256'],guidance_appended=bool(plan['guidance']),dry_run=dry_run),indent=2),flush=True)
    if not dry_run:asyncio.run(run(root,recipe,plan,gpu_override))
    return 0
