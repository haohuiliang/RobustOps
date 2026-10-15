"""Single-fault evaluation using the common inference and MCP lifecycle."""
import argparse,asyncio,json,os,sys,subprocess,time
from pathlib import Path
import httpx
from robustops.evaluation.selected_fault import SelectedFaultController
from robustops.evaluation.fault_task_caps import FaultTaskCaps
from robustops.evaluation.fault_schedule import FaultScheduler, assign_budgets, validate_schedule, load_schedule, observation_budgets, select_clean_branches
from robustops.evaluation.clean_fault_branch import run_clean_branch
from copy import deepcopy
from robustops.evaluation.fault_scheduling_controller import RuntimePriorityFault, PRIORITY, run_with_nohit_replay
from robustops.evaluation.runtime_services import launch,stop_processes,Model
from robustops.evaluation.clean_baseline import TokenCounter,run_task_attempt,_tool_content
from robustops.envs.enterpriseops.task import load_tasks
ROOT=Path(__file__).resolve().parents[3]

def read(p):return json.loads(Path(p).read_text())
def save(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
async def run(args):
 OUT=args.output.resolve();OUT.mkdir(parents=True,exist_ok=False)
 config=read(args.inference_config) if args.inference_config else {}
 if config.get('output') and (ROOT/config['output']).resolve()!=OUT:
  config['output']=str(OUT)
 paired_clean=config.get('clean_first',False)
 clean_sources={};branch_plans={}
 repair=config.get('fixed_fault_replay') or {}
 fixed=read(ROOT/config['base_fixed_mechanisms']) if config.get('base_fixed_mechanisms') else None
 if fixed and (not args.runtime_priority or repair or args.cells or args.fault_budget is not None):raise ValueError('Base-fixed Val requires full runtime evaluation without a task subset or dose override')
 panel=read(ROOT/config['frozen_panel']) if config.get('frozen_panel') else None
 if fixed and panel:raise ValueError('Base-fixed Val cannot use the frozen panel')
 frozen={str(r['cell']):r for r in panel['rows']} if panel else {}
 if panel and (not args.runtime_priority or args.resume_from or repair):raise ValueError('frozen panel requires independent runtime evaluation')
 plan=read(args.policy);rows=plan['rows'];all_rows=list(rows);scheduled=False;balance=None;dose_table=None;results=[];procs=[];gpu=None;log=None;reused={};resume_root=args.resume_from.resolve() if args.resume_from else None
 if resume_root and (not args.runtime_priority or args.cells):raise ValueError('resume requires full runtime batch')
 assert rows and len({(r['task_id'],r['mode']) for r in rows})==len(rows)
 partitions={r['partition'] for r in rows}
 if len(partitions)!=1 or not partitions<={'test','validation'}:raise ValueError('one held-out partition required')
 fixed_by_cell={}
 if fixed:
  if fixed.get('partition')!='validation' or partitions!={'validation'} or len(rows)!=127 or len(fixed.get('rows',[]))!=127:raise ValueError('Base-fixed manifest must cover all 127 Val cells')
  fixed_by_cell={str(s['cell']):s for s in fixed['rows']}
  if len(fixed_by_cell)!=127 or set(fixed_by_cell)!={str(r['cell']) for r in rows}:raise ValueError('Base-fixed cell coverage mismatch')
  for r in rows:
   s=fixed_by_cell[str(r['cell'])]
   if any(s[k]!=r[k] for k in ('task_id','mode','domain')):raise ValueError(f"Base-fixed task mismatch: {r['cell']}")
   if s['kind'] not in {kind for tool in r['selection']['tools'] for kind in tool['faults']}:raise ValueError(f"Base-fixed mechanism disabled: {r['cell']}")
  save(OUT/'base_fixed_mechanisms.json',fixed)
 for rel in ([] if args.runtime_priority else plan['frozen_sources']):
  if (ROOT/rel).read_bytes()!=(args.policy.parent/'code'/rel).read_bytes():raise RuntimeError(f'Frozen implementation changed: {rel}; do not silently compare different policies')
 if args.cells:
  if not args.runtime_priority:raise ValueError('--cells only applies to runtime evaluation')
  rows=[r for r in rows if r['cell'] in args.cells]
  if {r['cell'] for r in rows}!=set(args.cells):raise ValueError('unknown selected cell')
 if panel:
  from robustops.evaluation.frozen_panel import validate_panel
  validate_panel(ROOT,config)
  if set(frozen)!={str(r['cell']) for r in rows}:raise ValueError('panel/cells mismatch')
  save(OUT/'frozen_panel.json',panel)
 if args.runtime_priority:
  schedule=load_schedule(args.schedule)
  if args.inference_config:save(OUT/'resolved_inference_config.json',config)
  scheduled=schedule.get('scheduler')=='finite_budget'
  if fixed and not scheduled:raise ValueError('Base-fixed Val requires the frozen scheduled dose schedule')
  if scheduled:
   validate_schedule(schedule)
   if args.fault_budget is not None:raise ValueError('scheduled repetition limits come from the shared strategy or fixed panel')
   dose_table=(read(ROOT/fixed['candidate_budgets']) if fixed else
               read(ROOT/schedule['candidate_budgets']) if panel else
               read(resume_root/'candidate_budgets.json') if paired_clean and resume_root else
               {} if paired_clean else assign_budgets(all_rows,schedule))
   if fixed and any(s['budget']!=dose_table[str(s['cell'])].get(s['kind']) for s in fixed['rows']):raise ValueError('Base-fixed dose table changed')
   save(OUT/'candidate_budgets.json',dose_table)
   balance=None if panel or fixed else FaultScheduler(all_rows,schedule['seed']);task_caps=None
  else:
   budget=args.fault_budget if args.fault_budget is not None else schedule['injection_budget_per_task']
   if budget not in (1,2,3,4):raise ValueError('invalid injection budget')
   if schedule['point_policy']!='earliest_strictly_eligible' or schedule['max_zero_hit_replays']!=1 or schedule['cap_rounding']!='ceil':raise ValueError('unsupported schedule')
   schedule['injection_budget_per_task']=budget
   task_caps=FaultTaskCaps(schedule,len(rows),{str(r['cell']):(r['mode'],r['domain']) for r in rows})
  if resume_root:
   prior=read(resume_root/'policy_used.json')
   if fixed and read(resume_root/'base_fixed_mechanisms.json')!=fixed:raise ValueError('resume Base mechanism manifest changed')
   if prior['temperature']!=args.temperature or prior.get('sampling_seed_offset',0)!=args.sampling_seed_offset or prior['base_model']['model']!=args.served_name:raise ValueError('resume sampling/model mismatch')
   if prior['runtime_schedule']!=schedule or bool(prior.get('clean_first'))!=paired_clean:raise ValueError('resume schedule/collection flow changed')
   if paired_clean:
    inherited_clean=read(resume_root/'clean_sources.json') if (resume_root/'clean_sources.json').exists() else {}
    for row in rows:
     key=str(row['cell']);folder=resume_root/'clean'/f"cell_{row['cell']:02}"
     if not (folder/'attempt.json').exists() and key in inherited_clean:folder=Path(inherited_clean[key])
     if (folder/'attempt.json').exists() and (folder/'checkpoints.json').exists():
      donor=read(folder/'attempt.json')
      if donor['valid'] and donor['cleanup']['succeeded']:clean_sources[key]=str(folder)
   current={(r['task_id'],r['mode']):r for r in rows}
   inherited=read(resume_root/'reused_results.json') if (resume_root/'reused_results.json').exists() else {}
   for old in prior['rows']:
    r=current[(old['task_id'],old['mode'])]
    if any(r[k]!=old[k] for k in ('cell','payload','selection','seed')):raise ValueError('resume task/selection changed')
    path=resume_root/('base_fixed_kind' if fixed else 'runtime_priority')/f"cell_{old['cell']:02}"/'attempt.json'
    if not path.exists() and str(old['cell']) in inherited:path=Path(inherited[str(old['cell'])])
    if path.exists():
     a=read(path)
     if a['valid'] and a['cleanup']['succeeded']:
      reused[r['cell']]=str(path)
      results.append(dict(cell=r['cell'],condition='base_fixed_kind' if fixed else 'runtime_priority',success=a['success'],valid=True,hits=a['fault']['injected_calls'],target_mechanism=a['fault'].get('target_mechanism'),no_hit=a['fault'].get('no_hit'),termination=(a.get('trajectory') or {}).get('termination'),reused_from=str(path)))
   if fixed:pass
   elif scheduled:
    if read(resume_root/'candidate_budgets.json')!=dose_table:raise ValueError('resume dose assignments changed')
    for key,path in reused.items():
     f=read(Path(path))['fault']
     if f['injected_calls']:balance.claim(f['selected_mechanism'],str(key))
    save(OUT/'balance_at_resume.json',balance.report())
   else:
    sampled=[str(int(p.name.split('_')[1])) for p in (resume_root/'runtime_priority').glob('cell_*') if p.is_dir()]
    cap_report=read(resume_root/'task_caps.json')
    sampled=set(sampled)|set(cap_report.get('sampled_task_keys',[]))
    task_caps.restore(cap_report,sampled)
    save(OUT/'caps_at_resume.json',task_caps.report())
   save(OUT/'reused_results.json',reused)
  sources=[str(Path(m.__file__).resolve().relative_to(ROOT)) for m in list(sys.modules.values()) if getattr(m,'__file__',None) and str(Path(m.__file__).resolve()).startswith(str(ROOT/'src/robustops')) and str(m.__file__).endswith('.py')]
  rows=[{k:r[k] for k in ('cell','task_id','domain','mode','partition','payload','selection','selection_source','seed')} for r in rows]
  plan={k:plan[k] for k in ('base_model','tokenizer','temperature','max_turns','max_context_tokens','max_generation_tokens','replicas','concurrency')}
  plan=dict(plan,rows=rows,selector_source=str(args.policy),policy_id='runtime_priority_replay',frozen=False,conditions=['runtime_priority'],runtime_schedule=schedule,restoration='actual_boundary_snapshot',continued_from=str(resume_root) if resume_root else None,experiment_id='RUNTIME-PRIORITY-REPLAY')
  if scheduled:plan.update(policy_id='scheduled_single_fault',restoration='actual_boundary_snapshot' if schedule['max_zero_hit_replays'] else 'disabled',experiment_id='BASE-FAULT-'+next(iter(partitions)).upper())
  if fixed:plan.update(policy_id='base_fixed_kind_val',conditions=['base_fixed_kind'],restoration='disabled',experiment_id='BASE-FIXED-KIND-VAL',base_fixed_mechanisms=config['base_fixed_mechanisms'])
  save(OUT/'code_state.json',dict(revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=subprocess.check_output(['git','status','--short'],cwd=ROOT,text=True)))
  (OUT/'runtime_code.patch').write_text(subprocess.check_output(['git','diff','--','src/robustops/evaluation','src/robustops/data/fault_candidates.py','src/robustops/launch/fault_assignment.py'],cwd=ROOT,text=True))
  for rel in sources+['src/robustops/evaluation/fault_scheduling_controller.py','src/robustops/evaluation/clean_fault_branch.py','src/robustops/evaluation/fault_task_caps.py','src/robustops/evaluation/seed_state.py','src/robustops/evaluation/fault_checkpoint.py','src/robustops/envs/enterpriseops/sqlite_snapshot.py','src/robustops/evaluation/fault_schedule.py','src/robustops/launch/fault_assignment.py']:
   dst=OUT/'code'/rel;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/rel).read_bytes())
 if paired_clean:plan.update(conditions=['clean','runtime_priority'],clean_first=True,restoration='clean_boundary_snapshot')
 if panel:plan.update(conditions=['clean','runtime_priority'],frozen_panel=config['frozen_panel'],priority_policy='historical_per_cell_initial_priority')
 assert plan['temperature']==0 and plan['base_model']['temperature']==0, 'Evaluation temperature must be zero'
 plan['temperature']=args.temperature
 plan['sampling_seed_offset']=args.sampling_seed_offset
 plan['base_model']=dict(plan['base_model'],model=args.served_name,temperature=args.temperature,endpoint=f'http://127.0.0.1:{args.model_port}/v1')
 save(OUT/'policy_used.json',plan)
 from robustops.launch.resources import check_host_memory
 memory=check_host_memory(config.get('hardware',{}).get('min_host_memory_gib',0))
 devices=subprocess.check_output(['nvidia-smi','-L'],text=True)
 used=subprocess.check_output(['nvidia-smi','-i',str(args.gpu_id),'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
 if used:raise RuntimeError('GPU remains in use; refusing competing model launch')
 if 'GPU ' not in devices:raise RuntimeError('A visible GPU is required for local model evaluation')
 tasks={};modelpath=args.model_path.resolve();counter=TokenCounter(ROOT/plan['tokenizer'],enable_thinking=False)
 for r in rows:
  cfg=read(ROOT/f"recipes/enterpriseops/paired/{'oracle' if r['mode']=='oracle' else 'plus5'}/{r['domain']}.json");tasks[r['cell']]=next(t for t in load_tasks(ROOT/cfg['data_path'],expected_domain=r['domain'],expected_count=cfg['expected_task_count']) if t.task_id==r['task_id'])
 for r in rows:
  t=tasks[r['cell']]
  assert t.system_prompt==r['payload']['task']['system_policy'] and t.user_prompt==r['payload']['task']['user_request'] and list(t.verifiers)==r['payload']['verifiers'],'Frozen task definition changed'
 runtime=OUT/'runtime';runtime.mkdir(exist_ok=False);gate=asyncio.Semaphore(2)
 async def episode(r,condition):
  if paired_clean and ((condition=='clean' and str(r['cell']) in clean_sources) or (condition=='runtime_priority' and r['cell'] in reused)):return
  async with gate:
   if args.runtime_priority and task_caps is not None:task_caps.start(str(r['cell']))
   path=OUT/condition/f"cell_{r['cell']:02}";path.mkdir(parents=True,exist_ok=False);(path/'calls').mkdir()
   footprint=lambda x:counter.text(_tool_content(x))
   cell_schedule=dict(schedule,priority=frozen[str(r['cell'])]['priority']) if panel else schedule
   def factory(target):
    controller=RuntimePriorityFault(r['payload'],r['selection'],dose_table[str(r['cell'])] if scheduled else {k:budget for k in PRIORITY},SelectedFaultController,footprint,target,task_caps,str(r['cell']),cell_schedule,balance)
    if panel:
     # Preserve the scheduled scheduler's single reserved label without reranking.
     controller.live={controller.kinds[0]} if controller.kinds else set()
    return controller
   f=None if args.runtime_priority else SelectedFaultController(r['payload'],r['selection'],r['fault_kind'],r['budget'],r['binding'],footprint)
   m=Model(plan['base_model'],path/'calls',r['seed']+args.sampling_seed_offset)
   try:
    kw=dict(task=tasks[r['cell']],token_counter=counter,archive_path=ROOT/'enterpriseops_gym/gym_dbs.zip',mcp_url_overrides=urls,max_turns=plan['max_turns'],max_context_tokens=plan['max_context_tokens'],max_generation_tokens=plan['max_generation_tokens'],experiment_id=plan['experiment_id'],run_id=OUT.name,mode=r['mode'],model_name=plan['base_model']['model'],model_endpoint=plan['base_model']['endpoint'],seed=r['seed'],plan_preserving=True)
    if panel:kw['initial_state']=read(ROOT/frozen[str(r['cell'])]['initial_state'])
    if paired_clean and condition=='clean':
     kw['initial_state']={}
     f=RuntimePriorityFault(r['payload'],r['selection'],observation_budgets(r['selection']),SelectedFaultController,footprint,schedule=dict(schedule,observe_only=True))
     a=await run_task_attempt(model=m,tool_intervention=f,**kw)
     save(path/'initial_state.attempt.json',kw['initial_state']);save(path/'checkpoints.json',f.checkpoints)
     clean_sources[str(r['cell'])]=str(path);save(OUT/'clean_sources.json',clean_sources)
    elif paired_clean:
     spec=branch_plans[str(r['cell'])];donor_path=Path(clean_sources[str(r['cell'])]);donor=read(donor_path/'attempt.json')
     if spec['kind'] is None:
      a=deepcopy(donor)
      save(path/'branch_selection.json',dict(scored_branch='clean_no_opportunity',reason=spec['reason'],replay_attempted=False))
     else:
      f=RuntimePriorityFault(r['payload'],r['selection'],{spec['kind']:spec['budget']},SelectedFaultController,footprint,spec['target'],schedule=schedule)
      checkpoint=read(donor_path/'checkpoints.json')[spec['kind']]
      a=await run_clean_branch(donor=donor,checkpoint=checkpoint,target=spec['target'],controller=f,model=m,attempt_kwargs=kw)
      save(path/'branch_selection.json',dict(scored_branch='clean_checkpoint',target=spec['target'],budget=spec['budget'],replay_attempted=False))
    elif panel and condition=='clean':
     a=await run_task_attempt(model=m,tool_intervention=None,**kw)
     save(path/'initial_state.attempt.json',kw['initial_state'])
    elif fixed:
     spec=fixed_by_cell[str(r['cell'])]
     f=SelectedFaultController(r['payload'],r['selection'],spec['kind'],spec['budget'],'eligible_calls',footprint)
     a=await run_task_attempt(model=m,tool_intervention=f,**kw)
     hits=a['fault']['injected_calls']
     a['fault'].update(target_mechanism=spec['kind'],selected_mechanism=spec['kind'] if hits else None,no_hit=hits==0,base_injected_calls=spec['base_injected_calls'])
     save(path/'branch_selection.json',dict(scored_branch='base_fixed_kind',replay_attempted=False,target_mechanism=spec['kind'],no_hit=hits==0))
    elif repair:
     spec=repair[str(r['cell'])]
     assert spec['kind']=='unknown_commit' and spec['budget']==2
     kw['initial_state']=read(ROOT/spec['initial_state'])
     f=SelectedFaultController(r['payload'],r['selection'],spec['kind'],spec['budget'],'eligible_calls',footprint)
     a=await run_task_attempt(model=m,tool_intervention=f,**kw)
     a['fault']['selected_mechanism']=spec['kind']
     save(path/'branch_selection.json',dict(scored_branch='fixed_mechanism_cap2',replay_attempted=False,original_attempt=spec['source_attempt']))
    elif scheduled and not schedule['max_zero_hit_replays']:
     f=factory(None)
     a=await run_task_attempt(model=m,tool_intervention=f,**kw)
     save(path/'initial.attempt.json',a)
     save(path/'branch_selection.json',dict(scored_branch='initial',replay_attempted=False,target=None))
     if balance is not None:save(OUT/'balance.json',balance.report())
    elif args.runtime_priority:
     resume_episode=None
     if resume_root:
      prior_state=resume_root/'runtime_priority'/f"cell_{r['cell']:02}"/'initial_state.attempt.json'
      if prior_state.exists():kw['initial_state']=read(prior_state)
      checkpoint=prior_state.parent/'checkpoint.attempt.json'
      initial=prior_state.parent/'initial.attempt.json'
      if checkpoint.exists() and initial.exists():
       resume_episode=dict(checkpoint=read(checkpoint),initial=read(initial))
     outcome=await run_with_nohit_replay(controller_factory=factory,model=m,attempt_kwargs=kw,resume_episode=resume_episode,save_attempt=lambda branch,a:save(path/(branch+'.attempt.json'),a))
     a=outcome.pop('final');save(path/'branch_selection.json',outcome)
     if balance is not None:save(OUT/'balance.json',balance.report())
     elif task_caps is not None:save(OUT/'task_caps.json',task_caps.report())
    else:a=await run_task_attempt(model=m,tool_intervention=f,**kw)
    a['sampling']=dict(temperature=args.temperature,seed=r['seed']+args.sampling_seed_offset)
    save(path/'attempt.json',a);results.append(dict(cell=r['cell'],condition=condition,success=a['success'],valid=a['valid'],hits=a['fault'].get('injected_calls'),target_mechanism=a['fault'].get('target_mechanism'),no_hit=a['fault'].get('no_hit'),termination=(a.get('trajectory') or {}).get('termination')));print('DONE',r['cell'],condition,results[-1],flush=True)
    if not a['valid'] or not a['cleanup']['succeeded']:raise RuntimeError('Invalid or cleanup failure; stop batch')
   finally:
    if f is not None:save(path/'tool_audit.json',f.calls)
    await m.client.close()
 try:
  cmd=[sys.executable,'-m','vllm.entrypoints.openai.api_server','--model',str(modelpath),'--tokenizer',str(ROOT/plan['tokenizer']),'--served-model-name',args.served_name,'--host','127.0.0.1','--port',str(args.model_port),'--dtype','bfloat16','--max-model-len',str(plan['max_context_tokens']),'--enable-auto-tool-choice','--tool-call-parser','hermes']
  if panel:cmd.extend(['--max-num-seqs','2'])
  save(OUT/'environment.json',dict(devices=devices,host_memory_bytes=memory,command=cmd,gpu_id=args.gpu_id,model_port=args.model_port,mcp_port_offset=args.mcp_port_offset,temperature=args.temperature,sampling_seed_offset=args.sampling_seed_offset,global_concurrency=2,replicas=1,normal_source=None if args.runtime_priority else plan['rows'][0]['prior_clean_source']))
  log=(OUT/'gpu_server.log').open('w');gpu=subprocess.Popen(cmd,env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(args.gpu_id),PATH=str(Path(sys.executable).parent)+os.pathsep+os.environ.get('PATH','')),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  async with httpx.AsyncClient(trust_env=False) as client:
   for _ in range(600):
    if gpu.poll() is not None:raise RuntimeError('Owned model startup failed')
    try:
     if (await client.get(f'http://127.0.0.1:{args.model_port}/health',timeout=2)).status_code==200:break
    except httpx.HTTPError:pass
    await asyncio.sleep(1)
   else:raise RuntimeError('Model readiness timeout')
  print('MODEL READY',flush=True);procs,urls=await launch({g['mcp_server_name'] for t in tasks.values() for g in t.gym_servers},runtime,port_offset=args.mcp_port_offset)
  save(OUT/'owned_processes.json',dict(gpu=gpu.pid,mcp=[p.pid for p in procs]))
  if not paired_clean:rows=[r for r in rows if r['cell'] not in reused]
  for condition in plan['conditions']:
   if paired_clean and condition=='runtime_priority':
    donors={key:dict(attempt=read(Path(folder)/'attempt.json'),checkpoints=read(Path(folder)/'checkpoints.json')) for key,folder in clean_sources.items()}
    branch_plans,balance_report=select_clean_branches(rows,donors,schedule)
    if resume_root and (resume_root/'branch_plans.json').exists() and read(resume_root/'branch_plans.json')!=branch_plans:
     raise ValueError('Resumed clean branch assignments changed')
    dose_table={key:({spec['kind']:spec['budget']} if spec['kind'] else {}) for key,spec in branch_plans.items()}
    save(OUT/'branch_plans.json',branch_plans);save(OUT/'candidate_budgets.json',dose_table);save(OUT/'balance.json',balance_report)
    save(OUT/'clean_sources.json',clean_sources)
   async with asyncio.TaskGroup() as group:
    for r in rows:group.create_task(episode(r,condition))
   print('BATCH COMPLETE',len(results),flush=True)
  if paired_clean:save(OUT/'clean_completion.json',[dict(cell=int(key),source=folder) for key,folder in clean_sources.items()])
  save(OUT/'completion.json',[r for r in results if r['condition']!='clean'] if paired_clean else results)
  if fixed:
   save(OUT/'fixed_fault_summary.json',dict(assigned=len(results),hit=sum((r['hits'] or 0)>0 for r in results),no_hit=sum(r['no_hit'] is True for r in results),by_mechanism={kind:dict(assigned=sum(r['target_mechanism']==kind for r in results),hit=sum(r['target_mechanism']==kind and (r['hits'] or 0)>0 for r in results),no_hit=sum(r['target_mechanism']==kind and r['no_hit'] is True for r in results)) for kind in sorted({r['target_mechanism'] for r in results})}))
 finally:
  stop_processes(procs)
  if gpu is not None:stop_processes([gpu])
  if log:log.close()
  save(OUT/'resource_cleanup.json',dict(gpu_stopped=gpu is None or gpu.poll() is not None,mcp_stopped=all(p.poll() is not None for p in procs),completed=len(results),reused=len(reused),new_completed=len(results)-len(reused)))
def main():
 parser=argparse.ArgumentParser(description='Evaluate once per task-mode using a frozen Val/Test fault policy')
 parser.add_argument('--gpu-id',type=int,default=0)
 parser.add_argument('--model-port',type=int,default=8010)
 parser.add_argument('--mcp-port-offset',type=int,default=1000)
 parser.add_argument('--temperature',type=float,choices=(0.0,0.6),default=0.0)
 parser.add_argument('--sampling-seed-offset',type=int,default=0)
 parser.add_argument('--policy',type=Path,default=ROOT/'artifacts/test_fault_policy/policy.json')
 parser.add_argument('--output',type=Path,required=True)
 parser.add_argument('--model-path',type=Path,default=ROOT/'models/base/Arctic-MidTool-RL-4B')
 parser.add_argument('--served-name',default='Arctic-MidTool-RL-4B')
 parser.add_argument('--inference-config',type=Path,help='Resolved inference recipe provenance')
 parser.add_argument('--resume-from',type=Path,help='Reuse valid completed results and cap state from an interrupted runtime batch')
 parser.add_argument('--runtime-priority',action='store_true',help='Reuse LLM candidates; runtime qualification with one no-hit prefix replay')
 parser.add_argument('--cells',type=int,nargs='+',help='Select a task subset for runtime evaluation')
 parser.add_argument('--schedule',type=Path,default=ROOT/'recipes/enterpriseops/faults/test.json')
 parser.add_argument('--fault-budget',type=int,choices=(1,2,3,4),default=None,help='Per-mechanism maximum in runtime mode; actual eligible calls may be fewer')
 asyncio.run(run(parser.parse_args()))

if __name__=="__main__":main()
