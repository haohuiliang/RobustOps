"""Prepare, select and freeze held-out fault candidates without solver rollouts."""
import argparse
import asyncio
from collections import Counter
import json
from pathlib import Path

from robustops.data import fault_selector_batch as batch
from robustops.data.fault_candidate_prompt import PROMPT
from robustops.data.fault_selection import parse_message
from robustops.evaluation.fault_schedule import GROUPS, load_schedule

ROOT = Path(__file__).resolve().parents[3]

def read(path):
    return json.loads(Path(path).read_text())

def fixed_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and read(path) != value:
        raise ValueError(f'frozen input changed: {path}')
    batch.write_json(path, value)

def validate_selection(raw, payload):
    names = {t['function']['name'] for t in payload['tools']}
    if (set(raw) != {'tools'} or len(raw['tools']) != len(names)
            or {t['tool'] for t in raw['tools']} != names):
        raise ValueError('selector must cover each exposed tool exactly once')
    for t in raw['tools']:
        if (not isinstance(t['faults'], list) or len(t['faults']) != len(set(t['faults']))
                or not set(t['faults']) <= GROUPS.keys()
                or not isinstance(t.get('reason'), str) or not t['reason']
                or not isinstance(t.get('recovery_tools'), list)
                or not isinstance(t.get('replay_basis'), str)
                or not set(t['recovery_tools']) <= names):
            raise ValueError('invalid selector structure or unexposed recovery reference')

async def select(payload, directory, client, *, observations=None, **kwargs):
    observations = observations or {}
    shaped = dict(payload, observed_response_shapes={
        t['function']['name']: observations[key]['shape']
        for t in payload['tools']
        if (key := payload['tool_servers'][t['function']['name']]+'::'+t['function']['name']) in observations})
    messages = [dict(role='system', content=PROMPT), dict(role='user', content=json.dumps(shaped, ensure_ascii=False))]
    batch.write_json(directory/'request.json', messages)
    message, usage = await client.complete(messages, [])
    batch.write_json(directory/'response.json', dict(message=message, usage=usage))
    raw = parse_message(message)
    batch.write_json(directory/'selection.json', raw)
    validate_selection(raw, payload)
    batch.write_json(directory/'proposal.json', dict(task_id=payload['task_id'], mode=payload['mode'],
        tools=[dict(tool=t['tool'], faults=t['faults'], ood_faults=[]) for t in raw['tools']]))
    batch.write_json(directory/'validation.json', dict(structural_valid=True, rejected_paths=[],
        semantic_accepted=False, manual_label_changes=0, capability_catalog_used=False))

def prepare(config, output):
    output.mkdir(parents=True, exist_ok=True)
    fixed_write(output/'config.json', config)
    fixed_write(output/'prompts.json', dict(system=PROMPT))
    fixed_write(output/'response_shape_observations.json', read(config['response_shapes']))
    rows = batch.prepare(config, output, include_runtime=True)
    if len(rows) != config['expected_task_modes']:
        raise ValueError('unexpected held-out task-mode count')
    return rows

def reuse(config, output, rows):
    observations = read(config['response_shapes'])
    sources = {}
    for source in config.get('reuse_policies', []):
        for r in read(source)['rows']:
            sources[r['task_id'], r['mode']] = r
    reused = 0
    for r in rows:
        if batch.successful_attempt(output/f"cell_{r['cell']:04}"):
            continue
        old = sources.get((r['task_id'], r['mode']))
        if old is None:
            continue
        selection_path = Path(old['selection_source'])
        request_path = selection_path.parent/'request.json'
        if not request_path.exists():
            continue
        request = read(request_path); payload = read(r['input'])
        shaped = dict(payload, observed_response_shapes={t['function']['name']: observations[key]['shape']
            for t in payload['tools'] if (key := payload['tool_servers'][t['function']['name']]+'::'+t['function']['name']) in observations})
        if request != [dict(role='system',content=PROMPT),dict(role='user',content=json.dumps(shaped,ensure_ascii=False))]:
            # JSON whitespace may differ, but the prompt and structured input must match.
            if (len(request)!=2 or request[0]!=dict(role='system',content=PROMPT)
                    or request[1].get('role')!='user' or json.loads(request[1]['content'])!=shaped):
                continue
        selection = read(selection_path)
        validate_selection(selection,payload)
        attempt = output/f"cell_{r['cell']:04}"/'attempt_0001'
        attempt.mkdir(parents=True,exist_ok=False)
        fixed_write(attempt/'selection.json',selection)
        fixed_write(attempt/'request.json',request)
        fixed_write(attempt/'reused_from.json',dict(selection=str(selection_path)))
        fixed_write(attempt/'proposal.json',dict(task_id=r['task_id'],mode=r['mode'],tools=[dict(tool=t['tool'],faults=t['faults'],ood_faults=[]) for t in selection['tools']]))
        fixed_write(attempt/'validation.json',dict(structural_valid=True,rejected_paths=[],semantic_accepted=False,manual_label_changes=0,capability_catalog_used=False))
        reused += 1
    return reused

def freeze(config, output, rows):
    selected = []
    missing = []
    for r in rows:
        attempt = batch.successful_attempt(output/f"cell_{r['cell']:04}")
        if attempt is None:
            missing.append(r['cell']); continue
        payload = read(r['input']); selection = read(attempt/'selection.json')
        validate_selection(selection, payload)
        selected.append(dict(cell=r['cell'], task_id=r['task_id'], domain=r['domain'], mode=r['mode'],
            partition=config['partition'], payload=payload, selection=selection,
            selection_source=str(attempt/'selection.json'), seed=20260830))
    if missing:
        raise ValueError(f'selector incomplete; rerun to fill failed/missing cells: {missing}')
    # Execution settings match the Test protocol; no historical trajectory supplies injection points.
    split = 'val' if config['partition'] == 'validation' else config['partition']
    policy = dict(policy_id=split+'_fault_policy', frozen=True, rows=selected,
        base_model=dict(provider='local_vllm', endpoint='http://127.0.0.1:8010/v1',
            model='Arctic-MidTool-RL-4B', temperature=0), tokenizer=config['tokenizer'],
        temperature=0, max_turns=50, max_context_tokens=32768, max_generation_tokens=2048,
        replicas=1, concurrency=2, frozen_sources=[])
    # Selection freezes labels, not fault plans. Caps are drawn only after the
    # Base run has collected clean opportunities and selected a mechanism.
    load_schedule(config['schedule'])
    fixed_write(config['policy_output'], policy)
    return dict(task_modes=len(selected), candidate_coverage=dict(Counter(
        kind for r in selected for kind in {k for t in r['selection']['tools'] for k in t['faults']})))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--phase', choices=['prepare','select','freeze','all'], default='all')
    args = parser.parse_args()
    config = read(args.config); output = Path(config['output'])
    rows = prepare(config, output)
    print(json.dumps(dict(reused=reuse(config,output,rows))),flush=True)
    if args.phase in ('select', 'all'):
        observations = read(output/'response_shape_observations.json')
        async def configured_select(*args, **kwargs):
            return await select(*args, observations=observations, **kwargs)
        asyncio.run(batch.run(config, output, rows, selector=configured_select))
    if args.phase in ('freeze', 'all'):
        print(json.dumps(freeze(config, output, rows), ensure_ascii=False))
    else:
        print(json.dumps(dict(phase=args.phase, task_modes=len(rows)), ensure_ascii=False))

if __name__ == '__main__':
    main()
