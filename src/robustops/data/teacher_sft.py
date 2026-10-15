"""Convert guided teacher inference artifacts into masked SFT trajectories and Parquet.

Read attempt.json and the recorded teacher requests, remove guidance without
changing executed history, apply existing action/narration masks, and export.
"""
from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

from robustops.data.trajectory_messages import build_messages
from robustops.evaluation.clean_baseline import _argument_schema_diagnostics
from robustops.training.selective_supervision import validate_supervision_policy

ROOT = Path(__file__).resolve().parents[3]


def student_token_proof(record, tokenizer, limit=32768, *, include_spans=False):
    from robustops.training.selective_supervision import _build_selective_token_spans
    if not 0 < limit <= 32768:
        raise ValueError('student sequence limit must not exceed 32768')
    ids, spans = _build_selective_token_spans(renderer=tokenizer, tokenizer=tokenizer,
        messages=record['messages'], tools=record['tools'], policies=record['assistant_supervision'],
        template_kwargs={'enable_thinking': False})
    proof = dict(sequence_tokens=len(ids), positive_tokens=sum(end-start for _,start,end in spans),
                within_limit=len(ids)<=limit, max_sequence_tokens=limit,
                teacher_guidance_removed=True, original_prompt_restored=True, visible_context_matches=True)
    if include_spans:
        proof['supervised_spans'] = [dict(message_index=i, start=start, end=end) for i, start, end in spans]
    return proof


def student_export_allowed(attempt):
    trajectory = attempt.get('trajectory') or {}
    if trajectory.get('termination') == 'model_stop':
        return True
    return bool(trajectory.get('termination') == 'context_limit_pre_request'
                and attempt.get('status') == 'success' and attempt.get('valid')
                and attempt.get('cleanup', {}).get('succeeded')
                and trajectory.get('protocol_valid')
                and attempt.get('verifiers')
                and all(v['passed'] for v in attempt['verifiers']))


def student_record(original_task, attempt, tools, teacher_messages, record_id, guidance=None):
    """Rebuild from executed calls/visible results, then prove context-only removal."""
    from robustops.data.recovery_sampling import TEACHER_GUIDANCE, teacher_guidance_fragments
    if guidance is None:
        guidance = TEACHER_GUIDANCE
    trajectory = attempt.get('trajectory')
    if not student_export_allowed(attempt):
        raise ValueError('export requires model stop or verified-success context stop')
    messages, indices = build_messages(original_task, trajectory,
        allow_tool_ending=trajectory['termination'] == 'context_limit_pre_request')
    # Last request contains all prior executed assistant/tool turns. Normalize args only.
    def canonical(msgs):
        result = deepcopy(msgs)
        for m in result:
            for c in m.get('tool_calls', []):
                if isinstance(c['function']['arguments'], str):
                    c['function']['arguments'] = json.loads(c['function']['arguments'])
        return result
    # Last API request precedes the final executed turn, including its tool receipts.
    suffix = 1 + len(trajectory['turns'][-1].get('tool_calls') or [])
    assert canonical(messages[:-suffix])[1:] == canonical(teacher_messages)[1:], 'visible execution/context drift'
    expected = original_task.system_prompt + ('\n\n' + guidance if guidance else '')
    assert teacher_messages[0] == {'role':'system','content':expected}, 'teacher prompt mismatch'
    assert messages[0] == {'role':'system','content':original_task.system_prompt}
    encoded = json.dumps(messages, ensure_ascii=False)
    if any(json.dumps(block, ensure_ascii=False)[1:-1] in encoded
           for block in teacher_guidance_fragments(guidance)):
        raise ValueError('teacher guidance leaked into student messages')
    return dict(record_id=record_id,task_id=original_task.task_id,domain=original_task.domain,
                mode=attempt['mode'],partition='train',messages=messages,tools=deepcopy(tools),
                assistant_supervision=[dict(message_index=i,target='full') for i in indices],
                admission='pending_review_not_training_data',
                trajectory_termination=trajectory['termination'],
                final_answer_present=not bool(trajectory['turns'][-1].get('tool_calls')))


def filter_record(record, attempt, annotations=()):
    """Match recorded calls by ID; preserve schema-valid errors and recovery turns."""
    result = deepcopy(record)
    policies = validate_supervision_policy(record['messages'], record['assistant_supervision'])
    schemas = {t['function']['name']: t['function']['parameters'] for t in record['tools']}
    events = {e['tool_call_id']: e for e in attempt['trajectory']['tool_events']}
    changes = []
    for index, current in policies.items():
        message = record['messages'][index]
        calls = message.get('tool_calls') or []
        # Current dataset emits one executed call per assistant turn. Do not
        # accidentally mask a valid sibling in a future parallel-call dataset.
        if len(calls) > 1:
            raise ValueError('Parallel assistant calls need component-level supervision')
        for call in calls:
            function = call['function']
            args = function['arguments']
            if isinstance(args, str):
                args = json.loads(args)
            event = events[call['id']]
            if event['name'] != function['name'] or event['arguments'] != args:
                raise ValueError(f"Call/source mismatch: {record['record_id']} {index}")
            violations = (_argument_schema_diagnostics(
                {'inputSchema': schemas[function['name']]}, args)['schema_violations']
                if function['name'] in schemas else
                [{'path': '', 'kind': 'unexposed_tool'}])
            if violations and event['result']['is_error'] and current != 'masked':
                policies[index] = 'masked'
                changes.append(dict(record_id=record['record_id'], message_index=index,
                    previous_target=current, target='masked', rule='rejected_exposed_schema_violation',
                    tool_call_id=call['id'], violations=violations,
                    reason='Rejected call uses an unexposed tool or violates its already exposed argument schema.'))
    for annotation in annotations:
        index = annotation['message_index']
        if record['messages'][index] != annotation['assistant_message']:
            raise ValueError(f"Stale reviewed annotation: {record['record_id']} {index}")
        if annotation['recommended_target'] != 'tool_call_only':
            raise ValueError('Only reviewed narration masks are accepted here')
        if policies[index] == 'masked':
            continue
        previous = policies[index]
        policies[index] = 'tool_call_only'
        if previous != policies[index]:
            changes.append(dict(record_id=record['record_id'], message_index=index,
                previous_target=previous, target=policies[index], rule='reviewed_narration',
                reason=annotation['reason']))
    result['assistant_supervision'] = [dict(message_index=i, target=t) for i, t in policies.items()]
    validate_supervision_policy(result['messages'], result['assistant_supervision'])
    assert result['messages'] == record['messages'] and result['tools'] == record['tools']
    return result, changes



def read(path):
    return json.loads(Path(path).read_text())


def process_attempt(task, attempt, request, record_id, tokenizer, *, partition,
                    guidance, annotations=()):
    """Remove teacher guidance and mask identified bad decisions in one pass."""
    if not (attempt['success'] and attempt['valid'] and attempt['cleanup']['succeeded']
            and student_export_allowed(attempt)):
        return None
    record = student_record(task, attempt, request['tools'], request['messages'],
                            record_id, guidance=guidance)
    record['partition'] = partition
    record, changes = filter_record(record, attempt, annotations)
    proof = student_token_proof(record, tokenizer, include_spans=True)
    if not proof['within_limit']:
        return None
    record['admission'] = 'accepted_verifier_context_and_schema_gates'
    return record, proof, changes


def run(root, config, *, tokenizer=None):
    """Process one guided inference run or both rounds of a teacher collection."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    from robustops.data.recovery_sampling import teacher_guidance

    root = Path(root)
    partition = config.get('partition', 'train')
    if partition not in ('train', 'validation'):
        raise ValueError('SFT processing partition must be train or validation')
    manifest = {r['task_id']: r for r in map(json.loads, (root/config['manifest']).read_text().splitlines())}
    source = root/config['source']
    folders = [source] if (source/'config.json').exists() else [
        source/name for name in ['temperature_0', 'temperature_0.6'] if (source/name).exists()]
    if not folders:
        raise ValueError(f'No teacher inference runs found: {source}')
    if tokenizer is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(root/config['tokenizer'], local_files_only=True)
    annotations = []
    if config.get('reviewed_annotations'):
        annotations = read(root/config['reviewed_annotations'])['recommendations']
    template = pq.read_table(root/config['template_parquet'])
    template_row = template.to_pylist()[0]
    records, provenance, masks, changes, table_rows = [], [], [], [], []
    seen, record_ids, used_annotations = set(), set(), set()
    for folder in folders:
        spec = read(folder/'config.json')
        tasks = {r['cell']: r for r in read(root/spec['tasks'])}
        guidance = teacher_guidance(spec.get('guidance_variant', 'consistent_narration'))
        for path in sorted(folder.glob('*/cell_*/attempt.json')):
            condition = path.parent.parent.name
            if condition not in ('clean', 'fault'):
                continue
            cell = tasks[int(path.parent.name.removeprefix('cell_'))]
            if cell['task_id'] not in manifest:
                continue
            meta = manifest[cell['task_id']]
            if (meta['partition'] != partition or meta.get('execution_blocked_reason')
                    or cell['mode'] not in meta['runnable_modes'] or cell['domain'] != meta['domain']):
                raise ValueError('Teacher source does not match the selected task manifest/partition')
            key = cell['task_id'], cell['mode'], condition
            if key in seen:
                continue
            attempt = read(path)
            if (attempt['task_id'], attempt['mode']) != key[:2]:
                raise ValueError('Teacher attempt identity differs from its configured task')
            if not (attempt['success'] and attempt['valid'] and attempt['cleanup']['succeeded']
                    and student_export_allowed(attempt)):
                continue
            if condition == 'fault' and not attempt['fault']['injected_calls']:
                continue
            allowed_faults = set(spec['train_faults'])
            if ((partition == 'train' and 'ignored_filter' in allowed_faults)
                    or any(e.get('selected_fault') not in allowed_faults for e in attempt['fault'].get('events', []))):
                raise ValueError('Teacher source fault exposure differs from its configured scope')
            request_paths = sorted((path.parent/'calls').glob('*.request.json'))
            if not request_paths:
                raise ValueError(f'Missing recorded teacher requests: {path.parent}')
            record_id = f"{folder.name}-{condition}-{cell['cell']:02}"
            selected = [a for a in annotations if a['record_id'] == record_id]
            task = SimpleNamespace(task_id=cell['task_id'], domain=cell['domain'],
                system_prompt=cell['payload']['task']['system_policy'],
                user_prompt=cell['payload']['task']['user_request'])
            result = process_attempt(task, attempt, read(request_paths[-1]), record_id, tokenizer,
                partition=partition, guidance=guidance, annotations=selected)
            if result is None:
                continue
            record, proof, edits = result
            if record_id in record_ids:
                raise ValueError(f'Duplicate processed record: {record_id}')
            seen.add(key); record_ids.add(record_id)
            used_annotations.update((a['record_id'], a['message_index']) for a in selected)
            records.append(record); changes.extend(edits)
            masks.append(dict(record_id=record_id, tokenizer=config['tokenizer'],
                              assistant_supervision=record['assistant_supervision'], **proof))
            provenance.append(dict(record_id=record_id, source=str(path.parent), partition=partition,
                                   condition=condition, temperature=spec['temperature']))
            item = dict(template_row)
            item.update({k: record[k] for k in ['record_id', 'task_id', 'domain', 'mode']},
                source_layer=folder.name, source_model=spec['model'], sequence_tokens=proof['sequence_tokens'],
                positive_decision_events=len(proof['supervised_spans']))
            item.update({key+'_json': json.dumps(record[key], ensure_ascii=False)
                         for key in ['messages', 'tools', 'assistant_supervision']})
            table_rows.append(item)
    if used_annotations != {(a['record_id'], a['message_index']) for a in annotations}:
        raise ValueError('Some reviewed annotations did not match the selected teacher records')
    if not records:
        raise ValueError('No teacher trajectories passed export')
    validation = None
    if partition == 'train' and config.get('validation_data'):
        validation = root/config['validation_data']
        validation_records = [json.loads(line) for line in (validation/'records.jsonl').read_text().splitlines()]
        if {r['task_id'] for r in records} & {r['task_id'] for r in validation_records}:
            raise ValueError('Teacher Train/validation overlap')
    output = root/config['output']
    output.mkdir(parents=True, exist_ok=False)
    destination = output/partition
    destination.mkdir()
    for name, values in [('records.jsonl', records), ('provenance.jsonl', provenance),
                         ('masks.jsonl', masks), ('mask_changes.jsonl', changes)]:
        (destination/name).write_text(''.join(json.dumps(row, ensure_ascii=False)+'\n' for row in values))
    pq.write_table(pa.Table.from_pylist(table_rows, schema=template.schema), destination/f'{partition}.parquet')
    summary = dict(partition=partition, records=len(records), output=str(output),
                   masked_decisions=sum(p['target']=='masked' for r in records for p in r['assistant_supervision']),
                   mask_changes_by_rule=dict(Counter(c['rule'] for c in changes)),
                   **{partition+'_records': len(records)})
    if validation is not None:
        shutil.copytree(validation, output/'validation')
        summary['validation_records'] = len(validation_records)
    (output/'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2)+'\n')
    (output/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    print(json.dumps(run(ROOT, read(args.config)), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
