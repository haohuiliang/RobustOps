"""Run selected fixed demonstrations and render evidence from actual attempts."""
import argparse
import asyncio
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(path.read_text())


def write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + '\n')


def outcome(attempt, condition):
    if not attempt:
        return 'NOT RUN'
    if not attempt.get('valid') or not attempt.get('cleanup', {}).get('succeeded'):
        return 'INVALID'
    if condition == 'fault' and not attempt.get('fault', {}).get('injected_calls', 0):
        return 'NO FAULT HIT'
    return 'PASS' if attempt.get('success') else 'FAIL'


def summarize(attempt, condition):
    status = outcome(attempt, condition)
    if attempt is None:
        return {'status': status}
    trajectory = attempt.get('trajectory', {})
    turns = trajectory.get('turns', [])
    last = turns[-1] if turns else {}
    final = (last.get('assistant_content', '')
             if trajectory.get('termination') == 'model_stop' and not last.get('tool_calls') else '')
    return dict(status=status, task_success=attempt.get('success'),
                checks_passed=sum(bool(v['passed']) for v in attempt.get('verifiers', [])),
                checks_total=len(attempt.get('verifiers', [])),
                tool_calls=trajectory.get('tool_call_count', 0),
                injected_calls=attempt.get('fault', {}).get('injected_calls', 0),
                termination=trajectory.get('termination'), final_response=final,
                verifiers=attempt.get('verifiers', []))


def compact(value, limit=180):
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = ' '.join(text.split())
    return text if len(text) <= limit else text[:limit - 3] + '...'


def wrapped(text, indent='  '):
    import textwrap
    return '\n'.join(textwrap.fill(line, width=100, initial_indent=indent,
                                  subsequent_indent=indent, break_long_words=False,
                                  break_on_hyphens=False) if line else ''
                     for line in str(text).splitlines())


def render_attempt(attempt, condition, source):
    s = summarize(attempt, condition)
    if attempt is None:
        return '  NOT RUN\n'
    lines = [f'  Result: {s["status"]} | Checks: {s["checks_passed"]}/{s["checks_total"]}'
             f' | Tool calls: {s["tool_calls"]} | Fault hits: {s["injected_calls"]}',
             f'  End: {s["termination"]}', '', '  State checks:']
    for v in s['verifiers']:
        lines.append(f'    [{"PASS" if v["passed"] else "FAIL"}] {v["name"]}')
        if not v['passed']:
            lines.append(f'           Expected: {compact(v.get("expected"))}; actual: {compact(v.get("actual"))}')
    lines += ['', '  Tool calls (arguments and responses abbreviated; full records below):']
    faults = {e.get('call_index'): e for e in attempt.get('fault', {}).get('events', []) if e.get('injected')}
    for index, event in enumerate(attempt.get('trajectory', {}).get('tool_events', []), 1):
        injected = faults.get(index)
        tag = ' [FAULT INJECTED]' if injected else ''
        lines.append(f'    {index:02}. {event["name"]}{tag}')
        lines.append(wrapped('Args: ' + compact(event.get('arguments')), '        '))
        result = event.get('result', {})
        content = result.get('result', {}).get('content', [])
        visible = '\n'.join(b.get('text', '') for b in content if b.get('type') == 'text')
        lines.append(wrapped('Reply: ' + compact(visible or result), '        '))
    lines += ['', '  Final response:', wrapped(s['final_response'] or '(No final response recorded.)'),
              '', f'  Full record: {source}']
    if attempt.get('error'):
        lines += [wrapped('Runtime error: ' + compact(attempt['error']))]
    return '\n'.join(lines) + '\n'


def report(output):
    plans = {m: read(output / m / 'plan.json') for m in ('base', 'rl') if (output / m / 'plan.json').exists()}
    if not plans:
        raise ValueError('No saved demo plans in the output directory')
    first = next(iter(plans.values()))
    for plan in plans.values():
        for key in ('cases', 'limits', 'tokenizer', 'mode', 'guidance', 'controller'):
            if plan[key] != first[key]:
                raise ValueError(f'Base/RL plans differ in {key}; use a fresh output directory')
    rows, sections = [], []
    detail_dir = output / 'details'
    detail_dir.mkdir(exist_ok=True)
    for number, case in enumerate(first['cases'], 1):
        condition = case['conditions'][0]
        row = dict(case_id=case['case_id'], title=case.get('title', case['case_id']),
                   condition=condition, task_id=case['task_id'], fault=case.get('fault'))
        fault = case.get('fault', {})
        section = ['=' * 100, f'{number}. {row["title"]} [{condition}]',
                   f'   Case: {case["case_id"]}', '', 'Task:', wrapped(case['task']['user_request']), '',
                   f'Fault: {fault.get("kind", "none")} (maximum hits: {fault.get("budget", 0)})']
        brief = [f'{number}. {row["title"]} [{condition}]', '-' * 94,
                 wrapped(compact(case['task']['user_request'], 240)),
                 f'  Fault: {fault.get("kind", "none")}']
        for model in ('base', 'rl'):
            rel = Path(model) / case['case_id'] / condition / 'attempt.json'
            path = output / rel
            attempt = read(path) if path.exists() else None
            if attempt and (attempt['task_id'] != case['task_id'] or attempt['mode'] != case['mode']):
                raise ValueError(f'Attempt does not match plan: {path}')
            row[model] = summarize(attempt, condition)
            section += ['', model.upper(), '-' * 100, render_attempt(attempt, condition, rel.as_posix())]
            s = row[model]
            brief += ['', f'  {model.upper()}: {s["status"]}']
            if attempt:
                brief += [f'    Checks: {s["checks_passed"]}/{s["checks_total"]}; end: {s["termination"]}']
                failed = [v['name'] for v in s['verifiers'] if not v['passed']]
                if failed:
                    brief += [wrapped('Failed checks: ' + '; '.join(failed), '    ')]
                hits = [f'call {e.get("call_index")}: {e.get("tool")}'
                        for e in attempt.get('fault', {}).get('events', []) if e.get('injected')]
                if condition == 'fault':
                    brief += [wrapped('Fault positions: ' + ('; '.join(hits) or 'none'), '    ')]
                brief += ['    Final response:', wrapped(compact(s['final_response'], 500)
                          if s['final_response'] else '(No complete final response recorded.)', '      ')]
        rows.append(row)
        detail_path = Path('details') / (case['case_id'] + '.txt')
        (output / detail_path).write_text('\n'.join(section) + '\n', encoding='utf-8')
        brief += ['', f'  Full task, tool calls and responses: {detail_path}', '']
        sections.append('\n'.join(brief))
    data = dict(note='Selected demonstrations. Outcomes below are from this run.',
                models={m: p['backend'] for m, p in plans.items()}, results=rows)
    write_json(output / 'comparison.json', data)
    lines = ['BASE / ROBUSTOPS COMPARISON', '',
             f'{"Case":<22} {"Condition":<10} {"Base":<14} {"RL":<14} {"Calls B/R":<12} Fault hits B/R',
             '-' * 94]
    for r in rows:
        b, a = r['base'], r['rl']
        calls = f'{b.get("tool_calls", "-")} / {a.get("tool_calls", "-")}'
        hits = f'{b.get("injected_calls", "-")} / {a.get("injected_calls", "-")}'
        lines.append(f'{r["case_id"]:<22} {r["condition"]:<10} {b["status"]:<14} {a["status"]:<14} {calls:<12} {hits}')
    summary = '\n'.join(lines)
    lines += ['', 'PASS = all state checks passed in a valid run; fault examples also require a fault hit.',
              'FAIL = task incomplete; INVALID = execution/cleanup issue; NO FAULT HIT = injection did not occur.',
              'NOT RUN = no recorded attempt. The final response is not used to decide task success.',
              'Responses below are excerpts. Complete responses and tool traces are in details/<case>.txt.',
              '', *sections]
    path = output / 'comparison.txt'
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(summary, flush=True)
    print(f'\nResults: {path}', flush=True)
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=('base', 'rl', 'both'), default='both')
    parser.add_argument('--case', action='append', dest='case_ids', help='case ID; repeat to select several (default: all five)')
    parser.add_argument('--output', type=Path, default=Path('runs/demo'))
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--report-only', action='store_true')
    args = parser.parse_args(argv)
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if args.report_only:
        report(output)
        return 0
    from robustops.evaluation import cases
    models = ('base', 'rl') if args.model == 'both' else (args.model,)
    prepared = []
    for model in models:
        recipe = read(ROOT / f'recipes/inference/demo_{model}.json')
        recipe['output'] = str(output / model)
        recipe['case_ids'] = args.case_ids
        plan = cases.build_plan(ROOT, recipe)
        # Fail before allocating a GPU if an output belongs to a different run.
        saved = output / model / 'plan.json'
        if saved.exists() and read(saved) != plan:
            raise ValueError(f'Plan changed: choose a fresh --output ({saved})')
        other = output / ('rl' if model == 'base' else 'base') / 'plan.json'
        if other.exists():
            old = read(other)
            for key in ('cases', 'limits', 'tokenizer', 'mode', 'guidance', 'controller'):
                if old[key] != plan[key]:
                    raise ValueError(f'Other model has different {key}; choose a fresh --output')
        prepared.append((model, recipe, plan))
    for model, recipe, plan in prepared:
        print(f'{model}: ' + ', '.join(c['case_id'] for c in plan['cases']), flush=True)
        if not args.dry_run:
            try:
                asyncio.run(cases.run(ROOT, recipe, plan, args.gpus))
            finally:
                if (output / model / 'plan.json').exists():
                    report(output)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
