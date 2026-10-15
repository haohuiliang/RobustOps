"""Validate task identities and fault settings in a frozen evaluation panel."""
import json
from pathlib import Path

def read(path):
    return json.loads(Path(path).read_text())

def validate_panel(root, recipe):
    from robustops.evaluation.fault_schedule import validate_schedule, load_schedule
    from robustops.evaluation.fault_scheduling_controller import PRIORITY

    policy = read(root / recipe['selector_policy'])
    panel = read(root / recipe['frozen_panel'])
    schedule = load_schedule(root / recipe['schedule'], root=root)
    validate_schedule(schedule)
    budgets = read(root / schedule['candidate_budgets'])
    assert len(policy['rows']) == recipe['expected_task_modes'] == 127
    assert recipe['partition'] == panel['partition'] == 'validation'
    assert recipe['temperature'] == policy['temperature'] == 0
    indexed = {r['cell']: r for r in policy['rows']}
    assert len(panel['rows']) == len({r['cell'] for r in panel['rows']}) == recipe.get('expected_panel_task_modes', 32)
    for row in panel['rows']:
        original = indexed[row['cell']]
        assert original['partition'] == 'validation'
        assert all(row[k] == original[k] for k in ('task_id', 'mode', 'domain', 'seed'))
        assert row['candidate_budgets'] == budgets[str(row['cell'])]
        assert len(row['priority']) == len(PRIORITY) and set(row['priority']) == set(PRIORITY)
        assert row['priority'] == read(root / row['source_attempt'])['fault']['priority']
        assert read(root / row['initial_state'])
    assert (root / policy['tokenizer'] / 'tokenizer_config.json').is_file()
    return panel
