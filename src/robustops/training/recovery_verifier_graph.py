"""Read the published verifier graphs for private training progress scoring."""
from functools import lru_cache
import json
from pathlib import Path

from robustops.data.verifier_graph import validate_graph


@lru_cache(maxsize=4)
def load_graphs(directory):
    directory = Path(directory)
    summary = json.loads((directory / 'summary.json').read_text())
    if summary.get('status') != 'complete' or not summary.get('all_full_success_equivalent'):
        raise ValueError('verifier graph dataset is not complete and boundary-verified')
    graphs = {}
    for filename, partition in (('train.jsonl', 'train'), ('val.jsonl', 'validation')):
        rows = [json.loads(line) for line in (directory / filename).read_text().splitlines() if line.strip()]
        if len(rows) != summary['coverage']['partition_mode_pairs'][partition]:
            raise ValueError(f'verifier graph coverage mismatch: {partition}')
        for row in rows:
            key = row['task_id'], row['mode']
            if key in graphs or row['partition'] != partition:
                raise ValueError(f'duplicate or wrong-partition verifier graph: {key}')
            proposal = dict(task_id=row['task_id'], final_success=row['final_success'],
                nodes=[dict(index=v['index'], depends_on=v['depends_on']) for v in row['verifiers']])
            deps = validate_graph(proposal, row)
            graphs[key] = dict(row=row, dependencies=deps)
    return graphs


def task_graph(directory, task, mode):
    """Require the exact current-mode author checks before any rollout starts."""
    key = task.task_id, mode
    graph = load_graphs(str(Path(directory).resolve())).get(key)
    if graph is None:
        raise ValueError(f'missing verifier graph: {key}')
    row = graph['row']
    original = [{k: value for k, value in v.items() if k not in ('index', 'depends_on')}
                for v in row['verifiers']]
    if row['domain'] != task.domain or original != list(task.verifiers):
        raise ValueError(f'verifier graph differs from current task contract/order: {key}')
    return graph
