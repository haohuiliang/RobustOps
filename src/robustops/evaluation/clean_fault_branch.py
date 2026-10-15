"""Branch from a completed clean path or continue after its first actual fault."""
from copy import deepcopy

from robustops.evaluation.clean_baseline import run_task_attempt
from robustops.evaluation.fault_checkpoint import CheckpointBranchModel, prefix_messages


def branch_inputs(donor, checkpoint, target):
    trajectory = donor['trajectory']
    exposed = [e for e in trajectory['tool_events'] if not e.get('agent_protocol_error')]
    event = exposed[target['call_index'] - 1]
    if (checkpoint['call_index'] != target['call_index']
            or (event['name'], event['arguments']) != (target['tool'], target['arguments'])):
        raise ValueError('Clean checkpoint does not match its qualified tool call')
    prefix = {key: [x for x in trajectory[key] if x['turn'] < event['turn']]
              for key in ('turns', 'tool_events')}
    turn = next(t for t in trajectory['turns'] if t['turn'] == event['turn'])
    return prefix, turn


def first_fault_state(controller, attempt):
    snapshot = controller.first_fault_checkpoint
    if snapshot is None:
        return None
    snapshot = deepcopy(snapshot)
    snapshot['prefix'] = {key: [x for x in attempt['trajectory'][key]
                              if x['turn'] <= snapshot['after_turn']]
                          for key in ('turns', 'tool_events')}
    return snapshot


async def run_clean_branch(*, donor, checkpoint, target, controller, model,
                           attempt_kwargs, run_attempt=None):
    if not donor['valid'] or not donor['cleanup']['succeeded'] or donor['fault']['injected_calls']:
        raise ValueError('A clean branch requires a valid, unmodified donor')
    prefix, turn = branch_inputs(donor, checkpoint, target)
    kwargs = dict(attempt_kwargs, initial_state=deepcopy(checkpoint['databases']), resume_trajectory=prefix)
    controller.restore_checkpoint = deepcopy(checkpoint)
    branch = CheckpointBranchModel(model, prefix_messages(kwargs['task'], prefix), turn)
    return await (run_attempt or run_task_attempt)(model=branch, tool_intervention=controller, **kwargs)


async def resume_after_first_fault(*, snapshot, controller, model, attempt_kwargs, run_attempt=None):
    """Keep the first hit, database, dialogue and total cap; sample only the suffix."""
    kind = snapshot['runtime']['bound']
    if len(snapshot['runtime']['events']) != 1 or controller.budgets != snapshot['budgets']:
        raise ValueError('Continuation requires exactly one inherited hit and the original cap')
    if kind not in controller.budgets or not snapshot['prefix']['tool_events']:
        raise ValueError('First-fault snapshot has no matching mechanism or executed prefix')
    controller.restore_checkpoint = deepcopy(snapshot)
    kwargs = dict(attempt_kwargs, initial_state=deepcopy(snapshot['databases']),
                  resume_trajectory=deepcopy(snapshot['prefix']))
    return await (run_attempt or run_task_attempt)(model=model, tool_intervention=controller, **kwargs)
