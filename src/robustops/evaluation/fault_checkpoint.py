"""Resume an actual saved fault boundary; never execute its prefix again."""
from copy import deepcopy
from robustops.envs.enterpriseops.runtime import InfrastructureError


def prefix_messages(task, trajectory):
    from robustops.data.trajectory_messages import build_messages
    messages, indices = build_messages(task, trajectory, allow_tool_ending=True)
    # Keep the original argument string formatting, not a reserialized variant.
    for index, turn in zip(indices, trajectory['turns']):
        if turn.get('tool_calls'):
            messages[index]['tool_calls'] = deepcopy(turn['tool_calls'])
    return messages


class CheckpointBranchModel:
    def __init__(self, model, expected_prefix, target_turn):
        self.model = model
        self.expected_prefix = deepcopy(expected_prefix)
        self.target = dict(role='assistant', content=target_turn.get('assistant_content') or '',
                           tool_calls=deepcopy(target_turn['tool_calls']))
        if len(self.target['tool_calls']) != 1:
            raise InfrastructureError('checkpoint requires one effective target call')
        self.target_sent = False
        self.generated = 0

    async def complete(self, messages, tools):
        if not self.target_sent:
            if messages != self.expected_prefix:
                raise InfrastructureError('checkpoint conversation prefix mismatch')
            self.target_sent = True
            return deepcopy(self.target), dict(prompt_tokens=0, completion_tokens=0,
                                                finish_reason='tool_calls')
        self.generated += 1
        return await self.model.complete(messages, tools)

    def report(self):
        return dict(restoration='actual_boundary_snapshot', prefix_reexecuted_calls=0,
                    target_dispatched=self.target_sent, generated_turns=self.generated,
                    restored_prefix_tool_calls=sum(m['role']=='tool' for m in self.expected_prefix))
