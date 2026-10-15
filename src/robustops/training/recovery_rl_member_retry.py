"""Bounded whole-member retries, with model and infrastructure causes retained."""
from pathlib import Path
import json


class InvalidJointRollout(RuntimeError):
    """The isolated episode is invalid, but its environment cleanup succeeded."""


class ModelProtocolRollout(RuntimeError):
    """A model tool-call format failure; the failed episode was saved and cleaned."""


class MemberRetryBudget:
    def __init__(self, events, log_path, context, limit=4):
        self.events, self.log_path, self.context = events, Path(log_path), context
        self.limit, self.used, self.exhausted = limit, 0, False
        self.failure_counts = {}

    def discard_reason(self, phase='member'):
        kinds = set(self.failure_counts)
        prefix = ('model_protocol' if kinds == {'model_protocol_error'}
                  else 'mixed' if 'model_protocol_error' in kinds else 'invalid')
        return f'{prefix}_{phase}_retry_limit'

    async def run(self, submit, sample):
        retry = 0
        while not self.exhausted:
            try:
                return await submit(retry)
            except Exception as exc:
                cause = getattr(exc, 'cause', exc)
                if not isinstance(cause, (InvalidJointRollout, ModelProtocolRollout)):
                    raise
                kind = 'model_protocol_error' if isinstance(cause, ModelProtocolRollout) else 'infrastructure_invalid'
                self.failure_counts[kind] = self.failure_counts.get(kind, 0) + 1
                event = dict(self.context, sample=sample, member_retry=retry,
                             group_retries_used=self.used, error=str(exc), failure_kind=kind)
                if self.used >= self.limit:
                    self.exhausted = True
                    event.update(action='discard_group', reason=self.discard_reason(),
                                 failure_counts=dict(self.failure_counts))
                else:
                    self.used += 1
                    retry = self.used  # atomic shared claim before next await
                    event.update(action='retry_member', next_member_retry=retry)
                self.events.append(event)
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with self.log_path.open('a') as stream:
                    stream.write(json.dumps(event)+'\n')
        return None
