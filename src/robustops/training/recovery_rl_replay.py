"""Delayed zero-PG task/anchor queue; trajectories are always generated afresh."""
from copy import deepcopy


class ZeroSignalReplay:
    def __init__(self, delay, state=None):
        if type(delay) is not int or delay < 0:
            raise ValueError('zero-PG replay delay must be a nonnegative integer')
        self.delay = delay
        state = state or {}
        if state and state['delay'] != delay:
            raise ValueError('replay delay changed across checkpoint resume')
        self.pending = deepcopy(state.get('pending', []))
        self.serial = state.get('serial', 0)

    @staticmethod
    def key(flow, row, state_path):
        return [flow, row['task_id'], row['mode'], state_path if flow == 'fault' else None]

    def reject(self, flow, row, plan, state_path, step, previous=None):
        if not self.delay or flow not in ('clean', 'fault'):
            return None
        if flow == 'fault' and not state_path:
            raise ValueError('fault replay requires the original post-fault snapshot')
        key = self.key(flow, row, state_path)
        old = next((x for x in self.pending if x['key'] == key), previous)
        self.pending = [x for x in self.pending if x['key'] != key]
        self.serial += 1
        entry = dict(key=key, flow=flow, row=deepcopy(row), plan=deepcopy(plan),
                     state_path=state_path, first_zero_step=old['first_zero_step'] if old else step,
                     last_zero_step=step, eligible_step=step+self.delay, order=self.serial,
                     zero_count=(old['zero_count'] if old else 0)+1)
        self.pending.append(entry)
        return deepcopy(entry)

    def claim(self, flow, step, excluded_cells):
        eligible = [x for x in self.pending if x['flow'] == flow and x['eligible_step'] <= step
                    and (x['row']['task_id'], x['row']['mode']) not in excluded_cells]
        if not eligible:
            return None
        entry = min(eligible, key=lambda x: (
            x['last_zero_step'], x['order']))
        self.pending.remove(entry)
        return deepcopy(entry)

    def resolve(self, flow, row, state_path):
        key = self.key(flow, row, state_path)
        self.pending = [x for x in self.pending if x['key'] != key]

    def state_dict(self):
        return dict(delay=self.delay, serial=self.serial, pending=deepcopy(self.pending))

    def summary(self, step):
        return {flow: dict(pending=sum(x['flow'] == flow for x in self.pending),
                          eligible=sum(x['flow'] == flow and x['eligible_step'] <= step for x in self.pending))
                for flow in ('clean', 'fault')}
