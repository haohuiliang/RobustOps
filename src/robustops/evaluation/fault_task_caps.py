"""Batch-level caps count first actual injections, not candidate assignments."""
from math import ceil
from copy import deepcopy


class FaultTaskCaps:
    def __init__(self, config, total_tasks, task_strata=None):
        self.config = deepcopy(config)
        self.total_tasks = total_tasks
        self.sampled = set()
        self.task_strata = task_strata or {}
        self.claims = {g: set() for g in config['task_caps']}
        self.groups = {}
        for group, spec in config['task_caps'].items():
            if not 0 < spec['max_sampled_fraction'] <= 1 or spec['max_total_tasks'] < 0:
                raise ValueError('invalid task cap')
            for kind in spec['labels']:
                if kind in self.groups:
                    raise ValueError('overlapping cap groups')
                self.groups[kind] = group

    def start(self, task_key):
        self.sampled.add(task_key)
        if len(self.sampled) > self.total_tasks:
            raise ValueError('sampled beyond declared batch size')

    def limit(self, group):
        spec = self.config['task_caps'][group]
        return spec['max_total_tasks'] if self.task_strata else min(spec['max_total_tasks'], ceil(len(self.sampled) * spec['max_sampled_fraction']))

    def available(self, kind, task_key):
        if task_key not in self.sampled:
            raise ValueError('task not registered as sampled')
        group = self.groups.get(kind)
        if group is None or task_key in self.claims[group]:
            return True
        spec = self.config['task_caps'][group]
        if len(self.claims[group]) >= self.limit(group):
            return False
        if self.task_strata:
            mode, domain = self.task_strata[task_key]
            mode_started = sum(self.task_strata[k][0] == mode for k in self.sampled)
            domain_started = sum(self.task_strata[k][1] == domain for k in self.sampled)
            domain_total = sum(v[1] == domain for v in self.task_strata.values())
            mode_hits = sum(self.task_strata[k][0] == mode for k in self.claims[group])
            domain_hits = sum(self.task_strata[k][1] == domain for k in self.claims[group])
            mode_limit = min(spec['max_total_tasks_per_mode'], ceil(mode_started * spec['max_sampled_fraction_per_mode']))
            domain_limit = min(ceil(domain_started * spec['max_domain_fraction']), ceil(domain_total * spec['max_domain_fraction']))
            return mode_hits < mode_limit and domain_hits < domain_limit
        return True

    def claim(self, kind, task_key):
        # Synchronous check+update: atomic between awaits in this process's event loop.
        if not self.available(kind, task_key):
            return False
        group = self.groups.get(kind)
        if group is not None:
            self.claims[group].add(task_key)
        return True

    def restore(self, report, sampled_keys):
        if self.sampled or any(self.claims.values()):
            raise ValueError('restore caps before starting work')
        if report['total_tasks'] != self.total_tasks:
            raise ValueError('resume batch size changed')
        self.sampled = set(sampled_keys)
        if self.task_strata and not self.sampled <= self.task_strata.keys():
            raise ValueError('resume sampled an unknown task')
        for group, prior in report['groups'].items():
            keys = set(prior['task_keys'])
            if not keys <= self.sampled or len(keys) != prior['triggered_tasks']:
                raise ValueError('inconsistent prior cap evidence')
            self.claims[group] = keys

    def report(self):
        return dict(sampled_tasks=len(self.sampled), sampled_task_keys=sorted(self.sampled), total_tasks=self.total_tasks,
            groups={g:dict(triggered_tasks=len(self.claims[g]), current_limit=self.limit(g),
                           task_keys=sorted(self.claims[g]),
                           by_mode=dict((m, sum(self.task_strata[k][0] == m for k in self.claims[g])) for m in sorted({v[0] for v in self.task_strata.values()})),
                           by_domain=dict((d, sum(self.task_strata[k][1] == d for k in self.claims[g])) for d in sorted({v[1] for v in self.task_strata.values()})),
                           **deepcopy(spec))
                    for g, spec in self.config['task_caps'].items()})
