"""Actual-task exposure balancing and reproducible random repetition limits."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random

GROUPS = {
    'transient_unexecuted': 'unexecuted',
    'partial_read': 'missing_read', 'truncated_read': 'missing_read',
    'unknown_commit': 'unknown_commit', 'unknown_commit_create': 'unknown_commit',
    'ignored_filter': 'ignored_filter', 'reply_mixup': 'reply_mixup',
}

STRATEGY_PATH = 'recipes/enterpriseops/faults/strategy.json'


def shared_strategy(path=STRATEGY_PATH, *, root=None):
    config = load_schedule(path, root=root)
    validate_schedule(config)
    return config


def draw_repeat_limit(kind, config, identity):
    """Draw once when a mechanism is selected; persist the resulting plan."""
    weights = config['repeat_probabilities'][kind]
    levels = sorted(map(int, weights))
    rng = random.Random(f"{config['seed']}/{identity}/{kind}")
    return rng.choices(levels, weights=[weights[str(k)] for k in levels], k=1)[0]


def observation_budgets(selection):
    """Unit budgets only qualify opportunities; they are never injection plans."""
    entries = selection['tools'] if isinstance(selection, dict) else selection
    return {kind: 1 for tool in entries for kind in tool['faults']}


def select_clean_branches(rows, clean, schedule):
    """Balance after all clean runs finish, using only actual qualified points.

    clean maps cells to persisted attempts and checkpoints. No outcome or target
    label absent from the clean path participates in selection.
    """
    balance = FaultScheduler(rows, schedule['seed'])
    plans = {}
    for row in sorted(rows, key=lambda r: r['cell']):
        key = str(row['cell'])
        donor = clean[key]
        attempt = donor['attempt']
        if not attempt['valid'] or not attempt['cleanup']['succeeded']:
            plans[key] = dict(kind=None, reason='invalid_clean')
            continue
        checkpoints = donor['checkpoints']
        points = [p for p in attempt['fault']['opportunities'] if p['kind'] in checkpoints]
        if not points:
            plans[key] = dict(kind=None, reason='no_qualified_clean_opportunity')
            continue
        kind = balance.rank({p['kind'] for p in points}, key, schedule.get('preferred_mechanisms', ()))[0]
        target = min((p for p in points if p['kind'] == kind), key=lambda p: p['call_index'])
        cap = draw_repeat_limit(kind, schedule, f"{row['task_id']}/{row['mode']}")
        plans[key] = dict(kind=kind, budget=cap, target=target)
        balance.claim(kind, key)
    return plans, balance.report()


def load_schedule(path, *, root=None):
    """Resolve shared selection/dose rules with data-only schedule bindings."""
    root = Path(root) if root is not None else Path(__file__).resolve().parents[3]
    config = json.loads((root / path).read_text())
    if 'strategy' not in config:
        return config  # Existing run snapshots already contain expanded settings.
    if set(config) != {'strategy', 'candidate_budgets'}:
        raise ValueError('Schedule bindings may contain only strategy and data paths')
    strategy = json.loads((root / config['strategy']).read_text())
    if 'strategy' in strategy or 'candidate_budgets' in strategy:
        raise ValueError('Shared strategy must not contain schedule/data bindings')
    strategy['candidate_budgets'] = config['candidate_budgets']
    validate_schedule(strategy)
    return strategy


def validate_schedule(config):
    waiting = config.get('reservation_mode', 'eligible_now') == 'wait_for_priority_label'
    if (config.get('reservation_mode', 'eligible_now') not in ('eligible_now', 'wait_for_priority_label')
            or config.get('scheduler') != 'finite_budget' or config.get('max_zero_hit_replays') != int(waiting)
            or config.get('task_caps') != {}
            or not set(config.get('preferred_mechanisms', [])) <= GROUPS.keys()
            or config.get('point_policy') != 'earliest_strictly_eligible'
            or config.get('binding') != 'eligible_calls'
            or config.get('groups') != GROUPS
            or type(config.get('seed')) is not int):
        raise ValueError('finite-budget schedule requires five groups, no caps, matching reservation/replay mode, cumulative eligible calls')
    probabilities = config['repeat_probabilities']
    if set(probabilities) != set(GROUPS):
        raise ValueError('Every mechanism requires repetition probabilities')
    for kind, weights in probabilities.items():
        levels = ({'1', '2', '4'} if kind == 'transient_unexecuted' else
                  {'1', '2'} if kind in ('unknown_commit', 'unknown_commit_create', 'ignored_filter') else {'1', '4'})
        if (set(weights) != levels or any(type(v) not in (int, float) or not 0 < v < 1 for v in weights.values())
                or abs(sum(weights.values()) - 1) > 1e-9):
            raise ValueError(f'Invalid repetition probabilities: {kind}')


def assign_budgets(rows, config):
    """Draw independently per task/tool configuration/mechanism, then freeze.

    Stable identities make assignment independent of row order and pool size.
    Val and Test share probabilities; neither outcomes nor split names enter RNG.
    """
    validate_schedule(config)
    result = {}
    for row in rows:
        kinds = {kind for tool in row['selection']['tools'] for kind in tool['faults']}
        budgets = {}
        for kind in sorted(kinds):
            budgets[kind] = draw_repeat_limit(kind, config, f"{row['task_id']}/{row['mode']}")
        result[str(row['cell'])] = budgets
    return result


class FaultScheduler:
    def __init__(self, rows, seed):
        self.seed=seed
        self.rows={str(r['cell']):r for r in rows}
        self.claims={}
        self.groups=Counter();self.labels=Counter()
        self.domain_groups=defaultdict(Counter);self.domain_labels=defaultdict(Counter)

    def rank(self,kinds,key,preferred=()):
        row=self.rows[key];domain=row['domain']
        rng=random.Random(f"{self.seed}/{row['task_id']}/{row['mode']}")
        groups=sorted(set(GROUPS.values()));rng.shuffle(groups)
        labels=sorted(GROUPS);rng.shuffle(labels)
        preferred_groups={GROUPS[k] for k in preferred}
        return sorted(kinds,key=lambda k:(self.groups[GROUPS[k]],int(GROUPS[k] not in preferred_groups),self.domain_groups[domain][GROUPS[k]],groups.index(GROUPS[k]),self.labels[k],self.domain_labels[domain][k],labels.index(k)))

    def claim(self,kind,key):
        if key in self.claims:
            if self.claims[key]!=kind:raise ValueError('one mechanism per task-mode')
            return
        self.claims[key]=kind;g=GROUPS[kind];d=self.rows[key]['domain']
        self.groups[g]+=1;self.labels[kind]+=1
        self.domain_groups[d][g]+=1;self.domain_labels[d][kind]+=1

    def report(self):
        return dict(unit='distinct_task_mode_first_actual_hit',claims=dict(self.claims),groups=dict(self.groups),labels=dict(self.labels),by_domain={d:dict(c) for d,c in self.domain_groups.items()})


def main():
    import argparse

    parser = argparse.ArgumentParser(description='Allocate candidate budgets using shared fault selection/dose rules.')
    parser.add_argument('--config', type=Path, required=True, help='Schedule data bindings and shared strategy')
    parser.add_argument('--policy', type=Path, required=True, help='Prepared fault-candidate policy')
    parser.add_argument('--output', type=Path, required=True, help='Candidate-budget output; differing existing files are preserved')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[3]
    schedule = load_schedule(args.config, root=root)
    policy = json.loads((root / args.policy).read_text())
    budgets = assign_budgets(policy['rows'], schedule)
    output = root / args.output
    if output.exists() and json.loads(output.read_text()) != budgets:
        raise ValueError(f'Existing candidate budgets differ; choose a new output: {output}')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(budgets, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), task_modes=len(budgets),
        candidate_dose_counts={kind:dict(Counter(row[kind] for row in budgets.values() if kind in row))
                               for kind in GROUPS})))


if __name__ == '__main__':
    main()
