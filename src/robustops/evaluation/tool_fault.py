"""Episode-local tool-boundary faults. Never intercept reset or verification."""
from copy import deepcopy
from datetime import datetime, timezone
from jsonschema import Draft7Validator

CASE_PERCENTAGES = {0: 25, 1: 64, 2: 10, 3: 1}
CASE_KINDS = {'pre_dispatch', 'pre_dispatch_unknown', 'post_commit_response_loss',
              'truncated_read', 'partial_read'}
TOOL_FAULT_KINDS = CASE_KINDS | {'persistent', 'observe'}


def _case_schedule(config):
    import random
    from itertools import combinations
    n = config['planned_injections']
    if type(n) is not int or n not in CASE_PERCENTAGES:
        raise ValueError('planned injections must be 0..3')
    candidates = config['candidates']
    if any(c['kind'] not in CASE_KINDS for c in candidates):
        raise ValueError('case sampling only supports finite recovery faults')
    if any('count' in c or 'counts' in c for c in candidates):
        raise ValueError('case sampling assigns counts from the case quota')
    if not n:
        return []
    choices = []
    for size in range(1, min(2, n) + 1):
        allocations = [(n,)] if size == 1 else [(i, n-i) for i in range(1, n)]
        for combo in combinations(candidates, size):
            if len({c['tool'] for c in combo}) != size:
                continue
            if sum(c['kind'] != 'pre_dispatch' for c in combo) > 1:
                continue
            for allocation in allocations:
                if all(c['kind'] == 'pre_dispatch' or count == 1 for c, count in zip(combo, allocation)):
                    choices.append((combo, allocation))
    if not choices:
        raise ValueError('no compatible schedule for the assigned case quota')
    rng = random.Random(config['seed'])
    combo, allocation = rng.choice(choices)
    schedule = deepcopy(list(combo))
    for c, count in zip(schedule, allocation):
        positions = c.pop('trigger_positions', [1])
        if not positions or any(type(i) is not int or i < 1 for i in positions):
            raise ValueError('invalid trigger positions')
        c['match_from'] = rng.choice(positions)
        c['count'] = count
        if c['kind'] == 'pre_dispatch':
            c['error_profile'] = rng.choice(c.pop('error_profiles', [c.get('error_profile')] if c.get('error_profile') else ['unavailable', 'rate_limit', 'resource_busy']))
    return schedule


def subset(expected, actual):
    return all(k in actual and (subset(v, actual[k]) if isinstance(v, dict) and isinstance(actual[k], dict)
                               else actual[k] == v) for k, v in expected.items())


def incomplete_list(body, list_field=None):
    """Hide one list only when its existing sibling count exposes the omission."""
    if not isinstance(body, dict):
        return None
    counts = ('resultSizeEstimate', 'total_count', 'totalCount', 'count', '@odata.count', 'total')
    candidates = [key for key, value in body.items()
                  if isinstance(value, list) and value
                  and (list_field is None or key == list_field)
                  and any(type(body.get(k)) in (int, float) and body[k] >= len(value) for k in counts)]
    # A generic sibling count cannot identify which of multiple arrays it counts.
    if len(candidates) == 1:
        changed = deepcopy(body)
        changed[candidates[0]] = []
        return changed
    if len(candidates) > 1:
        return None
    for key, value in body.items():
        if isinstance(value, dict):
            nested = incomplete_list(value, list_field)
            if nested is not None:
                changed = deepcopy(body)
                changed[key] = nested
                return changed
    return None


def response_identity_values(response, field):
    """Extract documented returned IDs without issuing any confirmation query."""
    import json
    values = []
    def visit(value):
        if isinstance(value, dict):
            found = value.get(field)
            if isinstance(found, (str, int)) and not isinstance(found, bool) and found != '':
                values.append(found)
                return  # This resource's ID precedes IDs of embedded resources.
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)
    result = response.get('result', {})
    visit(result.get('structuredContent'))
    for block in result.get('content', []):
        if block.get('type') == 'text':
            try:
                visit(json.loads(block['text']))
            except (ValueError, TypeError):
                pass
    return list(dict.fromkeys(values))


class ToolFault:
    def __init__(self, config):
        self.config = deepcopy(config)
        if config['kind'] not in TOOL_FAULT_KINDS:
            raise ValueError('unsupported fault kind')
        if config.get('error_profile', 'unavailable') not in ('unavailable', 'rate_limit', 'resource_busy'):
            raise ValueError('unsupported transient error profile')
        if config['kind'] == 'truncated_read' and config.get('read_only') is not True:
            raise ValueError('truncated response requires a confirmed read-only tool')
        if config.get('count', 1) not in (1, 2, 3) or (config.get('count') == 3 and config['kind'] != 'pre_dispatch'):
            raise ValueError('only pre-dispatch transient faults support a three-call burst')
        if config['kind'] == 'post_commit_response_loss' and config.get('count', 1) != 1:
            raise ValueError('response loss is one event')
        if not isinstance(config.get('match_from', 1), int) or config.get('match_from', 1) < 1:
            raise ValueError('match_from must be positive')
        self.events = []
        self.calls = []
        self.matches = 0
        self.used = 0
        self.caller = None

    def bind(self, caller, tools):
        if self.caller is not None:
            raise ValueError('fault instances are single-episode; create a new instance after reset')
        self.caller = caller
        if self.config['kind'] == 'observe':
            return self
        schema = next(t['function']['parameters'] for t in tools if t['function']['name'] == self.config['tool'])
        if caller.routes[self.config['tool']] != self.config['server']:
            raise ValueError('fault tool route mismatch')
        self.validator = Draft7Validator(schema)
        self.exposed_arguments = set(schema.get('properties', {}))
        path = self.config.get('confirmation_path')
        if path:
            if self.config['kind'] not in ('pre_dispatch_unknown', 'post_commit_response_loss'):
                raise ValueError('confirmation path only supports unknown commit')
            if path['write_tool'] != self.config['tool'] or path['effect_kind'] not in {'parameters', 'action'}:
                raise ValueError('unsupported confirmation effect')
            if path['effect_kind'] == 'action' and (path['effect_parameters'] or not path.get('action_effect', '').strip()):
                raise ValueError('action confirmation requires a documented implicit effect')
            read_schema = next(t['function']['parameters'] for t in tools if t['function']['name'] == path['read_tool'])
            self.confirmation_validator = Draft7Validator(read_schema)
        creation = self.config.get('creation_confirmation_path')
        if creation:
            expected_tool = creation['write_tool']
            if self.config['tool'] != expected_tool:
                raise ValueError('creation path does not match the intercepted tool')
            allowed = {'pre_dispatch_unknown', 'post_commit_response_loss'}
            if self.config['kind'] not in allowed:
                raise ValueError('creation path used with an unsupported runtime mechanism')
            write_schema = next(t['function']['parameters'] for t in tools if t['function']['name'] == creation['write_tool'])
            read_schema = next(t['function']['parameters'] for t in tools if t['function']['name'] == creation['read_tool'])
            if creation['read_tool'] not in caller.routes or creation['write_tool'] not in caller.routes:
                raise ValueError('creation recovery route is not exposed')
            keys = creation['scope_parameters'] + creation['match_parameters']
            if not creation['match_parameters'] or any(k not in write_schema.get('properties', {}) for k in keys):
                raise ValueError('creation lookup lacks valid runtime scope/match parameters')
            if any(k not in read_schema.get('properties', {}) for k in creation['read_parameters']):
                raise ValueError('creation lookup requires unsupported read inputs')
            self.creation_read_validator = Draft7Validator(read_schema)
            self.creation_read_arguments = set(read_schema.get('properties', {}))
        return self

    @staticmethod
    def error(message):
        return {'is_error': True, 'outcome': {'transport_ok': True, 'application_ok': False,
                'source': 'tool_error', 'mcp_is_error': True, 'status_code': None, 'message': message},
                'result': {'isError': True, 'content': [{'type': 'text', 'text': message}]}}

    def eligibility(self, name, arguments):
        matched = (self.config['kind'] != 'observe' and name == self.config['tool'] and subset(self.config.get('arguments_match', {}), arguments)
                   and self.validator.is_valid(arguments))
        eligibility_reason = None
        if matched and not set(arguments) <= self.exposed_arguments:
            matched = False
            eligibility_reason = 'arguments not exposed by the tool schema'
        path = self.config.get('confirmation_path')
        if matched and path:
            ids = path['identity_parameters']
            effects = path['effect_parameters']
            query = {key: arguments[key] for key in ids if key in arguments}
            # Do not inject unless the actual call has a valid same-ID lookup
            # and all supplied non-identity arguments have annotated effects.
            if len(query) != len(ids) or any(value is None or value == '' for value in query.values()):
                eligibility_reason = 'missing runtime identity'
            elif not self.confirmation_validator.is_valid(query):
                eligibility_reason = 'confirmation query is not schema-valid'
            elif path['effect_kind'] == 'action' and (set(arguments) - set(ids)):
                eligibility_reason = 'actual action arguments beyond annotated identity'
            elif path['effect_kind'] == 'parameters' and (not (set(arguments) - set(ids)) or not (set(arguments) - set(ids)) <= set(effects)):
                eligibility_reason = 'actual write effects not fully covered by annotation'
            if eligibility_reason:
                matched = False
        creation = self.config.get('creation_confirmation_path')
        if matched and creation:
            keys = creation['scope_parameters'] + creation['match_parameters']
            if any(k not in arguments or arguments[k] is None or arguments[k] == '' or arguments[k] == [] for k in keys):
                matched = False
                eligibility_reason = 'missing runtime creation lookup scope or match'
            if matched:
                query = deepcopy(creation.get('read_constants', {}))
                sources = creation.get('read_sources', {})
                for target, source_keys in sources.items():
                    # More complex query construction needs an explicit adapter;
                    # never invent a lost generated ID or concatenate values.
                    if not isinstance(source_keys, list) or len(source_keys) != 1 or source_keys[0] not in arguments:
                        matched = False
                        eligibility_reason = 'creation query cannot be built from original arguments'
                        break
                    query[target] = arguments[source_keys[0]]
                if matched and (not set(query) <= self.creation_read_arguments
                                or not set(creation['read_parameters']) <= set(query)
                                or not self.creation_read_validator.is_valid(query)):
                    matched = False
                    eligibility_reason = 'creation recovery query is not schema-valid from original arguments'
        return matched, eligibility_reason

    async def call_tool(self, name, arguments):
        matched, eligibility_reason = self.eligibility(name, arguments)
        if matched:
            self.matches += 1
        inject = matched and self.matches >= self.config.get('match_from', 1) and (self.config['kind'] == 'persistent' or self.used < self.config.get('count', 1))
        entry = {'call_index': len(self.calls) + 1, 'server': self.caller.routes.get(name),
                 'tool': name, 'arguments': deepcopy(arguments), 'matched': matched,
                 'time': datetime.now(timezone.utc).isoformat(), 'dispatched': False}
        if eligibility_reason:
            entry['eligibility_skip'] = eligibility_reason
        if inject and self.config['kind'] in ('pre_dispatch', 'pre_dispatch_unknown', 'persistent'):
            message = {'pre_dispatch': 'Temporary service unavailable. The operation was not executed. Please retry.', 'pre_dispatch_unknown': 'Timeout waiting for the tool response. Execution status is unknown.', 'persistent': 'This tool is unavailable for the remainder of this session. The operation was not executed.'}[self.config['kind']]
            if self.config['kind'] == 'pre_dispatch':
                message = {
                    'unavailable': message,
                    'rate_limit': 'Rate limit exceeded. The operation was not executed. This is a temporary rejection; retry later.',
                    'resource_busy': 'Resource temporarily busy. The operation was not executed. Retry the operation.',
                }[self.config.get('error_profile', 'unavailable')]
            visible = self.error(message)
            entry.update(injected=True, committed=False, visible_result=deepcopy(visible))
            self.used += 1
            self.events.append(deepcopy(entry))
        else:
            entry['dispatched'] = True
            actual = await self.caller.call_tool(name, arguments)
            entry['actual_result'] = deepcopy(actual)
            visible = actual
            if inject and self.config['kind'] == 'truncated_read' and not actual.get('is_error') and actual.get('outcome', {}).get('application_ok') is True:
                import json
                for block in actual.get('result', {}).get('content', []):
                    body = block.get('text', '') if block.get('type') == 'text' else ''
                    try:
                        parsed = json.loads(body)
                    except (ValueError, TypeError):
                        continue
                    if not isinstance(parsed, (dict, list)) or len(body) < 16:
                        continue
                    visible = deepcopy(actual)
                    visible['result']['content'] = [{'type': 'text', 'text': body[:len(body)//2]}]
                    visible['result'].pop('structuredContent', None)
                    entry.update(injected=True, committed=False, visible_result=deepcopy(visible))
                    self.used += 1
                    self.events.append(deepcopy(entry))
                    break
            if inject and self.config['kind'] == 'partial_read' and not actual.get('is_error') and actual.get('outcome', {}).get('application_ok') is True:
                import json
                entry['eligibility_skip'] = 'no unambiguous nonempty collection with native sibling count'
                for i, block in enumerate(actual.get('result', {}).get('content', [])):
                    if block.get('type') != 'text':
                        continue
                    try:
                        body = json.loads(block['text'])
                    except (ValueError, TypeError):
                        continue
                    body = incomplete_list(body, self.config.get('list_field'))
                    if body is not None:
                        entry.pop('eligibility_skip', None)
                        visible = deepcopy(actual)
                        visible['result']['content'] = [{'type': 'text', 'text': json.dumps(body)}]
                        visible['result'].pop('structuredContent', None)
                        entry.update(injected=True, committed=False, visible_result=deepcopy(visible))
                        self.used += 1
                        self.events.append(deepcopy(entry))
                        break
            creation = self.config.get('creation_confirmation_path')
            creation_has_id = not creation or len(response_identity_values(actual, creation['returned_identity_field'])) == 1
            if inject and self.config['kind'] == 'post_commit_response_loss' and creation_has_id and not actual.get('is_error', False) and actual.get('outcome', {}).get('application_ok') is True:
                visible = self.error('Timeout waiting for the tool response. Execution status is unknown.')
                entry.update(injected=True, committed=True, visible_result=deepcopy(visible))
                self.used += 1
                self.events.append(deepcopy(entry))
        self.calls.append(entry)
        return visible

    def report(self):
        return {'enabled': self.config['kind'] != 'observe', 'triggered': bool(self.events), 'config': self.config,
                'matching_calls': self.matches, 'injected_calls': self.used,
                'events': deepcopy(self.events)}


class RandomToolFault:
    """Bounded schedule sampled before rollout; one intervention per exposed tool."""
    def __init__(self, config):
        import random
        from itertools import combinations
        self.config = deepcopy(config)
        if config.get('kind', 'random') not in {'random', 'case_random'}:
            raise ValueError('unsupported fault schedule kind')
        if any(c['kind'] not in TOOL_FAULT_KINDS for c in config['candidates']):
            raise ValueError('unsupported fault kind')
        if config.get('kind') == 'case_random':
            self.limit = config['planned_injections']
            self.schedule = _case_schedule(config)
            self._init_handlers()
            return
        n = config.get('event_count', 1)
        if n not in (1, 2):
            raise ValueError('one or two fault event segments required')
        self.limit = config.get('max_affected_calls', 4)
        if not isinstance(self.limit, int) or not 1 <= self.limit <= 4:
            raise ValueError('affected-call budget must be 1..4')
        candidates = config['candidates']
        # Unknown-commit branches are one family. With <=2 events, <=2 families.
        choices = [c for c in combinations(candidates, n)
                   if len({x['tool'] for x in c}) == n
                   and sum(x['kind'] == 'persistent' for x in c) <= 1]
        if not choices:
            raise ValueError('no eligible distinct-tool combination')
        rng = random.Random(config['seed'])
        self.schedule = deepcopy(list(rng.choice(choices)))
        for c in self.schedule:
            positions = c.pop('trigger_positions', [1])
            if not positions or any(not isinstance(i, int) or i < 1 for i in positions):
                raise ValueError('invalid trigger positions')
            c['match_from'] = rng.choice(positions)
            counts = c.pop('counts', [c.get('count', 1)])
            if c['kind'] != 'pre_dispatch' and counts != [1]:
                raise ValueError('burst supported only for transient pre-dispatch')
            c['count'] = rng.choice(counts)
        self._init_handlers()

    def _init_handlers(self):
        self.handlers = {c['tool']: ToolFault(c) for c in self.schedule}
        self.observer = ToolFault({'kind': 'observe'})
        self.calls, self.events = [], []
        self.caller = None
        self.exhausted = False
        self.injection_budget_exhausted = self.limit == 0
        self.has_persistent = any(c['kind'] == 'persistent' for c in self.schedule)

    def bind(self, caller, tools):
        if self.caller is not None:
            raise ValueError('schedule is episode-local')
        self.caller = caller
        for handler in [*self.handlers.values(), self.observer]:
            handler.bind(caller, tools)
        return self

    async def call_tool(self, name, arguments):
        if self.exhausted:
            raise RuntimeError('fault budget exhausted; controller must stop rollout')
        handler = self.observer if self.injection_budget_exhausted else self.handlers.get(name, self.observer)
        result = await handler.call_tool(name, arguments)
        entry = deepcopy(handler.calls[-1])
        entry['call_index'] = len(self.calls) + 1
        if entry.get('injected'):
            entry['fault_kind'] = handler.config['kind']
            self.events.append(deepcopy(entry))
        self.calls.append(entry)
        self.injection_budget_exhausted = len(self.events) >= self.limit
        # Finite faults stop injecting, not executing. Preserve the old persistent
        # contract: a permanently unavailable endpoint must never silently revive.
        self.exhausted = self.injection_budget_exhausted and self.has_persistent
        return result

    def report(self):
        return dict(enabled=bool(self.schedule), triggered=bool(self.events), config=deepcopy(self.config),
                    planned_injections=(self.limit if self.config.get('kind') == 'case_random' else None),
                    schedule=deepcopy(self.schedule), injected_calls=len(self.events),
                    triggered_segments=sum(bool(h.events) for h in self.handlers.values()),
                    budget_exhausted=self.injection_budget_exhausted,
                    rollout_stop_required=self.exhausted, events=deepcopy(self.events))
