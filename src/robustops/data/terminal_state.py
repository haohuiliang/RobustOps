"""Typed, cross-tool state observations for terminal preparation.

Contracts belong to trusted code, never model output. The ledger proves an
observed field/identity proposition within one trajectory, not interpretation
of a user request or completeness of a natural-language answer. Unknown effects
invalidate later-state claims until a subsequent observation resolves them.
"""
from copy import deepcopy
from dataclasses import dataclass
import json

from .terminal_evidence import atom, build_index, decode, validate_atom


@dataclass(frozen=True)
class ToolContract:
    resource: str
    identity_field: str
    effect: str  # read, create, update, delete
    row_paths: tuple = ((), ('data',))
    scope_arguments: tuple = ()
    target_argument: str | None = None
    observed_fields: tuple = ()
    invalidates_family: bool = False
    # (argument name, result field, declared JSON type); successful operations
    # only, with a matching visible schema and one observed result identity.
    argument_effects: tuple = ()

    def __post_init__(self):
        if self.effect not in {'read', 'create', 'update', 'delete'}:
            raise ValueError('unknown contract effect')
        if self.effect in {'update', 'delete'} and not self.target_argument:
            raise ValueError('mutation needs an explicit target argument')


def _key(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':'))


def _resolve(index, location):
    value = index.packet
    for step in location:
        value = decode(value) if step is None else value[step]
    return value


def _rows(value, location, path):
    if not path:
        if isinstance(value, dict):
            yield location, value
        return
    part, *rest = path
    if part == '*':
        if isinstance(value, list):
            for i, child in enumerate(value):
                yield from _rows(child, [*location, i], rest)
    elif isinstance(value, dict) and part in value:
        yield from _rows(value[part], [*location, part], rest)


def _payloads(index, event_id):
    # Exact MCP JSON-text contract already validated by EvidenceIndex. Also
    # support structuredContent; conflicting duplicates are not cherry-picked.
    for ref, node in index.nodes.items():
        if node['event_id'] != event_id or node['role'] != 'observation':
            continue
        if node['whole_document'] and node['field'] == 'text':
            try:
                value = decode(index.resolve(ref))
            except ValueError:
                continue
            if isinstance(value, (dict, list)):
                yield [*node['location'], None], value
    pos = next(i for i, e in enumerate(index.packet['evidence']) if e['id'] == event_id)
    location = ['evidence', pos, 'observation', 'result', 'structuredContent']
    result = index.packet['evidence'][pos]['observation'].get('result', {})
    value = result.get('structuredContent') if isinstance(result, dict) else None
    if isinstance(value, (dict, list)):
        yield location, value


class StateLedger:
    def __init__(self, packet, contracts):
        self.index = build_index(packet)
        self.contracts = contracts
        self.entities = {}
        self.initial_entities = {}
        self.history = []
        self.unmodeled_events = []
        self.barrier = 0
        self._compile()

    @staticmethod
    def identity(resource, value, scope=()):
        return {'resource': resource, 'id': atom(value),
                'scope': {k: atom(v) for k, v in scope}}

    def _entity(self, identity):
        key = _key(identity)
        if key not in self.entities:
            self.entities[key] = {'identity': identity, 'facts': {},
                                  'status': 'unknown', 'event_id': 0,
                                  'invalidated_fields': {}}
        return self.entities[key]

    def _barrier(self, event_id, reason):
        self.barrier = event_id
        self.unmodeled_events.append({'event_id': event_id, 'reason': reason})

    def _invalidate_family(self, resource, event_id, reason):
        # The trusted contract restricts the operation's effect to this resource
        # family. For example, an unresolved permission creation cannot rename
        # a file; it does invalidate earlier permission-state assumptions.
        for entity in self.entities.values():
            if entity['identity']['resource'] == resource:
                entity.update(status='unknown', event_id=event_id, facts={})
        self.unmodeled_events.append({'event_id':event_id,'resource':resource,'reason':reason})

    def _argument_effects(self, event, contract, returned):
        if not contract.argument_effects or len(returned) != 1:
            return
        schemas = [t['function'] for t in self.index.packet['used_tool_schemas']
                   if t.get('function',{}).get('name') == event['tool']]
        if len(schemas) != 1:
            return
        props = schemas[0].get('parameters',{}).get('properties',{})
        item = next(iter(returned.values()))
        eid = event['id']
        identity_facts = item['fields'].get(contract.identity_field,{})
        if not identity_facts:
            return
        for argument, field, kind in contract.argument_effects:
            if field in item['fields'] or argument not in event['arguments']:
                continue  # an actual returned value always wins over a request
            value = event['arguments'][argument]
            if props.get(argument,{}).get('type') != kind:
                continue
            if kind == 'string' and type(value) is not str:
                continue
            if kind == 'array' and (type(value) is not list or
                    props[argument].get('items',{}).get('type') != 'string' or
                    any(type(v) is not str for v in value)):
                continue
            facts = {}
            for ref, node in self.index.nodes.items():
                if node['event_id'] != eid or node['role'] != 'arguments' or node['whole_document']:
                    continue
                loc = node['location']
                tail = loc[loc.index('arguments')+1:]
                if not tail or tail[0] != argument:
                    continue
                path = [field, *tail[1:]]
                facts[_key(path)] = {'path':path,'value':atom(self.index.resolve(ref)),
                    'ref':ref,'event_id':eid,
                    'proof':{'kind':'successful_operation_effect','tool':event['tool'],
                             'argument':argument,'field':field,
                             'result_identity':item['identity'],
                             'identity_refs':[f['ref'] for f in identity_facts.values()],
                             'observation_ref':self.index.events[eid]['observation']['ref'],
                             'schema_name':schemas[0]['name']}}
            if facts:
                item['fields'][field] = facts

    def _compile(self):
        for event in self.index.packet['evidence']:
            eid, tool, args = event['id'], event['tool'], event['arguments']
            contract = self.contracts.get(tool)
            if contract is None:
                self._barrier(eid, 'tool effect not covered by trusted contracts')
                continue
            if any(k not in args for k in contract.scope_arguments):
                self._barrier(eid, 'resource scope absent from arguments')
                continue
            scope = tuple((k, args[k]) for k in contract.scope_arguments)
            target = None
            if contract.target_argument:
                value = args.get(contract.target_argument)
                if value is None or type(value) in (dict, list, bool):
                    self._barrier(eid, 'typed target identity absent')
                    continue
                target = self.identity(contract.resource, value, scope)
            response = self.index.events[eid]['response_status']
            if response != 'reported_success':
                # Do not apply requested values after an error. A failed write
                # is not assumed atomic: later reads can resolve uncertainty.
                if contract.effect in {'update', 'delete'} and target is not None:
                    entity = self._entity(target)
                    entity.update(status='unknown', event_id=eid)
                    entity['facts'] = {}
                elif contract.effect != 'read':
                    self._invalidate_family(contract.resource, eid, 'mutation effect not established')
                self.history.append({'event_id': eid, 'tool': tool,
                                     'response_status': response, 'applied': False})
                continue
            if contract.invalidates_family and contract.effect != 'read':
                for existing in self.entities.values():
                    if existing['identity']['resource'] == contract.resource:
                        existing.update(status='unknown', event_id=eid, facts={})
            if contract.effect == 'delete':
                entity = self._entity(target)
                entity.update(status='deleted', event_id=eid, facts={}, invalidated_fields={})
                self.history.append({'event_id': eid, 'tool': tool, 'effect': 'delete',
                                     'identity': target, 'observation_ref': self.index.events[eid]['observation']['ref']})
                continue
            # Group all representations of one resource in this response. Each
            # returned top-level field replaces its prior subtree, even []/null.
            returned = {}
            for location, payload in _payloads(self.index, eid):
                for path in contract.row_paths:
                    for row_location, row in _rows(payload, location, path):
                        wrappers = [_resolve(self.index, row_location[:j])
                                    for j in range(len(location), len(row_location))]
                        if any(isinstance(w,dict) and w.get('success') is False for w in wrappers):
                            continue
                        value = row.get(contract.identity_field)
                        if value is None or type(value) in (dict, list, bool):
                            continue
                        identity = self.identity(contract.resource, value, scope)
                        if target is not None and identity != target:
                            self._barrier(eid, 'response identity differs from explicit target')
                            continue
                        key = _key(identity)
                        entry = returned.setdefault(key, {'identity': identity, 'fields': {}})
                        row_fields = {}
                        for ref, node in self.index.nodes.items():
                            loc = node['location']
                            if (node['event_id'] != eid or node['role'] != 'observation'
                                    or node['whole_document'] or loc[:len(row_location)] != row_location):
                                continue
                            relative = loc[len(row_location):]
                            if not relative:
                                continue
                            field = relative[0]
                            if contract.observed_fields and field not in contract.observed_fields:
                                continue
                            path_key = _key(relative)
                            facts = row_fields.setdefault(field, {})
                            fact = {'path': relative, 'value': atom(self.index.resolve(ref)),
                                    'ref': ref, 'event_id': eid}
                            old = facts.get(path_key)
                            if old is not None and old['value'] != fact['value']:
                                fact['conflict'] = True
                            if old is not None and old.get('conflict'):
                                fact['conflict'] = True
                            facts[path_key] = fact
                        for field, facts in row_fields.items():
                            previous = entry['fields'].get(field)
                            if previous is not None and (
                                    {k: f['value'] for k, f in previous.items()} !=
                                    {k: f['value'] for k, f in facts.items()} or
                                    any(f.get('conflict') for f in previous.values())):
                                for fact in facts.values():
                                    fact['conflict'] = True
                            entry['fields'][field] = facts
            self._argument_effects(event, contract, returned)
            if target is not None and contract.effect == 'update':
                entity = self._entity(target)
                # A successful write can change fields omitted by a partial
                # response. Invalidate the old snapshot, then re-establish only
                # fields actually returned; arguments never become state.
                entity.update(status='unknown', event_id=eid, facts={})
            if not returned and contract.effect == 'create':
                self._invalidate_family(contract.resource, eid, 'created resource identity not observed')
            if not returned and contract.effect == 'read':
                self.history.append({'event_id':eid,'tool':tool,'effect':'read',
                                     'observation_projected':False})
            for item in returned.values():
                entity = self._entity(item['identity'])
                for field, facts in item['fields'].items():
                    entity['facts'] = {k: f for k, f in entity['facts'].items() if f['path'][0] != field}
                    entity['facts'].update(facts)
                    entity['invalidated_fields'].pop(field, None)
                entity.update(status='observed', event_id=eid)
                if _key(item['identity']) not in self.initial_entities:
                    self.initial_entities[_key(item['identity'])] = {**deepcopy(entity), 'barrier':self.barrier}
                self.history.append({'event_id': eid, 'tool': tool,
                                     'effect': contract.effect, 'identity': item['identity'],
                                     'returned_fields': sorted(item['fields'])})

    def check(self, resource, identity, path, expected, *, scope=()):
        """Check latest observed value; never interpret the associated sentence."""
        validate_atom(expected)
        key = self.identity(resource, identity, scope)
        entity = self.entities.get(_key(key))
        result = {'identity': key, 'path': list(path), 'expected': expected,
                  'scope_note': 'Typed observations or explicit successful effects under supplied tool contracts; not whole-answer approval.'}
        if entity is None:
            return {**result, 'status': 'unknown', 'reason': 'resource not observed'}
        if entity['status'] == 'deleted' and entity['event_id'] > self.barrier:
            return {**result, 'status': 'fail', 'reason': 'resource deleted', 'event_id': entity['event_id']}
        fact = entity['facts'].get(_key(list(path)))
        if fact is None:
            return {**result, 'status': 'unknown', 'reason': 'field not observed or invalidated'}
        if fact['event_id'] <= self.barrier:
            return {**result, 'status': 'unknown', 'reason': 'later unmodeled effect', 'event_id': self.barrier}
        if fact.get('conflict'):
            return {**result, 'status': 'unknown', 'reason': 'conflicting same-event observations'}
        return {**result, 'status': 'pass' if fact['value'] == expected else 'fail',
                'actual': fact['value'], 'ref': fact['ref'], 'event_id': fact['event_id'],
                'proof':fact.get('proof',{'kind':'observation'})}

    def as_dict(self):
        return {'entities': list(self.entities.values()), 'history': self.history,
                'unmodeled_events': self.unmodeled_events,
                'scope': 'Observed fields and explicit successful operation effects; contracts do not establish user-request coverage or terminal semantics.'}


# These endpoint families are a first integration of the generic ledger. Exact
# names and row paths are deliberate; unlisted tools create uncertainty barriers.
DIRECT = ((), ('data',))
FILE_FIELDS = ('id', 'name', 'mimeType', 'description', 'parents', 'driveId', 'trashed', 'starred')
PERMISSION_FIELDS = ('id', 'type', 'role', 'emailAddress', 'displayName', 'deleted')
COMMENT_FIELDS = ('id', 'content', 'resolved', 'deleted')
CONTRACTS = {
    'list_files': ToolContract('drive.file', 'id', 'read', (('files', '*'), ('data', 'files', '*')), observed_fields=FILE_FIELDS),
    'get_files': ToolContract('drive.file', 'id', 'read', target_argument='fileId', observed_fields=FILE_FIELDS),
    'create_file': ToolContract('drive.file', 'id', 'create', observed_fields=FILE_FIELDS),
    'copy_files': ToolContract('drive.file', 'id', 'create', ((), ('data',), ('file',), ('data','file')), observed_fields=FILE_FIELDS, argument_effects=(('parents','parents','array'),('description','description','string'))),
    'update_files': ToolContract('drive.file', 'id', 'update', target_argument='fileId', observed_fields=FILE_FIELDS),
    'delete_files': ToolContract('drive.file', 'id', 'delete', target_argument='fileId', invalidates_family=True),
    'list_drives': ToolContract('drive.drive', 'id', 'read', (('drives','*'),('data','drives','*'))),
    'list_permission': ToolContract('drive.permission', 'id', 'read', (('permissions','*'),('data','permissions','*'),('data','*')), scope_arguments=('fileId',), observed_fields=PERMISSION_FIELDS),
    'get_permission': ToolContract('drive.permission', 'id', 'read', scope_arguments=('fileId',), target_argument='permissionId', observed_fields=PERMISSION_FIELDS),
    'create_permission': ToolContract('drive.permission', 'id', 'create', scope_arguments=('fileId',), observed_fields=PERMISSION_FIELDS),
    'update_permission': ToolContract('drive.permission', 'id', 'update', scope_arguments=('fileId',), target_argument='permissionId', observed_fields=PERMISSION_FIELDS),
    'delete_permission': ToolContract('drive.permission', 'id', 'delete', scope_arguments=('fileId',), target_argument='permissionId', invalidates_family=True),
    'list_comments': ToolContract('drive.comment', 'id', 'read', (('comments','*'),('data','comments','*')), scope_arguments=('fileId',), observed_fields=COMMENT_FIELDS),
    'create_comments': ToolContract('drive.comment', 'id', 'create', scope_arguments=('fileId',), observed_fields=COMMENT_FIELDS),
    'update_comments': ToolContract('drive.comment', 'id', 'update', scope_arguments=('fileId',), target_argument='commentId', observed_fields=COMMENT_FIELDS),
    'delete_comments': ToolContract('drive.comment', 'id', 'delete', scope_arguments=('fileId',), target_argument='commentId'),
    'download_files': ToolContract('drive.download', 'operation_id', 'read'),
    'watch_files': ToolContract('drive.watch', 'id', 'create', scope_arguments=('fileId',)),
    'find_configuration_item_by_serial_number': ToolContract('itsm.ci', 'configuration_item_id', 'read'),
    'register_configuration_item': ToolContract('itsm.ci', 'configuration_item_id', 'create'),
    'update_configuration_item': ToolContract('itsm.ci', 'configuration_item_id', 'update', target_argument='configuration_item_id'),
    'get_user': ToolContract('itsm.user', 'user_id', 'read', target_argument='user_id'),
    'get_user_using_email': ToolContract('itsm.user', 'user_id', 'read'),
    'find_location_by_given_name': ToolContract('itsm.location', 'location_id', 'read'),
}

# These are resource relationships, not language patterns. A model cannot turn
# user.location into CI.location by selecting another field path.
RELATIONS = {
    ('drive.file', ('parents', '*')): ('drive.file', 'id', 'name'),
    ('drive.file', ('driveId',)): ('drive.drive', 'id', 'name'),
    ('itsm.ci', ('owner_id',)): ('itsm.user', 'user_id', 'first_name'),
    ('itsm.ci', ('location_id',)): ('itsm.location', 'location_id', 'name'),
    ('itsm.user', ('location_id',)): ('itsm.location', 'location_id', 'name'),
}


def join_observed_name(ledger, resource, identity, path, expected_name):
    """Check a declared foreign-key relation and target's latest observed name.

    This only proves the explicitly supplied resource/path; the renderer must
    retain that subject and relation. It does not validate a free-text claim.
    """
    pattern = tuple('*' if type(p) is int else p for p in path)
    relation = RELATIONS.get((resource, pattern))
    if relation is None:
        return {'status':'unknown','reason':'relationship not declared'}
    source = ledger.entities.get(_key(ledger.identity(resource, identity)))
    fact = source['facts'].get(_key(list(path))) if source else None
    if fact is None:
        return {'status':'unknown','reason':'source relationship not observed'}
    check_source = ledger.check(resource, identity, path, fact['value'])
    if check_source['status'] != 'pass':
        return {'status':'unknown','source_check':check_source}
    target_resource, identity_field, name_field = relation
    target_id = ledger.index.resolve(fact['ref'])
    check_identity = ledger.check(target_resource, target_id, [identity_field], fact['value'])
    check_name = ledger.check(target_resource, target_id, [name_field], atom(expected_name))
    status = ('unknown' if check_identity['status'] != 'pass' else check_name['status'])
    if resource == 'drive.file' and pattern == ('parents','*'):
        folder = ledger.check('drive.file', target_id, ['mimeType'], atom('application/vnd.google-apps.folder'))
        if folder['status'] != 'pass':
            status = 'unknown'
    return {'status':status,'source_resource':resource,'source_identity':atom(identity),
            'relation_path':list(path),'target_resource':target_resource,
            'target_identity':atom(target_id),'source_check':check_source,
            'identity_check':check_identity,'name_check':check_name}
