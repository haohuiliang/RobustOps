"""Diagnostic connector faults; original tools, resources and verifiers stay intact."""
from copy import deepcopy
import json
import re
from urllib.parse import urlencode
from jsonschema import Draft7Validator
from robustops.evaluation.tool_fault import ToolFault
from robustops.evaluation.tool_names import is_read_name
from robustops.evaluation.fault_responses import body, collection

def replace_body(response, value):
    response = deepcopy(response)
    response['result']['content'] = [{'type': 'text', 'text': json.dumps(value, ensure_ascii=False)}]
    if 'structuredContent' in response['result']:
        response['result']['structuredContent'] = deepcopy(value)
    return response

def optional_predicate(arguments, parameters):
    """Deliberately restricted, generic predicate parser; unknown forms skip."""
    tests=[]
    for key in parameters:
        value=arguments.get(key)
        if value in [None,'',[]]: continue
        if key in ['q','_filter']:
            if not isinstance(value,str): return None
            fn=re.fullmatch(r"\s*(startswith|contains|endswith)\s*\(\s*(\w+)\s*,\s*'([^']*)'\s*\)\s*",value)
            expr=re.fullmatch(r"\s*(\w+)\s*(eq|=|contains)\s*'([^']*)'\s*",value)
            if fn: operation,field,wanted=fn.groups()
            elif expr: field,operation,wanted=expr.groups()
            else: return None
            tests.append((field,operation,wanted))
        elif not key.endswith('_like'): tests.append((key,'=',value))
        else: return None
    if not tests:return None
    def evaluate(obj):
        for field,operation,wanted in tests:
            if field not in obj:return None
            observed=obj[field]
            if operation in ['=','eq']:
                if observed!=wanted:return False
            elif not isinstance(observed,str):return None
            elif operation=='contains' and wanted not in observed:return False
            elif operation=='startswith' and not observed.startswith(wanted):return False
            elif operation=='endswith' and not observed.endswith(wanted):return False
        return True
    evaluate.fields={field for field,_,_ in tests}
    return evaluate

class ConnectorFault:
    def __init__(self, kind, candidates, footprint=None):
        assert kind in ['index_unavailable', 'ignored_filter', 'reply_mixup', 'short_page', 'cursor_expired', 'observe']
        if kind=='reply_mixup' and footprint is None:raise ValueError('Reply replay requires the real observation token counter')
        self.kind = kind; self.candidates = deepcopy(candidates)
        self.footprint=footprint;self.previous={}
        self.caller = None; self.calls = []; self.events = []
        self.bound = None; self.phase = 'initial'; self.matches = 0
        self.pending = None; self.read_restart = False; self.enumeration_recovery = False

    def bind(self, caller, tools):
        if self.caller is not None: raise ValueError('New fault instance required after reset')
        self.caller = caller
        schemas = {t['function']['name']: t['function']['parameters'] for t in tools}
        self.validators = {k: Draft7Validator(v) for k, v in schemas.items()}
        self.exposed_arguments = {k: set(v.get('properties', {})) for k, v in schemas.items()}
        for c in self.candidates:
            schema = schemas[c['tool']]
            if caller.routes[c['tool']] != c['server']: raise ValueError('Route mismatch')
            if self.kind == 'ignored_filter':
                # These are reviewed tool contracts, not authority granted by a label.
                if c.get('read_only') is not True or c.get('predicate_semantics_verified') is not True:
                    raise ValueError('Ignored filter requires reviewed pure-read and native predicate contracts')
                protected = set(c.get('original_required_scope', []))
                if protected & set(c['predicate_parameters']):
                    raise ValueError('Cannot remove protected collection scope')
                guard_keys = protected | {k for group in c.get('required_any_after_drop', []) for k in group}
                if not guard_keys <= set(schema.get('properties', {})):
                    raise ValueError('Unexposed scope guard parameter')
            params = c['predicate_parameters'] if self.kind in ['index_unavailable','ignored_filter'] else [c['page_parameter']]
            if any(k not in schema.get('properties', {}) or k in schema.get('required', []) for k in params):
                raise ValueError('Recovery requires removing a required or nonexistent parameter')
        return self

    async def call_tool(self, name, arguments):
        entry = dict(call_index=len(self.calls)+1, tool=name, server=self.caller.routes.get(name),
                     arguments=deepcopy(arguments), dispatched=False, injected=False)
        valid = self.validators[name].is_valid(arguments) and set(arguments) <= self.exposed_arguments[name]
        candidate = next((c for c in self.candidates if c['tool'] == name), None)
        if self.bound is not None and name != self.bound['tool']: candidate = None
        if self.kind=='ignored_filter' and candidate and valid and self.phase=='initial':
            predicate=optional_predicate(arguments,candidate['predicate_parameters'])
            if predicate and not predicate.fields <= set(candidate['observable_fields']):predicate=None
            if predicate and arguments.get('_select'):
                selected={x.strip() for x in arguments['_select'].split(',')}
                if not predicate.fields <= selected:predicate=None
            dispatched={k:v for k,v in arguments.items() if k not in candidate['predicate_parameters']}
            if not self.validators[name].is_valid(dispatched): predicate=None
            if any(arguments.get(k) in (None, '', []) for k in candidate.get('original_required_scope', [])):
                predicate=None
            if any(not any(dispatched.get(k) not in (None, '', []) for k in group)
                   for group in candidate.get('required_any_after_drop', [])):
                predicate=None
            if predicate:
                entry.update(dispatched=True,dispatched_arguments=deepcopy(dispatched))
                actual=await self.caller.call_tool(name,dispatched);entry['actual_result']=deepcopy(actual)
                value=body(actual)
                key=candidate['collection_field']
                objects=value
                if key != '$':
                    for component in key.split('.'):
                        objects=objects.get(component) if isinstance(objects, dict) else None
                if not isinstance(objects, list) or not all(isinstance(o, dict) for o in objects):
                    objects=[]
                outcomes=[predicate(o) for o in objects]
                if (actual.get('outcome',{}).get('application_ok') is True and
                    not actual.get('is_error') and
                    key==candidate['collection_field'] and outcomes and None not in outcomes and
                    any(x is True for x in outcomes) and any(x is False for x in outcomes)):
                    self.bound=candidate;self.phase='ignored_once';self.matches+=1
                    entry.update(injected=True,committed=False,event_kind='optional_predicate_ignored',
                                 matching_items=sum(x is True for x in outcomes),outside_predicate_items=sum(x is False for x in outcomes),
                                 visible_result=deepcopy(actual),preserved_required_scope={k:arguments[k] for k in candidate.get('original_required_scope',[])})
                    self.events.append(deepcopy(entry))
                else:
                    entry['eligibility_skip']='No proven observable matching and nonmatching objects; no mixed-result fault claim'
                    # Only a reviewed pure read reaches here. Never expose a broadened
                    # query while claiming a clean skip; obtain the original response.
                    entry['probe_result'] = entry.pop('actual_result')
                    entry['probe_arguments'] = entry.pop('dispatched_arguments')
                    actual=await self.caller.call_tool(name,arguments)
                    entry.update(actual_result=deepcopy(actual), visible_result=deepcopy(actual),
                                 dispatched_arguments=deepcopy(arguments), backend_dispatch_count=2,
                                 fallback='original_query_after_unqualified_probe')
                entry.setdefault('backend_dispatch_count', 1)
                self.calls.append(entry);return actual
        if self.kind == 'index_unavailable' and candidate and valid and self.phase != 'recovered':
            filtered = any(arguments.get(k) not in [None, '', []] for k in candidate['predicate_parameters'])
            if filtered:
                self.bound = candidate; self.matches += 1; self.phase = 'index_failed'
                visible = ToolFault.error('503 SearchIndexUnavailable: indexed predicates cannot be evaluated by this connector. The request was not executed. Scoped collection retrieval without indexed predicates remains available.')
                entry.update(injected=True, committed=False, event_kind='indexed_query_rejected', visible_result=deepcopy(visible))
                self.calls.append(entry); self.events.append(deepcopy(entry)); return visible

        # Existing continuation requests are served by the connector snapshot.
        # They do not fetch hidden state or synthesize objects/IDs.
        if self.pending and name == self.bound['tool'] and valid:
            c = self.bound; p = c['page_parameter']; token = arguments.get(p)
            scope = {k: v for k, v in arguments.items() if k not in [p, 'pageSize', 'maxResults', '_top']}
            original = {k: v for k, v in self.pending['arguments'].items() if k not in [p, 'pageSize', 'maxResults', '_top']}
            if token == self.pending['token'] and scope == original:
                self.matches += 1
                if self.kind == 'cursor_expired':
                    visible = ToolFault.error('410 InvalidPageToken: the continuation cursor has expired. Discard the token and restart listing from the first page using the original query and scope.')
                    entry.update(injected=True, committed=False, event_kind='cursor_rejected', visible_result=deepcopy(visible))
                    self.phase = 'cursor_invalid'; self.events.append(deepcopy(entry))
                else:
                    value = deepcopy(self.pending['body']); value[c['collection_field']] = self.pending['remaining']
                    value.pop('nextPageToken', None); value.pop('@odata.nextLink', None)
                    visible = replace_body(self.pending['response'], value)
                    entry.update(actual_result=deepcopy(visible), visible_result=deepcopy(visible),
                                 dispatch_kind='connector_snapshot_continuation', source_call=self.pending['source_call'])
                    self.phase = 'recovered'; self.pending = None
                self.calls.append(entry); return visible
            if not token and scope == original and self.phase == 'cursor_invalid':
                self.read_restart = True; self.phase = 'recovered'; self.pending = None

        entry['dispatched'] = True
        actual = await self.caller.call_tool(name, arguments)
        entry['actual_result'] = deepcopy(actual); visible = actual
        ok = actual.get('outcome', {}).get('application_ok') is True and not actual.get('is_error')
        if self.kind=='reply_mixup' and valid and ok and is_read_name(name):
            server=self.caller.routes[name];old=self.previous.get(server)
            if (self.phase=='initial' and old and (old['tool']!=name or old['arguments']!=arguments)
                and old['response']!=actual):
                cached_tokens=self.footprint(old['response']);normal_tokens=self.footprint(actual)
                if cached_tokens<=normal_tokens:
                    visible=deepcopy(old['response']);self.phase='replayed_once';self.matches+=1
                    entry.update(injected=True,committed=False,event_kind='prior_reply_attached_to_current_read',
                        visible_result=deepcopy(visible),cached_source_call=old['call_index'],cached_source_tool=old['tool'],
                        cached_response_tokens=cached_tokens,normal_response_tokens=normal_tokens)
                    self.events.append(deepcopy(entry))
                else:entry['eligibility_skip']='Cached reply would amplify observation tokens'
            self.previous[server]=dict(call_index=entry['call_index'],tool=name,arguments=deepcopy(arguments),response=deepcopy(actual))
        if self.kind == 'index_unavailable' and self.bound and name == self.bound['tool'] and valid and ok:
            if not any(arguments.get(k) not in [None, '', []] for k in self.bound['predicate_parameters']):
                self.phase = 'recovered'; self.enumeration_recovery = True
        if self.kind in ['short_page', 'cursor_expired'] and candidate and valid and ok and self.phase == 'initial':
            value = body(actual); key, objects = collection(value); head = candidate['first_page_size']
            if key == candidate['collection_field'] and len(objects) > head and not arguments.get(candidate['page_parameter']):
                # Do not overwrite a native continuation that is already needed.
                if not value.get('nextPageToken') and not value.get('@odata.nextLink'):
                    self.bound = candidate; self.matches += 1; self.phase = 'page_issued'
                    token = 1 if candidate['page_parameter'] == '_skip' else 'connector_snapshot_1'
                    self.pending = dict(arguments=deepcopy(arguments), response=deepcopy(actual), body=deepcopy(value),
                                        remaining=deepcopy(objects[head:]), token=token, source_call=entry['call_index'])
                    value = deepcopy(value); value[key] = deepcopy(objects[:head])
                    if 'returned_count' in value: value['returned_count'] = head
                    if candidate['page_parameter'] == 'pageToken': value['nextPageToken'] = token
                    else:
                        endpoint = 'teams' if name == 'list_teams' else 'users'
                        param = '$skiptoken' if candidate['page_parameter'] == '_skiptoken' else '$skip'
                        value['@odata.nextLink'] = 'https://graph.microsoft.com/v1.0/' + endpoint + '?' + urlencode({param: token})
                    visible = replace_body(actual, value)
                    entry.update(injected=True, committed=False, event_kind='short_page_issued', visible_result=deepcopy(visible),
                                 actual_items=len(objects), visible_items=head, preserved_scope=deepcopy(arguments))
                    self.events.append(deepcopy(entry))
        self.calls.append(entry); return visible

    def report(self):
        return dict(enabled=self.kind != 'observe', triggered=bool(self.events), config=dict(kind=self.kind, candidates=self.candidates),
                    matching_calls=self.matches, injected_calls=len(self.events), events=deepcopy(self.events),
                    phase=self.phase, enumeration_recovery=self.enumeration_recovery, cursor_restart=self.read_restart,
                    cursor_rejections=sum(e.get('event_kind') == 'cursor_rejected' for e in self.events))

class ScopedReplyFault(ConnectorFault):
    """Only approved reads can supply history or receive swapped replies."""
    def __init__(self, allowed_tools, footprint):
        super().__init__('reply_mixup', [], footprint)
        self.allowed_tools = frozenset(allowed_tools)

    def bind(self, caller, tools):
        names = {t['function']['name'] for t in tools}
        if not self.allowed_tools <= names:
            raise ValueError('Unexposed approved tool')
        if any(not is_read_name(n) for n in self.allowed_tools):
            raise ValueError('Unsupported reply adapter tool prefix')
        return super().bind(caller, tools)

    async def call_tool(self, name, arguments):
        if name in self.allowed_tools:
            return await super().call_tool(name, arguments)
        actual = await self.caller.call_tool(name, arguments)
        self.calls.append(dict(call_index=len(self.calls)+1, tool=name,
            arguments=deepcopy(arguments), server=self.caller.routes.get(name),
            dispatched=True, injected=False, actual_result=deepcopy(actual),
            eligibility_skip='tool not approved for this fault'))
        return actual
