"""Runtime qualification for selected commit-receipt loss opportunities.

Reuses ToolFault error semantics and the existing tool_intervention protocol.
No model-generated executable mappings or per-tool capability catalog. This
opt-in module does not alter frozen historical evaluation/scheduling policies.
"""
from copy import deepcopy
from robustops.evaluation.fault_responses import text_body as body
from robustops.evaluation.tool_names import is_read_name, is_getter_name
import json
import re
from jsonschema import Draft7Validator
from robustops.evaluation.tool_fault import ToolFault

READ=('get_','list_','find_','search_','retrieve_','check_')
PAGING={'limit','offset','page','pageSize','page_size','pageToken','maxResults','_top','_skip','_skiptoken'}
COUNT={'count','total','total_count','totalCount','resultSizeEstimate','@odata.count'}


def successful(r):return not r.get('is_error') and r.get('outcome',{}).get('application_ok') is True

MISSING=object()
def get(value,path):
 if path=='$':return value
 for part in path.split('.'):
  if not isinstance(value,dict) or part not in value:return MISSING
  value=value[part]
 return value

def contains(expected,value):
 if isinstance(expected,dict):return isinstance(value,dict) and all(k in value and contains(v,value[k]) for k,v in expected.items())
 return expected==value

def exhausted(value,path,objects,properties,schema=None):
 # Recognized explicit terminal cursor or exact native count proves this page complete.
 parent=value
 for part in path.split('.')[:-1] if path!='$' else []:
  parent=parent.get(part,{}) if isinstance(parent,dict) else {}
 if isinstance(parent,dict):
  cursor_keys=('nextPageToken','@odata.nextLink','next_cursor')
  if any(parent.get(k) for k in cursor_keys) or parent.get('incompleteSearch'):return False
  counts=[parent[k] for k in COUNT if type(parent.get(k)) in (int,float)]
  if counts:return all(n==len(objects) for n in counts)
  if any(k in parent for k in cursor_keys):return True
 # Native APIs may omit the terminal cursor. Accept that only where the
 # exposed pagination parameter explicitly documents that output cursor.
 if schema and isinstance(parent,dict):
  descriptions=' '.join(v.get('description','') for v in schema.get('properties',{}).values())
  if 'nextPageToken' in descriptions and 'nextPageToken' not in parent:return True
 return not (PAGING & set(properties))

def id_field(key):
 return key=='id' or key.endswith('_id') or key.endswith('Id')

def dict_nodes(value,path=()):
 if not isinstance(value,dict):return
 yield path,value
 for key,child in value.items():
  if isinstance(child,dict):yield from dict_nodes(child,path+(key,))

def at_path(value,path):
 for key in path:
  try:value=value[key]
  except (KeyError,IndexError,TypeError):return MISSING
 return value

def ids(record):
 keys=[k for k,v in record.items() if id_field(k) and type(v) in (str,int)]
 return ['id'] if 'id' in keys else keys

def same(a,b):return type(a) is type(b) and a==b

def terminal(value):
 return isinstance(value,dict) and not any(value.get(k) for k in ('nextPageToken','@odata.nextLink','next_cursor','incompleteSearch','has_more','hasMore'))

def public_records(value,properties,description,path=(),parents=(),complete=True,schema=None):
 """Walk fully returned object collections, retaining public ancestor identities."""
 # Some native tools return the collection itself, without an object envelope.
 # No cursor/count can be inferred from a bare array: paginated schemas stay
 # unqualified. Apply the same nonpaginated completeness rule as wrapped lists.
 if isinstance(value,list):
  if complete and not (PAGING & set(properties)) and all(isinstance(x,dict) for x in value):
   for i,obj in enumerate(value):
    yield from public_records(obj,properties,description,path+(i,),parents,True,schema)
  return
 if not isinstance(value,dict):return
 if ids(value) and complete:yield path,value,parents
 for key,child in value.items():
  if isinstance(child,dict):
   yield from public_records(child,properties,description,path+(key,),parents+((value,) if ids(value) else ()),complete and terminal(value),schema)
  elif isinstance(child,list) and all(isinstance(x,dict) for x in child):
   if parents or ids(value):
    # Nested arrays must be named by the public all-records read description;
    # continuation/count evidence at this child boundary still takes precedence.
    named=bool(re.search(r'\ball\b',description,re.I) and re.search(r'\b'+re.escape(key)+r'\b',description,re.I))
    counts=[value[k] for k in COUNT if type(value.get(k)) in (int,float)]
    full=named and terminal(value) and (not counts or all(c==len(child) for c in counts))
   else:
    full=exhausted(value,key,child,properties,schema) and terminal(value)
   for i,obj in enumerate(child):
    yield from public_records(obj,properties,description,path+(key,i),parents+((value,) if ids(value) else ()),complete and full,schema)

class DirectCommitReadbackFault:
 """Original labels -> runtime evidence; no capability catalog or LLM recipes.
 Qualification reads are injector work, not solver recovery. Unsupported or
 unobserved routes pass through the original result without consuming budget.
 """
 def __init__(self,tool,fault,recovery_tools):
  self.tool=tool;self.fault=fault;self.recovery_tools=list(recovery_tools);self.caller=None;self.calls=[];self.events=[];self.used=0
 def bind(self,caller,tools):
  if self.caller is not None:raise ValueError('new instance per episode')
  if self.fault not in ('unknown_commit','unknown_commit_create'):raise ValueError('wrong fault')
  self.schemas={t['function']['name']:t['function']['parameters'] for t in tools}
  self.descriptions={t['function']['name']:t['function'].get('description','') for t in tools}
  if self.tool not in self.schemas:raise ValueError('unexposed write')
  self.invalid_recovery_tools=[n for n in self.recovery_tools if n not in self.schemas or not is_read_name(n) or caller.routes.get(n)!=caller.routes[self.tool]]
  self.caller=caller;self.routes=caller.routes;return self
 def report(self):
  return dict(enabled=True,triggered=bool(self.events),injected_calls=self.used,
              matching_calls=sum(c['tool']==self.tool for c in self.calls),
              config=dict(kind='post_commit_response_loss',tool=self.tool,server=self.routes[self.tool],count=1,candidate_label=self.fault),
              qualification_read_calls=sum(c['backend_dispatch_count']-1 for c in self.calls),events=deepcopy(self.events))

 @property
 def fault_call_audit(self):return self.calls

 def queries(self,arguments):
  for read in self.recovery_tools:
   schema=self.schemas[read]
   query={k:deepcopy(v) for k,v in arguments.items() if k in schema.get('properties',{})}
   if Draft7Validator(schema).is_valid(query):yield read,query
 async def probe(self,e,read,query,stage='after'):
  e['backend_dispatch_count']+=1
  item=dict(tool=read,arguments=deepcopy(query),stage=stage);e['probes'].append(item)
  try:item['result']=await self.caller.call_tool(read,query)
  except Exception as exc:item['error']=dict(type=type(exc).__name__,message=str(exc));return None
  return item['result']
 def emit(self,e,route,**qualification):
  visible=ToolFault.error('Timeout waiting for the tool response. Execution status is unknown; verify the intended operation before repeating it.')
  self.used+=1;e.update(injected=True,committed=True,event_kind='commit_receipt_lost_'+route,visible_result=deepcopy(visible),qualification=dict(route=route,**qualification));self.events.append(deepcopy(e));return visible
 def valid(self,name,args):
  schema=self.schemas[name]
  return set(args)<=set(schema.get('properties',{})) and Draft7Validator(schema).is_valid(args)
 async def deletion_before(self,e,args):
  witnesses=[]
  for read,query in self.queries(args):
   remaining=set(args)-set(query)
   # A full listing keeps parent/auth scope and leaves precisely the target ID.
   target_keys=[k for k in remaining if id_field(k)]
   getter=not remaining and is_getter_name(read)
   if getter:target_keys=[k for k in query if id_field(k)]
   if not target_keys or (remaining and set(target_keys)!=remaining):continue
   result=await self.probe(e,read,query,'before')
   if not result or not successful(result):continue
   value=body(result);matches=[]
   for path,obj,parents in public_records(value,self.schemas[read].get('properties',{}),self.descriptions[read],schema=self.schemas[read]):
    for field in ids(obj):
     for key in target_keys:
      if same(obj[field],args[key]):matches.append(dict(read=read,query=query,path=path,field=field,identity=args[key],source=key,kind=obj.get('kind'),getter=getter,prior=deepcopy(obj)))
   if len(matches)==1:witnesses.append(matches[0])
  return witnesses
 async def deletion_after(self,e,witnesses):
  for w in witnesses:
   observed=await self.probe(e,w['read'],w['query'])
   if not observed:continue
   value=body(observed)
   if w['getter']:
    codes=[observed.get('outcome',{}).get('status_code')]+[d.get(k) for _,d in dict_nodes(value) for k in ('status_code','code')]
    absent=not successful(observed) and any(c in (404,'404') for c in codes)
   else:
    # Same public collection path, complete, and target no longer present.
    if not w['path'] or not isinstance(w['path'][-1],int):continue
    path=w['path'][:-1];objects=at_path(value,path);parent=at_path(value,path[:-1])
    absent=successful(observed) and isinstance(objects,list) and all(isinstance(o,dict) for o in objects) and isinstance(parent,dict) and exhausted(parent,path[-1],objects,self.schemas[w['read']].get('properties',{}),self.schemas[w['read']]) and terminal(parent) and not any(same(o.get(w['field']),w['identity']) for o in objects)
   if absent:return self.emit(e,'verified_deletion',read_tool=w['read'],read_arguments=w['query'],identity_field=w['field'],known_identity_source=w['source'],preexisting=True,public_absence_confirmed=True)
   # Some native delete endpoints leave a public tombstone instead of
   # removing the record. Require an observed false -> true transition.
   if successful(observed):
    matches=[obj for _,obj,_ in public_records(value,self.schemas[w['read']].get('properties',{}),self.descriptions[w['read']],schema=self.schemas[w['read']]) if same(obj.get(w['field']),w['identity']) and (not w['kind'] or obj.get('kind')==w['kind'])]
    if len(matches)==1:
     changed=[k for k in ('deleted','trashed','is_deleted','isDeleted') if w['prior'].get(k) is False and matches[0].get(k) is True]
     if changed:return self.emit(e,'verified_deletion_tombstone',read_tool=w['read'],read_arguments=w['query'],identity_field=w['field'],known_identity_source=w['source'],preexisting=True,changed_deletion_flags=changed,public_absence_confirmed=False)

  return None
 async def singleton(self,e,args,receipt):
  # Same named settings/configuration endpoint and original scope, without an ID.
  nouns={'settings','preferences','configuration'}
  write_words=set(re.findall(r'\w+',self.descriptions[self.tool].lower()))
  if not nouns & write_words:return None
  for read,query in self.queries(args):
   if not query or not is_getter_name(read):continue
   if not nouns & write_words & set(re.findall(r'\w+',self.descriptions[read].lower())):continue
   expected={k:v for k,v in args.items() if k not in query}
   if not expected:continue
   write_matches=[(path,obj) for path,obj in dict_nodes(receipt) if not ids(obj) and all(k in obj and contains(v,obj[k]) for k,v in expected.items())]
   if len(write_matches)!=1:continue
   observed=await self.probe(e,read,query)
   if not observed or not successful(observed):continue
   matches=[(path,obj) for path,obj in dict_nodes(body(observed)) if not ids(obj) and all(k in obj and contains(v,obj[k]) for k,v in expected.items())]
   if len(matches)==1:return self.emit(e,'scoped_singleton_readback',read_tool=read,read_arguments=query,checked_effects=list(expected),object_path=matches[0][0])
  return None
 async def multi_id_creation(self,e,args,value):
  # Preserve the entire native identity tuple instead of guessing which of
  # several IDs is primary. Public recovery must uniquely rediscover the same
  # record, with every identity component and requested effect intact.
  nodes=[(path,obj) for path,obj in dict_nodes(value) if len(ids(obj))>1]
  if not nodes:return None
  depth=min(len(path) for path,_ in nodes);nodes=[n for n in nodes if len(n[0])==depth]
  if len(nodes)!=1:return None
  receipt_path,receipt=nodes[0];fields=ids(receipt)
  if not any(not any(same(receipt[k],v) for v in args.values()) for k in fields):return None
  for read,query in self.queries(args):
   scope={k for k in query if k not in receipt}
   effects={k:v for k,v in args.items() if k not in scope}
   if not effects or any(k not in receipt or not contains(v,receipt[k]) for k,v in effects.items()):continue
   observed=await self.probe(e,read,query)
   if not observed or not successful(observed):continue
   matches=[(path,obj) for path,obj,_ in public_records(body(observed),self.schemas[read].get('properties',{}),self.descriptions[read],schema=self.schemas[read]) if all(k in obj and contains(v,obj[k]) for k,v in effects.items())]
   if len(matches)!=1:continue
   path,obj=matches[0]
   if not all(same(obj.get(k),receipt[k]) for k in fields):continue
   if receipt.get('kind') and obj.get('kind')!=receipt['kind']:continue
   return self.emit(e,'composite_identity_public_readback',read_tool=read,read_arguments=query,receipt_path=receipt_path,object_path=path,identity_fields=fields,checked_effects=list(effects))
  return None

 async def call_tool(self,name,arguments):
  e=dict(call_index=len(self.calls)+1,tool=name,server=self.routes.get(name),fault_kind='post_commit_response_loss',arguments=deepcopy(arguments),dispatched=True,injected=False,backend_dispatch_count=1,probes=[])
  selected=name==self.tool and not self.used and not self.invalid_recovery_tools and self.valid(name,arguments)
  deletion=selected and self.fault=='unknown_commit' and name.startswith('delete_') and bool(re.search(r'\bdeletes?\b',self.descriptions[name],re.I))
  before=await self.deletion_before(e,arguments) if deletion else []
  actual=await self.caller.call_tool(name,arguments)
  e.update(actual_result=deepcopy(actual),visible_result=deepcopy(actual));self.calls.append(e)
  if name!=self.tool or self.used:return actual
  def skip(why):e['eligibility_skip']=why;return actual
  if self.invalid_recovery_tools:
   e['rejected_recovery_tools']=list(self.invalid_recovery_tools)
   return skip('unexposed/nonread/cross-server recovery tool')
  if not selected:return skip('invalid/unexposed original arguments')
  if not successful(actual):return skip('write did not succeed')
  if deletion:
   result=await self.deletion_after(e,before)
   return result if result is not None else skip('deletion lacks same-query prior existence and confirmed absence')
  schema=self.schemas[name];value=body(actual)
  if self.fault=='unknown_commit_create':
   composite=await self.multi_id_creation(e,arguments,value)
   if composite is not None:return composite
  if self.fault=='unknown_commit':
   singleton=await self.singleton(e,arguments,value)
   if singleton is not None:return singleton
  records=[]
  for path,record in dict_nodes(value):
   for field in ids(record):
    known=[k for k,v in arguments.items() if same(v,record[field])]
    if self.fault=='unknown_commit' and len(known)==1:records.append((path,record,field,known))
    elif self.fault=='unknown_commit_create':records.append((path,record,field,known))
  # A required target-ID argument can distinguish the updated resource from
  # optional foreign-key fields (owner_id, assignment_group_id, etc.). Do not
  # choose when two required identities remain, as in relation operations.
  if self.fault=='unknown_commit' and len(records)>1:
   target_records=[r for r in records if r[3][0] in schema.get('required',[])
                   and id_field(r[3][0]) and r[2] in ('id',r[3][0])]
   if len(target_records)==1:records=target_records
  # Prefer shallow resource identity over nested relation IDs.
  if records:
   depth=min(len(r[0]) for r in records);records=[r for r in records if len(r[0])==depth]
  if len(records)!=1:return skip('no unique native committed resource identity')
  receipt_path,receipt,identity_field,known=records[0];identity=receipt[identity_field]
  context={k for k in arguments if k not in receipt and re.search(r'authenticated user',schema.get('properties',{}).get(k,{}).get('description',''),re.I)}
  keys=[k for k,v in schema.get('properties',{}).items() if re.search(r'\bidempoten\w*\b',v.get('description',''),re.I) and isinstance(arguments.get(k),str) and arguments[k].strip()]
  if self.fault=='unknown_commit_create' and len(keys)==1 and known==keys:
   effects={k:v for k,v in arguments.items() if k not in set(keys)|context}
   if effects and all(k in receipt and contains(v,receipt[k]) for k,v in effects.items()):
    return self.emit(e,'documented_idempotency_key',key_argument=keys[0],identity_field=identity_field,identity_equals_original_key=True,checked_effects=list(effects),basis=schema['properties'][keys[0]]['description'],replay_outcome='backend may confirm existing key with conflict rather than success')
  if self.fault=='unknown_commit_create' and known:return skip('creation has known identity without supported idempotency-key route')
  description=self.descriptions[name]
  declared_replace=bool(re.search(r'\breplaces?\b',description,re.I)) and not re.search(r'\b(send|sends|append|appends|increment|increments|notify|notifies|notification|create|creates)\b',description,re.I)
  if self.fault=='unknown_commit' and known and declared_replace:
   effects={k:v for k,v in arguments.items() if k not in set(known)|context}
   if effects and all(k in receipt and contains(v,receipt[k]) for k,v in effects.items()):return self.emit(e,'declared_exact_replacement',exposed_description=description,identity_field=identity_field,known_identity_source=known[0],checked_effects=list(effects),preserved_context_arguments=sorted(context))
  for read,query in self.queries(arguments):
   # Only non-returned read scope can be covered by the query itself.
   scope={k for k in query if k not in receipt}|set(known)
   missing={k for k in arguments if k not in scope and k not in receipt}
   if any(not id_field(k) for k in missing):continue
   effects={k:v for k,v in arguments.items() if k not in scope|missing}
   if not effects or any(not contains(v,receipt[k]) for k,v in effects.items()):continue
   observed=await self.probe(e,read,query)
   if not observed or not successful(observed):continue
   candidates=[]
   for path,obj,parents in public_records(body(observed),self.schemas[read].get('properties',{}),self.descriptions[read],schema=self.schemas[read]):
    if not all(any(same(parent.get(f),arguments[k]) for parent in parents for f in ids(parent)) for k in missing):continue
    if known and not same(obj.get(identity_field),arguments[known[0]]):continue
    if all(k in obj and contains(v,obj[k]) for k,v in effects.items()):candidates.append((path,obj,parents))
   if len(candidates)!=1:continue
   path,obj,parents=candidates[0]
   if not same(obj.get(identity_field),identity) or (receipt.get('kind') and obj.get('kind')!=receipt['kind']):continue
   route='nested_public_readback' if missing else 'direct_public_readback'
   return self.emit(e,route,read_tool=read,read_arguments=query,receipt_path=receipt_path,identity_field=identity_field,object_path=path,checked_effects=list(effects),parent_identity_sources=sorted(missing),known_identity_source=known[0] if known else None)
  return skip('no directly executable complete recovery witness; replay/transform routes not certified')


def make_selected_write_fault(payload, selection, tool, fault):
    """Current no-catalog write path: labels -> actual-call qualification.

    No model-generated field recipes or old compile_labels contracts are used.
    Older label outputs without recovery_tools can use relevant exposed reads;
    their names alone never license injection. Explicit [] stays empty.
    """
    entries = selection['tools'] if isinstance(selection, dict) else selection
    entry = next(t for t in entries if t['tool'] == tool)
    if fault not in entry['faults'] or fault not in {'unknown_commit', 'unknown_commit_create'}:
        raise ValueError('write mechanism not selected')
    reads = entry.get('recovery_tools')
    if reads is None:
        reads = [t['tool'] for t in entries if t['faults'] and is_read_name(t['tool'])
                 and payload['tool_servers'][t['tool']] == payload['tool_servers'][tool]]
    return DirectCommitReadbackFault(tool, fault, reads)
