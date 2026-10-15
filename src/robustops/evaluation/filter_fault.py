"""Filter omission qualified by actual native result sets, not a tool whitelist."""
from copy import deepcopy
from robustops.evaluation.fault_responses import text_body as body
from robustops.evaluation.tool_names import is_read_name
import json
from jsonschema import Draft7Validator
READ=('get_','list_','find_','search_','retrieve_','check_')
PAGING={'limit','offset','page','pageSize','page_size','pageToken','maxResults','_top','_skip','_skiptoken'}
CONTROL=PAGING|{'order_by','order_direction','orderBy','_orderby','_select','fields','projection','format','_count','access_token','token','authorization'}
COUNT={'count','total','total_count','totalCount','resultSizeEstimate','@odata.count'}

def arrays(value,path=()):
 if isinstance(value,list) and value and all(isinstance(x,dict) for x in value):yield path,value
 elif isinstance(value,dict):
  for k,v in value.items():
   if isinstance(v,(dict,list)):yield from arrays(v,path+(k,))
def successful(r):return not r.get('is_error') and r.get('outcome',{}).get('application_ok') is True
def complete(value,path,items,properties):
 parent=value
 for k in path[:-1]:parent=parent[k]
 # Reject continuation indicators anywhere in object wrappers.
 def paged(v):
  return isinstance(v,dict) and (any(v.get(k) for k in ('nextPageToken','@odata.nextLink','next_cursor','incompleteSearch')) or any(paged(x) for x in v.values()))
 if paged(value):return False
 counts=[parent[k] for k in COUNT if isinstance(parent,dict) and type(parent.get(k)) in (int,float)]
 if counts:return all(n==len(items) for n in counts)
 return not PAGING.intersection(properties)

class ObservedFilterFault:
 def __init__(self,tool,parameter):self.tool=tool;self.parameter=parameter;self.calls=[];self.events=[];self.caller=None;self.used=0
 def bind(self,caller,tools):
  if self.caller is not None:raise ValueError('new instance per episode')
  schema=next(t['function']['parameters'] for t in tools if t['function']['name']==self.tool)
  if not is_read_name(self.tool):raise ValueError('not a supported pure read')
  self.properties=set(schema.get('properties',{}));self.validator=Draft7Validator(schema)
  if self.parameter not in self.properties or self.parameter in set(schema.get('required',[]))|CONTROL:raise ValueError('not a removable object predicate')
  self.caller=caller;return self
 async def call_tool(self,name,arguments):
  original=await self.caller.call_tool(name,arguments)
  entry=dict(tool=name,arguments=deepcopy(arguments),dispatched=True,injected=False,actual_result=deepcopy(original),visible_result=deepcopy(original),backend_dispatch_count=1)
  self.calls.append(entry)
  if name!=self.tool or self.used:return original
  if not self.validator.is_valid(arguments) or not set(arguments)<=self.properties or self.parameter not in arguments:return original
  if not successful(original):entry['eligibility_skip']='original query failed';return original
  value=body(original);lists=list(arrays(value))
  if len(lists)!=1:entry['eligibility_skip']='no unique nonempty object collection';return original
  path,items=lists[0]
  if not complete(value,path,items,self.properties):entry['eligibility_skip']='original result completeness unproven';return original
  widened={k:v for k,v in arguments.items() if k!=self.parameter}
  if not self.validator.is_valid(widened):entry['eligibility_skip']='broadened query not schema-valid';return original
  broad=await self.caller.call_tool(name,widened)
  entry.update(probe_arguments=deepcopy(widened),probe_result=deepcopy(broad),backend_dispatch_count=2)
  if not successful(broad):entry['eligibility_skip']='broadened query failed';return original
  expanded=body(broad);b_lists=list(arrays(expanded))
  if len(b_lists)!=1 or b_lists[0][0]!=path:entry['eligibility_skip']='collection shape changed';return original
  _,objects=b_lists[0]
  if not complete(expanded,path,objects,self.properties):entry['eligibility_skip']='broadened completeness unproven';return original
  encode=lambda o:json.dumps(o,sort_keys=True,ensure_ascii=False)
  a={encode(x) for x in items};b={encode(x) for x in objects}
  if len(a)!=len(items) or len(b)!=len(objects) or not a<b:
   entry['eligibility_skip']='not a strict superset of the original complete result';return original
  entry.update(injected=True,event_kind='native_predicate_omission',visible_result=deepcopy(broad),omitted_parameter=self.parameter)
  self.used+=1;self.events.append(deepcopy(entry));return broad
