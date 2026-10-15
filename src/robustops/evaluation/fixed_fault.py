"""Frozen scope-level fault policies; never dispatch a tool more than once."""
from copy import deepcopy
from robustops.evaluation.tool_fault import ToolFault, TOOL_FAULT_KINDS


class ScopeStage:
    def __init__(self, config):
        if config and config['kind'] not in TOOL_FAULT_KINDS:
            raise ValueError('unsupported fault kind')
        self.config=config
        self.bound=None
        self.calls=[]
        self.events=[]
        self.handlers=[]

    def bind(self, caller, tools):
        self.caller=caller;self.routes=caller.routes
        self.observer=ToolFault({'kind':'observe'}).bind(caller,tools)
        for candidate in (self.config or {}).get('candidates',[]):
            c=deepcopy(candidate);c.pop('trigger_positions',None)
            c.update(count=self.config['budget'],match_from=1)
            if c['kind']=='pre_dispatch':c['error_profile']=self.config['error_profile']
            self.handlers.append(ToolFault(c).bind(caller,tools))
        return self

    def accepts(self,name):
        return any(h.config['tool']==name for h in self.handlers)

    @property
    def fault_call_audit(self):
        return self.calls

    async def call_tool(self,name,arguments):
        handler=self.bound
        if handler is None:
            handler=next((h for h in self.handlers if h.eligibility(name,arguments)[0]),self.observer)
        result=await handler.call_tool(name,arguments)
        entry=deepcopy(handler.calls[-1]);entry['fault_kind']=handler.config['kind']
        if entry.get('injected'):
            self.bound=handler;self.events.append(entry)
        elif handler is self.observer and self.handlers:
            entry['scope_not_injected']='no_schema_and_identity_eligible_candidate'
        self.calls.append(entry)
        return result


class FixedPolicyFault:
    def __init__(self, row):
        self.row=deepcopy(row);self.config=row['fault'];self.calls=[];self.events=[];self.exhausted=False
        self.primary=ScopeStage(self.config['primary'])
        self.secondary=ScopeStage(self.config.get('secondary'))

    def bind(self,caller,tools):
        self.caller=caller;self.secondary.bind(caller,tools)
        self.observer=ToolFault({'kind':'observe'}).bind(caller,tools)
        self.primary.bind(caller,tools)
        return self

    async def call_tool(self,name,arguments):
        bound=self.primary.bound
        if (bound and bound.config['tool']==name) or (bound is None and self.primary.accepts(name)):
            stage=self.primary
        elif bound and self.config.get('secondary') and name!=bound.config['tool']:
            stage=self.secondary
        else:stage=self.observer
        result=await stage.call_tool(name,arguments);entry=deepcopy(stage.calls[-1])
        entry['call_index']=len(self.calls)+1;self.calls.append(entry)
        if entry.get('injected'):self.events.append(entry)
        return result

    def report(self):
        n=self.config['planned_injections'];hits=len(self.events)
        return dict(enabled=True,triggered=bool(hits),case_id=self.row['case_id'],config=deepcopy(self.config),
                    planned_injections=n,injected_calls=hits,budget_exhausted=hits>=n,rollout_stop_required=False,
                    events=deepcopy(self.events),call_audit=deepcopy(self.calls))
