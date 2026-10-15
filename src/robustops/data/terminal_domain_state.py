"""Explicit cross-domain resource contracts and query-bound read receipts.

Contracts are based on the visible local tool schemas and recorded payloads.
A query receipt describes that call's returned rows, never global completeness.
The original Drive/ITSM development registry remains unchanged for replay.
"""
import re
from .terminal_evidence import atom, Number
from .terminal_state import CONTRACTS, StateLedger, ToolContract, _payloads, _rows
from .terminal_tool_catalog import resolve_contracts

# Calendar metadata and the user's display settings are different resources.
CALENDAR_FIELDS=('summary','description','location','timeZone','conferenceProperties')
CALENDAR_LIST_FIELDS=('summaryOverride','colorId','backgroundColor','foregroundColor','hidden','selected','accessRole','defaultReminders','notificationSettings','primary','deleted')
HR_CASE_FIELDS=('hr_case_id','number','status','priority','source','short_description','worknotes','skills','coe_type','request_status','account_number','account_type','org_id','created_at','updated_at')
HR_TASK_FIELDS=('hr_case_task_id','number','status','short_description','task_type','external_url','worknotes','active','org_id','created_at','updated_at')
EVENT_FIELDS=('summary','description','status','location','colorId','start','end','attendees','attachments','reminders','recurrence','conferenceData','visibility','extendedProperties','source','guestsCanInviteOthers','guestsCanModify','guestsCanSeeOtherGuests','eventType')
DOMAIN_CONTRACTS={**CONTRACTS,
    'get_calendar_list':ToolContract('calendar.list_entry','id','read',(('items','*'),),observed_fields=CALENDAR_LIST_FIELDS),
    'get_calendar_from_list':ToolContract('calendar.list_entry','id','read',target_argument='calendarId',observed_fields=CALENDAR_LIST_FIELDS),
    'add_calendar_to_list':ToolContract('calendar.list_entry','id','create',observed_fields=CALENDAR_LIST_FIELDS),
    'get_calendar':ToolContract('calendar.calendar','id','read',target_argument='calendarId',observed_fields=CALENDAR_FIELDS),
    'create_calendar':ToolContract('calendar.calendar','id','create',observed_fields=CALENDAR_FIELDS),
    'patch_calendar':ToolContract('calendar.calendar','id','update',target_argument='calendarId',observed_fields=CALENDAR_FIELDS),
    'list_events':ToolContract('calendar.event','id','read',(('items','*'),),scope_arguments=('calendarId',),observed_fields=EVENT_FIELDS),
    'get_event':ToolContract('calendar.event','id','read',scope_arguments=('calendarId',),target_argument='eventId',observed_fields=EVENT_FIELDS),
    'create_event':ToolContract('calendar.event','id','create',scope_arguments=('calendarId',),observed_fields=EVENT_FIELDS),
    'delete_event':ToolContract('calendar.event','id','delete',scope_arguments=('calendarId',),target_argument='eventId'),
    'list_acl_rules':ToolContract('calendar.acl','id','read',(('items','*'),),scope_arguments=('calendarId',)),
    'delete_acl_rule':ToolContract('calendar.acl','id','delete',scope_arguments=('calendarId',),target_argument='ruleId'),
    'find_user':ToolContract('csm.user','user_id','read'),
    'find_user_by_email':ToolContract('csm.user','user_id','read'),
    'find_contact_by_portal_user':ToolContract('csm.contact','contact_id','read'),
    'find_contacts':ToolContract('csm.contact','contact_id','read',(('*',),)),
    'find_account':ToolContract('csm.account','account_id','read'),
    'find_product':ToolContract('csm.product','product_id','read'),
    'find_product_by_id':ToolContract('csm.product','product_id','read',target_argument='product_id'),
    'find_entitlements':ToolContract('csm.entitlement','entitlement_id','read',(('*',),)),
    'retrieve_knowledge':ToolContract('csm.knowledge','knowledge_id','read',(('*',),)),
    'find_sla_definitions':ToolContract('csm.sla','sla_def_id','read',(('*',),)),
    'create_new_case':ToolContract('csm.case','case_id','create'),
    'link_case_knowledge':ToolContract('csm.case_knowledge','case_kb_id','create'),
    'list_labels':ToolContract('email.label','id','read',(('labels','*'),),scope_arguments=('userId',)),
    'list_messages':ToolContract('email.message','id','read',(('messages','*'),),scope_arguments=('userId',)),
    'list_cse_identities':ToolContract('email.cse_identity','emailAddress','read',(('cseIdentities','*'),),scope_arguments=('userId',)),
    'delete_cse_identity':ToolContract('email.cse_identity','emailAddress','delete',scope_arguments=('userId',),target_argument='cseEmailAddress'),
    'list_hr_cases':ToolContract('hr.case','hr_case_id','read',(('hr_cases','*'),),observed_fields=HR_CASE_FIELDS),
    'get_cases_assigned_to':ToolContract('hr.case','hr_case_id','read',(('cases','*'),),observed_fields=HR_CASE_FIELDS),
    'find_hr_services':ToolContract('hr.service','hr_service_id','read',(('hr_services','*'),)),
    'list_hr_case_tasks':ToolContract('hr.case_task','hr_case_task_id','read',(('hr_case_tasks','*'),),observed_fields=HR_TASK_FIELDS),
    'update_hr_case_task':ToolContract('hr.case_task','hr_case_task_id','update',target_argument='hr_case_task_id',observed_fields=HR_TASK_FIELDS),
    # Single-ID updates; filtered/batch variants deliberately remain unresolved.
    'update_hr_case':ToolContract('hr.case','hr_case_id','update',( ('case',), ),target_argument='hr_case_id',observed_fields=HR_CASE_FIELDS),
    'create_problem':ToolContract('itsm.problem','problem_id','create'),
    'create_user':ToolContract('teams.user','id','create',observed_fields=('userPrincipalName','displayName','accountEnabled','jobTitle','mail','mailNickname','department','companyName','userType')),
    'list_teams':ToolContract('teams.team','id','read',(('value','*'),),observed_fields=('displayName','description','visibility','isArchived','webUrl')),
    'list_channels':ToolContract('teams.channel','id','read',(('value','*'),),scope_arguments=('teamId',),observed_fields=('displayName','description','membershipType','isArchived')),
    'add_channel_member':ToolContract('teams.channel_member','id','create',scope_arguments=('teamId','channelId'),observed_fields=('userId','displayName','email','roles')),
    'create_teamwork_tag':ToolContract('teams.tag','id','create',scope_arguments=('team_id',),observed_fields=('displayName','description','tagType','teamId','memberCount','members')),
}


class DomainLedger(StateLedger):
    """HR tools transport integer database IDs as decimal request strings.

    This conversion is restricted to the two declared HR resource types; it is
    not an equality rule for arbitrary numeric-looking IDs in other domains.
    Raw request values remain in the evidence packet and query receipts.
    """
    def __init__(self,packet):
        self.transport_normalizations=[]
        super().__init__(packet,resolve_contracts(packet['used_tool_schemas'],DOMAIN_CONTRACTS))

    def identity(self,resource,value,scope=()):
        if resource in {'hr.case','hr.case_task'} and type(value) is str and re.fullmatch(r'0|[1-9][0-9]*',value):
            self.transport_normalizations.append({'resource':resource,'raw':atom(value),
                'canonical':atom(Number(value)),
                'contract':'HR integer database IDs encoded as decimal strings in tool arguments'})
            value=Number(value)
        return super().identity(resource,value,scope)


def query_receipts(ledger):
    """Keep actual query scope, raw rows and pagination evidence together."""
    index=ledger.index;receipts=[]
    for event in index.packet['evidence']:
        contract=ledger.contracts.get(event['tool'])
        if contract is None or contract.effect!='read':continue
        eid=event['id'];rows=[];pagination=[];containers=[]
        for location,payload in _payloads(index,eid):
            containers.append(location)
            for path in contract.row_paths:
                for row_location,row in _rows(payload,location,path):
                    identity=row.get(contract.identity_field)
                    if identity is None or type(identity) in (dict,list,bool):continue
                    refs=[ref for ref,node in index.nodes.items() if node['event_id']==eid and node['role']=='observation' and node['location']==row_location]
                    rows.append({'identity':atom(identity),'location':row_location,'refs':refs})
            if isinstance(payload,dict):
                for key in ('nextPageToken','nextLink','@odata.nextLink','total_count','returned_count','resultSizeEstimate'):
                    if key in payload:pagination.append({'field':key,'value':atom(payload[key]),'location':[*location,key]})
        receipts.append({'event_id':eid,'tool':event['tool'],'resource':contract.resource,
                         'response_status':index.events[eid]['response_status'],
                         'arguments':event['arguments'],'observation_ref':index.events[eid]['observation']['ref'],
                         'payload_locations':containers,'rows':rows,'pagination':pagination,
                         'scope':'Only rows returned by this exact call. Filters and pagination remain part of the result; global completeness not established.'})
    return receipts


def project(packet):
    ledger=DomainLedger(packet)
    def raw(value):return Number(value['value']) if value['type']=='number' else value['value']
    entities=[]
    for entity in ledger.entities.values():
        identity=entity['identity']
        checks={key:ledger.check(identity['resource'],raw(identity['id']),fact['path'],fact['value'],
                                scope=tuple((k,raw(v)) for k,v in identity['scope'].items()))
                for key,fact in entity['facts'].items()}
        entities.append({**entity,'final_field_checks':checks})
    return {'entities':entities,'history':ledger.history,
            'queries':query_receipts(ledger),'unmodeled_events':ledger.unmodeled_events,
            'transport_normalizations':ledger.transport_normalizations,
            'whole_request_coverage_proven':False,'training_recipe_approved':False}
