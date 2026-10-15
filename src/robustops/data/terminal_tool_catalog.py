"""Schema-bound contracts for actual cross-domain endpoint name collisions.

The schema catalog contains only tool interfaces from the fixed source pool,
bound through pure-domain provenance. No task text or tool result enters it.
Unknown schema variants remain unmodeled instead of borrowing another backend.
"""
import json
from functools import lru_cache
from pathlib import Path
from .terminal_state import ToolContract

USER=('user_id','user_name','first_name','last_name','email','phone','role','active','location_id','org_id')
GROUP=('group_id','name','type','active','description','email','manager_id')
MEMBER=('member_id','group_id','user_id','org_id')
INCIDENT=('incident_id','number','caller_id','service','service_offering','configuration_item','problem','assigned_to','assignment_group','status','category','short_description','description','worknotes','impact','urgency','priority','resolved','created_at','updated_at')
SERVICE=('service_id','name','owned_by','used_for','status','service_classification','business_criticality','created_on')
OFFERING=('service_offering_id','name','owned_by','parent','used_for','status','service_classification','business_criticality','short_description','created_on')
CASE=('case_id','number','account_id','contact_id','product_id','channel','priority','state','short_description','assigned_to','assignment_group_id','closed_on')
KNOWLEDGE=('knowledge_id','kb_number','title','body','state','visibility','product_id','owner_id')

EXTRA={
 ('itsm','get_user_using_name'):ToolContract('itsm.user','user_id','read',observed_fields=USER),
 ('itsm','list_users'):ToolContract('itsm.user','user_id','read',(('users','*'),),observed_fields=USER),
 ('itsm','get_user_using_email'):ToolContract('itsm.user','user_id','read',observed_fields=USER),
 ('itsm','add_new_user'):ToolContract('itsm.user','user_id','create',observed_fields=USER),
 ('itsm','update_user_details'):ToolContract('itsm.user','user_id','update',target_argument='user_id',observed_fields=USER),
 ('itsm','list_user_groups'):ToolContract('itsm.group','group_id','read',(('*',),('user_groups','*')),observed_fields=GROUP),
 ('itsm','add_new_user_group'):ToolContract('itsm.group','group_id','create',observed_fields=GROUP),
 ('itsm','list_incidents'):ToolContract('itsm.incident','incident_id','read',(('*',),),observed_fields=INCIDENT),
 ('itsm','get_incidents_assigned_to'):ToolContract('itsm.incident','incident_id','read',(('*',),),observed_fields=INCIDENT),
 ('itsm','create_incident'):ToolContract('itsm.incident','incident_id','create',observed_fields=INCIDENT),
 ('itsm','update_incident'):ToolContract('itsm.incident','incident_id','update',target_argument='incident_id',observed_fields=INCIDENT),
 ('itsm','find_services'):ToolContract('itsm.service','service_id','read',(('services','*'),),observed_fields=SERVICE),
 ('itsm','add_new_service'):ToolContract('itsm.service','service_id','create',observed_fields=SERVICE),
 ('itsm','update_service'):ToolContract('itsm.service','service_id','update',target_argument='service_id',observed_fields=SERVICE),
 ('itsm','find_service_offerings'):ToolContract('itsm.offering','service_offering_id','read',(('service_offerings','*'),),observed_fields=OFFERING),
 ('itsm','register_new_service_offering'):ToolContract('itsm.offering','service_offering_id','create',observed_fields=OFFERING),
 ('itsm','update_service_offering'):ToolContract('itsm.offering','service_offering_id','update',target_argument='service_offering_id',observed_fields=OFFERING),
 ('itsm','send_notification'):ToolContract('itsm.notification','notification_id','create',observed_fields=('notification_id','incident_id','email','type','status','subject','message')),
 ('csm','list_users'):ToolContract('csm.user','user_id','read',(('*',),),observed_fields=USER),
 ('csm','add_new_user'):ToolContract('csm.user','user_id','create',observed_fields=USER),
 ('csm','update_user_details'):ToolContract('csm.user','user_id','update',target_argument='user_id',observed_fields=USER),
 ('csm','find_user_group'):ToolContract('csm.group','group_id','read',observed_fields=GROUP),
 ('csm','list_group_members'):ToolContract('csm.member','member_id','read',(('*',),),observed_fields=MEMBER),
 ('csm','add_new_group_member'):ToolContract('csm.member','member_id','create',observed_fields=MEMBER),
 ('csm','search_cases'):ToolContract('csm.case','case_id','read',(('*',),),observed_fields=CASE),
 ('csm','update_case'):ToolContract('csm.case','case_id','update',target_argument='case_id',observed_fields=CASE),
 ('csm','create_knowledge'):ToolContract('csm.knowledge','knowledge_id','create',observed_fields=KNOWLEDGE),
 ('csm','send_notification'):ToolContract('csm.notification','notification_id','create',observed_fields=('notification_id','case_id','knowledge_id','email','type','status')),
 ('hr','list_users'):ToolContract('hr.user','user_id','read',(('users','*'),),observed_fields=USER),
 ('hr','get_user_using_email'):ToolContract('hr.user','user_id','read',(('user',),),observed_fields=USER),
 ('hr','add_new_user'):ToolContract('hr.user','user_id','create',(('user',),),observed_fields=USER),
 ('hr','update_user_details'):ToolContract('hr.user','user_id','update',(('user',),),target_argument='user_id',observed_fields=USER),
 ('hr','find_group_by_name'):ToolContract('hr.group','group_id','read',(('user_group',),),observed_fields=GROUP),
 ('hr','add_new_user_group'):ToolContract('hr.group','group_id','create',(('user_group',),),observed_fields=GROUP),
 ('hr','list_group_members'):ToolContract('hr.member','member_id','read',(('user_group_memberships','*'),),observed_fields=MEMBER),
 ('hr','add_new_group_member'):ToolContract('hr.member','member_id','create',(('user_group_membership',),),observed_fields=MEMBER),
 ('hr','send_notification'):ToolContract('hr.notification','notification_id','create',(('notification',),),observed_fields=('notification_id','hr_case_id','email','type','status')),
 ('teams','list_users'):ToolContract('teams.user','id','read',(('value','*'),),observed_fields=('id','userPrincipalName','displayName','givenName','surname','mail','accountEnabled','jobTitle')),
}

# Every variant here must match an exact known task tool interface.
VARIANT_TOOLS={name for domain,name in EXTRA if sum(n==name for d,n in EXTRA)>1}

def signature(schema):return json.dumps(schema,sort_keys=True,ensure_ascii=False,separators=(',',':'))

@lru_cache(maxsize=1)
def schema_bindings():
    rows=json.loads(Path(__file__).with_name('terminal_tool_schemas.json').read_text())
    result={}
    for row in rows:
        key=signature(row['schema']);binding=(row['domain'],row['schema']['name'])
        if key in result and result[key]!=binding:raise ValueError('ambiguous schema catalog')
        result[key]=binding
    return result

def resolve_contracts(schemas,defaults):
    contracts={k:v for k,v in defaults.items() if k not in VARIANT_TOOLS}
    for (domain,name),contract in EXTRA.items():
        if name not in VARIANT_TOOLS:contracts[name]=contract
    bindings=schema_bindings()
    for item in schemas:
        schema=item.get('function',item);name=schema['name']
        if name not in VARIANT_TOOLS:continue
        binding=bindings.get(signature(schema))
        if binding in EXTRA:contracts[name]=EXTRA[binding]
    return contracts

def build_schema_catalog(records,provenance):
    """Only pure-domain interfaces; mixed-domain tasks cannot relabel tools."""
    domains={r['record_id']:r['domain'] for r in provenance};rows={}
    for record in records:
        domain=domains[record['record_id']]
        if domain=='hybrid':continue
        for item in record['tools']:
            schema=item['function'];name=schema['name']
            if name not in VARIANT_TOOLS or (domain,name) not in EXTRA:continue
            key=signature(schema);row={'domain':domain,'schema':schema}
            if key in rows and rows[key]['domain']!=domain:raise ValueError('same schema belongs to conflicting domains')
            rows[key]=row
    return [rows[k] for k in sorted(rows)]
