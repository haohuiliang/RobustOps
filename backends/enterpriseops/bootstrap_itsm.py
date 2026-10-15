"""Request-owned constraint feedback for the pinned ITSM update managers."""
import inspect
import os

if __package__:
    from .backend_errors import wrap_backend_call
else:
    from backend_errors import wrap_backend_call


def patch_update_errors(manager_class, method_name, table, expected_parameters):
    original = getattr(manager_class, method_name)
    if tuple(inspect.signature(original).parameters) != expected_parameters:
        raise RuntimeError(f'pinned ITSM manager signature differs: {method_name}')
    setattr(manager_class, method_name, wrap_backend_call(
        original, table=table, context_fields=('org_id',), manager_session=True))


def main():
    from database.managers.service_offering_manager import ServiceOfferingManager
    from database.models.service_offering import ServiceOffering
    from database.managers.incident_sla_manager import IncidentSLAManager
    from database.managers.incident_sla_manager import IncidentSLA
    patch_update_errors(ServiceOfferingManager, 'update_service_offering', ServiceOffering.__table__,
                        ('self', 'service_offering_id', 'service_offering_data', 'org_id'))
    patch_update_errors(IncidentSLAManager, 'update_incident_sla_details', IncidentSLA.__table__,
                        ('self', 'org_id', 'incident_sla_id', 'update_data'))
    from database.managers.user_manager import UserManager
    from database.models.user import User
    patch_update_errors(UserManager, 'update_user', User.__table__,
                        ('self', 'user_id', 'user_data', 'org_id'))
    from database.managers.location_manager import LocationManager
    from database.models.location import Location
    patch_update_errors(LocationManager, 'create_location', Location.__table__,
                        ('self', 'location_data'))
    import main as backend
    import uvicorn
    uvicorn.run(backend.app, host='0.0.0.0', port=int(os.environ.get('API_PORT', '8005')))


if __name__ == '__main__':
    main()
