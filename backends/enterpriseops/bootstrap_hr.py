"""Repair KB allocation, case-number schemas and survey conflicts in pinned HR."""
import os
from functools import wraps
if __package__:
    from .backend_errors import patch_route_errors
else:
    from backend_errors import patch_route_errors


def patch_case_number_schema(case_tools):
    targets = {"find_hr_case_by_number", "list_hr_cases", "update_hr_case"}
    selected = [tool for tool in case_tools if tool.name in targets]
    if len(selected) != len(targets):
        raise RuntimeError("HR case-number tools differ from the pinned catalog")
    for tool in selected:
        number = tool.inputSchema["properties"]["number"]
        if (number.get("pattern") not in (r"^HRC\d{7}$", r"^HRC\\d{7}$")
                or number.get("minLength") != 10 or number.get("maxLength") != 10):
            raise RuntimeError("HR case-number schema differs from the pinned catalog")
    for tool in selected:
        number = tool.inputSchema["properties"]["number"]
        # The API and manager perform an exact string lookup. Seed numbers can
        # contain a hyphen; removing it or padding digits changes the identity.
        for key in ("pattern", "minLength", "maxLength"):
            number.pop(key)
        number["description"] = (
            "Exact case number as supplied by the user or returned by the API. "
            "Preserve all characters, including hyphens and zero padding; "
            "do not reformat it. Seed and generated cases may use different formats."
        )


def patch_kb_number_allocation(manager_class, knowledge_model):
    original = manager_class._generate_next_kb_number

    @wraps(original)
    def allocate(self, org_id):
        candidate = original(self, org_id)
        # Preserve the pinned allocator's normal result and database-wide scope.
        occupied = {row[0] for row in self.session.query(knowledge_model.kb_number).all()}
        if candidate not in occupied:
            return candidate
        prefix, number = candidate.rsplit('-', 1)
        next_number = int(number)
        while candidate in occupied:
            next_number += 1
            candidate = f'{prefix}-{next_number:03d}'
        return candidate

    manager_class._generate_next_kb_number = allocate


def patch_survey_instance_errors(module, table):
    """The native manager owns rollback/close; share typed request rejection mapping."""
    patch_route_errors(module, 'create_survey_instance', table=table, context_fields=('org_id',))


def main():
    from database.managers.knowledge_manager import KnowledgeManager, Knowledge
    patch_kb_number_allocation(KnowledgeManager, Knowledge)
    from tools.case_tools import CASE_TOOLS
    patch_case_number_schema(CASE_TOOLS)
    from apis.survey_instances import router as survey_router
    from database.models.survey_instance import SurveyInstance
    patch_survey_instance_errors(survey_router, SurveyInstance.__table__)
    import main as backend
    import uvicorn
    uvicorn.run(backend.app, host='0.0.0.0', port=int(os.environ.get('API_PORT', '8005')))


if __name__ == '__main__':
    main()
