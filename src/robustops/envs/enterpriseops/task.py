"""Load and validate the fixed EnterpriseOps task rows used by this project."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Iterable, Mapping
from zipfile import ZipFile

import pyarrow.parquet as pq

from robustops.envs.enterpriseops.revised_overlay import (
    OfficialRevisedOverlay,
    OfficialRevisedOverlayError,
    apply_official_revised_overlay,
    official_revised_supplement_rows,
)
from robustops.envs.enterpriseops.verifier_contract import (
    VerifierContractCorrection,
    VerifierContractSourceError,
    VerifierSetNormalization,
    apply_verifier_contract,
)


class TaskContractError(ValueError):
    """The local task data does not satisfy the registered experiment contract."""


_HYBRID_LOCKDOWN_QUERY = (
    "SELECT \n"
    "    (SELECT COUNT(user_id) FROM users \n"
    "     WHERE email = 'christina.oliver@carsonllc.com' AND active = FALSE)\n"
    "    +\n"
    "    (SELECT COUNT(incident_id) FROM incident i\n"
    "     JOIN users u ON i.caller_id = u.user_id\n"
    "     WHERE u.email = 'christina.oliver@carsonllc.com' \n"
    "       AND i.short_description = 'Security Breach: Account Compromised'\n"
    "       AND i.priority = 'critical');"
)

_DRIVE_BAD_COMPLIANCE_PERMISSION_QUERY = (
    "WITH new_folder AS (SELECT id FROM files WHERE name = 'Audit Prep Q4' "
    "AND trashed = 0)\n"
    "SELECT COUNT(*) FROM permissions p\n"
    "JOIN new_folder nf ON p.file_id = nf.id\n"
    "WHERE p.email_address = 'compliance@catalystcorp.com'\n"
    "  AND p.role = 'reader'\n"
    "  AND p.deleted = 0;"
)

_DRIVE_FIXED_COMPLIANCE_PERMISSION_QUERY = (
    "WITH new_folder AS (SELECT id FROM files WHERE name = 'Audit Prep Q4' "
    "AND trashed = 0)\n"
    "SELECT COUNT(*) FROM permissions p\n"
    "JOIN new_folder nf ON p.file_id = nf.id\n"
    "JOIN groups g ON p.group_id = g.id\n"
    "WHERE g.email = 'compliance@catalystcorp.com'\n"
    "  AND p.role = 'reader'\n"
    "  AND p.type = 'group'\n"
    "  AND p.deleted = 0;"
)

_DRIVE_BAD_ENGINEERING_ORIGINAL_PERMISSION_QUERY = (
    "SELECT COUNT(*) as group_permission_count\n"
    "FROM permissions\n"
    "WHERE file_id = '01JBQM3K8TGHZX9RNVQM5Y2P8R'\n"
    "  AND type = 'group'\n"
    "  AND email_address = 'engineering@company.com'\n"
    "  AND deleted = false;"
)

_DRIVE_FIXED_ENGINEERING_ORIGINAL_PERMISSION_QUERY = (
    "SELECT COUNT(*) as group_permission_count\n"
    "FROM permissions p\n"
    "JOIN groups g ON p.group_id = g.id\n"
    "WHERE p.file_id = '01JBQM3K8TGHZX9RNVQM5Y2P8R'\n"
    "  AND p.type = 'group'\n"
    "  AND g.email = 'engineering@company.com'\n"
    "  AND p.deleted = false;"
)

_DRIVE_BAD_ENGINEERING_CHANGELOG_PERMISSION_QUERY = (
    "SELECT COUNT(*) as changelog_permission_count\n"
    "FROM permissions p\n"
    "JOIN files f ON p.file_id = f.id\n"
    "WHERE f.name = 'Technical Spec - Review Changelog'\n"
    "  AND f.created_time > '2024-01-25 00:00:00'\n"
    "  AND p.type = 'group'\n"
    "  AND p.email_address = 'engineering@company.com'\n"
    "  AND p.deleted = false;"
)

_DRIVE_FIXED_ENGINEERING_CHANGELOG_PERMISSION_QUERY = (
    "SELECT COUNT(*) as changelog_permission_count\n"
    "FROM permissions p\n"
    "JOIN files f ON p.file_id = f.id\n"
    "JOIN groups g ON p.group_id = g.id\n"
    "WHERE f.name = 'Technical Spec - Review Changelog'\n"
    "  AND f.created_time > '2024-01-25 00:00:00'\n"
    "  AND p.type = 'group'\n"
    "  AND g.email = 'engineering@company.com'\n"
    "  AND p.deleted = false;"
)

_DRIVE_BAD_EMILY_WILLIAMS_PERMISSION_QUERY = (
    "SELECT COUNT(*) as emily_permission FROM permissions p \n"
    "JOIN files f ON p.file_id = f.id \n"
    "WHERE f.name = 'Financials Q1 - Revision Cleanup Report' \n"
    "AND p.email_address = '_emily_williams@ibm.com' \n"
    "AND p.role = 'reader';"
)

_DRIVE_FIXED_EMILY_WILLIAMS_PERMISSION_QUERY = (
    "SELECT COUNT(*) as emily_permission FROM permissions p\n"
    "JOIN files f ON p.file_id = f.id\n"
    "JOIN users u ON p.user_id = u.id\n"
    "WHERE f.name = 'Financials Q1 - Revision Cleanup Report'\n"
    "  AND u.email = '_emily_williams@ibm.com'\n"
    "  AND p.type = 'user'\n"
    "  AND p.role = 'reader'\n"
    "  AND p.deleted = 0;"
)


_VERIFIER_CORRECTIONS = {
    "task_20251128_123025_545_13762b97_ad80b87c": {
        7: {
            "name": "Engineering Group Has Access to Technical Spec",
            "query": _DRIVE_BAD_ENGINEERING_ORIGINAL_PERMISSION_QUERY,
            "source_gym_name": "gym-google-drive-mcp",
            "target_gym_name": "gym-google-drive-mcp",
            "target_query": _DRIVE_FIXED_ENGINEERING_ORIGINAL_PERMISSION_QUERY,
            "rationale": (
                "Drive stores the existing Engineering Team principal through "
                "permissions.group_id, not permissions.email_address."
            ),
        },
        8: {
            "name": "Engineering Group Has Access to Changelog",
            "query": _DRIVE_BAD_ENGINEERING_CHANGELOG_PERMISSION_QUERY,
            "source_gym_name": "gym-google-drive-mcp",
            "target_gym_name": "gym-google-drive-mcp",
            "target_query": _DRIVE_FIXED_ENGINEERING_CHANGELOG_PERMISSION_QUERY,
            "rationale": (
                "Drive stores the existing Engineering Team principal through "
                "permissions.group_id, not permissions.email_address."
            ),
        },
    },
    "task_20251222_113047_572_64a4ebb9_cd2083a8": {
        0: {
            "name": "Verify User Deactivated",
            "query": (
                "SELECT COUNT(*) FROM users WHERE email = "
                "'christina.oliver@carsonllc.com' AND active = 0;"
            ),
            "source_gym_name": "gym-google-drive-mcp",
            "target_gym_name": "gym-itsm-mcp",
            "rationale": "The SQL targets the ITSM users table, not Drive.",
        }
    },
    "task_20251222_120551_813_1628b966_8e21a82f": {
        index: {
            "name": "Verify ITSM Lockdown",
            "query": _HYBRID_LOCKDOWN_QUERY,
            "source_gym_name": "gym-email-mcp",
            "target_gym_name": "gym-itsm-mcp",
            "rationale": "The SQL targets ITSM users and incident tables, not Email.",
        }
        for index in (0, 2)
    },
    "task_20260109_134319_026_64a4ebb9_98c48f81": {
        3: {
            "name": "Internal Compliance Permission Granted",
            "query": _DRIVE_BAD_COMPLIANCE_PERMISSION_QUERY,
            "source_gym_name": "gym-google-drive-mcp",
            "target_gym_name": "gym-google-drive-mcp",
            "target_query": _DRIVE_FIXED_COMPLIANCE_PERMISSION_QUERY,
            "rationale": (
                "Drive stores group permissions through permissions.group_id; "
                "permissions.email_address is null for type='group'."
            ),
        }
    },
    "task_20260108_162316_574_13762b97_098a086c": {
        5: {
            "name": "Emily Williams Has Reader Access to Report",
            "query": _DRIVE_BAD_EMILY_WILLIAMS_PERMISSION_QUERY,
            "source_gym_name": "gym-google-drive-mcp",
            "target_gym_name": "gym-google-drive-mcp",
            "target_query": _DRIVE_FIXED_EMILY_WILLIAMS_PERMISSION_QUERY,
            "rationale": (
                "Drive resolves the existing Emily Williams principal to "
                "permissions.user_id and clears permissions.email_address."
            ),
        }
    },
}


# The fixed Oracle snapshot serializes every numeric verifier target as a
# string for these exact tasks, while all three distractor-mode rows use the
# corresponding JSON integer.  The runtime equality comparator is type-strict,
# so leaving the Oracle source value untouched would change task success rather
# than merely provenance.  The complete source tuple is pinned and checked
# before converting it; all other string targets remain unchanged.
_NUMERIC_EXPECTED_VALUE_CORRECTIONS = {
    "task_20260121_144251_116_38405923_9438dabd": ("1", "2", "2"),
    "task_20260121_161936_323_cbcf38f8_50ccd164": ("0", "1", "1", "1", "1"),
    "task_20260107_162554_472_accab84d_41d2e871": ("1", "1", "1", "1", "1", "1", "1", "1"),
    "task_20260108_160928_699_391f1eff_905701cb": ("1", "1", "2", "1"),
    "task_20260121_143201_590_1afb323a_5a03bf03": ("1", "1", "0", "0", "1", "1", "1"),
    "task_20260121_145409_648_1afb323a_4e203caa": ("1", "1", "1", "1"),
    "task_20260121_150447_596_1afb323a_0a1e4085": ("1", "1", "0", "2"),
    "task_20260121_163208_054_1afb323a_7f40ee66": ("1", "1"),
    "task_20260107_181520_716_64a4ebb9_ec0278c5": ("1", "1", "1", "1"),
    "task_20260108_165812_020_4d9df647_eeacd121": ("1", "1", "1", "0", "1"),
    "task_20260106_071510_179_911d75d7_0ae789ae": ("1", "1"),
    "task_20260116_055008_115_c5cac200_a905eff3": ("0", "0", "0", "0"),
    "task_20260108_041838_219_df0ecdeb_3f8c0642": ("1", "1", "1"),
    "task_20260109_155926_434_df0ecdeb_405562a2": (
        "1",
        "0",
        "0",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
        "1",
    ),
    "task_20260113_152354_165_9a1e131b_fc5e2cac": ("1", "1", "1"),
    "task_20260114_124106_138_1628b966_ddc2587a": ("1", "1"),
    "task_20260115_154856_907_1628b966_275e920f": ("1", "1", "1"),
    "task_20260116_080150_885_1628b966_5a58d8e3": ("1", "1"),
    "task_20260116_125857_450_1628b966_25b7eec5": ("0", "0"),
    "task_20260117_084409_361_4d9df647_9892db8c": ("2", "2", "1", "1", "1"),
}


_CONTEXT_KEY_CORRECTIONS: Mapping[str, Mapping[str, Any]] = {}


_BAD_CALENDAR_ACCESS_TOKEN = (
    "ya29.A0ARrdaM-t3K9JkB6Qw4Yx7R8hS5lM2nP3qL0aZ9cBvN0."
    "ZxYcVbNmLkJhGfDsAaQwErTyUiOpAsDfGhJk_77-11"
)

_SEED_DATABASE_FILE_CORRECTIONS = {
    "task_20251217_030654_663_99ba2325_f931fd42": {
        "domain": "itsm",
        "server_name": "gym-itsm-mcp",
        "source_seed_database_file": (
            "Domain Wise DBs and Task-DB Mappings/itsm/dbs/"
            "db_1765893060767_9rwtfbbp7.sql"
        ),
        "target_seed_database_file": (
            "Domain Wise DBs and Task-DB Mappings/itsm/dbs/"
            "db_1765301900121_3mwjj54xy.sql"
        ),
        "context_key": "x-itsm-user-token",
        "context_value": "admin_token_marcus_2024_secure",
        "prompt_fragment": "create a new incident for Elena Petrov",
        "rationale": (
            "Use the source snapshot in which Elena Petrov, Dev Server Access, "
            "Desktop Boot Failure Troubleshooting, Service Desk Operations, and "
            "the next incident ID INC_024 match the prompt and both verifiers."
        ),
    },
}

_CONTEXT_VALUE_CORRECTIONS = {
    "task_20251205_032409_945_0a0bf089_61f5be29": (
        {
            "domain": "calendar",
            "server_name": "gym-calendar",
            "seed_database_file": (
                "Domain Wise DBs and Task-DB Mappings/calendar/dbs/"
                "db_1764361046464_47vxzhsli.sql"
            ),
            "key": "x-access-token",
            "source_value": _BAD_CALENDAR_ACCESS_TOKEN,
            "target_value": (
                "ya29.A0ARrdaM-Zx8Nw3Q4pVb6Ls9R1mT0cG2uF5yH7kJd8sA1Lq2."
                "wErtYuIoPaSdFgHjKlZxCvBnM987_65-43"
            ),
            "target_identity": "bob_developer",
            "prompt_fragment": "calendars for bob.smith@techcorp.com",
            "rationale": (
                "Use the static token belonging to the Bob Smith identity named by "
                "the prompt and verifier."
            ),
        },
    ),
    "task_20251217_081417_887_0a0bf089_487054de": (
        {
            "domain": "hybrid",
            "server_name": "sn-csm-server",
            "seed_database_file": (
                "Domain Wise DBs and Task-DB Mappings/hybrid/dbs/"
                "db_1765300846591_aekfq9lke.sql"
            ),
            "key": "x-user-email",
            "source_missing": True,
            "target_value": "aaron.le@servicenow.com",
            "target_identity": "aaron_le",
            "prompt_fragment": "High-Priority Incident Review",
            "rationale": (
                "Use the Aaron Le identity already selected by this task's Calendar "
                "token when calling the CSM reporting tools."
            ),
        },
    ),
    "task_20251224_153415_365_0a0bf089_01602883": (
        {
            "domain": "hybrid",
            "server_name": "sn-csm-server",
            "seed_database_file": (
                "Domain Wise DBs and Task-DB Mappings/hybrid/dbs/"
                "db_1765300846591_aekfq9lke.sql"
            ),
            "key": "x-user-email",
            "source_missing": True,
            "target_value": "aaron.le@servicenow.com",
            "target_identity": "aaron_le",
            "prompt_fragment": "Wayne Enterprises – Support Planning",
            "rationale": (
                "Use the Aaron Le identity already selected by this task's Calendar "
                "token when calling the CSM account tools."
            ),
        },
    ),
    "task_20260109_145235_208_64a4ebb9_9befb3ae": (
        {
            "domain": "drive",
            "server_name": "gym-google-drive-mcp",
            "seed_database_file": (
                "Domain Wise DBs and Task-DB Mappings/drive/dbs/"
                "db_1764755314410_f227g8ov4.sql"
            ),
            "key": "x_auth_token",
            "source_value": (
                "ya29.ZH_vhrq26vaob5lBJuSNVEh4K5Gl_KZutOkssEBqgJjrpL28omzvdfu"
                "hm4HTryVB2WJwvOn2PpopjPcWq73EK14LjHNpfiBTywRYc0mtPGZZQqk5uLjui"
                "ubsAhNVcJew"
            ),
            "target_value": (
                "ya29.E79oYmqohz4_yuNhAYfz7UsMx_YpFIXLCr1kMAgGtrvs9aINJEW2HfI54"
                "QtDpnvBx6yx7zaZkX9Y2Nekrf6VGmITgDUfwmeioLLJOXTp_eb6qfjubiyfAH1"
                "loUHmYL9g"
            ),
            "target_identity": "9e2a66d40a13450dbef56d074f",
            "prompt_fragment": "Monthly Financials.xls.xlsx",
            "rationale": (
                "Use the Carmen Alvarez token that matches this row's user_info and "
                "Drive seed."
            ),
        },
    ),
    "task_20260112_125407_584_1628b966_f5dd38b6": (
        {
            "domain": "hybrid",
            "server_name": "gym-calendar",
            "seed_database_file": (
                "Domain Wise DBs and Task-DB Mappings/hybrid/dbs/"
                "db_1765386439582_vm0389iwq.sql"
            ),
            "key": "x-access-token",
            "source_value": _BAD_CALENDAR_ACCESS_TOKEN,
            "target_value": (
                "tok_3469aec5aa7efb31d87435d2e8e84e6225dbabf1bbf5a309"
            ),
            "target_identity": "aaron_le",
            "prompt_fragment": "Aaron Le – Active Case Review",
            "rationale": (
                "Use Aaron Le's static token for the task that targets Aaron's "
                "primary calendar and Aaron Le event."
            ),
        },
        {
            "domain": "hybrid",
            "server_name": "sn-csm-server",
            "seed_database_file": (
                "Domain Wise DBs and Task-DB Mappings/hybrid/dbs/"
                "db_1765300846591_aekfq9lke.sql"
            ),
            "key": "x-user-email",
            "source_missing": True,
            "target_value": "aaron.le@servicenow.com",
            "target_identity": "aaron_le",
            "prompt_fragment": "Aaron Le – Active Case Review",
            "rationale": (
                "Use Aaron Le's CSM identity for the task that looks up Aaron and "
                "summarizes Aaron's assigned cases."
            ),
        },
    ),
}


_HR_MISSING_DISTRACTOR_CORRECTIONS = {
    "task_20251212_114043_514_b25153b4_e98b68ff": "Leadership Rotation Program",
    "task_20251216_123530_079_ec0cb5dd_3b89aab7": "operationalize David Kim",
    "task_20251217_073807_153_761f6b82_cba63024": "urgent field deployment",
    "task_20251217_161056_114_d9bcd1e5_c8f3ccb9": "targeted routing",
    "task_20251217_211723_514_87437950_3bbbb519": (
        "skills-based routing for US payroll bank account change"
    ),
    "task_20251217_231709_248_b25153b4_4c5f9bde": "Executive Relocation Program",
    "task_20251218_234518_453_d9bcd1e5_72c94403": "workplace misconduct",
    "task_20251219_114935_971_a8eea1c0_188fc892": "Samantha Stone primary account",
}


_SELECTED_TOOL_ADDITION_CORRECTIONS = {
    "task_20260108_194921_186_8e9e30d7_a087d6f8": {
        "domain": "teams",
        "task_server_names": ("gym-teams-mcp",),
        "prompt_fragment": "Add a Planner tab there called 'Release Checklist'",
        "tool_name": "list_teams_apps",
        "server_name": "gym-teams-mcp",
        "required_existing_tool": "add_tabs_to_channels",
        "source_variants": {
            "7312a52463013e9ea59b5735fb8eea452cf3e7345ac6e35fb9407f5853e87be8": (
                "oracle",
                10,
                10,
            ),
            "c96bdbbfedfb433f7266947cc1c9434a4f6c61b20cfccbc485a18639cfbc1609": (
                "plus_5_tools",
                18,
                15,
            ),
            "05929ce809919c1a99153b46ebf931e31f3acbff97cc70d8874d07fd5c1e20b4": (
                "plus_10_tools",
                23,
                20,
            ),
            "20eb36f4ab2e1b7c55146a9692adb0fb442c2acfa81a111a3f3f601193dc67e8": (
                "plus_15_tools",
                28,
                25,
            ),
        },
        "rationale": (
            "The task requires adding a Planner tab through an opaque Teams app ID, "
            "but its source tool set omits the app-catalog lookup used by equivalent "
            "Planner-tab tasks in the same fixed dataset."
        ),
    }
}


_HYBRID_DUPLICATE_TOOLS_BY_SERVER_SET = {
    frozenset(("gym-google-drive-mcp", "gym-itsm-mcp")): ("list_changes",),
    frozenset(("gym-itsm-mcp", "gym-teams-mcp")): ("get_user", "list_users"),
    frozenset(("sn-csm-server", "gym-teams-mcp")): ("list_users",),
    frozenset(("sn-hr-internal", "gym-teams-mcp")): ("get_user", "list_users"),
}


def _owner_rules(
    expected_server_names: tuple[str, str],
    target_server_name: str,
    rationale: str,
    *tool_names: str,
) -> Mapping[str, Mapping[str, Any]]:
    return {
        tool_name: {
            "expected_server_names": expected_server_names,
            "target_server_name": target_server_name,
            "rationale": rationale,
        }
        for tool_name in tool_names
    }


_SELECTED_TOOL_OWNER_CORRECTIONS = {
    "task_20251222_131624_825_8ce75c22_535e7ced": {
        "domain": "hybrid",
        "task_server_names": ("sn-csm-server", "gym-teams-mcp"),
        "prompt_fragment": "customer contact on this case is Elizabeth Chase",
        "tools": _owner_rules(
            ("sn-csm-server", "gym-teams-mcp"),
            "sn-csm-server",
            "The named customer contact belongs to the CSM case context.",
            "list_users",
        ),
    },
    "task_20251223_005949_519_13762b97_e08c84a3": {
        "domain": "hybrid",
        "task_server_names": ("gym-google-drive-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "Pull all change requests from ITSM",
        "tools": _owner_rules(
            ("gym-google-drive-mcp", "gym-itsm-mcp"),
            "gym-itsm-mcp",
            "The prompt explicitly requests ITSM change requests, not Drive change feed entries.",
            "list_changes",
        ),
    },
    "task_20251224_215807_175_8ce75c22_1c31b786": {
        "domain": "hybrid",
        "task_server_names": ("sn-hr-internal", "gym-teams-mcp"),
        "prompt_fragment": "Reach out to that person in Teams",
        "tools": _owner_rules(
            ("sn-hr-internal", "gym-teams-mcp"),
            "gym-teams-mcp",
            "The HR assignee must be resolved as a Teams user before chat actions.",
            "get_user",
            "list_users",
        ),
    },
    "task_20251225_130546_796_4d9df647_7e32b59a": {
        "domain": "hybrid",
        "task_server_names": ("gym-teams-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "set up something on Teams with Ian Johnson",
        "tools": _owner_rules(
            ("gym-teams-mcp", "gym-itsm-mcp"),
            "gym-teams-mcp",
            "The named participant is used for a Teams troubleshooting session.",
            "get_user",
            "list_users",
        ),
    },
    "task_20251225_133905_252_060e2c81_7e4fd228": {
        "domain": "hybrid",
        "task_server_names": ("gym-teams-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "Assign the incident to myself",
        "tools": _owner_rules(
            ("gym-teams-mcp", "gym-itsm-mcp"),
            "gym-itsm-mcp",
            "Joanne Simpson is resolved as the ITSM incident assignee.",
            "get_user",
            "list_users",
        ),
    },
    "task_20251225_152926_506_4d9df647_e4e518ca": {
        "domain": "hybrid",
        "task_server_names": ("gym-teams-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "spin up a live War Room",
        "tools": _owner_rules(
            ("gym-teams-mcp", "gym-itsm-mcp"),
            "gym-teams-mcp",
            "These added user tools are distractors; preserve the first declared Teams owner.",
            "get_user",
            "list_users",
        ),
    },
    "task_20251225_165311_201_4d9df647_7b3f1641": {
        "domain": "hybrid",
        "task_server_names": ("gym-teams-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "Technical Deep Dive: Hooli Teams Fix",
        "tools": _owner_rules(
            ("gym-teams-mcp", "gym-itsm-mcp"),
            "gym-teams-mcp",
            "These added user tools are distractors; preserve the first declared Teams owner.",
            "get_user",
            "list_users",
        ),
    },
    "task_20251226_150609_837_4d9df647_84917c0b": {
        "domain": "hybrid",
        "task_server_names": ("gym-teams-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "assign to myselft",
        "tools": _owner_rules(
            ("gym-teams-mcp", "gym-itsm-mcp"),
            "gym-itsm-mcp",
            "Joanne Simpson is resolved as the ITSM incident assignee.",
            "get_user",
            "list_users",
        ),
    },
    "task_20251226_180126_405_bca89e7d_c0d5b9ee": {
        "domain": "hybrid",
        "task_server_names": ("sn-hr-internal", "gym-teams-mcp"),
        "prompt_fragment": "annual performance review rollout",
        "tools": _owner_rules(
            ("sn-hr-internal", "gym-teams-mcp"),
            "sn-hr-internal",
            "These added user tools are distractors; preserve the first declared HR owner.",
            "get_user",
            "list_users",
        ),
    },
    "task_20251226_225931_685_bca89e7d_597733d7": {
        "domain": "hybrid",
        "task_server_names": ("sn-hr-internal", "gym-teams-mcp"),
        "prompt_fragment": "assign it to John Ramirez and Carl Brown",
        "tools": _owner_rules(
            ("sn-hr-internal", "gym-teams-mcp"),
            "sn-hr-internal",
            "The named users receive HR skills and belong to the HR user directory.",
            "list_users",
        ),
    },
    "task_20260116_054437_417_59bb807c_d73ca3e4": {
        "domain": "hybrid",
        "task_server_names": ("sn-hr-internal", "gym-teams-mcp"),
        "prompt_fragment": "Find Tabitha in Teams",
        "tools": _owner_rules(
            ("sn-hr-internal", "gym-teams-mcp"),
            "gym-teams-mcp",
            "The prompt explicitly resolves Tabitha in Teams before the HR lookup.",
            "get_user",
            "list_users",
        ),
    },
    "task_20260116_095226_245_4d9df647_08f9f5b1": {
        "domain": "hybrid",
        "task_server_names": ("gym-teams-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "provision his Microsoft Teams identity",
        "tools": _owner_rules(
            ("gym-teams-mcp", "gym-itsm-mcp"),
            "gym-teams-mcp",
            "These added user tools are distractors; preserve the first declared Teams owner.",
            "get_user",
            "list_users",
        ),
    },
    "task_20260116_155214_819_baf0b358_0cf4418f": {
        "domain": "hybrid",
        "task_server_names": ("gym-itsm-mcp", "gym-teams-mcp"),
        "prompt_fragment": "change ticket CHG0000001",
        "tools": _owner_rules(
            ("gym-itsm-mcp", "gym-teams-mcp"),
            "gym-itsm-mcp",
            "These added user tools are distractors; preserve the first declared ITSM owner.",
            "get_user",
            "list_users",
        ),
    },
    "task_20260116_174925_124_d8f93f2d_9e20df73": {
        "domain": "hybrid",
        "task_server_names": ("gym-google-drive-mcp", "gym-itsm-mcp"),
        "prompt_fragment": "set up a change freeze",
        "tools": _owner_rules(
            ("gym-google-drive-mcp", "gym-itsm-mcp"),
            "gym-itsm-mcp",
            "The change freeze is an ITSM change, not a Drive change feed entry.",
            "list_changes",
        ),
    },
    "task_20260117_084409_361_4d9df647_9892db8c": {
        "domain": "hybrid",
        "task_server_names": ("sn-hr-internal", "gym-teams-mcp"),
        "prompt_fragment": "find Thomas Green in Teams",
        "tools": _owner_rules(
            ("sn-hr-internal", "gym-teams-mcp"),
            "gym-teams-mcp",
            "Thomas Green must be resolved as the Teams webinar co-organizer.",
            "get_user",
            "list_users",
        ),
    },
}


@dataclass(frozen=True)
class ToolOwnerCorrection:
    """Pinned owner for one ambiguous selected tool in a known source row."""

    tool_name: str
    expected_server_names: tuple[str, ...]
    target_server_name: str
    rationale: str


@dataclass(frozen=True)
class MissingToolCorrection:
    """A known source distractor that the fixed official runtime does not publish."""

    tool_name: str
    server_name: str
    rationale: str


@dataclass(frozen=True)
class SelectedToolAdditionCorrection:
    """Pinned addition of one required lookup tool omitted by a source task row."""

    tool_name: str
    server_name: str
    source_variant: str
    source_declared_tool_count: int
    source_unique_declared_tool_count: int
    rationale: str


@dataclass(frozen=True)
class ContextKeyCorrection:
    """Pinned repair for one malformed context key in a known source row."""

    server_name: str
    source_key: str
    target_key: str
    rationale: str


@dataclass(frozen=True)
class ContextValueCorrection:
    """Pinned repair for one invalid context value in a known source row."""

    server_name: str
    key: str
    target_identity: str
    rationale: str


@dataclass(frozen=True)
class SeedDatabaseFileCorrection:
    """Pinned replacement for one source row bound to the wrong seed snapshot."""

    server_name: str
    source_seed_database_file: str
    target_seed_database_file: str
    rationale: str


@dataclass(frozen=True)
class VerifierCorrection:
    """Pinned repair for one verifier route or query in a known source row."""

    index: int
    source_gym_name: str
    target_gym_name: str
    query_changed: bool
    rationale: str


@dataclass(frozen=True)
class VerifierExpectedValueCorrection:
    """Pinned numeric type repair for one known Oracle verifier target."""

    index: int
    source_value: str
    target_value: int
    rationale: str


@dataclass(frozen=True)
class EnterpriseOpsTask:
    task_id: str
    domain: str
    system_prompt: str
    user_prompt: str
    selected_tools: tuple[str, ...]
    restricted_tools: tuple[str, ...]
    mcp_endpoint: str
    number_of_runs: int
    reset_database_between_runs: bool
    gym_servers: tuple[Mapping[str, Any], ...]
    verifiers: tuple[Mapping[str, Any], ...]
    official_revised_overlays: tuple[OfficialRevisedOverlay, ...] = ()
    tool_owner_corrections: tuple[ToolOwnerCorrection, ...] = ()
    missing_tool_corrections: tuple[MissingToolCorrection, ...] = ()
    selected_tool_addition_corrections: tuple[SelectedToolAdditionCorrection, ...] = ()
    context_key_corrections: tuple[ContextKeyCorrection, ...] = ()
    context_value_corrections: tuple[ContextValueCorrection, ...] = ()
    seed_database_file_corrections: tuple[SeedDatabaseFileCorrection, ...] = ()
    verifier_corrections: tuple[VerifierCorrection, ...] = ()
    verifier_expected_value_corrections: tuple[VerifierExpectedValueCorrection, ...] = ()
    verifier_contract_corrections: tuple[VerifierContractCorrection, ...] = ()
    verifier_set_normalization: VerifierSetNormalization | None = None

    @property
    def unique_selected_tools(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.selected_tools))

    @property
    def duplicate_selected_tools(self) -> tuple[str, ...]:
        seen: set[str] = set()
        duplicates: list[str] = []
        for name in self.selected_tools:
            if name in seen and name not in duplicates:
                duplicates.append(name)
            seen.add(name)
        return tuple(duplicates)


def _parse_json_list(value: Any, field: str, task_id: str) -> list[Mapping[str, Any]]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise TaskContractError(f"{task_id}: {field} is not valid JSON: {exc}") from exc
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise TaskContractError(f"{task_id}: {field} must be a list of objects")
    return value


def _string_list(value: Any, field: str, task_id: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise TaskContractError(f"{task_id}: {field} must be a list")
    if not all(isinstance(item, str) and item for item in value):
        raise TaskContractError(f"{task_id}: {field} must contain non-empty strings")
    return tuple(value)


def _apply_known_verifier_corrections(
    task_id: str,
    verifiers: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], tuple[VerifierCorrection, ...]]:
    """Apply pinned source-data corrections after checking the exact bad row."""

    corrections = _VERIFIER_CORRECTIONS.get(task_id, {})
    if not corrections:
        return verifiers, ()

    corrected = [dict(verifier) for verifier in verifiers]
    applied: list[VerifierCorrection] = []
    for index, correction in corrections.items():
        if index >= len(corrected):
            raise TaskContractError(
                f"{task_id}: verifier correction index {index} is out of range"
            )
        verifier = corrected[index]
        validation = verifier.get("validation_config")
        observed = {
            "name": verifier.get("name"),
            "query": validation.get("query") if isinstance(validation, Mapping) else None,
            "source_gym_name": verifier.get("gym_name"),
        }
        expected = {
            key: correction[key]
            for key in ("name", "query", "source_gym_name")
        }
        if observed != expected:
            raise TaskContractError(
                f"{task_id}: verifier {index} no longer matches its pinned correction source"
            )
        verifier["gym_name"] = correction["target_gym_name"]
        target_query = correction.get("target_query")
        if target_query is not None:
            verifier["validation_config"] = dict(validation)
            verifier["validation_config"]["query"] = target_query
        applied.append(
            VerifierCorrection(
                index=index,
                source_gym_name=correction["source_gym_name"],
                target_gym_name=correction["target_gym_name"],
                query_changed=target_query is not None,
                rationale=correction["rationale"],
            )
        )
    return corrected, tuple(applied)


def _apply_known_expected_value_corrections(
    task_id: str,
    verifiers: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], tuple[VerifierExpectedValueCorrection, ...]]:
    """Normalize exact source numeric strings without coercing other targets."""

    source_values = _NUMERIC_EXPECTED_VALUE_CORRECTIONS.get(task_id)
    if source_values is None:
        return verifiers, ()

    observed: list[Any] = []
    for verifier in verifiers:
        validation = verifier.get("validation_config")
        observed.append(
            validation.get("expected_value")
            if isinstance(validation, Mapping)
            else None
        )

    if all(isinstance(value, int) and not isinstance(value, bool) for value in observed):
        return verifiers, ()
    if tuple(observed) != source_values:
        raise TaskContractError(
            f"{task_id}: verifier expected values no longer match their pinned source"
        )

    corrected: list[Mapping[str, Any]] = []
    applied: list[VerifierExpectedValueCorrection] = []
    for index, (verifier, source_value) in enumerate(zip(verifiers, source_values)):
        updated = dict(verifier)
        validation = dict(updated["validation_config"])
        target_value = int(source_value)
        validation["expected_value"] = target_value
        updated["validation_config"] = validation
        corrected.append(updated)
        applied.append(
            VerifierExpectedValueCorrection(
                index=index,
                source_value=source_value,
                target_value=target_value,
                rationale=(
                    "The fixed Oracle row serializes a numeric database target as a "
                    "string while the paired distractor rows use the JSON integer."
                ),
            )
        )
    return corrected, tuple(applied)


def _apply_known_context_key_correction(
    *,
    task_id: str,
    domain: Any,
    user_prompt: Any,
    gym_servers: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], tuple[ContextKeyCorrection, ...]]:
    """Repair a registered malformed header only after its source shape matches."""

    correction = _CONTEXT_KEY_CORRECTIONS.get(task_id)
    if correction is None:
        return gym_servers, ()
    if (
        domain != correction["domain"]
        or not isinstance(user_prompt, str)
        or correction["prompt_fragment"] not in user_prompt
    ):
        raise TaskContractError(
            f"{task_id}: task no longer matches its pinned context-key correction source"
        )
    matching_indices = [
        index
        for index, server in enumerate(gym_servers)
        if server.get("mcp_server_name") == correction["server_name"]
    ]
    if len(matching_indices) != 1:
        raise TaskContractError(
            f"{task_id}: context-key correction server is not uniquely present"
        )
    index = matching_indices[0]
    server = gym_servers[index]
    context = server.get("context")
    source_key = correction["source_key"]
    target_key = correction["target_key"]
    if (
        not isinstance(context, dict)
        or source_key not in context
        or target_key in context
    ):
        raise TaskContractError(
            f"{task_id}: context no longer matches its pinned key correction source"
        )

    corrected = [dict(item) for item in gym_servers]
    corrected_context = dict(context)
    corrected_context[target_key] = corrected_context.pop(source_key)
    corrected[index]["context"] = corrected_context
    applied = ContextKeyCorrection(
        server_name=correction["server_name"],
        source_key=source_key,
        target_key=target_key,
        rationale=correction["rationale"],
    )
    return corrected, (applied,)


def _apply_known_seed_database_file_correction(
    *,
    task_id: str,
    domain: Any,
    user_prompt: Any,
    gym_servers: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], tuple[SeedDatabaseFileCorrection, ...]]:
    """Repair one proven wrong seed binding after matching its exact source row."""

    correction = _SEED_DATABASE_FILE_CORRECTIONS.get(task_id)
    if correction is None:
        return gym_servers, ()
    if (
        domain != correction["domain"]
        or not isinstance(user_prompt, str)
        or correction["prompt_fragment"] not in user_prompt
    ):
        raise TaskContractError(
            f"{task_id}: task no longer matches its pinned seed correction source"
        )
    matching_indices = [
        index
        for index, server in enumerate(gym_servers)
        if server.get("mcp_server_name") == correction["server_name"]
    ]
    if len(matching_indices) != 1:
        raise TaskContractError(
            f"{task_id}: seed correction server is not uniquely present"
        )
    index = matching_indices[0]
    server = gym_servers[index]
    context = server.get("context")
    if (
        server.get("seed_database_file")
        != correction["source_seed_database_file"]
        or not isinstance(context, dict)
        or context.get(correction["context_key"]) != correction["context_value"]
    ):
        raise TaskContractError(
            f"{task_id}: gym server no longer matches its pinned seed correction source"
        )

    corrected = [dict(item) for item in gym_servers]
    corrected[index] = dict(server)
    corrected[index]["seed_database_file"] = correction["target_seed_database_file"]
    applied = SeedDatabaseFileCorrection(
        server_name=correction["server_name"],
        source_seed_database_file=correction["source_seed_database_file"],
        target_seed_database_file=correction["target_seed_database_file"],
        rationale=correction["rationale"],
    )
    return corrected, (applied,)


def _apply_known_context_value_correction(
    *,
    task_id: str,
    domain: Any,
    user_prompt: Any,
    gym_servers: list[Mapping[str, Any]],
) -> tuple[list[Mapping[str, Any]], tuple[ContextValueCorrection, ...]]:
    """Repair a registered invalid auth value only after its source shape matches."""

    corrections = _CONTEXT_VALUE_CORRECTIONS.get(task_id)
    if corrections is None:
        return gym_servers, ()
    corrected = [dict(item) for item in gym_servers]
    applied: list[ContextValueCorrection] = []
    for correction in corrections:
        if (
            domain != correction["domain"]
            or not isinstance(user_prompt, str)
            or correction["prompt_fragment"] not in user_prompt
        ):
            raise TaskContractError(
                f"{task_id}: task no longer matches its pinned context-value "
                "correction source"
            )
        matching_indices = [
            index
            for index, server in enumerate(corrected)
            if server.get("mcp_server_name") == correction["server_name"]
        ]
        if len(matching_indices) != 1:
            raise TaskContractError(
                f"{task_id}: context-value correction server is not uniquely present"
            )
        index = matching_indices[0]
        server = corrected[index]
        context = server.get("context")
        key = correction["key"]
        source_matches = (
            isinstance(context, dict)
            and (
                key not in context
                if correction.get("source_missing", False)
                else key in context and context[key] == correction["source_value"]
            )
        )
        if (
            server.get("seed_database_file") != correction["seed_database_file"]
            or not source_matches
        ):
            raise TaskContractError(
                f"{task_id}: context no longer matches its pinned value correction source"
            )

        corrected_context = dict(context)
        corrected_context[key] = correction["target_value"]
        corrected[index] = dict(server)
        corrected[index]["context"] = corrected_context
        applied.append(
            ContextValueCorrection(
                server_name=correction["server_name"],
                key=key,
                target_identity=correction["target_identity"],
                rationale=correction["rationale"],
            )
        )
    return corrected, tuple(applied)


def _known_missing_tool_corrections(
    *,
    task_id: str,
    domain: Any,
    user_prompt: Any,
    selected_tools: tuple[str, ...],
    restricted_tools: tuple[str, ...],
    gym_servers: list[Mapping[str, Any]],
) -> tuple[MissingToolCorrection, ...]:
    """Register only the confirmed HR distractor absent from the official runtime."""

    selected_count = selected_tools.count("get_user_group")
    prompt_fragment = _HR_MISSING_DISTRACTOR_CORRECTIONS.get(task_id)
    if selected_count == 0:
        return ()
    observed_server_names = tuple(
        server.get("mcp_server_name") for server in gym_servers
    )
    if (
        prompt_fragment is None
        or selected_count != 1
        or domain != "hr"
        or not isinstance(user_prompt, str)
        or prompt_fragment not in user_prompt
        or observed_server_names != ("sn-hr-internal",)
        or "get_user_group" in restricted_tools
    ):
        raise TaskContractError(
            f"{task_id}: get_user_group no longer matches the registered missing-distractor source"
        )
    return (
        MissingToolCorrection(
            tool_name="get_user_group",
            server_name="sn-hr-internal",
            rationale=(
                "The fixed official HR executor omits this selected distractor because "
                "the official HR MCP catalog has no matching schema or implementation."
            ),
        ),
    )


def _apply_known_selected_tool_additions(
    *,
    task_id: str,
    domain: Any,
    user_prompt: Any,
    selected_tools: tuple[str, ...],
    restricted_tools: tuple[str, ...],
    gym_servers: list[Mapping[str, Any]],
) -> tuple[tuple[str, ...], tuple[SelectedToolAdditionCorrection, ...]]:
    """Add only a registered, source-pinned lookup omitted by a task row."""

    correction = _SELECTED_TOOL_ADDITION_CORRECTIONS.get(task_id)
    if correction is None:
        return selected_tools, ()

    serialized = json.dumps(
        selected_tools,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    source_signature = sha256(serialized).hexdigest()
    source_variant = correction["source_variants"].get(source_signature)
    observed_server_names = tuple(
        server.get("mcp_server_name") for server in gym_servers
    )
    tool_name = correction["tool_name"]
    if (
        domain != correction["domain"]
        or observed_server_names != correction["task_server_names"]
        or not isinstance(user_prompt, str)
        or correction["prompt_fragment"] not in user_prompt
        or source_variant is None
        or source_variant[1] != len(selected_tools)
        or source_variant[2] != len(tuple(dict.fromkeys(selected_tools)))
        or correction["required_existing_tool"] not in selected_tools
        or tool_name in selected_tools
        or tool_name in restricted_tools
    ):
        raise TaskContractError(
            f"{task_id}: task no longer matches its pinned selected-tool addition source"
        )

    corrected = (*selected_tools, tool_name)
    return corrected, (
        SelectedToolAdditionCorrection(
            tool_name=tool_name,
            server_name=correction["server_name"],
            source_variant=source_variant[0],
            source_declared_tool_count=source_variant[1],
            source_unique_declared_tool_count=source_variant[2],
            rationale=correction["rationale"],
        ),
    )


def _known_tool_owner_corrections(
    *,
    task_id: str,
    domain: Any,
    user_prompt: Any,
    selected_tools: tuple[str, ...],
    restricted_tools: tuple[str, ...],
    gym_servers: list[Mapping[str, Any]],
) -> tuple[ToolOwnerCorrection, ...]:
    """Return pinned owner corrections only when the exact source shape still matches."""

    observed_server_names = tuple(server.get("mcp_server_name") for server in gym_servers)
    known_duplicates = set(
        _HYBRID_DUPLICATE_TOOLS_BY_SERVER_SET.get(
            frozenset(observed_server_names),
            (),
        )
    )
    selected_duplicate_names = {
        name
        for name in selected_tools
        if name in known_duplicates and name not in restricted_tools
    }
    correction = _SELECTED_TOOL_OWNER_CORRECTIONS.get(task_id)
    if correction is None:
        if selected_duplicate_names:
            raise TaskContractError(
                f"{task_id}: selected Hybrid tool-owner collisions are not registered: "
                f"{sorted(selected_duplicate_names)}"
            )
        return ()

    if (
        domain != correction["domain"]
        or observed_server_names != correction["task_server_names"]
        or not isinstance(user_prompt, str)
        or correction["prompt_fragment"] not in user_prompt
    ):
        raise TaskContractError(
            f"{task_id}: task no longer matches its pinned tool-owner correction source"
        )

    registered_names = set(correction["tools"])
    missing_registrations = selected_duplicate_names - registered_names
    if missing_registrations:
        raise TaskContractError(
            f"{task_id}: selected Hybrid tool-owner collisions are not registered: "
            f"{sorted(missing_registrations)}"
        )

    result: list[ToolOwnerCorrection] = []
    for tool_name, tool_correction in correction["tools"].items():
        selected_count = selected_tools.count(tool_name)
        if selected_count == 0:
            continue
        if selected_count != 1 or tool_name in restricted_tools:
            raise TaskContractError(
                f"{task_id}: selected tool {tool_name!r} no longer matches its pinned "
                "owner correction source"
            )
        result.append(
            ToolOwnerCorrection(
                tool_name=tool_name,
                expected_server_names=tool_correction["expected_server_names"],
                target_server_name=tool_correction["target_server_name"],
                rationale=tool_correction["rationale"],
            )
        )
    return tuple(result)


def task_from_row(row: Mapping[str, Any]) -> EnterpriseOpsTask:
    required = {
        "task_id",
        "domain",
        "system_prompt",
        "user_prompt",
        "selected_tools",
        "restricted_tools",
        "mcp_endpoint",
        "number_of_runs",
        "reset_database_between_runs",
        "gym_servers_config",
        "verifiers",
    }
    missing = sorted(required - set(row))
    if missing:
        raise TaskContractError(f"task row missing fields: {missing}")

    try:
        row, official_revised_overlays = apply_official_revised_overlay(row)
    except OfficialRevisedOverlayError as exc:
        raise TaskContractError(str(exc)) from exc

    task_id = row["task_id"]
    if not isinstance(task_id, str) or not task_id:
        raise TaskContractError("task_id must be a non-empty string")

    selected_tools = _string_list(row["selected_tools"], "selected_tools", task_id)
    restricted_raw = row["restricted_tools"]
    if restricted_raw is None:
        restricted_raw = []
    restricted_tools = _string_list(restricted_raw, "restricted_tools", task_id)
    gym_servers, seed_database_file_corrections = (
        _apply_known_seed_database_file_correction(
            task_id=task_id,
            domain=row["domain"],
            user_prompt=row["user_prompt"],
            gym_servers=_parse_json_list(
                row["gym_servers_config"], "gym_servers_config", task_id
            ),
        )
    )
    gym_servers, context_key_corrections = _apply_known_context_key_correction(
        task_id=task_id,
        domain=row["domain"],
        user_prompt=row["user_prompt"],
        gym_servers=gym_servers,
    )
    gym_servers, context_value_corrections = _apply_known_context_value_correction(
        task_id=task_id,
        domain=row["domain"],
        user_prompt=row["user_prompt"],
        gym_servers=gym_servers,
    )
    verifiers, verifier_corrections = _apply_known_verifier_corrections(
        task_id,
        _parse_json_list(row["verifiers"], "verifiers", task_id),
    )
    verifiers, verifier_expected_value_corrections = (
        _apply_known_expected_value_corrections(task_id, verifiers)
    )
    try:
        (
            verifiers,
            verifier_contract_corrections,
            verifier_set_normalization,
        ) = apply_verifier_contract(
            task_id=task_id,
            domain=row["domain"],
            user_prompt=row["user_prompt"],
            verifiers=verifiers,
        )
    except VerifierContractSourceError as exc:
        raise TaskContractError(str(exc)) from exc
    selected_tools, selected_tool_addition_corrections = (
        _apply_known_selected_tool_additions(
            task_id=task_id,
            domain=row["domain"],
            user_prompt=row["user_prompt"],
            selected_tools=selected_tools,
            restricted_tools=restricted_tools,
            gym_servers=gym_servers,
        )
    )

    if row["number_of_runs"] != 1:
        raise TaskContractError(f"{task_id}: number_of_runs must be 1")
    if row["reset_database_between_runs"] is not True:
        raise TaskContractError(f"{task_id}: database reset must be enabled")
    if not selected_tools:
        raise TaskContractError(f"{task_id}: selected_tools cannot be empty")
    if not gym_servers:
        raise TaskContractError(f"{task_id}: gym_servers_config cannot be empty")
    if not verifiers:
        raise TaskContractError(f"{task_id}: verifiers cannot be empty")

    server_names: set[str] = set()
    for index, server in enumerate(gym_servers):
        for field in ("mcp_server_name", "mcp_server_url", "seed_database_file", "context"):
            if field not in server:
                raise TaskContractError(f"{task_id}: gym server {index} missing {field}")
        name = server["mcp_server_name"]
        if not isinstance(name, str) or not name:
            raise TaskContractError(f"{task_id}: gym server {index} has invalid name")
        if name in server_names:
            raise TaskContractError(f"{task_id}: duplicate gym server name {name}")
        server_names.add(name)
        if not isinstance(server["context"], dict):
            raise TaskContractError(f"{task_id}: gym server {name} context must be an object")

    for index, verifier in enumerate(verifiers):
        if verifier.get("verifier_type") != "database_state":
            raise TaskContractError(
                f"{task_id}: verifier {index} is not a database_state verifier"
            )
        if verifier.get("gym_name") not in server_names:
            raise TaskContractError(
                f"{task_id}: verifier {index} targets unknown gym {verifier.get('gym_name')!r}"
            )
        validation = verifier.get("validation_config")
        if not isinstance(validation, dict):
            raise TaskContractError(f"{task_id}: verifier {index} lacks validation_config")
        if not validation.get("query") or "expected_value" not in validation:
            raise TaskContractError(f"{task_id}: verifier {index} lacks query/expected_value")
        if validation.get("comparison_type", "equals") not in {
            "equals",
            "greater_than",
            "less_than",
            "contains",
        }:
            raise TaskContractError(f"{task_id}: verifier {index} has unsupported comparison")

    tool_owner_corrections = _known_tool_owner_corrections(
        task_id=task_id,
        domain=row["domain"],
        user_prompt=row["user_prompt"],
        selected_tools=selected_tools,
        restricted_tools=restricted_tools,
        gym_servers=gym_servers,
    )
    missing_tool_corrections = _known_missing_tool_corrections(
        task_id=task_id,
        domain=row["domain"],
        user_prompt=row["user_prompt"],
        selected_tools=selected_tools,
        restricted_tools=restricted_tools,
        gym_servers=gym_servers,
    )
    return EnterpriseOpsTask(
        task_id=task_id,
        domain=row["domain"],
        system_prompt=row["system_prompt"],
        user_prompt=row["user_prompt"],
        selected_tools=selected_tools,
        restricted_tools=restricted_tools,
        mcp_endpoint=row["mcp_endpoint"],
        number_of_runs=row["number_of_runs"],
        reset_database_between_runs=row["reset_database_between_runs"],
        gym_servers=tuple(gym_servers),
        verifiers=tuple(verifiers),
        official_revised_overlays=official_revised_overlays,
        tool_owner_corrections=tool_owner_corrections,
        missing_tool_corrections=missing_tool_corrections,
        selected_tool_addition_corrections=selected_tool_addition_corrections,
        context_key_corrections=context_key_corrections,
        context_value_corrections=context_value_corrections,
        seed_database_file_corrections=seed_database_file_corrections,
        verifier_corrections=verifier_corrections,
        verifier_expected_value_corrections=verifier_expected_value_corrections,
        verifier_contract_corrections=verifier_contract_corrections,
        verifier_set_normalization=verifier_set_normalization,
    )


def load_tasks(data_path: Path, *, expected_domain: str, expected_count: int) -> list[EnterpriseOpsTask]:
    if not data_path.is_file():
        raise TaskContractError(f"task parquet not found: {data_path}")
    rows = pq.read_table(data_path).to_pylist()
    if len(rows) != expected_count:
        raise TaskContractError(
            f"expected {expected_count} raw tasks in {data_path}, found {len(rows)}"
        )
    mode = data_path.parent.name
    rows.extend(
        official_revised_supplement_rows(domain=expected_domain, mode=mode)
    )
    tasks = [task_from_row(row) for row in rows]
    ids = [task.task_id for task in tasks]
    if len(set(ids)) != len(ids):
        raise TaskContractError(f"duplicate task_id found in {data_path}")
    wrong_domains = sorted({task.domain for task in tasks if task.domain != expected_domain})
    if wrong_domains:
        raise TaskContractError(
            f"expected domain {expected_domain!r}, found {wrong_domains}"
        )
    return sorted(tasks, key=lambda task: task.task_id)


def load_task_scope(
    tasks: Iterable[EnterpriseOpsTask],
    *,
    scope: str,
    scopes: Mapping[str, Any],
) -> list[EnterpriseOpsTask]:
    by_id = {task.task_id: task for task in tasks}
    if scope not in scopes:
        raise TaskContractError(f"scope {scope!r} is not registered")
    selection = scopes[scope]
    if selection == "all":
        return sorted(by_id.values(), key=lambda task: task.task_id)
    if not isinstance(selection, list) or not all(isinstance(item, str) for item in selection):
        raise TaskContractError(f"scope {scope!r} must be 'all' or a task_id list")
    if len(selection) != len(set(selection)):
        raise TaskContractError(f"scope {scope!r} contains duplicate task IDs")
    missing = [task_id for task_id in selection if task_id not in by_id]
    if missing:
        raise TaskContractError(f"scope {scope!r} contains unknown task IDs: {missing}")
    return [by_id[task_id] for task_id in selection]


def validate_seed_members(tasks: Iterable[EnterpriseOpsTask], archive_path: Path) -> None:
    if not archive_path.is_file():
        raise TaskContractError(f"database archive not found: {archive_path}")
    with ZipFile(archive_path) as archive:
        members = set(archive.namelist())
    missing = sorted(
        {
            server["seed_database_file"]
            for task in tasks
            for server in task.gym_servers
            if server["seed_database_file"] not in members
        }
    )
    if missing:
        raise TaskContractError(f"database archive is missing seed files: {missing}")
