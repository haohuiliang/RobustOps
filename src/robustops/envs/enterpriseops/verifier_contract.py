"""Pinned repairs and lossless normalization for EnterpriseOps verifiers.

The upstream Parquet files remain immutable.  Every semantic repair below is
limited to one task and validates the exact broken verifier before changing it.
After semantic repairs, exact duplicate verifiers are folded and non-identical
same-name checks receive stable unique names so name-keyed executors cannot
silently overwrite them.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping


class VerifierContractSourceError(ValueError):
    """A pinned verifier repair no longer matches its fixed source row."""


@dataclass(frozen=True)
class VerifierContractCorrection:
    """Provenance for one task-level semantic verifier repair."""

    source_verifier_count: int
    source_unique_verifier_count: int
    removed_verifier_count: int
    replaced_verifier_count: int
    added_verifier_count: int
    rationale: str


@dataclass(frozen=True)
class VerifierNameCorrection:
    """One stable rename that prevents a same-name result overwrite."""

    index: int
    source_name: str
    target_name: str


@dataclass(frozen=True)
class VerifierSetNormalization:
    """Source/effective counts for exact folding and unique-name assignment."""

    source_count: int
    effective_count: int
    exact_duplicate_count: int
    name_corrections: tuple[VerifierNameCorrection, ...]


@dataclass(frozen=True)
class _VerifierSpec:
    name: str
    query: str
    expected_value: Any = 1
    comparison_type: str = "equals"
    gym_name: str | None = None


@dataclass(frozen=True)
class _Replacement:
    source: _VerifierSpec
    query: str | None = None
    expected_value: Any | None = None
    comparison_type: str | None = None
    name: str | None = None


@dataclass(frozen=True)
class _TaskPatch:
    domain: str
    prompt_fragment: str
    source_count: int | tuple[int, ...]
    source_unique_count: int
    rationale: str
    remove: tuple[_VerifierSpec, ...] = ()
    replace: tuple[_Replacement, ...] = ()
    add: tuple[Mapping[str, Any], ...] = ()


def _sql(value: str) -> str:
    normalized = " ".join(value.split())
    return re.sub(r"\s*([(),;])\s*", r"\1", normalized)


def _spec(
    name: str,
    query: str,
    *,
    expected_value: Any = 1,
    comparison_type: str = "equals",
    gym_name: str | None = None,
) -> _VerifierSpec:
    return _VerifierSpec(
        name=name,
        query=_sql(query),
        expected_value=expected_value,
        comparison_type=comparison_type,
        gym_name=gym_name,
    )


def _new(
    name: str,
    description: str,
    query: str,
    *,
    gym_name: str,
    expected_value: Any = 1,
    comparison_type: str = "equals",
) -> Mapping[str, Any]:
    return {
        "verifier_type": "database_state",
        "name": name,
        "description": description,
        "gym_name": gym_name,
        "validation_config": {
            "query": query,
            "expected_value": expected_value,
            "comparison_type": comparison_type,
        },
    }


_CSM_TRANSITION_RESOLVED = _spec(
    "Transition Case Created",
    """SELECT COUNT(*) AS cnt FROM customer_case WHERE account_id = 4
    AND contact_id = 3 AND priority = 'moderate' AND state = 'resolved'
    AND channel = 'web';""",
    gym_name="sn-csm-server",
)
_CSM_RESOLVED_NOTIFICATION = _spec(
    "Notification Sent to New Primary",
    """SELECT COUNT(*) AS cnt FROM notification
    WHERE email = 'derek.blake7@globex.com' AND type = 'update'
    AND case_id = (SELECT case_id FROM customer_case WHERE account_id = 4
    AND contact_id = 3 AND priority = 'moderate' AND state = 'resolved'
    ORDER BY case_id DESC LIMIT 1);""",
    gym_name="sn-csm-server",
)
_CSM_REDUNDANT_PRIMARY = _spec(
    "Update Primary Contact",
    "SELECT is_primary FROM contact WHERE contact_id = 3;",
    gym_name="sn-csm-server",
)

_ITSM_PHONE_INCIDENT = _spec(
    "Report incident",
    """SELECT COUNT(*) FROM incident WHERE incident_id = "INC_024"
    AND configuration_item = "CI_005" AND caller_id = "USER_005"
    AND channel = "phone" AND status = "new" AND category = "hardware"
    AND impact = "low" AND urgency = "high" AND priority = "moderate";""",
    gym_name="gym-itsm-mcp",
)
_ITSM_AISHA_HIGH = _spec(
    "Verify if the incident is created correctly.",
    """SELECT COUNT(*) FROM incident WHERE incident_id = 'INC_024'
    AND impact = 'high' AND priority = 'high' AND urgency = 'high'
    AND caller_id = 'USER_004';""",
    gym_name="gym-itsm-mcp",
)
_ITSM_AISHA_REMINDER = _spec(
    "Verify if the notification is sent correctly.",
    """SELECT COUNT(*) FROM notification WHERE notification_id = 'NOTIF_008'
    AND email = 'aisha.williams@techcorp.com' AND incident_id = 'INC_024'
    AND type = 'reminder';""",
    gym_name="gym-itsm-mcp",
)
_ITSM_AISHA_OTHER = _spec(
    "Verify if the notification is sent correctly.",
    """SELECT COUNT(*) FROM notification WHERE notification_id = 'NOTIF_008'
    AND email = 'aisha.williams@techcorp.com' AND incident_id = 'INC_024'
    AND type = 'other';""",
    gym_name="gym-itsm-mcp",
)

_HR_WRONG_SPECS = (
    _spec(
        "Verify task Creation",
        """SELECT COUNT(*) AS cnt FROM hr_case_task WHERE task_type = 'mark when complete'
        AND short_description = 'Verify routing number with bank portal'
        AND assignment_group = 3 AND assigned_to = 11 AND status = 'ready'
        AND parent_case = 1;""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Send Notification",
        """SELECT COUNT(*) AS cnt FROM notification WHERE hr_case_id = 1
        AND email = 'lauren.diaz70@acme.com' AND type = 'alert'
        AND status = 'work_in_progress';""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Verify Case priority and State",
        """SELECT COUNT(*) AS cnt FROM hr_case WHERE hr_case_id = 47
        AND status = 'work_in_progress' AND priority = 'critical'
        AND assignment_group = 3 AND assigned_to = 11;""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Send Notification",
        """SELECT COUNT(*) AS cnt FROM notification WHERE hr_case_id = 47
        AND email = 'lauren.diaz70@acme.com' AND type = 'alert'
        AND status = 'work_in_progress';""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Verify task Creation",
        """SELECT COUNT(*) AS cnt FROM hr_case_task WHERE task_type = 'mark when complete'
        AND short_description = 'Verify routing number with bank portal'
        AND assignment_group = 3 AND assigned_to = 11 AND status = 'ready'
        AND parent_case = 47;""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Verify Case priority and State",
        """SELECT COUNT(*) AS cnt FROM hr_case WHERE hr_case_id = 47
        AND status = 'work_in_progress' AND priority = 'critical'
        AND assignment_group = 18 AND assigned_to = 40;""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Verify task Creation",
        """SELECT COUNT(*) AS cnt FROM hr_case_task WHERE task_type = 'mark when complete'
        AND short_description = 'Verify routing number with bank portal'
        AND assignment_group = 18 AND assigned_to = 40 AND status = 'ready'
        AND parent_case = 47;""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Verify Case priority and State",
        """SELECT COUNT(*) AS cnt FROM hr_case WHERE hr_case_id = 47
        AND status = 'work_in_progress' AND priority = 'critical'
        AND assignment_group = 18;""",
        gym_name="sn-hr-internal",
    ),
    _spec(
        "Verify task Creation",
        """SELECT COUNT(*) AS cnt FROM hr_case_task WHERE task_type = 'mark when complete'
        AND short_description = 'Verify routing number with bank portal'
        AND assignment_group = 18 AND status = 'ready' AND parent_case = 47;""",
        gym_name="sn-hr-internal",
    ),
)

_HR_DENISE_APPLIED = _spec(
    "Verify Knowledge",
    """SELECT COUNT(*) FROM hr_case_knowledge WHERE hr_case_kb_id= 13
    AND hr_case_id =2 AND knowledge_id = 2 AND used_as = 'applied';""",
    gym_name="sn-hr-internal",
)
_HR_DENISE_SUGGESTED = _spec(
    "Verify Knowledge",
    """SELECT COUNT(*) FROM hr_case_knowledge WHERE hr_case_kb_id= 13
    AND hr_case_id =2 AND knowledge_id = 2 AND used_as = 'suggested';""",
    gym_name="sn-hr-internal",
)
_HR_DENISE_TASK = _spec(
    "Verify Case Task",
    """SELECT COUNT(*) FROM hr_case_task WHERE hr_case_task_id = 150
    AND number = 'TASK-0001' AND status = 'draft' AND parent_case =2;""",
    gym_name="sn-hr-internal",
)


def _calendar_time_replacement(name: str, source_query: str) -> _Replacement:
    target = source_query
    for pattern in ("start_datetime", "end_datetime"):
        # Replace exact ISO-T equality with a wall-clock comparison that accepts
        # SQLAlchemy's space separator and microseconds without changing timezone
        # interpretation or the intended local time.
        target = re.sub(
            rf"{pattern}\s*=\s*'([^']+)'",
            lambda match: (
                f"substr(replace({pattern}, 'T', ' '), 1, 19) = "
                f"'{match.group(1).replace('T', ' ')[:19]}'"
            ),
            target,
        )
    return _Replacement(source=_spec(name, source_query, gym_name="gym-calendar"), query=target)


_CALENDAR_PATCHES = {
    "task_20251217_081417_887_0a0bf089_487054de": (
        "hybrid",
        "High-Priority Incident Review",
        4,
        _calendar_time_replacement(
            "Verify priority incident review event created",
            """SELECT COUNT(*) FROM events WHERE summary = 'High-Priority Incident Review'
            AND start_datetime = '2025-12-20T16:00:00-05:00'
            AND end_datetime = '2025-12-20T16:30:00-05:00'
            AND calendar_id = 'aaron_le-primary';""",
        ),
    ),
    "task_20251220_151043_591_0a0bf089_22634f28": (
        "hybrid",
        "Rescheduled training call",
        5,
        _calendar_time_replacement(
            "Verify rescheduled training call event created",
            """SELECT COUNT(*) FROM events WHERE calendar_id = 'aaron_le-primary'
            AND summary = 'INC-000040 – Rescheduled training call'
            AND start_datetime = '2025-12-23T14:00:00'
            AND end_datetime = '2025-12-23T15:00:00';""",
        ),
    ),
    "task_20260121_171958_007_cbcf38f8_0349e59e": (
        "calendar",
        "Team Sync",
        9,
        _calendar_time_replacement(
            "Team Sync event scheduled",
            """SELECT COUNT(*) FROM events WHERE calendar_id = 'alice-team'
            AND summary = 'Team Sync'
            AND start_datetime = '2025-12-12T15:30:00+00:00'
            AND end_datetime = '2025-12-12T16:30:00+00:00'
            AND start_timezone = 'Asia/Kolkata' AND end_timezone = 'Asia/Kolkata';""",
        ),
    ),
    "task_20251224_153415_365_0a0bf089_01602883": (
        "hybrid",
        "Support Planning",
        3,
        _calendar_time_replacement(
            "Verify planning session scheduled",
            """SELECT COUNT(*) FROM events
            WHERE summary = 'Wayne Enterprises – Support Planning Session'
            AND start_datetime = '2026-01-15T13:00:00-05:00'
            AND end_datetime = '2026-01-15T14:00:00-05:00';""",
        ),
    ),
    "task_20251124_112741_995_0a0bf089_3906a767": (
        "calendar",
        "Helios Kickoff",
        5,
        _calendar_time_replacement(
            "Verify event creation",
            """SELECT COUNT(*) AS count FROM events WHERE summary = 'Helios Kickoff'
            AND start_datetime = '2026-01-15T11:00:00'
            AND end_datetime = '2026-01-15T12:00:00';""",
        ),
    ),
    "task_20251202_171004_871_0a0bf089_489e2b2d": (
        "calendar",
        "Orion Launch Briefing",
        5,
        _calendar_time_replacement(
            "Verify event creation",
            """SELECT COUNT(*) AS count FROM events WHERE summary = 'Orion Launch Briefing'
            AND start_datetime = '2026-01-15T11:00:00'
            AND end_datetime = '2026-01-15T12:00:00';""",
        ),
    ),
    "task_20251205_032409_945_0a0bf089_61f5be29": (
        "calendar",
        "Q1 Launch Readiness Review",
        3,
        _calendar_time_replacement(
            "Verify Q1 Launch Readiness Review event",
            """SELECT COUNT(*) AS count FROM events
            WHERE summary = 'Q1 Launch Readiness Review'
            AND start_datetime = '2026-01-15T10:00:00-08:00'
            AND end_datetime = '2026-01-15T12:00:00-08:00'
            AND calendar_id IN (SELECT calendar_id FROM calendars
            WHERE summary = 'Product Rollouts');""",
        ),
    ),
}


_TASK_PATCHES: dict[str, _TaskPatch] = {
    "task_20251219_111409_528_c5ca08eb_db61cf35": _TaskPatch(
        domain="itsm",
        prompt_fragment="link the same service, service offering, and configuration item",
        source_count=1,
        source_unique_count=1,
        rationale=(
            "The source rewards missing or incorrect service linkage. Require the new "
            "Travis incident to exist with the reference incident's service, offering "
            "and CI; absence must fail rather than pass by reversing expected_value."
        ),
        replace=(
            _Replacement(
                _spec(
                    "Verify Service and Service Offering Linkage",
                    """SELECT COUNT(*) FROM incident where incident_id="INC_101"
                    AND ( service IS NULL OR service_offering IS NULL
                    OR service != 'SVC_020' OR service_offering != 'SVCOFF_025')""",
                    gym_name="gym-itsm-mcp",
                ),
                query="""SELECT COUNT(*) FROM incident
                WHERE incident_id = 'INC_101' AND caller_id = 'USER_011'
                AND service = 'SVC_020' AND service_offering = 'SVCOFF_025'
                AND configuration_item = 'CI_072'""",
            ),
        ),
    ),
    "task_20251204_231522_245_e6427839_5d09c3c2": _TaskPatch(
        domain="csm",
        prompt_fragment="Find the case regarding this and escalate it, also notify the primary contact",
        source_count=22,
        source_unique_count=3,
        rationale=(
            "The request requires escalation and primary-contact notification; policy "
            "requires customer_request as the escalation reason. Neither specifies "
            "the exact high priority enum. Remove the unsupported priority constraint "
            "and retain the source escalation check without it and the notification check."
        ),
        remove=(
            _spec(
                "Verify if the case was escalated.",
                "SELECT COUNT(*) FROM customer_case WHERE case_id = '102' AND account_id = '5' "
                "AND product_id = '48' AND escalation = '1' "
                "AND escalation_reason = 'customer_request' AND priority = 'high';",
                gym_name="sn-csm-server",
            ),
        ),
    ),
    "task_20251201_164641_724_109c89c0_1b29acb9": _TaskPatch(
        domain="csm",
        prompt_fragment="primary contact has resigned",
        source_count=(44, 36),
        source_unique_count=7,
        rationale=(
            "Keep the policy-default new handover case, remove the contradictory resolved "
            "copy, and verify deactivation of the former contact and portal user."
        ),
        remove=(_CSM_TRANSITION_RESOLVED, _CSM_RESOLVED_NOTIFICATION, _CSM_REDUNDANT_PRIMARY),
        add=(
            _new(
                "Former Primary Contact Deactivated",
                "The former Globex primary contact is inactive and is no longer primary.",
                "SELECT COUNT(*) FROM contact WHERE contact_id = 2 AND active = 0 AND is_primary = 0;",
                gym_name="sn-csm-server",
            ),
            _new(
                "Former Primary Portal User Deactivated",
                "The portal user linked to the former Globex primary contact is inactive.",
                """SELECT COUNT(*) FROM user WHERE user_id = 347 AND active = 0
                AND email = 'joshua.morgan@globex.com';""",
                gym_name="sn-csm-server",
            ),
        ),
    ),
    "task_20251210_205620_593_391f1eff_ad65c160": _TaskPatch(
        domain="itsm",
        prompt_fragment="WIFI-UK-005",
        source_count=10,
        source_unique_count=3,
        rationale="Use the policy-default chat channel; remove the contradictory phone copy.",
        remove=(_ITSM_PHONE_INCIDENT,),
    ),
    "task_20251212_184405_886_e6427839_6413728a": _TaskPatch(
        domain="itsm",
        prompt_fragment="scanned copy of her identity proof",
        source_count=18,
        source_unique_count=5,
        rationale=(
            "High impact and high urgency imply critical priority; the first identity-proof "
            "request is an other notification, not a reminder."
        ),
        remove=(_ITSM_AISHA_HIGH, _ITSM_AISHA_REMINDER),
        replace=(
            _Replacement(
                source=_ITSM_AISHA_OTHER,
                query="""SELECT COUNT(*) FROM notification
                WHERE notification_id = 'NOTIF_008'
                AND email = 'aisha.williams@techcorp.com'
                AND incident_id = 'INC_024' AND type = 'other'
                AND LOWER(message) LIKE '%identity%'
                AND LOWER(message) LIKE '%proof%';""",
            ),
        ),
    ),
    "task_20251219_155215_361_00364ece_c1c6ffc8": _TaskPatch(
        domain="hr",
        prompt_fragment="HRC0000711",
        source_count=30,
        source_unique_count=12,
        rationale=(
            "Keep only HRC0000711, Travis Wood, and Global Service Desk; remove copied Lauren, "
            "case 47, group 18, and assignee 40 worlds."
        ),
        remove=_HR_WRONG_SPECS,
    ),
    "task_20251226_104747_127_30600918_559da326": _TaskPatch(
        domain="hr",
        prompt_fragment="using for the resolution",
        source_count=20,
        source_unique_count=5,
        rationale=(
            "The prompt explicitly uses the article for resolution and never requests a new "
            "case task; require used_as=resolution and drop the hidden task check."
        ),
        remove=(_HR_DENISE_SUGGESTED, _HR_DENISE_TASK),
        replace=(
            _Replacement(
                source=_HR_DENISE_APPLIED,
                query="""SELECT COUNT(*) FROM hr_case_knowledge
                WHERE hr_case_kb_id = 13 AND hr_case_id = 2 AND knowledge_id = 2
                AND used_as = 'resolution';""",
            ),
        ),
    ),
}


for _task_id, (_domain, _fragment, _count, _replacement) in _CALENDAR_PATCHES.items():
    _TASK_PATCHES[_task_id] = _TaskPatch(
        domain=_domain,
        prompt_fragment=_fragment,
        source_count=_count,
        source_unique_count=_count,
        rationale=(
            "Compare the intended wall-clock value after normalizing only the ISO separator "
            "and stored microseconds; do not change the requested time or timezone."
        ),
        replace=(_replacement,),
    )


_TASK_PATCHES.update(
    {
        "task_20251204_115043_305_e81f6083_4acfd9ea": _TaskPatch(
            domain="calendar",
            prompt_fragment="removing all existing access permissions",
            source_count=10,
            source_unique_count=10,
            rationale=(
                "Require the requested product-team reader to be the only non-owner ACL; "
                "the source query proves that row exists but does not reject leftover access."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "verify_team_calendar_only_product_team_reader",
                        """SELECT COUNT(*) FROM acls a JOIN calendars c
                        ON a.calendar_id = c.calendar_id JOIN scopes s
                        ON a.scope_id = s.id WHERE c.user_id = 'alice_manager'
                        AND (c.summary LIKE '%Team%Archived%'
                        OR c.summary LIKE '%Team%Archive%'
                        OR c.summary LIKE '%archived%'
                        OR c.summary LIKE '%Archive%') AND a.role != 'owner'
                        AND s.type = 'group'
                        AND s.value = 'product-team@techcorp.com'
                        AND a.role = 'reader';""",
                        gym_name="gym-calendar",
                    ),
                    query="""SELECT COUNT(*) FROM calendars c
                    WHERE c.user_id = 'alice_manager'
                    AND (c.summary LIKE '%Team%Archived%'
                      OR c.summary LIKE '%Team%Archive%'
                      OR c.summary LIKE '%archived%'
                      OR c.summary LIKE '%Archive%')
                    AND (SELECT COUNT(*) FROM acls a
                      WHERE a.calendar_id = c.calendar_id
                      AND a.role != 'owner') = 1
                    AND EXISTS (SELECT 1 FROM acls a
                      JOIN scopes s ON a.scope_id = s.id
                      WHERE a.calendar_id = c.calendar_id
                      AND a.role = 'reader' AND s.type = 'group'
                      AND s.value = 'product-team@techcorp.com');""",
                ),
            ),
        ),
        "task_20251204_153830_645_1628b966_3df25b52": _TaskPatch(
            domain="email",
            prompt_fragment="Delete that draft immediately",
            source_count=3,
            source_unique_count=3,
            rationale="Check Subject and To as two header rows instead of one impossible row.",
            replace=(
                _Replacement(
                    source=_spec(
                        "Verify Draft Deleted",
                        """SELECT COUNT(D.id) FROM drafts AS D JOIN messages AS M
                        ON D.message_id = M.id JOIN message_parts AS MP ON M.id = MP.message_id
                        JOIN message_headers AS MH ON MP.id = MH.message_part_id
                        WHERE M.user_id = 'user_001' AND MH.name = 'Subject'
                        AND MH.value = 'Salaries' AND MH.name = 'To'
                        AND MH.value = 'all-staff@company.com';""",
                        expected_value=0,
                        gym_name="gym-email-mcp",
                    ),
                    query="""SELECT COUNT(D.id) FROM drafts AS D
                    JOIN messages AS M ON D.message_id = M.id
                    WHERE M.user_id = 'user_001'
                    AND EXISTS (SELECT 1 FROM message_parts MP
                      JOIN message_headers MH ON MP.id = MH.message_part_id
                      WHERE MP.message_id = M.id AND MH.name = 'Subject'
                      AND MH.value = 'Salaries')
                    AND EXISTS (SELECT 1 FROM message_parts MP
                      JOIN message_headers MH ON MP.id = MH.message_part_id
                      WHERE MP.message_id = M.id AND MH.name = 'To'
                      AND MH.value = 'all-staff@company.com');""",
                ),
            ),
        ),
        "task_20260107_141130_029_1628b966_8e264839": _TaskPatch(
            domain="email",
            prompt_fragment="VIP-Board",
            source_count=2,
            source_unique_count=2,
            rationale="Restore the label-name predicate that the inline SQL comment disabled.",
            replace=(
                _Replacement(
                    source=_spec(
                        "Verify Priority Automation",
                        """SELECT COUNT(F.id) FROM filters AS F JOIN labels AS L
                        ON F.user_id = L.user_id
                        WHERE F.user_id = '05a1fa8d-35a9-4a77-a57f-fa0751e7a99e' --
                        AND L.name = 'Vip-Board'
                        AND json_extract(F.criteria_json, '$.from') = 'board@corp.com'
                        AND json_extract(F.action_json, '$.addLabelIds') LIKE '%' || L.id || '%';""",
                        gym_name="gym-email-mcp",
                    ),
                    query="""SELECT COUNT(F.id) FROM filters AS F JOIN labels AS L
                    ON F.user_id = L.user_id
                    WHERE F.user_id = '05a1fa8d-35a9-4a77-a57f-fa0751e7a99e'
                    AND L.name = 'VIP-Board'
                    AND json_extract(F.criteria_json, '$.from') = 'board@corp.com'
                    AND json_extract(F.action_json, '$.addLabelIds') LIKE '%' || L.id || '%';""",
                ),
            ),
        ),
        "task_20260105_212045_832_e0ac0c6d_84c7f524": _TaskPatch(
            domain="calendar",
            prompt_fragment="Q4 Technical Roadmap Review",
            source_count=6,
            source_unique_count=6,
            rationale="Make the third event a positive existence check like its peer events.",
            replace=(
                _Replacement(
                    source=_spec(
                        "Q4 Technical Roadmap Review Event Created",
                        """SELECT COUNT(*) as count FROM events
                        WHERE calendar_id = 'alice-projects'
                        AND summary = 'Q4 Technical Roadmap Review'
                        AND start_datetime > datetime('now')""",
                        expected_value=0,
                        gym_name="gym-calendar",
                    ),
                    comparison_type="greater_than",
                ),
            ),
        ),
        "task_20260108_075149_000_df0ecdeb_ead1b33a": _TaskPatch(
            domain="teams",
            prompt_fragment="Welcome to Operations Hub",
            source_count=5,
            source_unique_count=5,
            rationale="Remove the accidental second SQL statement delimiter.",
            replace=(
                _Replacement(
                    source=_spec(
                        "Welcome Message Posted Today",
                        """SELECT COUNT(*) AS count FROM channel_messages
                        WHERE team_id = 'team_techcorp_001'
                        AND DATE(created_datetime) = CURRENT_DATE
                        AND body_json LIKE '%Welcome to Operations Hub, the new space for managing collaboration, project updates, and cross-department operations.%';;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channel_messages m
                    JOIN channels c ON c.channel_id = m.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Operations Hub'
                    AND LOWER(m.body_json) LIKE '%welcome%operations hub%'
                    AND LOWER(m.body_json) LIKE '%collaboration%'
                    AND LOWER(m.body_json) LIKE '%project update%'
                    AND LOWER(m.body_json) LIKE '%cross-department operation%';""",
                ),
            ),
        ),
        "task_20260107_174701_432_464ee3e0_16e768aa": _TaskPatch(
            domain="teams",
            prompt_fragment="Call Quality Analysis Report",
            source_count=3,
            source_unique_count=3,
            rationale=(
                "Move the explanatory comment outside SQL and check the requested report's "
                "call types, durations, participant counts, and issue follow-up."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "Check Summary Report Message Posted in Channel",
                        """SELECT COUNT(*) AS verification_result FROM channel_messages AS m
                        INNER JOIN channels AS c ON m.channel_id = c.channel_id
                        WHERE c.team_id = 'team_techcorp_001'
                        AND c.display_name = 'Call Quality Review'
                        -- The key to robustness: Convert both the column content and the search terms to lower case
                        AND LOWER(m.body_json) LIKE '%call quality analysis report%'
                        AND LOWER(m.body_json) LIKE '%call records%'
                        AND LOWER(m.body_json) LIKE '%past month%';""",
                        expected_value=0,
                        comparison_type="greater_than",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) AS verification_result
                    FROM channel_messages AS m
                    INNER JOIN channels AS c ON m.channel_id = c.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Call Quality Review'
                    AND LOWER(m.body_json) LIKE '%call quality analysis report%'
                    AND LOWER(m.body_json) LIKE '%call records%'
                    AND LOWER(m.body_json) LIKE '%past month%'
                    AND LOWER(m.body_json) LIKE '%call type%'
                    AND LOWER(m.body_json) LIKE '%duration%'
                    AND LOWER(m.body_json) LIKE '%participant%'
                    AND LOWER(m.body_json) LIKE '%issue%';""",
                ),
            ),
        ),
    }
)


_TASK_PATCHES.update(
    {
        "task_20251129_042357_079_d809b0d6_b11e3d3b": _TaskPatch(
            domain="teams",
            prompt_fragment="primary space for integration-related updates",
            source_count=4,
            source_unique_count=4,
            rationale=(
                "Require both documentation tabs to use OneNote and add the missing final "
                "announcement check in the Development channel."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "Verify the Development channel contains both required documentation tabs",
                        """SELECT CASE WHEN (
                          SELECT COUNT(*) FROM teams_channel_tabs ct
                          JOIN channels c ON ct.channel_id = c.channel_id
                          JOIN teams t ON c.team_id = t.team_id
                          WHERE LOWER(t.display_name) LIKE '%techcorp%'
                          AND LOWER(c.display_name) LIKE '%development%'
                          AND ct.display_name IN ('API Notes', 'Release Notes')
                        ) = 2 THEN 1 ELSE 0 END AS verification_passed;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT tab.display_name) = 2
                    THEN 1 ELSE 0 END
                    FROM teams_channel_tabs tab
                    JOIN channels c ON c.channel_id = tab.channel_id
                    JOIN teams t ON t.team_id = c.team_id
                    WHERE LOWER(t.display_name) = 'techcorp solutions team'
                    AND LOWER(c.display_name) = 'development'
                    AND tab.display_name IN ('API Notes', 'Release Notes')
                    AND tab.app_id IN (SELECT app_id FROM teams_apps
                      WHERE LOWER(display_name) = 'onenote');""",
                ),
            ),
            add=(
                _new(
                    "Development Integration Announcement Posted",
                    "The team is told that Development is the primary integration-update space.",
                    """SELECT COUNT(*) FROM channel_messages m
                    JOIN channels c ON c.channel_id = m.channel_id
                    JOIN teams t ON t.team_id = c.team_id
                    WHERE LOWER(t.display_name) = 'techcorp solutions team'
                    AND LOWER(c.display_name) = 'development'
                    AND LOWER(m.body_json) LIKE '%primary%'
                    AND LOWER(m.body_json) LIKE '%integration%'
                    AND LOWER(m.body_json) LIKE '%update%';""",
                    gym_name="gym-teams-mcp",
                ),
            ),
        ),
        "task_20251205_150335_071_464ee3e0_19d263cd": _TaskPatch(
            domain="teams",
            prompt_fragment="Acme Q4 Strategy Review",
            source_count=6,
            source_unique_count=6,
            rationale=(
                "Use the task's fixed December 5 date to check the next-Friday Pacific "
                "schedule, keep both call checks on the exact two participants, require John "
                "Davis as co-organizer, and remove hidden message metadata."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "Verifier 1: Call Created with EOD Security Checkpoint",
                        """SELECT COUNT(*) FROM calls
                        WHERE subject = 'EOD Security Checkpoint'
                        AND created_at >= datetime('now', '-5 minutes')
                        AND (LOWER(targets) LIKE '%alice%johnson%'
                          OR LOWER(targets) LIKE '%alice.johnson%')
                        AND (LOWER(targets) LIKE '%mike%thompson%'
                          OR LOWER(targets) LIKE '%mike.thompson%')
                        AND soft_delete = 0;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM calls
                    WHERE subject = 'EOD Security Checkpoint'
                    AND created_at >= datetime('now', '-5 minutes')
                    AND json_array_length(targets) = 2
                    AND LOWER(targets) LIKE '%alice%johnson%'
                    AND LOWER(targets) LIKE '%mike%thompson%'
                    AND soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "Verifier 2: Call Content Sharing Disabled",
                        """SELECT COUNT(*) FROM calls
                        WHERE subject = 'EOD Security Checkpoint'
                        AND created_at >= datetime('now', '-5 minutes')
                        AND (LOWER(call_options) LIKE '%iscontentsharingnotificationenabled%false%'
                          OR LOWER(call_options) LIKE '%\"iscontentsharingnotificationenabled\":false%')
                        AND soft_delete = 0;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM calls
                    WHERE subject = 'EOD Security Checkpoint'
                    AND created_at >= datetime('now', '-5 minutes')
                    AND json_array_length(targets) = 2
                    AND LOWER(targets) LIKE '%alice%johnson%'
                    AND LOWER(targets) LIKE '%mike%thompson%'
                    AND (LOWER(call_options) LIKE '%iscontentsharingnotificationenabled%false%'
                      OR LOWER(call_options) LIKE '%\"iscontentsharingnotificationenabled\":false%')
                    AND soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "Verifier 5: Townhall Scheduled for Friday 2-3 PM Pacific",
                        """SELECT COUNT(*) FROM virtual_event_townhalls
                        WHERE display_name = 'Acme Q4 Strategy Review'
                        AND strftime('%w', start_date_time) = '5'
                        AND strftime('%H:%M', start_date_time) = '14:00'
                        AND strftime('%H:%M', end_date_time) = '15:00'
                        AND start_date_time > datetime('now')
                        AND start_date_time < datetime('now', '+7 days')
                        AND created_at >= datetime('now', '-5 minutes');""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM virtual_event_townhalls
                    WHERE display_name = 'Acme Q4 Strategy Review'
                    AND substr(replace(start_date_time, 'T', ' '), 1, 16) = '2025-12-12 14:00'
                    AND substr(replace(end_date_time, 'T', ' '), 1, 16) = '2025-12-12 15:00'
                    AND start_time_zone = 'Pacific Standard Time'
                    AND end_time_zone = 'Pacific Standard Time';""",
                ),
                _Replacement(
                    source=_spec(
                        "Verifier 6: Urgent MFA Policy Message Sent to Development Channel",
                        """SELECT COUNT(*) FROM channel_messages
                        WHERE channel_id IN (SELECT channel_id FROM channels
                        WHERE display_name = 'Development' AND team_id IN
                        (SELECT team_id FROM teams WHERE display_name LIKE '%TechCorp Solutions%'))
                        AND (LOWER(body_json) LIKE '%mfa%policy%'
                        OR LOWER(body_json) LIKE '%two-factor%authentication%')
                        AND importance = 'urgent'
                        AND created_datetime >= datetime('now', '-5 minutes')
                        AND (LOWER(from_json) LIKE '%james%wilson%'
                        OR LOWER(from_json) LIKE '%james.wilson%');""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channel_messages m
                    JOIN channels c ON c.channel_id = m.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Development'
                    AND LOWER(m.body_json) LIKE '%mfa policy%'
                    AND LOWER(m.body_json) LIKE '%monday%'
                    AND LOWER(m.body_json) LIKE '%two-factor authentication%'
                    AND LOWER(m.body_json) LIKE '%eod friday%'
                    AND LOWER(m.body_json) LIKE '%it portal%';""",
                ),
            ),
            add=(
                _new(
                    "Acme Q4 Strategy Review Has John Davis Co-organizer",
                    "John Davis is a co-organizer of the requested townhall.",
                    """SELECT CASE WHEN EXISTS (
                      SELECT 1 FROM virtual_event_townhall_coorganizers co
                      JOIN virtual_event_townhalls v ON v.id = co.townhall_id
                      WHERE v.display_name = 'Acme Q4 Strategy Review'
                      AND co.user_id = 'john.davis'
                    ) THEN 1 ELSE 0 END;""",
                    gym_name="gym-teams-mcp",
                ),
            ),
        ),
        "task_20260105_163515_719_8e9e30d7_d86e69de": _TaskPatch(
            domain="teams",
            prompt_fragment="direct audio call between Emma and Frank",
            source_count=6,
            source_unique_count=6,
            rationale=(
                "Require the newly needed HR Knowledge Hub tab to use SharePoint and add the "
                "missing audio-call check for both named participants."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "hr_tab_configured_in_general",
                        """SELECT COUNT(*) AS count FROM teams_channel_tabs t
                        JOIN channels c ON c.channel_id = t.channel_id
                        JOIN teams tm ON tm.team_id = c.team_id
                        WHERE LOWER(tm.display_name) = 'globalex industries team'
                        AND LOWER(c.display_name) = 'general'
                        AND LOWER(t.display_name) = 'hr knowledge hub';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM teams_channel_tabs tab
                    JOIN channels c ON c.channel_id = tab.channel_id
                    JOIN teams tm ON tm.team_id = c.team_id
                    WHERE LOWER(tm.display_name) = 'globalex industries team'
                    AND LOWER(c.display_name) = 'general'
                    AND LOWER(tab.display_name) = 'hr knowledge hub'
                    AND tab.app_id IN (SELECT app_id FROM teams_apps
                      WHERE LOWER(display_name) = 'sharepoint');""",
                ),
            ),
            add=(
                _new(
                    "Emma and Frank Audio Call Created",
                    "An audio call includes both Emma Wilson and Frank Chen.",
                    """SELECT COUNT(*) FROM calls
                    WHERE json_array_length(targets) = 2
                    AND LOWER(requested_modalities) LIKE '%audio%'
                    AND LOWER(targets) LIKE '%emma%wilson%'
                    AND LOWER(targets) LIKE '%frank%chen%'
                    AND soft_delete = 0;""",
                    gym_name="gym-teams-mcp",
                ),
            ),
        ),
        "task_20260106_102032_602_8e9e30d7_286c0816": _TaskPatch(
            domain="teams",
            prompt_fragment="Temp#456",
            source_count=4,
            source_unique_count=4,
            rationale=(
                "Bind the welcome and audio call to Liam, and verify both team and channel "
                "membership. The fixed backend does not persist the temporary password itself."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "Channel Membership Added",
                        """SELECT COUNT(*) AS count FROM channel_members
                        WHERE channel_id = 'channel_shared_001'
                        AND (email = 'liam.turner@techcorp.com');""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channel_members cm
                    JOIN teams_users u ON u.user_id = cm.user_id
                    WHERE cm.channel_id = 'channel_shared_001'
                    AND u.user_principal_name = 'liam.turner@techcorp.com';""",
                ),
                _Replacement(
                    source=_spec(
                        "Welcome Message Posted Today",
                        """SELECT COUNT(*) AS count FROM channel_messages
                        WHERE team_id = 'team_techcorp_001'
                        AND DATE(created_datetime) = CURRENT_DATE;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channel_messages
                    WHERE team_id = 'team_techcorp_001'
                    AND channel_id = 'channel_shared_001'
                    AND LOWER(body_json) LIKE '%welcome liam turner%'
                    AND LOWER(body_json) LIKE '%business analyst%'
                    AND LOWER(body_json) LIKE '%techcorp solutions%';""",
                ),
                _Replacement(
                    source=_spec(
                        "Audio Call Created (source & target[0])",
                        """SELECT COUNT(*) FROM calls
                        WHERE json_extract(source, '$.identity.applicationInstance.id') = 'james.wilson'
                        AND json_extract(targets, '$[0].identity.user.displayName') = 'Liam Turner';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM calls
                    WHERE LOWER(requested_modalities) LIKE '%audio%'
                    AND LOWER(targets) LIKE '%liam%turner%'
                    AND soft_delete = 0;""",
                ),
            ),
            add=(
                _new(
                    "Liam Turner Added to TechCorp Solutions Team",
                    "The newly created Liam Turner user is a member of the team.",
                    """SELECT CASE WHEN EXISTS (
                      SELECT 1 FROM teams t, json_each(t.members) member
                      WHERE t.team_id = 'team_techcorp_001'
                      AND json_extract(member.value, '$.userId') IN (
                        SELECT user_id FROM teams_users
                        WHERE user_principal_name = 'liam.turner@techcorp.com'
                      )
                    ) THEN 1 ELSE 0 END;""",
                    gym_name="gym-teams-mcp",
                ),
            ),
        ),
        "task_20260106_123222_425_8e9e30d7_7dea421b": _TaskPatch(
            domain="teams",
            prompt_fragment="Product Launch 2026",
            source_count=5,
            source_unique_count=5,
            rationale="Bind the private channel, Planner app, tab, and announcement to one channel.",
            replace=(
                _Replacement(
                    source=_spec(
                        "Product Launch 2026 Channel Created",
                        """SELECT COUNT(*) AS count FROM channels c JOIN teams t
                        ON c.team_id = t.team_id WHERE c.display_name = 'Product Launch 2026'
                        AND t.display_name = 'TechCorp Solutions Team'
                        AND c.is_archived = 0 AND c.soft_delete = 0;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channels c JOIN teams t
                    ON c.team_id = t.team_id
                    WHERE c.display_name = 'Product Launch 2026'
                    AND t.display_name = 'TechCorp Solutions Team'
                    AND c.membership_type = 'private'
                    AND c.is_archived = 0 AND c.soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "Planner Tab Added",
                        """SELECT COUNT(*) FROM teams_channel_tabs
                        WHERE display_name = 'Launch Tasks'
                        AND team_id = 'team_techcorp_001';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM teams_channel_tabs tab
                    JOIN channels c ON c.channel_id = tab.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Product Launch 2026'
                    AND tab.display_name = 'Launch Tasks'
                    AND tab.app_id IN (SELECT app_id FROM teams_apps
                      WHERE LOWER(display_name) = 'planner');""",
                ),
                _Replacement(
                    source=_spec(
                        "Announcement Message Sent",
                        """SELECT COUNT(*) FROM channel_messages
                        WHERE team_id = 'team_techcorp_001'
                        AND LOWER(body_json) LIKE '%launch%'
                        AND LOWER(body_json) LIKE '%planning%'
                        AND LOWER(body_json) LIKE '%has started%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channel_messages m
                    JOIN channels c ON c.channel_id = m.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Product Launch 2026'
                    AND LOWER(m.body_json) LIKE '%launch%'
                    AND LOWER(m.body_json) LIKE '%planning%'
                    AND LOWER(m.body_json) LIKE '%started%';""",
                ),
            ),
        ),
        "task_20260108_160714_025_8e9e30d7_80cf63dd": _TaskPatch(
            domain="teams",
            prompt_fragment="Executive Leadership tag",
            source_count=5,
            source_unique_count=5,
            rationale="Check the requested description/message meaning and both tag members.",
            replace=(
                _Replacement(
                    source=_spec(
                        "exec_updates_channel_description_updated",
                        """SELECT COUNT(*) FROM channels c JOIN teams t ON t.team_id = c.team_id
                        WHERE c.channel_id = 'b5c4d7e9f2a6b8c5d9e7f1a4b6c8d2e5'
                        AND t.team_id = 'team_globalex_002'
                        AND c.display_name = 'Executive Updates'
                        AND LOWER(c.description) LIKE '%official%'
                        AND LOWER(c.description) LIKE '%ceo%'
                        AND LOWER(c.description) LIKE '%board%'
                        AND LOWER(c.description) LIKE '%formal%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channels c
                    WHERE c.channel_id = 'b5c4d7e9f2a6b8c5d9e7f1a4b6c8d2e5'
                    AND c.team_id = 'team_globalex_002'
                    AND c.display_name = 'Executive Updates'
                    AND LOWER(c.description) LIKE '%ceo%'
                    AND LOWER(c.description) LIKE '%board%'
                    AND LOWER(c.description) LIKE '%not%general%discussion%';""",
                ),
                _Replacement(
                    source=_spec(
                        "exec_updates_announcement_posted_with_emma_wilson",
                        """SELECT COUNT(*) FROM channel_messages m JOIN channels c
                        ON c.channel_id = m.channel_id JOIN teams t ON t.team_id = c.team_id
                        WHERE t.team_id = 'team_globalex_002'
                        AND c.channel_id = 'b5c4d7e9f2a6b8c5d9e7f1a4b6c8d2e5'
                        AND c.display_name = 'Executive Updates'
                        AND LOWER(m.body_json) LIKE '%emma wilson%'
                        AND LOWER(m.body_json) LIKE '%sponsor%'
                        AND LOWER(m.body_json) LIKE '%board%'
                        AND LOWER(m.body_json) LIKE '%structure%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channel_messages m
                    WHERE m.channel_id = 'b5c4d7e9f2a6b8c5d9e7f1a4b6c8d2e5'
                    AND m.team_id = 'team_globalex_002'
                    AND LOWER(m.body_json) LIKE '%emma wilson%'
                    AND LOWER(m.body_json) LIKE '%sponsor%'
                    AND LOWER(m.body_json) LIKE '%final%board%material%'
                    AND LOWER(m.body_json) LIKE '%stor%channel%';""",
                ),
                _Replacement(
                    source=_spec(
                        "executive_updates_channel_email_provisioned",
                        """SELECT COUNT(*) FROM channels
                        WHERE channel_id = 'b5c4d7e9f2a6b8c5d9e7f1a4b6c8d2e5'
                        AND team_id = 'team_globalex_002'
                        AND email IS NOT NULL AND email LIKE '%teams.ms';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channels
                    WHERE channel_id = 'b5c4d7e9f2a6b8c5d9e7f1a4b6c8d2e5'
                    AND team_id = 'team_globalex_002'
                    AND email IS NOT NULL AND email LIKE '%@%.%';""",
                ),
                _Replacement(
                    source=_spec(
                        "executive_leadership_teamwork_tag_created",
                        """SELECT COUNT(*) FROM teamwork_tags tt JOIN teams t
                        ON t.team_id = tt.team_id WHERE t.team_id = 'team_globalex_002'
                        AND LOWER(tt.display_name) LIKE '%executive%'
                        AND LOWER(tt.display_name) LIKE '%leadership%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT member.userId) = 2
                    AND COUNT(DISTINCT CASE WHEN member.userId IN
                      ('emma.wilson', 'frank.chen') THEN member.userId END) = 2
                    THEN 1 ELSE 0 END FROM teamwork_tag_members member
                    JOIN teamwork_tags tag ON tag.id = member.tagId
                    WHERE tag.team_id = 'team_globalex_002'
                    AND tag.display_name = 'Executive Leadership';""",
                ),
            ),
        ),
        "task_20260108_160846_466_8e9e30d7_7e50c8e2": _TaskPatch(
            domain="teams",
            prompt_fragment="Incident Response Triage",
            source_count=8,
            source_unique_count=8,
            rationale=(
                "Verify both chat tabs and tag members; replace hidden message/call literals "
                "with the participants and content actually requested."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "techcorp_incident_responders_tag_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM teamwork_tags tt WHERE tt.team_id = 'team_techcorp_001'
                        AND tt.display_name = 'Incident Responders';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT member.userId) = 3
                    AND COUNT(DISTINCT CASE WHEN member.userId IN
                      ('john.davis', 'alice.johnson', 'bob.smith')
                      THEN member.userId END) = 3
                    THEN 1 ELSE 0 END FROM teamwork_tag_members member
                    JOIN teamwork_tags tag ON tag.id = member.tagId
                    WHERE tag.team_id = 'team_techcorp_001'
                    AND tag.display_name = 'Incident Responders';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_incident_triage_chat_created_with_members",
                        """SELECT CASE WHEN COUNT(DISTINCT cm.user_id) = 4 THEN 1 ELSE 0 END
                        FROM chats ch JOIN chat_members cm ON ch.chat_id = cm.chat_id
                        WHERE ch.topic = 'Incident Response Triage'
                        AND ch.chat_type = 'group'
                        AND cm.user_id IN ('james.wilson','john.davis','alice.johnson','bob.smith');""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT cm.user_id) = 4
                    AND COUNT(DISTINCT CASE WHEN cm.user_id IN
                      ('james.wilson','john.davis','alice.johnson','bob.smith')
                      THEN cm.user_id END) = 4
                    THEN 1 ELSE 0 END FROM chats ch
                    JOIN chat_members cm ON ch.chat_id = cm.chat_id
                    WHERE ch.topic = 'Incident Response Triage'
                    AND ch.chat_type = 'group';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_incident_triage_kickoff_message_posted",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                        WHERE ch.topic = 'Incident Response Triage'
                        AND LOWER(m.body_json) LIKE '%welcome%'
                        AND LOWER(m.body_json) LIKE '%p1 and p2%'
                        AND LOWER(m.body_json) LIKE '%planner tab%'
                        AND LOWER(m.body_json) LIKE '%onenote tab%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                    WHERE ch.topic = 'Incident Response Triage'
                    AND LOWER(m.body_json) LIKE '%hub%'
                    AND LOWER(m.body_json) LIKE '%p1%'
                    AND LOWER(m.body_json) LIKE '%p2%'
                    AND LOWER(m.body_json) LIKE '%production%issue%';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_incident_triage_kickoff_message_pinned",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                        JOIN pinned_chat_messages pcm ON pcm.chat_id = ch.chat_id
                        AND pcm.message_id = m.message_id
                        WHERE ch.topic = 'Incident Response Triage'
                        AND LOWER(m.body_json) LIKE '%welcome to incident response triage%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                    JOIN pinned_chat_messages pcm ON pcm.chat_id = ch.chat_id
                    AND pcm.message_id = m.message_id
                    WHERE ch.topic = 'Incident Response Triage'
                    AND LOWER(m.body_json) LIKE '%hub%'
                    AND LOWER(m.body_json) LIKE '%p1%'
                    AND LOWER(m.body_json) LIKE '%p2%';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_incident_triage_team_member_list_message_posted",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                        WHERE ch.topic = 'Incident Response Triage'
                        AND LOWER(m.body_json) LIKE '%team%'
                        AND LOWER(m.body_json) LIKE '%james.wilson@techcorp.com%'
                        AND LOWER(m.body_json) LIKE '%alice.manager@techcorp.com%'
                        AND LOWER(m.body_json) LIKE '%bob.smith@techcorp.com%'
                        AND LOWER(m.body_json) LIKE '%carol.white@techcorp.com%'
                        AND LOWER(m.body_json) LIKE '%dave.brown@techcorp.com%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                    WHERE ch.topic = 'Incident Response Triage'
                    AND LOWER(m.body_json) LIKE '%james.wilson@techcorp.com%'
                    AND LOWER(m.body_json) LIKE '%alice.manager@techcorp.com%'
                    AND LOWER(m.body_json) LIKE '%bob.smith@techcorp.com%'
                    AND LOWER(m.body_json) LIKE '%carol.white@techcorp.com%'
                    AND LOWER(m.body_json) LIKE '%dave.brown@techcorp.com%'
                    AND LOWER(m.body_json) LIKE '%nathan.richards@techcorp.com%'
                    AND LOWER(m.body_json) LIKE '%sophia.bennett@techcorp.com%';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_incident_triage_escalation_message_posted",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                        WHERE ch.topic = 'Incident Response Triage'
                        AND LOWER(m.body_json) LIKE '%escalation procedure%'
                        AND LOWER(m.body_json) LIKE '%p1%'
                        AND LOWER(m.body_json) LIKE '%p2%'
                        AND LOWER(m.body_json) LIKE '%15 minutes%'
                        AND LOWER(m.body_json) LIKE '%30 minutes%'
                        AND LOWER(m.body_json) LIKE '%mike thompson%'
                        AND LOWER(m.body_json) LIKE '%it.support@techcorp.com%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM chats ch JOIN chat_messages m ON ch.chat_id = m.chat_id
                    WHERE ch.topic = 'Incident Response Triage'
                    AND LOWER(m.body_json) LIKE '%p1%'
                    AND LOWER(m.body_json) LIKE '%p2%'
                    AND LOWER(m.body_json) LIKE '%15 minute%'
                    AND LOWER(m.body_json) LIKE '%30 minute%'
                    AND LOWER(m.body_json) LIKE '%mike thompson%';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_incident_group_call_created_with_correct_participants",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                        WHERE subject = 'Incident Response System Walkthrough'
                        AND callback_uri = 'https://meetings.techcorp.com/api/calls'
                        AND source LIKE '%james.wilson%'
                        AND targets LIKE '%john.davis%'
                        AND targets LIKE '%alice.johnson%'
                        AND targets LIKE '%bob.smith%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                    WHERE json_array_length(targets) = 3
                    AND LOWER(targets) LIKE '%john%davis%'
                    AND LOWER(targets) LIKE '%alice%johnson%'
                    AND LOWER(targets) LIKE '%bob%smith%'
                    AND soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_incident_one_on_one_call_with_bob_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                        WHERE subject = 'Quick Sync - Incident Response'
                        AND callback_uri = 'https://meetings.techcorp.com/api/calls'
                        AND source LIKE '%james.wilson%'
                        AND targets LIKE '%bob.smith%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                    WHERE json_array_length(targets) = 1
                    AND LOWER(targets) LIKE '%bob%smith%'
                    AND soft_delete = 0;""",
                ),
            ),
            add=(
                _new(
                    "Incident Response Planner Chat Tab Added",
                    "The triage group chat contains a Planner tab.",
                    """SELECT COUNT(*) FROM teams_chat_tabs tab
                    JOIN chats ch ON ch.chat_id = tab.chat_id
                    WHERE ch.topic = 'Incident Response Triage'
                    AND tab.app_id IN (SELECT app_id FROM teams_apps
                      WHERE LOWER(display_name) = 'planner');""",
                    gym_name="gym-teams-mcp",
                ),
                _new(
                    "Incident Post-Mortems OneNote Chat Tab Added",
                    "The triage group chat contains the named OneNote tab.",
                    """SELECT COUNT(*) FROM teams_chat_tabs tab
                    JOIN chats ch ON ch.chat_id = tab.chat_id
                    WHERE ch.topic = 'Incident Response Triage'
                    AND tab.display_name = 'Incident Post-Mortems'
                    AND tab.app_id IN (SELECT app_id FROM teams_apps
                      WHERE LOWER(display_name) = 'onenote');""",
                    gym_name="gym-teams-mcp",
                ),
            ),
        ),
        "task_20260108_172650_003_8e9e30d7_a1ff8588": _TaskPatch(
            domain="teams",
            prompt_fragment="Customer Escalation Review",
            source_count=11,
            source_unique_count=11,
            rationale=(
                "Verify the fixed townhall schedule and Planner tab, remove favorite/creator/"
                "callback/subject hidden conditions, and check the requested call participants."
            ),
            remove=(
                _spec(
                    "townhall_created_by_james_wilson",
                    """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM virtual_event_townhalls
                    WHERE display_name = 'Customer Escalation Review'
                    AND created_by_user_id = 'james.wilson' AND org_id = 'techcorp-001';""",
                    gym_name="gym-teams-mcp",
                ),
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "escalation_reviewers_tag_with_three_members",
                        """SELECT CASE WHEN COUNT(*) = 3 THEN 1 ELSE 0 END
                        FROM teamwork_tag_members ttm
                        JOIN teamwork_tags tt ON tt.id = ttm.tagId
                        JOIN teams t ON t.team_id = tt.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND tt.display_name = 'Escalation Reviewers'
                        AND ttm.userId IN
                          ('alice.johnson', 'dave.brown', 'carol.white');""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT ttm.userId) = 3
                    AND COUNT(DISTINCT CASE WHEN ttm.userId IN
                      ('alice.johnson', 'dave.brown', 'carol.white')
                      THEN ttm.userId END) = 3
                    THEN 1 ELSE 0 END FROM teamwork_tag_members ttm
                    JOIN teamwork_tags tt ON tt.id = ttm.tagId
                    JOIN teams t ON t.team_id = tt.team_id
                    WHERE t.display_name = 'TechCorp Solutions Team'
                    AND tt.display_name = 'Escalation Reviewers';""",
                ),
                _Replacement(
                    source=_spec(
                        "escalation_coordination_group_chat_with_three_members",
                        """SELECT CASE WHEN COUNT(*) = 3 THEN 1 ELSE 0 END
                        FROM chat_members cm JOIN chats c ON c.chat_id = cm.chat_id
                        WHERE c.topic = 'Escalation Review Coordination'
                        AND LOWER(c.chat_type) = 'group'
                        AND cm.user_id IN
                          ('james.wilson', 'alice.johnson', 'dave.brown');""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT cm.user_id) = 3
                    AND COUNT(DISTINCT CASE WHEN cm.user_id IN
                      ('james.wilson', 'alice.johnson', 'dave.brown')
                      THEN cm.user_id END) = 3
                    THEN 1 ELSE 0 END FROM chat_members cm
                    JOIN chats c ON c.chat_id = cm.chat_id
                    WHERE c.topic = 'Escalation Review Coordination'
                    AND LOWER(c.chat_type) = 'group';""",
                ),
                _Replacement(
                    source=_spec(
                        "customer_escalation_townhall_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM virtual_event_townhalls v WHERE v.org_id = 'techcorp-001'
                        AND v.display_name = 'Customer Escalation Review'
                        AND LOWER(v.audience) = 'organization'
                        AND v.is_attendee_email_notification_enabled = 1;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM virtual_event_townhalls v WHERE v.org_id = 'techcorp-001'
                    AND v.display_name = 'Customer Escalation Review'
                    AND LOWER(v.audience) = 'organization'
                    AND substr(replace(v.start_date_time, 'T', ' '), 1, 16) = '2026-01-10 15:00'
                    AND substr(replace(v.end_date_time, 'T', ' '), 1, 16) = '2026-01-10 16:00'
                    AND v.start_time_zone = 'Pacific Standard Time'
                    AND v.end_time_zone = 'Pacific Standard Time';""",
                ),
                _Replacement(
                    source=_spec(
                        "customer_escalations_channel_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM channels c
                        JOIN teams t ON t.team_id = c.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND c.display_name = 'Customer Escalations'
                        AND LOWER(c.membership_type) = 'standard'
                        AND c.is_favorite_by_default = 1;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM channels c JOIN teams t ON t.team_id = c.team_id
                    WHERE t.display_name = 'TechCorp Solutions Team'
                    AND c.display_name = 'Customer Escalations'
                    AND LOWER(c.membership_type) = 'standard'
                    AND c.is_archived = 0 AND c.soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "customer_escalations_channel_welcome_message_posted",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                        JOIN teams t ON t.team_id = c.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND c.display_name = 'Customer Escalations'
                        AND LOWER(m.body_json) LIKE '%welcome%'
                        AND LOWER(m.body_json) LIKE '%escalation tracker%'
                        AND LOWER(m.body_json) LIKE '%escalation reviewers%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Customer Escalations'
                    AND LOWER(m.body_json) LIKE '%customer escalation review%'
                    AND LOWER(m.body_json) LIKE '%january 10%'
                    AND LOWER(m.body_json) LIKE '%3:00%'
                    AND LOWER(m.body_json) LIKE '%4:00%'
                    AND LOWER(m.body_json) LIKE '%pacific%'
                    AND LOWER(m.body_json) NOT LIKE '%this saturday%';""",
                ),
                _Replacement(
                    source=_spec(
                        "individual_test_call_with_alice_johnson_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                        WHERE subject = 'Test Call from James Wilson to Alice'
                        AND callback_uri = 'https://meetings.techcorp.com/api/calls'
                        AND JSON_EXTRACT(source, '$.identity.user.id') = 'james.wilson'
                        AND JSON_EXTRACT(targets, '$[0].identity.user.id') = 'alice.johnson';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                    WHERE json_array_length(targets) = 1
                    AND LOWER(targets) LIKE '%alice%johnson%'
                    AND LOWER(requested_modalities) LIKE '%audio%'
                    AND soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "individual_test_call_with_dave_brown_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                        WHERE subject = 'Test Call from James Wilson to Dave'
                        AND callback_uri = 'https://meetings.techcorp.com/api/calls'
                        AND JSON_EXTRACT(source, '$.identity.user.id') = 'james.wilson'
                        AND JSON_EXTRACT(targets, '$[0].identity.user.id') = 'dave.brown';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                    WHERE json_array_length(targets) = 1
                    AND LOWER(targets) LIKE '%dave%brown%'
                    AND LOWER(requested_modalities) LIKE '%audio%'
                    AND soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "group_test_call_with_alice_and_dave_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                        WHERE subject = 'Group Call from James Wilson'
                        AND callback_uri = 'https://meetings.techcorp.com/api/calls'
                        AND JSON_EXTRACT(source, '$.identity.user.id') = 'james.wilson'
                        AND JSON_EXTRACT(targets, '$[0].identity.user.id') = 'alice.johnson'
                        AND JSON_EXTRACT(targets, '$[1].identity.user.id') = 'dave.brown';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM calls
                    WHERE json_array_length(targets) = 2
                    AND LOWER(targets) LIKE '%alice%johnson%'
                    AND LOWER(targets) LIKE '%dave%brown%'
                    AND LOWER(requested_modalities) LIKE '%audio%'
                    AND soft_delete = 0;""",
                ),
            ),
            add=(
                _new(
                    "Escalation Tracker Planner Tab Added",
                    "The Customer Escalations channel contains the requested Planner tab.",
                    """SELECT COUNT(*) FROM teams_channel_tabs tab
                    JOIN channels c ON c.channel_id = tab.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Customer Escalations'
                    AND tab.display_name = 'Escalation Tracker'
                    AND tab.app_id IN (SELECT app_id FROM teams_apps
                      WHERE LOWER(display_name) = 'planner');""",
                    gym_name="gym-teams-mcp",
                ),
            ),
        ),
        "task_20260108_194921_186_8e9e30d7_a087d6f8": _TaskPatch(
            domain="teams",
            prompt_fragment="TechCorp Weekly Release Readiness",
            source_count=9,
            source_unique_count=9,
            rationale=(
                "Check the fixed townhall schedule/co-organizers, real Planner app, requested "
                "message meaning, tag members, and audio-call topic/participants without hidden fields."
            ),
            replace=(
                _Replacement(
                    source=_spec(
                        "techcorp_release_townhall_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM virtual_event_townhalls v WHERE v.org_id = 'techcorp-001'
                        AND v.display_name = 'TechCorp Weekly Release Readiness'
                        AND v.audience = 'organization'
                        AND v.is_attendee_email_notification_enabled = 1;""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM virtual_event_townhalls v WHERE v.org_id = 'techcorp-001'
                    AND v.display_name = 'TechCorp Weekly Release Readiness'
                    AND v.audience = 'organization'
                    AND substr(replace(v.start_date_time, 'T', ' '), 1, 16) = '2026-01-09 10:00'
                    AND substr(replace(v.end_date_time, 'T', ' '), 1, 16) = '2026-01-09 11:00'
                    AND v.start_time_zone = 'Pacific Standard Time'
                    AND v.end_time_zone = 'Pacific Standard Time';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_release_readiness_channel_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END FROM channels c
                        JOIN teams t ON t.team_id = c.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND c.display_name = 'Release Readiness'
                        AND c.membership_type = 'standard'
                        AND c.is_favorite_by_default = 1
                        AND LOWER(c.description) LIKE '%release coordination%'
                        AND LOWER(c.description) LIKE '%techcorp weekly release readiness townhall%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM channels c JOIN teams t ON t.team_id = c.team_id
                    WHERE t.display_name = 'TechCorp Solutions Team'
                    AND c.display_name = 'Release Readiness'
                    AND c.membership_type = 'standard'
                    AND c.is_archived = 0 AND c.soft_delete = 0;""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_release_checklist_tab_added",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM teams_channel_tabs tct JOIN teams t ON t.team_id = tct.team_id
                        JOIN channels c ON c.team_id = tct.team_id
                        AND c.channel_id = tct.channel_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND c.display_name = 'Release Readiness'
                        AND tct.display_name = 'Release Checklist';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM teams_channel_tabs tab JOIN channels c ON c.channel_id = tab.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Release Readiness'
                    AND tab.display_name = 'Release Checklist'
                    AND tab.app_id IN (SELECT app_id FROM teams_apps
                      WHERE LOWER(display_name) = 'planner');""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_release_townhall_announced_in_general_channel",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                        JOIN teams t ON t.team_id = c.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND c.display_name = 'General'
                        AND LOWER(m.body_json) LIKE '%weekly release%'
                        AND LOWER(m.body_json) LIKE '%single place%'
                        AND LOWER(m.body_json) LIKE '%release updates%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                    WHERE c.team_id = 'team_techcorp_001' AND c.display_name = 'General'
                    AND LOWER(m.body_json) LIKE '%techcorp weekly release readiness%'
                    AND LOWER(m.body_json) LIKE '%10%11%pacific%'
                    AND LOWER(m.body_json) LIKE '%live%release%update%'
                    AND LOWER(m.body_json) LIKE '%meeting chat%';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_release_townhall_announced_in_development_channel",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                        JOIN teams t ON t.team_id = c.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND c.display_name = 'Development'
                        AND LOWER(m.body_json) LIKE '%discussions%'
                        AND LOWER(m.body_json) LIKE '%release readiness%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                    WHERE c.team_id = 'team_techcorp_001' AND c.display_name = 'Development'
                    AND LOWER(m.body_json) LIKE '%techcorp weekly release readiness%'
                    AND LOWER(m.body_json) LIKE '%10%11%pacific%'
                    AND LOWER(m.body_json) LIKE '%live%release%update%'
                    AND LOWER(m.body_json) LIKE '%meeting chat%';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_release_readiness_channel_welcome_and_townhall_mention",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                        JOIN teams t ON t.team_id = c.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND c.display_name = 'Release Readiness'
                        AND LOWER(m.body_json) LIKE '%welcome%'
                        AND LOWER(m.body_json) LIKE '%release readiness%'
                        AND LOWER(m.body_json) LIKE '%townhall%';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                    FROM channel_messages m JOIN channels c ON c.channel_id = m.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Release Readiness'
                    AND LOWER(m.body_json) LIKE '%techcorp weekly release readiness%'
                    AND LOWER(m.body_json) LIKE '%10%11%pacific%'
                    AND LOWER(m.body_json) LIKE '%live%release%update%'
                    AND LOWER(m.body_json) LIKE '%meeting chat%';""",
                ),
                _Replacement(
                    source=_spec(
                        "techcorp_release_reviewers_tag_created",
                        """SELECT CASE WHEN COUNT(*) > 0 THEN 1 ELSE 0 END
                        FROM teamwork_tags tt JOIN teams t ON t.team_id = tt.team_id
                        WHERE t.display_name = 'TechCorp Solutions Team'
                        AND tt.display_name = 'Release Reviewers'
                        AND LOWER(tt.description) LIKE '%release readiness%'
                        AND tt.tag_type = 'standard';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT member.userId) = 3
                    AND COUNT(DISTINCT CASE WHEN member.userId IN
                      ('alice.johnson', 'dave.brown', 'carol.white')
                      THEN member.userId END) = 3
                    THEN 1 ELSE 0 END FROM teamwork_tag_members member
                    JOIN teamwork_tags tag ON tag.id = member.tagId
                    WHERE tag.team_id = 'team_techcorp_001'
                    AND tag.display_name = 'Release Reviewers';""",
                ),
                _Replacement(
                    source=_spec(
                        "audio_call_created",
                        """SELECT COUNT(*) FROM calls
                        WHERE json_extract(source, '$.identity.applicationInstance.id') = 'james.wilson'
                        AND (json_extract(targets, '$[0].identity.user.id') = 'carol.white'
                        OR json_extract(targets, '$[0].identity.user.id') = 'alice.johnson');""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM calls
                    WHERE json_array_length(targets) = 2
                    AND LOWER(requested_modalities) LIKE '%audio%'
                    AND LOWER(subject) LIKE '%release%'
                    AND (LOWER(subject) LIKE '%agenda%' OR LOWER(subject) LIKE '%readiness%')
                    AND LOWER(targets) LIKE '%alice%johnson%'
                    AND LOWER(targets) LIKE '%carol%white%'
                    AND soft_delete = 0;""",
                ),
            ),
            add=(
                _new(
                    "TechCorp Release Townhall Co-organizers",
                    "Alice Johnson and Dave Brown are both townhall co-organizers.",
                    """SELECT CASE WHEN COUNT(DISTINCT co.user_id) = 2
                    THEN 1 ELSE 0 END FROM virtual_event_townhall_coorganizers co
                    JOIN virtual_event_townhalls v ON v.id = co.townhall_id
                    WHERE v.display_name = 'TechCorp Weekly Release Readiness'
                    AND co.user_id IN ('alice.johnson', 'dave.brown');""",
                    gym_name="gym-teams-mcp",
                ),
            ),
        ),
        "task_20260109_042243_802_df0ecdeb_e6cc6185": _TaskPatch(
            domain="teams",
            prompt_fragment="Operational Efficiency Hub",
            source_count=8,
            source_unique_count=8,
            rationale="Verify actual removals, all five remaining members, and the requested audio call.",
            replace=(
                _Replacement(
                    source=_spec(
                        "Bob and Dave Removed From Team",
                        """SELECT COUNT(*) AS team_members_remaining FROM teams t
                        WHERE t.team_id = 'team_techcorp_001'
                        AND (t.members LIKE '%userId: bob.smith%'
                        OR t.members LIKE '%userId: dave.brown%');""",
                        expected_value=0,
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM teams t, json_each(t.members) member
                    WHERE t.team_id = 'team_techcorp_001'
                    AND json_extract(member.value, '$.userId') IN ('bob.smith', 'dave.brown');""",
                ),
                _Replacement(
                    source=_spec(
                        "All Remaining Members Added To Operational Efficiency Hub",
                        """SELECT COUNT(*) FROM teams t, json_each(t.members) AS m
                        WHERE t.org_id = 'techcorp-001' AND t.team_id = 'team_techcorp_001'
                        AND json_extract(m.value, '$.userId') IN ('bob.smith', 'dave.brown');""",
                        expected_value=0,
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT CASE WHEN COUNT(DISTINCT cm.user_id) = 5
                    AND COUNT(DISTINCT CASE WHEN cm.user_id IN
                      ('james.wilson', 'alice.johnson', 'carol.white',
                       'nathan.richards', 'sophia.bennett')
                      THEN cm.user_id END) = 5
                    THEN 5 ELSE 0 END FROM channel_members cm
                    JOIN channels c ON c.channel_id = cm.channel_id
                    WHERE c.team_id = 'team_techcorp_001'
                    AND c.display_name = 'Operational Efficiency Hub';""",
                    expected_value=5,
                ),
                _Replacement(
                    source=_spec(
                        "Audio Call Created Between Alice and Carol",
                        """SELECT COUNT(*) FROM calls
                        WHERE JSON_EXTRACT(source, '$.identity.user.displayName') = 'James Wilson'
                        AND JSON_EXTRACT(targets, '$[0].identity.user.displayName') = 'Alice Johnson';""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM calls
                    WHERE json_array_length(targets) = 1
                    AND LOWER(requested_modalities) LIKE '%audio%'
                    AND LOWER(targets) LIKE '%alice%johnson%'
                    AND soft_delete = 0;""",
                    name="Audio Call Created Between Alice and James",
                ),
            ),
        ),
        "task_20260109_060901_602_8e9e30d7_06581f9f": _TaskPatch(
            domain="teams",
            prompt_fragment="call durations, number of participants",
            source_count=3,
            source_unique_count=3,
            rationale="Require the structured report content instead of accepting any channel message.",
            replace=(
                _Replacement(
                    source=_spec(
                        "analysis_report_posted",
                        """SELECT COUNT(*) as count FROM channel_messages
                        WHERE channel_id IN (SELECT channel_id FROM channels
                        WHERE display_name = 'Call Quality Review'
                        AND team_id = 'team_techcorp_001');""",
                        expected_value=0,
                        comparison_type="greater_than",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) FROM channel_messages
                    WHERE channel_id IN (SELECT channel_id FROM channels
                      WHERE display_name = 'Call Quality Review'
                      AND team_id = 'team_techcorp_001')
                    AND LOWER(body_json) LIKE '%duration%'
                    AND LOWER(body_json) LIKE '%participant%'
                    AND (LOWER(body_json) LIKE '%call quality%'
                      OR LOWER(body_json) LIKE '%audio%'
                      OR LOWER(body_json) LIKE '%dropped%')
                    AND LOWER(body_json) LIKE '%investigat%';""",
                ),
            ),
        ),
        "task_20260109_075544_800_464ee3e0_7852c2fb": _TaskPatch(
            domain="teams",
            prompt_fragment="https://bot.techcorp.com/api/calls",
            source_count=2,
            source_unique_count=2,
            rationale="Add the exact callback URL explicitly required by the prompt.",
            replace=(
                _Replacement(
                    source=_spec(
                        "Check Call from James Wilson to Alice Johnson",
                        """SELECT COUNT(*) AS verification_result FROM calls
                        WHERE org_id = 'techcorp-001' AND source LIKE '%james.wilson%'
                        AND targets LIKE '%alice.johnson%'
                        AND targets LIKE '%sarah.martinez%'
                        AND requested_modalities LIKE '%audio%'
                        AND requested_modalities LIKE '%video%'""",
                        gym_name="gym-teams-mcp",
                    ),
                    query="""SELECT COUNT(*) AS verification_result FROM calls
                    WHERE org_id = 'techcorp-001'
                    AND source LIKE '%james.wilson%'
                    AND json_array_length(targets) = 2
                    AND targets LIKE '%alice.johnson%'
                    AND targets LIKE '%sarah.martinez%'
                    AND requested_modalities LIKE '%audio%'
                    AND requested_modalities LIKE '%video%'
                    AND callback_uri = 'https://bot.techcorp.com/api/calls';""",
                ),
            ),
        ),
    }
)


def _canonical(verifier: Mapping[str, Any]) -> str:
    return json.dumps(verifier, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _matches(verifier: Mapping[str, Any], spec: _VerifierSpec) -> bool:
    validation = verifier.get("validation_config")
    if not isinstance(validation, Mapping):
        return False
    return (
        verifier.get("name") == spec.name
        and (spec.gym_name is None or verifier.get("gym_name") == spec.gym_name)
        and _sql(str(validation.get("query", ""))) == spec.query
        and validation.get("expected_value") == spec.expected_value
        and validation.get("comparison_type", "equals") == spec.comparison_type
    )


def _find_one(
    *, task_id: str, verifiers: list[Mapping[str, Any]], spec: _VerifierSpec
) -> int:
    matches = [index for index, verifier in enumerate(verifiers) if _matches(verifier, spec)]
    if len(matches) != 1:
        raise VerifierContractSourceError(
            f"{task_id}: expected one source verifier {spec.name!r}, found {len(matches)}"
        )
    return matches[0]


def _dedupe_exact(verifiers: list[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    seen: set[str] = set()
    result: list[Mapping[str, Any]] = []
    for verifier in verifiers:
        key = _canonical(verifier)
        if key in seen:
            continue
        seen.add(key)
        result.append(verifier)
    return result


def apply_verifier_contract(
    *,
    task_id: str,
    domain: Any,
    user_prompt: Any,
    verifiers: list[Mapping[str, Any]],
) -> tuple[
    list[Mapping[str, Any]],
    tuple[VerifierContractCorrection, ...],
    VerifierSetNormalization,
]:
    """Apply exact task repairs, fold exact duplicates, and uniquify names."""

    source_count = len(verifiers)
    working = _dedupe_exact(verifiers)
    source_unique_count = len(working)
    patch = _TASK_PATCHES.get(task_id)
    corrections: tuple[VerifierContractCorrection, ...] = ()
    if patch is not None:
        allowed_source_counts = (
            (patch.source_count,)
            if isinstance(patch.source_count, int)
            else patch.source_count
        )
        if (
            domain != patch.domain
            or not isinstance(user_prompt, str)
            or patch.prompt_fragment not in user_prompt
            or source_count not in allowed_source_counts
            or source_unique_count != patch.source_unique_count
        ):
            raise VerifierContractSourceError(
                f"{task_id}: task no longer matches its pinned verifier-contract source"
            )

        remove_indices = {
            _find_one(task_id=task_id, verifiers=working, spec=spec)
            for spec in patch.remove
        }
        replacement_indices: dict[int, _Replacement] = {}
        for replacement in patch.replace:
            index = _find_one(task_id=task_id, verifiers=working, spec=replacement.source)
            if index in remove_indices or index in replacement_indices:
                raise VerifierContractSourceError(
                    f"{task_id}: verifier repair operations overlap at index {index}"
                )
            replacement_indices[index] = replacement

        repaired: list[Mapping[str, Any]] = []
        for index, verifier in enumerate(working):
            if index in remove_indices:
                continue
            replacement = replacement_indices.get(index)
            if replacement is None:
                repaired.append(verifier)
                continue
            updated = dict(verifier)
            validation = dict(updated["validation_config"])
            if replacement.query is not None:
                validation["query"] = replacement.query
            if replacement.expected_value is not None:
                validation["expected_value"] = replacement.expected_value
            if replacement.comparison_type is not None:
                validation["comparison_type"] = replacement.comparison_type
            updated["validation_config"] = validation
            if replacement.name is not None:
                updated["name"] = replacement.name
            repaired.append(updated)
        working = repaired + [dict(verifier) for verifier in patch.add]
        working = _dedupe_exact(working)
        corrections = (
            VerifierContractCorrection(
                source_verifier_count=source_count,
                source_unique_verifier_count=source_unique_count,
                removed_verifier_count=len(remove_indices),
                replaced_verifier_count=len(replacement_indices),
                added_verifier_count=len(patch.add),
                rationale=patch.rationale,
            ),
        )

    names: dict[str, int] = {}
    existing_names = {str(verifier.get("name") or "") for verifier in working}
    normalized: list[Mapping[str, Any]] = []
    name_corrections: list[VerifierNameCorrection] = []
    for index, verifier in enumerate(working):
        source_name = str(verifier.get("name") or f"verifier_{index + 1}")
        ordinal = names.get(source_name, 0) + 1
        names[source_name] = ordinal
        updated = dict(verifier)
        if not verifier.get("name"):
            target_name = source_name
            updated["name"] = target_name
            name_corrections.append(
                VerifierNameCorrection(index=index, source_name="", target_name=target_name)
            )
        elif ordinal > 1:
            suffix = ordinal
            target_name = f"{source_name} [source check {suffix}]"
            while target_name in existing_names:
                suffix += 1
                target_name = f"{source_name} [source check {suffix}]"
            existing_names.add(target_name)
            updated["name"] = target_name
            name_corrections.append(
                VerifierNameCorrection(
                    index=index,
                    source_name=source_name,
                    target_name=target_name,
                )
            )
        normalized.append(updated)

    normalization = VerifierSetNormalization(
        source_count=source_count,
        effective_count=len(normalized),
        exact_duplicate_count=source_count - source_unique_count,
        name_corrections=tuple(name_corrections),
    )
    return normalized, corrections, normalization
