#!/usr/bin/env python3
"""Audit the pinned Teams 3.11 list_channels implementation and compatibility."""

from __future__ import annotations

import dis
import inspect
import json
import sys
from datetime import UTC, datetime
from typing import Any


def query_initialization_evidence(method: Any) -> dict[str, Any]:
    """Prove query is assigned before the filter helper is reached."""

    instructions = list(dis.get_instructions(method))
    query_stores = [
        instruction
        for instruction in instructions
        if instruction.opname == "STORE_FAST" and instruction.argval == "query"
    ]
    filter_loads = [
        instruction
        for instruction in instructions
        if instruction.opname in {"LOAD_ATTR", "LOAD_METHOD"}
        and instruction.argval == "_apply_odata_filter"
    ]
    if not query_stores or not filter_loads:
        raise AssertionError(
            "list_channels no longer exposes the expected query/filter flow"
        )
    if query_stores[0].offset >= filter_loads[0].offset:
        raise AssertionError(
            "list_channels reaches its filter helper before assigning query"
        )
    return {
        "query_store_offset": query_stores[0].offset,
        "filter_helper_offset": filter_loads[0].offset,
        "query_initialized_before_filter": True,
    }


def startswith_helper_defect_evidence(method: Any) -> dict[str, Any]:
    """Prove the pinned helper reads an unbound local query in startswith."""

    instructions = list(dis.get_instructions(method))
    query_loads = [
        instruction
        for instruction in instructions
        if instruction.opname == "LOAD_FAST" and instruction.argval == "query"
    ]
    query_stores = [
        instruction
        for instruction in instructions
        if instruction.opname == "STORE_FAST" and instruction.argval == "query"
    ]
    if "query" in inspect.signature(method).parameters:
        raise AssertionError("filter helper unexpectedly accepts query")
    if (
        not query_loads
        or not query_stores
        or query_loads[0].offset >= query_stores[0].offset
    ):
        raise AssertionError("pinned startswith unbound-query defect was not found")
    return {
        "first_query_load_offset": query_loads[0].offset,
        "first_query_store_offset": query_stores[0].offset,
        "query_is_unbound_at_first_load": True,
    }


def _ids(rows: list[dict[str, Any]]) -> list[str]:
    return [row["id"] for row in rows]


def behavior_evidence() -> dict[str, Any]:
    """Run the loaded manager against an isolated in-memory SQLite database."""

    import database.managers.channel_manager as channel_manager_module
    from database.managers.channel_manager import ChannelManager
    from database.models.channel import (
        Channel,
        ChannelMember,
        ChannelMembershipType,
    )
    from database.models.org import Org
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from sqlalchemy.pool import StaticPool

    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Org.__table__.create(engine)
    Channel.__table__.create(engine)
    ChannelMember.__table__.create(engine)
    session_factory = sessionmaker(bind=engine)
    session = session_factory()
    now = datetime(2026, 9, 7, tzinfo=UTC).replace(tzinfo=None)
    session.add(
        Org(
            org_id="org-a",
            tenant_id="tenant-a",
            display_name="Audit Org",
            soft_deleted=False,
        )
    )
    channels = [
        Channel(
            org_id="org-a",
            channel_id="channel-alpha",
            team_id="team-a",
            display_name="Alpha",
            membership_type=ChannelMembershipType.STANDARD,
            is_favorite_by_default=False,
            is_archived=False,
            created_date_time=now.replace(day=4),
            soft_delete=False,
        ),
        Channel(
            org_id="org-a",
            channel_id="channel-secret",
            team_id="team-a",
            display_name="Secret",
            membership_type=ChannelMembershipType.PRIVATE,
            is_favorite_by_default=False,
            is_archived=False,
            created_date_time=now.replace(day=3),
            soft_delete=False,
        ),
        Channel(
            org_id="org-a",
            channel_id="channel-hidden",
            team_id="team-a",
            display_name="Hidden",
            membership_type=ChannelMembershipType.PRIVATE,
            is_favorite_by_default=False,
            is_archived=False,
            created_date_time=now.replace(day=2),
            soft_delete=False,
        ),
        Channel(
            org_id="org-a",
            channel_id="channel-beta",
            team_id="team-a",
            display_name="Beta",
            membership_type=ChannelMembershipType.STANDARD,
            is_favorite_by_default=False,
            is_archived=False,
            created_date_time=now.replace(day=1),
            soft_delete=False,
        ),
    ]
    session.add_all(channels)
    session.add_all(
        [
            ChannelMember(
                channel_id="channel-secret",
                user_id="user-a",
                roles="member",
                added_at=now,
            ),
            ChannelMember(
                channel_id="channel-hidden",
                user_id="user-b",
                roles="member",
                added_at=now,
            ),
        ]
    )
    session.commit()
    session.close()

    original_get_session = channel_manager_module.get_session
    channel_manager_module.get_session = lambda database_id: session_factory()
    manager = ChannelManager("in-memory-audit")
    try:
        visible = manager.list_channels("team-a", user_id="user-a")
        startswith = manager.list_channels(
            "team-a", user_id="user-a", filter_expr="startswith(displayName,'Al')"
        )
        membership = manager.list_channels(
            "team-a", user_id="user-a", filter_expr="membershipType eq 'private'"
        )
        combined = manager.list_channels(
            "team-a",
            user_id="user-a",
            filter_expr=(
                "startswith(displayName,'Al') and membershipType eq 'standard'"
            ),
        )
        disjunction = manager.list_channels(
            "team-a",
            user_id="user-a",
            filter_expr="displayName eq 'Alpha' or displayName eq 'Beta'",
        )
        page = manager.list_channels(
            "team-a",
            user_id="user-a",
            filter_expr="membershipType eq 'standard'",
            limit=1,
            offset=1,
        )
        selected = manager.list_channels(
            "team-a",
            user_id="user-a",
            filter_expr="displayName eq 'Alpha'",
            select_fields="displayName,membershipType",
        )
        try:
            manager.list_channels(
                "team-a", user_id="user-a", filter_expr="unsupported(foo)"
            )
        except Exception as exc:  # noqa: BLE001 - exact image exception is evidence
            invalid_filter = type(exc).__name__
        else:
            raise AssertionError("unsupported Teams filter was silently accepted")
    finally:
        channel_manager_module.get_session = original_get_session
        engine.dispose()

    expected = {
        "visible": ["channel-alpha", "channel-secret", "channel-beta"],
        "startswith": ["channel-alpha"],
        "membership": ["channel-secret"],
        "combined": ["channel-alpha"],
        "disjunction": ["channel-alpha", "channel-beta"],
        "page": ["channel-beta"],
    }
    observed = {
        "visible": _ids(visible),
        "startswith": _ids(startswith),
        "membership": _ids(membership),
        "combined": _ids(combined),
        "disjunction": _ids(disjunction),
        "page": _ids(page),
    }
    if observed != expected:
        raise AssertionError(f"Teams list_channels behavior drifted: {observed!r}")
    if len(selected) != 1 or set(selected[0]) != {
        "@odata.id",
        "displayName",
        "membershipType",
    }:
        raise AssertionError(f"Teams select projection drifted: {selected!r}")
    return {
        **observed,
        "private_nonmember_excluded": "channel-hidden" not in observed["visible"],
        "select_keys": sorted(selected[0]),
        "invalid_filter_exception": invalid_filter,
    }


def main() -> None:
    import database.managers.channel_manager as channel_manager_module
    from database.managers.channel_manager import ChannelManager
    from database.models.channel import Channel

    from backends.enterpriseops.bootstrap_teams import (
        patch_list_channels_filter_contract,
    )

    module_file = channel_manager_module.__file__
    if not module_file.endswith("channel_manager.pyc"):
        raise AssertionError(f"unexpected loaded Teams manager: {module_file}")
    if sys.version_info[:2] != (3, 11):
        raise AssertionError(f"unexpected Teams Python: {sys.version}")
    original_filter_helper = ChannelManager._build_channel_filter_condition
    defect = startswith_helper_defect_evidence(original_filter_helper)
    patch_list_channels_filter_contract(ChannelManager, Channel)
    print(
        json.dumps(
            {
                "python": sys.version.split()[0],
                "module_file": module_file,
                "signature": str(inspect.signature(ChannelManager.list_channels)),
                "bytecode": query_initialization_evidence(ChannelManager.list_channels),
                "pinned_startswith_defect": defect,
                "behavior": behavior_evidence(),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
