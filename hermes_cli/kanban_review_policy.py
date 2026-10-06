"""Opt-in same-card review authority on native task events and runs.

No schema or parallel workflow: the marker and provenance live in task_events.
Review-origin/profile independence follows upstream PRs #98305 and #97933.
Operator sources are trusted local API conventions, not OS authentication.
"""
from __future__ import annotations

import json
import os
import sqlite3
from typing import Optional


class ReviewRequiredError(ValueError):
    """Completion or review handoff lacks native review authority."""


def is_review_required(conn: sqlite3.Connection, task_id: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM task_events WHERE task_id = ? AND kind = 'review_required' LIMIT 1",
        (task_id,),
    ).fetchone() is not None


def _payload(row, description: str) -> dict:
    try:
        value = json.loads(row["payload"]) if row is not None else None
    except (TypeError, ValueError):
        value = None
    if not isinstance(value, dict):
        raise ReviewRequiredError(f"required review has missing/malformed {description} provenance")
    return value


def _required(conn, task_id: str) -> bool:
    rows = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_required'",
        (task_id,),
    ).fetchall()
    if not rows:
        return False
    if len(rows) != 1 or _payload(rows[0], "policy").get("required") is not True:
        raise ReviewRequiredError("required review has malformed/ambiguous policy provenance")
    return True


def _profile(value) -> str:
    from hermes_cli.profiles import normalize_profile_name

    if not isinstance(value, str) or not value.strip():
        raise ReviewRequiredError("required review has missing profile provenance")
    return normalize_profile_name(value)


def _review_origin(conn, task_id: str, run_id: int) -> bool:
    row = conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND run_id = ? "
        "AND kind = 'claimed' ORDER BY id DESC LIMIT 1", (task_id, run_id),
    ).fetchone()
    return _payload(row, "run claim").get("source_status") == "review"


def _independent(conn, task_id: str, reviewer: str, implementer: str) -> None:
    if reviewer == implementer:
        raise ReviewRequiredError("required review needs an independent reviewer; self-review is rejected")
    rows = conn.execute(
        "SELECT id, profile FROM task_runs WHERE task_id = ?", (task_id,),
    ).fetchall()
    for row in rows:
        if _profile(row["profile"]) == reviewer and not _review_origin(conn, task_id, row["id"]):
            raise ReviewRequiredError("required reviewer previously performed implementation work on this task")


def validate_handoff(conn, task_id: str, reviewer: Optional[str], implementer: Optional[str]) -> None:
    if not _required(conn, task_id):
        return
    from hermes_cli.profiles import profile_exists

    reviewer = _profile(reviewer)
    implementer = _profile(implementer)
    if not profile_exists(reviewer):
        raise ReviewRequiredError(f"required reviewer profile {reviewer!r} is not installed")
    _independent(conn, task_id, reviewer, implementer)
    current = conn.execute("SELECT current_run_id FROM tasks WHERE id = ?", (task_id,)).fetchone()
    if current and current["current_run_id"] is not None:
        if _review_origin(conn, task_id, current["current_run_id"]):
            raise ReviewRequiredError("review run must complete or request changes, not replace its review handoff")


def authorize_completion(
    conn, task_id: str, expected_run_id: Optional[int], *, force: bool,
    override_reason: Optional[str], override_actor: Optional[str], override_source: Optional[str],
) -> Optional[dict]:
    """Read under the completion transaction; return authoritative audit facts.

    A cheap preflight invokes this too, before existing gates can stage artifacts
    or collect PR receipts. Only the locked invocation authorizes the write.
    """
    if override_reason is not None:
        from agent.delegation_context import is_delegated_child_process_context

        if (expected_run_id is not None or os.environ.get("HERMES_KANBAN_TASK")
                or os.environ.get("HERMES_KANBAN_RUN_ID") or is_delegated_child_process_context()):
            raise ReviewRequiredError("worker context cannot use an operator review override")
        if (not force or override_source not in {"cli", "dashboard", "native"}
                or not isinstance(override_actor, str) or not override_actor.strip()
                or not isinstance(override_reason, str) or not override_reason.strip()):
            raise ReviewRequiredError("operator review override requires force, source, actor and an explicit reason")
        from agent.redact import redact_sensitive_text

        return {"decision": "operator_override", "source": override_source,
                "actor": redact_sensitive_text(override_actor.strip()),
                "reason": redact_sensitive_text(override_reason.strip()),
                "review_required": is_review_required(conn, task_id)}
    if not _required(conn, task_id):
        return None
    row = conn.execute(
        "SELECT status, current_run_id FROM tasks WHERE id = ?", (task_id,),
    ).fetchone()
    if (row is None or row["status"] != "running" or expected_run_id is None
            or row["current_run_id"] != int(expected_run_id)):
        raise ReviewRequiredError("required review completion needs the owned current review run; operator recovery requires an explicit override reason")
    run = conn.execute(
        "SELECT profile, status, ended_at FROM task_runs WHERE task_id = ? AND id = ?",
        (task_id, int(expected_run_id)),
    ).fetchone()
    if run is None or run["ended_at"] is not None or run["status"] != "running":
        raise ReviewRequiredError("required review completion has stale/malformed run provenance")
    if not _review_origin(conn, task_id, int(expected_run_id)):
        raise ReviewRequiredError("required review cannot be completed by an implementation run; request review instead")
    handoff = _payload(conn.execute(
        "SELECT payload FROM task_events WHERE task_id = ? AND kind = 'review_requested' "
        "ORDER BY id DESC LIMIT 1", (task_id,),
    ).fetchone(), "review handoff")
    reviewer, implementer = _profile(handoff.get("reviewer")), _profile(handoff.get("implementer"))
    if _profile(run["profile"]) != reviewer:
        raise ReviewRequiredError("required review can only be completed by the designated reviewer")
    _independent(conn, task_id, reviewer, implementer)
    return {"decision": "reviewer_approval", "reviewer": reviewer,
            "implementer": implementer, "run_id": int(expected_run_id)}
