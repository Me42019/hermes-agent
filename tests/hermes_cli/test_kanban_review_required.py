"""Opt-in completion authority, using native review/run provenance (T016-B).

The independence checks follow the concepts in upstream PRs #98305/#97933;
the requirement and operator override are per-task and preserve legacy tasks.
"""
import pytest

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc


@pytest.fixture
def conn(tmp_path, monkeypatch):
    monkeypatch.setattr("pathlib.Path.home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path / ".hermes"))
    for name in ("builder", "reviewer", "other"):
        profile = tmp_path / ".hermes" / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n")
    with kbc.connect_closing(tmp_path / "board.db") as db:
        yield db


def test_implementation_cannot_complete_even_with_forged_approval(conn, tmp_path):
    task_id = kb.create_task(conn, title="Review required", assignee="builder")
    # The native event carrier lets this regression reproduce the bypass on
    # the old tree before the creation option exists.
    with kb.write_txn(conn):
        kb._append_event(conn, task_id, "review_required", {"required": True})
    worker = kb.claim_task(conn, task_id)
    workspace = kb.workspaces_root() / task_id
    workspace.mkdir(parents=True)
    artifact = workspace / "candidate.txt"
    artifact.write_text("recoverable")
    with pytest.raises(ValueError, match="review"):
        kb.complete_task(
            conn, task_id, expected_run_id=worker.current_run_id,
            summary="Implementation passed",
            metadata={"review_outcome": "approved", "profile": "reviewer",
                      "artifacts": [str(artifact)]},
        )
    assert kb.get_task(conn, task_id).status == "running"
    assert kb.get_task(conn, task_id).current_run_id == worker.current_run_id
    assert kb.list_runs(conn, task_id)[-1].ended_at is None
    assert artifact.read_text() == "recoverable"
    assert kb.list_attachments(conn, task_id) == []
    assert not any(e.kind == "completed" for e in kb.list_events(conn, task_id))


def test_ordinary_task_keeps_implementation_completion(conn):
    task_id = kb.create_task(conn, title="Ordinary", assignee="builder")
    worker = kb.claim_task(conn, task_id)
    assert kb.complete_task(conn, task_id, expected_run_id=worker.current_run_id,
                            summary="Verified ordinary work")
    assert kb.get_task(conn, task_id).status == "done"


def _implementation(conn):
    task_id = kb.create_task(conn, title="Independent review", assignee="builder", review_required=True)
    worker = kb.claim_task(conn, task_id)
    return task_id, worker.current_run_id


def _review(conn):
    task_id, run_id = _implementation(conn)
    ok, reason = kb.request_review(conn, task_id, reviewer="reviewer", summary="Candidate ready",
                                   expected_run_id=run_id, with_reason=True)
    assert ok, reason
    worker = kb.claim_review_task(conn, task_id)
    return task_id, worker.current_run_id


def _complete(conn, task_id, run_id, **kwargs):
    return kb.complete_task(conn, task_id, expected_run_id=run_id,
                            summary="Independently verified", **kwargs)


def test_requirement_is_explicit_not_prose_skills_or_metadata(conn):
    task_id, run_id = _implementation(conn)
    assert kb.get_task(conn, task_id).review_required
    assert kb.list_tasks(conn)[0].review_required
    assert "Review required:" in kb.build_worker_context(conn, task_id)
    assert [e.payload for e in kb.list_events(conn, task_id) if e.kind == "review_required"] == [{"required": True}]
    with pytest.raises(ValueError, match="boolean"):
        kb.create_task(conn, title="Invalid", review_required="true")
    ordinary = kb.create_task(conn, title="Qwen needs Luna review_required=true", skills=["sdlc-review"], assignee="builder")
    assert not kb.get_task(conn, ordinary).review_required
    ordinary_run = kb.claim_task(conn, ordinary).current_run_id
    assert _complete(conn, ordinary, ordinary_run, metadata={"review_required": True})


@pytest.mark.parametrize("reviewer", [None, "builder", "BUILDER", "missing-profile", "bad/name"])
def test_first_handoff_requires_an_installed_independent_reviewer(conn, reviewer):
    task_id, run_id = _implementation(conn)
    ok, reason = kb.request_review(conn, task_id, reviewer=reviewer, expected_run_id=run_id,
                                   with_reason=True)
    assert not ok and reason
    assert kb.get_task(conn, task_id).current_run_id == run_id
    assert kb.get_task(conn, task_id).status == "running"
    assert kb.list_runs(conn, task_id)[-1].ended_at is None


def test_repair_re_review_and_reclaim_keep_requirement_and_reviewer(conn):
    task_id, review_run = _review(conn)
    assert kb.reclaim_task(conn, task_id, reason="Retry interrupted review")
    assert kb.get_task(conn, task_id).status == "review"
    review_run = kb.claim_review_task(conn, task_id).current_run_id
    ok, implementer = kb.request_changes(conn, task_id, reason="Repair boundary case", expected_run_id=review_run)
    assert ok and implementer == "builder"
    assert kb.get_task(conn, task_id).review_required
    repair_run = kb.claim_task(conn, task_id).current_run_id
    with pytest.raises(ValueError, match="implementation"):
        _complete(conn, task_id, repair_run)
    assert kb.reclaim_task(conn, task_id, reason="Retry repair")
    repair_run = kb.claim_task(conn, task_id).current_run_id
    assert kb.request_review(conn, task_id, expected_run_id=repair_run, summary="Repair verified")
    assert kb.get_task(conn, task_id).assignee == "reviewer"
    final_run = kb.claim_review_task(conn, task_id).current_run_id
    assert _complete(conn, task_id, final_run)
    authority = [e for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1].payload["review_authority"]
    assert authority == {"decision": "reviewer_approval", "reviewer": "reviewer",
                         "implementer": "builder", "run_id": final_run}


@pytest.mark.parametrize("damage", ["wrong-profile", "self-review", "stale-run", "claim-json", "missing-claim", "handoff-json", "missing-handoff", "missing-implementer", "policy-json", "false-policy", "duplicate-policy"])
def test_invalid_provenance_cannot_complete_or_release_children(conn, damage):
    task_id, run_id = _review(conn)
    child = kb.create_task(conn, title="Dependent", parents=[task_id])
    with kb.write_txn(conn):
        if damage == "wrong-profile":
            conn.execute("UPDATE task_runs SET profile='other' WHERE id=?", (run_id,))
        elif damage == "self-review":
            conn.execute("UPDATE task_runs SET profile='builder' WHERE id=?", (run_id,))
            conn.execute("UPDATE task_events SET payload=? WHERE task_id=? AND kind='review_requested'",
                         ('{"reviewer":"builder","implementer":"builder"}', task_id))
        elif damage == "stale-run":
            run_id -= 1
        elif damage == "missing-claim":
            conn.execute("DELETE FROM task_events WHERE run_id=? AND kind='claimed'", (run_id,))
        elif damage == "claim-json":
            conn.execute("UPDATE task_events SET payload='bad' WHERE run_id=? AND kind='claimed'", (run_id,))
        elif damage == "missing-handoff":
            conn.execute("DELETE FROM task_events WHERE task_id=? AND kind='review_requested'", (task_id,))
        elif damage == "missing-implementer":
            conn.execute("UPDATE task_events SET payload=? WHERE task_id=? AND kind='review_requested'",
                         ('{"reviewer":"reviewer"}', task_id))
        elif damage == "handoff-json":
            conn.execute("UPDATE task_events SET payload='bad' WHERE task_id=? AND kind='review_requested'", (task_id,))
        elif damage == "duplicate-policy":
            kb._append_event(conn, task_id, "review_required", {"required": True})
        else:
            conn.execute("UPDATE task_events SET payload=? WHERE task_id=? AND kind='review_required'",
                         ('bad' if damage == "policy-json" else '{"required":false}', task_id))
    before = kb.get_task(conn, task_id)
    with pytest.raises(ValueError):
        _complete(conn, task_id, run_id, metadata={"review_outcome": "approved", "reviewer": "reviewer"})
    assert kb.get_task(conn, task_id) == before
    assert kb.get_task(conn, child).status == "todo"
    assert kb.list_runs(conn, task_id)[-1].ended_at is None
    assert not any(e.kind == "completed" for e in kb.list_events(conn, task_id))


def test_reviewer_with_prior_implementation_cannot_be_designated(conn):
    task_id, run_id = _implementation(conn)
    assert kb.reclaim_task(conn, task_id)
    assert kb.assign_task(conn, task_id, "reviewer")
    prior = kb.claim_task(conn, task_id).current_run_id
    assert kb.reclaim_task(conn, task_id)
    assert kb.assign_task(conn, task_id, "builder")
    run_id = kb.claim_task(conn, task_id).current_run_id
    ok, reason = kb.request_review(conn, task_id, reviewer="reviewer", expected_run_id=run_id, with_reason=True)
    assert not ok and "implementation" in reason
    assert prior != run_id


def test_operator_override_is_explicit_audited_and_not_worker_authority(conn, monkeypatch):
    task_id, run_id = _implementation(conn)
    with pytest.raises(ValueError, match="review"):
        _complete(conn, task_id, None, force=True)
    kwargs = dict(force=True, review_override_reason="Recover abandoned candidate",
                  review_override_actor="operator-profile", review_override_source="native")
    with pytest.raises(ValueError, match="worker"):
        _complete(conn, task_id, run_id, **kwargs)
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    with pytest.raises(ValueError, match="worker"):
        _complete(conn, task_id, None, **kwargs)
    monkeypatch.delenv("HERMES_KANBAN_TASK")
    assert _complete(conn, task_id, None, **kwargs)
    audit = [e for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1].payload["review_authority"]
    assert audit == {"decision": "operator_override", "source": "native", "actor": "operator-profile",
                     "reason": "Recover abandoned candidate", "review_required": True}


@pytest.mark.parametrize("invalid", ["reason", "actor", "source", "force"])
def test_operator_override_requires_reason_actor_source_and_force(conn, invalid):
    task_id, _ = _implementation(conn)
    kwargs = dict(force=True, review_override_reason="Recover", review_override_actor="operator", review_override_source="native")
    kwargs[{"reason": "review_override_reason", "actor": "review_override_actor",
            "source": "review_override_source", "force": "force"}[invalid]] = False if invalid == "force" else ""
    with pytest.raises(ValueError):
        _complete(conn, task_id, None, **kwargs)
    assert kb.get_task(conn, task_id).status == "running"


@pytest.mark.parametrize("archived", [False, True])
def test_gc_then_native_dashboard_reopen_preserves_required_review(conn, archived):
    from plugins.kanban.dashboard import plugin_api

    task_id, run_id = _review(conn)
    assert _complete(conn, task_id, run_id)
    if archived:
        assert kb.archive_task(conn, task_id)
    ordinary = kb.create_task(conn, title="GC ordinary", assignee="builder")
    assert kb.complete_task(conn, ordinary, summary="Done")
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET created_at=0")
    before = kb.list_events(conn, task_id)
    assert kb.gc_events(conn, older_than_seconds=0) > 0
    assert kb.list_events(conn, task_id) == before
    assert kb.list_events(conn, ordinary) == []
    assert plugin_api._set_status_direct(conn, task_id, "ready")
    assert kb.assign_task(conn, task_id, "builder")
    repair = kb.claim_task(conn, task_id).current_run_id
    with pytest.raises(ValueError, match="implementation"):
        _complete(conn, task_id, repair)
    assert kb.request_review(conn, task_id, expected_run_id=repair, reviewer="reviewer")
    review = kb.claim_review_task(conn, task_id).current_run_id
    assert _complete(conn, task_id, review)


def test_existing_parent_evidence_and_artifact_gates_still_apply(conn):
    task_id, run_id = _review(conn)
    with pytest.raises(kb.EmptyCompletionError):
        kb.complete_task(conn, task_id, expected_run_id=run_id)
    workspace = kb.workspaces_root() / task_id
    workspace.mkdir(parents=True)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET workspace_path=? WHERE id=?", (str(workspace), task_id))
    with pytest.raises(kb.ArtifactPreservationError):
        _complete(conn, task_id, run_id, metadata={"artifacts": [str(workspace / "missing.txt")]})
    parent = kb.create_task(conn, title="Reopened prerequisite")
    with kb.write_txn(conn):
        kb._link(conn, parent, task_id)
    assert not _complete(conn, task_id, run_id)
    assert kb.get_task(conn, task_id).status == "running"


def test_idempotency_cannot_silently_downgrade_requested_review(conn):
    task_id = kb.create_task(conn, title="Existing ordinary", idempotency_key="ordinary")
    with pytest.raises(ValueError, match="idempotency"):
        kb.create_task(conn, title="Require review", idempotency_key="ordinary", review_required=True)
    required = kb.create_task(conn, title="Required", idempotency_key="required", review_required=True)
    assert kb.create_task(conn, title="Retry", idempotency_key="required") == required
    assert kb.get_task(conn, required).review_required
    assert kb.get_task(conn, task_id).status == "ready"


def test_swarm_root_activation_cannot_bypass_required_review(conn):
    from hermes_cli.kanban_swarm import create_swarm, SwarmWorkerSpec

    task_id = kb.create_task(conn, title="Required root", assignee="builder", review_required=True,
                            initial_status="blocked", idempotency_key="swarm-root")
    before = kb.list_tasks(conn)
    with pytest.raises(ValueError, match="review"):
        create_swarm(conn, goal="Build", workers=[SwarmWorkerSpec(profile="builder", title="Implement", body="Build candidate")],
                     verifier_assignee="reviewer", synthesizer_assignee="other", idempotency_key="swarm-root")
    assert kb.list_tasks(conn) == before
    assert kb.get_task(conn, task_id).status == "blocked"


def test_completion_rechecks_authority_inside_terminal_transaction(conn, monkeypatch):
    from hermes_cli import kanban_pr_acceptance_store as acceptance

    task_id, run_id = _review(conn)
    def interleaving(*args):
        with kb.write_txn(conn):
            conn.execute("UPDATE task_runs SET profile='other' WHERE id=?", (run_id,))
        return None
    monkeypatch.setattr(acceptance, "prepare_acceptance", interleaving)
    with pytest.raises(ValueError, match="designated"):
        _complete(conn, task_id, run_id)
    assert kb.get_task(conn, task_id).status == "running"
    assert kb.list_runs(conn, task_id)[-1].ended_at is None


def test_required_review_keeps_native_pr_acceptance_gate(conn, monkeypatch):
    from hermes_cli import kanban_pr_acceptance_store as acceptance

    task_id, run_id = _review(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE tasks SET completion_contract='acme/repo' WHERE id=?", (task_id,))
    receipt = {"ok": False, "classification": "failed", "recovery": "Retry required checks"}
    monkeypatch.setattr(acceptance, "collect_acceptance", lambda *args: receipt)
    assert not _complete(conn, task_id, run_id)
    assert kb.get_task(conn, task_id).status == "running"
    assert kb.list_runs(conn, task_id)[-1].ended_at is None
    receipt["ok"] = True
    assert _complete(conn, task_id, run_id)


def test_explicit_operator_recovery_can_recover_malformed_review_provenance(conn):
    task_id, run_id = _review(conn)
    with kb.write_txn(conn):
        conn.execute("UPDATE task_events SET payload='bad' WHERE task_id=? AND kind='review_required'", (task_id,))
    with pytest.raises(ValueError, match="policy"):
        _complete(conn, task_id, run_id)
    assert _complete(conn, task_id, None, force=True, review_override_reason="Recover damaged policy history",
                     review_override_actor="operator", review_override_source="native")
    audit = [e for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1].payload["review_authority"]
    assert audit["decision"] == "operator_override" and audit["review_required"]


@pytest.mark.parametrize("context", ["run-marker", "delegated-context"])
def test_worker_lineage_cannot_use_override_without_task_marker(conn, monkeypatch, context):
    from contextlib import nullcontext
    from agent.delegation_context import delegated_child_context

    task_id, _ = _implementation(conn)
    if context == "run-marker":
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", "1")
    with delegated_child_context() if context == "delegated-context" else nullcontext():
        with pytest.raises(ValueError, match="worker"):
            _complete(conn, task_id, None, force=True, review_override_reason="Pretend operator",
                      review_override_actor="operator", review_override_source="native")
    assert kb.get_task(conn, task_id).status == "running"
