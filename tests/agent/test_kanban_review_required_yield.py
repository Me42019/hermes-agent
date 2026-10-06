"""T016-B authority rejection and approval preserve T016-A turn semantics."""
from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from tests.agent.test_kanban_handoff_yield import call, response, worker as worker


@pytest.mark.parametrize("transition", ["implementation_complete", "request_review", "reviewer_complete"])
def test_required_review_handoffs_yield_only_after_accepted_authority(worker, monkeypatch, transition):
    agent, task_id, run_id = worker
    with kbc.connect() as conn, kb.write_txn(conn):
        kb._append_event(conn, task_id, "review_required", {"required": True})
    if transition == "reviewer_complete":
        with kbc.connect() as conn:
            assert kb.request_review(conn, task_id, reviewer="reviewer", expected_run_id=run_id)
            run_id = kb.claim_review_task(conn, task_id).current_run_id
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    name = "kanban_request_review" if transition == "request_review" else "kanban_complete"
    args = {"summary": "Verified", **({"reviewer": "reviewer"} if transition == "request_review" else {})}
    calls = [call(name, args, "handoff"), call("kanban_comment", {"task_id": task_id, "body": "Next tool"}, "later")]
    client = Mock()
    client.chat.completions.create.side_effect = [response(calls), response()]
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kwargs: client)
    agent.client = client
    result = agent.run_conversation("Perform the lifecycle transition")
    accepted = transition != "implementation_complete"
    assert result["interrupted"] is accepted
    assert client.chat.completions.create.call_count == (1 if accepted else 2)
    with kbc.connect() as conn:
        comments = kb.list_comments_after(conn, task_id, after_id=0)
        assert any(c.body == "Next tool" for c in comments) is (not accepted)
        assert (kb.goal_run_status(conn, task_id, run_id) != "running") is accepted
