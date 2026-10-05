"""Native handoffs persist before yielding, and cannot execute the rest of a batch."""
import json
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc
from tools import kanban_tools as kt


@pytest.fixture
def worker(tmp_path, monkeypatch):
    from tests.agent.test_iteration_budget_warning import _agent

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    with kbc.connect() as conn:
        tid = kb.create_task(conn, title="handoff", assignee="coder")
        kb.claim_task(conn, tid)
        run_id = kb.get_task(conn, tid).current_run_id
    monkeypatch.setenv("HERMES_KANBAN_TASK", tid)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    monkeypatch.setenv("HERMES_PROFILE", "coder")
    monkeypatch.setattr(kt, "_goal_gate", lambda *a: None)
    monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: True)
    agent = _agent(tmp_path, monkeypatch, "null")
    agent._disable_streaming = True
    yield agent, tid, run_id
    agent.clear_interrupt()
    agent._session_db.close()


def call(name, args, cid):
    return SimpleNamespace(id=cid, type="function", function=SimpleNamespace(
        name=name, arguments=json.dumps(args)))


def response(calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason="tool_calls" if calls else "stop",
        message=SimpleNamespace(role="assistant", content=None if calls else "finished",
                                tool_calls=calls, reasoning=None))], usage=None)


@pytest.mark.parametrize("handoff", ["kanban_request_review", "kanban_request_changes",
                                    "kanban_complete", "kanban_block"])
@pytest.mark.parametrize("last_iteration", [False, True])
def test_accepted_handoff_stops_model_and_later_tool(worker, monkeypatch, handoff, last_iteration):
    agent, tid, run_id = worker
    if last_iteration:
        agent.max_iterations = 1
    monkeypatch.setattr("agent.turn_tool_round.compress_after_tool_results",
                        lambda *a, **kw: pytest.fail("compression admitted after handoff"))
    if handoff == "kanban_request_changes":
        with kbc.connect() as conn:
            assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=run_id)
            kb.claim_review_task(conn, tid)
            run_id = kb.get_task(conn, tid).current_run_id
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))
    args = {"task_id": None, "summary": "verified", "reviewer": "reviewer"} if handoff == "kanban_request_review" else (
        {"summary": "verified"} if handoff == "kanban_complete" else {"reason": "needs changes"})
    calls = [call(handoff, args, "handoff"),
             call("kanban_comment", {"task_id": tid, "body": "OLD WORKER MUST NOT ACT"}, "late")]
    client = Mock()
    client.chat.completions.create.side_effect = [response(calls), response()]
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    agent.client = client
    result = agent.run_conversation("perform the handoff")
    assert client.chat.completions.create.call_count == 1
    assert result["interrupted"] is True
    assert "kanban_handoff" in result["turn_exit_reason"]
    persisted = agent._session_db.get_messages(agent.session_id)
    assert any(m.get("role") == "tool" and m.get("tool_call_id") == "handoff"
               and '"ok": true' in str(m.get("content")) for m in persisted)
    with kbc.connect() as conn:
        assert kb.goal_run_status(conn, tid, run_id) != "running"
        assert not any("OLD WORKER" in c.body for c in kb.list_comments_after(conn, tid, after_id=0))
        kt._comment_watermark[tid] = 0
        kt._comment_poll_last_attempt = 0
        kb.add_comment(conn, tid, author="operator", body="note for successor")
    agent.steer = Mock()
    assert kt.inject_new_comments_from_env(agent) is False
    agent.steer.assert_not_called()


@pytest.mark.parametrize("rejection", ["empty_summary", "unknown_reviewer", "stale_run"])
def test_rejected_handoff_keeps_worker_running(worker, monkeypatch, rejection):
    agent, tid, run_id = worker
    args = {"summary": ""} if rejection == "empty_summary" else {"summary": "verified", "reviewer": "reviewer"}
    if rejection == "unknown_reviewer":
        monkeypatch.setattr("hermes_cli.profiles.profile_exists", lambda name: False)
    if rejection == "stale_run":
        monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id + 1000))
    calls = [call("kanban_request_review", args, "rejected"),
             call("kanban_comment", {"task_id": tid, "body": "recovered normally"}, "next")]
    client = Mock()
    client.chat.completions.create.side_effect = [response(calls), response()]
    monkeypatch.setattr(agent, "_create_request_openai_client", lambda **kw: client)
    agent.client = client
    result = agent.run_conversation("attempt handoff then recover")
    assert client.chat.completions.create.call_count == 2
    assert result["interrupted"] is False
    with kbc.connect() as conn:
        assert kb.goal_run_status(conn, tid, run_id) == "running"
        assert any(c.body == "recovered normally" for c in kb.list_comments_after(conn, tid, after_id=0))


def test_accepted_handoff_stops_later_parallel_file_segment(worker, monkeypatch, tmp_path):
    agent, tid, run_id = worker
    from agent.tool_executor import execute_tool_calls_segmented
    from agent.tool_dispatch_helpers import _plan_tool_batch_segments

    path = tmp_path / "must-not-be-written.txt"
    calls = [call("kanban_request_review", {"summary": "verified", "reviewer": "reviewer"}, "handoff"),
             call("write_file", {"path": str(path), "content": "stale edit"}, "late-write"),
             call("read_file", {"path": str(tmp_path / "other.txt")}, "late-read")]
    segments = _plan_tool_batch_segments(calls, execution_cwd=tmp_path)
    assert [kind for kind, _ in segments] == ["sequential", "parallel"]
    messages = [{"role": "user", "content": "handoff"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": tc.id, "type": "function", "function": {
                        "name": tc.function.name, "arguments": tc.function.arguments}} for tc in calls]}]
    execute_tool_calls_segmented(agent, SimpleNamespace(tool_calls=calls), messages, tid, segments=segments)
    assert agent._interrupt_requested is True
    assert not path.exists()
    assert all("skipped" in m["content"] for m in messages[-2:])


def test_orchestrator_handoff_does_not_interrupt_its_own_conversation(worker, monkeypatch):
    agent, tid, run_id = worker
    from agent.kanban_stop import yield_after_kanban_handoff
    from agent.delegation_context import non_dispatcher_owned_context

    with kbc.connect() as conn:
        assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=run_id)
    with non_dispatcher_owned_context():
        yield_after_kanban_handoff(agent, "kanban_request_review", {})
    assert agent._interrupt_requested is False
