"""Real tool/CLI/dashboard resolution against isolated native boards (T016-B)."""
import argparse
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from hermes_cli import kanban as kc
from hermes_cli import kanban_db as kb
from hermes_cli import kanban_db_connect as kbc
from plugins.kanban.dashboard import plugin_api
from tools import kanban_tools as tools


@pytest.fixture
def board(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    for name in ("builder", "reviewer"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n")
    monkeypatch.setenv("HERMES_PROFILE", "operator-profile")
    app = FastAPI()
    app.include_router(plugin_api.router)
    with TestClient(app) as client:
        yield client


def _cli(*argv):
    parser = argparse.ArgumentParser()
    kc.build_parser(parser.add_subparsers(dest="command"))
    return kc.kanban_command(parser.parse_args(["kanban", *argv]))


def _worker(monkeypatch, task_id, run_id):
    monkeypatch.setenv("HERMES_KANBAN_TASK", task_id)
    monkeypatch.setenv("HERMES_KANBAN_RUN_ID", str(run_id))


def _create(required):
    with kbc.connect_closing() as conn:
        task_id = kb.create_task(conn, title="Surface authority", assignee="builder", review_required=required)
        run_id = kb.claim_task(conn, task_id).current_run_id
        return task_id, run_id


@pytest.mark.parametrize("surface", ["tool", "cli", "dashboard", "bulk"])
@pytest.mark.parametrize("required", [False, True])
def test_every_surface_preserves_ordinary_completion_and_rejects_required_implementation(board, monkeypatch, surface, required):
    task_id, run_id = _create(required)
    if surface in {"tool", "cli"}:
        _worker(monkeypatch, task_id, run_id)
    if surface == "tool":
        result = json.loads(tools._handle_complete({"summary": "Verified implementation"}))
        accepted = result.get("ok") is True
    elif surface == "cli":
        accepted = _cli("complete", task_id, "--summary", "Verified implementation") == 0
    elif surface == "dashboard":
        result = board.patch(f"/tasks/{task_id}", json={"status": "done", "summary": "Verified implementation"})
        accepted = result.status_code == 200
        if required:
            assert "review" in result.json()["detail"]
    else:
        result = board.post("/tasks/bulk", json={"ids": [task_id], "status": "done", "summary": "Verified implementation"})
        accepted = result.json()["results"][0]["ok"]
    assert accepted is (not required)
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).status == ("running" if required else "done")
        assert (kb.list_runs(conn, task_id)[-1].ended_at is None) is required


@pytest.mark.parametrize("surface", ["tool", "cli"])
def test_designated_review_run_completes_on_worker_surfaces(board, monkeypatch, surface):
    task_id, run_id = _create(True)
    _worker(monkeypatch, task_id, run_id)
    requested = json.loads(tools._handle_request_review({"reviewer": "reviewer", "summary": "Ready"}))
    assert requested["ok"] is True
    with kbc.connect_closing() as conn:
        run_id = kb.claim_review_task(conn, task_id).current_run_id
    _worker(monkeypatch, task_id, run_id)
    if surface == "tool":
        assert json.loads(tools._handle_complete({"summary": "Independently verified"}))["ok"] is True
    else:
        assert _cli("complete", task_id, "--summary", "Independently verified") == 0
    with kbc.connect_closing() as conn:
        audit = [e for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1].payload
        assert audit["review_authority"]["reviewer"] == "reviewer"


@pytest.mark.parametrize("surface", ["cli", "dashboard", "bulk"])
def test_operator_override_is_explicit_and_audited_at_each_surface(board, surface):
    task_id, run_id = _create(True)
    reason = "Recover abandoned implementation"
    if surface == "cli":
        assert _cli("complete", task_id, "--force", "--summary", "Recovered") != 0
        assert _cli("complete", task_id, "--force", "--override-review", reason, "--summary", "Recovered") == 0
    elif surface == "dashboard":
        assert board.patch(f"/tasks/{task_id}", json={"status": "done", "summary": "Recovered"}).status_code != 200
        result = board.patch(f"/tasks/{task_id}", json={"status": "done", "summary": "Recovered", "review_override_reason": reason})
        assert result.status_code == 200, result.text
    else:
        result = board.post("/tasks/bulk", json={"ids": [task_id], "status": "done", "summary": "Recovered", "review_override_reason": reason})
        assert result.json()["results"][0]["ok"]
    with kbc.connect_closing() as conn:
        audit = [e for e in kb.list_events(conn, task_id) if e.kind == "completed"][-1].payload["review_authority"]
        assert audit["decision"] == "operator_override"
        assert audit["reason"] == reason
        assert audit["source"] == ("cli" if surface == "cli" else "dashboard")
        assert audit["actor"] == ("operator-profile" if surface == "cli" else "dashboard")


@pytest.mark.parametrize("surface", ["cli", "dashboard", "bulk", "tool"])
def test_worker_cannot_obtain_override_by_arguments(board, monkeypatch, surface):
    task_id, run_id = _create(True)
    _worker(monkeypatch, task_id, run_id)
    if surface == "cli":
        assert _cli("complete", task_id, "--force", "--override-review", "Pretend operator", "--summary", "Done") != 0
    elif surface == "dashboard":
        assert board.patch(f"/tasks/{task_id}", json={"status": "done", "summary": "Done", "review_override_reason": "Pretend operator"}).status_code != 200
    elif surface == "bulk":
        result = board.post("/tasks/bulk", json={"ids": [task_id], "status": "done", "summary": "Done", "review_override_reason": "Pretend operator"})
        assert not result.json()["results"][0]["ok"]
    else:
        result = json.loads(tools._handle_complete({"summary": "Done", "review_override_reason": "Pretend operator"}))
        assert result.get("ok") is not True
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).current_run_id == run_id
        assert kb.get_task(conn, task_id).status == "running"


@pytest.mark.parametrize("surface", ["tool", "cli", "dashboard"])
def test_creation_and_display_expose_explicit_review_requirement(board, capsys, surface):
    if surface == "tool":
        result = json.loads(tools._handle_create({"title": "Required creation", "assignee": "builder", "review_required": True}))
        assert result["ok"] is True, result
        task_id = result["task_id"]
    elif surface == "cli":
        assert _cli("create", "Required creation", "--assignee", "builder", "--review-required") == 0
        with kbc.connect_closing() as conn:
            task_id = kb.list_tasks(conn)[0].id
        assert _cli("show", task_id, "--json") == 0
        assert '"review_required": true' in capsys.readouterr().out
    else:
        result = board.post("/tasks", json={"title": "Required creation", "assignee": "builder", "review_required": True})
        assert result.status_code == 200, result.text
        assert result.json()["task"]["review_required"] is True
        task_id = result.json()["task"]["id"]
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).review_required
        assert "Review required:" in kb.build_worker_context(conn, task_id)
    shown = json.loads(tools._handle_show({"task_id": task_id}))
    assert shown["task"]["review_required"] is True


@pytest.mark.parametrize("judge", ["allows", "unavailable", "failure"])
def test_goal_judge_cannot_grant_implementation_review_authority(board, monkeypatch, judge):
    task_id, run_id = _create(True)
    _worker(monkeypatch, task_id, run_id)
    with kbc.connect_closing() as conn, kb.write_txn(conn):
        conn.execute("UPDATE tasks SET goal_mode=1 WHERE id=?", (task_id,))
    monkeypatch.setattr(tools, "_goal_judge_available", lambda: judge != "unavailable")
    def verdict(**kwargs):
        if judge == "failure":
            raise RuntimeError("synthetic judge outage")
        return "done", "Looks complete", {}, None, False
    monkeypatch.setattr(tools, "judge_goal", verdict)
    result = json.loads(tools._handle_complete({"summary": "Implementation verified", "metadata": {"review_outcome": "approved"}}))
    assert result.get("ok") is not True
    assert "review" in result["error"]
    with kbc.connect_closing() as conn:
        assert kb.get_task(conn, task_id).current_run_id == run_id
