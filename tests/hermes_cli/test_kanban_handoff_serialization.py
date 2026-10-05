"""A successor waits for the retained previous process, independently of reaper grace."""
import subprocess
import sys

import pytest

from hermes_cli import kanban_db as kb, kanban_db_connect as kbc, kanban_db_dispatch as kbd


@pytest.mark.parametrize("repair", [False, True])
def test_successor_waits_for_actual_previous_process_exit(tmp_path, monkeypatch, repair):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setattr(kbd, "_profile_exists_fn", lambda: lambda name: True)
    monkeypatch.setattr(kbd, "review_dispatch_enabled", lambda: True)
    monkeypatch.setattr(kbd, "_dispatch_profile_allowlist", lambda *a: None)
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="serialized", assignee="coder",
                                 workspace_kind="dir", workspace_path=str(tmp_path))
            kb.claim_task(conn, tid)
            old_run = kb.get_task(conn, tid).current_run_id
            if repair:
                assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=old_run)
                kb.claim_review_task(conn, tid)
                old_run = kb.get_task(conn, tid).current_run_id
            kbd._set_worker_pid(conn, tid, proc.pid)
            if repair:
                assert kb.request_changes(conn, tid, reason="repair", expected_run_id=old_run)[0]
            else:
                assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=old_run)
            spawned = []

            def spawn(task, workspace, **kw):
                assert proc.poll() is not None, "successor overlaps former worker"
                spawned.append(task.id)
                return 0

            result = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
            assert spawned == []
            assert (tid, "previous_worker_unwinding") in result.respawn_guarded, vars(result)
            assert result.reaped_terminal_workers == []
            assert proc.poll() is None
            # Clean unwind, without a kill or grace timer: the next tick may dispatch.
            proc.stdin.close()
            assert proc.wait(timeout=5) == 0
            result = kbd.dispatch_once(conn, spawn_fn=spawn, max_spawn=1)
            assert spawned == [tid]
            assert result.spawned[0][0] == tid
    finally:
        if proc.poll() is None:
            proc.kill()
        proc.wait(timeout=5)


@pytest.mark.parametrize("identity", [None, "unverified", "recycled", "unreadable"])
def test_previous_worker_guard_preserves_process_identity_rules(tmp_path, monkeypatch, identity):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    proc = subprocess.Popen([sys.executable, "-c", "import sys; sys.stdin.read()"], stdin=subprocess.PIPE)
    try:
        with kbc.connect() as conn:
            tid = kb.create_task(conn, title="identity", assignee="coder")
            kb.claim_task(conn, tid)
            run_id = kb.get_task(conn, tid).current_run_id
            kbd._set_worker_pid(conn, tid, proc.pid)
            assert kb.request_review(conn, tid, summary="ready", reviewer="reviewer", expected_run_id=run_id)
            if identity in {None, "unverified", "recycled"}:
                fingerprint = "foreign-boot|1" if identity == "recycled" else identity
                conn.execute("UPDATE task_runs SET worker_started_at=? WHERE id=?", (fingerprint, run_id))
            else:
                monkeypatch.setattr(kbd, "_process_fingerprint", lambda pid: None)
            expected = None if identity == "recycled" else "previous_worker_unwinding"
            assert kbd.check_respawn_guard(conn, tid, lane="review") == expected
            assert proc.poll() is None
    finally:
        proc.stdin.close()
        proc.wait(timeout=5)
