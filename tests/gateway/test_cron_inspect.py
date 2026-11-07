"""cron.inspect reads an admission that is not in the live map and does not restore it.

cron.status on that same session is the mutating path: dispatch adopts a session id
that is absent from the live map, and authorize restores it. A live listing cannot
make a later cron.status read-only, so the diagnostic must not send it.
"""
import pytest


def _fire(profile="profile"):
    from hermes_state_keys import CRON_PRINCIPAL_ID, cron_request_id, local_identity
    request_id = cron_request_id("job", "fire")
    return request_id, local_identity(profile, CRON_PRINCIPAL_ID, request_id)


@pytest.mark.asyncio
async def test_unmounted_inspect_does_not_restore_or_adopt(tmp_path, monkeypatch):
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionActor
    from gateway.session_controls import AuthorityConnection
    from gateway import session_cron
    from hermes_state import SessionDB
    from hermes_state_keys import CRON_PRINCIPAL_ID
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, get_session_admission

    request_id, sid = _fire()
    db = SessionDB(tmp_path / "state.db")
    authority = SessionAuthority(
        type("Runner", (), {"_draining": False})(), profile_id="profile", instance_id="test",
        db=db, epoch=begin_runtime_epoch(db, instance_id="test"),
    )
    # chat_id prefix is what authorize uses to choose the local restore. A cron fire's
    # session is local; this row is that shape and is deliberately not in the live map.
    db.create_session(sid, source="cron", chat_id="local-cron")
    assert sid not in authority.sessions
    actor = SessionActor(
        "viewer", "profile",
        frozenset({"session:create", "session:submit", "session:read", "session:control"}), "cli",
    )
    calls = []

    def restore(*_args, **_kwargs):
        calls.append("restore")
        raise RuntimeError("restore refused")

    def adopt(*_args, **_kwargs):
        calls.append("adopt")
        raise RuntimeError("adopt refused")

    monkeypatch.setattr("gateway.session_local_recovery.restore_local_session", restore)
    monkeypatch.setattr("hermes_state_local_lineage.adopt_untracked_compression", adopt)
    monkeypatch.setattr("cron.jobs.pause_job", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("paused")))
    monkeypatch.setattr(
        "cron.scheduler_authority.reconcile_pending",
        lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("reconciled")),
    )

    with db:
        row = admit_session_input(
            db, epoch=authority.epoch, principal_id=CRON_PRINCIPAL_ID, session_id=sid,
            request_id=request_id, payload={"text": ""},
        )
        before = get_session_admission(db, admission_id=row["admission_id"])
        params = {"job_id": "job", "request_id": "fire"}
        state = session_cron._inspect(authority, session_cron._actor(authority, actor), params)
        connection = AuthorityConnection(authority, object(), {
            "user_id": "viewer", "profile_id": "profile",
            "capabilities": ["session:create", "session:read", "session:submit", "session:control"],
        })
        try:
            response = await connection.dispatch({"id": 1, "method": "cron.inspect", "params": params})
            rejected = await connection.dispatch({
                "id": 2, "method": "cron.inspect",
                "params": {**params, "session_id": sid},
            })
            assert calls == []
            with pytest.raises(RuntimeError, match="restore refused"):
                await connection.dispatch({
                    "id": 3, "method": "cron.status",
                    "params": {"session_id": sid, "admission_id": row["admission_id"]},
                })
        finally:
            await connection.close()
        after = get_session_admission(db, admission_id=row["admission_id"])
    assert state["status"] == "queued" and state["result"] is None
    assert response["result"]["status"] == "queued"
    assert rejected["error"]["code"] == 4001
    assert "restore" in calls
    assert sid not in authority.sessions
    assert after == before
    assert not (tmp_path / "cron" / "jobs.json").exists()


@pytest.mark.asyncio
async def test_session_not_live_and_no_row_stays_unknown_without_restore(tmp_path, monkeypatch):
    from gateway import session_cron
    from gateway.session_authority import SessionAuthority
    from gateway.session_contract import SessionActor
    from hermes_state import SessionDB
    from hermes_state_runtime import begin_runtime_epoch

    db = SessionDB(tmp_path / "state.db")
    authority = SessionAuthority(
        type("Runner", (), {"_draining": False})(), profile_id="profile", instance_id="test",
        db=db, epoch=begin_runtime_epoch(db, instance_id="test"),
    )
    actor = SessionActor(
        "viewer", "profile",
        frozenset({"session:create", "session:submit", "session:read", "session:control"}), "cli",
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("inspect touched a session")

    monkeypatch.setattr("gateway.session_local_recovery.restore_local_session", explode)
    monkeypatch.setattr("hermes_state_local_lineage.adopt_untracked_compression", explode)
    monkeypatch.setattr("gateway.session_authority.SessionAuthority.authorize", explode)
    state = session_cron._inspect(
        authority, session_cron._actor(authority, actor), {"job_id": "job", "request_id": "fire"})
    assert state == {"status": "unknown", "result": None, "job_flags": None, "progress_at": None}
    assert authority.sessions == {}
