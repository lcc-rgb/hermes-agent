"""Mounted cron.status is a ledger read. The public verb does not pause or restore that session."""
import pytest


@pytest.mark.asyncio
async def test_mounted_cron_status_does_not_restore_pause_or_rewrite(tmp_path, monkeypatch):
    from gateway.session_authority import LiveSession, SessionAuthority
    from gateway.session_contract import SessionActor
    from gateway.session_controls import AuthorityConnection
    from gateway import session_cron
    from hermes_state import SessionDB
    from hermes_state_runtime import admit_session_input, begin_runtime_epoch, get_session_admission

    db = SessionDB(tmp_path / "state.db")
    authority = SessionAuthority(
        type("Runner", (), {"_draining": False})(), profile_id="profile", instance_id="test",
        db=db, epoch=begin_runtime_epoch(db, instance_id="test"),
    )
    db.create_session("s", source="cron")
    authority.sessions["s"] = LiveSession(type("Source", (), {"platform": None, "user_id": "cron-owner"})(), "route")
    actor = SessionActor(
        "cron-owner", "profile",
        frozenset({"session:create", "session:submit", "session:read", "session:control"}), "ticker",
    )

    def explode(*_args, **_kwargs):
        raise AssertionError("cron.status mutated owner state")

    monkeypatch.setattr("gateway.session_local_recovery.restore_local_session", explode)
    monkeypatch.setattr("hermes_state_local_lineage.adopt_untracked_compression", explode)
    monkeypatch.setattr("cron.jobs.pause_job", explode)
    monkeypatch.setattr("cron.scheduler_authority.reconcile_pending", explode)

    with db:
        row = admit_session_input(
            db, epoch=authority.epoch, principal_id="cron-owner", session_id="s",
            request_id="cron:job:fire", payload={"text": ""},
        )
        before = get_session_admission(db, admission_id=row["admission_id"])
        params = {"session_id": "s", "admission_id": row["admission_id"]}
        state = await session_cron.operation(authority, "status", params, actor)
        connection = AuthorityConnection(authority, object(), {
            "user_id": "viewer", "profile_id": "profile",
            "capabilities": ["session:create", "session:read", "session:submit", "session:control"],
        })
        try:
            response = await connection.dispatch({"id": 1, "method": "cron.status", "params": params})
        finally:
            await connection.close()
        after = get_session_admission(db, admission_id=row["admission_id"])
    assert state["status"] == "queued" and state["result"] is None
    assert response["result"]["status"] == "queued"
    assert after == before
    assert not (tmp_path / "cron" / "jobs.json").exists()
