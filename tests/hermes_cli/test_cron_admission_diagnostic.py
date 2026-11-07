"""The cron admission diagnostic reads cron.status and leaves the fire untouched."""
import json
import os
from argparse import Namespace
from contextlib import asynccontextmanager

import pytest

from hermes_cli.cron_admission_diagnostic import (
    AdmissionDiagnostic,
    OwnerRefused,
    OwnerUnreachable,
    diagnose,
    interpret_status,
    load_journal,
    query_cron_status,
    render_json,
    render_text,
)
from hermes_cli.gateway_runtime import GatewayDiscovery, GatewayEndpoint

_SECRET = "SECRET_PROMPT sk-test do-not-leak"
_FORBIDDEN = (
    "process is absent",
    "process absent",
    "no local fallback",
    "did not run",
    "not running",
    "never admitted",
    "gateway is down",
    _SECRET,
)


def _endpoint():
    return GatewayEndpoint(
        profile_id="profile", instance_id="instance", authority_epoch=1, runtime_protocol=1,
        api_origin="http://127.0.0.1:9", supervisor="none", capabilities=frozenset(),
    )


def _ready():
    return GatewayDiscovery("ready", endpoint=_endpoint())


def _absent():
    return GatewayDiscovery("absent")


def _write_journal(job_id, request_id, record):
    from cron.scheduler_authority import journal_path

    path = journal_path(job_id, request_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(json.dumps(record).encode())
    return path


def _snapshot(root):
    found = {}
    if not root.exists():
        return found
    for path in root.rglob("*"):
        if path.is_symlink() or not path.is_file():
            continue
        found[str(path.relative_to(root))] = path.read_bytes()
    return found


def _assert_clean(text):
    for needle in _FORBIDDEN:
        assert needle not in text


def test_terminal_report_drops_output_text():
    from hermes_cli.cron_admission_diagnostic import _UNREACHABLE_FLAG

    payload = {
        "status": "terminal",
        "result": [True, _SECRET, "answer " + _SECRET, "err " + _SECRET],
        "job": {"prompt": _SECRET},
        "job_flags": {_UNREACHABLE_FLAG: True, "api_key": _SECRET, "_quota_hold_seconds": 30},
        "progress_at": 10.0,
        "extra_prompt": _SECRET,
    }
    report = interpret_status(payload)
    assert report.verdict == "authoritative" and report.status == "terminal" and report.terminal_success is True
    assert report.progress_observed is True
    assert report.job_flags == ((_UNREACHABLE_FLAG, True), ("_quota_hold_seconds", 30))
    rendered = render_text(report, "job", "fire") + render_json(report)
    _assert_clean(rendered)
    assert "extra_prompt" not in rendered and "api_key" not in rendered


def test_owner_unknown_stays_unknown():
    report = interpret_status({"status": "unknown", "result": [True, _SECRET, "", ""], "progress_at": 5})
    assert report.verdict == "unknown" and report.evidence == "owner_status_unknown"
    assert report.terminal_success is None and report.progress_observed is False
    assert _SECRET not in render_json(report)


def test_malformed_terminal_result_stays_unknown():
    report = interpret_status({"status": "terminal", "result": [1, _SECRET, "", ""]})
    assert report.verdict == "unknown" and report.evidence == "malformed_status"
    assert _SECRET not in render_json(report)


def test_started_is_authoritative_without_touching_the_home(monkeypatch):
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    record = {
        "params": {"job_id": "job", "request_id": "fire", "extra_prompt": _SECRET},
        "receipt": {"session_id": "local-" + "a" * 64, "admission_id": "b" * 32, "extra_prompt": _SECRET},
    }
    path = _write_journal("job", "fire", record)
    before = path.read_bytes()
    home_before = _snapshot(home)
    calls = []

    def discover():
        calls.append("discover")
        return _ready()

    def call_status(endpoint, session_id, admission_id):
        calls.append((endpoint.api_origin, session_id, admission_id))
        return {"status": "started", "result": None, "progress_at": 4, "job": {"prompt": _SECRET}}

    def explode(*_args, **_kwargs):
        raise AssertionError("diagnostic mutated cron state")

    monkeypatch.setattr("cron.jobs.pause_job", explode)
    monkeypatch.setattr("cron.jobs.resume_job", explode)
    monkeypatch.setattr("cron.scheduler_authority.pause_for_unknown", explode)
    monkeypatch.setattr("cron.scheduler_authority.reconcile_pending", explode)
    monkeypatch.setattr("hermes_cli.gateway_runtime.ensure_gateway_runtime", explode)
    report = diagnose("job", "fire", discover=discover, call_status=call_status, attachable_fn=lambda item: True)
    assert report.verdict == "authoritative" and report.status == "started" and report.queried is True
    assert report.exit_code() == 0
    assert calls[0] == "discover"
    assert calls[1][1:] == ("local-" + "a" * 64, "b" * 32)
    assert path.read_bytes() == before
    assert _snapshot(home) == home_before
    text = render_text(report, "job", "fire")
    _assert_clean(text)
    assert "cron.recover, and cron.cancel were not called." in text


def test_refusal_is_not_a_missing_admission():
    def call_status(*_args):
        raise OwnerRefused("permission_denied")

    report = diagnose(
        "job", "fire",
        load=lambda *_args: {"params": {"job_id": "job", "request_id": "fire"},
                             "receipt": {"session_id": "local-a", "admission_id": "admit"}},
        discover=_ready, call_status=call_status, attachable_fn=lambda item: True,
    )
    assert report.verdict == "unknown" and report.evidence == "owner_refused"
    assert report.owner_reason == "permission_denied" and report.status is None
    text = render_text(report, "job", "fire")
    assert "unresolved" in text
    _assert_clean(text)


def test_no_receipt_does_not_query_or_recover():
    def discover():
        raise AssertionError("discover")

    def call_status(*_args):
        raise AssertionError("rpc")

    report = diagnose(
        "job", "fire",
        load=lambda *_args: {"params": {"job_id": "job", "request_id": "fire", "extra_prompt": _SECRET},
                             "receipt": None},
        discover=discover, call_status=call_status, attachable_fn=lambda item: True,
    )
    assert report == AdmissionDiagnostic(verdict="unknown", evidence="no_receipt")
    text = render_text(report, "job", "fire")
    assert "not diagnosable on the already-running owner" in text
    assert "sent no cron verb" in text
    _assert_clean(text)


def test_missing_journal_does_not_claim_the_fire_is_absent():
    report = diagnose("job", "fire", load=lambda *_args: None, discover=lambda: (_ for _ in ()).throw(AssertionError("discover")),
                      call_status=lambda *_args: (_ for _ in ()).throw(AssertionError("rpc")))
    assert report.evidence == "no_journal" and report.queried is False
    _assert_clean(render_text(report, "job", "fire"))


def test_discovery_absent_does_not_connect(monkeypatch):
    def call_status(*_args):
        raise AssertionError("connected")

    monkeypatch.setattr("hermes_cli.gateway_runtime.ensure_gateway_runtime", lambda *_a, **_k: (_ for _ in ()).throw(AssertionError("started")))
    report = diagnose(
        "job", "fire",
        load=lambda *_args: {"params": {"job_id": "job", "request_id": "fire"},
                             "receipt": {"session_id": "local-a", "admission_id": "admit"}},
        discover=_absent, call_status=call_status,
    )
    assert report.verdict == "unknown" and report.evidence == "owner_not_attachable"
    assert report.discovery_state == "absent" and report.queried is False
    text = render_text(report, "job", "fire")
    assert "discovery_state=absent" in text
    assert "attach result only" in text
    _assert_clean(text)


def test_connection_failure_message_is_not_repeated():
    def call_status(*_args):
        raise OwnerUnreachable

    report = diagnose(
        "job", "fire",
        load=lambda *_args: {"params": {"job_id": "job", "request_id": "fire"},
                             "receipt": {"session_id": "local-a", "admission_id": "admit"}},
        discover=_ready, call_status=call_status, attachable_fn=lambda item: True,
    )
    assert report.evidence == "owner_unreachable" and report.queried is True
    text = render_text(report, "job", "fire") + render_json(report)
    assert "Gateway connection failed" not in text
    _assert_clean(text)


def test_identity_conflict_skips_the_owner():
    report = diagnose(
        "job", "fire",
        load=lambda *_args: {"params": {"job_id": "other", "request_id": "fire", "extra_prompt": _SECRET},
                             "receipt": {"session_id": "local-a", "admission_id": "admit"}},
        discover=lambda: (_ for _ in ()).throw(AssertionError("discover")),
        call_status=lambda *_args: (_ for _ in ()).throw(AssertionError("rpc")),
    )
    assert report.evidence == "identity_conflict"
    _assert_clean(render_text(report, "job", "fire"))


def test_rejected_ids_are_not_echoed_or_read(monkeypatch):
    monkeypatch.setattr("hermes_cli.cron_admission_diagnostic.load_journal", lambda *_args: (_ for _ in ()).throw(AssertionError("read")))
    report = diagnose("../" + _SECRET, "fire")
    assert report.evidence == "invalid_identity"
    text = render_text(report, "../" + _SECRET, "fire")
    assert "<rejected>" in text
    _assert_clean(text)


def test_symlink_journal_fails_closed(tmp_path, monkeypatch):
    from cron.scheduler_authority import journal_path
    from hermes_constants import get_hermes_home

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    target = tmp_path / "elsewhere.json"
    target.write_text(json.dumps({"params": {"job_id": "job", "request_id": "fire"}, "extra_prompt": _SECRET}), encoding="utf-8")
    path = journal_path("job", "fire")
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    from hermes_cli.cron_admission_diagnostic import JournalUnreadable

    with pytest.raises(JournalUnreadable):
        load_journal("job", "fire")
    report = diagnose("job", "fire")
    assert report.evidence == "journal_unreadable"
    assert _SECRET not in render_text(report, "job", "fire")
    assert target.read_text(encoding="utf-8").count(_SECRET) == 1


def test_query_sends_only_cron_status_to_the_discovered_endpoint(monkeypatch):
    from hermes_cli import gateway_client

    seen = {}

    class Client:
        async def rpc(self, method, **params):
            seen["method"] = method
            seen["params"] = params
            return {"status": "queued", "result": [_SECRET]}

    @asynccontextmanager
    async def connected(endpoint=None, subject=None, *, max_size=8 * 1024 * 1024):
        seen["endpoint"] = endpoint
        seen["calls"] = seen.get("calls", 0) + 1
        yield Client()

    monkeypatch.setattr(gateway_client, "connect_gateway", connected)
    monkeypatch.setattr(
        "hermes_cli.gateway_runtime.ensure_gateway_runtime",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("started a gateway")),
    )
    endpoint = _endpoint()
    assert query_cron_status(endpoint, "local-a", "admit") == {"status": "queued", "result": [_SECRET]}
    assert seen["endpoint"] is endpoint
    assert seen["method"] == "cron.status"
    assert set(seen["params"]) == {"session_id", "admission_id"}
    with pytest.raises(OwnerUnreachable):
        query_cron_status(None, "local-a", "admit")
    assert seen["calls"] == 1


def test_query_maps_the_connection_failure_without_its_text(monkeypatch):
    from hermes_cli import gateway_client

    class Client:
        async def rpc(self, method, **params):
            raise gateway_client.GatewayClientError("Gateway connection failed; no local fallback")

    @asynccontextmanager
    async def connected(endpoint=None, subject=None, *, max_size=8 * 1024 * 1024):
        yield Client()

    monkeypatch.setattr(gateway_client, "connect_gateway", connected)
    with pytest.raises(OwnerUnreachable) as caught:
        query_cron_status(_endpoint(), "local-a", "admit")
    assert "no local fallback" not in str(caught.value)


def test_cli_json_uses_the_diagnostic_exit_code(capsys, monkeypatch):
    from hermes_cli.cron import cron_command

    report = AdmissionDiagnostic(verdict="unknown", evidence="no_journal")
    monkeypatch.setattr("hermes_cli.cron_admission_diagnostic.diagnose", lambda job_id, request_id: report)
    code = cron_command(Namespace(cron_command="admission-status", job_id="job", request_id="fire", json=True))
    assert code == 2
    body = json.loads(capsys.readouterr().out)
    assert body["verdict"] == "unknown" and body["evidence"] == "no_journal" and body["queried"] is False


def test_parser_accepts_admission_status():
    import argparse

    from hermes_cli.subcommands.cron import build_cron_parser

    parser = argparse.ArgumentParser()
    build_cron_parser(parser.add_subparsers(dest="command"), cmd_cron=lambda args: 0)
    args = parser.parse_args(["cron", "admission-status", "job", "fire", "--json"])
    assert args.cron_command == "admission-status" and args.job_id == "job" and args.json is True


def test_load_journal_round_trip_is_read_only():
    path = _write_journal("job", "fire", {"params": {"job_id": "job", "request_id": "fire"}, "receipt": None})
    before = path.read_bytes()
    assert load_journal("job", "fire")["receipt"] is None
    assert path.read_bytes() == before
    assert load_journal("job", "other") is None
