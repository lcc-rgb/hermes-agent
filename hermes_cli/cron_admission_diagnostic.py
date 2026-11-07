"""Read-only diagnostic of one cron admission through the public ``cron.status`` RPC.

An immediate fire that was admitted and then lost its client (``Gateway connection failed;
no local fallback``, history UNKNOWN, protective pause) still has one authoritative
source: the owner's admission row. This command reports that row. It does not submit,
recover, cancel, reconcile, deliver, pause, or write the journal.

Live owner, no restart
----------------------
``cron.status`` is already on the public ``hermes-gateway-v1`` catalog
(``session_id`` + ``admission_id``). A client dials an owner
``discover_gateway_endpoint`` already found and calls only that verb. Nothing here
starts a gateway or restarts the owner. When the fire's session is still mounted,
that verb is a ledger read (``authorize`` restores only a session missing from the
live map; ``dispatch`` runs a compression catch-up only in that same case).

A journal with no receipt cannot be addressed by ``cron.status``. ``cron.recover``
can look the fire up by job and request, and it always calls ``restore_local_session``.
That is a persistence change, so this command does not call it. The verdict stays
UNKNOWN. A side-effect-free lookup by ``(job_id, request_id)`` would be a new
canonical read that omits ``session_id`` (so dispatch does not adopt the session)
and does not call ``authorize``. It would return the same status projection as
``cron.status``, and UNKNOWN when the row is absent or its outcome is unknown —
never ``missing``, which is recover's definite non-admission. That verb is not
added here: the process already running would not serve it until it restarts, and
an ``unknown_method`` answer must not be reported as a resolution of the live fire.

UNKNOWN stays unresolved. A discovery state, a refused RPC, a timeout, or an empty
journal is an attach or client fact. None of those is evidence that a process is
gone or that the fire did not run.
"""
from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass

_TOKEN = re.compile(r"^[A-Za-z0-9_-]{1,256}$")
_CODE = re.compile(r"^[a-z0-9_]{1,80}$")
_JOURNAL_CAP = 1_000_000
_KNOWN_STATUSES = frozenset({"queued", "started", "terminal", "unknown"})
_QUOTA_FLAG = "_quota_hold_seconds"
_UNREACHABLE_FLAG = "_model_unreachable"


class JournalUnreadable(Exception):
    """The local journal could not be read. Says nothing about the owner."""


class OwnerUnreachable(Exception):
    """``cron.status`` was not answered. Says nothing about process liveness."""


class OwnerRefused(Exception):
    """The owner returned a bounded refusal. Not a missing admission."""

    def __init__(self, reason=None):
        self.reason = reason if isinstance(reason, str) and _CODE.fullmatch(reason) else None


@dataclass(frozen=True)
class AdmissionDiagnostic:
    """Sanitized report. Fields are codes and booleans; no prompt, output, or path."""

    verdict: str
    evidence: str
    queried: bool = False
    status: str | None = None
    terminal_success: bool | None = None
    progress_observed: bool = False
    job_flags: tuple = ()
    discovery_state: str | None = None
    discovery_reason: str | None = None
    owner_reason: str | None = None

    def exit_code(self) -> int:
        return 0 if self.verdict == "authoritative" else 2


def _unknown(evidence, **fields) -> AdmissionDiagnostic:
    return AdmissionDiagnostic(verdict="unknown", evidence=evidence, **fields)


def _code(value):
    return value if isinstance(value, str) and _CODE.fullmatch(value) else None


def _safe_id(value) -> bool:
    return isinstance(value, str) and _TOKEN.fullmatch(value) is not None


def _finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _flags(raw):
    if not isinstance(raw, dict):
        return ()
    found = []
    unreachable = raw.get(_UNREACHABLE_FLAG)
    if isinstance(unreachable, bool):
        found.append((_UNREACHABLE_FLAG, unreachable))
    hold = raw.get(_QUOTA_FLAG)
    if _finite_number(hold) and 0 <= hold <= 7 * 86400:
        found.append((_QUOTA_FLAG, hold))
    return tuple(found)


def _progress(raw) -> bool:
    return isinstance(raw, dict) and _finite_number(raw.get("progress_at")) and raw["progress_at"] > 0


def _terminal_success(result):
    if isinstance(result, (str, bytes)) or not isinstance(result, (list, tuple)) or len(result) != 4:
        return None
    return result[0] if isinstance(result[0], bool) else None


def interpret_status(payload) -> AdmissionDiagnostic:
    """Project one ``cron.status`` result. Drops output text, prompts, and unknown flags."""
    if not isinstance(payload, dict):
        return _unknown("malformed_status", queried=True)
    status = payload.get("status")
    if status not in _KNOWN_STATUSES:
        return _unknown("malformed_status", queried=True)
    if status == "unknown":
        return _unknown("owner_status_unknown", queried=True, status="unknown")
    flags = _flags(payload.get("job_flags"))
    if status == "terminal":
        success = _terminal_success(payload.get("result"))
        if success is None:
            return _unknown("malformed_status", queried=True)
        return AdmissionDiagnostic(
            verdict="authoritative", evidence="cron_status", queried=True, status="terminal",
            terminal_success=success, progress_observed=_progress(payload), job_flags=flags,
        )
    return AdmissionDiagnostic(
        verdict="authoritative", evidence="cron_status", queried=True, status=status,
        progress_observed=_progress(payload), job_flags=flags,
    )


def load_journal(job_id, request_id):
    """Read one admission journal. ``None`` when the file is absent. Never creates or rewrites it."""
    from cron.scheduler_authority import journal_path

    path = journal_path(job_id, request_id)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise JournalUnreadable from exc
    try:
        try:
            data = os.read(fd, _JOURNAL_CAP + 1)
        except OSError as exc:
            raise JournalUnreadable from exc
    finally:
        os.close(fd)
    if len(data) > _JOURNAL_CAP:
        raise JournalUnreadable
    try:
        record = json.loads(data.decode("utf-8-sig"))
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise JournalUnreadable from exc
    if not isinstance(record, dict):
        raise JournalUnreadable
    return record


def _receipt(record, job_id, request_id):
    if not isinstance(record, dict):
        return "journal_unreadable", None
    params = record.get("params")
    if not isinstance(params, dict) or params.get("job_id") != job_id or params.get("request_id") != request_id:
        return "identity_conflict", None
    receipt = record.get("receipt")
    if not isinstance(receipt, dict):
        return "no_receipt", None
    session_id, admission_id = receipt.get("session_id"), receipt.get("admission_id")
    if not _safe_id(session_id) or not _safe_id(admission_id):
        return "no_receipt", None
    return None, (session_id, admission_id)


def _attach_fields(discovery):
    return {
        "discovery_state": _code(getattr(discovery, "state", None)),
        "discovery_reason": _code(getattr(discovery, "reason_code", None)),
    }


def diagnose_record(record, job_id, request_id, *, discover, call_status, attachable_fn) -> AdmissionDiagnostic:
    """Decide from an already-loaded journal. ``discover`` and ``call_status`` perform the only I/O."""
    problem, receipt = _receipt(record, job_id, request_id)
    if problem:
        return _unknown(problem)
    discovery = discover()
    fields = _attach_fields(discovery)
    if not attachable_fn(discovery) or getattr(discovery, "endpoint", None) is None:
        return _unknown("owner_not_attachable", **fields)
    session_id, admission_id = receipt
    try:
        payload = call_status(discovery.endpoint, session_id, admission_id)
    except OwnerRefused as exc:
        return _unknown("owner_refused", queried=True, owner_reason=exc.reason, **fields)
    except OwnerUnreachable:
        return _unknown("owner_unreachable", queried=True, **fields)
    report = interpret_status(payload)
    return AdmissionDiagnostic(
        verdict=report.verdict, evidence=report.evidence, queried=True, status=report.status,
        terminal_success=report.terminal_success, progress_observed=report.progress_observed,
        job_flags=report.job_flags, owner_reason=report.owner_reason, **fields,
    )


def _discover():
    from hermes_constants import get_hermes_home
    from hermes_cli.gateway_runtime import discover_gateway_endpoint

    return discover_gateway_endpoint(get_hermes_home())


def query_cron_status(endpoint, session_id, admission_id):
    """One ``cron.status`` against an already-discovered owner. Never starts a gateway."""
    if endpoint is None or not _safe_id(session_id) or not _safe_id(admission_id):
        raise OwnerUnreachable
    import asyncio

    from hermes_cli.gateway_client import GatewayClientError, GatewayRPCError, connect_gateway

    async def ask():
        async with connect_gateway(endpoint) as client:
            return await client.rpc("cron.status", session_id=session_id, admission_id=admission_id)

    try:
        return asyncio.run(ask())
    except GatewayRPCError as exc:
        raise OwnerRefused(str(exc)) from None
    except (GatewayClientError, TimeoutError, OSError):
        raise OwnerUnreachable from None


def diagnose(job_id, request_id, *, load=load_journal, discover=None, call_status=None,
             attachable_fn=None) -> AdmissionDiagnostic:
    """Diagnose one fire. Default I/O is the real journal, discovery, and ``cron.status``."""
    if not _safe_id(job_id) or not _safe_id(request_id):
        return _unknown("invalid_identity")
    if discover is None:
        discover = _discover
    if call_status is None:
        call_status = query_cron_status
    if attachable_fn is None:
        from hermes_cli.gateway_runtime import attachable as attachable_fn
    try:
        record = load(job_id, request_id)
    except JournalUnreadable:
        return _unknown("journal_unreadable")
    if record is None:
        return _unknown("no_journal")
    return diagnose_record(
        record, job_id, request_id, discover=discover, call_status=call_status, attachable_fn=attachable_fn,
    )


_EVIDENCE_TEXT = {
    "invalid_identity": "The job id or request id is not a single token of letters, digits, underscore, or hyphen. Nothing was read.",
    "journal_unreadable": "The local admission journal could not be read. Execution remains UNKNOWN.",
    "no_journal": "No local admission journal exists for this job and request. Execution remains UNKNOWN.",
    "identity_conflict": "The local journal identity does not match this job and request. Nothing was queried. Execution remains UNKNOWN.",
    "no_receipt": (
        "The local journal has no admission receipt, so cron.status cannot be addressed. "
        "cron.recover was not called, because that verb restores the session. "
        "Execution remains UNKNOWN. This fire is not diagnosable on the already-running owner without that restore."
    ),
    "owner_not_attachable": "Discovery did not yield an attachable owner. Execution remains UNKNOWN. The discovery state is an attach result only.",
    "owner_unreachable": "The owner did not return cron.status. Execution remains UNKNOWN. A failed attach is an attach result only.",
    "owner_refused": "The owner refused cron.status. Execution remains UNKNOWN. A refusal leaves the admission unresolved.",
    "owner_status_unknown": "The owner reports this admission as unknown. Execution remains UNKNOWN.",
    "malformed_status": "The owner answer failed validation. Execution remains UNKNOWN.",
    "cron_status": "The owner answered cron.status for the stored receipt.",
}


def _shown(value) -> str:
    return value if _safe_id(value) else "<rejected>"


def _scalar(value) -> str:
    if value is True:
        return "true"
    if value is False:
        return "false"
    if isinstance(value, int):
        return str(value)
    return format(value, "g")


def render_text(report: AdmissionDiagnostic, job_id, request_id) -> str:
    """Human report. Interpolates only safe ids and the report's own codes."""
    lines = [
        "Cron admission diagnostic",
        f"job_id={_shown(job_id)}",
        f"request_id={_shown(request_id)}",
        f"verdict={report.verdict}",
        f"evidence={report.evidence}",
        f"queried={'true' if report.queried else 'false'}",
    ]
    if report.status:
        lines.append(f"status={report.status}")
    if report.terminal_success is not None:
        lines.append("terminal_success=" + ("true" if report.terminal_success else "false"))
    if report.status in {"queued", "started"}:
        lines.append("The admission is still open on the owner.")
    if report.verdict == "authoritative":
        lines.append("progress_observed=" + ("true" if report.progress_observed else "false"))
    for key, value in report.job_flags:
        lines.append(f"{key}={_scalar(value)}")
    if report.discovery_state:
        lines.append(f"discovery_state={report.discovery_state}")
    if report.discovery_reason:
        lines.append(f"discovery_reason={report.discovery_reason}")
    if report.owner_reason:
        lines.append(f"owner_reason={report.owner_reason}")
    lines.append(_EVIDENCE_TEXT[report.evidence])
    if report.queried:
        lines.append("This command sent cron.status.")
    else:
        lines.append("This command sent no cron verb.")
    lines.append("cron.submit, cron.recover, and cron.cancel were not called.")
    lines.append("This command did not write the journal, change the pause, reconcile, or deliver.")
    lines.append(
        "On the running owner, cron.status restores a local session that is not in its live map. "
        "A mounted admission is a ledger read. Removing that restore needs an owner change and a restart, "
        "which this command does not perform."
    )
    if report.verdict != "authoritative":
        lines.append("UNKNOWN stays unresolved. Do not start another run from this report.")
    return "\n".join(lines)


def render_json(report: AdmissionDiagnostic) -> str:
    body = {
        "verdict": report.verdict,
        "evidence": report.evidence,
        "queried": report.queried,
        "status": report.status,
        "terminal_success": report.terminal_success,
        "progress_observed": report.progress_observed,
        "job_flags": {key: value for key, value in report.job_flags},
        "discovery_state": report.discovery_state,
        "discovery_reason": report.discovery_reason,
        "owner_reason": report.owner_reason,
    }
    return json.dumps(body, sort_keys=True, allow_nan=False)
