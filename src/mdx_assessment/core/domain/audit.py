"""Audit events and the SHA-256 hash chain (PDF §3.4: append-only, timestamped, integrity hashes).

Every event stores the hash of the event before it (`prev_hash`) and its own
`hash` over its content plus `prev_hash`. So:

- changing any stored field breaks that event's hash;
- recomputing that hash to cover the change breaks the next event's `prev_hash`;
- deleting an event leaves a gap in `seq` and breaks the next event's `prev_hash`.

The first event links to `GENESIS_HASH`. One limitation of any hash chain: deleting
the *last* events can't be detected from the chain alone; the session record keeps
its own event count for that cross-check (plan §2.4 `audit.events`).

These functions are storage-independent; Postgres and in-memory trails share them.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from typing import Annotated

from pydantic import AwareDatetime, Field, JsonValue, StringConstraints

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.canonical import content_hash

GENESIS_HASH = "0" * 64

HexHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
Name = Annotated[str, StringConstraints(min_length=1, max_length=100)]


class NewAuditEvent(DomainModel):
    """What a caller records. The trail assigns `seq`, `ts` and the hashes."""

    session_id: Annotated[str, StringConstraints(min_length=1, max_length=64)] | None = None
    actor: Name  # e.g. "master_agent", "input_validation", "analyst:jdoe", "system"
    event_type: Name  # e.g. "session_started", "tool_call", "llm_call", "decision"
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class AuditEvent(DomainModel):
    seq: int = Field(gt=0)
    ts: AwareDatetime
    session_id: str | None
    actor: str
    event_type: str
    payload: dict[str, JsonValue]
    prev_hash: HexHash
    hash: HexHash


class ChainVerification(DomainModel):
    valid: bool
    events_checked: int
    first_broken_seq: int | None = None
    reason: str | None = None


def canonical_ts(ts: datetime) -> str:
    """UTC, microsecond precision: the same instant always hashes the same way."""
    return ts.astimezone(UTC).isoformat(timespec="microseconds")


def compute_event_hash(
    *,
    seq: int,
    ts: datetime,
    session_id: str | None,
    actor: str,
    event_type: str,
    payload: dict[str, JsonValue],
    prev_hash: str,
) -> str:
    """SHA-256 over the canonical JSON of every stored field except `hash` itself.

    Raises ValueError if the payload holds NaN or Infinity (not representable).
    """
    return content_hash(
        {
            "seq": seq,
            "ts": canonical_ts(ts),
            "session_id": session_id,
            "actor": actor,
            "event_type": event_type,
            "payload": payload,
            "prev_hash": prev_hash,
        }
    )


def chain_event(new: NewAuditEvent, *, seq: int, ts: datetime, prev_hash: str) -> AuditEvent:
    """The next event in a chain whose last hash is `prev_hash`."""
    digest = compute_event_hash(
        seq=seq,
        ts=ts,
        session_id=new.session_id,
        actor=new.actor,
        event_type=new.event_type,
        payload=new.payload,
        prev_hash=prev_hash,
    )
    return AuditEvent(
        seq=seq,
        ts=ts,
        session_id=new.session_id,
        actor=new.actor,
        event_type=new.event_type,
        payload=new.payload,
        prev_hash=prev_hash,
        hash=digest,
    )


def _recomputed(event: AuditEvent) -> str:
    return compute_event_hash(
        seq=event.seq,
        ts=event.ts,
        session_id=event.session_id,
        actor=event.actor,
        event_type=event.event_type,
        payload=event.payload,
        prev_hash=event.prev_hash,
    )


def verify_chain(events: Iterable[AuditEvent]) -> ChainVerification:
    """Verify a whole chain: `events` must be every event, in `seq` order, from seq 1."""
    expected_seq, expected_prev, checked = 1, GENESIS_HASH, 0
    for event in events:
        checked += 1
        if event.seq != expected_seq:
            return _broken(
                checked, event.seq, f"expected seq {expected_seq}, found {event.seq} (gap)"
            )
        if event.prev_hash != expected_prev:
            return _broken(checked, event.seq, "prev_hash does not match the previous event")
        if _recomputed(event) != event.hash:
            return _broken(checked, event.seq, "hash does not match the event's content")
        expected_seq, expected_prev = event.seq + 1, event.hash
    return ChainVerification(valid=True, events_checked=checked)


def verify_linked_events(
    events: Iterable[AuditEvent], predecessor_hashes: dict[int, str]
) -> ChainVerification:
    """Verify a subset of a chain (e.g. one session's events).

    Each event's own hash is recomputed, and its `prev_hash` is compared with the
    actual event at `seq - 1` (looked up in `predecessor_hashes`, which may contain
    events from other sessions). A missing predecessor means a deleted event.
    """
    checked = 0
    for event in events:
        checked += 1
        if _recomputed(event) != event.hash:
            return _broken(checked, event.seq, "hash does not match the event's content")
        if event.seq == 1:
            expected = GENESIS_HASH
        elif (event.seq - 1) in predecessor_hashes:
            expected = predecessor_hashes[event.seq - 1]
        else:
            return _broken(checked, event.seq, f"event {event.seq - 1} is missing (gap)")
        if event.prev_hash != expected:
            return _broken(checked, event.seq, "prev_hash does not match the previous event")
    return ChainVerification(valid=True, events_checked=checked)


def _broken(checked: int, seq: int, reason: str) -> ChainVerification:
    return ChainVerification(
        valid=False, events_checked=checked, first_broken_seq=seq, reason=reason
    )
