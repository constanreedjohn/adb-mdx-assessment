"""Hash-chain rules, exercised through the in-memory trail (no database)."""

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta, timezone

import pytest

from mdx_assessment.core.domain import GENESIS_HASH, AuditEvent, NewAuditEvent, verify_chain
from mdx_assessment.core.domain.audit import compute_event_hash, verify_linked_events
from mdx_assessment.core.persistence import AuditTrail, InMemoryAuditTrail

T0 = datetime(2026, 10, 15, 9, 20, tzinfo=UTC)


def fixed_clock() -> datetime:
    return T0


def ev(
    session: str | None = "sess-1", event_type: str = "tool_call", **payload: object
) -> NewAuditEvent:
    return NewAuditEvent.model_validate(
        {
            "session_id": session,
            "actor": "input_validation",
            "event_type": event_type,
            "payload": payload,
        }
    )


def filled_trail(n: int = 5) -> InMemoryAuditTrail:
    trail = InMemoryAuditTrail(clock=fixed_clock)
    for i in range(n):
        trail.append(ev(step=i))
    return trail


def test_in_memory_trail_satisfies_the_protocol() -> None:
    assert isinstance(InMemoryAuditTrail(), AuditTrail)


def test_chain_links_from_genesis() -> None:
    events = filled_trail(3).read()
    assert [e.seq for e in events] == [1, 2, 3]
    assert events[0].prev_hash == GENESIS_HASH
    assert events[1].prev_hash == events[0].hash
    assert events[2].prev_hash == events[1].hash
    assert len({e.hash for e in events}) == 3


def test_valid_chain_verifies() -> None:
    result = filled_trail(5).verify()
    assert result.valid
    assert result.events_checked == 5
    assert result.first_broken_seq is None


def test_empty_chain_is_valid() -> None:
    assert InMemoryAuditTrail().verify().valid


def tamper(events: list[AuditEvent], index: int, **changes: object) -> list[AuditEvent]:
    changed = list(events)
    changed[index] = events[index].model_copy(update=changes)
    return changed


def test_tampered_payload_is_detected() -> None:
    events = tamper(filled_trail(5).read(), 2, payload={"step": 99})
    result = verify_chain(events)
    assert (result.valid, result.first_broken_seq) == (False, 3)
    assert result.reason == "hash does not match the event's content"


@pytest.mark.parametrize("field", ["actor", "event_type", "session_id"])
def test_any_tampered_field_is_detected(field: str) -> None:
    events = tamper(filled_trail(3).read(), 1, **{field: "forged"})
    assert verify_chain(events).first_broken_seq == 2


def test_tampered_timestamp_is_detected() -> None:
    events = filled_trail(3).read()
    events = tamper(events, 1, ts=events[1].ts + timedelta(seconds=1))
    assert verify_chain(events).first_broken_seq == 2


def test_tampering_with_a_recomputed_hash_breaks_the_next_link() -> None:
    events = filled_trail(5).read()
    forged = events[2].model_copy(update={"payload": {"step": 99}})
    forged = forged.model_copy(
        update={
            "hash": compute_event_hash(
                seq=forged.seq,
                ts=forged.ts,
                session_id=forged.session_id,
                actor=forged.actor,
                event_type=forged.event_type,
                payload=forged.payload,
                prev_hash=forged.prev_hash,
            )
        }
    )
    events[2] = forged
    result = verify_chain(events)
    assert (result.valid, result.first_broken_seq) == (False, 4)
    assert result.reason == "prev_hash does not match the previous event"


def test_deleted_event_is_detected() -> None:
    events = filled_trail(5).read()
    del events[2]
    result = verify_chain(events)
    assert (result.valid, result.first_broken_seq) == (False, 4)
    assert "gap" in (result.reason or "")


def test_deleted_first_event_is_detected() -> None:
    result = verify_chain(filled_trail(3).read()[1:])
    assert (result.valid, result.first_broken_seq) == (False, 2)


def test_reordered_events_are_detected() -> None:
    events = filled_trail(4).read()
    events[1], events[2] = events[2], events[1]
    assert not verify_chain(events).valid


def test_same_instant_in_another_timezone_hashes_the_same() -> None:
    manila = timezone(timedelta(hours=8))
    common = {
        "seq": 1,
        "session_id": None,
        "actor": "a",
        "event_type": "e",
        "payload": {},
        "prev_hash": GENESIS_HASH,
    }
    assert compute_event_hash(ts=T0, **common) == compute_event_hash(  # type: ignore[arg-type]
        ts=T0.astimezone(manila),
        **common,  # type: ignore[arg-type]
    )


def test_nan_payload_is_rejected_before_anything_is_stored() -> None:
    trail = InMemoryAuditTrail()
    with pytest.raises(ValueError, match="Out of range float"):
        trail.append(ev(value=float("nan")))
    assert trail.read() == []


# --- per-session verification ----------------------------------------------------------


def interleaved() -> InMemoryAuditTrail:
    trail = InMemoryAuditTrail(clock=fixed_clock)
    for i in range(6):
        trail.append(ev(session="sess-a" if i % 2 == 0 else "sess-b", step=i))
    return trail


def test_session_verification_follows_links_through_other_sessions() -> None:
    trail = interleaved()
    assert [e.seq for e in trail.read("sess-b")] == [2, 4, 6]
    assert trail.verify("sess-a").valid
    assert trail.verify("sess-b").events_checked == 3


def test_session_verification_detects_a_deleted_neighbour() -> None:
    events = interleaved().read()
    by_seq = {e.seq: e.hash for e in events if e.seq != 3}  # event 3 (sess-a) deleted
    session_b = [e for e in events if e.session_id == "sess-b"]
    result = verify_linked_events(session_b, by_seq)
    assert (result.valid, result.first_broken_seq) == (False, 4)


# --- concurrency -----------------------------------------------------------------------


def test_concurrent_appends_keep_one_valid_chain() -> None:
    trail = InMemoryAuditTrail()
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: trail.append(ev(step=i)), range(200)))
    events = trail.read()
    assert [e.seq for e in events] == list(range(1, 201))
    assert trail.verify().valid
