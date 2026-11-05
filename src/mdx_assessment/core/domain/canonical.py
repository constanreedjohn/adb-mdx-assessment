"""Canonical JSON and content hashing.

One serialization for everything that is hashed (evidence IDs, the audit hash
chain, the config version), so the same content always gives the same hash:
keys sorted, no insignificant whitespace, UTF-8, NaN/Infinity rejected.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import JsonValue


def canonical_json(value: JsonValue) -> str:
    """Deterministic JSON text for `value`. Raises ValueError on NaN or Infinity."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_hash(value: JsonValue) -> str:
    """SHA-256 (hex) of the canonical JSON of `value`."""
    return sha256_hex(canonical_json(value))
