"""Shared base for domain models."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class DomainModel(BaseModel):
    """Immutable, strict-shaped model: unknown fields are errors, instances never change.

    Immutability keeps evidence, findings and decisions trustworthy once recorded;
    "changing" one means creating a new instance (`model_copy(update=...)`).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")
