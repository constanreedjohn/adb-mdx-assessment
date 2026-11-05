"""Evidence: a recorded fact that findings, explanations and decisions cite by ID.

The `evidence_id` is derived from the content (kind, source and data), so the
same fact gets the same ID in every run and replay (plan §3.4 reproducibility),
and an ID can't silently point at different content.
"""

from __future__ import annotations

from typing import Annotated, Self

from pydantic import Field, JsonValue, StringConstraints, model_validator

from mdx_assessment.core.domain.base import DomainModel
from mdx_assessment.core.domain.canonical import content_hash

EVIDENCE_ID_PREFIX = "ev-"
_ID_HEX_LENGTH = 16  # 64 bits: ample for one deployment's evidence, short enough to cite

EvidenceId = Annotated[str, StringConstraints(pattern=r"^ev-[0-9a-f]{16}$")]
NonEmptyStr = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]


def compute_evidence_id(kind: str, source: str, data: dict[str, JsonValue]) -> str:
    digest = content_hash({"kind": kind, "source": source, "data": data})
    return EVIDENCE_ID_PREFIX + digest[:_ID_HEX_LENGTH]


class Evidence(DomainModel):
    """A fact with a stable, content-derived ID.

    kind:   what sort of fact, e.g. "file_metadata", "point_change", "history_stats"
    source: where it came from, e.g. "mdx_data.get_file_metadata" (tool.operation)
    data:   the fact itself, JSON-only so it hashes and round-trips exactly
    """

    evidence_id: EvidenceId
    kind: NonEmptyStr
    source: NonEmptyStr
    data: dict[str, JsonValue] = Field(default_factory=dict)

    @classmethod
    def create(cls, kind: str, source: str, data: dict[str, JsonValue] | None = None) -> Self:
        payload = data or {}
        return cls(
            evidence_id=compute_evidence_id(kind, source, payload),
            kind=kind,
            source=source,
            data=payload,
        )

    @model_validator(mode="after")
    def _id_matches_content(self) -> Self:
        expected = compute_evidence_id(self.kind, self.source, self.data)
        if self.evidence_id != expected:
            raise ValueError(
                f"evidence_id {self.evidence_id} does not match its content (expected {expected})"
            )
        return self
