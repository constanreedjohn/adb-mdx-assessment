"""Dataset completeness: did every expected file arrive with the expected record count?

One pure function, used by the run-context facts (E6-03) and the Input Validation Agent
(E7-01), so the two can never disagree. No I/O: callers pass what the MDX tool returned.

Per expected file (from `config/expected_files.yaml`):

| Status    | When                                                    |
| --------- | ------------------------------------------------------- |
| `ok`      | Received with exactly the expected record count         |
| `short`   | Received with fewer records than expected               |
| `over`    | Received with more records than expected                |
| `missing` | Not received                                            |

The dataset is complete only if every expected file is `ok`. Files MDX received that
the manifest doesn't expect are listed as `unexpected` but don't make the dataset
incomplete: nothing downstream reads them.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import Protocol

from pydantic import Field

from mdx_assessment.core.domain import DomainModel


class FileStatus(StrEnum):
    OK = "ok"
    SHORT = "short"
    OVER = "over"
    MISSING = "missing"


class ExpectedFileLike(Protocol):
    @property
    def file_name(self) -> str: ...

    @property
    def expected_record_count(self) -> int: ...


class ReceivedFileLike(Protocol):
    @property
    def file_name(self) -> str: ...

    @property
    def record_count(self) -> int: ...


class FileCheck(DomainModel):
    file_name: str = Field(min_length=1)
    status: FileStatus
    expected_records: int = Field(ge=0)
    received_records: int | None = Field(default=None, ge=0)
    """None when the file wasn't received."""


class Completeness(DomainModel):
    files: tuple[FileCheck, ...]
    unexpected_files: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return all(f.status is FileStatus.OK for f in self.files)

    @property
    def files_expected(self) -> int:
        return len(self.files)

    @property
    def files_received(self) -> int:
        return sum(f.status is not FileStatus.MISSING for f in self.files)

    @property
    def records_expected(self) -> int:
        return sum(f.expected_records for f in self.files)

    @property
    def records_received(self) -> int:
        """Records in the expected files that arrived."""
        return sum(f.received_records or 0 for f in self.files)

    @property
    def problems(self) -> tuple[FileCheck, ...]:
        return tuple(f for f in self.files if f.status is not FileStatus.OK)


def _status(expected: int, received: int | None) -> FileStatus:
    if received is None:
        return FileStatus.MISSING
    if received < expected:
        return FileStatus.SHORT
    if received > expected:
        return FileStatus.OVER
    return FileStatus.OK


def assess_completeness(
    expected: Sequence[ExpectedFileLike], received: Sequence[ReceivedFileLike]
) -> Completeness:
    by_name = {f.file_name: f for f in received}
    checks = []
    for e in expected:
        got = by_name.get(e.file_name)
        count = got.record_count if got is not None else None
        checks.append(
            FileCheck(
                file_name=e.file_name,
                status=_status(e.expected_record_count, count),
                expected_records=e.expected_record_count,
                received_records=count,
            )
        )
    wanted = {e.file_name for e in expected}
    return Completeness(
        files=tuple(checks),
        unexpected_files=tuple(sorted(n for n in by_name if n not in wanted)),
    )
