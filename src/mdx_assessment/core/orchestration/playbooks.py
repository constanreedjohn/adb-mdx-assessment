"""Playbook documents (`config/playbooks/*.yaml`), owned by core.

Placeholder schema for SP1: identity and status only. SP2 (E6-03) adds mandatory
steps, allowed follow-ups, limits, the default plan, `selected_when`, precedence
and `extends`, and `resolve_playbook(facts)` (plan §2.5).
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated

from pydantic import StringConstraints

from mdx_assessment.core.domain import DomainModel

PLAYBOOKS_PATTERN = "playbooks/*.yaml"

PlaybookId = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]*$")]


class PlaybookStatus(StrEnum):
    PLACEHOLDER = "placeholder"  # declared, not usable for planning yet
    ACTIVE = "active"


class PlaybookDocument(DomainModel):
    id: PlaybookId
    version: Annotated[str, StringConstraints(min_length=1)]
    title: Annotated[str, StringConstraints(min_length=1)]
    status: PlaybookStatus
    description: str = ""
