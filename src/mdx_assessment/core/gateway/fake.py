"""Deterministic fake provider: scripted replies per prompt ID (CI, tests, offline runs).

    fake = FakeProvider({
        "master_plan": {"steps": [...], ...},               # same reply every call
        "mda_explain": [first_reply, second_reply],          # one per call, last repeats
        "master_opinion": lambda request: {...},             # computed from the request
        "broken": '{"not": "valid"',                         # raw text: exercises bad output
    })

A reply can be a model instance, a dict, or raw text. Dicts and models go through
the output model like a real structured reply; raw text is parsed as JSON. Token
counts are derived from text length so tests see stable usage numbers.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from pydantic import BaseModel, ValidationError

from mdx_assessment.core.gateway.provider import (
    LLMOutputError,
    LLMRequest,
    LLMResponse,
    LLMUnavailableError,
)

FakeReply = BaseModel | Mapping[str, Any] | str
Script = FakeReply | Sequence[FakeReply] | Callable[[LLMRequest], FakeReply]

FAKE_MODEL = "fake-model"


class FakeProvider:
    name = "fake"
    deployment = "fake"

    def __init__(self, script: Mapping[str, Script] | None = None) -> None:
        self._script = dict(script or {})
        self._calls_per_prompt: dict[str, int] = {}
        self.requests: list[LLMRequest] = []

    def set_reply(self, prompt_id: str, reply: Script) -> None:
        self._script[prompt_id] = reply

    async def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if request.prompt_id not in self._script:
            raise LLMUnavailableError(f"fake provider has no reply for {request.prompt_id!r}")
        reply = self._next_reply(request)
        raw = self._as_text(reply)
        try:
            parsed = request.output_model.model_validate(
                reply.model_dump() if isinstance(reply, BaseModel) else self._as_data(reply)
            )
        except (ValidationError, json.JSONDecodeError) as exc:
            raise LLMOutputError(f"output did not match {request.output_model.__name__}") from exc
        return LLMResponse(
            parsed=parsed,
            refusal=None,
            raw_text=raw,
            model=FAKE_MODEL,
            prompt_tokens=max(1, (len(request.system_prompt) + len(request.user_message)) // 4),
            completion_tokens=max(1, len(raw) // 4),
        )

    async def aclose(self) -> None:
        """Nothing to release."""

    def _next_reply(self, request: LLMRequest) -> FakeReply:
        entry = self._script[request.prompt_id]
        if callable(entry) and not isinstance(entry, BaseModel):
            return entry(request)
        if isinstance(entry, Sequence) and not isinstance(entry, str):
            n = self._calls_per_prompt.get(request.prompt_id, 0)
            self._calls_per_prompt[request.prompt_id] = n + 1
            return entry[min(n, len(entry) - 1)]
        return entry

    @staticmethod
    def _as_data(reply: Mapping[str, Any] | str) -> Any:
        return json.loads(reply) if isinstance(reply, str) else dict(reply)

    @staticmethod
    def _as_text(reply: FakeReply) -> str:
        if isinstance(reply, BaseModel):
            return reply.model_dump_json()
        if isinstance(reply, str):
            return reply
        return json.dumps(dict(reply), default=str)
