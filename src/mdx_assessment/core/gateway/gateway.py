"""The governed LLM gateway: the single path from any agent to a model (PDF §2.1, §3.2).

    result = await gateway.call(
        "master_plan", "v1", inputs={...}, output_model=MasterPlan,
        actor="master_agent", session_id=session_id,
    )
    result.output        # a validated MasterPlan

What every call gets, whichever provider (Azure OpenAI, or the fake in CI):
- **Versioned prompt**: the system prompt is `config/prompts/<id>/<version>.md`; the
  version is recorded with the call, so a run says exactly which wording it used.
- **Inputs as data**: inputs travel as JSON in the user message, never pasted into
  the prompt text.
- **Structured output with allowed values**: the reply is parsed into `output_model`
  and validated again through it, so `Literal`/`Enum` fields and validators always apply.
- **Logging**: one audit event (`llm_call`) and one trace span per call, with
  provider, deployment, model, prompt version, tokens and latency.

Added in SP3: evidence citation and number matching (E5-03); untrusted-content
wrapping, one retry and the template fallback (E5-04).
"""

from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, JsonValue, ValidationError

from mdx_assessment.core.config import ConfigError, load_prompt
from mdx_assessment.core.domain import NewAuditEvent
from mdx_assessment.core.gateway.provider import (
    LLMError,
    LLMOutputError,
    LLMProvider,
    LLMRequest,
)
from mdx_assessment.core.observability.tracing import Tracer
from mdx_assessment.core.persistence.audit import AuditTrail

LLM_CALL_EVENT = "llm_call"
Outcome = Literal["ok", "output_error", "unavailable", "prompt_error"]


@dataclass
class PromptStore:
    """Prompt texts from `config/prompts/`, loaded once per (id, version)."""

    config_dir: Path
    _cache: dict[tuple[str, str], str] = field(default_factory=dict)

    def get(self, prompt_id: str, version: str) -> str:
        key = (prompt_id, version)
        if key not in self._cache:
            self._cache[key] = load_prompt(self.config_dir, prompt_id, version)
        return self._cache[key]


@dataclass(frozen=True)
class GatewayResult[M: BaseModel]:
    output: M
    prompt_id: str
    prompt_version: str
    provider: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: float


def render_user_message(inputs: dict[str, JsonValue]) -> str:
    """Inputs as indented JSON with sorted keys: readable for the model, stable for tests."""
    return "Inputs (JSON):\n" + json.dumps(inputs, indent=2, sort_keys=True, ensure_ascii=False)


class LLMGateway:
    def __init__(
        self,
        provider: LLMProvider,
        prompts: PromptStore,
        audit: AuditTrail,
        tracer: Tracer | None = None,
    ) -> None:
        self.provider = provider
        self.prompts = prompts
        self._audit = audit
        self._tracer = tracer or Tracer(None)

    async def call[M: BaseModel](
        self,
        prompt_id: str,
        version: str,
        *,
        inputs: dict[str, JsonValue],
        output_model: type[M],
        actor: str,
        session_id: str | None,
    ) -> GatewayResult[M]:
        """One governed call. Raises LLMOutputError or LLMUnavailableError (both audited)."""
        started = time.perf_counter()
        outcome: Outcome = "unavailable"
        model, tokens_in, tokens_out, error = self.provider.deployment, 0, 0, ""
        try:
            try:
                system_prompt = self.prompts.get(prompt_id, version)
            except ConfigError as exc:
                outcome = "prompt_error"
                raise LLMOutputError(str(exc)) from exc
            request = LLMRequest(
                prompt_id=prompt_id,
                prompt_version=version,
                system_prompt=system_prompt,
                user_message=render_user_message(inputs),
                output_model=output_model,
            )
            with self._tracer.generation(
                f"llm:{prompt_id}",
                model=self.provider.deployment,
                input=inputs,
                version=version,
                metadata={"prompt_id": prompt_id, "provider": self.provider.name, "actor": actor},
            ) as generation:
                response = await self.provider.complete(request)
                model = response.model
                tokens_in, tokens_out = response.prompt_tokens, response.completion_tokens
                output = self._validated(response.parsed, response.refusal, output_model)
                if generation is not None:
                    generation.update(
                        output=output.model_dump(mode="json"),
                        model=model,
                        input_tokens=tokens_in,
                        output_tokens=tokens_out,
                    )
            outcome = "ok"
            return GatewayResult(
                output=output,
                prompt_id=prompt_id,
                prompt_version=version,
                provider=self.provider.name,
                model=model,
                prompt_tokens=tokens_in,
                completion_tokens=tokens_out,
                latency_ms=_ms_since(started),
            )
        except LLMOutputError as exc:
            if outcome != "prompt_error":
                outcome = "output_error"
            error = str(exc)
            raise
        except LLMError as exc:  # LLMUnavailableError and any other provider failure
            outcome, error = "unavailable", str(exc)
            raise
        finally:
            await self._record(
                actor,
                session_id,
                prompt_id,
                version,
                outcome,
                model,
                tokens_in,
                tokens_out,
                error,
                started,
            )

    @staticmethod
    def _validated[M: BaseModel](
        parsed: BaseModel | None, refusal: str | None, output_model: type[M]
    ) -> M:
        if refusal:
            raise LLMOutputError(f"the model refused: {refusal[:200]}")
        if parsed is None:
            raise LLMOutputError("no structured output was returned")
        try:  # re-validate: allowed values and validators apply whatever the provider did
            return output_model.model_validate(parsed.model_dump())
        except ValidationError as exc:
            fields = sorted({".".join(map(str, e["loc"])) for e in exc.errors()})
            raise LLMOutputError(f"invalid {output_model.__name__}: {', '.join(fields)}") from exc

    async def _record(
        self,
        actor: str,
        session_id: str | None,
        prompt_id: str,
        version: str,
        outcome: Outcome,
        model: str,
        tokens_in: int,
        tokens_out: int,
        error: str,
        started: float,
    ) -> None:
        payload: dict[str, JsonValue] = {
            "prompt_id": prompt_id,
            "prompt_version": version,
            "provider": self.provider.name,
            "deployment": self.provider.deployment,
            "model": model,
            "outcome": outcome,
            "prompt_tokens": tokens_in,
            "completion_tokens": tokens_out,
            "latency_ms": _ms_since(started),
        }
        if error:
            payload["error"] = error[:500]
        event = NewAuditEvent(
            session_id=session_id, actor=actor, event_type=LLM_CALL_EVENT, payload=payload
        )
        await asyncio.to_thread(self._audit.append, event)


def _ms_since(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 1)
