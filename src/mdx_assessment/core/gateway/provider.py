"""LLM provider interface: one structured-output completion, nothing else.

Providers know how to talk to a model. Everything that makes a call *governed*
(prompts, validation, logging, guardrails, retry, fallback) lives in `LLMGateway`,
so it applies equally to Azure OpenAI and to the fake used in CI.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from pydantic import BaseModel, ValidationError


class LLMError(RuntimeError):
    """Base class for gateway errors."""


class LLMUnavailableError(LLMError):
    """The model couldn't be reached or refused the request (network, auth, quota)."""


class LLMOutputError(LLMError):
    """The model replied, but not with usable output (refusal, invalid structure, bad value)."""


class LLMPromptError(LLMOutputError):
    """The prompt file is missing or unreadable: a configuration fault, never retried."""


class LLMRejectedError(LLMOutputError):
    """Valid output that a check rejected, with every reason (retried with them attached)."""

    prefix = "rejected"

    def __init__(self, problems: tuple[str, ...]) -> None:
        self.problems = problems
        super().__init__(f"{self.prefix}: " + "; ".join(problems))


class LLMGroundingError(LLMRejectedError):
    """The output cites evidence that isn't there, or states numbers the evidence lacks."""

    prefix = "not backed by the evidence"


def invalid_output(model: type[BaseModel], exc: ValidationError) -> LLMOutputError:
    """Names the failing fields (never their values), so a retry can say what to fix."""
    fields = sorted({".".join(map(str, e["loc"])) or "(root)" for e in exc.errors()})
    return LLMOutputError(f"invalid {model.__name__}: {', '.join(fields)}")


@dataclass(frozen=True)
class LLMRequest:
    prompt_id: str
    prompt_version: str
    system_prompt: str
    user_message: str
    output_model: type[BaseModel]


@dataclass(frozen=True)
class LLMResponse:
    parsed: BaseModel | None
    refusal: str | None
    raw_text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMProvider(Protocol):
    @property
    def name(self) -> str:
        """e.g. "azure_openai", "fake"."""
        ...

    @property
    def deployment(self) -> str:
        """Deployment or model name requests are routed to."""
        ...

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """One completion. Raises LLMUnavailableError if the model can't be reached."""
        ...

    async def aclose(self) -> None:
        """Release connections. Call it on the event loop the provider was used on."""
        ...
