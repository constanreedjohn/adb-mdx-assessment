"""`make smoke-llm`: one structured-output call to the Azure OpenAI deployment.

Proves, before any agent work depends on it (plan A-04, Q-07):
- the endpoint, deployment, API version and credentials work;
- structured output parses into a Pydantic model (what the gateway relies on);
- the reply follows the prompt (it echoes a random token back unchanged).

Prints the deployment, API version, reported model, tokens and latency. Never
prints keys. Exit code 0 on success, 1 otherwise.
"""

from __future__ import annotations

import secrets
import sys
import time
from dataclasses import dataclass
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, ConfigDict

from mdx_assessment.core.config import load_prompt
from mdx_assessment.core.gateway.azure_openai import (
    LLMNotConfiguredError,
    create_azure_openai_client,
)
from mdx_assessment.settings import get_settings

PROMPT_ID = "smoke_test"
PROMPT_VERSION = "v1"


class SmokeReply(BaseModel):
    """What the prompt asks for. `extra=forbid` keeps the schema strict for the model."""

    model_config = ConfigDict(extra="forbid")

    status: Literal["ok"]
    model_family: str
    echo: str


@dataclass(frozen=True)
class SmokeResult:
    reply: SmokeReply
    model: str
    prompt_tokens: int
    completion_tokens: int
    latency_ms: int
    echo_matches: bool


class SmokeError(RuntimeError):
    pass


def run_smoke(client: OpenAI, deployment: str, system_prompt: str) -> SmokeResult:
    """One call. `client` is an `AzureOpenAI` (a subclass of `OpenAI`) or a test fake."""
    token = "mdx-" + secrets.token_hex(4)
    started = time.perf_counter()
    completion = client.chat.completions.parse(
        model=deployment,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Token: {token}"},
        ],
        response_format=SmokeReply,
    )
    latency_ms = round((time.perf_counter() - started) * 1000)
    message = completion.choices[0].message
    if message.refusal:
        raise SmokeError(f"the model refused: {message.refusal}")
    reply = message.parsed
    if not isinstance(reply, SmokeReply):
        raise SmokeError("no structured reply was returned")
    usage = completion.usage
    return SmokeResult(
        reply=reply,
        model=str(completion.model),
        prompt_tokens=int(usage.prompt_tokens) if usage else 0,
        completion_tokens=int(usage.completion_tokens) if usage else 0,
        latency_ms=latency_ms,
        echo_matches=reply.echo == token,
    )


def main() -> int:
    settings = get_settings()
    azure = settings.azure_openai
    print(f"Endpoint host : {azure.endpoint.split('//')[-1].split('/')[0] or '(not set)'}")
    print(f"Deployment    : {azure.deployment or '(not set)'}")
    print(f"API version   : {azure.api_version}")
    print(f"Auth mode     : {azure.auth_mode.value}")
    try:
        client = create_azure_openai_client(azure)
        prompt = load_prompt(settings.app.config_dir, PROMPT_ID, PROMPT_VERSION)
        result = run_smoke(client, azure.deployment, prompt)
    except LLMNotConfiguredError as exc:
        print(f"FAILED: {exc}")
        return 1
    except Exception as exc:  # report any API/auth/network problem readably
        print(f"FAILED: {type(exc).__name__}: {_first_line(str(exc))}")
        return 1

    print(f"Model         : {result.model}")
    print(f"Prompt        : {PROMPT_ID}/{PROMPT_VERSION}")
    print(f"Reply         : {result.reply.model_dump_json()}")
    print(f"Tokens        : {result.prompt_tokens} in, {result.completion_tokens} out")
    print(f"Latency       : {result.latency_ms} ms")
    if not result.echo_matches:
        print("FAILED: the reply did not echo the token back unchanged.")
        return 1
    print("Result        : ok (structured output parsed into SmokeReply)")
    return 0


def _first_line(text: str) -> str:
    return text.strip().splitlines()[0][:300] if text.strip() else ""


if __name__ == "__main__":
    sys.exit(main())
