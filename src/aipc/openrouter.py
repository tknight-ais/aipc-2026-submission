"""Thin client for calling models via OpenRouter's OpenAI-compatible API."""

import os
from dataclasses import dataclass

import httpx
from openai import OpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REQUEST_TIMEOUT_SECONDS = 120.0
# A cap, not a target: about three times the longest essay asked for, so a
# truncated reply means something went wrong. finish_reason catches it.
MAX_OUTPUT_TOKENS = 32_000

_client: OpenAI | None = None


@dataclass
class CompletionResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    served_model: str
    provider: str | None
    generation_id: str
    billed_usd: float | None
    finish_reason: str | None = "stop"
    reasoning_tokens: int | None = None


def get_client() -> OpenAI:
    """One client per process: a run makes dozens of calls, and a fresh
    client per call means a fresh connection pool each time, a plausible
    cause of a hung run seen live.

    The SDK's own retries are off: they are invisible to the call log and
    the cost ceiling. LLMClient retries, visibly.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        _client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=api_key, max_retries=0, timeout=REQUEST_TIMEOUT_SECONDS)
    return _client


def complete(
    model: str,
    prompt: str,
    temperature: float | None = 1.0,
    seed: int | None = None,
    system: str | None = None,
    reasoning: bool = True,
) -> CompletionResult:
    """`reasoning=False` asks OpenRouter to switch off hidden reasoning, which
    otherwise multiplied Haiku 5.5's billed output about 5x. Some models
    (Opus 5.5) refuse to switch it off.

    `temperature` and `seed` are sent only when given: OpenRouter silently
    drops parameters a model doesn't support, so the caller leaves them out
    for such models rather than record values that were never applied.

    OpenRouter adds the serving provider and the billed cost to the
    response; both are kept, because one model id can be served by
    different providers.
    """
    client = get_client()
    messages = []
    if system is not None:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    sampling = {k: v for k, v in {"temperature": temperature, "seed": seed}.items() if v is not None}
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=MAX_OUTPUT_TOKENS,
        extra_body=None if reasoning else {"reasoning": {"enabled": False}},
        **sampling,
    )
    usage = response.usage
    choice = response.choices[0]
    details = usage.completion_tokens_details
    return CompletionResult(
        text=choice.message.content or "",
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=usage.completion_tokens,
        served_model=response.model,
        provider=(response.model_extra or {}).get("provider"),
        generation_id=response.id,
        billed_usd=(usage.model_extra or {}).get("cost"),
        finish_reason=choice.finish_reason,
        reasoning_tokens=details.reasoning_tokens if details else None,
    )


def list_models() -> list[dict]:
    """OpenRouter's public model listing: ids, prices, supported parameters."""
    response = httpx.get(f"{OPENROUTER_BASE_URL}/models", timeout=30)
    response.raise_for_status()
    return response.json()["data"]
