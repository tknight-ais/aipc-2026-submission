"""LLM call wrapper: cost accounting, per-run cost ceiling, retries,
dry-run support.

Wraps openrouter.py so stages never call it directly - every call is
priced, logged, counted against the run's cost ceiling and handed to the
`record` callback (the store's `calls` table) in one place.
"""

import logging
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime

from openai import (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
)
from pydantic import BaseModel

from aipc.config import Config, ModelConfig
from aipc.openrouter import CompletionResult, complete

logger = logging.getLogger(__name__)

# Used only to size a dry-run cost estimate, where no real completion
# exists yet to measure. Measured on Claude Haiku 5.5 essay calls: 3.3
# prompt characters per token, 1.86 tokens per essay word.
# Rounded so the estimate stays an upper bound.
CHARS_PER_TOKEN = 3
TOKENS_PER_WORD = 2.0
# Hidden reasoning on a model that reasons: one Opus 5.5 essay billed 1.32x
# its visible output and overshot the word target by 5%.
REASONING_ALLOWANCE = 1.5
# Prompts whose reply is an essay or a judgment of two essays are priced
# at essay length; every other reply at SHORT_REPLY_TOKENS. The longest
# other reply measured was 2,188 tokens (ledger_flat).
LONG_REPLY_PROMPTS = {
    "zero_shot_thesis", "draft", "revise", "revise_free", "baseline", "degrade_essay",
    "judge_pairwise", "judge_pairwise_neutral", "judge_pairwise_reasoned", "judge_absolute",
}
SHORT_REPLY_TOKENS = 2500

MAX_ATTEMPTS = 4
RETRY_BASE_SECONDS = 2
TRANSIENT_ERRORS = (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError)


class CallResult(BaseModel):
    text: str
    model: str
    temperature: float
    seed: int
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class CallRecord(BaseModel):
    """One real model call, as persisted: enough to audit what was asked,
    who served it, what it cost and what came back.

    `temperature` and `seed` are the configured values; `sampling_sent`
    names the ones actually sent (empty for models that support neither).
    `attempts` counts tries, failed ones included; a failed try may still
    have been billed, so attempts > 1 means check the provider's records.
    """

    prompt_name: str | None
    tier: str
    requested_model: str
    served_model: str
    provider: str | None
    generation_id: str
    temperature: float
    seed: int
    sampling_sent: str
    prompt_tokens: int
    completion_tokens: int
    reasoning_tokens: int | None
    cost_usd: float
    billed_usd: float | None
    finish_reason: str | None
    attempts: int
    latency_s: float
    called_at: str
    prompt: str
    response: str


class CostCeilingExceeded(Exception):
    pass


class IncompleteReply(Exception):
    """The model stopped for a reason other than finishing (length cap,
    content filter): the text would otherwise pass as a complete essay.
    """


def _cost(model_config: ModelConfig, prompt_tokens: int, completion_tokens: int) -> float:
    return (
        prompt_tokens / 1_000_000 * model_config.usd_per_mtok_in
        + completion_tokens / 1_000_000 * model_config.usd_per_mtok_out
    )


def _complete_with_retries(prompt_name: str | None, **kwargs) -> tuple[CompletionResult, int]:
    """Retries transient failures with backoff, logging each one, and
    returns the result with the number of attempts made.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return complete(**kwargs), attempt
        except TRANSIENT_ERRORS as error:
            if attempt == MAX_ATTEMPTS:
                raise
            wait = RETRY_BASE_SECONDS**attempt
            logger.warning("attempt %d of call name=%s failed: %r; retrying in %ds", attempt, prompt_name, error, wait)
            time.sleep(wait)


class LLMClient:
    """Tracks cumulative cost for one run and enforces its ceiling."""

    def __init__(self, config: Config, dry_run: bool = False, record: Callable[[CallRecord], None] | None = None):
        self.config = config
        self.dry_run = dry_run
        self.record = record
        self.total_cost_usd = 0.0
        # Critics run as concurrent calls from a thread pool; this guards
        # the shared running total against a lost-update race.
        self._cost_lock = threading.Lock()

    def call(
        self,
        tier: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        prompt_name: str | None = None,
    ) -> CallResult:
        model_config = self.config.model_for_tier(tier)
        temperature = self.config.defaults.temperature if temperature is None else temperature
        seed = self.config.defaults.seed if seed is None else seed

        if self.dry_run:
            text = "[dry-run: no call made]"
            prompt_tokens = int(len(prompt) / CHARS_PER_TOKEN)
            allowance = REASONING_ALLOWANCE if model_config.reasoning else 1.0
            essay_tokens = self.config.limits.target_words[1] * TOKENS_PER_WORD
            reply_tokens = essay_tokens if prompt_name in LONG_REPLY_PROMPTS else SHORT_REPLY_TOKENS
            completion_tokens = int(reply_tokens * allowance)
            cost_usd = _cost(model_config, prompt_tokens, completion_tokens)
            log_prefix = "dry_run call"
        else:
            # Logged before the call, not just after - a call that hangs
            # otherwise leaves no trace of which one it was.
            logger.info("starting call name=%s tier=%s model=%s", prompt_name, tier, model_config.id)
            called_at = datetime.now(UTC).isoformat()
            started = time.monotonic()
            sent = {"temperature": model_config.supports_temperature, "seed": model_config.supports_seed}
            result, attempts = _complete_with_retries(
                prompt_name, model=model_config.id, prompt=prompt, system=system,
                temperature=temperature if sent["temperature"] else None,
                seed=seed if sent["seed"] else None,
                reasoning=model_config.reasoning,
            )
            latency_s = time.monotonic() - started
            text = result.text
            prompt_tokens = result.prompt_tokens
            completion_tokens = result.completion_tokens
            computed_usd = _cost(model_config, prompt_tokens, completion_tokens)
            cost_usd = computed_usd if result.billed_usd is None else result.billed_usd
            log_prefix = "call"
            if self.record is not None:
                self.record(CallRecord(
                    prompt_name=prompt_name, tier=tier, requested_model=model_config.id,
                    served_model=result.served_model, provider=result.provider, generation_id=result.generation_id,
                    temperature=temperature, seed=seed, sampling_sent=",".join(k for k, v in sent.items() if v),
                    prompt_tokens=prompt_tokens, completion_tokens=completion_tokens,
                    reasoning_tokens=result.reasoning_tokens, cost_usd=computed_usd, billed_usd=result.billed_usd,
                    finish_reason=result.finish_reason, attempts=attempts, latency_s=latency_s, called_at=called_at,
                    prompt=prompt if system is None else f"[system]\n{system}\n\n[user]\n{prompt}", response=text,
                ))

        with self._cost_lock:
            self.total_cost_usd += cost_usd
            total_cost_usd = self.total_cost_usd
        if total_cost_usd > self.config.limits.max_cost_usd:
            raise CostCeilingExceeded(
                f"run cost {total_cost_usd:.4f} exceeds ceiling "
                f"{self.config.limits.max_cost_usd:.4f}"
            )
        if not self.dry_run and result.finish_reason != "stop":
            raise IncompleteReply(f"call name={prompt_name} model={model_config.id} finish_reason={result.finish_reason}")
        logger.info(
            "%s name=%s tier=%s model=%s temperature=%s seed=%s "
            "prompt_tokens=%d completion_tokens=%d cost_usd=%.4f run_total_usd=%.4f",
            log_prefix, prompt_name, tier, model_config.id, temperature, seed,
            prompt_tokens, completion_tokens, cost_usd, total_cost_usd,
        )
        return CallResult(
            text=text,
            model=model_config.id,
            temperature=temperature,
            seed=seed,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost_usd=cost_usd,
        )
