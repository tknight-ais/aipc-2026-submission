"""LLM call wrapper: cost accounting, per-run cost ceiling, retries,
dry-run support.

Wraps openrouter.py so stages never call it directly - every call is
priced, logged, counted against the run's cost ceiling and handed to the
`record` callback (the store's `calls` table) in one place.
"""

import logging
import threading
import time
from collections import defaultdict, deque
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
    "zero_shot_thesis",
    "draft",
    "revise",
    "revise_free",
    "baseline",
    "degrade_essay",
    "judge_pairwise",
    "judge_pairwise_neutral",
    "judge_pairwise_reasoned",
    "judge_absolute",
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
    `replayed_from` names the failed run whose recorded reply was reused
    instead of calling the model; such a call costs nothing.
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
    replayed_from: str | None = None


ReplayKey = tuple[str | None, str, str, int]


def _recorded_prompt(prompt: str, system: str | None) -> str:
    return prompt if system is None else f"[system]\n{system}\n\n[user]\n{prompt}"


def _replay_key(prompt_name: str | None, model: str, recorded_prompt: str, seed: int) -> ReplayKey:
    return prompt_name, model, recorded_prompt, seed


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
    returns the result with the number of attempts made. The last attempt's
    failure propagates.
    """
    for attempt in range(1, MAX_ATTEMPTS):
        try:
            return complete(**kwargs), attempt
        except TRANSIENT_ERRORS as error:
            wait = RETRY_BASE_SECONDS**attempt
            logger.warning("attempt %d of call name=%s failed: %r; retrying in %ds", attempt, prompt_name, error, wait)
            time.sleep(wait)
    return complete(**kwargs), MAX_ATTEMPTS


class LLMClient:
    """Tracks cumulative cost for one run and enforces its ceiling.

    Given the calls of a failed run (`replay`, from run `replay_run`), a
    real call whose prompt name, model, prompt and seed match a finished
    recorded call reuses that reply instead of paying for it again. The
    stages are deterministic given their replies, so a rerun with the same
    settings replays up to the point of failure, then calls live. Matching
    calls replay in their recorded order, so repeated identical prompts
    (ideation's candidates) get their replies back in turn.
    """

    def __init__(
        self,
        config: Config,
        dry_run: bool = False,
        record: Callable[[CallRecord], None] | None = None,
        replay: list[CallRecord] | None = None,
        replay_run: str | None = None,
    ):
        self.config = config
        self.dry_run = dry_run
        self.record = record
        self.total_cost_usd = 0.0
        self.replay_run = replay_run
        self._replay: dict[ReplayKey, deque[CallRecord]] = defaultdict(deque)
        for call in replay or []:
            if call.finish_reason == "stop":
                key = _replay_key(call.prompt_name, call.requested_model, call.prompt, call.seed)
                self._replay[key].append(call)
        # Critics run as concurrent calls from a thread pool; this guards
        # the shared running total and the replay queues against races.
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
            result = self._estimate(model_config, prompt, prompt_name)
        else:
            result = self._send(tier, model_config, prompt, system, temperature, seed, prompt_name)

        computed_usd = _cost(model_config, result.prompt_tokens, result.completion_tokens)
        cost_usd = computed_usd if result.billed_usd is None else result.billed_usd
        with self._cost_lock:
            self.total_cost_usd += cost_usd
            total_cost_usd = self.total_cost_usd
        if total_cost_usd > self.config.limits.max_cost_usd:
            raise CostCeilingExceeded(
                f"run cost {total_cost_usd:.4f} exceeds ceiling {self.config.limits.max_cost_usd:.4f}"
            )
        if result.finish_reason != "stop":
            raise IncompleteReply(
                f"call name={prompt_name} model={model_config.id} finish_reason={result.finish_reason}"
            )
        logger.info(
            "%s name=%s tier=%s model=%s temperature=%s seed=%s "
            "prompt_tokens=%d completion_tokens=%d cost_usd=%.4f run_total_usd=%.4f",
            "dry_run call" if self.dry_run else "call",
            prompt_name,
            tier,
            model_config.id,
            temperature,
            seed,
            result.prompt_tokens,
            result.completion_tokens,
            cost_usd,
            total_cost_usd,
        )
        return CallResult(
            text=result.text,
            model=model_config.id,
            temperature=temperature,
            seed=seed,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cost_usd=cost_usd,
        )

    def _estimate(self, model_config: ModelConfig, prompt: str, prompt_name: str | None) -> CompletionResult:
        """A dry run's stand-in reply: no call is made, and the token counts
        are sized so the cost estimate stays an upper bound.
        """
        allowance = REASONING_ALLOWANCE if model_config.reasoning else 1.0
        essay_tokens = self.config.limits.target_words[1] * TOKENS_PER_WORD
        reply_tokens = essay_tokens if prompt_name in LONG_REPLY_PROMPTS else SHORT_REPLY_TOKENS
        return CompletionResult(
            text="[dry-run: no call made]",
            prompt_tokens=int(len(prompt) / CHARS_PER_TOKEN),
            completion_tokens=int(reply_tokens * allowance),
            served_model=model_config.id,
            provider=None,
            generation_id="dry-run",
            billed_usd=None,
        )

    def _send(
        self,
        tier: str,
        model_config: ModelConfig,
        prompt: str,
        system: str | None,
        temperature: float,
        seed: int,
        prompt_name: str | None,
    ) -> CompletionResult:
        """One real call, retried on transient failure and recorded whatever
        its finish reason, unless a failed run's recorded reply replays it.
        """
        recorded_prompt = _recorded_prompt(prompt, system)
        with self._cost_lock:
            queue = self._replay.get(_replay_key(prompt_name, model_config.id, recorded_prompt, seed))
            previous = queue.popleft() if queue else None
        if previous is not None:
            return self._replayed(previous)
        # Logged before the call, not just after - a call that hangs
        # otherwise leaves no trace of which one it was.
        logger.info("starting call name=%s tier=%s model=%s", prompt_name, tier, model_config.id)
        called_at = datetime.now(UTC).isoformat()
        started = time.monotonic()
        sent = {"temperature": model_config.supports_temperature, "seed": model_config.supports_seed}
        result, attempts = _complete_with_retries(
            prompt_name,
            model=model_config.id,
            prompt=prompt,
            system=system,
            temperature=temperature if sent["temperature"] else None,
            seed=seed if sent["seed"] else None,
            reasoning=model_config.reasoning,
        )
        if self.record is not None:
            self.record(
                CallRecord(
                    prompt_name=prompt_name,
                    tier=tier,
                    requested_model=model_config.id,
                    served_model=result.served_model,
                    provider=result.provider,
                    generation_id=result.generation_id,
                    temperature=temperature,
                    seed=seed,
                    sampling_sent=",".join(k for k, v in sent.items() if v),
                    prompt_tokens=result.prompt_tokens,
                    completion_tokens=result.completion_tokens,
                    reasoning_tokens=result.reasoning_tokens,
                    cost_usd=_cost(model_config, result.prompt_tokens, result.completion_tokens),
                    billed_usd=result.billed_usd,
                    finish_reason=result.finish_reason,
                    attempts=attempts,
                    latency_s=time.monotonic() - started,
                    called_at=called_at,
                    prompt=recorded_prompt,
                    response=result.text,
                )
            )
        return result

    def _replayed(self, previous: CallRecord) -> CompletionResult:
        """A recorded reply reused at no cost, and recorded again for this
        run with the run it came from.
        """
        logger.info("replayed call name=%s from run %s", previous.prompt_name, self.replay_run)
        if self.record is not None:
            self.record(
                previous.model_copy(
                    update={
                        "cost_usd": 0.0,
                        "billed_usd": 0.0,
                        "attempts": 0,
                        "latency_s": 0.0,
                        "called_at": datetime.now(UTC).isoformat(),
                        "replayed_from": self.replay_run,
                    }
                )
            )
        return CompletionResult(
            text=previous.response,
            prompt_tokens=previous.prompt_tokens,
            completion_tokens=previous.completion_tokens,
            served_model=previous.served_model,
            provider=previous.provider,
            generation_id=previous.generation_id,
            billed_usd=0.0,
            finish_reason=previous.finish_reason,
            reasoning_tokens=previous.reasoning_tokens,
        )
