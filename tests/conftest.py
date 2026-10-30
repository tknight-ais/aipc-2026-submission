"""Shared test fixtures: a fake LLM client and hand-written fixture
artifacts. No test in this suite makes a network call.
"""

import json
from pathlib import Path

import pytest

from aipc.config import load_config
from aipc.llm import CallRecord, CallResult
from aipc.models import ContentLedger

FIXTURES_DIR = Path(__file__).parent / "fixtures"
# Replies whose sections the parsers require; a test can override them.
WELL_FORMED = {
    "fidelity": "DROPPED OBJECTIONS:\nnone found\n\nUNSUPPORTED CLAIMS:\nnone found",
    "literature_background": "BACKGROUND:\nA background.\n\nENGAGE:\nnone of the above are directly relevant",
}


class FakeLLM:
    """Drop-in replacement for LLMClient. Returns canned text keyed by
    prompt_name; never makes a network call. Prompts whose replies must
    have named sections get a well-formed default.
    """

    def __init__(self, responses: dict[str, str | list[str]] | None = None):
        self.responses = {**WELL_FORMED, **(responses or {})}
        self.calls = []
        self.total_cost_usd = 0.0

    def call(
        self,
        tier: str,
        prompt: str,
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        prompt_name: str | None = None,
    ) -> CallResult:
        self.calls.append({"tier": tier, "prompt": prompt, "prompt_name": prompt_name, "seed": seed})
        value = self.responses.get(prompt_name or "", "stubbed response")
        if isinstance(value, list):
            # Cycles through canned responses in call order for this
            # prompt_name - lets a test give round 1 and round 2 of a
            # revision loop different critic output, repeating the last
            # entry once the list runs out.
            prior_calls = sum(1 for c in self.calls if c["prompt_name"] == prompt_name)
            text = value[min(prior_calls - 1, len(value) - 1)]
        else:
            text = value
        return CallResult(
            text=text,
            model=f"fake/{tier}",
            temperature=temperature if temperature is not None else 1.0,
            seed=seed if seed is not None else 1,
            prompt_tokens=10,
            completion_tokens=10,
            cost_usd=0.0,
        )


@pytest.fixture
def fake_llm():
    return FakeLLM()


@pytest.fixture
def call_record():
    """Builds a finished call on the cheap tier as the calls table stores
    it; keyword arguments override any field.
    """
    config = load_config()

    def build(**overrides) -> CallRecord:
        fields = {
            "prompt_name": "draft",
            "tier": "cheap",
            "requested_model": config.model_for_tier("cheap").id,
            "served_model": "test/served",
            "provider": "TestProvider",
            "generation_id": "gen-old",
            "temperature": 1.0,
            "seed": config.defaults.seed,
            "sampling_sent": "",
            "prompt_tokens": 100,
            "completion_tokens": 100,
            "reasoning_tokens": None,
            "cost_usd": 0.05,
            "billed_usd": 0.05,
            "finish_reason": "stop",
            "attempts": 1,
            "latency_s": 3.0,
            "called_at": "2026-10-26T00:00:00+00:00",
            "prompt": "write it",
            "response": "a reply",
        }
        return CallRecord(**(fields | overrides))

    return build


@pytest.fixture
def sample_ledger() -> ContentLedger:
    data = json.loads((FIXTURES_DIR / "ledger.json").read_text())
    return ContentLedger(**data)
