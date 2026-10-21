"""Shared test fixtures: a fake LLM client and hand-written fixture
artifacts. No test in this suite makes a network call.
"""

import json
from pathlib import Path

import pytest

from aipc.llm import CallResult
from aipc.models import ContentLedger

FIXTURES_DIR = Path(__file__).parent / "fixtures"


class FakeLLM:
    """Drop-in replacement for LLMClient. Returns canned text keyed by
    prompt_name; never makes a network call.
    """

    def __init__(self, responses: dict[str, str] | None = None):
        self.responses = responses or {}
        self.calls = []
        self.total_cost_usd = 0.0

    def call(self, tier, prompt, system=None, temperature=None, seed=None, prompt_name=None):
        self.calls.append({"tier": tier, "prompt": prompt, "prompt_name": prompt_name, "seed": seed})
        value = self.responses.get(prompt_name, "stubbed response")
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
def sample_ledger() -> ContentLedger:
    data = json.loads((FIXTURES_DIR / "ledger.json").read_text())
    return ContentLedger(**data)
