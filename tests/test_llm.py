from unittest.mock import patch

import httpx2
import pytest
from openai import APITimeoutError

from aipc.config import load_config
from aipc.llm import MAX_ATTEMPTS, CostCeilingExceeded, IncompleteReply, LLMClient
from aipc.openrouter import CompletionResult

TIMEOUT = APITimeoutError(request=httpx2.Request("POST", "https://openrouter.ai/api/v1/chat/completions"))


def _completion(
    text="essay text", prompt_tokens=100, completion_tokens=100, billed_usd: float | None = 0.01, finish_reason="stop"
):
    return CompletionResult(
        text=text,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        served_model="test/served",
        provider="TestProvider",
        generation_id="gen-1",
        billed_usd=billed_usd,
        finish_reason=finish_reason,
    )


@pytest.fixture
def config():
    return load_config()


def test_dry_run_makes_no_api_call(config, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    llm = LLMClient(config, dry_run=True)
    result = llm.call(tier="cheap", prompt="write an essay about X")
    assert result.text == "[dry-run: no call made]"
    assert llm.total_cost_usd > 0


@patch("aipc.llm.complete")
def test_call_prices_from_usage_and_accumulates_cost(mock_complete, config):
    # Priced explicitly here rather than relying on default.yaml's live
    # cheap-tier price, so this test doesn't break every time that price
    # is retuned (it already has been once).
    config.models["cheap"].usd_per_mtok_in = 1.0
    config.models["cheap"].usd_per_mtok_out = 5.0
    mock_complete.return_value = _completion(prompt_tokens=100_000, completion_tokens=100_000, billed_usd=None)
    llm = LLMClient(config, dry_run=False)
    result = llm.call(tier="cheap", prompt="write")
    # $1.0/Mtok in + $5.0/Mtok out, at 0.1 Mtok each way
    assert result.cost_usd == pytest.approx(0.6)
    assert llm.total_cost_usd == pytest.approx(0.6)


@patch("aipc.llm.complete")
def test_call_raises_when_cost_ceiling_exceeded(mock_complete, config):
    config.limits.max_cost_usd = 1.0
    config.models["cheap"].usd_per_mtok_in = 1.0
    config.models["cheap"].usd_per_mtok_out = 5.0
    mock_complete.return_value = _completion(prompt_tokens=1_000_000, completion_tokens=1_000_000, billed_usd=None)
    llm = LLMClient(config, dry_run=False)
    with pytest.raises(CostCeilingExceeded):
        llm.call(tier="cheap", prompt="write")


def test_dry_run_also_raises_when_estimate_exceeds_ceiling(config):
    config.limits.max_cost_usd = 0.0001
    llm = LLMClient(config, dry_run=True)
    with pytest.raises(CostCeilingExceeded):
        llm.call(tier="cheap", prompt="write an essay about X")


@patch("aipc.llm.complete")
def test_real_call_is_recorded_with_provenance(mock_complete, config):
    mock_complete.return_value = _completion(text="a reply")
    recorded = []
    LLMClient(config, record=recorded.append).call(tier="cheap", prompt="the prompt", prompt_name="critic", seed=3)

    [call] = recorded
    assert (call.prompt_name, call.prompt, call.response, call.seed) == ("critic", "the prompt", "a reply", 3)
    assert (call.requested_model, call.served_model, call.provider) == (
        config.models["cheap"].id,
        "test/served",
        "TestProvider",
    )
    assert call.billed_usd == 0.01 and call.latency_s >= 0


def test_dry_run_is_not_recorded(config):
    recorded = []
    LLMClient(config, dry_run=True, record=recorded.append).call(tier="cheap", prompt="p")
    assert recorded == []


@patch("aipc.llm.complete")
def test_billed_cost_counts_against_the_ceiling_when_present(mock_complete, config):
    mock_complete.return_value = _completion(billed_usd=0.25)
    llm = LLMClient(config)
    assert llm.call(tier="cheap", prompt="write").cost_usd == 0.25
    assert llm.total_cost_usd == 0.25


@patch("aipc.llm.complete")
def test_unsupported_sampling_parameters_are_not_sent_and_recorded_as_such(mock_complete, config):
    config.models["cheap"].supports_seed = False
    config.models["cheap"].supports_temperature = False
    mock_complete.return_value = _completion()
    recorded = []
    LLMClient(config, record=recorded.append).call(tier="cheap", prompt="p", seed=3)

    assert mock_complete.call_args.kwargs["seed"] is None
    assert mock_complete.call_args.kwargs["temperature"] is None
    assert recorded[0].sampling_sent == ""


@patch("aipc.llm.time.sleep")
@patch("aipc.llm.complete")
def test_transient_failure_is_retried_and_counted(mock_complete, mock_sleep, config):
    mock_complete.side_effect = [TIMEOUT, _completion()]
    recorded = []
    LLMClient(config, record=recorded.append).call(tier="cheap", prompt="p")
    assert recorded[0].attempts == 2
    assert mock_sleep.call_count == 1


@patch("aipc.llm.time.sleep")
@patch("aipc.llm.complete")
def test_a_failure_on_the_last_attempt_propagates(mock_complete, mock_sleep, config):
    mock_complete.side_effect = TIMEOUT
    with pytest.raises(APITimeoutError):
        LLMClient(config).call(tier="cheap", prompt="p")
    assert mock_complete.call_count == MAX_ATTEMPTS
    assert mock_sleep.call_count == MAX_ATTEMPTS - 1


@patch("aipc.llm.complete")
def test_incomplete_reply_is_recorded_then_raised(mock_complete, config):
    mock_complete.return_value = _completion(text="cut off mid-sent", finish_reason="length")
    recorded = []
    llm = LLMClient(config, record=recorded.append)
    with pytest.raises(IncompleteReply):
        llm.call(tier="cheap", prompt="p")
    assert recorded[0].finish_reason == "length"
    assert llm.total_cost_usd == 0.01


def test_dry_run_allows_for_hidden_reasoning(config):
    config.models["cheap"].reasoning = False
    without = LLMClient(config, dry_run=True).call(tier="cheap", prompt="p", prompt_name="draft").completion_tokens
    config.models["cheap"].reasoning = True
    with_reasoning = (
        LLMClient(config, dry_run=True).call(tier="cheap", prompt="p", prompt_name="draft").completion_tokens
    )
    assert with_reasoning == int(without * 1.5)


def test_dry_run_prices_short_replies_below_essays(config):
    llm = LLMClient(config, dry_run=True)
    essay = llm.call(tier="cheap", prompt="p", prompt_name="zero_shot_thesis").completion_tokens
    verdict = llm.call(tier="cheap", prompt="p", prompt_name="validity_gate").completion_tokens
    assert verdict < essay


@patch("aipc.llm.complete")
def test_a_recorded_reply_is_replayed_free_and_recorded_with_its_source(mock_complete, config, call_record):
    recorded = []
    llm = LLMClient(config, record=recorded.append, replay=[call_record(response="old reply")], replay_run="run-old")

    result = llm.call(tier="cheap", prompt="write it", prompt_name="draft")

    assert result.text == "old reply"
    assert mock_complete.call_count == 0
    assert llm.total_cost_usd == 0.0
    assert recorded[0].replayed_from == "run-old"
    assert recorded[0].cost_usd == 0.0
    assert recorded[0].generation_id == "gen-old"


@patch("aipc.llm.complete")
def test_identical_prompts_replay_in_recorded_order_then_call_live(mock_complete, config, call_record):
    mock_complete.return_value = _completion(text="live reply")
    llm = LLMClient(config, replay=[call_record(response="first"), call_record(response="second")], replay_run="r")

    texts = [llm.call(tier="cheap", prompt="write it", prompt_name="draft").text for _ in range(3)]

    assert texts == ["first", "second", "live reply"]
    assert mock_complete.call_count == 1


@patch("aipc.llm.complete")
def test_only_a_finished_reply_to_the_same_prompt_and_seed_is_replayed(mock_complete, config, call_record):
    mock_complete.return_value = _completion(text="live reply")
    replay = [
        call_record(response="cut off", finish_reason="length"),
        call_record(response="other prompt", prompt="write something else"),
        call_record(response="other seed", seed=config.defaults.seed + 1),
    ]
    llm = LLMClient(config, replay=replay, replay_run="r")

    assert llm.call(tier="cheap", prompt="write it", prompt_name="draft").text == "live reply"
    assert mock_complete.call_count == 1
