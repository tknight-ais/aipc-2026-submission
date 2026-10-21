from unittest.mock import MagicMock, patch

import pytest

import aipc.openrouter
from aipc.openrouter import complete, get_client


@pytest.fixture(autouse=True)
def _reset_cached_client():
    # get_client() caches its OpenAI client at module level (reused
    # across calls in one real run); reset it so tests don't leak state
    # into each other via that cache.
    aipc.openrouter._client = None
    yield
    aipc.openrouter._client = None


def test_get_client_raises_without_api_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        get_client()


def test_get_client_reads_api_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    client = get_client()
    assert client.api_key == "test-key"


def _mock_response(content, prompt_tokens, completion_tokens):
    mock_response = MagicMock()
    mock_response.choices[0].message.content = content
    mock_response.choices[0].finish_reason = "stop"
    mock_response.usage.completion_tokens_details = None
    mock_response.usage.prompt_tokens = prompt_tokens
    mock_response.usage.completion_tokens = completion_tokens
    mock_response.usage.model_extra = {"cost": 0.002}
    mock_response.model = "test/model-served"
    mock_response.id = "gen-1"
    mock_response.model_extra = {"provider": "TestProvider"}
    return mock_response


@patch("aipc.openrouter.get_client")
def test_complete_returns_text_and_usage(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response(
        "stubbed essay text", prompt_tokens=10, completion_tokens=20
    )
    mock_get_client.return_value = mock_client

    result = complete(model="test/model", prompt="write something")

    assert result.text == "stubbed essay text"
    assert result.prompt_tokens == 10
    assert result.completion_tokens == 20
    assert (result.served_model, result.provider, result.generation_id, result.billed_usd) == (
        "test/model-served", "TestProvider", "gen-1", 0.002
    )
    mock_client.chat.completions.create.assert_called_once_with(
        model="test/model",
        messages=[{"role": "user", "content": "write something"}],
        max_tokens=aipc.openrouter.MAX_OUTPUT_TOKENS,
        extra_body=None,
        temperature=1.0,
    )


@patch("aipc.openrouter.get_client")
def test_complete_passes_seed_and_system_message(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response(
        "stubbed essay text", prompt_tokens=10, completion_tokens=20
    )
    mock_get_client.return_value = mock_client

    complete(
        model="test/model",
        prompt="write something",
        seed=7,
        system="You are a helpful assistant.",
    )

    mock_client.chat.completions.create.assert_called_once_with(
        model="test/model",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "write something"},
        ],
        max_tokens=aipc.openrouter.MAX_OUTPUT_TOKENS,
        extra_body=None,
        temperature=1.0,
        seed=7,
    )


@patch("aipc.openrouter.get_client")
def test_complete_can_switch_off_reasoning(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("text", prompt_tokens=1, completion_tokens=1)
    mock_get_client.return_value = mock_client

    complete(model="test/model", prompt="p", reasoning=False)

    assert mock_client.chat.completions.create.call_args.kwargs["extra_body"] == {"reasoning": {"enabled": False}}


@patch("aipc.openrouter.get_client")
def test_complete_omits_unsupported_sampling_parameters(mock_get_client):
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_response("text", prompt_tokens=1, completion_tokens=1)
    mock_get_client.return_value = mock_client

    complete(model="test/model", prompt="p", temperature=None, seed=None)

    kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert "temperature" not in kwargs and "seed" not in kwargs


@patch("aipc.openrouter.get_client")
def test_complete_reports_finish_reason_and_empty_content(mock_get_client):
    mock_client = MagicMock()
    response = _mock_response(None, prompt_tokens=1, completion_tokens=1)
    response.choices[0].finish_reason = "content_filter"
    mock_client.chat.completions.create.return_value = response
    mock_get_client.return_value = mock_client

    result = complete(model="test/model", prompt="p")

    assert (result.text, result.finish_reason) == ("", "content_filter")
