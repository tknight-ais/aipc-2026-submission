import pytest

from aipc.config import load_config
from aipc.context import Context
from aipc.stages import literature as literature_stage
from aipc.store import Store


def _ctx(fake_llm, tmp_path, dry_run=False):
    config = load_config()
    config.dry_run = dry_run
    store = Store(runs_dir=tmp_path)
    return Context(llm=fake_llm, config=config, run_id="run1", store=store)


def test_run_formats_sources_and_parses_background(fake_llm, tmp_path, monkeypatch):
    fake_works = [
        {
            "id": "W1",
            "display_name": "Freedom of the Will and the Concept of a Person",
            "authorships": [{"author": {"display_name": "Harry G. Frankfurt"}}],
            "publication_year": 1971,
            "primary_location": {"source": {"display_name": "Journal of Philosophy"}},
        },
        {
            "id": "W2",
            "display_name": "Alternate Possibilities and Moral Responsibility",
            "authorships": [{"author": {"display_name": "Harry G. Frankfurt"}}],
            "publication_year": 1969,
            "primary_location": {"source": None},
        },
    ]
    monkeypatch.setattr(literature_stage, "search_openalex", lambda query, limit=8: fake_works)
    fake_llm.responses["literature_queries"] = (
        "consequence argument free will\ncompatibilism\nfrankfurt freedom of the will"
    )
    fake_llm.responses["literature_background"] = (
        "BACKGROUND:\nCompatibilists and incompatibilists disagree about determinism.\n\n"
        "ENGAGE:\n- Frankfurt's paper directly challenges incompatibilist positions."
    )
    ctx = _ctx(fake_llm, tmp_path)

    pack = literature_stage.run("free will", ctx)

    assert pack.queries == ["consequence argument free will", "compatibilism", "frankfurt freedom of the will"]
    assert pack.sources == [
        "Freedom of the Will and the Concept of a Person - Harry G. Frankfurt (1971), Journal of Philosophy",
        "Alternate Possibilities and Moral Responsibility - Harry G. Frankfurt (1969)",
    ], "deduplicated by id, venue where OpenAlex has one"
    assert pack.background == "Compatibilists and incompatibilists disagree about determinism."
    assert pack.engagement_list == ["Frankfurt's paper directly challenges incompatibilist positions."]
    assert [c["prompt_name"] for c in fake_llm.calls] == ["literature_queries", "literature_background"]
    call = fake_llm.calls[1]
    assert "Harry G. Frankfurt" in call["prompt"]


def test_a_title_the_topic_quotes_is_searched_first(fake_llm, tmp_path, monkeypatch):
    searched = []

    def fake_search(query, limit=8):
        searched.append(query)
        return []

    monkeypatch.setattr(literature_stage, "search_openalex", fake_search)
    fake_llm.responses["literature_queries"] = "compatibilism\nconsequence argument"
    topic = 'Frankfurt ("Freedom of the Will and the Concept of a Person", 1971) and “Alternate Possibilities and Moral Responsibility” say "ought" ...'

    literature_stage.run(topic, _ctx(fake_llm, tmp_path))

    assert searched == [
        "Freedom of the Will and the Concept of a Person",
        "Alternate Possibilities and Moral Responsibility",
        "compatibilism",
        "consequence argument",
    ]


def test_run_skips_search_in_dry_run(fake_llm, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("search_openalex should not be called in dry_run")

    monkeypatch.setattr(literature_stage, "search_openalex", fail)
    fake_llm.responses["literature_background"] = (
        "BACKGROUND:\nnone\n\nENGAGE:\nnone of the above are directly relevant"
    )
    ctx = _ctx(fake_llm, tmp_path, dry_run=True)

    pack = literature_stage.run("free will", ctx)

    assert pack.sources == []


class _Response:
    def __init__(self, status_code=200):
        self.status_code = status_code

    def raise_for_status(self):
        if self.status_code != 200:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return {"results": [{"display_name": "A work"}]}


def test_search_openalex_drops_wildcard_characters(monkeypatch):
    sent = {}

    def fake_get(url, params, headers, timeout):
        sent.update(params)
        return _Response()

    monkeypatch.setattr(literature_stage.httpx, "get", fake_get)
    literature_stage.search_openalex("Why be moral? A *study*")
    assert "?" not in sent["search"] and "*" not in sent["search"]


def test_search_openalex_retries_a_rate_limit_then_gives_up(monkeypatch):
    statuses = [429, 200]
    monkeypatch.setattr(literature_stage.httpx, "get", lambda url, params, headers, timeout: _Response(statuses.pop(0)))
    monkeypatch.setattr(literature_stage.time, "sleep", lambda seconds: None)
    assert literature_stage.search_openalex("free will") == [{"display_name": "A work"}]

    monkeypatch.setattr(literature_stage.httpx, "get", lambda url, params, headers, timeout: _Response(429))
    with pytest.raises(RuntimeError, match="HTTP 429"):
        literature_stage.search_openalex("free will")


def test_search_openalex_sends_a_key_as_a_header_not_in_the_url(monkeypatch):
    sent = {}

    def fake_get(url, params, headers, timeout):
        sent.update(params=params, headers=headers)
        return _Response()

    monkeypatch.setattr(literature_stage.httpx, "get", fake_get)
    monkeypatch.setenv("OPENALEX_API_KEY", "secret")
    literature_stage.search_openalex("free will")
    assert sent["headers"] == {"Authorization": "Bearer secret"}
    assert "secret" not in str(sent["params"])


def test_search_queries_drop_numbering_and_a_preamble(fake_llm, tmp_path):
    fake_llm.responses["literature_queries"] = "Queries:\n1. consequence argument\n2. compatibilism\n3. frankfurt cases"
    queries = literature_stage.search_queries("topic", _ctx(fake_llm, tmp_path))
    assert queries == ["consequence argument", "compatibilism", "frankfurt cases"]
