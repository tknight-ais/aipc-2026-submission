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
        }
    ]
    monkeypatch.setattr(literature_stage, "search_openalex", lambda query, limit=8: fake_works)
    fake_llm.responses["literature_queries"] = "consequence argument free will\ncompatibilism\nfrankfurt freedom of the will"
    fake_llm.responses["literature_background"] = (
        "BACKGROUND:\nCompatibilists and incompatibilists disagree about determinism.\n\n"
        "ENGAGE:\n- Frankfurt's paper directly challenges incompatibilist positions."
    )
    ctx = _ctx(fake_llm, tmp_path)

    pack = literature_stage.run("free will", ctx)

    assert pack.queries == ["consequence argument free will", "compatibilism", "frankfurt freedom of the will"]
    assert pack.sources == ["Freedom of the Will and the Concept of a Person - Harry G. Frankfurt (1971)"], "deduplicated by id"
    assert pack.background == "Compatibilists and incompatibilists disagree about determinism."
    assert pack.engagement_list == ["Frankfurt's paper directly challenges incompatibilist positions."]
    assert [c["prompt_name"] for c in fake_llm.calls] == ["literature_queries", "literature_background"]
    call = fake_llm.calls[1]
    assert "Harry G. Frankfurt" in call["prompt"]


def test_run_skips_search_in_dry_run(fake_llm, tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("search_openalex should not be called in dry_run")

    monkeypatch.setattr(literature_stage, "search_openalex", fail)
    fake_llm.responses["literature_background"] = "BACKGROUND:\nnone\n\nENGAGE:\nnone of the above are directly relevant"
    ctx = _ctx(fake_llm, tmp_path, dry_run=True)

    pack = literature_stage.run("free will", ctx)

    assert pack.sources == []


def test_search_openalex_drops_wildcard_characters(monkeypatch):
    sent = {}

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"results": []}

    def fake_get(url, params, timeout):
        sent.update(params)
        return Response()

    monkeypatch.setattr(literature_stage.httpx, "get", fake_get)
    literature_stage.search_openalex("Why be moral? A *study*")
    assert "?" not in sent["search"] and "*" not in sent["search"]
