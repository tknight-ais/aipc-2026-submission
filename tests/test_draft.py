import pytest

from aipc.config import load_config
from aipc.context import Context
from aipc.models import Essay
from aipc.stages import draft as draft_stage
from aipc.store import Store


def _ctx(fake_llm, tmp_path, critics=1, max_revision_rounds=3):
    config = load_config()
    config.pipeline.critics = critics
    config.pipeline.max_revision_rounds = max_revision_rounds
    store = Store(runs_dir=tmp_path)
    return Context(llm=fake_llm, config=config, run_id="run1", store=store)


def test_draft_essay_produces_essay_from_ledger(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["draft"] = "A full essay body developing the thesis and its objections."
    ctx = _ctx(fake_llm, tmp_path)

    essay = draft_stage.draft_essay(sample_ledger, ctx)

    assert essay.thesis == sample_ledger.thesis
    assert essay.text == fake_llm.responses["draft"]
    assert essay.word_count == len(essay.text.split())

    call = fake_llm.calls[0]
    assert call["prompt_name"] == "draft"
    assert sample_ledger.thesis in call["prompt"]
    assert sample_ledger.objections[0].statement in call["prompt"]


def test_writing_prompts_show_the_question_engagements_and_sources(sample_ledger, fake_llm, tmp_path):
    source = "Freedom of the Will and the Concept of a Person - Harry G. Frankfurt (1971)"
    ledger = sample_ledger.model_copy(
        update={
            "question": "Is free will compatible with determinism?",
            "engagements": ["Frankfurt (1971)"],
            "sources": [source],
        }
    )
    ctx = _ctx(fake_llm, tmp_path)

    draft_stage.draft_essay(ledger, ctx)
    draft_stage.revise(Essay.from_text("Body.", thesis=ledger.thesis), ledger, ["An objection."], ctx)
    draft_stage.check_fidelity(Essay.from_text("Body.", thesis=ledger.thesis), ledger, ctx)

    for call in fake_llm.calls:
        assert "Is free will compatible with determinism?" in call["prompt"], call["prompt_name"]
        assert "Frankfurt (1971)" in call["prompt"], call["prompt_name"]
        assert source in call["prompt"], call["prompt_name"]


def test_a_ledger_without_a_question_renders_as_before(sample_ledger):
    assert sample_ledger.question is None and sample_ledger.sources is None
    rendered = draft_stage.ledger_json(sample_ledger)
    assert '"question"' not in rendered and '"sources"' not in rendered
    assert rendered == sample_ledger.model_dump_json(indent=2, exclude={"question", "sources"})


def test_run_critics_returns_one_critique_per_critic(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["critic"] = "The analogy to perception begs the question."
    ctx = _ctx(fake_llm, tmp_path, critics=3)
    essay = draft_stage.draft_essay(sample_ledger, ctx)

    critiques = draft_stage.run_critics(essay, sample_ledger.thesis, previously_raised=[], ctx=ctx)

    assert len(critiques) == 3
    assert {c.critic_id for c in critiques} == {0, 1, 2}
    assert all(c.text == fake_llm.responses["critic"] for c in critiques)
    critic_calls = [c for c in fake_llm.calls if c["prompt_name"] == "critic"]
    assert sample_ledger.thesis in critic_calls[0]["prompt"]
    assert sorted(c["seed"] for c in critic_calls) == [ctx.config.defaults.seed + i for i in range(3)]


def test_revise_addresses_objections(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["draft"] = "First draft text."
    fake_llm.responses["revise"] = "Revised essay text addressing the objection."
    ctx = _ctx(fake_llm, tmp_path)
    essay = draft_stage.draft_essay(sample_ledger, ctx)

    revised = draft_stage.revise(essay, sample_ledger, objections=["an objection"], ctx=ctx)

    assert revised.text == fake_llm.responses["revise"]
    assert revised.thesis == sample_ledger.thesis
    call = fake_llm.calls[-1]
    assert call["prompt_name"] == "revise"
    assert "an objection" in call["prompt"]


def test_both_revise_prompts_give_the_current_body_length(sample_ledger, fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path)
    essay = Essay.from_text("Title\n\nFive words in the body.\n\nReferences\nA, B. 2001. C.", thesis="T")

    draft_stage.revise(essay, sample_ledger, objections=["an objection"], ctx=ctx)
    draft_stage.revise_free(essay, objections=["an objection"], ctx=ctx)

    assert essay.word_count == 6
    for call in fake_llm.calls:
        assert "The current draft's body is 6 words" in call["prompt"], call["prompt_name"]


def test_last_within_keeps_the_latest_version_under_the_limit():
    versions = [Essay.from_text(" ".join(["w"] * n), thesis="T") for n in (8, 9, 12)]
    for i, version in enumerate(versions):
        version.rounds_used = i

    assert draft_stage.last_within(versions, max_words=12).rounds_used == 2
    assert draft_stage.last_within(versions, max_words=10).rounds_used == 1
    with pytest.raises(ValueError, match="every version is over 7 words"):
        draft_stage.last_within(versions, max_words=7)


def test_run_with_a_word_limit_drops_a_revision_that_grows_past_it(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["draft"] = "Short first draft."
    fake_llm.responses["critic"] = "A persistent objection that is never fully resolved."
    fake_llm.responses["revise"] = ["A revision within the limit.", "A revision that has grown far past the limit."]
    ctx = _ctx(fake_llm, tmp_path, critics=1, max_revision_rounds=2)

    essay, rounds, _ = draft_stage.run(sample_ledger, ctx, max_words=6)

    assert len(rounds) == 2
    assert (essay.text, essay.rounds_used) == ("A revision within the limit.", 1)
    fidelity_call = next(c for c in fake_llm.calls if c["prompt_name"] == "fidelity")
    assert "A revision within the limit." in fidelity_call["prompt"]


def test_check_fidelity_parses_both_sections(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["fidelity"] = "DROPPED OBJECTIONS:\nnone found\n\nUNSUPPORTED CLAIMS:\nnone found"
    ctx = _ctx(fake_llm, tmp_path)
    essay = draft_stage.draft_essay(sample_ledger, ctx)

    review = draft_stage.check_fidelity(essay, sample_ledger, ctx)

    assert review.dropped_objections == "none found"
    assert review.unsupported_claims == "none found"


def test_run_stops_when_critics_find_nothing_new(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["draft"] = "First draft text."
    fake_llm.responses["critic"] = "NONE"
    fake_llm.responses["fidelity"] = "DROPPED OBJECTIONS:\nnone found\n\nUNSUPPORTED CLAIMS:\nnone found"
    ctx = _ctx(fake_llm, tmp_path, critics=1, max_revision_rounds=3)

    essay, rounds, fidelity = draft_stage.run(sample_ledger, ctx)

    assert essay.text == "First draft text."
    assert essay.rounds_used == 0
    assert len(rounds) == 1
    assert rounds[0].has_new_objection is False
    assert fidelity.dropped_objections == "none found"
    assert not any(c["prompt_name"] == "revise" for c in fake_llm.calls)


def test_run_hits_hard_cap_when_objections_keep_coming(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["draft"] = "First draft text."
    fake_llm.responses["critic"] = "A persistent objection that is never fully resolved."
    fake_llm.responses["revise"] = "Revised draft text."
    fake_llm.responses["fidelity"] = "DROPPED OBJECTIONS:\nnone found\n\nUNSUPPORTED CLAIMS:\nnone found"
    ctx = _ctx(fake_llm, tmp_path, critics=1, max_revision_rounds=2)

    essay, rounds, _ = draft_stage.run(sample_ledger, ctx)

    assert len(rounds) == 2
    assert all(r.has_new_objection for r in rounds)
    assert essay.rounds_used == 2
    assert essay.text == "Revised draft text."


def test_hedge_density_counts_whole_phrases_in_the_body_only():
    body = "Perhaps so. A process we cannot audit may be wrong, and it is impossibly hard."
    text = f"{body}\n\nReferences\nPerhaps, A. 2001. Arguably True. Possibly Press."
    assert draft_stage.hedge_density(text) == pytest.approx(1 / len(body.split()) * 1000)
