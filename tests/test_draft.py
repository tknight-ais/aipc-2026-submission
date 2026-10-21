from aipc.config import load_config
from aipc.context import Context
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


def test_check_fidelity_parses_both_sections(sample_ledger, fake_llm, tmp_path):
    fake_llm.responses["fidelity"] = (
        "DROPPED OBJECTIONS:\nnone found\n\nUNSUPPORTED CLAIMS:\nnone found"
    )
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

    essay, rounds, fidelity = draft_stage.run(sample_ledger, ctx)

    assert len(rounds) == 2
    assert all(r.has_new_objection for r in rounds)
    assert essay.rounds_used == 2
    assert essay.text == "Revised draft text."
