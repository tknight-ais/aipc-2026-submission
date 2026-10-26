from aipc.config import load_config
from aipc.context import Context
from aipc.models import LiteraturePack, ThesisCandidate
from aipc.stages import ideation as ideation_stage
from aipc.store import Store

THESIS_TEXT = "THESIS:\nMoral facts are mind-independent.\n\nRATIONALE:\nIt resists a common objection."


def _ctx(fake_llm, tmp_path, n=2):
    config = load_config()
    config.pipeline.ideation_candidates = n
    store = Store(runs_dir=tmp_path)
    return Context(llm=fake_llm, config=config, run_id="run1", store=store)


def test_generate_cold_produces_n_candidates(fake_llm, tmp_path):
    fake_llm.responses["ideation_cold"] = THESIS_TEXT
    ctx = _ctx(fake_llm, tmp_path, n=3)

    candidates = ideation_stage.generate_cold("metaethics", 3, ctx)

    assert len(candidates) == 3
    assert all(c.origin == "cold" for c in candidates)
    assert candidates[0].claim == "Moral facts are mind-independent."
    assert candidates[0].rationale == "It resists a common objection."


def test_generate_informed_includes_background_in_prompt(fake_llm, tmp_path):
    fake_llm.responses["ideation_informed"] = THESIS_TEXT
    lit_pack = LiteraturePack(background="Compatibilists and incompatibilists disagree.", sources=["Frankfurt (1971)"])
    ctx = _ctx(fake_llm, tmp_path, n=1)

    candidates = ideation_stage.generate_informed("metaethics", lit_pack, 1, ctx)

    assert candidates[0].origin == "informed"
    call = fake_llm.calls[0]
    assert "Compatibilists and incompatibilists disagree." in call["prompt"]
    assert "Frankfurt (1971)" in call["prompt"]


def test_check_significance_parses_pass(fake_llm, tmp_path):
    fake_llm.responses["significance_filter"] = "VERDICT: PASS\nREASONING:\nIt is a live dispute."
    ctx = _ctx(fake_llm, tmp_path)
    candidate = ThesisCandidate(claim="X", origin="cold", rationale="Y")

    verdict = ideation_stage.check_significance(candidate, ctx)

    assert verdict.passed is True
    assert verdict.reasoning == "It is a live dispute."


def test_check_novelty_parses_fail(fake_llm, tmp_path):
    fake_llm.responses["novelty_filter"] = "VERDICT: FAIL\nREASONING:\nFrankfurt already argues this."
    ctx = _ctx(fake_llm, tmp_path)
    candidate = ThesisCandidate(claim="X", origin="cold", rationale="Y")
    lit_pack = LiteraturePack(background="", sources=["Frankfurt (1971)"])

    verdict = ideation_stage.check_novelty(candidate, lit_pack, ctx)

    assert verdict.passed is False


def test_screen_skips_novelty_when_significance_fails(fake_llm, tmp_path):
    fake_llm.responses["significance_filter"] = "VERDICT: FAIL\nREASONING:\nAlready settled."
    ctx = _ctx(fake_llm, tmp_path)
    candidate = ThesisCandidate(claim="X", origin="cold", rationale="Y")
    lit_pack = LiteraturePack(background="", sources=[])

    [result] = ideation_stage.screen([candidate], lit_pack, ctx)

    assert result.significance.passed is False
    assert result.novelty is None
    assert not any(c["prompt_name"] == "novelty_filter" for c in fake_llm.calls)


def test_screen_estimates_novelty_for_every_candidate_in_a_dry_run(fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path)
    ctx.config.dry_run = True
    candidates = [ThesisCandidate(claim=c, origin="cold", rationale="Y") for c in ("X", "Z")]

    results = ideation_stage.screen(candidates, LiteraturePack(background="", sources=[]), ctx)

    assert all(r.significance.passed is False for r in results)
    assert sum(c["prompt_name"] == "novelty_filter" for c in fake_llm.calls) == 2


def test_run_pools_cold_and_informed_batches(fake_llm, tmp_path):
    fake_llm.responses["ideation_cold"] = THESIS_TEXT
    fake_llm.responses["ideation_informed"] = THESIS_TEXT
    fake_llm.responses["significance_filter"] = "VERDICT: PASS\nREASONING:\nFine."
    fake_llm.responses["novelty_filter"] = "VERDICT: PASS\nREASONING:\nFine."
    ctx = _ctx(fake_llm, tmp_path, n=2)
    lit_pack = LiteraturePack(background="bg", sources=[])

    results = ideation_stage.run("metaethics", lit_pack, ctx)

    assert len(results) == 4
    assert {r.candidate.origin for r in results} == {"cold", "informed"}


def test_generate_reads_a_thesis_whose_keys_are_bold(fake_llm, tmp_path):
    fake_llm.responses["ideation_cold"] = (
        "**THESIS:**\nMoral facts are mind-independent.\n\n**RATIONALE:** It resists one."
    )
    [candidate] = ideation_stage.generate_cold("metaethics", 1, _ctx(fake_llm, tmp_path, n=1))
    assert (candidate.claim, candidate.rationale) == ("Moral facts are mind-independent.", "It resists one.")
