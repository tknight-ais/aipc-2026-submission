import pytest

from aipc.config import load_config
from aipc.context import Context
from aipc.models import Decomposition
from aipc.stages import decomposition as decomposition_stage
from aipc.store import Store

THESIS = "Compatibilism survives the consequence argument."


def _ctx(fake_llm, tmp_path):
    return Context(llm=fake_llm, config=load_config(), run_id="run1", store=Store(runs_dir=tmp_path))


def test_generate_candidates_parses_one_per_line(fake_llm, tmp_path):
    fake_llm.responses["decompose"] = "Is genealogy relevant to justification?\nDoes the perceptual analogy hold?"
    ctx = _ctx(fake_llm, tmp_path)

    candidates = decomposition_stage.generate_candidates(THESIS, 2, ctx)

    assert len(candidates) == 2
    assert candidates[0].sub_questions == [
        "Is genealogy relevant to justification?",
        "Does the perceptual analogy hold?",
    ]
    assert candidates[0].thesis == THESIS


def test_check_validity_parses_fail(fake_llm, tmp_path):
    fake_llm.responses["validity_gate"] = "VERDICT: FAIL\nREASONING:\nThe two questions overlap heavily."
    ctx = _ctx(fake_llm, tmp_path)
    candidate = Decomposition(thesis=THESIS, sub_questions=["Q1?", "Q2?"])

    verdict = decomposition_stage.check_validity(candidate, ctx)

    assert verdict.passed is False
    assert verdict.reasoning == "The two questions overlap heavily."


def test_select_parses_choice_number(fake_llm, tmp_path):
    fake_llm.responses["decomposition_select"] = "CHOICE: 2\nREASONING:\nIt builds the strongest case."
    ctx = _ctx(fake_llm, tmp_path)
    survivors = [
        Decomposition(thesis=THESIS, sub_questions=["Q1?"]),
        Decomposition(thesis=THESIS, sub_questions=["Q2?"]),
    ]

    selection = decomposition_stage.select(THESIS, survivors, ctx)

    assert selection.chosen_index == 1
    assert selection.reasoning == "It builds the strongest case."


def test_run_raises_when_nothing_passes_validity_gate(fake_llm, tmp_path):
    fake_llm.responses["decompose"] = "Q1?\nQ2?"
    fake_llm.responses["validity_gate"] = "VERDICT: FAIL\nREASONING:\nBroken."
    ctx = _ctx(fake_llm, tmp_path)

    with pytest.raises(RuntimeError):
        decomposition_stage.run(THESIS, 2, ctx)


def test_run_selects_among_survivors(fake_llm, tmp_path):
    fake_llm.responses["decompose"] = "Q1?\nQ2?"
    fake_llm.responses["validity_gate"] = "VERDICT: PASS\nREASONING:\nFine."
    fake_llm.responses["decomposition_select"] = "CHOICE: 1\nREASONING:\nBest shape."
    ctx = _ctx(fake_llm, tmp_path)

    screened, chosen = decomposition_stage.run(THESIS, 2, ctx)

    assert len(screened) == 2
    assert all(s.validity.passed for s in screened)
    assert chosen.sub_questions == ["Q1?", "Q2?"]


def test_select_reads_the_first_number_after_choice(fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path)
    survivors = [Decomposition(thesis="T", sub_questions=[f"Q{i}"]) for i in range(4)]
    fake_llm.responses["decomposition_select"] = "CHOICE: 2 (of 4)\nREASONING: candidate 3 is weaker."
    assert decomposition_stage.select("T", survivors, ctx).chosen_index == 1


def test_select_raises_on_an_out_of_range_choice(fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path)
    survivors = [Decomposition(thesis="T", sub_questions=["Q"])]
    fake_llm.responses["decomposition_select"] = "CHOICE: 0\nREASONING: x"
    with pytest.raises(ValueError):
        decomposition_stage.select("T", survivors, ctx)


def test_dry_run_plans_for_the_most_sub_questions(fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path)
    ctx.config.dry_run = True
    _, chosen = decomposition_stage.run(THESIS, 2, ctx)
    assert len(chosen.sub_questions) == decomposition_stage.MAX_SUB_QUESTIONS


def test_generate_candidates_drops_numbering_and_a_preamble(fake_llm, tmp_path):
    fake_llm.responses["decompose"] = (
        "Here is one decomposition:\n\n1. Is genealogy relevant?\n2. Does the analogy hold?"
    )
    [candidate] = decomposition_stage.generate_candidates(THESIS, 1, _ctx(fake_llm, tmp_path))
    assert candidate.sub_questions == ["Is genealogy relevant?", "Does the analogy hold?"]
