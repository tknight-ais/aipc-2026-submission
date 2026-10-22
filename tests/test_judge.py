from aipc.config import load_config
from aipc.context import Context
from aipc.models import Essay, Persona
from aipc.stages import judge as judge_stage
from aipc.store import Store

PERSONA = Persona(name="test_judge", description="a test judge")

SCORE_BLOCKS = (
    "CRITERION: argument_quality\nSCORE: 7\nREASONING: Solid but not airtight.\n\n"
    "CRITERION: rigor\nSCORE: 6\nREASONING: Some gaps in the inference.\n\n"
    "CRITERION: originality\nSCORE: 8\nREASONING: A fresh combination of positions.\n\n"
    "CRITERION: clarity\nSCORE: 9\nREASONING: Easy to follow throughout."
)


def _ctx(fake_llm, tmp_path):
    return Context(llm=fake_llm, config=load_config(), run_id="run1", store=Store(runs_dir=tmp_path))


def test_score_absolute_parses_all_four_criteria(fake_llm, tmp_path):
    fake_llm.responses["judge_absolute"] = SCORE_BLOCKS
    ctx = _ctx(fake_llm, tmp_path)
    essay = Essay(text="An essay.", word_count=2, thesis="X")

    scores = judge_stage.score_absolute(essay, essay_id="candidate", persona=PERSONA, panel_type="mirror", ctx=ctx)

    assert len(scores) == 4
    assert {s.criterion for s in scores} == set(judge_stage.CRITERIA)
    rigor = next(s for s in scores if s.criterion == "rigor")
    assert rigor.score == 6.0
    assert rigor.essay_id == "candidate"
    assert rigor.panel_type == "mirror"


def test_compare_pairwise_parses_winner_a(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise"] = "WINNER: A\nREASONING:\nA engages the objection more directly."
    ctx = _ctx(fake_llm, tmp_path)
    essay_a = Essay(text="Essay A.", word_count=2, thesis="X")
    essay_b = Essay(text="Essay B.", word_count=2, thesis="X")

    winner, reasoning = judge_stage.compare_pairwise(essay_a, essay_b, PERSONA, ctx)

    assert winner == "A"
    assert reasoning == "A engages the objection more directly."


def test_compare_both_orderings_consistent_when_same_essay_wins_each_time(fake_llm, tmp_path):
    # Candidate is "A" in the first call and "B" in the second - a
    # consistent preference for the candidate looks like A then B.
    fake_llm.responses["judge_pairwise"] = [
        "WINNER: A\nREASONING:\nFirst order.",
        "WINNER: B\nREASONING:\nSecond order.",
    ]
    ctx = _ctx(fake_llm, tmp_path)
    candidate = Essay(text="Candidate essay.", word_count=2, thesis="X")
    reference = Essay(text="Reference essay.", word_count=2, thesis="X")

    result = judge_stage.compare_both_orderings(candidate, reference, PERSONA, ctx)

    assert result["candidate_wins"] is True
    assert result["consistent"] is True


def test_compare_both_orderings_inconsistent_when_orderings_disagree(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise"] = [
        "WINNER: A\nREASONING:\nFirst order.",
        "WINNER: A\nREASONING:\nSecond order.",
    ]
    ctx = _ctx(fake_llm, tmp_path)
    candidate = Essay(text="Candidate essay.", word_count=2, thesis="X")
    reference = Essay(text="Reference essay.", word_count=2, thesis="X")

    result = judge_stage.compare_both_orderings(candidate, reference, PERSONA, ctx)

    assert result["consistent"] is False
    assert result["candidate_wins"] is False


def test_score_absolute_drops_unreadable_scores(fake_llm, tmp_path):
    fake_llm.responses["judge_absolute"] = "CRITERION: rigor\nSCORE: high\n\nCRITERION: clarity\nSCORE: 7.5"
    scores = judge_stage.score_absolute(
        Essay(text="E.", word_count=1, thesis="X"), "e", PERSONA, "mirror", _ctx(fake_llm, tmp_path)
    )
    assert [(s.criterion, s.score) for s in scores] == [("clarity", 7.5)]


def test_unreadable_pairwise_verdict_is_not_read_as_a(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise"] = "I cannot choose."
    ctx = _ctx(fake_llm, tmp_path)
    candidate = Essay(text="Candidate essay.", word_count=2, thesis="X")
    reference = Essay(text="Reference essay.", word_count=2, thesis="X")

    winner, _ = judge_stage.compare_pairwise(candidate, reference, PERSONA, ctx)
    result = judge_stage.compare_both_orderings(candidate, reference, PERSONA, ctx)

    assert winner is None
    assert result["consistent"] is False
    assert result["candidate_wins"] is False
