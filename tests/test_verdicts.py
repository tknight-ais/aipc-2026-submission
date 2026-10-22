from aipc.models import Essay
from aipc.stages import verdicts
from aipc.store import ExperimentEssay, Judgment


def _judgment(shown_first, winner, reasoning="Long reasoning. A is better."):
    return Judgment(
        experiment="x",
        tournament_run_id="t",
        thesis_id="t1",
        essay_a_run="a",
        essay_b_run="b",
        judge_model="judge/model",
        shown_first_run=shown_first,
        winner_run=winner,
        reasoning=reasoning,
    )


def test_report_summarises_and_lists_each_pair():
    essays = [
        ExperimentEssay(thesis_id="t1", condition="A_zero_shot", essay_index=0, run_id="a", essay_path="pa"),
        ExperimentEssay(thesis_id="t1", condition="C_structured", essay_index=0, run_id="b", essay_path="pb"),
    ]
    texts = {"a": Essay.from_text("Title A\n\nBody.", thesis="t"), "b": Essay.from_text("Title B\n\nBody.", thesis="t")}
    report = verdicts.build("x", [_judgment("a", "a"), _judgment("b", "a")], essays, texts)

    assert "Same winner in both orderings: 1/1 (100%)" in report
    assert "Picked the essay shown first: 1/2 (50%)" in report
    assert "A_zero_shot vs C_structured (t1): A_zero_shot" in report
    assert "Title A" in report and "A is better." in report
