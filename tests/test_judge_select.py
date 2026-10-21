from aipc.config import load_config
from aipc.context import Context
from aipc.models import Essay
from aipc.stages import judge_select
from aipc.replies import parse_winner
from aipc.store import Store

WIN_A = "WINNER: A\nREASONING:\nFine."
WIN_B = "WINNER: B\nREASONING:\nFine."


def _ctx(fake_llm, tmp_path):
    return Context(llm=fake_llm, config=load_config(), run_id="run1", store=Store(runs_dir=tmp_path))


def _levels():
    return [(name, Essay(text=f"{name} essay.", word_count=2, thesis="X")) for name in judge_select.LEVEL_ORDER]


def test_parse_winner_reads_common_formats():
    assert parse_winner("WINNER: A\nREASONING: x") == "A"
    assert parse_winner("**WINNER:** b\nREASONING: x") == "B"
    assert parse_winner("Some preamble.\nWINNER: B") == "B"


def test_parse_winner_returns_none_rather_than_defaulting():
    assert parse_winner("I cannot decide.") is None
    assert parse_winner("WINNER: neither") is None


def test_correct_judge_scores_every_trial(fake_llm, tmp_path):
    # Each trial is better-as-A then better-as-B, so a correct judge alternates.
    fake_llm.responses["judge_pairwise_reasoned"] = [WIN_A, WIN_B] * 9
    result = judge_select.run_judge(_levels(), "judge_qwen", repeats=3, ctx=_ctx(fake_llm, tmp_path))

    assert result.trials == 9
    assert result.correct == 9
    assert result.position_consistent == 9
    assert result.unparsed == 0
    assert result.model == "qwen/qwen3-30b-a3b-instruct-2507"
    assert len(fake_llm.calls) == 18
    assert {c["tier"] for c in fake_llm.calls} == {"judge_qwen"}


def test_position_biased_judge_scores_zero(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise_reasoned"] = WIN_A
    result = judge_select.run_judge(_levels(), "judge_qwen", repeats=3, ctx=_ctx(fake_llm, tmp_path))

    assert result.correct == 0
    assert result.position_consistent == 0


def test_unparsed_verdict_is_neither_correct_nor_consistent(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise_reasoned"] = ["no idea", WIN_B] + [WIN_A, WIN_B] * 2
    result = judge_select.run_judge(_levels(), "judge_qwen", repeats=1, ctx=_ctx(fake_llm, tmp_path))

    assert result.unparsed == 1
    assert result.correct == 2
    assert result.position_consistent == 2


def test_repeats_use_distinct_seeds(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise_reasoned"] = WIN_A
    result = judge_select.run_judge(_levels(), "judge_qwen", repeats=3, ctx=_ctx(fake_llm, tmp_path))

    assert sorted({v.seed for v in result.verdicts}) == [1, 2, 3]


def test_rank_prefers_correct_then_consistent_then_cheap():
    def make(tier, correct, consistent, cost):
        return judge_select.JudgeSelectionResult(
            judge_tier=tier, model=tier, trials=9, correct=correct,
            position_consistent=consistent, unparsed=0, cost_usd=cost, verdicts=[],
        )

    ranked = judge_select.rank([make("a", 5, 9, 0.01), make("b", 7, 7, 0.20), make("c", 7, 7, 0.05)])
    assert [r.judge_tier for r in ranked] == ["c", "b", "a"]


def test_parse_winner_reads_last_line_of_reasoning_first_reply():
    assert parse_winner("REASONING:\nWINNER: A looked likely at first.\nWINNER: B") == "B"


def test_configured_prompt_is_used(fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path)
    ctx.config.judge_prompt = "judge_pairwise_reasoned"
    fake_llm.responses["judge_pairwise_reasoned"] = ["REASONING:\nChecked.\nWINNER: A", "REASONING:\nChecked.\nWINNER: B"] * 3
    result = judge_select.run_judge(_levels(), "judge_qwen", repeats=1, ctx=ctx)

    assert result.correct == 3
    assert result.verdicts[0].reasoning == "Checked."
    assert {c["prompt_name"] for c in fake_llm.calls} == {"judge_pairwise_reasoned"}
