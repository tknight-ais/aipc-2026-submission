from aipc.config import load_config
from aipc.context import Context
from aipc.models import Essay
from aipc.stages import calibrate as calibrate_stage
from aipc.store import Store

SCORE_BLOCKS_HIGH = (
    "CRITERION: argument_quality\nSCORE: 9\nREASONING: Strong.\n\n"
    "CRITERION: rigor\nSCORE: 9\nREASONING: Strong.\n\n"
    "CRITERION: originality\nSCORE: 9\nREASONING: Strong.\n\n"
    "CRITERION: clarity\nSCORE: 9\nREASONING: Strong."
)
SCORE_BLOCKS_LOW = (
    "CRITERION: argument_quality\nSCORE: 3\nREASONING: Weak.\n\n"
    "CRITERION: rigor\nSCORE: 3\nREASONING: Weak.\n\n"
    "CRITERION: originality\nSCORE: 3\nREASONING: Weak.\n\n"
    "CRITERION: clarity\nSCORE: 3\nREASONING: Weak."
)


def _ctx(fake_llm, tmp_path):
    return Context(llm=fake_llm, config=load_config(), run_id="run1", store=Store(runs_dir=tmp_path))


def test_build_graded_set_has_four_cumulative_levels(fake_llm, tmp_path):
    fake_llm.responses["degrade_essay"] = "Damaged essay text."
    ctx = _ctx(fake_llm, tmp_path)
    original = Essay(text="Original essay text.", word_count=3, thesis="X")

    levels = calibrate_stage.build_graded_set(original, ctx)

    assert [name for name, _ in levels] == ["original", "light", "heavy", "severe"]
    assert levels[0][1] is original
    assert levels[1][1].text == "Damaged essay text."
    degrade_calls = [c for c in fake_llm.calls if c["prompt_name"] == "degrade_essay"]
    assert len(degrade_calls) == 3


def test_score_levels_detects_a_correctly_ordered_panel(fake_llm, tmp_path):
    # Every judge call after the first returns the "low" block - the
    # FakeLLM list-cycling means index 0 ("original") sees SCORE_BLOCKS_HIGH's
    # judge_absolute response if we tailor per level instead; simplest here
    # is to make absolute scores depend on essay text via pairwise instead.
    fake_llm.responses["judge_absolute"] = [
        SCORE_BLOCKS_HIGH,
        SCORE_BLOCKS_LOW,
        SCORE_BLOCKS_LOW,
        SCORE_BLOCKS_LOW,
    ] * len(calibrate_stage.judge_stage.MIRROR_PANEL)
    # compare_both_orderings makes two calls per pair: (less_damaged,
    # more_damaged) then (more_damaged, less_damaged). A judge that
    # correctly prefers the less-damaged essay says "A" first (it's in
    # slot A) and "B" second (it's moved to slot B) - alternating,
    # regardless of content, since FakeLLM can't inspect the essays.
    # FakeLLM's list responses don't cycle, they just repeat the last
    # entry once exhausted, so the full 24-call sequence (3 pairs x 4
    # personas x 2 orderings) is spelled out explicitly.
    fake_llm.responses["judge_pairwise"] = ["WINNER: A\nREASONING:\nFine.", "WINNER: B\nREASONING:\nFine."] * 12
    ctx = _ctx(fake_llm, tmp_path)
    levels = [
        ("original", Essay(text="Original.", word_count=1, thesis="X")),
        ("light", Essay(text="Light damage.", word_count=2, thesis="X")),
        ("heavy", Essay(text="Heavy damage.", word_count=2, thesis="X")),
        ("severe", Essay(text="Severe damage.", word_count=2, thesis="X")),
    ]

    results = calibrate_stage.score_levels(levels, ctx)

    assert len(results) == len(calibrate_stage.judge_stage.MIRROR_PANEL)
    for result in results:
        assert result.monotonic is True
        assert set(result.pairwise_results) == {"original_vs_light", "light_vs_heavy", "heavy_vs_severe"}
        # WINNER: A every time means the less-damaged essay (passed as
        # "A"/candidate) won every comparison - a correctly calibrated panel.
        assert all(result.pairwise_results.values())


def test_score_levels_flags_a_non_monotonic_panel(fake_llm, tmp_path):
    # The panel scores "heavy" damage higher than "light" - miscalibrated.
    fake_llm.responses["judge_absolute"] = [SCORE_BLOCKS_HIGH, SCORE_BLOCKS_LOW, SCORE_BLOCKS_HIGH, SCORE_BLOCKS_LOW]
    fake_llm.responses["judge_pairwise"] = "WINNER: A\nREASONING:\nFine."
    ctx = _ctx(fake_llm, tmp_path)
    levels = [
        ("original", Essay(text="Original.", word_count=1, thesis="X")),
        ("light", Essay(text="Light damage.", word_count=2, thesis="X")),
        ("heavy", Essay(text="Heavy damage.", word_count=2, thesis="X")),
        ("severe", Essay(text="Severe damage.", word_count=2, thesis="X")),
    ]

    results = calibrate_stage.score_levels(levels, ctx)

    assert results[0].monotonic is False
