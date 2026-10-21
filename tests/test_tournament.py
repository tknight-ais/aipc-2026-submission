from collections import Counter

from aipc.config import load_config
from aipc.context import Context
from aipc.blind import normalise
from aipc.stages import tournament
from aipc.store import ExperimentEssay, Store

WIN_A = "WINNER: A\nREASONING:\nFine."


def _essays(theses=("t1", "t2"), conditions=("A", "B", "C", "D"), per=3):
    return [
        ExperimentEssay(thesis_id=t, condition=c, essay_index=i, run_id=f"{t}-{c}-{i}", essay_path="x")
        for t in theses for c in conditions for i in range(per)
    ]


def test_normalise_strips_markup_and_plain_headings_but_keeps_prose():
    text = (
        "# A Title About Free Will\n\n"
        "## Introduction\nThe first paragraph argues **boldly** for the thesis.\n\n"
        "Objection One: The Regress\n\n"
        "- First listed point.\n- Second listed point.\n\n"
        "A short sentence."
    )
    assert normalise(text) == (
        "The first paragraph argues boldly for the thesis.\n\n"
        "First listed point. Second listed point.\n\n"
        "A short sentence."
    )


def test_normalise_drops_section_break_lines():
    for rule in ("***", "___", "* * *", "---"):
        assert normalise(f"First paragraph.\n\n{rule}\n\nSecond paragraph.") == "First paragraph.\n\nSecond paragraph."


def test_normalise_leaves_plain_prose_unchanged():
    text = "First paragraph of argument, which ends properly.\n\nSecond paragraph, also ending properly."
    assert normalise(text) == text


def test_schedule_pairs_within_thesis_across_conditions_with_capped_degree():
    essays = _essays()
    pairs = tournament.schedule_pairs(essays, per_essay=6, seed=1)

    by_run = {e.run_id: e for e in essays}
    for p in pairs:
        a, b = by_run[p.essay_a_run], by_run[p.essay_b_run]
        assert a.thesis_id == b.thesis_id == p.thesis_id
        assert a.condition != b.condition
    degree = Counter(r for p in pairs for r in (p.essay_a_run, p.essay_b_run))
    assert max(degree.values()) <= 6
    assert len({(p.essay_a_run, p.essay_b_run) for p in pairs}) == len(pairs)
    assert pairs == tournament.schedule_pairs(essays, per_essay=6, seed=1)
    assert pairs != tournament.schedule_pairs(essays, per_essay=6, seed=2)


def _ctx(fake_llm, tmp_path, dry_run=False):
    config = load_config()
    config.dry_run = dry_run
    return Context(llm=fake_llm, config=config, run_id="tour1", store=Store(runs_dir=tmp_path))


def _schedule():
    pair = tournament.Pair(thesis_id="t1", essay_a_run="ea", essay_b_run="eb")
    return tournament.Schedule(experiment="x", seed=1, per_essay=1, pairs=[pair])


def test_run_judges_both_orderings_and_maps_positions_to_runs(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise_reasoned"] = WIN_A
    ctx = _ctx(fake_llm, tmp_path)
    texts = {"ea": "Essay a text.", "eb": "Essay b text."}

    judgments = tournament.run("x", _schedule(), texts, "judge_gemini", ctx)

    assert [(j.shown_first_run, j.winner_run) for j in judgments] == [("ea", "ea"), ("eb", "eb")]
    assert fake_llm.calls[0]["prompt"].index("Essay a text.") < fake_llm.calls[0]["prompt"].index("Essay b text.")
    assert fake_llm.calls[1]["prompt"].index("Essay b text.") < fake_llm.calls[1]["prompt"].index("Essay a text.")
    assert {j.judge_model for j in judgments} == {"google/gemini-2.5-flash"}
    assert len(ctx.store.judgments("x")) == 2


def test_run_resumes_without_rejudging(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise_reasoned"] = WIN_A
    ctx = _ctx(fake_llm, tmp_path)
    texts = {"ea": "a", "eb": "b"}
    tournament.run("x", _schedule(), texts, "judge_gemini", ctx)

    assert tournament.run("x", _schedule(), texts, "judge_gemini", ctx) == []
    assert len(tournament.run("x", _schedule(), texts, "judge_qwen", ctx)) == 2


def test_run_rejudges_when_the_judge_prompt_changes(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise_reasoned"] = WIN_A
    fake_llm.responses["judge_pairwise_neutral"] = WIN_A
    ctx = _ctx(fake_llm, tmp_path)
    texts = {"ea": "a", "eb": "b"}
    tournament.run("x", _schedule(), texts, "judge_gemini", ctx)

    ctx.config.judge_prompt = "judge_pairwise_neutral"
    judgments = tournament.run("x", _schedule(), texts, "judge_gemini", ctx)
    assert len(judgments) == 2
    assert len({j.judge_prompt_hash for j in ctx.store.judgments("x")}) == 2


def test_run_saves_nothing_on_dry_run(fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path, dry_run=True)
    judgments = tournament.run("x", _schedule(), {"ea": "a", "eb": "b"}, "judge_gemini", ctx)

    assert len(judgments) == 2
    assert all(j.winner_run is None for j in judgments)
    assert ctx.store.judgments("x") == []
