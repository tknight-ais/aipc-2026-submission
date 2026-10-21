from aipc.config import load_config
from aipc.context import Context
from aipc.stages import select as select_stage
from aipc.store import Store
from conftest import FakeLLM


class ContentJudge(FakeLLM):
    """Prefers the essay with more occurrences of 'strong'; ties go to A."""

    def call(self, tier, prompt, system=None, temperature=None, seed=None, prompt_name=None):
        essay_a = prompt.split("Essay A:\n")[1].split("\n\nEssay B:\n")[0]
        essay_b = prompt.split("\n\nEssay B:\n")[1].split("\n\nBefore deciding")[0]
        winner = "B" if essay_b.count("strong") > essay_a.count("strong") else "A"
        self.responses[prompt_name] = f"REASONING:\nChecked.\nWINNER: {winner}"
        return super().call(tier, prompt, system, temperature, seed, prompt_name)


def _ctx(llm, tmp_path):
    return Context(llm=llm, config=load_config(), run_id="run1", store=Store(runs_dir=tmp_path))


def test_select_ranks_by_consistent_wins(tmp_path):
    llm = ContentJudge()
    candidates = {"weak": "A weak essay.", "best": "A strong strong essay.", "mid": "A strong essay."}
    result = select_stage.run(candidates, "judge_gemini", repeats=2, ctx=_ctx(llm, tmp_path))

    assert [r.name for r in result.ranking] == ["best", "mid", "weak"]
    assert (result.winner, result.tied_at_top) == ("best", False)
    assert result.ranking[0].wins == 4 and result.ranking[0].trials == 4
    assert len(llm.calls) == 3 * 2 * 2
    assert {c["tier"] for c in llm.calls} == {"judge_gemini"}


def test_position_biased_judge_gives_only_ties(fake_llm, tmp_path):
    fake_llm.responses["judge_pairwise_reasoned"] = "REASONING:\nx\nWINNER: A"
    result = select_stage.run({"x": "One.", "y": "Two."}, "judge_gemini", repeats=1, ctx=_ctx(fake_llm, tmp_path))

    assert result.trials[0].winner is None
    assert result.tied_at_top
    assert all(r.wins == r.losses == 0 for r in result.ranking)


def test_formatting_is_stripped_before_judging(tmp_path):
    llm = ContentJudge()
    select_stage.run({"x": "# Title\n\nA strong essay.", "y": "Plain."}, "judge_gemini", repeats=1, ctx=_ctx(llm, tmp_path))

    assert all("# Title" not in c["prompt"] for c in llm.calls)
