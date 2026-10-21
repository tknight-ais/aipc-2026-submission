import json

import pytest

from aipc.config import load_config
from aipc.context import Context
from aipc.stages import experiment
from aipc.store import Store

THESIS = "Free will is compatible with determinism."


def _ctx(fake_llm, tmp_path, condition="A_zero_shot"):
    config = experiment.condition_config(load_config(), condition, "cheap", seed=7)
    store = Store(runs_dir=tmp_path)
    store.start_run("run1", "label1", "hash", variant=condition, started_at="t")
    return Context(llm=fake_llm, config=config, run_id="run1", store=store)


def _artifacts(tmp_path):
    return sorted(p.stem for p in (tmp_path / "label1").glob("*.json"))


def test_condition_config_sets_strategy_rounds_and_seed():
    base = load_config()
    c = experiment.condition_config(base, "C_structured", "cheap", seed=4)
    d = experiment.condition_config(base, "D_debate", "cheap", seed=4)
    assert (c.pipeline.strategy, d.pipeline.strategy) == ("flat", "debate_room")
    assert c.pipeline.max_revision_rounds == d.pipeline.max_revision_rounds == 1
    assert c.defaults.seed == 4
    assert base.pipeline.max_revision_rounds == 3


def test_condition_config_sets_generator_tier_and_frontier_for_e():
    base = load_config()
    assert experiment.condition_config(base, "C_structured", "frontier", seed=1).defaults.tier == "frontier"
    assert experiment.condition_config(base, "A_zero_shot", "cheap", seed=1).defaults.tier == "cheap"
    assert experiment.condition_config(base, "E_frontier_zero_shot", "cheap", seed=1).defaults.tier == "frontier"


def test_structured_condition_uses_generator_tier_throughout(fake_llm, tmp_path):
    ctx = _ctx(fake_llm, tmp_path, "C_structured")
    ctx.config = experiment.condition_config(load_config(), "C_structured", "standard", seed=7)
    fake_llm.responses["decompose"] = "- Q1?\n- Q2?"
    fake_llm.responses["validity_gate"] = "VERDICT: PASS\nREASONING: fine"
    fake_llm.responses["decomposition_select"] = "CHOICE: 1\nREASONING: fine"
    experiment.write_essay("C_structured", THESIS, ctx)

    assert {c["tier"] for c in fake_llm.calls} == {"standard"}


def test_zero_shot_is_one_call_with_the_thesis(fake_llm, tmp_path):
    fake_llm.responses["zero_shot_thesis"] = "An essay."
    essay, path = experiment.write_essay("A_zero_shot", THESIS, _ctx(fake_llm, tmp_path))

    assert [c["prompt_name"] for c in fake_llm.calls] == ["zero_shot_thesis"]
    assert THESIS in fake_llm.calls[0]["prompt"]
    assert essay.thesis == THESIS
    assert json.loads(path.read_text())["text"] == "An essay."


def test_critique_revises_once_without_a_ledger(fake_llm, tmp_path):
    fake_llm.responses.update({"zero_shot_thesis": "Draft.", "critic": "A real objection.", "revise_free": "Revised."})
    essay, _ = experiment.write_essay("B_critique", THESIS, _ctx(fake_llm, tmp_path, "B_critique"))

    names = [c["prompt_name"] for c in fake_llm.calls]
    assert names == ["zero_shot_thesis", "critic", "critic", "critic", "revise_free"]
    assert "ledger" not in fake_llm.calls[-1]["prompt"].lower()
    assert essay.text == "Revised."
    assert essay.rounds_used == 1
    assert _artifacts(tmp_path) == ["draft", "essay", "round1"]


def test_critique_keeps_draft_when_no_critic_objects(fake_llm, tmp_path):
    fake_llm.responses.update({"zero_shot_thesis": "Draft.", "critic": "NONE"})
    essay, _ = experiment.write_essay("B_critique", THESIS, _ctx(fake_llm, tmp_path, "B_critique"))

    assert essay.text == "Draft."
    assert "revise_free" not in [c["prompt_name"] for c in fake_llm.calls]


@pytest.mark.parametrize("condition, ledger_prompt", [("C_structured", "ledger_flat"), ("D_debate", "debater_for")])
def test_structured_conditions_use_their_strategy_and_one_round(fake_llm, tmp_path, condition, ledger_prompt):
    fake_llm.responses.update({
        "decompose": "Q1?\nQ2?",
        "validity_gate": "VERDICT: PASS\nREASONING: ok",
        "decomposition_select": "CHOICE: 1\nREASONING: ok",
        "debate_converge": "NONE",
        "draft": "Draft.",
        "critic": "An objection.",
        "revise": "Revised.",
    })
    essay, _ = experiment.write_essay(condition, THESIS, _ctx(fake_llm, tmp_path, condition))

    names = [c["prompt_name"] for c in fake_llm.calls]
    assert ledger_prompt in names
    assert names.count("revise") == 1
    assert essay.text == "Revised."
    assert {"decomposition", "ledger", "round1", "fidelity", "essay"} <= set(_artifacts(tmp_path))


def test_load_experiment_rejects_empty_theses_and_unknown_conditions(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text("name: x\nseed: 1\nessays_per_condition: 1\ngenerator_tier: cheap\nconditions: [A_zero_shot]\ntheses: []\n")
    with pytest.raises(ValueError, match="no theses"):
        experiment.load_experiment(path)
    path.write_text("name: x\nseed: 1\nessays_per_condition: 1\ngenerator_tier: cheap\nconditions: [Z]\ntheses: [{id: t1, claim: c}]\n")
    with pytest.raises(ValueError, match="unknown conditions"):
        experiment.load_experiment(path)


def test_every_writing_prompt_carries_the_same_requirements(fake_llm, tmp_path, sample_ledger):
    from aipc.stages import draft

    ctx = _ctx(fake_llm, tmp_path)
    requirements = draft.essay_requirements(ctx)
    essay = experiment.write_zero_shot(THESIS, ctx)
    draft.draft_essay(sample_ledger, ctx)
    draft.revise(essay, sample_ledger, ["An objection."], ctx)
    draft.revise_free(essay, ["An objection."], ctx)

    prompts = {c["prompt_name"]: c["prompt"] for c in fake_llm.calls}
    assert set(prompts) == {"zero_shot_thesis", "draft", "revise", "revise_free"}
    for name, prompt in prompts.items():
        assert requirements in prompt, name
        assert "sources" not in prompt.replace(requirements, ""), f"{name} has its own source rule"
