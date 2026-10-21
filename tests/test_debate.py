from aipc.config import load_config
from aipc.context import Context
from aipc.models import Decomposition
from aipc.stages import debate as debate_stage
from aipc.store import Store

DECOMPOSITION = Decomposition(
    thesis="Compatibilism survives the consequence argument.",
    sub_questions=["Does genealogy entail unjustification?", "Does the perceptual analogy hold?"],
)

LEDGER_BLOCK = (
    "CLAIM: Genealogy and justification are distinct.\n"
    "SUPPORT: Frankfurt (1971) and responses.\n"
    "OBJECTION: Fitness-shaping undermines reliability regardless.\n"
    "REPLY: The perceptual analogy blocks that inference.\n"
    "REPLY_TYPE: decisive\n\n"
    "CONCESSION: Reliability remains an open empirical question."
)


def _ctx(fake_llm, tmp_path, strategy="flat", debate_max_rounds=2):
    config = load_config()
    config.pipeline.strategy = strategy
    config.pipeline.debate_max_rounds = debate_max_rounds
    return Context(llm=fake_llm, config=config, run_id="run1", store=Store(runs_dir=tmp_path))


def test_run_flat_produces_one_claim_block_per_sub_question(fake_llm, tmp_path):
    fake_llm.responses["ledger_flat"] = LEDGER_BLOCK
    ctx = _ctx(fake_llm, tmp_path, strategy="flat")

    ledger, transcripts = debate_stage.run(DECOMPOSITION, ctx)

    assert len(ledger.claims) == 2
    assert ledger.claims[0].id == "sq0_c1"
    assert ledger.claims[1].id == "sq1_c1"
    assert len(ledger.objections) == 2
    assert ledger.objections[0].reply_type == "decisive"
    assert ledger.concessions == [
        "Reliability remains an open empirical question.",
        "Reliability remains an open empirical question.",
    ]
    assert len(transcripts) == 2
    assert all(t.converged for t in transcripts)


def test_run_debate_room_stops_on_convergence(fake_llm, tmp_path):
    fake_llm.responses["debater_for"] = "The analogy to perception holds."
    fake_llm.responses["debater_against"] = "Fitness-shaping still undermines reliability."
    fake_llm.responses["debate_converge"] = "NONE"
    fake_llm.responses["debate_synthesize"] = LEDGER_BLOCK
    ctx = _ctx(fake_llm, tmp_path, strategy="debate_room", debate_max_rounds=3)

    ledger, transcripts = debate_stage.run(DECOMPOSITION, ctx)

    assert len(transcripts) == 2
    for transcript in transcripts:
        assert transcript.converged is True
        assert transcript.rounds_used == 2
        assert len(transcript.turns) == 4
    assert len(ledger.claims) == 2
    assert sum(c["prompt_name"] == "debate_converge" for c in fake_llm.calls) == len(transcripts)


def test_run_debate_room_hits_hard_cap_when_never_converging(fake_llm, tmp_path):
    fake_llm.responses["debater_for"] = "A new point for."
    fake_llm.responses["debater_against"] = "A new point against."
    fake_llm.responses["debate_converge"] = "There is a new objection on the table."
    fake_llm.responses["debate_synthesize"] = LEDGER_BLOCK
    ctx = _ctx(fake_llm, tmp_path, strategy="debate_room", debate_max_rounds=2)

    _, transcripts = debate_stage.run(DECOMPOSITION, ctx)

    for transcript in transcripts:
        assert transcript.converged is False
        assert transcript.rounds_used == 2
        assert len(transcript.turns) == 4


def test_debaters_only_see_published_turns(fake_llm, tmp_path):
    fake_llm.responses["debater_for"] = "For argument."
    fake_llm.responses["debater_against"] = "Against argument."
    fake_llm.responses["debate_converge"] = "NONE"
    fake_llm.responses["debate_synthesize"] = LEDGER_BLOCK
    ctx = _ctx(fake_llm, tmp_path, strategy="debate_room", debate_max_rounds=1)

    debate_stage.run(DECOMPOSITION, ctx)

    against_call = next(c for c in fake_llm.calls if c["prompt_name"] == "debater_against")
    assert "For argument." in against_call["prompt"]
