"""The whole component-ladder chain through the real CLI, with only the network
call faked: experiment -> tournament -> blind ratings -> analyze.

The fake judge prefers the longer essay, and only revised essays are
long, so A_zero_shot (never revised) must come out weakest.
"""

import json

from typer.testing import CliRunner

from aipc import llm
from aipc.cli import app
from aipc.openrouter import CompletionResult
from aipc.store import Store
from aipc.viewer import rating

runner = CliRunner()

REPLIES = {
    "You are decomposing": "What is free will?\nDoes determinism rule it out?",
    "You are the validity gate": "VERDICT: PASS\nREASONING: fine",
    "You are selecting the best decomposition": "CHOICE: 1\nREASONING: fine",
    "You are building part of": "CLAIM: c\nSUPPORT: s",
    "You are a philosopher debating": "A debate turn.",
    "You are the Manager judging": "NONE",
    "You are the Manager synthesizing": "CLAIM: c\nSUPPORT: s",
    "You are a Critic": "A real objection.",
    "You are the ledger-fidelity": "DROPPED OBJECTIONS: none\nUNSUPPORTED CLAIMS: none",
    "You are the Drafter": "## Draft heading\n\nShort draft.",
    "You are writing a single-author": "Short draft.",
    "You are the Reviser": "## Revised\n\nA much longer revised essay that answers the objection in full.",
}


def _judge(prompt: str) -> str:
    essay_a = prompt.split("Essay A:\n")[1].split("\n\nEssay B:\n")[0]
    essay_b = prompt.split("\n\nEssay B:\n")[1].split("\n\nBefore deciding")[0]
    assert "#" not in essay_a + essay_b, "formatting must be normalised before judging"
    if len(essay_a) == len(essay_b):
        return "WINNER: A\nREASONING: same length"
    return f"WINNER: {'A' if len(essay_a) > len(essay_b) else 'B'}\nREASONING: longer"


def fake_complete(model, prompt, temperature=1.0, seed=None, system=None, reasoning=True):
    if prompt.startswith("Compare two philosophy essays"):
        text = _judge(prompt)
    else:
        text = next(reply for opening, reply in REPLIES.items() if prompt.startswith(opening))
    return CompletionResult(
        text=text,
        prompt_tokens=100,
        completion_tokens=50,
        served_model=model,
        provider="FakeProvider",
        generation_id="gen-fake",
        billed_usd=None,
    )


def test_ladder_chain_end_to_end(monkeypatch, tmp_path):
    monkeypatch.setattr(llm, "complete", fake_complete)
    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("AIPC_RUNS_DIR", str(runs_dir))
    config_path = tmp_path / "exp.yaml"
    config_path.write_text(
        "name: e2e\nseed: 1\nessays_per_condition: 2\ngenerator_tier: cheap\n"
        "conditions: [A_zero_shot, B_critique, C_structured, D_debate]\n"
        "theses: [{id: t1, claim: Free will is compatible with determinism.}, {id: t2, claim: It does not.}]\n"
    )

    result = runner.invoke(app, ["experiment", "--config", str(config_path)])
    assert result.exit_code == 0, result.output
    assert len(Store(runs_dir).experiment_essays("e2e")) == 16
    db = Store(runs_dir).conn
    runs_without_calls = db.execute(
        "SELECT COUNT(*) FROM runs WHERE run_id NOT IN (SELECT run_id FROM calls)"
    ).fetchone()[0]
    assert runs_without_calls == 0, "every essay run must record its model calls"
    assert db.execute("SELECT COUNT(*) FROM calls WHERE provider = 'FakeProvider' AND response != ''").fetchone()[0] > 0
    assert db.execute("SELECT COUNT(*) FROM runs WHERE git_commit IS NULL").fetchone()[0] == 0

    rerun = runner.invoke(app, ["experiment", "--config", str(config_path)])
    assert rerun.output.count("already done, skipped") == 16

    result = runner.invoke(app, ["tournament", "--experiment", "e2e", "--judge", "judge_gemini", "--per-essay", "4"])
    assert result.exit_code == 0, result.output
    store = Store(runs_dir)
    judgments = store.judgments("e2e")
    assert judgments and all(j.winner_run is not None for j in judgments)

    assert all(j.judge_prompt_hash for j in judgments)

    result = runner.invoke(app, ["draw-anchors", "--experiment", "e2e"])
    assert result.exit_code == 0, result.output
    assert rating.anchors(store, "e2e")
    while (pending := rating.next_unrated(store, "e2e")) is not None:
        rating.record(store, "e2e", pending[0], "first")
    assert len(store.human_judgments("e2e")) == len(rating.anchors(store, "e2e"))
    assert runner.invoke(app, ["draw-anchors", "--experiment", "e2e"]).exit_code != 0, "anchors must stay frozen"

    result = runner.invoke(app, ["analyze", "--experiment", "e2e"])
    assert result.exit_code == 0, result.output
    report = json.loads(next(runs_dir.glob("*_analyze_e2e_*/analysis.json")).read_text())
    pooled = report["strengths"]["pooled"]
    assert pooled["A_zero_shot"]["log_strength"] == 0.0
    assert all(pooled[c]["log_strength"] > 0 for c in ["B_critique", "C_structured", "D_debate"])
    assert report["pairs"] == len(judgments) // 2

    result = runner.invoke(app, ["tournament", "--experiment", "e2e", "--judge", "judge_qwen", "--per-essay", "4"])
    assert result.exit_code == 0, result.output
    assert runner.invoke(app, ["analyze", "--experiment", "e2e"]).exit_code != 0, "two judges must not be pooled"
    result = runner.invoke(app, ["analyze", "--experiment", "e2e", "--judge", "judge_qwen"])
    assert result.exit_code == 0, result.output
    assert "judge qwen/qwen3-30b-a3b-instruct-2507" in result.output
