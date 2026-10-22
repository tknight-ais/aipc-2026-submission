import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from aipc.cli import app

runner = CliRunner()


def test_help_lists_command_group():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "baseline" in result.output


def test_baseline_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["baseline", "--topic", "metaethics", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def test_draft_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["draft", "--ledger", "tests/fixtures/ledger.json", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def test_ideate_dry_run_needs_no_api_key_or_network(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["ideate", "--topic", "metaethics", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def test_decompose_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(
        app, ["decompose", "--thesis", "Compatibilism survives the consequence argument.", "--dry-run"]
    )
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def test_ledger_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["ledger", "--decomposition", "tests/fixtures/decomposition.json", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def test_generate_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["generate", "--topic", "metaethics", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def test_select_dry_run_over_recorded_essays(monkeypatch, tmp_path):
    from aipc.models import Essay
    from aipc.store import Store

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    store = Store(tmp_path / "runs")
    for thesis in ["t1", "t2", "t3"]:
        path = tmp_path / f"{thesis}.json"
        path.write_text(Essay(text="Body.", word_count=1, thesis=thesis).model_dump_json())
        store.save_experiment_essay("contest", thesis, "C_structured", 0, f"run-{thesis}", path)

    result = runner.invoke(app, ["select", "--experiment", "contest", "--judge", "judge_gemini", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "trials=4" in result.output
    assert "winner essay:" in result.output
    assert "estimated cost" in result.output


def test_select_requires_at_least_two_essays(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(app, ["select", "--essay", "tests/fixtures/essay_candidate.json", "--dry-run"])
    assert result.exit_code != 0


def test_calibrate_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(
        app, ["calibrate", "--essay", "tests/fixtures/essay_candidate.json", "--dry-run", "--max-cost", "50"]
    )
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def test_evaluate_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path))
    result = runner.invoke(
        app,
        [
            "evaluate",
            "--essay",
            "tests/fixtures/essay_candidate.json",
            "--reference",
            "tests/fixtures/essay_reference.json",
            "--dry-run",
        ],
    )
    assert result.exit_code == 0, result.output
    assert "estimated cost" in result.output
    assert (tmp_path / "runs.db").exists()


def _levels_dir(tmp_path):
    levels_dir = tmp_path / "levels"
    levels_dir.mkdir()
    essay_json = Path("tests/fixtures/essay_candidate.json").read_text()
    for name in ["original", "light", "heavy", "severe"]:
        (levels_dir / f"level_{name}.json").write_text(essay_json)
    return levels_dir


def test_judge_select_dry_run_needs_no_api_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(app, ["judge-select", "--levels-dir", str(_levels_dir(tmp_path)), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "[judge_qwen]" in result.output
    assert "[judge_gemini]" in result.output
    assert "estimated cost" in result.output


def test_judge_select_rejects_unknown_judge(monkeypatch, tmp_path):
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(
        app, ["judge-select", "--levels-dir", str(_levels_dir(tmp_path)), "--judge", "nope", "--dry-run"]
    )
    assert result.exit_code != 0
    assert not (tmp_path / "runs" / "runs.db").exists()


def _experiment_yaml(tmp_path):
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: t\nseed: 1\nessays_per_condition: 1\ngenerator_tier: cheap\n"
        "conditions: [A_zero_shot, B_critique, C_structured, D_debate]\n"
        "theses: [{id: t1, claim: Free will is compatible with determinism.}]\n"
    )
    return path


def test_experiment_dry_run_covers_every_condition(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(app, ["experiment", "--config", str(_experiment_yaml(tmp_path)), "--dry-run"])
    assert result.exit_code == 0, result.output
    for condition in ["A_zero_shot", "B_critique", "C_structured", "D_debate"]:
        assert f"t1/{condition}/0: words=" in result.output
    assert "estimated total cost" in result.output


def test_experiment_writes_one_essay_per_thesis_and_condition_in_each_pass(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    path = tmp_path / "exp.yaml"
    path.write_text(
        "name: t\nseed: 1\nessays_per_condition: 2\ngenerator_tier: cheap\nconditions: [A_zero_shot, B_critique]\n"
        "theses: [{id: t1, claim: One.}, {id: t2, claim: Two.}]\n"
    )
    result = runner.invoke(app, ["experiment", "--config", str(path), "--dry-run"])
    assert result.exit_code == 0, result.output
    written = [line.split(":")[0] for line in result.output.splitlines() if ": words=" in line]
    assert written == [
        "t1/A_zero_shot/0",
        "t1/B_critique/0",
        "t2/A_zero_shot/0",
        "t2/B_critique/0",
        "t1/A_zero_shot/1",
        "t1/B_critique/1",
        "t2/A_zero_shot/1",
        "t2/B_critique/1",
    ]


def test_experiment_skips_essays_already_recorded(monkeypatch, tmp_path):
    from aipc.store import Store

    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    Store(tmp_path / "runs").save_experiment_essay("t", "t1", "A_zero_shot", 0, "r0", tmp_path / "e.json")
    result = runner.invoke(app, ["experiment", "--config", str(_experiment_yaml(tmp_path)), "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "t1/A_zero_shot/0: already done, skipped" in result.output


@pytest.mark.skipif(
    not Path("configs/experiments/poc.yaml").exists(), reason="experiment configs are private until judging ends"
)
def test_shipped_poc_config_has_the_expected_design():
    from aipc.stages.experiment import load_experiment

    exp = load_experiment(Path("configs/experiments/poc.yaml"))
    assert [t.id for t in exp.theses] == ["t1", "t2"]
    assert exp.conditions == ["A_zero_shot", "B_critique", "C_structured", "D_debate"]
    assert exp.generator_tier == "cheap"


def test_draw_theses_prints_yaml_the_experiment_config_accepts(tmp_path):
    import yaml

    from aipc.models import FilterVerdict, ScreenedThesis, ThesisCandidate
    from aipc.stages.experiment import ExperimentConfig

    run_dir = tmp_path / "ideate"
    run_dir.mkdir()
    for i, claim in enumerate(
        [
            'Free will is "compatible" with determinism: yes.',
            "Consciousness is not computational.",
            "Free will is compatible with determinism.",
        ]
    ):
        passed = FilterVerdict(passed=True, reasoning="")
        screened = ScreenedThesis(
            candidate=ThesisCandidate(claim=claim, origin="cold", rationale=""), significance=passed, novelty=passed
        )
        (run_dir / f"candidate{i}.json").write_text(screened.model_dump_json())

    result = runner.invoke(app, ["draw-theses", "--ideation-run", str(run_dir)])
    assert result.exit_code == 0, result.output
    parsed = yaml.safe_load(result.output)
    config = ExperimentConfig(
        name="x", seed=1, essays_per_condition=1, generator_tier="cheap", conditions=["A_zero_shot"], **parsed
    )
    assert len(config.theses) == 2


def test_tournament_dry_run_over_recorded_essays(monkeypatch, tmp_path):
    from aipc.models import Essay
    from aipc.store import Store

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    store = Store(tmp_path / "runs")
    for condition in ["A_zero_shot", "B_critique"]:
        for i in range(2):
            path = tmp_path / f"{condition}{i}.json"
            path.write_text(Essay(text="## Heading\n\nBody.", word_count=2, thesis="T").model_dump_json())
            store.save_experiment_essay("poc", "t1", condition, i, f"{condition}{i}", path)

    result = runner.invoke(app, ["tournament", "--experiment", "poc", "--judge", "judge_gemini", "--dry-run"])
    assert result.exit_code == 0, result.output
    assert "essays: 4  pairs: 4  new judgments: 8" in result.output
    assert "estimated cost" in result.output


def test_tournament_needs_recorded_essays(monkeypatch, tmp_path):
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(app, ["tournament", "--experiment", "x", "--judge", "judge_gemini", "--dry-run"])
    assert result.exit_code != 0


@pytest.mark.parametrize(
    "command", [["tournament", "--judge", "judge_gemini"], ["analyze"], ["verdicts"], ["draw-anchors"]]
)
def test_commands_on_an_experiment_require_its_name(monkeypatch, tmp_path, command):
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(app, command)
    assert result.exit_code == 2
    # Typer colours its errors under GITHUB_ACTIONS, which splits the flag's text.
    assert "--experiment" in re.sub(r"\x1b\[[0-9;]*m", "", result.output)


def test_experiment_budget_stops_the_run_mid_essay(monkeypatch, tmp_path):
    import sqlite3

    from aipc.llm import CostCeilingExceeded

    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    result = runner.invoke(
        app, ["experiment", "--config", str(_experiment_yaml(tmp_path)), "--dry-run", "--budget", "0.006"]
    )
    assert isinstance(result.exception, CostCeilingExceeded)
    assert "t1/A_zero_shot/0: words=" in result.output
    statuses = [
        r[0]
        for r in sqlite3.connect(tmp_path / "runs" / "runs.db").execute("SELECT status FROM runs ORDER BY started_at")
    ]
    assert statuses == ["dry_run", "over_ceiling"]


def test_git_state_is_unknown_outside_a_checkout(monkeypatch, tmp_path):
    import subprocess

    from aipc import cli

    real_run = subprocess.run
    monkeypatch.setattr(
        cli.subprocess, "run", lambda args, **kw: real_run(["git", "-C", str(tmp_path), *args[3:]], **kw)
    )
    assert cli._git_state() == (None, None)


def test_a_run_that_fails_is_closed_as_failed_with_its_cost(monkeypatch, tmp_path):
    import sqlite3

    from aipc import cli

    monkeypatch.setenv("AIPC_RUNS_DIR", str(tmp_path / "runs"))
    with pytest.raises(RuntimeError), cli._run("x", slug="s", dry_run=True) as ctx:
        ctx.llm.call(tier="cheap", prompt="p")
        raise RuntimeError("a stage failed")

    status, cost = (
        sqlite3.connect(tmp_path / "runs" / "runs.db").execute("SELECT status, cost_usd FROM runs").fetchone()
    )
    assert status == "failed"
    assert cost == ctx.llm.total_cost_usd > 0
