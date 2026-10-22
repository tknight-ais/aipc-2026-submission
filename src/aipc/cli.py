"""Command-line interface for the aipc pipeline.

Every command that calls a model does so inside `_run`, so each run logs
its resolved config, its git commit and every model call, names its
output directory the same way, and is closed with its cost however it
ends.
"""

import hashlib
import itertools
import json
import logging
import os
import re
import subprocess
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated

import typer
from dotenv import load_dotenv

from aipc.blind import normalise
from aipc.config import DEFAULT_CONFIG_PATH, Config, load_config
from aipc.context import Context
from aipc.llm import CostCeilingExceeded, LLMClient
from aipc.models import ContentLedger, Decomposition, Essay
from aipc.openrouter import list_models
from aipc.pipeline import run as run_pipeline
from aipc.pipeline import write_baseline
from aipc.prices import check as check_prices
from aipc.stages import analysis as analysis_stage
from aipc.stages import calibrate as calibrate_stage
from aipc.stages import checks as checks_stage
from aipc.stages import debate as debate_stage
from aipc.stages import decomposition as decomposition_stage
from aipc.stages import draft as draft_stage
from aipc.stages import experiment as experiment_stage
from aipc.stages import export as export_stage
from aipc.stages import ideation as ideation_stage
from aipc.stages import judge as judge_stage
from aipc.stages import judge_select as judge_select_stage
from aipc.stages import literature as literature_stage
from aipc.stages import power as power_stage
from aipc.stages import select as select_stage
from aipc.stages import thesis_draw as thesis_draw_stage
from aipc.stages import tournament as tournament_stage
from aipc.stages import verdicts as verdicts_stage
from aipc.store import Judgment, Store
from aipc.viewer import rating
from aipc.viewer.server import make_server

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer(add_completion=False, help="AIPC essay-generation pipeline.")

DryRun = Annotated[bool, typer.Option("--dry-run", help="Plan the calls and estimate cost without calling the API.")]
# Required, never defaulted: a default experiment would let a forgotten
# flag judge, analyse or rate the wrong one.
ExperimentName = Annotated[str, typer.Option("--experiment", help="Experiment name, as in its config file.")]
JudgeEntry = Annotated[str, typer.Option("--judge", help="Config model entry of the judge chosen by judge-select.")]
MaxCost = Annotated[
    float | None, typer.Option("--max-cost", help="Cost ceiling for this run in USD; overrides limits.max_cost_usd.")
]


@app.callback()
def _callback() -> None:
    """AIPC essay-generation pipeline.

    A no-op callback: Typer collapses a single-command app into a flat
    CLI otherwise, which would break `aipc baseline ...` once run
    through a real shell instead of directly through the app object.
    """


def _runs_dir() -> Path:
    return Path(os.environ.get("AIPC_RUNS_DIR", "runs"))


def _slugify(text: str, max_len: int = 40) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return slug[:max_len].rstrip("-") or "run"


def _load_essay(path: Path | str) -> Essay:
    return Essay.model_validate_json(Path(path).read_text())


def _require_models(*entries: str) -> None:
    models = load_config(DEFAULT_CONFIG_PATH).models
    unknown = [entry for entry in entries if entry not in models]
    if unknown:
        raise typer.BadParameter(f"no model entry in config for: {', '.join(unknown)}")


def _git_state() -> tuple[str | None, bool | None]:
    """(commit, has uncommitted changes) of the code doing the run, so every
    result can be traced to the exact source that produced it. Both are
    None outside a git checkout (a downloaded archive), recorded as such.
    """
    repo = Path(__file__).resolve().parents[2]
    commit = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    if commit.returncode != 0:
        logger.warning("not a git checkout with a commit: the run records no git commit")
        return None, None
    status = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"], capture_output=True, text=True, check=True
    )
    return commit.stdout.strip(), bool(status.stdout.strip())


def _start_run(variant: str, slug: str, dry_run: bool, config: Config | None = None) -> Context:
    """Loads config (unless given an already-resolved one), logs it, and
    opens a run row under a human-readable, time-sortable label.
    """
    config = config or load_config(DEFAULT_CONFIG_PATH)
    config.dry_run = dry_run
    config_json = config.model_dump_json(indent=2)
    config_hash = hashlib.sha256(config_json.encode()).hexdigest()
    logger.info("variant=%s config_hash=%s", variant, config_hash)
    logger.info("resolved config: %s", config_json)

    run_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)
    label = f"{started_at:%Y%m%d-%H%M%S}_{variant}_{_slugify(slug)}_{run_id[:8]}"
    git_commit, git_dirty = _git_state()
    store = Store(runs_dir=_runs_dir())
    store.save_config(config_hash, config_json)
    store.start_run(
        run_id,
        label,
        config_hash,
        variant=variant,
        started_at=started_at.isoformat(),
        git_commit=git_commit,
        git_dirty=git_dirty,
    )
    llm = LLMClient(config, dry_run=dry_run, record=lambda call: store.save_call(run_id, call))
    return Context(llm=llm, config=config, run_id=run_id, store=store)


def _with_ceiling(config: Config, max_cost_usd: float | None) -> Config:
    """A copy of config whose per-run cost ceiling is at most max_cost_usd."""
    if max_cost_usd is None:
        return config
    limits = config.limits.model_copy(update={"max_cost_usd": max_cost_usd})
    return config.model_copy(update={"limits": limits})


def _close_run(ctx: Context, status: str) -> None:
    ctx.store.finish_run(ctx.run_id, datetime.now(UTC).isoformat(), cost_usd=ctx.llm.total_cost_usd, status=status)


@contextmanager
def _run(variant: str, slug: str, dry_run: bool, config: Config | None = None) -> Iterator[Context]:
    """Opens a run and closes it however the command ends: `ok` (or
    `dry_run`), `over_ceiling` if its cost ceiling stopped it, `failed` on
    any other error or an interrupt. Every close records the cost spent so
    far, so a crashed run is never left `running` at $0.
    """
    ctx = _start_run(variant, slug, dry_run, config)
    try:
        yield ctx
    except CostCeilingExceeded:
        _close_run(ctx, "over_ceiling")
        raise
    except BaseException:
        _close_run(ctx, "failed")
        raise
    _close_run(ctx, "dry_run" if dry_run else "ok")


def _report_run(ctx: Context) -> None:
    """Prints the run's id and (estimated) cost."""
    label = "estimated cost" if ctx.config.dry_run else "cost"
    typer.echo(f"run_id: {ctx.run_id}")
    typer.echo(f"{label}: ${ctx.llm.total_cost_usd:.4f}")


def _save_essay(ctx: Context, essay: Essay) -> None:
    path = ctx.store.save_artifact(ctx.run_id, "essay", essay, markdown_text=essay.text)
    typer.echo(f"word_count: {essay.word_count}")
    typer.echo(f"essay: {path}")


@app.command(rich_help_panel="Component ladder")
def prices() -> None:
    """Compare every config model entry with OpenRouter's live listing:
    price, and whether seed and temperature are supported. No model calls.
    Exits 1 on any mismatch; fix the config before a paid run.
    """
    lines = check_prices(load_config(DEFAULT_CONFIG_PATH).models, list_models())
    for line in lines:
        typer.echo(line)
    if any(line.startswith("MISMATCH") for line in lines):
        raise typer.Exit(1)


@app.command("judge-select", rich_help_panel="Component ladder")
def judge_select(
    levels_dir: Annotated[
        Path, typer.Option("--levels-dir", help="A calibrate run directory holding level_<name>.json essays.")
    ],
    judge: Annotated[
        list[str] | None, typer.Option("--judge", help="Config model entry of a candidate judge. Repeatable.")
    ] = None,
    repeats: Annotated[int, typer.Option("--repeats", help="Seeds per pair; each repeat judges both orderings.")] = 3,
    prompt: Annotated[
        str | None, typer.Option("--prompt", help="Judge prompt name; overrides judge_prompt in config.")
    ] = None,
    max_cost: MaxCost = None,
    dry_run: DryRun = False,
) -> None:
    """Pick the ladder experiment's judge: run each candidate neutral judge on the
    saved graded-damage essays and rank them by correct, position-consistent
    verdicts. Reuses a calibrate run's levels, so nothing is regenerated.
    Candidates default to judge_qwen and judge_gemini.
    """
    judges = judge or ["judge_qwen", "judge_gemini"]
    _require_models(*judges)
    config = _with_ceiling(load_config(DEFAULT_CONFIG_PATH), max_cost)
    config.judge_prompt = prompt or config.judge_prompt
    levels = judge_select_stage.load_levels(levels_dir)
    with _run("judge-select", slug="-".join(judges), dry_run=dry_run, config=config) as ctx:
        ranked = judge_select_stage.rank([judge_select_stage.run_judge(levels, tier, repeats, ctx) for tier in judges])
        for result in ranked:
            ctx.store.save_artifact(ctx.run_id, f"judge_selection_{result.judge_tier}", result)

    for result in ranked:
        typer.echo(
            f"[{result.judge_tier}] {result.model}: correct={result.correct}/{result.trials} "
            f"position_consistent={result.position_consistent}/{result.trials} "
            f"unparsed={result.unparsed} cost=${result.cost_usd:.4f}"
        )
    if not dry_run:
        typer.echo(f"selected: {ranked[0].judge_tier}")
    _report_run(ctx)


@app.command(rich_help_panel="Component ladder")
def ideate(
    topic: Annotated[str, typer.Option(help="Topic area to generate candidate theses for.")],
    tier: Annotated[
        str | None, typer.Option("--tier", help="Model tier for every call; overrides defaults.tier.")
    ] = None,
    dry_run: DryRun = False,
) -> None:
    """Literature background, then cold and informed thesis batches,
    screened by the significance and novelty filters.
    """
    config = load_config(DEFAULT_CONFIG_PATH)
    config.defaults.tier = tier or config.defaults.tier
    with _run("ideate", slug=topic, dry_run=dry_run, config=config) as ctx:
        lit_pack = literature_stage.run(topic, ctx)
        ctx.store.save_artifact(ctx.run_id, "literature", lit_pack)
        for i, result in enumerate(ideation_stage.run(topic, lit_pack, ctx)):
            ctx.store.save_artifact(ctx.run_id, f"candidate{i}", result)
            verdicts = f"significance={result.significance.passed}"
            if result.novelty is not None:
                verdicts += f" novelty={result.novelty.passed}"
            typer.echo(f"[{result.candidate.origin}] {verdicts} :: {result.candidate.claim}")
    _report_run(ctx)


@app.command("draw-theses", rich_help_panel="Component ladder")
def draw_theses(
    ideation_run: Annotated[Path, typer.Option("--ideation-run", help="Run directory of an `aipc ideate` run.")],
    n: Annotated[int, typer.Option("--n", help="Number of theses to draw.")] = 2,
    seed: Annotated[int, typer.Option("--seed", help="First seed to try.")] = 1,
) -> None:
    """Draw the experiment's fixed theses from ideation survivors, with a
    fixed seed, rejecting near-duplicate draws. Makes no API calls. Prints
    YAML to paste unedited into the experiment config.
    """
    result = thesis_draw_stage.draw(thesis_draw_stage.load_survivors(ideation_run), n, seed)
    typer.echo(f"# ideation run: {ideation_run.name}")
    typer.echo(f"# survivors: {result.survivors}, seed used: {result.seed}, rejected seeds: {result.rejected_seeds}")
    typer.echo("theses:")
    for i, claim in enumerate(result.theses, start=1):
        typer.echo(f"  - id: t{i}")
        typer.echo(f"    claim: {json.dumps(claim)}")


@app.command(rich_help_panel="Component ladder")
def experiment(
    config_path: Annotated[
        Path, typer.Option("--config", help="Experiment YAML, e.g. configs/experiments/contest.yaml.")
    ],
    budget: Annotated[
        float | None, typer.Option("--budget", help="Hard cap in USD on this invocation's total spend across essays.")
    ] = None,
    dry_run: DryRun = False,
) -> None:
    """Generate the component-ladder essays: every thesis x condition x index, one
    run per essay with `variant` set to the condition. Essays already
    recorded for this experiment are skipped, so a crashed run resumes.

    Essays are written in passes, one essay per thesis and condition in
    each, so a run that stops early leaves every condition with the same
    number of essays, and any drift over a long run (provider routing,
    model updates) falls on every condition alike.

    With --budget, each essay's cost ceiling is what is left of the budget,
    so the run stops mid-essay rather than overspend.
    """
    exp = experiment_stage.load_experiment(config_path)
    base_config = load_config(DEFAULT_CONFIG_PATH)
    done = {(e.thesis_id, e.condition, e.essay_index) for e in Store(_runs_dir()).experiment_essays(exp.name)}
    total_cost = 0.0
    for index, thesis, condition in itertools.product(range(exp.essays_per_condition), exp.theses, exp.conditions):
        if (thesis.id, condition, index) in done:
            typer.echo(f"{thesis.id}/{condition}/{index}: already done, skipped")
            continue
        remaining = None if budget is None else budget - total_cost
        if remaining is not None and remaining <= 0:
            typer.echo(f"budget ${budget:.2f} used up; stopping")
            break
        total_cost += _experiment_essay(exp, thesis, condition, index, base_config, dry_run, remaining)
    label = "estimated total cost" if dry_run else "total cost"
    typer.echo(f"{label}: ${total_cost:.4f}")


def _experiment_essay(
    exp: experiment_stage.ExperimentConfig,
    thesis: experiment_stage.Thesis,
    condition: str,
    index: int,
    base_config: Config,
    dry_run: bool,
    max_cost_usd: float | None,
) -> float:
    """Writes one experiment essay as its own run and returns its cost."""
    config = experiment_stage.condition_config(base_config, condition, exp.generator_tier, seed=exp.seed + index)
    config = _with_ceiling(config, None if max_cost_usd is None else min(max_cost_usd, config.limits.max_cost_usd))
    with _run(condition, slug=f"{exp.name}-{thesis.id}", dry_run=dry_run, config=config) as ctx:
        essay, essay_path = experiment_stage.write_essay(condition, thesis.claim, ctx)
        if not dry_run:
            ctx.store.save_experiment_essay(exp.name, thesis.id, condition, index, ctx.run_id, essay_path)
    typer.echo(
        f"{thesis.id}/{condition}/{index}: words={essay.word_count} cost=${ctx.llm.total_cost_usd:.4f} run_id={ctx.run_id}"
    )
    return ctx.llm.total_cost_usd


@app.command(rich_help_panel="Component ladder")
def tournament(
    judge: JudgeEntry,
    experiment_name: ExperimentName,
    per_essay: Annotated[int, typer.Option("--per-essay", help="Opponents per essay.")] = 6,
    seed: Annotated[int, typer.Option("--seed", help="Pairing schedule seed.")] = 1,
    max_cost: MaxCost = None,
    dry_run: DryRun = False,
) -> None:
    """Blind pairwise tournament over an experiment's essays: seeded
    pairing within each thesis across conditions, formatting stripped,
    both orderings judged, every verdict saved to the judgments table.
    """
    _require_models(judge)
    essays = Store(_runs_dir()).experiment_essays(experiment_name)
    if not essays:
        raise typer.BadParameter(f"no essays recorded for experiment {experiment_name!r}; run `aipc experiment` first")

    config = _with_ceiling(load_config(DEFAULT_CONFIG_PATH), max_cost)
    schedule = tournament_stage.Schedule(
        experiment=experiment_name,
        seed=seed,
        per_essay=per_essay,
        pairs=tournament_stage.schedule_pairs(essays, per_essay, seed),
    )
    texts = {e.run_id: normalise(_load_essay(e.essay_path).text) for e in essays}
    with _run("tournament", slug=f"{experiment_name}-{judge}", dry_run=dry_run, config=config) as ctx:
        ctx.store.save_artifact(ctx.run_id, "schedule", schedule)
        judgments = tournament_stage.run(experiment_name, schedule, texts, judge, ctx)

    unparsed = sum(j.winner_run is None for j in judgments)
    typer.echo(
        f"essays: {len(essays)}  pairs: {len(schedule.pairs)}  new judgments: {len(judgments)}  unparsed: {unparsed}"
    )
    _report_run(ctx)


@app.command("draw-anchors", rich_help_panel="Component ladder")
def draw_anchors(experiment_name: ExperimentName) -> None:
    """Freeze the blind-rating set once the tournament is complete: a seeded
    draw of pairs the judge decided the same way in both orderings. The
    /rate page serves only this set. Refuses to redraw. No API calls.
    """
    store = Store(_runs_dir())
    if rating.anchors(store, experiment_name):
        raise typer.BadParameter(f"anchors for {experiment_name!r} are already drawn and frozen")
    judgments = store.judgments(experiment_name)
    _single_judge(judgments)
    anchor_set = rating.draw(experiment_name, judgments)

    with _run("draw-anchors", slug=experiment_name, dry_run=False) as ctx:
        for anchor in anchor_set.anchors:
            ctx.store.save_anchor_pair(anchor)
        ctx.store.save_artifact(ctx.run_id, "anchors", anchor_set)
    typer.echo(f"drew {len(anchor_set.anchors)} anchors from {anchor_set.decided_pairs} position-consistent pairs")
    typer.echo(f"run_id: {ctx.run_id}")


@app.command(rich_help_panel="Component ladder")
def view(
    port: Annotated[int, typer.Option(help="Local port to serve on.")] = 8765,
    experiment_name: Annotated[
        str | None, typer.Option("--experiment", help="Experiment whose anchors /rate serves; none, no rating.")
    ] = None,
) -> None:
    """Serve a browser view of past runs on localhost, plus /rate for
    blind anchor ratings of the experiment named with --experiment.
    """
    server = make_server(_runs_dir(), port, experiment_name)
    typer.echo(f"serving {_runs_dir()} at http://127.0.0.1:{port}  (ctrl-c to stop)")
    server.serve_forever()


@app.command(rich_help_panel="Component ladder")
def analyze(experiment_name: ExperimentName) -> None:
    """Pre-registered analysis of a judged experiment: Bradley-Terry
    strengths with bootstrap intervals, win rates, position consistency,
    verbosity check, hedge density, and judge-human agreement. No API calls.
    """
    reader = Store(_runs_dir())
    essays = reader.experiment_essays(experiment_name)
    judgments = reader.judgments(experiment_name)
    judge_model = _single_judge(judgments)
    texts = {e.run_id: _load_essay(e.essay_path) for e in essays}

    report = analysis_stage.build_report(
        experiment_name,
        judge_model,
        judgments,
        reader.human_judgments(experiment_name),
        condition_of={e.run_id: e.condition for e in essays},
        word_count={run_id: float(essay.word_count) for run_id, essay in texts.items()},
        hedge={run_id: draft_stage.hedge_density(essay.text) for run_id, essay in texts.items()},
    )
    with _run("analyze", slug=experiment_name, dry_run=False) as ctx:
        ctx.store.save_artifact(ctx.run_id, "analysis", report)
    _print_report(report)
    typer.echo(f"run_id: {ctx.run_id}")


def _single_judge(judgments: list[Judgment]) -> str:
    """The one judge model behind these judgments. Judgments from a second
    model or prompt would otherwise be pooled silently.
    """
    judges = sorted({(j.judge_model, j.judge_prompt_hash or "unrecorded") for j in judgments})
    if len(judges) != 1:
        raise typer.BadParameter(f"need judgments from exactly one judge model and prompt, found: {judges}")
    return judges[0][0]


def _print_report(report: analysis_stage.Report) -> None:
    typer.echo(
        f"experiment {report.experiment}, judge {report.judge_model}: {report.essays} essays, {report.pairs} pairs"
    )
    for scope, estimates in report.strengths.items():
        typer.echo(f"Bradley-Terry log-strength ({scope}), 95% bootstrap CI by pair [by essay]:")
        for condition, e in estimates.items():
            typer.echo(
                f"  {condition:14} {e.log_strength:+.2f}  [{e.ci_low:+.2f}, {e.ci_high:+.2f}]"
                f"  [{e.essay_ci_low:+.2f}, {e.essay_ci_high:+.2f}]"
            )
    typer.echo("win rates:")
    for line in report.win_rates:
        typer.echo(f"  {line}")
    passed = "PASS" if report.position_consistency_pass else "FAIL"
    typer.echo(f"position consistency: {report.position_consistency:.0%} (bar 70%: {passed})")
    typer.echo(f"verbosity correlation (A won vs A's word lead): {report.verbosity_correlation}")
    typer.echo(f"longer essay won {report.longer_won_share} of decided pairs")
    for condition, value in report.hedge_density.items():
        typer.echo(f"  {condition:14} hedges/1k words {value:.2f}  mean words {report.word_count[condition]:.0f}")
    a = report.agreement
    gate = "PASS" if report.h4_gate_pass else "FAIL"
    interval = f"95% CI [{a.ci_low:.2f}, {a.ci_high:.2f}]" if a.rated else "no ratings"
    typer.echo(
        f"judge-human agreement: {a.agreed}/{a.rated}, {interval}, kappa {a.kappa}, "
        f"chance p {a.binomial_p:.4f} (H4 bar 70% of {analysis_stage.AGREEMENT_MIN_RATED}: {gate})"
    )


@app.command(rich_help_panel="Component ladder")
def verdicts(
    experiment_name: ExperimentName,
    out: Annotated[
        Path | None, typer.Option("--out", help="Markdown file to write; default runs/verdicts_<experiment>.md.")
    ] = None,
) -> None:
    """Write a readable report of the judge's verdicts: how it behaved, and
    every pair with its reasoning. Names conditions, so it is not blind.
    No API calls.
    """
    store = Store(_runs_dir())
    essays = store.experiment_essays(experiment_name)
    judgments = store.judgments(experiment_name)
    if not judgments:
        raise typer.BadParameter(f"no judgments recorded for experiment {experiment_name!r}")
    texts = {e.run_id: _load_essay(e.essay_path) for e in essays}
    path = out or _runs_dir() / f"verdicts_{experiment_name}.md"
    path.write_text(verdicts_stage.build(experiment_name, judgments, essays, texts))
    typer.echo(f"report: {path}")


@app.command(rich_help_panel="Component ladder")
def power(
    essays: Annotated[
        list[int] | None, typer.Option("--essays", help="Essays per condition per thesis. Repeatable; default 4, 6, 8.")
    ] = None,
    effect: Annotated[
        list[float] | None,
        typer.Option(
            "--effect", help="True log-strength of the 2nd condition over the 1st. Repeatable; default 0.5, 1, 1.5."
        ),
    ] = None,
    essay_sd: Annotated[
        float, typer.Option("--essay-sd", help="SD of essay quality within a condition, in log-strength.")
    ] = 0.5,
    conditions: Annotated[
        list[str] | None, typer.Option("--condition", help="Conditions, reference first. Repeatable; default A, C, D.")
    ] = None,
    theses: Annotated[int, typer.Option("--theses")] = 2,
    per_essay: Annotated[int, typer.Option("--per-essay", help="Opponents per essay, as in the tournament.")] = 6,
    tie_rate: Annotated[float, typer.Option("--tie-rate", help="Share of pairs whose orderings disagree.")] = 0.26,
    reps: Annotated[int, typer.Option("--reps", help="Simulated experiments per design point.")] = 200,
    resamples: Annotated[int, typer.Option("--resamples", help="Bootstrap resamples per interval.")] = 200,
    seed: int = 1,
) -> None:
    """Simulated power of the pre-registered analysis for a ladder design:
    how often it would detect a true difference of each size. No API calls.
    """
    conditions = conditions or ["A_zero_shot", "C_structured", "D_debate"]
    typer.echo("essays  effect  judgments  pair-CI power  robust power")
    for n, size in itertools.product(essays or [4, 6, 8], effect or [0.5, 1.0, 1.5]):
        r = power_stage.simulate(conditions, theses, n, per_essay, size, essay_sd, tie_rate, reps, resamples, seed)
        typer.echo(f"{n:6}  {size:6.2f}  {r.judgments:9}  {r.pair_power:13.2f}  {r.robust_power:12.2f}")
    typer.echo(f"essay_sd={essay_sd} tie_rate={tie_rate} reps={reps} (Monte Carlo SE up to {0.5 / reps**0.5:.2f})")


@app.command(rich_help_panel="Contest essay")
def select(
    experiment_name: Annotated[str, typer.Option("--experiment", help="Experiment whose essays are the candidates.")],
    judge: JudgeEntry,
    repeats: Annotated[int, typer.Option("--repeats", help="Seeds per pair; each judges both orderings.")] = 2,
    max_cost: MaxCost = None,
    dry_run: DryRun = False,
) -> None:
    """Pick the essay to submit: round-robin of an experiment's essays with
    the calibrated neutral judge, both orderings, ranked by consistent wins.
    """
    _require_models(judge)
    essays = {e.name: e for e in Store(_runs_dir()).experiment_essays(experiment_name)}
    if len(essays) < 2:
        raise typer.BadParameter(f"experiment {experiment_name!r} has fewer than 2 essays")

    config = _with_ceiling(load_config(DEFAULT_CONFIG_PATH), max_cost)
    candidates = {name: _load_essay(e.essay_path).text for name, e in essays.items()}
    with _run("select", slug=f"{experiment_name}-{judge}", dry_run=dry_run, config=config) as ctx:
        result = select_stage.run(candidates, judge, repeats, ctx)
        ctx.store.save_artifact(ctx.run_id, "selection", result)

    for ranked in result.ranking:
        typer.echo(f"{ranked.name}: wins={ranked.wins} losses={ranked.losses} trials={ranked.trials}")
    tie_note = "  (TIED at top: rerun with more --repeats)" if result.tied_at_top else ""
    typer.echo(f"winner: {result.winner}{tie_note}")
    typer.echo(f"winner essay: {essays[result.winner].essay_path}")
    _report_run(ctx)


@app.command(rich_help_panel="Contest essay")
def check(
    essay: Annotated[Path, typer.Option("--essay", help="Path to an Essay JSON file, e.g. the one select picked.")],
) -> None:
    """Pre-submission checks, no model calls: body word count, OpenAlex
    lookup of every reference, in-text citations without an entry, and
    text that refers to how the essay was made.
    """
    with _run("check", slug=essay.parent.name, dry_run=False) as ctx:
        report = checks_stage.run(_load_essay(essay).text)
        ctx.store.save_artifact(ctx.run_id, "checks", report)

    typer.echo(f"words: {report.total_words} total, {report.body_words} body (limit {checks_stage.MAX_WORDS} total)")
    for ref in report.references:
        typer.echo(f"[{'found' if ref.found else 'CHECK BY HAND'}] {ref.entry}\n    best match: {ref.best_match}")
    for citation in report.unmatched_in_text:
        typer.echo(f"[NO ENTRY] in-text citation: {citation}")
    for mention in report.process_mentions:
        typer.echo(f"[PROCESS?] ...{mention}...")
    typer.echo("PASS" if report.passed else "REVIEW NEEDED")
    typer.echo(f"run_id: {ctx.run_id}")


@app.command(rich_help_panel="Contest essay")
def export(
    essay: Annotated[Path, typer.Option("--essay", help="Path to an Essay JSON file.")],
    out: Annotated[Path, typer.Option("--out", help="PDF path to write.")],
) -> None:
    """Export an essay to PDF via pandoc and Typst. The text is not changed."""
    typ_path = export_stage.to_pdf(_load_essay(essay).text, out)
    typer.echo(f"pdf: {out}")
    typer.echo(f"typst source: {typ_path}")


@app.command(rich_help_panel="Single stages")
def decompose(
    thesis: Annotated[str, typer.Option(help="The thesis to decompose into sub-questions.")],
    dry_run: DryRun = False,
) -> None:
    """Generate candidate decompositions, screen them for structural
    validity, and select the strongest survivor.
    """
    with _run("decompose", slug=thesis, dry_run=dry_run) as ctx:
        screened, chosen = decomposition_stage.run(thesis, ctx.config.pipeline.decomposition_candidates, ctx)
        for i, result in enumerate(screened):
            ctx.store.save_artifact(ctx.run_id, f"candidate{i}", result)
            typer.echo(f"[{i}] validity={result.validity.passed} :: {' / '.join(result.decomposition.sub_questions)}")
        ctx.store.save_artifact(ctx.run_id, "chosen", chosen)
        typer.echo(f"chosen: {' / '.join(chosen.sub_questions)}")
    _report_run(ctx)


@app.command(rich_help_panel="Single stages")
def ledger(
    decomposition: Annotated[Path, typer.Option("--decomposition", help="Path to a Decomposition JSON file.")],
    dry_run: DryRun = False,
) -> None:
    """Turn a decomposition into a ContentLedger via the configured
    generation strategy (`pipeline.strategy`: flat or debate_room).
    """
    decomp = Decomposition.model_validate_json(decomposition.read_text())
    with _run("ledger", slug=decomp.thesis, dry_run=dry_run) as ctx:
        ctx.store.save_artifact(ctx.run_id, "decomposition", decomp)
        content_ledger, transcripts = debate_stage.run(decomp, ctx)
        for i, transcript in enumerate(transcripts):
            ctx.store.save_artifact(ctx.run_id, f"transcript{i}", transcript)
        ctx.store.save_artifact(ctx.run_id, "ledger", content_ledger)

    typer.echo(f"strategy: {ctx.config.pipeline.strategy}")
    typer.echo(f"claims: {len(content_ledger.claims)}")
    typer.echo(f"objections: {len(content_ledger.objections)}")
    typer.echo(f"concessions: {len(content_ledger.concessions)}")
    _report_run(ctx)


@app.command(rich_help_panel="Single stages")
def draft(
    ledger: Annotated[Path, typer.Option("--ledger", help="Path to a ContentLedger JSON file.")],
    dry_run: DryRun = False,
) -> None:
    """Turn a content ledger into an essay via Drafter -> Critics -> Reviser,
    then a ledger-fidelity pass over the final essay.
    """
    content_ledger = ContentLedger.model_validate_json(ledger.read_text())
    with _run("draft", slug=content_ledger.thesis, dry_run=dry_run) as ctx:
        ctx.store.save_artifact(ctx.run_id, "ledger", content_ledger)
        essay, rounds, fidelity = draft_stage.run(content_ledger, ctx)
        for revision_round in rounds:
            ctx.store.save_artifact(ctx.run_id, f"round{revision_round.round_num}", revision_round)
            ctx.store.save_measurement(ctx.run_id, "word_count", float(revision_round.word_count))
            ctx.store.save_measurement(ctx.run_id, "hedge_density", revision_round.hedge_density)
            ctx.store.save_measurement(ctx.run_id, "has_new_objection", float(revision_round.has_new_objection))
            typer.echo(
                f"round {revision_round.round_num}: word_count={revision_round.word_count} "
                f"hedge_density={revision_round.hedge_density:.2f} "
                f"new_objection={revision_round.has_new_objection}"
            )
        ctx.store.save_artifact(ctx.run_id, "fidelity", fidelity)
        _save_essay(ctx, essay)
    _report_run(ctx)


@app.command(rich_help_panel="Original pipeline")
def generate(
    topic: Annotated[str, typer.Option(help="Topic area to generate a full essay for.")],
    dry_run: DryRun = False,
) -> None:
    """Full pipeline, one topic in: literature -> ideation -> decomposition
    -> ledger -> draft/critique/revise -> judge against a fresh baseline
    reference. See pipeline.py for the stage order and its one known
    simplification (the reference essay isn't frozen across runs yet).
    """
    with _run("generate", slug=topic, dry_run=dry_run) as ctx:
        result = run_pipeline(topic, ctx)

    typer.echo(f"thesis: {result.thesis}")
    typer.echo(f"sub_questions: {' / '.join(result.decomposition.sub_questions)}")
    typer.echo(f"ledger: {len(result.ledger.claims)} claims, {len(result.ledger.objections)} objections")
    typer.echo(f"essay: {result.essay.word_count} words, {len(result.rounds)} revision rounds")
    for name, comparison in result.comparisons.items():
        typer.echo(f"[{name}] beat reference={comparison['candidate_wins']} consistent={comparison['consistent']}")
    _report_run(ctx)


@app.command(rich_help_panel="Original pipeline")
def baseline(
    topic: Annotated[str, typer.Option(help="Topic area to write the essay about.")],
    dry_run: DryRun = False,
) -> None:
    """Generate a single-model baseline essay for a topic area.

    The frozen pairwise reference for later judging: one call, one stage,
    no pipeline.
    """
    with _run("baseline", slug=topic, dry_run=dry_run) as ctx:
        _save_essay(ctx, write_baseline(topic, ctx))
    _report_run(ctx)


@app.command(rich_help_panel="Original pipeline")
def evaluate(
    essay: Annotated[Path, typer.Option("--essay", help="Path to the candidate Essay JSON file.")],
    reference: Annotated[Path, typer.Option("--reference", help="Path to the frozen reference Essay JSON file.")],
    dry_run: DryRun = False,
) -> None:
    """Score the candidate essay with the frozen mirror panel (absolute
    rubric scores) and compare it against the frozen reference essay
    pairwise, in both orderings.
    """
    candidate_essay, reference_essay = _load_essay(essay), _load_essay(reference)
    with _run("evaluate", slug=candidate_essay.thesis or "essay", dry_run=dry_run) as ctx:
        ctx.store.save_measurement(ctx.run_id, "word_count", float(candidate_essay.word_count))
        for persona in judge_stage.MIRROR_PANEL:
            scores = judge_stage.score_absolute(candidate_essay, "candidate", persona, panel_type="mirror", ctx=ctx)
            for score in scores:
                ctx.store.save_score(ctx.run_id, score)
                typer.echo(f"[{persona.name}] {score.criterion}={score.score}")

            comparison = judge_stage.compare_both_orderings(candidate_essay, reference_essay, persona, ctx)
            winner = "candidate" if comparison["candidate_wins"] else "reference"
            ctx.store.save_comparison("candidate", "reference", persona.name, winner, "both_orders")
            typer.echo(f"[{persona.name}] winner={winner} consistent={comparison['consistent']}")
    _report_run(ctx)


@app.command(rich_help_panel="Original pipeline")
def calibrate(
    essay: Annotated[Path, typer.Option("--essay", help="Path to an Essay JSON file to use as the undamaged anchor.")],
    max_cost: MaxCost = None,
    dry_run: DryRun = False,
) -> None:
    """Grade deliberate, cumulative damage into the given essay (original >
    light > heavy > severe) and check whether the persona panel's scores
    and pairwise verdicts reproduce that order. judge-select reuses the
    graded essays to choose the ladder's judge.
    """
    original_essay = _load_essay(essay)
    config = _with_ceiling(load_config(DEFAULT_CONFIG_PATH), max_cost)
    with _run("calibrate", slug=original_essay.thesis or "essay", dry_run=dry_run, config=config) as ctx:
        levels, results = calibrate_stage.run(original_essay, ctx)
        for name, level_essay in levels:
            ctx.store.save_artifact(ctx.run_id, f"level_{name}", level_essay, markdown_text=level_essay.text)
        for result in results:
            ctx.store.save_artifact(ctx.run_id, f"calibration_{result.persona}", result)

    for result in results:
        typer.echo(
            f"[{result.persona}] monotonic={result.monotonic} "
            f"pairwise={result.pairwise_results} scores={result.absolute_scores}"
        )
    overall_pass = all(r.monotonic and all(r.pairwise_results.values()) for r in results)
    typer.echo(f"overall: {'PASS' if overall_pass else 'FAIL'}")
    _report_run(ctx)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
