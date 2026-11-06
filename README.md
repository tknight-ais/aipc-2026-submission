# Testing agent scaffolds for philosophy research

[![CI](https://github.com/tknight-ais/aipc-2026-submission/actions/workflows/ci.yml/badge.svg)](https://github.com/tknight-ais/aipc-2026-submission/actions/workflows/ci.yml)

A multi-agent pipeline that writes philosophy essays, and an experiment to
find which parts of an agent scaffold actually help, in a field with no
answer key. Built for the
[AI Philosophy Competition 2026](https://www.zacharygoodsell.com/ai-philosophy-competition).

**Status.** One essay from this pipeline is entered in the competition,
with a methodology report. Judging is blind, so the essay's topic and the
results are kept out of the public code until they are announced at the
end of January 2027. The full research record will be published then: the
designs fixed before each run, the decision log, and the run database.

**Who built it.** Tom Knight designed the project, chose the question and
the experiments, and approved every paid run. Claude Code, Anthropic's AI
coding agent, wrote and ran the code under his direction.

## The problem

Critique loops, structured outlines and debate between agents are often
assumed to beat a single strong model. Testing that means comparing
essays, and philosophy has no answer key. So the comparison is built to be
trustworthy without one: one component changes at a time, every judge is
tested before use on essays with planted errors of known severity, and a
human can check the judge.

## Design: a component ladder

Every condition uses the same generator model and the same writing
instructions (`prompts/essay_requirements.md`), and each adds one thing to
the one before:

| Condition | Adds |
| :--- | :--- |
| A, zero-shot | One call: thesis and requirements in, essay out |
| B, critique | Three critics, blind to each other, then one revision against their objections |
| C, structured | An argument built before drafting: the thesis split into sub-questions, a ledger of claims, objections and replies, a draft from the ledger; then B's critique round |
| D, debate | C, with the ledger built by two position-defined debaters and a manager instead of one call per sub-question |

C to D changes only how the ledger is built, which is a configuration value
(`pipeline.strategy`), not a code branch. Every condition argues the same
fixed thesis, drawn with a fixed seed from the system's own ideation. So
the ladder measures what each component adds to an argument, not how well
it generates ideas.

The contest comparison used a 2x2 built from these parts: A, B, D0 (D
without its critique round) and D, all given the thesis the contest run
chose, with the submitted essay added to the judging. The submitted essay
came from the full pipeline, run once from the question alone: literature
search, thesis generation and screening, a plan, debate rooms, a draft and
up to three rounds of critique (`aipc generate`).

## Judging

- **Blind and pairwise.** Essays on the same thesis from different conditions are paired by a seeded schedule. Formatting and titles are stripped first (`blind.py`), and every pair is judged in both orders. A pair is won only if both orders agree; otherwise it is a tie.
- **Tested before use.** Candidate judges compare versions of one essay with planted, cumulative damage (original > light > heavy > severe, true by construction). A trial counts only if the judge picks the better version in both orders, and the pass mark is fixed before any judge runs (`aipc judge-select`).
- **Other model families.** The generator is Claude, so the main judges are not. A Claude judge can still run, labelled as same-family.
- **One judge at a time.** Several judges can rate the same pairs; `aipc analyze --judge <entry>` analyses one judge's verdicts and refuses to pool them.
- **Checkable by a human.** `draw-anchors` freezes 20 pairs the judge decided consistently for blind rating at `/rate`, and judge-human agreement decides whether the results say something about the essays or only about the judge. The contest comparison skipped this step for lack of time.

## Analysis

`aipc analyze` runs the analysis fixed before each experiment
(`stages/analysis.py`; the statistics use only the standard library):

- Bradley-Terry log-strength per condition relative to A (Hunter's MM), pooled and per thesis.
- Two 95% bootstrap intervals: over pairs, which is too narrow because pairs share essays, and over essays, which errs wide. A difference is called robust only if both exclude zero.
- Win rates, position consistency, whether the longer essay wins, and judge-human agreement with a Clopper-Pearson interval.

`aipc power` simulates the design's false-positive rate and power with the
real schedule and analysis before anything is spent.

## Provenance and safeguards

- **Every model call is stored** in SQLite (`runs/runs.db`): the full prompt and reply, requested and served model, provider, tokens including hidden reasoning, billed cost, finish reason and attempts. Every run records its resolved config and git commit, with a flag for uncommitted changes.
- **Stage outputs are saved as JSON** as they complete, so any stage can be re-run against fixed input, and a failed run resumes by replaying its recorded calls at no cost.
- **Spend is capped** per run and per command (`--budget`, `--max-cost`). Every paid command has a `--dry-run` that prices an upper bound, and `aipc prices` checks the configured prices against OpenRouter's live listing.
- **Unreadable replies never become defaults.** An unreadable verdict is a tie, never "A"; a ledger reply with no readable claim stops the run; a reply cut off by a length cap raises.
- **Sampling is recorded as sent.** Seed and temperature go only to models that support them. The Claude 5.x models support neither, so generation can't be reproduced from a seed; the stored replies are the record.

## Running it

```bash
uv sync                    # Python 3.13, locked dependencies
uv run pytest              # offline: no API key needed
cp .env.example .env       # then add an OpenRouter key, for paid commands only
uv run aipc --help         # commands grouped by purpose, in workflow order
```

The ladder, in order: `prices`, `judge-select`, `ideate`, `draw-theses`,
`experiment`, `tournament`, `draw-anchors`, blind ratings at `/rate`
(`aipc view --experiment <name>`), then `analyze`. The contest essay:
`generate --tier frontier --strategy debate_room --no-evaluate`, then
`check` and `export`. An essay from another run joins an experiment's
judging with `add-essay`, before `tournament`.

Run commands from the project root, where `runs/` lives (or set
`AIPC_RUNS_DIR`). `export` needs [pandoc](https://pandoc.org) (tested with
3.8). CI runs ruff, pyright and the tests on pushes to `main` and on pull
requests. Experiment configurations name their theses, so they stay
private until judging ends; `configs/experiments/contest.yaml` shows the
format.

## Layout

```
src/aipc/
  cli.py          commands; every run is opened and closed by _run
  llm.py          the single call path: pricing, cost ceiling, retries, call log
  replies.py      parsers for every reply format the prompts ask for
  blind.py        formatting stripped before judging
  store.py        SQLite tables and JSON artifacts
  prompts/        every prompt, as a file; each run records the text it sent
  stages/
    experiment.py                         the ladder's conditions
    decomposition.py debate.py draft.py   building and drafting the argument
    judge_select.py tournament.py neutral_judge.py
    analysis.py power.py
    ideation.py literature.py thesis_draw.py
    select.py checks.py export.py         the contest essay
  viewer/         local run viewer and the /rate page
```

`baseline`, `evaluate` and `calibrate` predate the ladder and are kept
because the project's earliest results came from them; `calibrate` wrote
the planted-damage essays that judges are tested on.

## Limitations

- The judge is a language model. Testing it first and checking it against a human limit how far its verdicts can mislead; they don't remove the problem.
- Blinding is imperfect: formatting is stripped, but paragraph length and reference lists can still differ by condition.
- Pairs share essays and essays share theses, so the effective sample is smaller than the number of judgments. Only large effects are detectable at the planned size, and a null result is not evidence that an effect is absent.
- The thesis is fixed, so the ladder tests how components develop an argument, not how well they generate ideas.
- The significance, novelty and validity gates have not been calibrated on known-bad inputs. Treat them as prompts, not as tested filters.

## Licence

MIT: see [LICENSE](LICENSE).
