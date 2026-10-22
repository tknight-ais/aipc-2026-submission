# aipc-2026-submission

A multi-agent pipeline that writes philosophy essays for the
[AI Philosophy Competition 2026](https://www.zacharygoodsell.com/ai-philosophy-competition),
built as an experiment: which parts of an agent scaffold actually improve
the essays, in a domain with no ground truth?

Critique loops, structured outlines and debate between agents are often
assumed to beat a single strong model. Philosophy has no answer key to
check that against, so the comparison is built to be trustworthy without
one: one component changes at a time, the judge is validated before use,
and a human checks the judge.

## Design: a component ladder

Every condition argues the same fixed thesis, with the same generator model
and the same writing instructions (`prompts/essay_requirements.md`). Each
condition adds one thing to the one before:

| Condition | Adds |
| :--- | :--- |
| A, zero-shot | One call: thesis and requirements in, essay out |
| B, critique | Three critics, blind to each other, then one revision against their objections |
| C, structured | An argument built before drafting: the thesis split into sub-questions, a ledger of claims, objections and replies, a draft from the ledger; then B's critique round |
| D, debate | C, with the ledger built by two position-defined debaters and a manager instead of one call per sub-question |

C to D changes only how the ledger is built, and that is a configuration
value (`pipeline.strategy`), not a code branch. The theses are drawn with a
fixed seed from the system's own ideation; the only human input to the
essays is the topic.

## Judging

- **Blind and pairwise.** Essays on the same thesis from different conditions are paired by a seeded schedule. Formatting is stripped first (`blind.py`), and every pair is judged in both orderings. A pair is won only if both orderings agree; otherwise it is a tie.
- **A different model family.** The generator is Claude and the judge is not, to avoid self-preference.
- **Selected before use.** Candidate judges compare versions of one essay with planted, cumulative damage (original > light > heavy > severe, true by construction), and are ranked by a rule fixed in advance (`aipc judge-select`).
- **Checked by a human.** After the tournament, 20 pairs the judge decided consistently are frozen and rated blind at `/rate`. Judge-human agreement decides whether the results are read as evidence about the essays or only about the judge.

## Analysis

`aipc analyze` runs the pre-registered analysis (`stages/analysis.py`;
the statistics use only the standard library):

- Bradley-Terry log-strength per condition relative to A (Hunter's MM), pooled and per thesis.
- Two 95% bootstrap intervals: over pairs, which is too narrow because pairs share essays, and over essays (a vertex bootstrap), which errs wide. A difference is called robust only if both exclude zero.
- Win rates, position consistency, whether the longer essay wins, hedge density, and judge-human agreement with a Clopper-Pearson interval.

`aipc power` simulates the design's false-positive rate and power with the
real schedule and analysis, before anything is spent. Each experiment's
design, judge-selection rule, hypotheses and analysis are pre-registered
and tagged before its first essay, in the private research repository
(available on request).

## Provenance and safeguards

- **Every model call is stored** in SQLite (`runs/runs.db`): the full prompt and reply, requested and served model, provider, tokens including hidden reasoning, billed cost, finish reason and attempts. Every run records its resolved config and git commit, with a flag for uncommitted changes.
- **Stage outputs are saved as JSON** as they complete, so any stage can be re-run against fixed input (`aipc decompose`, `ledger`, `draft`).
- **Spend is capped** per run and per command (`--budget`, `--max-cost`). Every paid command has a `--dry-run` that prices an estimated upper bound, and `aipc prices` checks the configured prices against OpenRouter's live listing.
- **Unreadable replies never become defaults.** An unreadable verdict is a tie, never "A"; a ledger reply with no readable claim stops the run; a reply cut off by a length cap raises.
- **Every run is closed with its cost**, however it ends: `ok`, `over_ceiling` or `failed`.
- **Sampling is recorded as sent.** Seed and temperature are sent only to models that support them. The Claude 5.x models support neither, so generation is not reproducible from a seed; the stored replies are the record.

## Running it

```bash
uv sync                    # Python 3.13, locked dependencies
uv run pytest              # offline: no API key needed
cp .env.example .env       # then add an OpenRouter key
uv run aipc --help         # commands grouped by purpose, in workflow order
```

The ladder, in order: `prices`, `judge-select`, `ideate`, `draw-theses`,
`experiment`, `tournament`, `draw-anchors`, blind ratings at `/rate`
(`aipc view --experiment <name>`), then `analyze`. The contest essay:
`experiment` with the winning condition, then `select`, `check` and
`export`. Run commands from the project root, where `runs/` lives (or set
`AIPC_RUNS_DIR`). `export` needs [pandoc](https://pandoc.org) (tested with
3.8). CI runs ruff, pyright and the tests on pushes to `main` and on pull
requests.

Experiment configurations with specific theses stay private until judging
ends; `configs/experiments/contest.yaml` shows the format.

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

The original pipeline (`generate`, `baseline`, `evaluate`, `calibrate`,
scored by a panel of persona judges) predates the ladder and is kept
because the project's earliest results came from it. The ladder shares its
stages but not its judging, and selects its judge on the graded-damage
essays `calibrate` wrote.

## Limitations

- The judge is a language model. Selection and the human check limit how far its verdicts can mislead; they do not remove the problem.
- Blinding is imperfect: formatting is stripped, but paragraph length and reference lists can still differ by condition.
- Pairs share essays and essays share theses, so the effective sample is smaller than the number of judgments. Only large effects are detectable at the planned size, and a null result is not evidence that an effect is absent.
- The significance, novelty and validity gates have not been calibrated on known-bad inputs. Treat them as prompts, not as tested filters.

## Licence

MIT: see [LICENSE](LICENSE).
