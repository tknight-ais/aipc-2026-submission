# aipc-2026-submission

A multi-agent pipeline that generates philosophy essays for the
[AI Philosophy Competition 2026](https://www.zacharygoodsell.com/ai-philosophy-competition),
built so that comparisons between agent architectures remain valid in a
domain with no ground truth.

The pipeline runs AI-led ideation, literature grounding, adversarial debate
between philosopher personas, and iterative critique. Candidates are scored
by a panel of persona judges held out from generation. The central
measurement is whether the scaffold beats a plain single-model essay on the
same topic.

## Quick start

```bash
uv sync                          # install from the lockfile
cp .env.example .env             # then add your OpenRouter key
uv run pytest                    # passes without an API key
uv run aipc generate --topic metaethics   # full pipeline, needs the key
uv run aipc view                 # browse runs at http://127.0.0.1:8765
```

Run commands from the project root; `view` reads `runs/` relative to it
(or `AIPC_RUNS_DIR`).

Workflow, in order: `prices` (config against OpenRouter's live listing),
`ideate`, `draw-theses`, `experiment`, `tournament`, `draw-anchors`, blind
ratings at `/rate`, `analyze` (the component ladder); then for the contest
essay `experiment` with the winning condition, `select`, `check` and
`export`. `power` simulates a design's power before spending. Every paid
command takes `--dry-run`; `experiment --budget` and `--max-cost` cap
spend. Run long jobs under `caffeinate -i` so the machine doesn't sleep
mid-call. `export` needs [pandoc](https://pandoc.org) on the PATH.

## Design

The comparison is a component ladder: conditions that add pipeline
components one step at a time (zero-shot; structured argument with
critique and revision; the same with a debate-built argument), each
writing to the same fixed theses with the same generator and the same
writing instructions, compared by a blind pairwise judge in both
orderings. Experiment configurations with specific theses are not
included here; `configs/experiments/contest.yaml` shows the format.

Three design decisions carry most of the weight:

- **One seam.** Whole architectures are swapped by configuration, not by
  branching the codebase, so an architecture comparison is a config change.
- **Separated judging.** Generation critics, the selection judge, and a
  frozen evaluation panel are three different roles. The frozen panel never
  participates in generation, because a measuring instrument that shapes
  what it measures cannot support claims about what it measured.
- **Persisted artifacts.** Every intermediate is stored, so a single stage
  can be re-run against fixed upstream input — which turns a noisy
  end-to-end comparison into a controlled one.

## Reproducibility

Every run records its full resolved configuration, prompt versions and
git commit. Every model call records its prompt and reply, requested and
served model, provider, tokens (hidden reasoning included), billed cost,
finish reason and attempt count. Seed and temperature are sent, and
recorded as sent, only where the model supports them: the Claude 5.x
models support neither, so generation is not reproducible from a seed.
Environment reproducibility is `uv sync` against the committed lockfile. The only system dependency is pandoc, for
PDF export; the pipeline itself only makes HTTP calls and writes SQLite.

The pipeline generates text and never executes generated code.

## Licence

MIT — see [LICENSE](LICENSE).
