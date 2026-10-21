"""Readable report of a tournament's verdicts: a summary of how the judge
behaved, then every pair with both orderings and the judge's reasoning.

Not blind: it names conditions. Don't show it to anyone who still has
blind ratings to do on the same experiment.
"""

from collections import defaultdict

from aipc.models import Essay
from aipc.store import ExperimentEssay, Judgment

CONCLUSION_CHARS = 700


def _conclusion(reasoning: str) -> str:
    """The end of the judge's reasoning, starting at a sentence boundary."""
    tail = reasoning[-CONCLUSION_CHARS:]
    start = tail.find(". ") + 2 if len(reasoning) > CONCLUSION_CHARS and ". " in tail else 0
    return tail[start:].strip()


def _title(essay: Essay) -> str:
    return essay.text.strip().splitlines()[0].strip("# *")[:120]


def _summary(judgments: list[Judgment], by_pair: dict) -> list[str]:
    decided = [j for j in judgments if j.winner_run is not None]
    first_picked = sum(j.winner_run == j.shown_first_run for j in decided)
    consistent = sum(len({j.winner_run for j in js}) == 1 and js[0].winner_run is not None for js in by_pair.values())
    return [
        "## Summary",
        "",
        f"- Pairs: {len(by_pair)}; judgements: {len(judgments)}; unreadable: {len(judgments) - len(decided)}",
        f"- Same winner in both orderings: {consistent}/{len(by_pair)} ({consistent / len(by_pair):.0%})",
        f"- Picked the essay shown first: {first_picked}/{len(decided)} ({first_picked / len(decided):.0%}); 50% means no position bias",
        "",
    ]


def _pair_section(index: int, js: list[Judgment], essays: dict[str, ExperimentEssay], texts: dict[str, Essay]) -> list[str]:
    a, b = js[0].essay_a_run, js[0].essay_b_run
    winners = {j.winner_run for j in js}
    outcome = essays[next(iter(winners))].condition if len(winners) == 1 and None not in winners else "tie (orderings disagree)"
    lines = [f"## {index}. {essays[a].condition} vs {essays[b].condition} ({js[0].thesis_id}): {outcome}", ""]
    for run in (a, b):
        lines.append(f"- **{essays[run].name}**: {_title(texts[run])} ({texts[run].word_count} words)")
    lines.append("")
    for j in js:
        shown_first = essays[j.shown_first_run].condition
        winner = essays[j.winner_run].condition if j.winner_run else "unreadable"
        lines += [
            f"**Shown first: {shown_first}. Picked: {winner}.**",
            "",
            f"> {_conclusion(j.reasoning)}".replace("\n", "\n> "),
            "",
            "<details><summary>Full reasoning</summary>",
            "",
            j.reasoning,
            "",
            "</details>",
            "",
        ]
    return lines


def build(experiment: str, judgments: list[Judgment], essays: list[ExperimentEssay], texts: dict[str, Essay]) -> str:
    """Markdown report: summary first, then each pair in thesis order."""
    by_run = {e.run_id: e for e in essays}
    by_pair: dict[tuple[str, str], list[Judgment]] = defaultdict(list)
    for j in judgments:
        by_pair[(j.essay_a_run, j.essay_b_run)].append(j)
    pairs = sorted(by_pair.values(), key=lambda js: (js[0].thesis_id, by_run[js[0].essay_a_run].name))

    lines = [f"# Judge verdicts: {experiment}", "", f"Judge: {judgments[0].judge_model}", ""]
    lines += _summary(judgments, by_pair)
    for i, js in enumerate(pairs, start=1):
        lines += _pair_section(i, js, by_run, texts)
    return "\n".join(lines)
