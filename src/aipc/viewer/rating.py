"""Blind anchor ratings: the human check on the judge.

The anchor set is drawn once, after the tournament is complete, by
`aipc draw-anchors`, and stored; the rating page serves that frozen set,
so judgments added mid-rating cannot change it.

Anchors are drawn only from pairs the judge decided the same way in both
orderings. A judge tie counts against agreement, so anchors drawn from
all pairs could make the agreement bar unreachable whatever the rater
chose.

The rater sees only normalised essay text: no run ids, conditions or judge
verdicts. The browser posts back a pair index and "first" or "second";
the pair itself is always looked up here, never taken from the request.

This is the viewer's one write path, limited to the human_judgments table.
"""

import random
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from aipc.blind import normalise
from aipc.models import Essay
from aipc.stages.analysis import pair_outcomes
from aipc.store import AnchorPair, HumanJudgment, Judgment, Store

ANCHOR_COUNT = 20
ANCHOR_SEED = 1


class AnchorSet(BaseModel):
    """The frozen draw, as saved in the draw-anchors run."""

    experiment: str
    decided_pairs: int
    anchors: list[AnchorPair]


def draw(experiment: str, judgments: list[Judgment]) -> AnchorSet:
    """A seeded draw of up to ANCHOR_COUNT position-consistent pairs, each
    with a seeded on-screen order. Raises, through pair_outcomes, if the
    tournament isn't finished.
    """
    decided = sorted((o.essay_a_run, o.essay_b_run) for o in pair_outcomes(judgments) if o.position_consistent)
    rng = random.Random(ANCHOR_SEED)
    drawn = rng.sample(decided, min(ANCHOR_COUNT, len(decided)))
    anchors = [
        AnchorPair(
            experiment=experiment, anchor_index=i, essay_a_run=a, essay_b_run=b, shown_first_run=rng.choice([a, b])
        )
        for i, (a, b) in enumerate(drawn)
    ]
    return AnchorSet(experiment=experiment, decided_pairs=len(decided), anchors=anchors)


def anchors(store: Store, experiment: str) -> list[AnchorPair]:
    return store.anchor_pairs(experiment)


def next_unrated(store: Store, experiment: str) -> tuple[int, AnchorPair] | None:
    rated = {(h.essay_a_run, h.essay_b_run) for h in store.human_judgments(experiment)}
    for anchor in anchors(store, experiment):
        if (anchor.essay_a_run, anchor.essay_b_run) not in rated:
            return anchor.anchor_index, anchor
    return None


def essay_text(store: Store, runs_dir: Path, run_id: str) -> str:
    """Normalised text, read from runs_dir/<label>/essay.json."""
    label = store.conn.execute("SELECT label FROM runs WHERE run_id = ?", (run_id,)).fetchone()[0]
    return normalise(Essay.model_validate_json((runs_dir / label / "essay.json").read_text()).text)


def record(store: Store, experiment: str, index: int, choice: str) -> None:
    """Saves the rater's pick for the anchor at `index`, if it is the next unrated one."""
    pending = next_unrated(store, experiment)
    if pending is None or pending[0] != index or choice not in ("first", "second"):
        return
    anchor = pending[1]
    store.save_human_judgment(
        HumanJudgment(
            experiment=experiment,
            essay_a_run=anchor.essay_a_run,
            essay_b_run=anchor.essay_b_run,
            shown_first_run=anchor.shown_first_run,
            winner_run=anchor.shown_first_run if choice == "first" else anchor.shown_second_run,
            rated_at=datetime.now(UTC).isoformat(),
        )
    )
