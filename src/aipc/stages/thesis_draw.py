"""Draws the component ladder's fixed theses from an ideation run (step 1).

A human chooses only the topic, so the theses are drawn, not picked: the
survivors of both ideation filters, sampled with a fixed seed. Ideation
candidates have converged on near-duplicates before, so a draw whose
theses overlap too much is rejected and the next seed tried. Overlap is
word-set Jaccard similarity: crude, but fixed in advance and logged.

MAX_OVERLAP is set from earlier ideation runs: rewordings of one thesis
scored 0.31-0.37, distinct theses 0.07-0.23. Long
claims share few words even when they say the same thing, so a higher
threshold never fires. 16 pairs is a small sample.
"""

import itertools
import random
import re
from pathlib import Path

from pydantic import BaseModel

from aipc.models import ScreenedThesis

MAX_OVERLAP = 0.3
MAX_SEED_TRIES = 20


class ThesisDraw(BaseModel):
    theses: list[str]
    seed: int
    rejected_seeds: list[int]
    survivors: int


def load_survivors(run_dir: Path) -> list[str]:
    """Claims that passed both significance and novelty, in candidate order."""
    paths = sorted(run_dir.glob("candidate*.json"), key=lambda p: int(p.stem.removeprefix("candidate")))
    screened = [ScreenedThesis.model_validate_json(p.read_text()) for p in paths]
    return [s.candidate.claim for s in screened if s.survived]


def overlap(a: str, b: str) -> float:
    words_a, words_b = (set(re.findall(r"[a-z']+", text.lower())) for text in (a, b))
    return len(words_a & words_b) / len(words_a | words_b)


def draw(survivors: list[str], n: int, seed: int) -> ThesisDraw:
    if len(survivors) < n:
        raise ValueError(f"only {len(survivors)} survivors; need {n}")
    rejected = []
    for try_seed in range(seed, seed + MAX_SEED_TRIES):
        theses = random.Random(try_seed).sample(survivors, n)
        if all(overlap(a, b) <= MAX_OVERLAP for a, b in itertools.combinations(theses, 2)):
            return ThesisDraw(theses=theses, seed=try_seed, rejected_seeds=rejected, survivors=len(survivors))
        rejected.append(try_seed)
    raise ValueError(f"every draw from seed {seed} to {seed + MAX_SEED_TRIES - 1} had near-duplicate theses")
