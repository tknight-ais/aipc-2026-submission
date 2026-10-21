"""Pydantic artifact types: the contracts stages are written against, and
the shapes persisted as JSON under runs/.
"""

import re
from typing import Literal

from pydantic import BaseModel


class ThesisCandidate(BaseModel):
    claim: str
    origin: Literal["cold", "informed"]
    rationale: str


class LiteraturePack(BaseModel):
    background: str
    queries: list[str] = []
    sources: list[str] = []
    engagement_list: list[str] = []


class FilterVerdict(BaseModel):
    passed: bool
    reasoning: str


class ScreenedThesis(BaseModel):
    candidate: ThesisCandidate
    significance: FilterVerdict
    # None when the significance filter already rejected the candidate -
    # novelty is the more expensive (retrieval-backed) filter, spent only
    # on candidates already judged worth arguing.
    novelty: FilterVerdict | None = None

    @property
    def survived(self) -> bool:
        """Passed both filters."""
        return self.significance.passed and self.novelty is not None and self.novelty.passed


class Decomposition(BaseModel):
    thesis: str
    sub_questions: list[str]


class ScreenedDecomposition(BaseModel):
    decomposition: Decomposition
    validity: FilterVerdict


class DecompositionSelection(BaseModel):
    chosen_index: int
    reasoning: str


class DebateTurn(BaseModel):
    speaker: str
    text: str


class DebateTranscript(BaseModel):
    sub_question: str
    turns: list[DebateTurn]
    converged: bool
    rounds_used: int


class Claim(BaseModel):
    id: str
    statement: str
    support: str
    role: str


class Objection(BaseModel):
    statement: str
    target_claim_id: str
    reply: str
    reply_type: Literal["decisive", "concessive"]


class ContentLedger(BaseModel):
    thesis: str
    claims: list[Claim] = []
    objections: list[Objection] = []
    concessions: list[str] = []
    engagements: list[str] = []


BIBLIOGRAPHY_HEADING = re.compile(
    r"^\s*(?:#+\s*)?(?:\*\*)?(?:references|bibliography|works cited)(?:\*\*)?:?\s*$",
    re.IGNORECASE | re.MULTILINE,
)


def split_bibliography(text: str) -> tuple[str, str]:
    """Returns (body, bibliography), split at the last bibliography heading."""
    headings = list(BIBLIOGRAPHY_HEADING.finditer(text))
    if not headings:
        return text, ""
    return text[:headings[-1].start()], text[headings[-1].end():]


def count_words(text: str) -> int:
    """Words in the body; the contest's 6,000-word cap excludes the bibliography."""
    return len(split_bibliography(text)[0].split())


class Essay(BaseModel):
    text: str
    word_count: int
    # The specific claim argued, not the topic area handed in. None where
    # the stage that produced the essay has no way to know it (the bare
    # baseline prompt lets the model pick its own thesis from a topic
    # area and never extracts what it chose).
    thesis: str | None
    rounds_used: int = 0

    @classmethod
    def from_text(cls, text: str, thesis: str | None) -> "Essay":
        return cls(text=text, word_count=count_words(text), thesis=thesis)


class Critique(BaseModel):
    critic_id: int
    text: str


class RevisionRound(BaseModel):
    round_num: int
    critiques: list[Critique]
    has_new_objection: bool
    word_count: int
    hedge_density: float


class FidelityReview(BaseModel):
    # Raw model output, not parsed structure - the same choice baseline.py
    # and draft.py make for essay text. "none found" is a valid value, not
    # an absence of one.
    dropped_objections: str
    unsupported_claims: str


class Persona(BaseModel):
    name: str
    description: str
    model_tier: str = "cheap"


class Scorecard(BaseModel):
    essay_id: str
    persona: str
    panel_type: str
    criterion: str
    score: float
    reasoning: str
