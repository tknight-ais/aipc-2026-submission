"""Pre-submission checks on the chosen essay.

No model calls. Three checks:
- total word count, references included, at most 6,000 (the contest's
  limit; no source says references are excluded);
- every bibliography entry looked up on OpenAlex, and every in-text
  citation matched to a bibliography entry;
- no text that refers to how the essay was made.

A reference counts as found when one of the top OpenAlex results has a
title whose words nearly all appear in the entry, an author whose name
contains the entry's first name token, and a year within one of the
entry's. Author and year are checked because a real title under the
wrong author and year is a common model error. The query
is the title alone where the entry is author-year style, since journal
names and page numbers pushed the right work out of the top results.
OpenAlex metadata is messy (it dates some papers a year early, and puts
mis-indexed records near the top), so the best match is
reported for a person to review, and "not found" means "check by hand",
not "fabricated".
"""

import re
import unicodedata

from pydantic import BaseModel, computed_field

from aipc.models import count_words, split_bibliography
from aipc.stages.literature import search_openalex

MAX_WORDS = 6000
TITLE_OVERLAP_FOUND = 0.8
YEAR_TOLERANCE = 1
SEARCH_RESULTS = 5
ENTRY_YEAR = re.compile(r"\b((?:19|20)\d{2})\b")
ENTRY_FIRST_NAME = re.compile(r"\s*([A-Z][\w'’-]+)")
# The year may follow other text in the brackets: "Parfit (Reasons and Persons, 1984)".
CITATION_NARRATIVE = re.compile(r"\b([A-Z][\w'’-]+)(?: et al\.)?\s+\([^()]*?\b((?:19|20)\d{2})")
CITATION_PARENTHETICAL = re.compile(
    r"[(;]\s*(?:see\s+)?(?:[A-Z][\w'’-]+\s+(?:and|&)\s+)?([A-Z][\w'’-]+)(?: et al\.)?,?\s+((?:19|20)\d{2})"
)
POSSESSIVE = re.compile(r"['’]s$")
APA_TITLE = re.compile(r"\(?\b(?:19|20)\d{2}[a-z]?\)?\.?\s*[\"“']?(.+?[.?!])[\"”']?(?:\s|$)")
PROCESS_TERMS = re.compile(
    r"\b(?:content ledger|ledger|revision round|reviser|drafter|this draft|earlier draft|"
    r"word count|word limit|sub-questions?|as an AI|I am an AI|the prompt)\b",
    re.IGNORECASE,
)


class ReferenceCheck(BaseModel):
    entry: str
    found: bool
    best_match: str


class CheckReport(BaseModel):
    body_words: int
    total_words: int
    within_word_limit: bool
    references: list[ReferenceCheck]
    unmatched_in_text: list[str]
    process_mentions: list[str]

    @computed_field
    @property
    def passed(self) -> bool:
        return (
            self.within_word_limit
            and all(r.found for r in self.references)
            and not self.unmatched_in_text
            and not self.process_mentions
        )


def split_entries(text: str) -> tuple[str, list[str]]:
    """Returns (body, bibliography entries): the non-empty lines after the
    last bibliography heading, bullets stripped.
    """
    body, bibliography = split_bibliography(text)
    entries = [line.strip().lstrip("-*0123456789.) ").strip() for line in bibliography.splitlines()]
    return body, [e for e in entries if e]


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z]+", text.lower()) if len(w) > 3}


def title_overlap(title: str, entry: str) -> float:
    title_words = _words(title)
    return len(title_words & _words(entry)) / len(title_words) if title_words else 0.0


def title_query(entry: str) -> str:
    """The title of an author-year entry, else the whole entry."""
    match = APA_TITLE.search(entry)
    return match.group(1).rstrip(".") if match else entry


def _name_tokens(name: str) -> set[str]:
    """Lower-case letter runs, accents and any kind of hyphen dropped:
    OpenAlex writes 'Lasonen‐Aarnio' with U+2010.
    """
    spaced = re.sub(r"[^\w]", " ", name)
    plain = unicodedata.normalize("NFKD", spaced).encode("ascii", "ignore").decode().lower()
    return set(re.findall(r"[a-z]+", plain))


def author_matches(entry: str, work: dict) -> bool:
    """Every token of the entry's first capitalised word (normally the first
    author's surname) is a token of one of the work's author names.
    """
    match = ENTRY_FIRST_NAME.match(entry)
    if match is None:
        return False
    surname = _name_tokens(match.group(1))
    return any(surname <= _name_tokens(a["author"]["display_name"]) for a in work.get("authorships", []))


def year_matches(entry: str, work: dict) -> bool:
    match = ENTRY_YEAR.search(entry)
    year = work.get("publication_year")
    return match is not None and year is not None and abs(int(match.group(1)) - year) <= YEAR_TOLERANCE


def check_reference(entry: str) -> ReferenceCheck:
    results = search_openalex(title_query(entry), limit=SEARCH_RESULTS)
    scored = [(title_overlap(w["display_name"] or "", entry), w) for w in results]
    if not scored:
        return ReferenceCheck(entry=entry, found=False, best_match="(no results)")
    title_hits = [w for overlap, w in scored if overlap >= TITLE_OVERLAP_FOUND]
    found = any(author_matches(entry, w) and year_matches(entry, w) for w in title_hits)
    overlap, work = max(scored, key=lambda s: s[0])
    return ReferenceCheck(
        entry=entry,
        found=found,
        best_match=(
            f"{work['display_name']} ({work.get('publication_year')}) overlap={overlap:.2f} "
            f"author={'yes' if author_matches(entry, work) else 'no'} year={'yes' if year_matches(entry, work) else 'no'}"
        ),
    )


def in_text_citations(body: str) -> set[tuple[str, str]]:
    """(surname, year) pairs. A possessive, as in "Frankfurt's (1971)", names
    the surname without its "'s"."""
    pairs = set(CITATION_NARRATIVE.findall(body)) | set(CITATION_PARENTHETICAL.findall(body))
    return {(POSSESSIVE.sub("", surname), year) for surname, year in pairs}


def unmatched_citations(body: str, entries: list[str]) -> list[str]:
    """In-text (surname, year) pairs with no bibliography entry naming both."""
    return sorted(
        f"{surname} {year}"
        for surname, year in in_text_citations(body)
        if not any(surname in e and year in e for e in entries)
    )


def process_mentions(body: str) -> list[str]:
    return [body[max(0, m.start() - 60) : m.end() + 60].replace("\n", " ") for m in PROCESS_TERMS.finditer(body)]


def run(text: str) -> CheckReport:
    body, entries = split_entries(text)
    total_words = len(text.split())
    return CheckReport(
        body_words=count_words(body),
        total_words=total_words,
        within_word_limit=total_words <= MAX_WORDS,
        references=[check_reference(e) for e in entries],
        unmatched_in_text=unmatched_citations(body, entries),
        process_mentions=process_mentions(body),
    )
