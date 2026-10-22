from aipc.stages import checks

ESSAY = """# Title

Frankfurt (1971) argues that freedom needs only the right will. Others disagree (Fischer and Ravizza 2024; see Parfit, 2011).

## References

- Frankfurt, H. (1971). Freedom of the Will and the Concept of a Person. Journal of Philosophy.
- Parfit, D. (2011). On What Matters. Oxford University Press.
"""


def _fake_search(results):
    def search(query, limit=8):
        return results

    return search


def test_split_entries_excludes_references_from_body():
    body, entries = checks.split_entries(ESSAY)
    assert "Philosophical Studies" not in body
    assert len(entries) == 2 and entries[0].startswith("Frankfurt")


def test_essay_without_bibliography_is_all_body():
    assert checks.split_entries("Just text.") == ("Just text.", [])


def test_in_text_citation_without_entry_is_flagged():
    body, entries = checks.split_entries(ESSAY)
    assert checks.unmatched_citations(body, entries) == ["Ravizza 2024"]


def test_reference_found_when_a_title_matches(monkeypatch):
    monkeypatch.setattr(
        checks,
        "search_openalex",
        _fake_search(
            [
                {"display_name": "Something else entirely", "publication_year": 2001},
                {
                    "display_name": "Freedom of the Will and the Concept of a Person",
                    "publication_year": 1970,
                    "authorships": [{"author": {"display_name": "Harry G. Frankfurt"}}],
                },
            ]
        ),
    )
    result = checks.check_reference("Frankfurt, H. (1971). Freedom of the Will and the Concept of a Person.")
    assert result.found
    assert "(1970)" in result.best_match


def test_reference_not_found_when_no_title_matches(monkeypatch):
    monkeypatch.setattr(
        checks, "search_openalex", _fake_search([{"display_name": "Unrelated work on bees", "publication_year": 2001}])
    )
    assert not checks.check_reference("Invented, A. (2020). The Freedom of Machines.").found


def test_process_mentions_are_flagged():
    assert checks.process_mentions("The objection, raised in the revision round, fails.")
    assert not checks.process_mentions("Critics of compatibilism object that it fails.")


def test_run_reports_word_count_and_overall_result(monkeypatch):
    monkeypatch.setattr(checks, "search_openalex", _fake_search([]))
    report = checks.run(ESSAY)
    assert report.within_word_limit
    assert report.body_words < 30
    assert not report.passed
    assert report.model_dump()["passed"] is False


def test_title_query_extracts_author_year_titles():
    assert (
        checks.title_query("Parfit, D. (1984). Reasons and Persons. Oxford University Press.") == "Reasons and Persons"
    )
    assert checks.title_query("Smith, M. (1994). Why Be Moral? Blackwell.") == "Why Be Moral?"
    assert checks.title_query("No year here. A Title.") == "No year here. A Title."
    quoted = 'Frankfurt, Harry (1971). "Freedom of the Will and the Concept of a Person." Journal of Philosophy 68(1).'
    assert checks.title_query(quoted) == "Freedom of the Will and the Concept of a Person"


def test_count_words_excludes_the_bibliography():
    from aipc.models import count_words

    assert count_words("One two three.\n\nReferences\n\nFrankfurt, H. (1971). A Title.") == 3


def test_word_limit_counts_references(monkeypatch):
    monkeypatch.setattr(checks, "search_openalex", _fake_search([]))
    references = "\n".join(f"Author{i}, A. (2000). Title number {i}. Journal." for i in range(200))
    report = checks.run("word " * 5900 + "\n\nReferences\n\n" + references)
    assert report.body_words == 5900
    assert report.total_words > checks.MAX_WORDS
    assert not report.within_word_limit


def test_real_title_under_the_wrong_author_and_year_is_not_found(monkeypatch):
    monkeypatch.setattr(
        checks,
        "search_openalex",
        _fake_search(
            [
                {
                    "display_name": "Freedom of the Will and the Concept of a Person",
                    "publication_year": 1971,
                    "authorships": [{"author": {"display_name": "Harry G. Frankfurt"}}],
                },
            ]
        ),
    )
    result = checks.check_reference(
        "Smith, J. (2019). Freedom of the Will and the Concept of a Person. Journal of Moral Studies."
    )
    assert not result.found
    assert "author=no year=no" in result.best_match


def test_author_match_ignores_hyphen_style_but_not_partial_words():
    work = {"authorships": [{"author": {"display_name": "Maria Lasonen\u2010Aarnio"}}]}
    assert checks.author_matches("Lasonen-Aarnio, Maria. 2014. Higher-Order Evidence.", work)
    assert not checks.author_matches(
        "Li, J. 2020. A Title.", {"authorships": [{"author": {"display_name": "Elizabeth Smith"}}]}
    )
