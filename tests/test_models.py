import pytest

from aipc.models import Essay, count_words


def test_content_ledger_fixture_loads(sample_ledger):
    assert sample_ledger.thesis
    assert len(sample_ledger.claims) == 2
    assert sample_ledger.objections[0].target_claim_id == "c2"
    assert sample_ledger.objections[0].reply_type == "decisive"


def test_essay_round_trips_json():
    essay = Essay(text="a b c", word_count=3, thesis="metaethics")
    restored = Essay.model_validate_json(essay.model_dump_json())
    assert restored == essay


@pytest.mark.parametrize(
    "heading", ["References", "## References", "**References**", "**References:**", "References:", "Reference List"]
)
def test_count_words_excludes_the_bibliography_under_any_common_heading(heading):
    text = f"A title\n\n{'word ' * 100}\n\n{heading}\nParfit, D. 1984. Reasons and Persons. Oxford University Press."
    assert count_words(text) == 102
