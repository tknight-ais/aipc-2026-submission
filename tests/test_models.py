from aipc.models import Essay


def test_content_ledger_fixture_loads(sample_ledger):
    assert sample_ledger.thesis
    assert len(sample_ledger.claims) == 2
    assert sample_ledger.objections[0].target_claim_id == "c2"
    assert sample_ledger.objections[0].reply_type == "decisive"


def test_essay_round_trips_json():
    essay = Essay(text="a b c", word_count=3, thesis="metaethics")
    restored = Essay.model_validate_json(essay.model_dump_json())
    assert restored == essay
