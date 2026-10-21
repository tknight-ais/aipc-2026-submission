import threading
import urllib.parse
import urllib.request

import pytest

from aipc.models import Essay
from aipc.store import Judgment, Store
from aipc.viewer import rating
from aipc.viewer.server import make_server


def _judgment(a, b, shown_first, winner):
    return Judgment(
        experiment="poc", tournament_run_id="tour", thesis_id="t1", essay_a_run=a, essay_b_run=b,
        judge_model="m", shown_first_run=shown_first, winner_run=winner, reasoning="",
    )


def _judged_store(tmp_path, n_pairs, n_ties=0, freeze=True):
    """n_pairs position-consistent pairs plus n_ties pairs whose orderings
    disagree, over essays with distinct, recognisable text.
    """
    store = Store(runs_dir=tmp_path)
    store.save_config("h", "{}")
    for i in range((n_pairs + n_ties) * 2):
        run_id = f"r{i}"
        store.start_run(run_id, f"label{i}", "h", variant="A_zero_shot", started_at="t")
        store.save_artifact(run_id, "essay", Essay(text=f"## Heading\n\nEssay body {i}.", word_count=3, thesis="T"))
    for i in range(n_pairs + n_ties):
        a, b = f"r{2 * i}", f"r{2 * i + 1}"
        store.save_judgment(_judgment(a, b, a, a))
        store.save_judgment(_judgment(a, b, b, a if i < n_pairs else b))
    if freeze:
        for anchor in rating.draw("poc", store.judgments("poc")).anchors:
            store.save_anchor_pair(anchor)
    return store


def test_draw_takes_twenty_position_consistent_pairs_only(tmp_path):
    store = _judged_store(tmp_path, 25, n_ties=10, freeze=False)
    anchor_set = rating.draw("poc", store.judgments("poc"))
    assert anchor_set.decided_pairs == 25
    assert len(anchor_set.anchors) == rating.ANCHOR_COUNT
    tie_runs = {f"r{i}" for i in range(50, 70)}
    assert not any(a.essay_a_run in tie_runs for a in anchor_set.anchors)
    assert anchor_set == rating.draw("poc", store.judgments("poc"))
    assert {a.shown_first_run == a.essay_a_run for a in anchor_set.anchors} == {True, False}


def test_draw_refuses_an_unfinished_tournament(tmp_path):
    store = _judged_store(tmp_path, 2, freeze=False)
    store.save_judgment(_judgment("r0", "r3", "r0", "r0"))
    with pytest.raises(ValueError, match="finish the tournament"):
        rating.draw("poc", store.judgments("poc"))


def test_rating_serves_the_frozen_set_even_when_judgments_change(tmp_path):
    store = _judged_store(tmp_path, 25)
    before = rating.anchors(store, "poc")
    store.save_judgment(_judgment("r0", "r99", "r0", "r0"))
    assert rating.anchors(store, "poc") == before


def test_record_maps_choice_to_run_and_ignores_stale_or_bad_posts(tmp_path):
    store = _judged_store(tmp_path, 3)
    index, anchor = rating.next_unrated(store, "poc")

    rating.record(store, "poc", index + 1, "first")
    rating.record(store, "poc", index, "both")
    assert store.human_judgments("poc") == []

    rating.record(store, "poc", index, "second")
    [saved] = store.human_judgments("poc")
    assert saved.winner_run == anchor.shown_second_run
    assert saved.shown_first_run == anchor.shown_first_run
    assert rating.next_unrated(store, "poc")[0] == index + 1


@pytest.fixture
def served(tmp_path):
    _judged_store(tmp_path, 2).conn.close()
    server = make_server(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}", tmp_path
    server.shutdown()
    server.server_close()


def _post(url, fields):
    return urllib.request.urlopen(url, data=urllib.parse.urlencode(fields).encode()).read().decode()


def test_rate_page_is_blind_and_records_through_to_done(served):
    base_url, runs_dir = served
    html = urllib.request.urlopen(base_url + "/rate").read().decode()
    assert "Essay body" in html
    assert "0 of 2 rated" in html
    for leak in ["Heading", "A_zero_shot", "label", "r0", "r1"]:
        assert leak not in html

    html = _post(base_url + "/rate", {"pair": "0", "choice": "first"})
    assert "1 of 2 rated" in html
    html = _post(base_url + "/rate", {"pair": "1", "choice": "second"})
    assert "All 2 pairs rated" in html
    assert len(Store(runs_dir).human_judgments("poc")) == 2


def test_rate_page_before_any_tournament(tmp_path):
    Store(runs_dir=tmp_path).conn.close()
    server = make_server(tmp_path, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        html = urllib.request.urlopen(f"http://127.0.0.1:{server.server_address[1]}/rate").read().decode()
        assert "Run the tournament first" in html
    finally:
        server.shutdown()
        server.server_close()
