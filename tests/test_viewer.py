import threading
import urllib.error
import urllib.request

import pytest

from aipc.models import Essay
from aipc.store import Store
from aipc.viewer.server import make_server

LABEL = "20260930-000000_baseline_metaethics_run1abc"


@pytest.fixture
def runs_dir(tmp_path):
    store = Store(runs_dir=tmp_path)
    store.save_config("hash1", '{"strategy": "flat"}')
    store.start_run("run1", LABEL, "hash1", variant="baseline", started_at="2026-09-30T00:00:00Z")
    essay = Essay(text="hello <b>world</b>", word_count=2, thesis="metaethics")
    store.save_artifact("run1", "essay", essay, markdown_text=essay.text)
    store.finish_run("run1", "2026-09-30T00:01:00Z", cost_usd=0.05, status="ok")
    store.conn.close()
    return tmp_path


@pytest.fixture
def base_url(runs_dir):
    server = make_server(runs_dir, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def get(url: str) -> str:
    return urllib.request.urlopen(url).read().decode()


def test_index_lists_run(base_url):
    html = get(base_url + "/")
    assert LABEL in html
    assert "baseline" in html


def test_run_page_shows_stage_config_and_escapes_text(base_url):
    html = get(f"{base_url}/run/{LABEL}")
    assert "essay" in html
    assert "strategy" in html
    assert "hello &lt;b&gt;world&lt;/b&gt;" in html


def test_unknown_run_is_404(base_url):
    with pytest.raises(urllib.error.HTTPError) as err:
        get(base_url + "/run/nope")
    assert err.value.code == 404
