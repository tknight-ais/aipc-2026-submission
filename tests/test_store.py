import sqlite3

from aipc.models import Essay
from aipc.store import Store


def test_start_finish_run_and_save_artifact(tmp_path):
    store = Store(runs_dir=tmp_path)
    store.save_config("hash1", '{"a": 1}')
    store.start_run(
        "run1",
        "20260930-000000_baseline_metaethics_run1abc",
        "hash1",
        variant="baseline",
        started_at="2026-09-30T00:00:00Z",
    )

    essay = Essay(text="hello world", word_count=2, thesis="metaethics")
    path = store.save_artifact("run1", "essay", essay, markdown_text=essay.text)
    assert path.exists()
    assert path.read_text().strip().startswith("{")
    assert path.parent.name == "20260930-000000_baseline_metaethics_run1abc"

    md_path = path.with_suffix(".md")
    assert md_path.read_text() == "hello world"

    store.finish_run("run1", "2026-09-30T00:01:00Z", cost_usd=0.05, status="ok")

    conn = sqlite3.connect(tmp_path / "runs.db")
    row = conn.execute("SELECT status, cost_usd FROM runs WHERE run_id = ?", ("run1",)).fetchone()
    assert row == ("ok", 0.05)

    artifact_row = conn.execute("SELECT stage, type, location FROM artifacts WHERE run_id = ?", ("run1",)).fetchone()
    assert artifact_row == ("essay", "Essay", str(path))


def test_an_older_database_gains_the_columns_added_since(tmp_path):
    conn = sqlite3.connect(tmp_path / "runs.db")
    conn.execute("CREATE TABLE judgments (id INTEGER PRIMARY KEY AUTOINCREMENT, experiment TEXT NOT NULL)")
    conn.commit()
    conn.close()

    store = Store(runs_dir=tmp_path)

    judgment_columns = {row[1] for row in store.conn.execute("PRAGMA table_info(judgments)")}
    assert "judge_prompt_hash" in judgment_columns


def test_a_new_database_has_every_column_without_migrating(tmp_path):
    from aipc import store as store_module

    conn = sqlite3.connect(tmp_path / "fresh.db")
    conn.executescript(store_module.SCHEMA)
    for table, columns in store_module.ADDED_COLUMNS.items():
        existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        assert set(columns) <= existing, table
