"""Read-only access to runs.db and artifact files for the viewer.

Artifact paths are rebuilt from runs_dir/label/stage.json rather than
taken from the stored `location`, which is relative to whatever working
directory the run happened in.
"""

import json
import sqlite3
from contextlib import closing
from pathlib import Path


def _connect(runs_dir: Path) -> sqlite3.Connection:
    db_path = (runs_dir / "runs.db").resolve()
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _read_json(runs_dir: Path, label: str, stage: str) -> dict:
    return json.loads((runs_dir / label / f"{stage}.json").read_text())


def _essay_word_count(runs_dir: Path, label: str) -> int | None:
    path = runs_dir / label / "essay.json"
    return json.loads(path.read_text())["word_count"] if path.exists() else None


def list_runs(runs_dir: Path) -> list[dict]:
    """All runs, newest first, with the final essay's word count if any."""
    with closing(_connect(runs_dir)) as conn:
        rows = conn.execute("SELECT * FROM runs ORDER BY started_at DESC").fetchall()
    return [dict(row) | {"word_count": _essay_word_count(runs_dir, row["label"])} for row in rows]


def get_run(runs_dir: Path, label: str) -> dict | None:
    with closing(_connect(runs_dir)) as conn:
        row = conn.execute("SELECT * FROM runs WHERE label = ?", (label,)).fetchone()
    return dict(row) if row else None


def get_config(runs_dir: Path, config_hash: str) -> dict:
    with closing(_connect(runs_dir)) as conn:
        row = conn.execute(
            "SELECT config_json FROM configs WHERE config_hash = ?", (config_hash,)
        ).fetchone()
    return json.loads(row["config_json"])


def get_stages(runs_dir: Path, run: dict) -> list[dict]:
    """The run's artifacts in the order the pipeline wrote them."""
    with closing(_connect(runs_dir)) as conn:
        rows = conn.execute(
            "SELECT stage, type FROM artifacts WHERE run_id = ? ORDER BY id", (run["run_id"],)
        ).fetchall()
    return [
        {"stage": row["stage"], "type": row["type"], "data": _read_json(runs_dir, run["label"], row["stage"])}
        for row in rows
    ]
