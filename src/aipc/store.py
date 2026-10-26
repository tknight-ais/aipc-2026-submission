"""SQLite run/artifact persistence.

Artifacts are written as JSON files under runs/<run_label>/<stage>.json,
with a database row pointing at each one, so any stage can be replayed
against a fixed upstream artifact. run_label is a human-readable,
timestamp-sortable directory name; run_id (a uuid) stays the database
primary key. Every model call is a row in `calls`, with its full prompt
and reply. Scores are stored long and unaggregated - one row per
persona-criterion - because averaging at write time destroys the
disagreement signal.
"""

import sqlite3
import threading
from pathlib import Path

from pydantic import BaseModel

from aipc.llm import CallRecord
from aipc.models import Scorecard

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    label TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    variant TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    cost_usd REAL NOT NULL DEFAULT 0,
    status TEXT NOT NULL,
    git_commit TEXT,
    git_dirty INTEGER
);

CREATE TABLE IF NOT EXISTS configs (
    config_hash TEXT PRIMARY KEY,
    config_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    stage TEXT NOT NULL,
    type TEXT NOT NULL,
    location TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS scores (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    essay_id TEXT NOT NULL,
    persona TEXT NOT NULL,
    panel_type TEXT NOT NULL,
    criterion TEXT NOT NULL,
    score REAL NOT NULL,
    reasoning TEXT
);

CREATE TABLE IF NOT EXISTS comparisons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    essay_a TEXT NOT NULL,
    essay_b TEXT NOT NULL,
    judge TEXT NOT NULL,
    winner TEXT NOT NULL,
    order_label TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS prompts (
    hash TEXT PRIMARY KEY,
    agent_type TEXT NOT NULL,
    version TEXT NOT NULL,
    text TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS experiment_essays (
    experiment TEXT NOT NULL,
    thesis_id TEXT NOT NULL,
    condition TEXT NOT NULL,
    essay_index INTEGER NOT NULL,
    run_id TEXT NOT NULL,
    essay_path TEXT NOT NULL,
    PRIMARY KEY (experiment, thesis_id, condition, essay_index)
);

CREATE TABLE IF NOT EXISTS judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    tournament_run_id TEXT NOT NULL,
    thesis_id TEXT NOT NULL,
    essay_a_run TEXT NOT NULL,
    essay_b_run TEXT NOT NULL,
    judge_model TEXT NOT NULL,
    shown_first_run TEXT NOT NULL,
    winner_run TEXT,
    reasoning TEXT NOT NULL,
    judge_prompt_hash TEXT
);

CREATE TABLE IF NOT EXISTS human_judgments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    essay_a_run TEXT NOT NULL,
    essay_b_run TEXT NOT NULL,
    shown_first_run TEXT NOT NULL,
    winner_run TEXT NOT NULL,
    rated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS anchor_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment TEXT NOT NULL,
    anchor_index INTEGER NOT NULL,
    essay_a_run TEXT NOT NULL,
    essay_b_run TEXT NOT NULL,
    shown_first_run TEXT NOT NULL,
    UNIQUE (experiment, anchor_index)
);

CREATE TABLE IF NOT EXISTS calls (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    prompt_name TEXT,
    tier TEXT NOT NULL,
    requested_model TEXT NOT NULL,
    served_model TEXT NOT NULL,
    provider TEXT,
    generation_id TEXT NOT NULL,
    temperature REAL NOT NULL,
    seed INTEGER NOT NULL,
    prompt_tokens INTEGER NOT NULL,
    completion_tokens INTEGER NOT NULL,
    cost_usd REAL NOT NULL,
    billed_usd REAL,
    latency_s REAL NOT NULL,
    called_at TEXT NOT NULL,
    prompt TEXT NOT NULL,
    response TEXT NOT NULL,
    sampling_sent TEXT,
    reasoning_tokens INTEGER,
    finish_reason TEXT,
    attempts INTEGER,
    replayed_from TEXT
);

CREATE TABLE IF NOT EXISTS measurements (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT NOT NULL,
    metric TEXT NOT NULL,
    value REAL NOT NULL
);
"""


class ExperimentEssay(BaseModel):
    thesis_id: str
    condition: str
    essay_index: int
    run_id: str
    essay_path: str

    @property
    def name(self) -> str:
        return f"{self.thesis_id}/{self.condition}/{self.essay_index}"


class Judgment(BaseModel):
    experiment: str
    tournament_run_id: str
    thesis_id: str
    essay_a_run: str
    essay_b_run: str
    judge_model: str
    shown_first_run: str
    winner_run: str | None
    reasoning: str
    # None for judgments made before 2026-10-21, when it was not recorded.
    judge_prompt_hash: str | None = None


class AnchorPair(BaseModel):
    """One pair of the frozen blind-rating set, in its on-screen order."""

    experiment: str
    anchor_index: int
    essay_a_run: str
    essay_b_run: str
    shown_first_run: str

    @property
    def shown_second_run(self) -> str:
        return self.essay_b_run if self.shown_first_run == self.essay_a_run else self.essay_a_run


class HumanJudgment(BaseModel):
    experiment: str
    essay_a_run: str
    essay_b_run: str
    shown_first_run: str
    winner_run: str
    rated_at: str


# Columns added after runs.db files already existed. SCHEMA has them for a
# new database; an older one gets them when it is opened.
ADDED_COLUMNS = {
    "runs": {"git_commit": "TEXT", "git_dirty": "INTEGER"},
    "calls": {
        "sampling_sent": "TEXT",
        "reasoning_tokens": "INTEGER",
        "finish_reason": "TEXT",
        "attempts": "INTEGER",
        "replayed_from": "TEXT",
    },
    "judgments": {"judge_prompt_hash": "TEXT"},
}


class Store:
    """One connection, shared with the critic threads; writes take a lock."""

    def __init__(self, runs_dir: Path):
        self.runs_dir = runs_dir
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.runs_dir / "runs.db", check_same_thread=False)
        self.conn.executescript(SCHEMA)
        self._add_missing_columns()
        self._lock = threading.Lock()
        self._run_dirs: dict[str, Path] = {}

    def _add_missing_columns(self) -> None:
        for table, columns in ADDED_COLUMNS.items():
            existing = {row[1] for row in self.conn.execute(f"PRAGMA table_info({table})")}
            for name, sql_type in columns.items():
                if name not in existing:
                    self.conn.execute(f"ALTER TABLE {table} ADD COLUMN {name} {sql_type}")
        self.conn.commit()

    def _write(self, sql: str, params: tuple) -> None:
        with self._lock:
            self.conn.execute(sql, params)
            self.conn.commit()

    def save_config(self, config_hash: str, config_json: str) -> None:
        self._write(
            "INSERT OR IGNORE INTO configs (config_hash, config_json) VALUES (?, ?)",
            (config_hash, config_json),
        )

    def save_prompt(self, prompt_hash: str, agent_type: str, version: str, text: str) -> None:
        self._write(
            "INSERT OR IGNORE INTO prompts (hash, agent_type, version, text) VALUES (?, ?, ?, ?)",
            (prompt_hash, agent_type, version, text),
        )

    def start_run(
        self,
        run_id: str,
        label: str,
        config_hash: str,
        variant: str,
        started_at: str,
        git_commit: str | None = None,
        git_dirty: bool | None = None,
    ) -> None:
        self._write(
            "INSERT INTO runs (run_id, label, config_hash, variant, started_at, status, git_commit, git_dirty) "
            "VALUES (?, ?, ?, ?, ?, 'running', ?, ?)",
            (run_id, label, config_hash, variant, started_at, git_commit, git_dirty),
        )
        self._run_dirs[run_id] = self.runs_dir / label

    def finish_run(self, run_id: str, finished_at: str, cost_usd: float, status: str) -> None:
        self._write(
            "UPDATE runs SET finished_at = ?, cost_usd = ?, status = ? WHERE run_id = ?",
            (finished_at, cost_usd, status, run_id),
        )

    def save_score(self, run_id: str, scorecard: Scorecard) -> None:
        self._write(
            "INSERT INTO scores (run_id, essay_id, persona, panel_type, criterion, score, reasoning) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                run_id,
                scorecard.essay_id,
                scorecard.persona,
                scorecard.panel_type,
                scorecard.criterion,
                scorecard.score,
                scorecard.reasoning,
            ),
        )

    def save_comparison(self, essay_a: str, essay_b: str, judge: str, winner: str, order_label: str) -> None:
        self._write(
            "INSERT INTO comparisons (essay_a, essay_b, judge, winner, order_label) VALUES (?, ?, ?, ?, ?)",
            (essay_a, essay_b, judge, winner, order_label),
        )

    def save_measurement(self, run_id: str, metric: str, value: float) -> None:
        self._write(
            "INSERT INTO measurements (run_id, metric, value) VALUES (?, ?, ?)",
            (run_id, metric, value),
        )

    def save_call(self, run_id: str, call: CallRecord) -> None:
        self._insert("calls", call, run_id=run_id)

    def calls(self, run_id: str) -> list[CallRecord]:
        """A run's recorded calls, in the order they were made."""
        cursor = self.conn.execute("SELECT * FROM calls WHERE run_id = ? ORDER BY id", (run_id,))
        names = [d[0] for d in cursor.description]
        rows = [dict(zip(names, row)) for row in cursor.fetchall()]
        return [CallRecord(**{k: v for k, v in row.items() if k not in ("id", "run_id")}) for row in rows]

    def save_artifact(self, run_id: str, stage: str, artifact: BaseModel, markdown_text: str | None = None) -> Path:
        run_dir = self._run_dirs[run_id]
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / f"{stage}.json"
        path.write_text(artifact.model_dump_json(indent=2))
        if markdown_text is not None:
            (run_dir / f"{stage}.md").write_text(markdown_text)
        self._write(
            "INSERT INTO artifacts (run_id, stage, type, location) VALUES (?, ?, ?, ?)",
            (run_id, stage, type(artifact).__name__, str(path)),
        )
        return path

    def save_experiment_essay(
        self, experiment: str, thesis_id: str, condition: str, essay_index: int, run_id: str, essay_path: Path
    ) -> None:
        self._write(
            "INSERT INTO experiment_essays (experiment, thesis_id, condition, essay_index, run_id, essay_path) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (experiment, thesis_id, condition, essay_index, run_id, str(essay_path)),
        )

    def experiment_essays(self, experiment: str) -> list[ExperimentEssay]:
        rows = self.conn.execute(
            "SELECT thesis_id, condition, essay_index, run_id, essay_path FROM experiment_essays "
            "WHERE experiment = ? ORDER BY thesis_id, condition, essay_index",
            (experiment,),
        ).fetchall()
        return [
            ExperimentEssay(thesis_id=r[0], condition=r[1], essay_index=r[2], run_id=r[3], essay_path=r[4])
            for r in rows
        ]

    def save_judgment(self, judgment: Judgment) -> None:
        self._insert("judgments", judgment)

    def judgments(self, experiment: str) -> list[Judgment]:
        return [Judgment(**row) for row in self._select("judgments", experiment)]

    def save_anchor_pair(self, anchor: AnchorPair) -> None:
        self._insert("anchor_pairs", anchor)

    def anchor_pairs(self, experiment: str) -> list[AnchorPair]:
        return [AnchorPair(**row) for row in self._select("anchor_pairs", experiment)]

    def save_human_judgment(self, judgment: HumanJudgment) -> None:
        self._insert("human_judgments", judgment)

    def human_judgments(self, experiment: str) -> list[HumanJudgment]:
        return [HumanJudgment(**row) for row in self._select("human_judgments", experiment)]

    def _insert(self, table: str, row: BaseModel, **extra) -> None:
        data = extra | row.model_dump()
        columns = ", ".join(data)
        placeholders = ", ".join("?" for _ in data)
        self._write(f"INSERT INTO {table} ({columns}) VALUES ({placeholders})", tuple(data.values()))

    def _select(self, table: str, experiment: str) -> list[dict]:
        cursor = self.conn.execute(f"SELECT * FROM {table} WHERE experiment = ? ORDER BY id", (experiment,))
        names = [d[0] for d in cursor.description]
        return [dict(zip(names, row)) for row in cursor.fetchall()]
