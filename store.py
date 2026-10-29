"""SQLite schema and write helpers. One schema, one place."""
import sqlite3
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    subset TEXT NOT NULL,
    model_version TEXT NOT NULL,
    seed INTEGER NOT NULL,
    train_sha256 TEXT NOT NULL,
    test_sha256 TEXT NOT NULL,
    created_at TEXT NOT NULL,
    metrics_json TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS predictions (
    run_id TEXT NOT NULL,
    engine_id INTEGER NOT NULL,
    cycles_observed INTEGER NOT NULL,
    pred_rul REAL NOT NULL,
    true_rul REAL NOT NULL,
    severity TEXT NOT NULL,
    PRIMARY KEY (run_id, engine_id)
);

CREATE TABLE IF NOT EXISTS maintenance_records (
    idempotency_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    subset TEXT NOT NULL,
    engine_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    rul_prediction REAL NOT NULL,
    severity TEXT NOT NULL,
    action TEXT NOT NULL,
    status TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS exceptions (
    idempotency_key TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    subset TEXT NOT NULL,
    engine_id INTEGER NOT NULL,
    cycle INTEGER NOT NULL,
    reason TEXT NOT NULL,
    created_at TEXT NOT NULL
);
"""


def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)
    return con


def write_run(con, run):
    con.execute(
        "INSERT OR REPLACE INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (run["run_id"], run["subset"], run["model_version"], run["seed"],
         run["train_sha256"], run["test_sha256"], run["created_at"],
         run["metrics_json"]),
    )


def write_predictions(con, run_id, engines):
    con.executemany(
        "INSERT OR REPLACE INTO predictions VALUES (?, ?, ?, ?, ?, ?)",
        [(run_id, e["id"], e["cycles_observed"], e["pred_rul"],
          e["true_rul"], e["severity"]) for e in engines],
    )


def write_work_orders(con, run_id, subset, rows, created_at):
    """rows: iterable of dicts with engine_id, cycle, rul, severity, action, key.
    Returns the number of newly inserted records."""
    inserted = 0
    for r in rows:
        cur = con.execute(
            "INSERT OR IGNORE INTO maintenance_records VALUES (?,?,?,?,?,?,?,?,?,?)",
            (r["key"], run_id, subset, r["engine_id"], r["cycle"], r["rul"],
             r["severity"], r["action"], "pending_submission", created_at),
        )
        inserted += cur.rowcount
    return inserted


def write_exceptions(con, run_id, subset, rows, created_at):
    inserted = 0
    for r in rows:
        cur = con.execute(
            "INSERT OR IGNORE INTO exceptions VALUES (?,?,?,?,?,?,?)",
            (r["key"], run_id, subset, r["engine_id"], r["cycle"],
             r["reason"], created_at),
        )
        inserted += cur.rowcount
    return inserted


def count(con, table):
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def now_iso():
    return datetime.now(timezone.utc).isoformat()
