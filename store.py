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

CREATE TABLE IF NOT EXISTS search_trials (
    run_id TEXT NOT NULL,
    trial_index INTEGER NOT NULL,
    alpha REAL NOT NULL,
    diffusion REAL NOT NULL,
    width REAL NOT NULL,
    lambda REAL NOT NULL,
    val_rmse REAL NOT NULL,
    is_best INTEGER NOT NULL,
    PRIMARY KEY (run_id, trial_index)
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

CREATE INDEX IF NOT EXISTS idx_runs_subset ON runs (subset, created_at);
CREATE INDEX IF NOT EXISTS idx_predictions_run ON predictions (run_id, severity);
CREATE INDEX IF NOT EXISTS idx_maint_run ON maintenance_records (run_id, severity);
CREATE INDEX IF NOT EXISTS idx_exc_run ON exceptions (run_id);
"""

ALLOWED_TABLES = {"runs", "predictions", "search_trials",
                  "maintenance_records", "exceptions"}


def connect(path):
    con = sqlite3.connect(path)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(SCHEMA)  # idempotent: CREATE ... IF NOT EXISTS
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


def write_search_trials(con, run_id, trials):
    """trials: dicts with alpha, diffusion, width, lambda, val_rmse.
    The lowest validation RMSE trial is marked is_best = 1."""
    if not trials:
        return 0
    best_idx = min(range(len(trials)), key=lambda i: trials[i]["val_rmse"])
    con.executemany(
        "INSERT OR REPLACE INTO search_trials VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        [(run_id, i, t["alpha"], t["diffusion"], t["width"], t["lambda"],
          t["val_rmse"], int(i == best_idx)) for i, t in enumerate(trials)],
    )
    return len(trials)


def write_work_orders(con, run_id, subset, rows, created_at):
    """rows: dicts with key, engine_id, cycle, rul, severity, action.
    Returns the number of newly inserted records (re-runs insert nothing)."""
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
    if table not in ALLOWED_TABLES:
        raise ValueError(f"Unknown table: {table}")
    return con.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]


def now_iso():
    return datetime.now(timezone.utc).isoformat()
