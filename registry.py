"""Model versioning: list runs per subset and compare the latest with a previous one."""
import json
import os
import sqlite3

import store


def list_runs(out_dir, subset):
    path = os.path.join(out_dir, subset, "maintenance.sqlite")
    if not os.path.exists(path):
        return []
    con = store.connect(path)
    try:
        rows = con.execute(
            "SELECT run_id, model_version, created_at, metrics_json FROM runs "
            "ORDER BY created_at DESC").fetchall()
    finally:
        con.close()
    return [{"run_id": r["run_id"], "model_version": r["model_version"],
             "created_at": r["created_at"],
             "metrics": json.loads(r["metrics_json"])} for r in rows]


def compare(old, new):
    """Return the change in reservoir and baseline RMSE/MAE (negative = improvement)."""
    out = {}
    for model in ("reservoir", "ridge_baseline"):
        out[model] = {m: new[model][m] - old[model][m]
                      for m in ("rmse_capped", "mae_capped")}
    return out
