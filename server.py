
#!/usr/bin/env python3
"""AERO-SYNAPSE server: serves the dashboard, exposes pipeline outputs, runs jobs."""

import json
import os
import threading
from datetime import datetime, timezone

from flask import Flask, abort, jsonify, request, send_from_directory

import pipeline
import registry
import store

BASE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.environ.get("AERO_DATA_DIR", os.path.join(BASE, "data"))
OUT_DIR = os.environ.get("AERO_OUT_DIR", os.path.join(BASE, "runs"))
STATIC_DIR = os.path.join(BASE, "static")

# Render supplies the PORT environment variable.
PORT = int(os.environ.get("PORT", "10000"))

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

_lock = threading.Lock()
_status = {
    "state": "idle",
    "subset": None,
    "started": None,
    "finished": None,
    "error": None,
    "message": "ready",
}


def now_iso():
    return datetime.now(timezone.utc).isoformat()


def run_dir(subset):
    return os.path.join(OUT_DIR, subset)


def load_json(subset, name):
    path = os.path.join(run_dir(subset), name)
    if not os.path.exists(path):
        return None

    with open(path, encoding="utf-8") as f:
        return json.load(f)


def clean_subset(value):
    subset = (value or "FD001").upper()

    if subset not in pipeline.SUBSETS:
        abort(400, description=f"Unknown subset: {subset}")

    return subset


def available_subsets():
    return [
        s for s in pipeline.SUBSETS
        if os.path.exists(os.path.join(run_dir(s), "metrics.json"))
    ]


@app.errorhandler(400)
@app.errorhandler(404)
def json_error(err):
    return jsonify(
        error=getattr(err, "description", str(err))
    ), err.code


@app.get("/")
def index():
    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/health")
def health():
    return jsonify(
        status="ok",
        service="aero-synapse",
        timestamp=now_iso(),
    ), 200


@app.get("/api/status")
def status():
    return jsonify(
        dict(
            _status,
            available=available_subsets(),
            data_dir_ready=os.path.isdir(DATA_DIR),
        )
    )


@app.post("/api/run")
def start_run():
    payload = request.get_json(silent=True) or {}
    subset = clean_subset(payload.get("subset"))

    with _lock:
        if _status["state"] == "running":
            return jsonify(
                error="A run is already in progress",
                status=dict(_status),
            ), 409

        _status.update(
            state="running",
            subset=subset,
            started=now_iso(),
            finished=None,
            error=None,
            message=f"running {subset}",
        )

    threading.Thread(
        target=_worker,
        args=(subset,),
        daemon=True,
    ).start()

    return jsonify(dict(_status)), 202


def _worker(subset):
    try:
        os.makedirs(OUT_DIR, exist_ok=True)
        pipeline.run_pipeline(subset, DATA_DIR, OUT_DIR)

        with _lock:
            _status.update(
                state="idle",
                finished=now_iso(),
                error=None,
                message=f"completed {subset}",
            )

    except Exception as exc:
        app.logger.exception("Pipeline failed for %s", subset)

        with _lock:
            _status.update(
                state="error",
                finished=now_iso(),
                error=str(exc),
                message="run failed",
            )


@app.get("/api/fleet")
def fleet():
    subset = clean_subset(request.args.get("subset"))
    data = load_json(subset, "predictions.json")

    if data is None:
        abort(
            404,
            description=f"No run for {subset} yet. POST /api/run first.",
        )

    engines = [
        {k: v for k, v in e.items() if k not in ("history", "sensors")}
        for e in data["engines"]
    ]

    return jsonify(
        subset=subset,
        engines=engines,
        metrics=load_json(subset, "metrics.json"),
    )


@app.get("/api/engine/<int:engine_id>")
def engine(engine_id):
    subset = clean_subset(request.args.get("subset"))
    data = load_json(subset, "predictions.json")

    if data is None:
        abort(404, description=f"No run for {subset} yet.")

    for e in data["engines"]:
        if e["id"] == engine_id:
            return jsonify(subset=subset, **e)

    abort(404, description=f"Engine {engine_id} not found in {subset}")


@app.get("/api/workorders")
def workorders():
    subset = clean_subset(request.args.get("subset"))

    try:
        limit = int(request.args.get("limit", 50))
    except (TypeError, ValueError):
        abort(400, description="limit must be an integer")

    limit = min(max(limit, 1), 500)

    path = os.path.join(run_dir(subset), "maintenance.sqlite")

    if not os.path.exists(path):
        return jsonify(subset=subset, total=0, orders=[])

    con = store.connect(path)

    try:
        total = store.count(con, "maintenance_records")

        rows = con.execute(
            """
            SELECT *
            FROM maintenance_records
            ORDER BY rowid DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    finally:
        con.close()

    return jsonify(
        subset=subset,
        total=total,
        orders=[dict(r) for r in rows],
    )


@app.get("/api/exceptions")
def exceptions():
    subset = clean_subset(request.args.get("subset"))
    path = os.path.join(run_dir(subset), "maintenance.sqlite")

    if not os.path.exists(path):
        return jsonify(subset=subset, total=0, exceptions=[])

    con = store.connect(path)

    try:
        total = store.count(con, "exceptions")

        rows = con.execute(
            """
            SELECT *
            FROM exceptions
            ORDER BY rowid DESC
            LIMIT 100
            """
        ).fetchall()
    finally:
        con.close()

    return jsonify(
        subset=subset,
        total=total,
        exceptions=[dict(r) for r in rows],
    )


@app.get("/api/runs")
def runs():
    subset = clean_subset(request.args.get("subset"))

    return jsonify(
        subset=subset,
        runs=registry.list_runs(OUT_DIR, subset),
    )


@app.get("/api/metrics")
def metrics():
    subset = clean_subset(request.args.get("subset"))
    data = load_json(subset, "metrics.json")

    if data is None:
        abort(404, description=f"No metrics for {subset} yet.")

    return jsonify(data)


@app.get("/api/summary")
def summary():
    output = []

    for subset in pipeline.SUBSETS:
        metrics_data = load_json(subset, "metrics.json")

        if metrics_data is None:
            output.append({
                "subset": subset,
                "available": False,
            })
            continue

        reservoir = metrics_data["reservoir"]
        baseline = metrics_data["ridge_baseline"]

        output.append({
            "subset": subset,
            "available": True,
            "model_version": metrics_data.get("model_version"),
            "generated_at": metrics_data.get("generated_at"),
            "test_engines": metrics_data.get("test_engines"),
            "reservoir_rmse": reservoir["rmse_capped"],
            "reservoir_mae": reservoir["mae_capped"],
            "baseline_rmse": baseline["rmse_capped"],
            "baseline_mae": baseline["mae_capped"],
            "work_orders_total": metrics_data.get(
                "work_orders_total", 0
            ),
            "exceptions_total": metrics_data.get(
                "exceptions_total", 0
            ),
        })

    return jsonify(subsets=output)


if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    app.run(
        host="0.0.0.0",
        port=PORT,
        threaded=True,
        debug=False,
    )
