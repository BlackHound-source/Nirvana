
#!/usr/bin/env python3
"""
AERO-SYNAPSE Flask server.

Serves the fleet dashboard, exposes pipeline outputs, launches
background pipeline jobs, and provides metrics, work orders,
exceptions, and model-registry information.
"""

import json
import os
import threading
from datetime import datetime, timezone

from flask import Flask, abort, jsonify, request, send_from_directory

import pipeline
import registry
import store


# =========================================================
# Configuration
# =========================================================

BASE = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(
    os.environ.get("AERO_DATA_DIR", os.path.join(BASE, "data"))
)

OUT_DIR = os.path.abspath(
    os.environ.get("AERO_OUT_DIR", os.path.join(BASE, "runs"))
)

STATIC_DIR = os.path.join(BASE, "static")

# Render injects PORT. Local fallback is 10000.
PORT = int(os.environ.get("PORT", "10000"))

app = Flask(__name__, static_folder=STATIC_DIR, static_url_path="")

# One process-local lock protects pipeline status and run initiation.
_lock = threading.Lock()

_status = {
    "state": "idle",
    "subset": None,
    "started": None,
    "finished": None,
    "error": None,
    "message": "ready",
}


# =========================================================
# Utilities
# =========================================================

def now_iso():
    """Return the current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


def run_dir(subset):
    """Return the output directory for a dataset subset."""
    return os.path.join(OUT_DIR, subset)


def load_json(subset, filename):
    """Load a JSON artifact, returning None if it does not exist."""
    path = os.path.join(run_dir(subset), filename)

    if not os.path.isfile(path):
        return None

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, UnicodeDecodeError, OSError):
        app.logger.exception("Unable to read JSON artifact: %s", path)
        abort(500, description=f"Invalid or unreadable artifact: {filename}")


def clean_subset(value):
    """Validate a C-MAPSS subset identifier."""
    subset = (value or "FD001").strip().upper()

    if subset not in pipeline.SUBSETS:
        abort(400, description=f"Unknown subset: {subset}")

    return subset


def available_subsets():
    """Return subsets with an available metrics artifact."""
    return [
        subset
        for subset in pipeline.SUBSETS
        if os.path.isfile(
            os.path.join(run_dir(subset), "metrics.json")
        )
    ]


def status_snapshot():
    """Return a consistent copy of the current in-memory job status."""
    with _lock:
        return dict(_status)


def database_path(subset):
    """Return the SQLite path for a subset."""
    return os.path.join(run_dir(subset), "maintenance.sqlite")


# =========================================================
# Error handlers
# =========================================================

@app.errorhandler(400)
@app.errorhandler(404)
def handle_client_error(error):
    return jsonify(
        error=getattr(error, "description", str(error))
    ), error.code


@app.errorhandler(405)
def handle_method_error(error):
    return jsonify(error="Method not allowed"), 405


@app.errorhandler(500)
def handle_server_error(error):
    app.logger.error("Internal server error: %s", error)

    return jsonify(
        error=getattr(error, "description", "Internal server error")
    ), 500


# =========================================================
# Dashboard and health
# =========================================================

@app.get("/")
def index():
    """Serve the dashboard."""
    index_path = os.path.join(STATIC_DIR, "index.html")

    if not os.path.isfile(index_path):
        abort(404, description="Dashboard file not found")

    return send_from_directory(STATIC_DIR, "index.html")


@app.get("/health")
def health():
    """Health endpoint for deployment checks."""
    return jsonify(
        status="ok",
        service="aero-synapse",
        timestamp=now_iso(),
    ), 200


@app.get("/api/status")
def status():
    """Report job state, available subsets, and data-directory status."""
    current = status_snapshot()

    current["available"] = available_subsets()
    current["data_dir_ready"] = os.path.isdir(DATA_DIR)

    current["required_files"] = {
        subset: os.path.isfile(
            os.path.join(DATA_DIR, f"train_{subset}.txt")
        )
        for subset in pipeline.SUBSETS
    }

    return jsonify(current)


# =========================================================
# Pipeline execution
# =========================================================

@app.post("/api/run")
def start_run():
    """Start a background pipeline run for a requested subset."""
    payload = request.get_json(silent=True) or {}

    if not isinstance(payload, dict):
        abort(400, description="Request body must be a JSON object")

    subset = clean_subset(payload.get("subset"))

    with _lock:
        if _status["state"] == "running":
            return jsonify(
                error="A run is already in progress",
                status=dict(_status),
            ), 409

        # Reject missing training data before starting a worker.
        train_path = os.path.join(
            DATA_DIR,
            f"train_{subset}.txt",
        )

        if not os.path.isfile(train_path):
            return jsonify(
                error=f"Missing train_{subset}.txt under {DATA_DIR}",
                data_dir=DATA_DIR,
                expected_file=train_path,
            ), 400

        _status.update(
            state="running",
            subset=subset,
            started=now_iso(),
            finished=None,
            error=None,
            message=f"running {subset}",
        )

    worker = threading.Thread(
        target=_worker,
        args=(subset,),
        daemon=True,
        name=f"pipeline-{subset}",
    )
    worker.start()

    return jsonify(status_snapshot()), 202


def _worker(subset):
    """Run the pipeline and record its final state."""
    try:
        os.makedirs(OUT_DIR, exist_ok=True)

        pipeline.run_pipeline(
            subset,
            DATA_DIR,
            OUT_DIR,
        )

        with _lock:
            _status.update(
                state="idle",
                finished=now_iso(),
                error=None,
                message=f"completed {subset}",
            )

    except Exception as exc:
        app.logger.exception(
            "Pipeline failed for subset %s",
            subset,
        )

        with _lock:
            _status.update(
                state="error",
                finished=now_iso(),
                error=str(exc),
                message="run failed",
            )


# =========================================================
# Fleet predictions
# =========================================================

@app.get("/api/fleet")
def fleet():
    """Return predictions for all engines in a subset."""
    subset = clean_subset(request.args.get("subset"))

    data = load_json(subset, "predictions.json")

    if data is None:
        abort(
            404,
            description=(
                f"No run for {subset} yet. "
                "POST /api/run to start a pipeline run."
            ),
        )

    if not isinstance(data, dict) or not isinstance(
        data.get("engines"), list
    ):
        abort(500, description="Invalid predictions.json structure")

    engines = [
        {
            key: value
            for key, value in engine.items()
            if key not in ("history", "sensors")
        }
        for engine in data["engines"]
    ]

    return jsonify(
        subset=subset,
        engines=engines,
        metrics=load_json(subset, "metrics.json"),
    )


@app.get("/api/engine/<int:engine_id>")
def engine(engine_id):
    """Return detailed predictions for one engine."""
    subset = clean_subset(request.args.get("subset"))

    data = load_json(subset, "predictions.json")

    if data is None:
        abort(404, description=f"No run for {subset} yet.")

    for item in data.get("engines", []):
        if item.get("id") == engine_id:
            return jsonify(
                subset=subset,
                **item,
            )

    abort(
        404,
        description=f"Engine {engine_id} not found in {subset}",
    )


# =========================================================
# Maintenance records
# =========================================================

@app.get("/api/workorders")
def workorders():
    """Return maintenance records from the subset SQLite database."""
    subset = clean_subset(request.args.get("subset"))

    try:
        limit = int(request.args.get("limit", "50"))
    except (TypeError, ValueError):
        abort(400, description="limit must be an integer")

    limit = min(max(limit, 1), 500)

    path = database_path(subset)

    if not os.path.isfile(path):
        return jsonify(
            subset=subset,
            total=0,
            orders=[],
        )

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
        orders=[dict(row) for row in rows],
    )


@app.get("/api/exceptions")
def exceptions():
    """Return logged exceptions from the subset SQLite database."""
    subset = clean_subset(request.args.get("subset"))
    path = database_path(subset)

    if not os.path.isfile(path):
        return jsonify(
            subset=subset,
            total=0,
            exceptions=[],
        )

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
        exceptions=[dict(row) for row in rows],
    )


# =========================================================
# Model registry and metrics
# =========================================================

@app.get("/api/runs")
def runs():
    """Return model registry entries for a subset."""
    subset = clean_subset(request.args.get("subset"))

    return jsonify(
        subset=subset,
        runs=registry.list_runs(OUT_DIR, subset),
    )


@app.get("/api/metrics")
def metrics():
    """Return evaluation metrics for a subset."""
    subset = clean_subset(request.args.get("subset"))

    data = load_json(subset, "metrics.json")

    if data is None:
        abort(404, description=f"No metrics for {subset} yet.")

    return jsonify(data)


@app.get("/api/summary")
def summary():
    """Summarize metrics for all available subsets."""
    output = []

    for subset in pipeline.SUBSETS:
        metrics_data = load_json(subset, "metrics.json")

        if metrics_data is None:
            output.append({
                "subset": subset,
                "available": False,
            })
            continue

        reservoir = metrics_data.get("reservoir", {})
        baseline = metrics_data.get("ridge_baseline", {})

        output.append({
            "subset": subset,
            "available": True,
            "model_version": metrics_data.get("model_version"),
            "generated_at": metrics_data.get("generated_at"),
            "test_engines": metrics_data.get("test_engines"),
            "reservoir_rmse": reservoir.get("rmse_capped"),
            "reservoir_mae": reservoir.get("mae_capped"),
            "baseline_rmse": baseline.get("rmse_capped"),
            "baseline_mae": baseline.get("mae_capped"),
            "work_orders_total": metrics_data.get(
                "work_orders_total", 0
            ),
            "exceptions_total": metrics_data.get(
                "exceptions_total", 0
            ),
        })

    return jsonify(subsets=output)


# =========================================================
# Local development entry point
# =========================================================

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)

    # Render requires the service to bind to 0.0.0.0.
    app.run(
        host="0.0.0.0",
        port=PORT,
        threaded=True,
        debug=False,
    )
