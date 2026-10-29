import os
import tempfile

import numpy as np
import pytest

import pipeline


def test_rul_targets_cap():
    assert np.allclose(pipeline.rul_targets(5, 100), [4, 3, 2, 1, 0])
    assert pipeline.rul_targets(3, 1).tolist() == [1.0, 1.0, 0.0]


def test_severity_bands_and_review():
    assert pipeline.severity_band(5) == "critical"
    assert pipeline.severity_band(20) == "high"
    assert pipeline.severity_band(45) == "medium"
    assert pipeline.severity_band(100) == "low"
    assert pipeline.severity_band(float("nan")) == "review"


def test_metrics_match_hand_computation():
    y = [10.0, 20.0]
    p = [12.0, 16.0]
    assert pipeline.rmse(y, p) == pytest.approx(np.sqrt((4 + 16) / 2))
    assert pipeline.mae(y, p) == pytest.approx(3.0)


def test_phm_score_asymmetry():
    # late prediction (over-estimate by 10) is penalised more than early by 10
    late = pipeline.phm_score([20.0], [10.0])
    early = pipeline.phm_score([0.0], [10.0])
    assert late > early


def test_idempotency_key_is_stable():
    a = pipeline.idempotency_key("FD001", 1, 5, "x")
    assert a == pipeline.idempotency_key("FD001", 1, 5, "x")
    assert a != pipeline.idempotency_key("FD002", 1, 5, "x")


def test_triage_rows_split_work_and_exceptions():
    engines = [{"id": 1, "history": [8.0, 50.0, float("nan")]}]
    work, exc = pipeline.build_triage_rows("FD001", engines)
    assert len(work) == 1 and work[0]["severity"] == "critical"
    assert len(exc) == 1 and exc[0]["reason"] == "non-finite prediction"


def test_reservoir_is_deterministic_and_finite():
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(40, 3))
    r1 = pipeline.CSNFReservoir(input_dim=3).run_engine(X)
    r2 = pipeline.CSNFReservoir(input_dim=3).run_engine(X)
    assert np.array_equal(r1, r2)
    assert np.isfinite(r1).all()


def test_ridge_recovers_linear_signal():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(200, 4))
    y = X @ np.array([1.0, -2.0, 0.5, 3.0]) + 4.0
    model = pipeline.fit_ridge(X, y, regularization=1e-6)
    assert np.allclose(pipeline.predict_ridge(model, X), y, atol=1e-2)


def test_read_table_rejects_wrong_width():
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("1 1 2 3\n")
        path = f.name
    try:
        with pytest.raises(ValueError):
            pipeline.read_table(path)
    finally:
        os.remove(path)


def test_end_to_end_on_synthetic_cmapss():
    """Builds a tiny synthetic C-MAPSS-shaped dataset and runs the full pipeline,
    including persistence and reload. Validates plumbing, not accuracy."""
    rng = np.random.default_rng(7)

    def make_split(n_engines, lengths):
        rows = []
        for eid, L in zip(range(1, n_engines + 1), lengths):
            for c in range(1, L + 1):
                rows.append([eid, c, *rng.normal(size=3), *rng.normal(size=21)])
        return np.array(rows)

    train = make_split(6, [40, 45, 50, 42, 48, 46])
    test = make_split(4, [35, 38, 30, 33])
    rul = np.array([12.0, 60.0, 5.0, 40.0])

    with tempfile.TemporaryDirectory() as d:
        data_dir = os.path.join(d, "data")
        out_dir = os.path.join(d, "runs")
        os.makedirs(data_dir)
        np.savetxt(os.path.join(data_dir, "train_FD001.txt"), train)
        np.savetxt(os.path.join(data_dir, "test_FD001.txt"), test)
        np.savetxt(os.path.join(data_dir, "RUL_FD001.txt"), rul)

        m = pipeline.run_pipeline("FD001", data_dir, out_dir, seed=0)
        assert m["test_engines"] == 4
        assert np.isfinite(m["reservoir"]["rmse_capped"])

        run_dir = os.path.join(out_dir, "FD001")
        assert os.path.exists(os.path.join(run_dir, "model.pkl"))
        assert os.path.exists(os.path.join(run_dir, "audit.jsonl"))

        art = pipeline.Artifact.load(os.path.join(run_dir, "model.pkl"))
        assert art.model_version == m["model_version"]

        # Second run with identical inputs must not duplicate work orders.
        m2 = pipeline.run_pipeline("FD001", data_dir, out_dir, seed=0)
        assert m2["work_orders_total"] == m["work_orders_total"]
