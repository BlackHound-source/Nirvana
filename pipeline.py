#!/usr/bin/env python3
"""
AERO-SYNAPSE pipeline.
NASA C-MAPSS RUL prediction -> deterministic triage -> SQLite + audit log.

  python pipeline.py --subset FD001 --data-dir data --out-dir runs
  python pipeline.py --selftest
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import pickle
import sqlite3
import zipfile
from dataclasses import dataclass, asdict, field

import numpy as np

import store

__version__ = "1.1.0"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
R_MAX = 125.0
WASHOUT = 20
VAL_FRACTION = 0.20
SEED = 0
N_COLS = 26
GRID_SEARCH_ALPHA = (0.4, 0.7)
GRID_SEARCH_WIDTH = (0.25, 0.35)
GRID_SEARCH_LAMBDA = (0.1, 1.0, 10.0, 100.0)


def log(msg):
    print(msg, flush=True)


# ------------------------- DATA DISCOVERY -------------------------

def extract_zips(data_dir):
    """Safely extract ZIP archives found under data_dir, including nested folders."""
    root = os.path.realpath(data_dir)
    os.makedirs(root, exist_ok=True)
    archives = sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True))

    for zip_path in archives:
        # Avoid repeatedly extracting an archive into itself or processing the same file.
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = os.path.realpath(os.path.join(root, member.filename))
                if target != root and not target.startswith(root + os.sep):
                    raise ValueError(f"Unsafe ZIP member in {zip_path}: {member.filename}")
            archive.extractall(root)
        log(f"[DATASET] Extracted archive: {zip_path}")


def locate_dataset(subset, data_dir):
    """Find required C-MAPSS files regardless of nested ZIP folder structure."""
    data_dir = os.path.abspath(os.path.expanduser(str(data_dir)))
    log(f"[DATASET] Requested subset: {subset}")
    log(f"[DATASET] Data directory: {data_dir}")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}. "
            "Set AERO_DATA_DIR to the directory containing the C-MAPSS files."
        )

    extract_zips(data_dir)
    wanted = {
        "train": f"train_{subset}.txt",
        "test": f"test_{subset}.txt",
        "rul": f"RUL_{subset}.txt",
    }

    found_files = {}
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            found_files.setdefault(filename, []).append(os.path.join(root, filename))

    log(f"[DATASET] Files discovered: {sorted(found_files)[:100]}")
    paths = {}
    missing = []
    for key, filename in wanted.items():
        hits = sorted(found_files.get(filename, []))
        if not hits:
            missing.append(filename)
        else:
            # Prefer a direct child of data_dir, then a deterministic nested path.
            paths[key] = min(
                hits,
                key=lambda candidate: (os.path.dirname(candidate) != data_dir, candidate),
            )

    if missing:
        raise FileNotFoundError(
            f"Missing required C-MAPSS file(s): {', '.join(missing)}. "
            f"Searched recursively under {data_dir}. "
            f"Discovered files: {sorted(found_files)[:100]}"
        )

    log(f"[DATASET] Resolved paths: {paths}")
    return paths


# ------------------------- VALIDATION -----------------------------

def read_table(path):
    with open(path, "rb") as f:
        raw = f.read()
    digest = hashlib.sha256(raw).hexdigest()
    data = np.loadtxt(io.BytesIO(raw), dtype=float, ndmin=2)
    if data.shape[1] != N_COLS:
        raise ValueError(f"{path}: expected {N_COLS} columns, got {data.shape[1]}")
    if not np.isfinite(data).all():
        raise ValueError(f"{path} contains NaN or infinite values")
    return data, digest


def group_engines(data, source):
    ids = data[:, 0].astype(int)
    cycles = data[:, 1].astype(int)
    if np.any(cycles < 1):
        raise ValueError(f"{source}: cycles must start at 1")
    order = np.lexsort((cycles, ids))
    data, ids, cycles = data[order], ids[order], cycles[order]
    if np.any((np.diff(ids) == 0) & (np.diff(cycles) == 0)):
        raise ValueError(f"{source}: duplicate engine-cycle rows")
    result = {}
    for engine_id in np.unique(ids):
        rows = data[ids == engine_id]
        if not np.array_equal(rows[:, 1].astype(int), np.arange(1, len(rows) + 1)):
            raise ValueError(f"{source}: engine {engine_id} has missing cycles")
        result[int(engine_id)] = rows
    if not result:
        raise ValueError(f"{source}: no engines found")
    return result


def rul_targets(n_cycles, cap=R_MAX):
    return np.minimum(np.arange(n_cycles - 1, -1, -1, dtype=float), cap)


def split_engine_ids(ids, fraction=VAL_FRACTION, seed=SEED):
    ids = np.array(sorted(ids))
    if len(ids) < 2:
        raise ValueError("At least two training engines are required")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(ids)
    n_val = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    return sorted(perm[n_val:].tolist()), sorted(perm[:n_val].tolist())


# ------------------------- PREPROCESSING --------------------------

@dataclass
class Preprocessor:
    keep: np.ndarray | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    def fit(self, engines):
        matrix = np.vstack([rows[:, 2:] for rows in engines.values()])
        std = matrix.std(axis=0)
        self.keep = np.where(std > 1e-8)[0]
        if len(self.keep) == 0:
            raise ValueError("All input features are constant")
        matrix = matrix[:, self.keep]
        self.low = matrix.min(axis=0)
        self.high = matrix.max(axis=0)
        return self

    @property
    def n_features(self):
        return len(self.keep)

    def transform(self, rows):
        x = rows[:, 2:][:, self.keep]
        span = np.maximum(self.high - self.low, 1e-12)
        return (x - self.low) / span  # test values outside training range are not clipped

    def to_dict(self):
        return {"keep": self.keep.tolist(), "low": self.low.tolist(),
                "high": self.high.tolist()}


# ------------------------- R-CSNF-STYLE RESERVOIR -----------------

class CSNFReservoir:
    """Deterministic recurrent complex-valued reservoir inspired by
    split-step Schrödinger field dynamics."""

    def __init__(self, input_dim, grid_n=32, num_lags=3, pool_bins=8,
                 evolve_steps=5, alpha_inject=0.6, packet_width=0.35,
                 diffusion=0.01, potential=0.4):
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if grid_n % pool_bins:
            raise ValueError("grid_n must be divisible by pool_bins")
        self.input_dim = input_dim
        self.n = grid_n
        self.num_lags = num_lags
        self.pool_bins = pool_bins
        self.evolve_steps = evolve_steps
        self.alpha = alpha_inject
        self.packet_width = packet_width
        self.length = 2 * np.pi
        self.x = np.linspace(0, self.length, grid_n, endpoint=False)
        dx = self.length / grid_n
        self.k = 2 * np.pi * np.fft.fftfreq(grid_n, d=dx)
        dt = 1.0 / evolve_steps
        self.propagator = np.exp(-0.5j * diffusion * self.k ** 2 * dt)
        self.vpsi = np.exp(-1j * potential * np.cos(2 * self.x) * dt)
        self.vphi = np.exp(-1j * potential * np.sin(3 * self.x) * dt)

    @property
    def n_features_out(self):
        return 7 * self.pool_bins + self.num_lags * self.input_dim

    def kinetic(self, field):
        return np.fft.ifft(np.fft.fft(field) * self.propagator)

    @staticmethod
    def normalize(psi, phi):
        norm2 = np.vdot(psi, psi).real + np.vdot(phi, phi).real
        if not np.isfinite(norm2) or norm2 < 1e-12:
            n = len(psi)
            return np.ones(n, dtype=complex) / np.sqrt(n), np.zeros(n, dtype=complex)
        norm = np.sqrt(norm2)
        return psi / norm, phi / norm

    def encode(self, lags):
        lags = np.asarray(lags, dtype=float)
        if lags.shape != (self.num_lags, self.input_dim):
            raise ValueError(f"Unexpected lag shape: {lags.shape}")
        arc = self.length / self.input_dim
        width = max(1e-8, min(self.packet_width, arc * 0.5))
        weights = np.exp(-0.35 * np.arange(self.num_lags))
        weights /= weights.sum()
        if self.num_lags > 1:
            velocity = np.angle(np.exp(1j * (lags[0] - lags[1])))
        else:
            velocity = np.zeros(self.input_dim)
        signal = 0.5 * (1 + np.tanh(lags))
        phi = np.zeros(self.n, dtype=complex)
        for lag in range(self.num_lags):
            for feature in range(self.input_dim):
                center = (feature * arc + arc * (0.1 + 0.8 * signal[lag, feature])) % self.length
                distance = np.angle(np.exp(1j * (self.x - center)))
                packet = np.exp(-0.5 * (distance / width) ** 2)
                tilt = np.exp(1j * velocity[feature] * (lag + 1) * np.pi * self.x / self.length)
                phi += weights[lag] * packet * tilt
        psi = np.ones(self.n, dtype=complex) / np.sqrt(self.n)
        return self.normalize(psi, phi)

    def evolve(self, psi, phi):
        for step in range(self.evolve_steps):
            psi = self.kinetic(psi)
            phi = self.kinetic(phi)
            coupling = 0.1 * np.exp(-1j * (step + 0.5))
            psi_old = psi.copy()
            psi = psi * self.vpsi - 1j * coupling * phi
            phi = phi * self.vphi - 1j * np.conj(coupling) * psi_old
            psi = self.kinetic(psi)
            phi = self.kinetic(phi)
        return self.normalize(psi, phi)

    def step(self, previous, lags):
        psi_in, phi_in = self.encode(lags)
        if previous is None:
            psi, phi = psi_in, phi_in
        else:
            a = self.alpha
            psi = (1 - a) * previous[:self.n] + a * psi_in
            phi = (1 - a) * previous[self.n:] + a * phi_in
            psi, phi = self.normalize(psi, phi)
        psi, phi = self.evolve(psi, phi)
        return np.concatenate([psi, phi])

    def features(self, state, lags):
        psi = state[:self.n]
        phi = state[self.n:]

        def pool(values):
            return values.reshape(self.pool_bins, -1).mean(axis=1)

        derivative = np.fft.ifft(1j * self.k * np.fft.fft(psi))
        values = np.concatenate([
            pool(np.abs(psi) ** 2), pool(np.abs(phi) ** 2),
            pool(psi.real), pool(psi.imag),
            pool(phi.real), pool(phi.imag),
            pool(np.imag(np.conj(psi) * derivative)),
            lags.reshape(-1),
        ])
        return np.nan_to_num(values)

    def run_engine(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim != 2 or X.shape[1] != self.input_dim:
            raise ValueError(f"Expected (T, {self.input_dim}), got {X.shape}")
        state = None
        output = np.empty((len(X), self.n_features_out))
        for t in range(len(X)):
            lags = np.stack([X[max(0, t - i)] for i in range(self.num_lags)])
            state = self.step(state, lags)
            output[t] = self.features(state, lags)
        return output


# ------------------------- RIDGE / METRICS ------------------------

def fit_ridge(X, y, regularization=1.0):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    mean = X.mean(axis=0)
    scale = X.std(axis=0)
    scale[scale < 1e-8] = 1.0
    Z = (X - mean) / scale
    y_mean = y.mean()
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    factors = S / (S ** 2 + regularization)
    if len(S):
        factors[S <= S[0] * 1e-10] = 0.0
    weights = Vt.T @ (factors * (U.T @ (y - y_mean)))
    return {"mean": mean, "scale": scale, "weights": weights, "y_mean": float(y_mean)}


def predict_ridge(model, X):
    return ((X - model["mean"]) / model["scale"]) @ model["weights"] + model["y_mean"]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def phm_score(prediction, truth):
    delta = np.asarray(prediction) - np.asarray(truth)
    return float(np.sum(np.where(delta < 0,
                                 np.exp(-delta / 13.0) - 1,
                                 np.exp(delta / 10.0) - 1)))


def causal_mean_features(X, window=5):
    result = np.empty_like(X)
    for t in range(len(X)):
        result[t] = X[max(0, t - window + 1):t + 1].mean(axis=0)
    return np.hstack([X, result])


# ------------------------- TRIAGE ---------------------------------

def severity_band(rul):
    if not np.isfinite(rul):
        return "review"
    if rul <= 10:
        return "critical"
    if rul <= 30:
        return "high"
    if rul <= 60:
        return "medium"
    return "low"


def triage_action(severity):
    return {
        "critical": "schedule_immediate_maintenance",
        "high": "schedule_urgent_maintenance",
        "medium": "schedule_planned_maintenance",
        "low": "monitor",
        "review": "raise_exception",
    }[severity]


def idempotency_key(subset, engine_id, cycle, action):
    return hashlib.sha256(f"{subset}|{engine_id}|{cycle}|{action}".encode()).hexdigest()


def build_triage_rows(subset, engines_out):
    """Return (work_orders, exceptions) as lists of dicts. Deterministic."""
    work, exc = [], []
    for e in engines_out:
        for t, rul in enumerate(e["history"], start=1):
            sev = severity_band(rul)
            if sev == "low":
                continue
            action = triage_action(sev)
            row = {"key": idempotency_key(subset, e["id"], t, action),
                   "engine_id": e["id"], "cycle": t, "rul": rul,
                   "severity": sev, "action": action}
            if sev == "review":
                exc.append({**row, "reason": "non-finite prediction"})
            else:
                work.append(row)
    return work, exc


# ------------------------- MODEL PERSISTENCE ----------------------

@dataclass
class Artifact:
    model_version: str
    subset: str
    seed: int
    best_parameters: dict
    preprocessor: dict
    reservoir_config: dict
    reservoir_model: dict
    baseline_model: dict
    train_sha256: str
    test_sha256: str

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(asdict(self), f)

    @staticmethod
    def load(path):
        with open(path, "rb") as f:
            return Artifact(**pickle.load(f))


def make_model_version(subset, seed, train_hash, best):
    raw = f"{__version__}|{subset}|{seed}|{train_hash}|{json.dumps(best, sort_keys=True)}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


# ------------------------- MAIN PIPELINE --------------------------

def run_pipeline(subset="FD001", data_dir=None, out_dir=None, seed=SEED):
    # Honour container/runtime configuration even when callers omit explicit paths.
    data_dir = data_dir or os.environ.get("AERO_DATA_DIR", "data")
    out_dir = out_dir or os.environ.get("AERO_OUT_DIR", "runs")
    if subset not in SUBSETS:
        raise ValueError(f"Unsupported subset: {subset}")
    run_dir = os.path.join(out_dir, subset)
    os.makedirs(run_dir, exist_ok=True)

    paths = locate_dataset(subset, data_dir)

    log("[1] Loading and validating data")
    train_raw, train_hash = read_table(paths["train"])
    test_raw, test_hash = read_table(paths["test"])
    test_rul = np.loadtxt(paths["rul"], dtype=float, ndmin=1)
    train_engines = group_engines(train_raw, "train")
    test_engines = group_engines(test_raw, "test")
    test_ids = sorted(test_engines)
    if len(test_rul) != len(test_ids):
        raise ValueError(f"RUL rows ({len(test_rul)}) != test engines ({len(test_ids)})")
    log(f"Training engines: {len(train_engines)} | test engines: {len(test_ids)}")

    checkpoint_path = os.path.join(run_dir, "training_checkpoint.pkl")

    def atomic_pickle_save(payload, path):
        """Write checkpoint atomically so an interrupted write cannot corrupt it."""
        temp_path = path + ".tmp"
        with open(temp_path, "wb") as handle:
            pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_path, path)

    checkpoint = None
    if os.path.isfile(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as handle:
                candidate = pickle.load(handle)
            expected = {
                "pipeline_version": __version__,
                "subset": subset,
                "seed": seed,
                "train_sha256": train_hash,
                "test_sha256": test_hash,
            }
            if all(candidate.get(k) == v for k, v in expected.items()):
                checkpoint = candidate
                log(f"[RESUME] Compatible training checkpoint found: {checkpoint_path}")
            else:
                log("[RESUME] Checkpoint does not match this dataset/configuration; retraining.")
        except Exception as exc:
            log(f"[RESUME] Checkpoint could not be read ({exc}); retraining.")

    if checkpoint is not None:
        best = checkpoint["best"]
        pre_dict = checkpoint["preprocessor"]
        pre = Preprocessor(
            keep=np.asarray(pre_dict["keep"], dtype=int),
            low=np.asarray(pre_dict["low"], dtype=float),
            high=np.asarray(pre_dict["high"], dtype=float),
        )
        cfg = checkpoint["reservoir_config"]
        reservoir = CSNFReservoir(
            input_dim=cfg["input_dim"],
            alpha_inject=cfg["alpha_inject"],
            packet_width=cfg["packet_width"],
        )
        model = plain_to_model(checkpoint["reservoir_model"])
        baseline = plain_to_model(checkpoint["baseline_model"])
        log("[RESUME] Skipped validation tuning and final training; continuing at test evaluation.")
    else:
        log("[2] Engine-level validation split")
        fit_ids, val_ids = split_engine_ids(train_engines.keys(), seed=seed)
        fit_engines = {i: train_engines[i] for i in fit_ids}
        val_engines = {i: train_engines[i] for i in val_ids}

        log("[3] Validation tuning")
        pre_val = Preprocessor().fit(fit_engines)

        def make_items(engines, pp):
            return [(pp.transform(engines[i]), rul_targets(len(engines[i])))
                    for i in sorted(engines)]

        fit_items = make_items(fit_engines, pre_val)
        val_items = make_items(val_engines, pre_val)

        best = None
        for alpha in GRID_SEARCH_ALPHA:
            for width in GRID_SEARCH_WIDTH:
                res = CSNFReservoir(input_dim=pre_val.n_features,
                                    alpha_inject=alpha, packet_width=width)
                X_fit = np.vstack([res.run_engine(X)[WASHOUT:] for X, _ in fit_items])
                y_fit = np.concatenate([y[WASHOUT:] for _, y in fit_items])
                X_val = np.vstack([res.run_engine(X)[WASHOUT:] for X, _ in val_items])
                y_val = np.concatenate([y[WASHOUT:] for _, y in val_items])
                for lam in GRID_SEARCH_LAMBDA:
                    candidate_model = fit_ridge(X_fit, y_fit, lam)
                    pred = np.clip(predict_ridge(candidate_model, X_val), 0, R_MAX)
                    score = rmse(y_val, pred)
                    log(f"  alpha={alpha} width={width} lambda={lam}: val RMSE={score:.3f}")
                    if best is None or score < best["score"]:
                        best = {"score": score, "alpha": alpha, "width": width, "lambda": lam}
        log(f"Best reservoir settings: {best}")

        log("[4] Final training on all training engines")
        pre = Preprocessor().fit(train_engines)
        train_items = make_items(train_engines, pre)
        reservoir = CSNFReservoir(input_dim=pre.n_features,
                                  alpha_inject=best["alpha"], packet_width=best["width"])
        X_train = np.vstack([reservoir.run_engine(X)[WASHOUT:] for X, _ in train_items])
        y_train = np.concatenate([y[WASHOUT:] for _, y in train_items])
        model = fit_ridge(X_train, y_train, best["lambda"])

        B_train = np.vstack([causal_mean_features(X)[WASHOUT:] for X, _ in train_items])
        baseline = fit_ridge(B_train, y_train, best["lambda"])

        # Save the expensive training result before the test-evaluation stage.
        # The checkpoint is reusable only for the same dataset hashes and settings.
        checkpoint_payload = {
            "pipeline_version": __version__,
            "subset": subset,
            "seed": seed,
            "train_sha256": train_hash,
            "test_sha256": test_hash,
            "best": best,
            "preprocessor": pre.to_dict(),
            "reservoir_config": {
                "alpha_inject": best["alpha"],
                "packet_width": best["width"],
                "input_dim": pre.n_features,
            },
            "reservoir_model": model_to_plain(model),
            "baseline_model": model_to_plain(baseline),
        }
        atomic_pickle_save(checkpoint_payload, checkpoint_path)
        log(f"[CHECKPOINT] Training checkpoint saved: {checkpoint_path}")

    log("[5] Test evaluation (same protocol for both models: last cycle of each test engine)")
    truth_capped = np.minimum(test_rul, R_MAX)
    engines_out, res_last, base_last = [], [], []
    for i, engine_id in enumerate(test_ids):
        rows = test_engines[engine_id]
        X = pre.transform(rows)
        res_pred = np.clip(predict_ridge(model, reservoir.run_engine(X)), 0, R_MAX)
        base_pred = np.clip(predict_ridge(baseline, causal_mean_features(X)), 0, R_MAX)
        res_last.append(res_pred[-1])
        base_last.append(base_pred[-1])
        sev = severity_band(float(res_pred[-1]))
        engines_out.append({
            "id": int(engine_id),
            "cycles_observed": int(len(rows)),
            "pred_rul": round(float(res_pred[-1]), 2),
            "true_rul": float(test_rul[i]),
            "severity": sev,
            "action": triage_action(sev),
            "history": [round(float(v), 2) for v in res_pred],
            "sensors": {f"s{k}": float(rows[-1][4 + k]) for k in range(1, 22)},
        })

    res_last = np.asarray(res_last)
    base_last = np.asarray(base_last)
    metrics = {
        "subset": subset,
        "pipeline_version": __version__,
        "seed": seed,
        "generated_at": store.now_iso(),
        "train_sha256": train_hash,
        "test_sha256": test_hash,
        "test_engines": len(test_ids),
        "best_parameters": best,
        "reservoir": {
            "rmse_capped": rmse(truth_capped, res_last),
            "mae_capped": mae(truth_capped, res_last),
            "rmse_uncapped": rmse(test_rul, res_last),
            "phm_score_uncapped": phm_score(res_last, test_rul),
        },
        "ridge_baseline": {
            "rmse_capped": rmse(truth_capped, base_last),
            "mae_capped": mae(truth_capped, base_last),
            "rmse_uncapped": rmse(test_rul, base_last),
            "phm_score_uncapped": phm_score(base_last, test_rul),
        },
    }
    model_version = make_model_version(subset, seed, train_hash, best)
    metrics["model_version"] = model_version
    for name in ("reservoir", "ridge_baseline"):
        log(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics[name].items()))

    log("[6] Persisting artifact, predictions, work orders and audit log")
    Artifact(
        model_version=model_version, subset=subset, seed=seed,
        best_parameters=best, preprocessor=pre.to_dict(),
        reservoir_config={"alpha_inject": best["alpha"],
                          "packet_width": best["width"],
                          "input_dim": pre.n_features},
        reservoir_model=model_to_plain(model),
        baseline_model=model_to_plain(baseline),
        train_sha256=train_hash, test_sha256=test_hash,
    ).save(os.path.join(run_dir, "model.pkl"))

    created_at = store.now_iso()
    work, exc = build_triage_rows(subset, engines_out)
    with open(os.path.join(run_dir, "audit.jsonl"), "w", encoding="utf-8") as audit:
        for r in work + exc:
            audit.write(json.dumps({**r, "subset": subset, "created_at": created_at}) + "\n")

    con = store.connect(os.path.join(run_dir, "maintenance.sqlite"))
    try:
        with con:
            store.write_run(con, {
                "run_id": model_version, "subset": subset,
                "model_version": model_version, "seed": seed,
                "train_sha256": train_hash, "test_sha256": test_hash,
                "created_at": created_at, "metrics_json": json.dumps(metrics),
            })
            store.write_predictions(con, model_version, engines_out)
            new_wo = store.write_work_orders(con, model_version, subset, work, created_at)
            new_exc = store.write_exceptions(con, model_version, subset, exc, created_at)
        total_wo = store.count(con, "maintenance_records")
        total_exc = store.count(con, "exceptions")
    finally:
        con.close()

    metrics["work_orders_new"] = new_wo
    metrics["work_orders_total"] = total_wo
    metrics["exceptions_new"] = new_exc
    metrics["exceptions_total"] = total_exc

    with open(os.path.join(run_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"subset": subset, "model_version": model_version,
                   "engines": engines_out}, f)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log(f"Work orders: {new_wo} new / {total_wo} total | "
        f"exceptions: {new_exc} new / {total_exc} total")
    log(f"Pipeline completed for {subset} (model {model_version}).")
    return metrics


def model_to_plain(m):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}


def plain_to_model(d):
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in d.items()}


def predict_with_artifact(artifact, rows):
    """Reproduce predictions for one engine's raw rows from a saved artifact."""
    pre = Preprocessor(keep=np.asarray(artifact.preprocessor["keep"]),
                       low=np.asarray(artifact.preprocessor["low"]),
                       high=np.asarray(artifact.preprocessor["high"]))
    cfg = artifact.reservoir_config
    res = CSNFReservoir(input_dim=cfg["input_dim"],
                        alpha_inject=cfg["alpha_inject"],
                        packet_width=cfg["packet_width"])
    X = pre.transform(rows)
    return np.clip(predict_ridge(plain_to_model(artifact.reservoir_model),
                                 res.run_engine(X)), 0, R_MAX)


# ------------------------- SELF-TEST ------------------------------

def run_selftest():
    assert np.allclose(rul_targets(5, 100), [4, 3, 2, 1, 0])
    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 4))
    y = np.linspace(30, 0, 30)
    res = CSNFReservoir(input_dim=4, grid_n=32, pool_bins=8)
    F = res.run_engine(X)
    assert F.shape == (30, res.n_features_out) and np.isfinite(F).all()
    pred = predict_ridge(fit_ridge(F, y, 1.0), F)
    assert pred.shape == y.shape and np.isfinite(pred).all()
    assert severity_band(5) == "critical"
    assert severity_band(20) == "high"
    assert severity_band(45) == "medium"
    assert severity_band(100) == "low"
    assert severity_band(float("nan")) == "review"
    print("Self-test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AERO-SYNAPSE C-MAPSS pipeline")
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--data-dir", default=os.environ.get("AERO_DATA_DIR", "data"))
    parser.add_argument("--out-dir", default=os.environ.get("AERO_OUT_DIR", "runs"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        run_pipeline(args.subset, args.data_dir, args.out_dir, args.seed)
