#!/usr/bin/env python3
"""
AERO-SYNAPSE pipeline (R-CSNF v3 reservoir, C-MAPSS RUL prediction).

  python pipeline.py --selftest
  python pipeline.py --subset FD001 --data-dir data --out-dir runs
  python pipeline.py --subset FD001 --trials 16 --seed 0

Protocol
  * Engine-level validation split on the training engines (no leakage).
  * Hyperparameters chosen by RANDOM SEARCH over the paper's search spaces
    (R-CSNF v3 row of Table 1), with a fixed trial budget.
  * Selection uses validation RUL RMSE only. The test set is scored once.
  * Test score = last cycle of each test engine vs. the official RUL file.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import io
import json
import os
import pickle
import zipfile
from dataclasses import dataclass, asdict, field

import numpy as np

import store

__version__ = "2.0.0"

SUBSETS = ("FD001", "FD002", "FD003", "FD004")
R_MAX = 125.0                 # RUL cap used in training (standard C-MAPSS practice)
N_COLS = 26
VAL_FRACTION = 0.20
SEED = 0

# Reservoir defaults (paper Sec. 7.6 and rcsnf.py)
GRID_N = 64
POOL_BINS = 8
NUM_LAGS = 3
EVOLVE_STEPS = 10
POTENTIAL = 0.4
WASHOUT = 30

# Search spaces (paper Table 1, R-CSNF v3 row). All selection is on validation.
SEARCH_ALPHA = (0.1, 0.9)        # injection strength, uniform
SEARCH_DIFFUSION = (1e-4, 1e-1)  # propagation strength, log-uniform
SEARCH_WIDTH = (0.15, 0.5)       # packet width, uniform
SEARCH_LAMBDA = (1e-6, 1e-2)     # readout penalty, log-uniform
DEFAULT_TRIALS = 16              # equal-budget trials, as in paper Fig. 3
BALANCE_RUL = True               # RUL-band row weighting (ablation switch)

# Post-processing grid (tuned on validation only)
SMOOTH_BETA_GRID = (0.2, 0.4, 0.6)
SMOOTH_DEADBAND_GRID = (1.0, 2.0, 4.0)

RUL_BINS = (0, 10, 30, 60, 90, 126)
CYCLE_BINS = (1, 30, 60, 100, 10**6)


def log(msg):
    print(msg, flush=True)


# ------------------------- DATA DISCOVERY -------------------------

def extract_zips(data_dir):
    """Safely extract ZIP archives under data_dir (nested folders included)."""
    root = os.path.realpath(data_dir)
    os.makedirs(root, exist_ok=True)
    for zip_path in sorted(glob.glob(os.path.join(root, "**", "*.zip"), recursive=True)):
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                target = os.path.realpath(os.path.join(root, member.filename))
                if target != root and not target.startswith(root + os.sep):
                    raise ValueError(f"Unsafe ZIP member in {zip_path}: {member.filename}")
            archive.extractall(root)
        log(f"[DATASET] Extracted archive: {zip_path}")


def locate_dataset(subset, data_dir):
    """Find the train/test/RUL files for a subset, wherever they sit under data_dir."""
    data_dir = os.path.abspath(os.path.expanduser(str(data_dir)))
    log(f"[DATASET] Requested subset: {subset} | data directory: {data_dir}")
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Dataset directory does not exist: {data_dir}. "
            "Set AERO_DATA_DIR to the directory containing the C-MAPSS files."
        )
    extract_zips(data_dir)
    wanted = {"train": f"train_{subset}.txt",
              "test": f"test_{subset}.txt",
              "rul": f"RUL_{subset}.txt"}
    found = {}
    for root, _, filenames in os.walk(data_dir):
        for filename in filenames:
            found.setdefault(filename, []).append(os.path.join(root, filename))
    paths, missing = {}, []
    for key, filename in wanted.items():
        hits = sorted(found.get(filename, []))
        if not hits:
            missing.append(filename)
        else:
            paths[key] = min(hits, key=lambda c: (os.path.dirname(c) != data_dir, c))
    if missing:
        raise FileNotFoundError(f"Missing C-MAPSS file(s): {', '.join(missing)} under {data_dir}")
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
    """Piecewise-linear RUL for a training engine that fails at its last cycle."""
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

def regime_ids(rows):
    """Operating-condition cluster per row (op settings are columns 3-5)."""
    settings = np.round(rows[:, 2:5], 1)
    _, inv = np.unique(settings, axis=0, return_inverse=True)
    return np.asarray(inv).ravel()


@dataclass
class Preprocessor:
    """Min-max scaler over a fixed set of feature columns (fit on training data only)."""
    keep: np.ndarray | None = None
    low: np.ndarray | None = None
    high: np.ndarray | None = None

    def fit_matrix(self, matrix):
        std = matrix.std(axis=0)
        self.keep = np.where(std > 1e-8)[0]
        if len(self.keep) == 0:
            raise ValueError("All input features are constant")
        m = matrix[:, self.keep]
        self.low, self.high = m.min(axis=0), m.max(axis=0)
        return self

    @property
    def n_features(self):
        return len(self.keep)

    def scale(self, x):
        span = np.maximum(self.high - self.low, 1e-12)
        return (x - self.low) / span


@dataclass
class RegimePreprocessor:
    """Global min-max bounds plus one set of bounds per operating regime.
    All regimes share the global keep set, so the feature width is identical."""
    per_regime: dict = field(default_factory=dict)
    global_pre: Preprocessor | None = None

    def fit(self, engines):
        matrix = np.vstack([rows[:, 2:] for rows in engines.values()])
        self.global_pre = Preprocessor().fit_matrix(matrix)
        keep = self.global_pre.keep
        grouped = {}
        for rows in engines.values():
            ids = regime_ids(rows)
            for r in np.unique(ids):
                grouped.setdefault(int(r), []).append(rows[ids == r][:, 2:][:, keep])
        self.per_regime = {}
        for r, chunks in grouped.items():
            mat = np.vstack(chunks)
            if len(mat) >= 10:
                self.per_regime[r] = Preprocessor(keep=keep, low=mat.min(axis=0),
                                                  high=mat.max(axis=0))
        return self

    @property
    def n_features(self):
        return self.global_pre.n_features

    def transform(self, rows):
        out = np.empty((len(rows), self.n_features))
        ids = regime_ids(rows)
        for r in np.unique(ids):
            m = ids == r
            pre = self.per_regime.get(int(r), self.global_pre)
            out[m] = pre.scale(rows[m][:, 2:][:, self.global_pre.keep])
        return out

    def to_dict(self):
        def plain(p):
            return {"keep": p.keep.tolist(), "low": p.low.tolist(), "high": p.high.tolist()}
        return {"global": plain(self.global_pre),
                "per_regime": {str(k): plain(v) for k, v in self.per_regime.items()}}

    @staticmethod
    def from_dict(d):
        def mk(v):
            return Preprocessor(keep=np.asarray(v["keep"], dtype=int),
                                low=np.asarray(v["low"], dtype=float),
                                high=np.asarray(v["high"], dtype=float))
        rp = RegimePreprocessor()
        rp.global_pre = mk(d["global"])
        rp.per_regime = {int(k): mk(v) for k, v in d["per_regime"].items()}
        return rp


# ------------------------- R-CSNF RESERVOIR -----------------------

class CSNFReservoir:
    """Recurrent complex-valued wave-field reservoir (R-CSNF v3 structure)."""

    def __init__(self, input_dim, alpha_inject=0.6, packet_width=0.35,
                 diffusion=0.01, grid_n=GRID_N, num_lags=NUM_LAGS,
                 pool_bins=POOL_BINS, evolve_steps=EVOLVE_STEPS, potential=POTENTIAL):
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
        self.coupling_amplitude = np.log(2.0)
        self.length = 2 * np.pi
        self.x = np.linspace(0, self.length, grid_n, endpoint=False)
        self.k = 2 * np.pi * np.fft.fftfreq(grid_n, d=self.length / grid_n)
        dt = 1.0 / evolve_steps
        self.propagator = np.exp(-0.5j * diffusion * self.k ** 2 * dt)
        self.vpsi = np.exp(-1j * potential * np.cos(2 * self.x) * dt)
        self.vphi = np.exp(-1j * potential * np.sin(3 * self.x) * dt)

    @property
    def n_features_out(self):
        # 7P - 1 pooled field features + num_lags * input_dim raw lag inputs
        return 7 * self.pool_bins - 1 + self.num_lags * self.input_dim

    def kinetic(self, f):
        return np.fft.ifft(np.fft.fft(f) * self.propagator)

    @staticmethod
    def normalize(psi, phi):
        norm2 = np.vdot(psi, psi).real + np.vdot(phi, phi).real
        if not np.isfinite(norm2) or norm2 < 1e-12:
            n = len(psi)
            return np.ones(n, dtype=complex) / np.sqrt(n), np.zeros(n, dtype=complex)
        norm = np.sqrt(norm2)
        return psi / norm, phi / norm

    def encode(self, lags):
        arc = self.length / self.input_dim
        width = max(1e-8, min(self.packet_width, arc * 0.5))
        weights = np.exp(-0.35 * np.arange(self.num_lags))
        weights /= weights.sum()
        velocity = (np.angle(np.exp(1j * (lags[0] - lags[1])))
                    if self.num_lags > 1 else np.zeros(self.input_dim))
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
            coupling = self.coupling_amplitude * np.exp(-1j * (step + 0.5))
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
        psi, phi = state[:self.n], state[self.n:]

        def pool(v):
            return v.reshape(self.pool_bins, -1).mean(axis=1)

        derivative = np.fft.ifft(1j * self.k * np.fft.fft(psi))
        values = np.concatenate([
            pool(np.abs(psi) ** 2), pool(np.abs(phi) ** 2)[:-1],
            pool(psi.real), pool(psi.imag), pool(phi.real), pool(phi.imag),
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


# ------------------------- READOUT / METRICS ----------------------

def fit_ridge(X, y, regularization=1e-4, rcond=1e-4, w=None):
    """SVD ridge readout on standardized features, with optional row weights."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    w = np.ones(len(y)) if w is None else np.asarray(w, dtype=float)
    wsum = w.sum()
    mean = (X * w[:, None]).sum(0) / wsum
    scale = np.sqrt(((X - mean) ** 2 * w[:, None]).sum(0) / wsum)
    scale[scale < 1e-8] = 1.0
    y_mean = float((y * w).sum() / wsum)
    sw = np.sqrt(w)[:, None]
    Z = (X - mean) / scale * sw
    yw = (y - y_mean) * sw[:, 0]
    U, S, Vt = np.linalg.svd(Z, full_matrices=False)
    factors = S / (S ** 2 + regularization)
    if len(S):
        factors[S <= S[0] * rcond] = 0.0
    weights = Vt.T @ (factors * (U.T @ yw))
    return {"mean": mean, "scale": scale, "weights": weights, "y_mean": y_mean}


def predict_ridge(model, X):
    return ((X - model["mean"]) / model["scale"]) @ model["weights"] + model["y_mean"]


def rul_balance_weights(y, bins=RUL_BINS):
    """Equal total weight per RUL band, so the healthy plateau does not dominate.
    Set BALANCE_RUL = False to train with uniform weights (ablation)."""
    if not BALANCE_RUL:
        return np.ones(len(y))
    idx = np.clip(np.digitize(y, bins[1:-1]), 0, len(bins) - 2)
    counts = np.bincount(idx, minlength=len(bins) - 1).astype(float)
    counts[counts == 0] = 1.0
    return (len(y) / (len(bins) - 1)) / counts[idx]


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((np.asarray(y_true) - np.asarray(y_pred)) ** 2)))


def mae(y_true, y_pred):
    return float(np.mean(np.abs(np.asarray(y_true) - np.asarray(y_pred))))


def phm_score(prediction, truth):
    d = np.asarray(prediction) - np.asarray(truth)
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def causal_mean_features(X, window=5):
    result = np.empty_like(X)
    for t in range(len(X)):
        result[t] = X[max(0, t - window + 1):t + 1].mean(axis=0)
    return np.hstack([X, result])


# ------------------------- POST-PROCESSING ------------------------

def causal_ema_deadband(pred, beta, deadband):
    """Causal smoothing. RUL falls by one per cycle; large jumps are damped."""
    out = np.empty_like(pred, dtype=float)
    out[0] = pred[0]
    for t in range(1, len(pred)):
        expected = out[t - 1] - 1.0
        target = pred[t]
        out[t] = expected if abs(target - expected) < deadband \
            else expected + beta * (target - expected)
    return out


def postprocess(pred, beta, deadband, washout=WASHOUT):
    p = np.asarray(pred, dtype=float).copy()
    if len(p) > washout:
        p[:washout] = p[washout]          # hold the first settled value
    p = causal_ema_deadband(p, beta, deadband)
    return np.clip(p, 0, R_MAX)


# ------------------------- ERROR PROFILE --------------------------

def error_profile(y_true, y_pred, cycles):
    err = np.asarray(y_pred, float) - np.asarray(y_true, float)
    rows = []
    for lo, hi in zip(RUL_BINS[:-1], RUL_BINS[1:]):
        m = (y_true >= lo) & (y_true < hi)
        if m.any():
            rows.append({"rul_band": f"[{lo},{hi})", "n": int(m.sum()),
                         "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                         "bias": float(np.mean(err[m]))})
    for lo, hi in zip(CYCLE_BINS[:-1], CYCLE_BINS[1:]):
        m = (cycles >= lo) & (cycles < hi)
        if m.any():
            rows.append({"cycle_band": f"[{lo},{hi})", "n": int(m.sum()),
                         "rmse": float(np.sqrt(np.mean(err[m] ** 2))),
                         "bias": float(np.mean(err[m]))})
    return rows


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


# ------------------------- PERSISTENCE ----------------------------

def model_to_plain(m):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}


def plain_to_model(d):
    return {k: (np.asarray(v) if isinstance(v, list) else v) for k, v in d.items()}


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


# ------------------------- TUNING ---------------------------------

def make_items(engines, pre):
    return [(pre.transform(engines[i]), rul_targets(len(engines[i])), engines[i])
            for i in sorted(engines)]


def sample_trials(n_trials, seed):
    """Random search over the paper's R-CSNF spaces (equal budget per trial)."""
    rng = np.random.default_rng(seed)
    trials = []
    for _ in range(n_trials):
        trials.append({
            "alpha": float(rng.uniform(*SEARCH_ALPHA)),
            "diffusion": float(10 ** rng.uniform(*np.log10(SEARCH_DIFFUSION))),
            "width": float(rng.uniform(*SEARCH_WIDTH)),
            "lambda": float(10 ** rng.uniform(*np.log10(SEARCH_LAMBDA))),
        })
    return trials


def evaluate_trial(trial, pre, fit_items, val_items):
    """Fit the readout on fit_items, score RUL RMSE over validation cycles after washout."""
    res = CSNFReservoir(input_dim=pre.n_features, alpha_inject=trial["alpha"],
                        packet_width=trial["width"], diffusion=trial["diffusion"])
    X_fit = np.vstack([res.run_engine(X)[WASHOUT:] for X, _, _ in fit_items])
    y_fit = np.concatenate([y[WASHOUT:] for _, y, _ in fit_items])
    model = fit_ridge(X_fit, y_fit, trial["lambda"], w=rul_balance_weights(y_fit))
    preds, truths = [], []
    for X, y, _ in val_items:
        p = np.clip(predict_ridge(model, res.run_engine(X)), 0, R_MAX)
        preds.append(p[WASHOUT:])
        truths.append(y[WASHOUT:])
    return rmse(np.concatenate(truths), np.concatenate(preds))


def best_smoothing(model, res, val_items):
    cache = [(res.run_engine(X), y) for X, y, _ in val_items]
    best = None
    for beta in SMOOTH_BETA_GRID:
        for db in SMOOTH_DEADBAND_GRID:
            preds, truths = [], []
            for F, y in cache:
                p = postprocess(predict_ridge(model, F), beta, db)
                preds.append(p[WASHOUT:])
                truths.append(y[WASHOUT:])
            score = rmse(np.concatenate(truths), np.concatenate(preds))
            if best is None or score < best["score"]:
                best = {"score": score, "beta": beta, "deadband": db}
    return best


# ------------------------- MAIN PIPELINE --------------------------

def run_pipeline(subset="FD001", data_dir=None, out_dir=None, seed=SEED,
                 trials=DEFAULT_TRIALS):
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

    checkpoint_path = os.path.join(run_dir, f"training_checkpoint_seed{seed}.pkl")
    checkpoint = None
    if os.path.isfile(checkpoint_path):
        try:
            with open(checkpoint_path, "rb") as h:
                cand = pickle.load(h)
            expected = {"pipeline_version": __version__, "subset": subset, "seed": seed,
                        "train_sha256": train_hash, "test_sha256": test_hash,
                        "trials": trials}
            if all(cand.get(k) == v for k, v in expected.items()):
                checkpoint = cand
                log(f"[RESUME] Compatible checkpoint: {checkpoint_path}")
            else:
                log("[RESUME] Checkpoint does not match configuration; retraining.")
        except Exception as exc:
            log(f"[RESUME] Checkpoint unreadable ({exc}); retraining.")

    if checkpoint is not None:
        best = checkpoint["best"]
        smooth = checkpoint["smoothing"]
        profile = checkpoint.get("validation_error_profile", [])
        search_log = checkpoint.get("search_log", [])
        pre = RegimePreprocessor.from_dict(checkpoint["preprocessor"])
        model = plain_to_model(checkpoint["reservoir_model"])
        baseline = plain_to_model(checkpoint["baseline_model"])
        reservoir = CSNFReservoir(input_dim=pre.n_features, alpha_inject=best["alpha"],
                                  packet_width=best["width"], diffusion=best["diffusion"])
    else:
        log("[2] Engine-level validation split")
        fit_ids, val_ids = split_engine_ids(train_engines.keys(), seed=seed)
        fit_engines = {i: train_engines[i] for i in fit_ids}
        val_engines = {i: train_engines[i] for i in val_ids}

        log(f"[3] Random search over paper's R-CSNF space ({trials} trials)")
        pre_val = RegimePreprocessor().fit(fit_engines)
        fit_items = make_items(fit_engines, pre_val)
        val_items = make_items(val_engines, pre_val)
        search_log, best = [], None
        for i, trial in enumerate(sample_trials(trials, seed)):
            score = evaluate_trial(trial, pre_val, fit_items, val_items)
            search_log.append({**trial, "val_rmse": score})
            log(f"  trial {i + 1:2d}: alpha={trial['alpha']:.3f} D={trial['diffusion']:.4g} "
                f"width={trial['width']:.3f} lambda={trial['lambda']:.2e} -> val RMSE={score:.3f}")
            if best is None or score < best["score"]:
                best = {**trial, "score": score}
        log(f"Best reservoir settings: {best}")

        log("[3b] Causal smoothing and validation error profile")
        res_val = CSNFReservoir(input_dim=pre_val.n_features, alpha_inject=best["alpha"],
                                packet_width=best["width"], diffusion=best["diffusion"])
        X_fit = np.vstack([res_val.run_engine(X)[WASHOUT:] for X, _, _ in fit_items])
        y_fit = np.concatenate([y[WASHOUT:] for _, y, _ in fit_items])
        cand_model = fit_ridge(X_fit, y_fit, best["lambda"], w=rul_balance_weights(y_fit))
        smooth = best_smoothing(cand_model, res_val, val_items)
        log(f"Best smoothing: {smooth}")
        vp_t, vp_p, vp_c = [], [], []
        for X, y, _ in val_items:
            p = postprocess(predict_ridge(cand_model, res_val.run_engine(X)),
                            smooth["beta"], smooth["deadband"])
            vp_t.append(y)
            vp_p.append(p)
            vp_c.append(np.arange(1, len(y) + 1))
        profile = error_profile(np.concatenate(vp_t), np.concatenate(vp_p),
                                np.concatenate(vp_c))

        log("[4] Final training on all training engines (selected settings)")
        pre = RegimePreprocessor().fit(train_engines)
        train_items = make_items(train_engines, pre)
        reservoir = CSNFReservoir(input_dim=pre.n_features, alpha_inject=best["alpha"],
                                  packet_width=best["width"], diffusion=best["diffusion"])
        X_train = np.vstack([reservoir.run_engine(X)[WASHOUT:] for X, _, _ in train_items])
        y_train = np.concatenate([y[WASHOUT:] for _, y, _ in train_items])
        w_train = rul_balance_weights(y_train)
        model = fit_ridge(X_train, y_train, best["lambda"], w=w_train)
        B_train = np.vstack([causal_mean_features(X)[WASHOUT:] for X, _, _ in train_items])
        baseline = fit_ridge(B_train, y_train, best["lambda"], w=w_train)

        payload = {"pipeline_version": __version__, "subset": subset, "seed": seed,
                   "train_sha256": train_hash, "test_sha256": test_hash,
                   "trials": trials, "best": best, "smoothing": smooth,
                   "search_log": search_log,
                   "validation_error_profile": profile,
                   "preprocessor": pre.to_dict(),
                   "reservoir_model": model_to_plain(model),
                   "baseline_model": model_to_plain(baseline)}
        tmp = checkpoint_path + ".tmp"
        with open(tmp, "wb") as h:
            pickle.dump(payload, h, protocol=pickle.HIGHEST_PROTOCOL)
            h.flush()
            os.fsync(h.fileno())
        os.replace(tmp, checkpoint_path)
        log(f"[CHECKPOINT] Saved: {checkpoint_path}")

    log("[5] Test evaluation (last cycle of each test engine, official RUL labels)")
    truth_capped = np.minimum(test_rul, R_MAX)
    engines_out, res_last, base_last = [], [], []
    for i, engine_id in enumerate(test_ids):
        rows = test_engines[engine_id]
        X = pre.transform(rows)
        raw = predict_ridge(model, reservoir.run_engine(X))
        res_pred = postprocess(raw, smooth["beta"], smooth["deadband"])
        base_pred = np.clip(predict_ridge(baseline, causal_mean_features(X)), 0, R_MAX)
        res_last.append(res_pred[-1])
        base_last.append(base_pred[-1])
        sev = severity_band(float(res_pred[-1]))
        engines_out.append({
            "id": int(engine_id), "cycles_observed": int(len(rows)),
            "pred_rul": round(float(res_pred[-1]), 2), "true_rul": float(test_rul[i]),
            "severity": sev, "action": triage_action(sev),
            "history": [round(float(v), 2) for v in res_pred],
            "sensors": {f"s{k}": float(rows[-1][4 + k]) for k in range(1, 22)},
        })
    res_last, base_last = np.asarray(res_last), np.asarray(base_last)

    metrics = {
        "subset": subset, "pipeline_version": __version__, "seed": seed,
        "generated_at": store.now_iso(),
        "train_sha256": train_hash, "test_sha256": test_hash,
        "test_engines": len(test_ids), "trials": trials,
        "best_parameters": best, "smoothing": smooth,
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
        "validation_error_profile": profile,
    }
    model_version = make_model_version(subset, seed, train_hash, best)
    metrics["model_version"] = model_version
    for name in ("reservoir", "ridge_baseline"):
        log(f"{name}: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics[name].items()))
    for row in profile:
        log("  profile: " + json.dumps(row))

    log("[6] Persisting artifact, predictions, work orders and audit log")
    Artifact(
        model_version=model_version, subset=subset, seed=seed,
        best_parameters={**best, "smoothing": smooth},
        preprocessor=pre.to_dict(),
        reservoir_config={"alpha_inject": best["alpha"], "packet_width": best["width"],
                          "diffusion": best["diffusion"], "input_dim": pre.n_features},
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
            store.write_run(con, {"run_id": model_version, "subset": subset,
                                  "model_version": model_version, "seed": seed,
                                  "train_sha256": train_hash, "test_sha256": test_hash,
                                  "created_at": created_at,
                                  "metrics_json": json.dumps(metrics)})
            store.write_predictions(con, model_version, engines_out)
            new_wo = store.write_work_orders(con, model_version, subset, work, created_at)
            new_exc = store.write_exceptions(con, model_version, subset, exc, created_at)
        total_wo = store.count(con, "maintenance_records")
        total_exc = store.count(con, "exceptions")
    finally:
        con.close()

    metrics.update({"work_orders_new": new_wo, "work_orders_total": total_wo,
                    "exceptions_new": new_exc, "exceptions_total": total_exc})
    with open(os.path.join(run_dir, "predictions.json"), "w", encoding="utf-8") as f:
        json.dump({"subset": subset, "model_version": model_version,
                   "engines": engines_out}, f)
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(run_dir, "search_log.json"), "w", encoding="utf-8") as f:
        json.dump(search_log, f, indent=2)

    log(f"Work orders: {new_wo} new / {total_wo} total | "
        f"exceptions: {new_exc} new / {total_exc} total")
    log(f"Pipeline completed for {subset} (model {model_version}).")
    return metrics


def predict_with_artifact(artifact, rows):
    """Reproduce predictions for one engine's raw rows from a saved artifact."""
    pre = RegimePreprocessor.from_dict(artifact.preprocessor)
    cfg = artifact.reservoir_config
    res = CSNFReservoir(input_dim=cfg["input_dim"], alpha_inject=cfg["alpha_inject"],
                        packet_width=cfg["packet_width"], diffusion=cfg["diffusion"])
    raw = predict_ridge(plain_to_model(artifact.reservoir_model),
                        res.run_engine(pre.transform(rows)))
    smooth = artifact.best_parameters["smoothing"]
    return postprocess(raw, smooth["beta"], smooth["deadband"])


# ------------------------- SELF-TEST ------------------------------

def run_selftest():
    assert np.allclose(rul_targets(5, 100), [4, 3, 2, 1, 0])
    rng = np.random.default_rng(123)
    X = rng.normal(size=(30, 4))
    y = np.linspace(30, 0, 30)
    res = CSNFReservoir(input_dim=4)
    F = res.run_engine(X)
    assert F.shape == (30, res.n_features_out) and np.isfinite(F).all()
    pred = predict_ridge(fit_ridge(F, y, 1e-4, w=rul_balance_weights(y)), F)
    assert pred.shape == y.shape and np.isfinite(pred).all()
    assert severity_band(5) == "critical" and severity_band(20) == "high"
    assert severity_band(45) == "medium" and severity_band(100) == "low"
    assert severity_band(float("nan")) == "review"
    assert len(sample_trials(4, 0)) == 4
    print("Self-test passed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AERO-SYNAPSE C-MAPSS pipeline")
    parser.add_argument("--subset", default="FD001", choices=SUBSETS)
    parser.add_argument("--data-dir", default=os.environ.get("AERO_DATA_DIR", "data"))
    parser.add_argument("--out-dir", default=os.environ.get("AERO_OUT_DIR", "runs"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        run_selftest()
    else:
        run_pipeline(args.subset, args.data_dir, args.out_dir, args.seed, args.trials)
