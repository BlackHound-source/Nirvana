# AERO-SYNAPSE

Turbofan remaining-useful-life prediction on NASA C-MAPSS, with deterministic
triage into an idempotent maintenance log and a dashboard.

## Reproduce
1. Download the NASA C-MAPSS archive and place `train_FD00x.txt`,
   `test_FD00x.txt`, `RUL_FD00x.txt` in `data/` (or a ZIP containing them).
2. `make install`
3. `make test`                  # unit + synthetic end-to-end tests
4. `make pipeline-all`          # runs FD001–FD004, writes runs/<subset>/
5. `python server.py`           # dashboard at http://127.0.0.1:8000

## Outputs per subset (runs/<SUBSET>/)
- `metrics.json`          RMSE/MAE/PHM for reservoir and ridge baseline
- `model.pkl`             trained artifact (weights, preprocessing, hyperparameters, version)
- `predictions.json`      per-engine predictions and history
- `maintenance.sqlite`    tables: runs, predictions, maintenance_records, exceptions
- `audit.jsonl`           one line per triage decision

## Evaluation protocol
Both models are scored on the last observed cycle of each test engine against
RUL_FDxxx.txt, capped at 125. Hyperparameters are chosen on a 20% engine-level
validation split of the training set; the test set is never used for selection.

## Limitations
- The reservoir is a deterministic feature map, not a validated physics model.
- Results are reported only for subsets actually run; the dashboard shows
  unavailable subsets greyed out.
- Synthetic test validates plumbing only, not accuracy.
- Operational thresholds (10/30/60 cycles) are policy choices, not calibrated.
