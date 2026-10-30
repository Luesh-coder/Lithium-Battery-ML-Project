# Early Prediction of Li-Ion Battery Cycle Life with Uncertainty Quantification

EEL 5825 course project. The goal is to predict a cell's cycle life (cycles until capacity
falls to 80% of nominal) from its first 100, 50, 20, or 10 cycles, with calibrated
prediction intervals. The data is the 124-cell LFP/graphite fast-charging dataset from
Severson et al. (2019). The full proposal is in [reports/proposal.pdf](reports/proposal.pdf).

## Project layout

```
├── configs/default.yaml        all paths, windows, and hyperparameters
├── data/
│   ├── raw/                    the 3 batch .mat files (download; see data/README.md)
│   └── preprocessed/           cells.pkl + cells_summary.csv (from scripts/preprocess.py)
├── notebooks/                  exploration and figures for the report
├── reports/                    proposal and final report
├── results/
│   ├── figures/
│   ├── metrics/                CSV scores and predictions from each script
│   └── models/                 saved checkpoints
├── scripts/                    entry points, run in order
│   ├── preprocess.py           raw .mat -> cleaned 124-cell dataset
│   ├── train_baselines.py      elastic net / random forest / gradient boosting
│   ├── plot_baselines.py       result figures -> results/figures/
│   ├── train_cnn.py            1D CNN on Q(V) curves              (to implement)
│   └── run_uncertainty.py      Gaussian process vs deep ensembles (to implement)
├── src/battery_ml/             importable package used by scripts and notebooks
│   ├── config.py
│   ├── data/                   loader.py (h5py reader + cleaning), splits.py
│   ├── features/               severson.py (variance/discharge/full), curves.py
│   ├── models/                 baselines.py, cnn.py, gaussian_process.py, deep_ensemble.py
│   └── evaluation/             metrics.py (RMSE, MAPE, interval coverage/width), plots.py
└── tests/                      pytest suite (runs on synthetic data, no download needed)
```

## Setup (Windows, PowerShell)

Use Python 3.12. PyTorch does not reliably ship wheels for the newest Python release.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
# CUDA build of PyTorch (RTX 50-series needs CUDA 12.8 or newer)
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
pip install -e .
pytest
```

If PowerShell blocks `Activate.ps1`, run
`Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once, or call
`.venv\Scripts\python.exe` directly.

## Running the pipeline

1. **Download the data** into `data/raw/` (see [data/README.md](data/README.md)).
2. **Preprocess** (reads each HDF5 file once; takes a few minutes):
   ```powershell
   python scripts/preprocess.py
   ```
3. **Baselines and the early-prediction study** (every window in the config):
   ```powershell
   python scripts/train_baselines.py --feature-set variance
   python scripts/train_baselines.py --feature-set full --windows 100 50
   ```
   Scores go to `results/metrics/baselines_<feature-set>.csv`. Plot them (PNG + PDF in
   `results/figures/`) with:
   ```powershell
   python scripts/plot_baselines.py --window 100
   ```
4. **1D CNN**: implement `src/battery_ml/models/cnn.py`, then run `scripts/train_cnn.py`.
5. **Uncertainty**: implement `gaussian_process.py` and `deep_ensemble.py`, then run
   `scripts/run_uncertainty.py`.

## Mapping to the proposal

| Proposal | Where | Status |
|---|---|---|
| Data processing | `data/loader.py`, `scripts/preprocess.py` | ready |
| Original train / primary / secondary test splits from [1] | `data/splits.py` | ready |
| ΔQ(V) variance feature (plus discharge and full sets from [1]) | `features/severson.py` | ready |
| 1. Elastic net, random forest, gradient boosting | `models/baselines.py`, `scripts/train_baselines.py` | ready |
| 2. 1D CNN on Q(V) curves | `models/cnn.py`, `scripts/train_cnn.py` | stub |
| 3. Early prediction with 100 / 50 / 20 / 10 cycles | `windows:` in `configs/default.yaml` | ready for baselines |
| 4. Gaussian process vs deep ensembles | `models/gaussian_process.py`, `models/deep_ensemble.py` | stub |
| RMSE, MAPE, interval coverage and width | `evaluation/metrics.py` | ready |

## Notes

- **Splits.** Train has 41 cells, primary test 43, secondary test 40, following [1].
  Severson et al. also report primary-test results with one outlier cell removed. That
  cell is b2c1, which reaches 80% capacity after 148 cycles. Our primary test set keeps
  it, so compare against their non-parenthetical numbers.
- **Target.** Models regress log10(cycle life) by default (`features.log_target`), and
  metrics are computed after converting back to cycles.
- **Shorter windows.** The reference cycles for the 50, 20, and 10-cycle windows in
  `configs/default.yaml` are placeholders. Choose and justify them in the report.

## References

1. K. A. Severson et al., "Data-driven prediction of battery cycle life before capacity
   degradation," *Nature Energy*, vol. 4, pp. 383–391, 2019.
2. P. M. Attia et al., "Closed-loop optimization of fast-charging protocols for batteries
   with machine learning," *Nature*, vol. 578, pp. 397–402, 2020.
3. B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive
   uncertainty estimation using deep ensembles," *NeurIPS*, 2017.
