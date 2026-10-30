# CLAUDE.md

Guidance for Claude Code when working in this repository.

## Project

EEL 5825 course project: predict a Li-ion cell's **cycle life** (cycles until capacity
falls to 80% of nominal) from only its first 100 / 50 / 20 / 10 cycles, with calibrated
prediction intervals. The data is the 124-cell LFP/graphite fast-charging dataset from
Severson et al. (2019) [1]. The proposal is `reports/proposal.pdf`, and `README.md` has
user-facing setup and run instructions.

Proposal timeline: **Weeks 1–2** data processing + baselines (done) · **Weeks 3–4** 1D CNN +
early-prediction study · **Weeks 5–6** Gaussian process vs deep ensembles · **Week 7**
analysis and report.

## Current status

| Component | Location | State |
|---|---|---|
| HDF5 loader + published cleaning (→ 124 cells) | `src/battery_ml/data/loader.py` | done, tested on a fake HDF5 file |
| Original train / primary / secondary splits (41 / 43 / 40) | `src/battery_ml/data/splits.py` | done, tested |
| Severson `variance` / `discharge` / `full` features, any window | `src/battery_ml/features/severson.py` | done, tested |
| ΔQ(V) curve stacking for deep models | `src/battery_ml/features/curves.py` | done, tested |
| Elastic net, random forest, gradient boosting | `src/battery_ml/models/baselines.py`, `scripts/train_baselines.py` | done |
| RMSE, MAPE, interval coverage and width | `src/battery_ml/evaluation/metrics.py` | done, tested |
| Result figures (metric bars, predicted vs observed) | `src/battery_ml/evaluation/plots.py`, `scripts/plot_baselines.py` | done, smoke-tested |
| 1D CNN (`CycleLifeCNN`) | `src/battery_ml/models/cnn.py`, `scripts/train_cnn.py` | **stub** (`NotImplementedError`) |
| Gaussian process (`fit_gp`, `predict_interval`) | `src/battery_ml/models/gaussian_process.py` | **stub** |
| Deep ensemble (`DeepEnsemble`) | `src/battery_ml/models/deep_ensemble.py` | **stub** |
| GP vs ensemble comparison | `scripts/run_uncertainty.py` | **stub** |

The raw data is in `data/raw/`, and `scripts/preprocess.py` has been run (2026-10-29). It
wrote `cells.pkl` (~120 MB, 124 cells, splits 41 / 43 / 40, cycle life 148–2237) and
`cells_summary.csv`. The baselines have been run for the 100-cycle window on all three
feature sets (`results/metrics/baselines_*.csv`). The variance elastic net reproduces
Severson's Table 1 (RMSE 103.6 / 138.0 / 196.0 cycles on train / primary / secondary).
Figures for the 100-cycle results are in `results/figures/baselines_w100_*.{png,pdf}`.
Not done yet: the 50 / 20 / 10-cycle windows, and there are no notebooks.
The stub docstrings and `# TODO` comments describe the intended design. Follow them when
implementing.

## Environment and commands

Windows 11, PowerShell. Python 3.12 venv at `.venv/` with the package installed editable
(`pip install -e .`). PyTorch is the CUDA 12.8 build (RTX 50-series GPU).

```powershell
.venv\Scripts\python.exe -m pytest                       # 18 tests, synthetic data, ~5 s
.venv\Scripts\python.exe scripts/preprocess.py           # raw .mat -> data/preprocessed/cells.pkl + cells_summary.csv
.venv\Scripts\python.exe scripts/train_baselines.py --feature-set variance
.venv\Scripts\python.exe scripts/train_baselines.py --feature-set full --windows 100 50 --models elastic_net
.venv\Scripts\python.exe scripts/plot_baselines.py --window 100   # -> results/figures/baselines_w100_*.png/.pdf
```

Every script takes `--config` (default `configs/default.yaml`). The training scripts take
`--windows`. Baseline scores are written to `results/metrics/baselines_<feature_set>.csv`,
and per-cell predictions to `..._predictions.csv`. New scripts should write the same
long format: `window, model, split, rmse, mape[, coverage, width]`.

## Architecture

The pipeline is `scripts/*.py` (thin CLI entry points) → `src/battery_ml/` (importable
package) → `configs/default.yaml` (every path, window, and hyperparameter).

- **`config.load_config`** resolves `paths.*` against the repo root, so scripts work from
  any working directory.
- **`data/loader.py`** reads the three MATLAB v7.3 structs (which are HDF5) with h5py and
  applies the cleaning from the authors' starter code. It drops 5 batch-1 cells that never
  reached end of life, merges 5 batch-2 continuation cells into their batch-1 cells (adding
  cycles to cycle life), and drops 6 noisy batch-3 cells. Each cell is a dict with
  `cycle_life`, `policy`, `batch`, `n_cycles`, `summary` (per-cycle arrays: `cycle, QD, QC,
  IR, Tavg, Tmin, Tmax, chargetime`), and `Qdlin` / `Tdlin` of shape `(keep_cycles, 1000)`
  on `VOLTAGE_GRID` (3.6 V → 2.0 V). Only the first `keep_cycles` (120) curves are kept.
- **`data/splits.py`** reproduces Severson's split. Batches 1 and 2 are sorted by
  (batch, channel). Even positions plus the last cell form the primary test set and odd
  positions form train. All of batch 3 is the secondary test set.
- **`features/severson.py`** builds every feature from ΔQ(V) = Q_last(V) − Q_ref(V), with
  cycles numbered from 1 (`delta_q` reads rows `last - 1` and `ref - 1`).
  The `variance` set has 1 feature, `discharge` has 6, and `full` has 9. The `(ref, last)`
  pair per window comes from `windows:` in the config.
- **`models/__init__.py`** exports only the sklearn baselines, so the classical pipeline
  works without importing torch. Import `models.cnn` and `models.deep_ensemble` directly.
- **`evaluation/plots.py`** builds figures from the long-format results CSVs. It uses
  `matplotlib.figure.Figure` directly (no pyplot), and `use_style()` applies the shared
  style. Color always means the split: train is gray, primary test is blue `#2a78d6`, and
  secondary test is orange `#eb6834` (validated for CVD). Keep that mapping in new figures.
  It also holds `SEVERSON_TABLE1`, the published elastic-net numbers. It isn't exported from
  `evaluation/__init__`, so training scripts don't import matplotlib.

## Conventions and gotchas

- **Target scale.** Models regress `log10(cycle_life)` when `features.log_target` is true.
  Convert predictions and interval bounds back with `10 ** x` **before** calling the
  metrics. Every function in `evaluation/metrics.py` expects cycles, not log cycles.
- **No test leakage.** Fit scalers, standardization statistics, and hyperparameter
  searches on the `train` split only. The test splits exist so results can be compared
  with the published numbers.
- **Cycle numbering.** Every cycle number in the code (config `windows`, `delta_q`,
  `_summary_window`) is 1-based, matching `summary["cycle"]`. `Qdlin` row `j` is cycle
  `j + 1`. This was verified on the real data: cycle-to-cycle changes in `Qdlin[:, -1]`
  correlate with changes in `summary["QD"]` only at the same array position (r ≈ 0.52;
  ≈ 0 at ±1, ±2). Index curves by cycle with `[cycle - 1]`. Fixing an earlier off-by-one
  here moved the variance elastic net from 122 / 137 / 211 to 104 / 138 / 196 RMSE,
  which matches [1].
- **Glitches in summary `QD` (unresolved).** Single-cycle spikes: b1c0 at cycle 12
  (1.539 Ah) and b1c18 at cycle 40 (2.884 Ah), against about 1.07 Ah typical. Both are
  primary-test cells. They inflate `qd_max_minus_cycle2` (`discharge` set), so the
  discharge elastic net predicts about 0 cycles for both and gets 316 primary RMSE
  (paper: about 91). b2c1's low `QD` near cycle 100 is real fast fade, not a glitch.
- **Batch-1 row 0.** `Qdlin` / `Tdlin` row 0 (summary cycle 1) is all-NaN for every
  batch-1 cell, and batches 2 and 3 have no NaN rows. No configured window touches it,
  but deep models that take raw curve stacks must skip or mask it.
- **Window reference cycles.** Only `100: {ref: 10, last: 100}` comes from the paper. The
  `ref` values for the 50, 20, and 10-cycle windows are placeholders that still need to be
  chosen and justified in the report.
- **Outlier cell and Table 1.** Severson's parenthetical primary-test numbers exclude "one
  battery in the test set [that] reaches 80% SOH rapidly". That is **b2c1** (148 cycles;
  the next-shortest primary-test cell lasts 335). Table 1 is in `evaluation/plots.py`
  (`SEVERSON_TABLE1`), taken from the matching table in US patent 11,226,374. Our primary
  test set includes b2c1, so compare against the non-parenthetical values.
- **NumPy version.** `features/severson.py` uses `np.trapezoid`, which requires NumPy ≥ 2.0,
  but `requirements.txt` still allows `numpy>=1.26`. The venv has NumPy 2.x.
- **Tests never need the real data.** `tests/conftest.py` builds synthetic cells, and
  `tests/test_loader.py` writes a fake HDF5 file with the same reference layout as the
  data.matr.io structs. Keep new tests self-contained in the same way.
- **Git.** `data/raw/*`, `data/preprocessed/*`, and `results/*/*` are gitignored (the raw
  files are several GB). Only the `.gitkeep` files are tracked.
- **Style.** Short module docstrings that cite references as [1] / [3], small functions,
  config-driven hyperparameters (don't hard-code them in scripts), and `seed` from the
  config passed to every estimator.

## References

1. K. A. Severson et al., "Data-driven prediction of battery cycle life before capacity
   degradation," *Nature Energy* 4, 383–391 (2019). Data: <https://data.matr.io/1/projects/5c48dd2bc625d700019f3204>
2. P. M. Attia et al., "Closed-loop optimization of fast-charging protocols…," *Nature* 578, 397–402 (2020).
3. B. Lakshminarayanan, A. Pritzel, C. Blundell, "Simple and scalable predictive uncertainty
   estimation using deep ensembles," *NeurIPS* (2017).
