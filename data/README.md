# Data

Neither folder is tracked by git; the raw files are several GB.

## `raw/` — download these three files

From the Toyota Research Institute data platform,
<https://data.matr.io/1/projects/5c48dd2bc625d700019f3204> ("Data-driven prediction of
battery cycle life before capacity degradation"), open each batch page and download its
MATLAB struct:

| Batch | File | Cells (raw → cleaned) |
|---|---|---|
| 2017-05-12 | `2017-05-12_batchdata_updated_struct_errorcorrect.mat` | 46 → 41 |
| 2017-06-30 | `2017-06-30_batchdata_updated_struct_errorcorrect.mat` | 48 → 43 |
| 2018-04-12 | `2018-04-12_batchdata_updated_struct_errorcorrect.mat` | 46 → 40 |

Put them directly in `data/raw/`. The file names are listed in `configs/default.yaml`.

## `preprocessed/` — written by `scripts/preprocess.py`

- `cells.pkl` — dict of 124 cleaned cells (`b1c0`, `b1c1`, ...). See the docstring in
  `src/battery_ml/data/loader.py` for the fields.
- `cells_summary.csv` — one row per cell: batch, charging policy, cycle life, split.

## Conventions to check during exploration

- **Curve indexing.** Row `k` of `Qdlin` / `Tdlin` is entry `k` of the struct's `cycles`
  array, and the code treats that as cycle number `k`. Severson's ΔQ₁₀₀₋₁₀ is then
  `Qdlin[100] - Qdlin[10]`. Confirm this against `summary["cycle"]` in your first notebook.
- **Summary indexing.** Summary features look cycles up by value in `summary["cycle"]`,
  not by array position, so they do not depend on that convention.
- **Voltage grid.** `Qdlin` is interpolated on 1000 points from 3.6 V down to 2.0 V
  (`VOLTAGE_GRID` in `loader.py`).
