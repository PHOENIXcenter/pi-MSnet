```markdown
# Peptide RT Confidence Toolkit

A complete pipeline to:
- convert peptide retention time data from Parquet files into **HDF5 shards** and compute per‑peptide statistics,
- load the shards into a **DuckDB** database for fast querying,
- compute **confidence scores** for predicted retention times (four methods + average),
- generate **SVG threshold‑count plots** grouped by prediction tool.

---

## Workflow Overview

```
Parquet files → HDF5 shards + summary CSV → DuckDB database → confidence CSV + SVG plots
```

1. **Build HDF5 shards & summary** – `build_hdf5_shards.py`  
   Reads `*.parquet` files, normalises RT values, groups peptides by first letter and saves them into compressed HDF5 shards. Then calculates per‑peptide statistics (count, mean, std, median, IQR, percentiles) and writes `peptide_rt_statistics_summary.csv`.

2. **Populate DuckDB** – `hdf5_to_duckdb.py`  
   Reads the HDF5 shards and inserts all RT values into a DuckDB table (`peptide_rt`). Creates indexes for fast lookup.

3. **Confidence & visualisation** – `batch_predict_confidence_and_plot_duckdb.py`  
   For a given CSV of peptide predictions (columns: `Peptide`, `Tool`, `Predict`) it calls the confidence calculator (four methods + average) and saves the enriched CSV. It also plots the number of peptides above each confidence threshold for every tool, exporting one SVG per method.

---

## Repository Structure

| File | Description |
|------|-------------|
| `build_hdf5_shards.py` | Convert Parquet RT data to HDF5 shards and compute peptide statistics. |
| `hdf5_to_duckdb.py` | Load HDF5 shards into a DuckDB database. |
| `duckdb_calculate_confidence.py` | Library containing `ShardedPeptideConfidenceCalculator`. |
| `batch_predict_confidence_and_plot_duckdb.py` | Batch confidence calculation and threshold‑count curve plotting. |

---

## Dependencies

- Python 3.7+
- `pandas`, `numpy`, `scipy`, `matplotlib`
- `h5py`
- `duckdb`
- `tqdm`
- `pyarrow` or `fastparquet` (for Parquet reading)
- (optional) `multiprocessing` is part of the standard library for parallel statistics

Install core packages:
```bash
pip install pandas numpy scipy matplotlib h5py duckdb tqdm pyarrow
```

---

## Detailed Usage

### Step 1 – Build HDF5 Shards and Summary Statistics

Place your Parquet files in the expected directory structure:
```
huizong/
└── */
    └── peptidoform_retention/
        └── *.parquet
```

Each Parquet file must contain the columns **`peptidoform`**, **`retention`**, and **`effective_rt`**.

Run:
```bash
python build_hdf5_shards.py
```

This will:
- create a `peptide_shards/` directory with HDF5 files (`peptide_rt_data_A.h5`, `peptide_rt_data_B.h5`, …)
- save a shard index mapping (`shard_index.json`)
- generate `peptide_rt_statistics_summary.csv` containing per‑peptide aggregated metrics

**Configuration** (edit inside the script):
- `root_dir`: directory containing the `huizong` folder (default: `'huizong'`)
- `shard_size`: max peptides per shard before splitting
- `summary_filename`: output CSV name

---

### Step 2 – Create the DuckDB Database

Make sure the HDF5 shards are in `peptide_shards/` (or adjust `H5_DIR` inside `hdf5_to_duckdb.py`). Then:
```bash
python hdf5_to_duckdb.py
```

This produces:
- `peptide_rt.duckdb` (the DuckDB database with a `peptide_rt` table)
- `filtered_records_log.csv` (logs of invalid or filtered values)

**Configuration** (inside the script):
- `H5_DIR`: path to HDF5 shards (default `peptide_shards`)
- `DB_FILE`: output database file
- `BATCH_SZ`: insert batch size
- `ENABLE_INDEX`: whether to build indexes on `peptidoform` and `rt_value`

---

### Step 3 – Compute Confidence Scores and Generate Plots

Prepare a prediction CSV with the following columns:
- **Peptide** – peptide identifier (must match the database)
- **Tool** – name of the prediction tool (used for plot grouping)
- **Predict** – predicted retention time

Run:
```bash
python batch_predict_confidence_and_plot_duckdb.py your_predictions.csv
```

**Outputs**:
- `your_predictions_confidence.csv` – original columns + 5 confidence columns:
  - `method1_quantile`
  - `method2_robust_zscore`
  - `method3_distance`
  - `method4_kde`
  - `average_confidence`
- Five SVG files (one per confidence method) showing peptide counts above each confidence threshold, split by tool.

**Configuration** (inside the script):
- `SUMMARY_CSV`: path to the statistics CSV (default `peptide_rt_statistics_summary.csv`)
- `THRESHOLDS`: array of confidence thresholds for the curves

---

## Confidence Methods Explained

| Method | Description |
|--------|-------------|
| **Quantile** | Measures how far the predicted RT is from the median based on the empirical distribution. |
| **Robust Z‑score** | Uses the median and IQR to compute a Z‑score; p‑value from t‑distribution adjusts for sample size. |
| **Distance** | Distance from the median relative to a spread metric (IQR or standard deviation). |
| **KDE** | Gaussian kernel density estimation; compares predicted density to the maximum observed density. Falls back to quantile if KDE fails. |
| **Average** | Arithmetic mean of the four methods. |

---

## Notes

- The DuckDB connection uses a singleton that keeps the database open during confidence calculations. Do not move or delete `peptide_rt.duckdb` while a script is running.
- All scripts print progress information and can be interrupted safely; partial data is saved.
- The parallel statistics computation (`calculate_peptide_statistics_parallel`) is automatically triggered when the number of peptides exceeds 100,000 (can be adjusted).