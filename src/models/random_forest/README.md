# Random forest (RF)

Random-forest models for the full-bloom forecast, from the first grid search to the
figures in the final report. Notebooks are numbered in run order; each writes to the
matching folder in [`results/rf/`](../../../results/rf/).

| # | Notebook | What it does | Output `results/rf/…` | Rerun in clean-up? |
|---|---|---|---|---|
| 01 | `01_rf_raw_target.ipynb` | First experiment: predict full-bloom DOY directly. 96 candidates × 6 datasets × 2 training windows (`all` / last `30y`) | `01_raw_target/` | No – record only. Stopped after 7 of 12 conditions (no Feb1, no Mar1-small-30y) |
| 02 | `02_rf_anomaly_grid.ipynb` | Main grid search on the **anomaly target** (DOY − 30-year station reference). 54 candidates × Feb1/Mar1/Mar15 × large/small | `02_anomaly_grid/` | No – code checked: it reproduces the saved scores exactly |
| 03 | `03_rf_mar15_refined_grid.ipynb` | Finer grid (27 candidates) around the best 15 March small-sample model | `03_mar15_refined/` | Yes (~10 min) |
| 04 | `04_rf_results_and_figures.ipynb` | Picks the best RF per issue date, rebuilds its validation predictions, makes the report tables and figures | `04_selected_models/` | Yes (~1 min) |
| 05 | `05_rf_permutation_importance.ipynb` | 5-fold permutation importance of the best 15 March RF, single features and correlated groups | `05_permutation_importance/` | Yes (~5 min) |

`rf_common.py` holds what the notebooks share: finding the repository root, loading a
sample, the station reference, fitting/predicting the anomaly RF and saving CSVs.

The input-drift (PSI) analysis is not an RF analysis and lives in
[`src/monitoring/`](../../monitoring/).

## Setup used by every notebook

- **Data:** `data/model/model_features_{feb1,mar1,mar15}.csv` through `model_data.load_features(..., source="csv")`.
  Feb1 has 20 features, Mar1 23, Mar15 36 (every non-ID column).
- **Samples:** `large` = all 82 stations (4,365 rows; 3,837 development).
  `small` = the 48 stations with a 2025 record (2,972 rows; 2,444 development).
- **Validation:** the five agreed expanding-window folds (validation seasons 2005–2014, two per fold).
  Models are ranked by the plain mean of the five fold scores. **Test seasons 2015–2025 are never used here.**
- **Anomaly target (02–05):** station reference = the station's mean full-bloom DOY over the last 30 years of the
  fold's training data (fewer than 10 records → region mean → overall mean). The forest learns DOY − reference on
  all training years of the fold; the reference is added back before scoring.
- `random_state=42` everywhere. Results do not depend on `n_jobs`.

## Results in one table

| Issue date | Best RF (all CV, mean of 5 folds) | Sample | MAE | RMSE | R² | ±3-day hit |
|---|---|---|---|---|---|---|
| 1 Feb | anomaly · 200 trees · depth 5 · leaf 30 · sqrt | large | 3.785 | 4.552 | 0.815 | 47.4% |
| 1 Mar | anomaly · 500 trees · depth 5 · leaf 30 · 1.0 | large | 3.555 | 4.264 | 0.831 | 51.2% |
| 15 Mar | anomaly · 500 trees · depth 8 · leaf 10 · 0.7 (refined grid) | small | 3.046 | 3.691 | 0.881 | 58.5% |

Raw target vs anomaly target (notebook 01 vs 02, best of each): Mar15 3.136 vs 3.058,
Mar1 3.878 vs 3.555. Raw Feb1 was never run. A 30-year training window was worse than
using all training years in every raw-target run.

## Where the report's RF numbers come from

| Report item (Section 9.1.2 / 9.6) | File |
|---|---|
| Table "Best random forest for each issue date" | `results/rf/04_selected_models/selected_models_cv_summary.csv` |
| Figure "Random forest refined grid, 15 March" | `results/rf/04_selected_models/rf_tuning_mar15_small.png` |
| Figure "Predicted against actual full-bloom DOY" | `results/rf/04_selected_models/rf_mar15_predicted_vs_actual_by_region.png` |
| Table "MAE by region and by year" | `results/rf/04_selected_models/mar15_region_mae.csv`, `mar15_year_mae.csv` |
| Figure "Random forest permutation importance" | `results/rf/05_permutation_importance/individual_pi_summary.csv`, `group_pi_summary.csv` (drawn in `pi_top8.png`) |
| Figures "PSI" and "Mean temperature, 1–14 March" | `results/monitoring/` (from `src/monitoring/psi_drift.ipynb`) |
| MLflow runs `rf-*` | `src/models/log_models.py` reads `results/rf/02_anomaly_grid/best_models.csv` and `*_best_candidate_fold_results.csv` |

## Running a notebook

Open it from this folder (VS Code / Jupyter) with the project's Python environment
(`requirements.txt` plus `matplotlib`, `scipy`, `jupyter`) and run all cells. Paths are
found from the notebook's location, so no drive letters need editing.

To try 01 or 02 without overwriting the saved results, set `MAX_CANDIDATES = 2` and
`OUTPUT_OVERRIDE = '<some scratch folder>'` in the first code cell.

The clean-up reruns (Oct 2026) used Python 3.13, scikit-learn 1.9.1 and pandas 3.0 and
reproduced the original results: refined grid best MAE 3.0465, PI of the 1–14 March
temperature group 1.13 days, PSI of `tmean_mar1_14` 0.588.
