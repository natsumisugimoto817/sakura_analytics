# Random-forest results

Made by the notebooks in [`src/models/random_forest/`](../../src/models/random_forest/)
(the folder number = the notebook number). All scores are 5-fold cross-validation
scores on the 2005–2014 validation seasons, in days of full-bloom DOY.
CSV files are UTF-8 with a BOM, so Excel opens them correctly.

File names: `{date}_{sample}_...` with date `feb1` / `mar1` / `mar15` and sample
`large` (82 stations) or `small` (48 stations with a 2025 record).

## `01_raw_target/` – first experiment, raw DOY target (record only)

| File | Content |
|---|---|
| `best_models.csv` | best candidate of each completed condition (7 rows) |
| `{date}_{sample}_train{all,30y}_grid_search_all_candidates.csv` | 96 candidates, mean CV scores, best first |
| `..._all_candidates_fold_results.csv` | every candidate × fold |
| `..._best_candidate_fold_results.csv` | the best candidate's five folds |

`trainall` = all training years of the fold, `train30y` = only the last 30.
Not run: Mar1-small-30y and all Feb1 conditions.

## `02_anomaly_grid/` – main grid, anomaly target

| File | Content |
|---|---|
| `best_models.csv` | best candidate per dataset (6 rows). Read by `src/models/log_models.py` |
| `{date}_{sample}_grid_search_all_candidates.csv` | 54 candidates, mean CV scores, best first |
| `{date}_{sample}_all_candidates_fold_results.csv` | every candidate × fold |
| `{date}_{sample}_best_candidate_fold_results.csv` | the best candidate's five folds |

## `03_mar15_refined/` – 15 March, small sample, finer grid

`mar15_small_refined_` + `grid_search_all_candidates.csv` (27 candidates),
`all_candidates_fold_results.csv`, `best_candidate_fold_results.csv`,
`best_candidate_oof_predictions.csv` (validation predictions of the best candidate) and
`config.json` (features, grid, station list).

## `04_selected_models/` – selected model per issue date, report tables and figures

| File | Content |
|---|---|
| `selected_models.csv` | the chosen model per issue date and which grid it came from |
| `reproduction_check.csv` | rebuilt vs saved CV scores (must agree within 0.01 days) |
| `{date}_{sample}_oof_predictions.csv` | validation predictions of each selected model |
| `selected_models_cv_summary.csv` | report table "Best random forest for each issue date" |
| `mar15_small_combined_grid.csv`, `rf_tuning_mar15_small.png` | report figure "Random forest refined grid" |
| `rf_mar15_predicted_vs_actual_by_region.png` | report figure "Predicted against actual" |
| `rf_mar15_predicted_vs_actual_by_year.png` | the same, coloured by year |
| `mar15_region_mae.csv`, `mar15_year_mae.csv` | report table "MAE by region and by year" (pooled records) |
| `mar15_region_fold_mae.csv` | MAE by region × fold |
| `mar15_station_mae.csv`, `mar15_hokkaido_year_mae.csv` | errors by station; Hokkaido by year |

## `05_permutation_importance/` – best 15 March RF

| File | Content |
|---|---|
| `group_membership.csv`, `config.json` | the 19 feature groups (absolute Spearman ρ ≥ 0.8 on the fold 1 training years) |
| `baseline_mae_by_fold.csv` | validation MAE of each refitted fold model |
| `individual_pi_summary.csv`, `group_pi_summary.csv` | mean PI over folds and SD across folds (days of MAE) |
| `*_by_fold.csv`, `*_all_repeats.csv`, `group_pi_fold_comparison.csv` | the same by fold and by shuffle |
| `pi_5fold_comparison.png`, `pi_top8.png` | figures |
