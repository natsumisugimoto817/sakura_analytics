# Monitoring

`psi_drift.ipynb` checks whether the inputs of the deployed 15 March SVR have shifted
between the training years (1964–2014) and the test years (2015–2025), using the
Population Stability Index (PSI) for each of its 28 features
(< 0.10 stable, 0.10–0.25 moderate, ≥ 0.25 large shift).

- Features: read from `app/model/svr_mar15_meta.json`, so the check always follows the deployed model.
- Records: the 48-station small sample, so both periods cover the same stations.
- Output: [`results/monitoring/`](../../results/monitoring/)
  - `mar15_small_train_test_drift.csv` – PSI, both KL directions, means and mean change per feature
  - `mar15_small_train_test_bin_details.csv` – every bin's train/test share and PSI contribution
  - `station_composition_train_test.csv` – each station's share of records in the two periods
  - `feature_psi.png`, `selected_feature_distributions.png`, `monitoring_overview.png` – figures
- Runtime: a few seconds.

Result (report Section 9.6): 11 features shifted strongly, led by the 1–14 March
temperature measures (PSI 0.43–0.59); mean temperature over 1–14 March was 2.1 °C
higher in the test years.
