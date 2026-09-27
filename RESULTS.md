# Results

This repository re-audits a private telecom churn panel and compares it with a public, cross-sectional Taiwan bankruptcy benchmark. The final claim is calibrated: leakage and resampling protocol problems are established; SMOTE-family methods did not improve ranking in either dataset; the churn-vs-Taiwan harmful/neutral difference is real in these two datasets; the mechanism remains open.

All figures and numbers regenerate from the pipeline (`python scripts/run_all.py`).

## 1. Data audit

Notebook 00 establishes the honest churn modelling window. The private churn panel has 99,807 rows, 55 features, and a positive rate of 0.038114. The month `N` leakage signal is visible in USAGE zeros: churn rows have `USAGE_N == 0` at 0.577550, while non-churn rows are 0.030374. The audit also flags global artifact months `N-5` and `N-2`, which are reported rather than hidden.

Figures:
- `figures/data_audit/usage_zero_fraction.png`
- `figures/data_audit/usage_median_trajectory.png`

## 2. Leakage inflation

Notebook 01 shows that contaminated temporal windows inflate measured performance. The fully leaky `ALL_12` window reaches PR-AUC 0.829622, while the honest `N-11..N-2` window reaches PR-AUC 0.375297. The relative PR-AUC inflation from `ALL_12` to `HONEST` is 0.547629. The leaky model's top permutation features include `USAGE_N`, `TAG_N-1`, `STATUS_BAYAR_N`, and `USAGE_N-1`, matching the audit's leakage warning.

## 3. Honest imbalance-strategy comparison

Notebook 02 evaluates imbalance handling on the churn panel with resampling fit inside the training fold only. The honest baseline (`none`) has PR-AUC 0.369116. `smote` falls to 0.289217, `adasyn` to 0.293157, and `smotenc` to 0.262593. The paired PR-AUC deltas versus `none` are negative and significant: SMOTE mean delta -0.079899 with Wilcoxon p=0.001953; ADASYN mean delta -0.075959 with Wilcoxon p=0.001953; SMOTENC mean delta -0.106523 with Wilcoxon p=0.001953.

The same notebook shows the operating-point and calibration story. Class weighting raises fixed-threshold recall at 0.5 from 0.205574 (`none`) to 0.622765, but Brier score degrades from 0.028810 to 0.139821. That is a threshold/calibration tradeoff, not an improvement in ranking.

Figure:
- `figures/imbalance_comparison/delta_pr_auc_vs_none.png`

## 4. Public Taiwan replication

Notebook 03 repeats the protocol on the public Taiwan bankruptcy dataset: 6,819 rows, 95 features, positive rate 0.032263. Pre-split SMOTE is again leaky: leaky PR-AUC is 0.999577 versus honest PR-AUC 0.484579, inflation 0.514998.

For honest ranking, Taiwan is neutral rather than harmful. The `none` baseline has PR-AUC 0.498675; `smote` has 0.489610, mean delta -0.009065, Wilcoxon p=0.276855; `adasyn` has 0.491410, mean delta -0.007265, Wilcoxon p=0.276855. The churn-vs-Taiwan demarcation therefore rests on two datasets and should not be overgeneralized.

Figure:
- `figures/taiwan_replication/cross_dataset_delta_pr_auc.png`

## 5. Mechanism investigation

Notebook 04 tests the original temporal-coherence/jaggedness hypothesis and mostly disconfirms it. Roughness moved the wrong way: SMOTE-synthetic trajectories were smoother, because interpolation averages curves. Autocorrelation medians differed in direction (`real` 0.027599, `smote_synthetic` 0.005001), but the 95% CIs overlapped, KS SMOTE-vs-real was not significant (p=0.192389), and size-matched tests detected the difference in only 0.015 of autocorrelation repeats. Roughness is therefore reported as a dead end, not evidence.

Notebook 05 asks a broader C2ST question on churn: are complete SMOTE-synthetic minority rows distinguishable from real minority rows? Yes. Churn real-vs-SMOTE C2ST AUC is 0.860849 with 95% CI [0.857158, 0.864540], while the exact-copy duplicated sanity floor is 0.500000. Top features include `TAG_N-2`, `TAG_N-7`, and `STATUS_BAYAR_N-2`.

Notebook 06 asks whether that detectability explains the churn-vs-Taiwan demarcation. It does not cleanly do so. Taiwan full C2ST AUC is 0.725551, but naive comparison is confounded by feature count and sample size. After controls, churn at Taiwan's minority count has mean AUC 0.618255 with empirical interval [0.549121, 0.673185], while Taiwan at churn's 40-feature count has mean AUC 0.611051 with empirical interval [0.507384, 0.694487]. The intervals overlap, so `demarcation_supported` is `inconclusive`.

Figures:
- `figures/temporal_coherence/autocorrelation_distributions.png`
- `figures/temporal_coherence/monotonicity_distributions.png`
- `figures/temporal_coherence/illustrative_trajectories.png`

## What holds

- Leakage from contaminated temporal windows and pre-split resampling can massively inflate PR-AUC.
- SMOTE-family oversampling did not improve ranking in these experiments.
- On the churn panel, synthetic oversampling significantly harmed ranking.
- On Taiwan, SMOTE/ADASYN were neutral for ranking.
- Fixed-threshold recall gains must be interpreted with ranking and calibration metrics.

## What remains open

The mechanism behind the churn-vs-Taiwan demarcation is not established. Temporal structure remains a plausible hypothesis, but the tested jaggedness/coherence proxy failed. C2ST detectability shows SMOTE creates distinguishable points, but controlled churn-vs-Taiwan detectability is comparable. Explaining when off-manifold synthetic points harm ranking remains future work.
