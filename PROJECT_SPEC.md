# Project Spec - Honest Evaluation of Class-Imbalance Handling

## Mission

A reproducible ML research study showing that common class-imbalance practices can be misleading, and that honest evaluation changes the conclusions. Origin: a re-examination of the author's 2016 undergraduate and 2020 master's theses (churn prediction on the same telecom dataset), audited with 2026 standards for temporal leakage, resampling-before-split, ranking metrics, calibration, and reproducibility.

## Final Claim Set

1. Contaminated churn-window features (`N`, `N-1`) and pre-split resampling both inflate measured PR-AUC.
2. SMOTE/ADASYN/SMOTENC did not improve ranking: significantly harmful on the private churn panel, neutral on the public Taiwan bankruptcy benchmark.
3. Resampling and class weighting change fixed-threshold recall/precision behavior and can degrade calibration. These are threshold shifts, not ranking improvements, so ranking and calibration metrics must be reported explicitly.
4. The temporal-incoherence/jaggedness hypothesis was tested and not supported: roughness moved the wrong way, monotonicity did not separate the groups, and per-signal coherence was inconclusive. C2ST shows SMOTE synthetics are detectable/off-manifold, but controlled churn-vs-Taiwan C2ST comparisons overlap, so off-manifold detectability alone does not explain the harmful-vs-neutral demarcation. The mechanism remains open.

## Evaluation Standard

Report PR-AUC, ROC-AUC, recall-at-budget, Brier score, and fixed-threshold operating metrics where relevant. Never headline accuracy. Fit all resampling and preprocessing inside the training fold only. Use repeated stratified folds and paired significance tests for strategy comparisons.

## Datasets

- **Telecom churn (private case study, not committed):** local file `data/raw/training_clean.xlsx`, panel structure (signals TAG/STATUS_BAYAR/GGN/USAGE across 12 months plus static fields), about 3.8% churn. Data are git-ignored and never shared.
- **Taiwanese Bankruptcy (public, cross-sectional):** downloaded at build time from the public CSV used in `src/data/taiwan.py`, about 3.2% positive. This is the public reproducibility anchor.

## Deliverables

Reproducible notebooks, reusable `src/` modules, generated figures, per-notebook verification reports, a master verification report, a publishable results narrative, and a paper draft/skeleton.

## Non-goals

This is not a leaderboard chase and not a claim that SMOTE is universally bad. The contribution is methodological: leakage-aware evaluation, calibrated imbalance-method comparisons, a two-dataset harmful-vs-neutral finding, and transparent reporting of mechanism probes that leave the explanation open.
