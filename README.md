# Thesis Recreation 2026

I re-audited my own earlier churn-prediction theses and turned it into a reproducible study of class-imbalance evaluation: private telecom panel data compared against a public Taiwan bankruptcy benchmark, looking hard at leakage, honest resampling, ranking metrics, calibration, and the mechanism behind the results (including where that investigation didn't pan out). The full narrative is in [`paper/paper.md`](paper/paper.md); this README covers reproduction.

## What I found

- Contaminated churn windows and pre-split resampling substantially inflate PR-AUC.
- SMOTE-family oversampling did not improve ranking: it was significantly harmful on the churn panel and neutral on Taiwan.
- Fixed-threshold recall gains from resampling/class weighting are threshold shifts and can degrade calibration.
- The mechanism remains open: temporal jaggedness was disconfirmed, and controlled C2ST detectability did not explain the churn-vs-Taiwan demarcation.

## Data Privacy

The telecom churn dataset is proprietary and is not shared. Place it locally at `data/raw/training_clean.xlsx` to reproduce private-data notebooks. The Taiwan bankruptcy dataset is public and is downloaded by `python scripts/run_all.py` (or `python -m src.data.taiwan`).

## Reproduction

```powershell
python -m pip install -r requirements.txt
python scripts/run_all.py
```

`python scripts/run_all.py` is the primary cross-platform entry point. It downloads/checks data, executes notebooks `00` through `06`, runs `tests/test_acceptance.py`, and writes a local reproduction report with headline drift checks. GNU Make is optional only:

```powershell
make all
make test
```

For quick smoke checks:

```powershell
python scripts/run_all.py --fast
```

`--fast` executes temporary notebook copies with reduced repeat constants where present, then restores the authoritative reports before running acceptance tests. It is a smoke run, not a source for paper numbers.

Measured on the author's Windows/Conda environment (`Windows 11`, `Python 3.12.3`, 8 logical processors, packages pinned in `requirements.txt`), representative notebook runtimes are:

| notebook | full-run expectation | `--fast` smoke measured |
| --- | ---: | ---: |
| `00_Data_Audit.ipynb` | ~1.7 min | 101.9 s |
| `01_leakage_window_ablation.ipynb` | ~2.3 min | 140.0 s |
| `02_imbalance_comparison.ipynb` | ~15.4 min; this is the long pole | 527.0 s |
| `03_taiwan_replication.ipynb` | ~7.8 min | 246.5 s |
| `04_temporal_coherence.ipynb` | ~8.3 min | 498.3 s |
| `05_c2st_mechanism.ipynb` | ~2.6 min | 113.9 s |
| `06_c2st_taiwan.ipynb` | ~9.6 min | 132.5 s |

The measured `--fast` smoke run completed end-to-end in 29.4 minutes, including data check and acceptance tests. Expect the full authoritative run to take roughly 50 minutes or more. The exact per-step timing is recorded locally by `scripts/run_all.py` after a full run.

## Repo Map

- `notebooks/`: experiment orchestration and reports.
- `src/`: reusable data, feature, model, evaluation, and visualization code.
- `figures/`: generated figures.
- `data/`: local datasets; private raw churn data are ignored, and local generated reports are ignored by git.
- [`paper/paper.md`](paper/paper.md): the write-up.
- [`RESULTS.md`](RESULTS.md): per-notebook results with the exact numbers behind each claim.
