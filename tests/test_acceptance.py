"""Acceptance checks for regenerated verification reports.

Run with ``python tests/test_acceptance.py`` after ``make all``.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output"


def _read(path: str) -> str:
    return (OUTPUT / path).read_text(encoding="utf-8")


def _json_after(text: str, heading: str) -> object:
    start = text.index(heading)
    match = re.search(r"```json\n(.*?)\n```", text[start:], re.DOTALL)
    if match is None:
        raise AssertionError(f"No JSON block after {heading!r}")
    return json.loads(match.group(1))


def _float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise AssertionError(f"Pattern not found: {pattern}")
    return float(match.group(1))


def check_leakage_inflation() -> None:
    text = _read("01_leakage_ablation_verification.md")
    inflation = _float(r"Relative PR-AUC inflation \(ALL_12 -> HONEST\)\n- ([0-9.]+)", text)
    assert inflation >= 0.45, inflation


def check_churn_ranking_harm() -> None:
    text = _read("02_imbalance_comparison_verification.md")
    summary = _json_after(text, "## Three-lens table")
    sig = {row["strategy"]: row for row in _json_after(text, "## Paired significance vs `none`")}
    none_pr = summary["none"]["pr_auc_mean"]
    smote_pr = summary["smote"]["pr_auc_mean"]
    assert 0.35 <= none_pr <= 0.39, none_pr
    assert smote_pr < none_pr, (smote_pr, none_pr)
    assert sig["smote"]["mean_delta"] < 0, sig["smote"]
    assert sig["smote"]["wilcoxon_p"] < 0.05, sig["smote"]


def check_taiwan_neutral() -> None:
    text = _read("03_taiwan_replication_verification.md")
    summary = _json_after(text, "## E2 table")
    sig = {row["strategy"]: row for row in _json_after(text, "## Paired significance vs `none`")}
    none_pr = summary["none"]["pr_auc_mean"]
    smote_pr = summary["smote"]["pr_auc_mean"]
    assert abs(smote_pr - none_pr) <= 0.03, (smote_pr, none_pr)
    assert sig["smote"]["wilcoxon_p"] > 0.05, sig["smote"]


def check_c2st_sanity_floor() -> None:
    text = _read("05_c2st_verification.md")
    auc = _json_after(text, "## C2ST AUC table")
    duplicated = auc["real_vs_duplicated"]["mean_auc"]
    assert 0.47 <= duplicated <= 0.53, duplicated


def main() -> None:
    checks = [
        check_leakage_inflation,
        check_churn_ranking_harm,
        check_taiwan_neutral,
        check_c2st_sanity_floor,
    ]
    for check in checks:
        check()
        print(f"PASS {check.__name__}")


if __name__ == "__main__":
    main()
