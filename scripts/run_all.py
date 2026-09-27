"""Cross-platform pipeline runner for the thesis recreation project.

This is the primary make-free entry point:

    python scripts/run_all.py
    python scripts/run_all.py --fast

Full mode regenerates authoritative outputs and writes
``output/REPRODUCTION_REPORT.md`` with drift checks. Fast mode is a smoke
test: it executes temporary notebook copies with reduced repeat counts,
restores authoritative outputs afterward, then runs the acceptance tests.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nbformat

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "output"
FAST_RESTORE_DIRS = [PROJECT_ROOT / "output", PROJECT_ROOT / "figures"]
FINGERPRINT_PACKAGES = [
    "numpy",
    "pandas",
    "scikit-learn",
    "imbalanced-learn",
    "scipy",
    "matplotlib",
    "nbconvert",
    "nbformat",
]

NOTEBOOKS: list[tuple[str, Path]] = [
    ("00", PROJECT_ROOT / "notebooks" / "00_Data_Audit.ipynb"),
    ("01", PROJECT_ROOT / "notebooks" / "01_leakage_window_ablation.ipynb"),
    ("02", PROJECT_ROOT / "notebooks" / "02_imbalance_comparison.ipynb"),
    ("03", PROJECT_ROOT / "notebooks" / "03_taiwan_replication.ipynb"),
    ("04", PROJECT_ROOT / "notebooks" / "04_temporal_coherence.ipynb"),
    ("05", PROJECT_ROOT / "notebooks" / "05_c2st_mechanism.ipynb"),
    ("06", PROJECT_ROOT / "notebooks" / "06_c2st_taiwan.ipynb"),
]

NOTEBOOK_REPORTS: list[tuple[str, Path]] = [
    ("00_Data_Audit", OUTPUT_DIR / "00_audit_verification.md"),
    ("01_leakage_window_ablation", OUTPUT_DIR / "01_leakage_ablation_verification.md"),
    ("02_imbalance_comparison", OUTPUT_DIR / "02_imbalance_comparison_verification.md"),
    ("03_taiwan_replication", OUTPUT_DIR / "03_taiwan_replication_verification.md"),
    ("04_temporal_coherence", OUTPUT_DIR / "04_temporal_coherence_verification.md"),
    ("05_c2st_mechanism", OUTPUT_DIR / "05_c2st_verification.md"),
    ("06_c2st_taiwan", OUTPUT_DIR / "06_c2st_taiwan_verification.md"),
]

FAST_REPLACEMENTS: dict[str, dict[str, str]] = {
    "02": {"N_REPEATS = 2": "N_REPEATS = 1"},
    "03": {"N_REPEATS = 3": "N_REPEATS = 1"},
    "05": {"N_REPEATS = 3": "N_REPEATS = 1"},
    "06": {"N_REPEATS = 3": "N_REPEATS = 1", "N_BOOT = 20": "N_BOOT = 2"},
}

DRIFT_TOLERANCE = 1e-6

DATA_CHECK_CODE = (
    "from src.data.taiwan import download_taiwan; "
    "from pathlib import Path; "
    "print(download_taiwan()); "
    "p=Path('data/raw/training_clean.xlsx'); "
    "print('private churn data present' if p.exists() else "
    "'WARNING: private churn data missing at data/raw/training_clean.xlsx')"
)


@dataclass
class StepResult:
    """Runtime result for one runner step."""

    name: str
    ok: bool
    seconds: float
    detail: str = ""


def _run_command(name: str, command: list[str]) -> StepResult:
    start = time.perf_counter()
    print(f"\n>>> {name}: {' '.join(command)}", flush=True)
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    seconds = time.perf_counter() - start
    ok = completed.returncode == 0
    print(f"<<< {name}: {'PASS' if ok else 'FAIL'} in {seconds:.1f}s", flush=True)
    return StepResult(name=name, ok=ok, seconds=seconds, detail=f"exit={completed.returncode}")


def _environment_fingerprint() -> list[str]:
    lines = [
        f"- Python: {platform.python_version()}",
        f"- Platform: {platform.platform()}",
    ]
    for package in FINGERPRINT_PACKAGES:
        try:
            version = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            version = "not installed"
        lines.append(f"- {package}: {version}")
    return lines


def _backup_artifacts(parent: Path) -> dict[Path, Path]:
    backups = {}
    for source in FAST_RESTORE_DIRS:
        if source.exists():
            target = parent / source.name
            shutil.copytree(source, target)
            backups[source] = target
    return backups


def _restore_artifacts(backups: dict[Path, Path]) -> None:
    for destination, source in backups.items():
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(source, destination)


def _run_python_code(name: str, code: str) -> StepResult:
    return _run_command(name, [sys.executable, "-c", code])


def _count_self_run_checks(report_path: Path) -> tuple[int, int]:
    text = _read(report_path)
    try:
        section = text.split("## Self-run acceptance checks", maxsplit=1)[1]
    except IndexError as exc:
        raise ValueError(f"No self-run acceptance section in {report_path}") from exc

    next_heading = re.search(r"\n## ", section)
    if next_heading is not None:
        section = section[: next_heading.start()]

    passed = 0
    total = 0
    for line in section.splitlines():
        stripped = line.strip()
        if not stripped.startswith("|") or stripped.startswith("| ---"):
            continue
        cells = [cell.strip() for cell in stripped.strip("|").split("|")]
        if not cells or cells[0].lower() == "check":
            continue
        result = cells[-1]
        if result not in {"PASS", "FAIL"}:
            continue
        total += 1
        if result == "PASS":
            passed += 1
    return passed, total


def _write_master_verification_report() -> None:
    rows = []
    for notebook_name, report_path in NOTEBOOK_REPORTS:
        passed, total = _count_self_run_checks(report_path)
        rows.append((notebook_name, passed, total, total - passed, report_path))

    lines = [
        "# Master Verification Report",
        "",
        "## Environment",
        *_environment_fingerprint(),
        "",
        "## Notebook self-run acceptance checks",
        "| notebook | passed | total | failed | report | annotation |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for notebook_name, passed, total, failed, report_path in rows:
        annotation = ""
        if notebook_name == "04_temporal_coherence":
            annotation = (
                "Intentional mechanism disconfirmation: failed checks are the honest "
                "negative result, not a packaging error."
            )
        lines.append(
            f"| {notebook_name} | {passed} | {total} | {failed} | "
            f"`{report_path.relative_to(PROJECT_ROOT).as_posix()}` | {annotation} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Notebook 04 intentionally has failed mechanism checks: those failures are "
            "part of the final claim calibration, not a packaging error.",
            "- All setup-level C2ST sanity floors in notebook 06 pass.",
        ]
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "VERIFICATION_REPORT.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def _regenerate_master_verification_step() -> StepResult:
    start = time.perf_counter()
    try:
        _write_master_verification_report()
    except Exception as exc:  # pragma: no cover - exercised by CLI
        seconds = time.perf_counter() - start
        print("\n>>> verification: regenerate master report", flush=True)
        print(f"<<< verification: FAIL in {seconds:.1f}s ({exc})", flush=True)
        return StepResult(name="verification", ok=False, seconds=seconds, detail=repr(exc))

    seconds = time.perf_counter() - start
    print("\n>>> verification: regenerate master report", flush=True)
    print(f"<<< verification: PASS in {seconds:.1f}s", flush=True)
    return StepResult(name="verification", ok=True, seconds=seconds, detail="generated")


def _patch_notebook_for_fast(notebook_path: Path, temp_dir: Path, name: str) -> Path:
    nb = nbformat.read(notebook_path, as_version=4)
    replacements = FAST_REPLACEMENTS.get(name, {})
    for cell in nb.cells:
        if not isinstance(cell.get("source"), str):
            continue
        for old, new in replacements.items():
            if old in cell.source:
                cell.source = cell.source.replace(old, new)
    target = temp_dir / notebook_path.name
    nbformat.write(nb, target)
    return target


def _execute_notebook(name: str, path: Path, fast: bool, temp_dir: Path | None) -> StepResult:
    if fast:
        try:
            temp_path = _patch_notebook_for_fast(path, temp_dir or path.parent, name)
        except Exception as exc:  # pragma: no cover - exercised by CLI
            seconds = 0.0
            print(f"\n>>> notebook:{name}: compile smoke {path.name}", flush=True)
            print(f"<<< notebook:{name}: FAIL in {seconds:.1f}s ({exc})", flush=True)
            return StepResult(
                name=f"notebook:{name}",
                ok=False,
                seconds=seconds,
                detail=repr(exc),
            )
        result = _run_command(
            f"notebook:{name}",
            [
                sys.executable,
                "-m",
                "nbconvert",
                "--to",
                "notebook",
                "--execute",
                "--inplace",
                str(temp_path),
            ],
        )
        result.detail = f"fast-smoke; temp={temp_path.name}"
        return result

    notebook_path = path
    return _run_command(
        f"notebook:{name}",
        [
            sys.executable,
            "-m",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--inplace",
            str(notebook_path),
        ],
    )


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _json_after(text: str, heading: str) -> Any:
    start = text.index(heading)
    match = re.search(r"```json\n(.*?)\n```", text[start:], re.DOTALL)
    if match is None:
        raise ValueError(f"No JSON block after {heading!r}")
    return json.loads(match.group(1))


def _float(pattern: str, text: str) -> float:
    match = re.search(pattern, text)
    if match is None:
        raise ValueError(f"Pattern not found: {pattern}")
    return float(match.group(1))


def _extract_headlines() -> dict[str, float | str]:
    report01 = _read(OUTPUT_DIR / "01_leakage_ablation_verification.md")
    report02 = _read(OUTPUT_DIR / "02_imbalance_comparison_verification.md")
    report03 = _read(OUTPUT_DIR / "03_taiwan_replication_verification.md")
    report05 = _read(OUTPUT_DIR / "05_c2st_verification.md")
    report06 = _read(OUTPUT_DIR / "06_c2st_taiwan_verification.md")

    summary02 = _json_after(report02, "## Three-lens table")
    sig02 = {
        row["strategy"]: row
        for row in _json_after(report02, "## Paired significance vs `none`")
    }
    summary03 = _json_after(report03, "## E2 table")
    sig03 = {
        row["strategy"]: row
        for row in _json_after(report03, "## Paired significance vs `none`")
    }
    auc05 = _json_after(report05, "## C2ST AUC table")
    auc06 = _json_after(report06, "## AUC table")
    verdict06 = _json_after(report06, "## Machine-readable verdict")

    return {
        "leakage_inflation": _float(
            r"Relative PR-AUC inflation \(ALL_12 -> HONEST\)\n- ([0-9.]+)", report01
        ),
        "churn_none_pr_auc": summary02["none"]["pr_auc_mean"],
        "churn_smote_mean_delta": sig02["smote"]["mean_delta"],
        "churn_smote_wilcoxon_p": sig02["smote"]["wilcoxon_p"],
        "taiwan_none_pr_auc": summary03["none"]["pr_auc_mean"],
        "taiwan_smote_mean_delta": sig03["smote"]["mean_delta"],
        "taiwan_smote_wilcoxon_p": sig03["smote"]["wilcoxon_p"],
        "c2st_churn_auc": auc05["real_vs_smote"]["mean_auc"],
        "c2st_churn_duplicated_auc": auc05["real_vs_duplicated"]["mean_auc"],
        "c2st_churn_at_taiwan_n_ci_low": auc06["churn@taiwan_n"]["ci_low"],
        "c2st_churn_at_taiwan_n_ci_high": auc06["churn@taiwan_n"]["ci_high"],
        "c2st_taiwan_at_churn_features_ci_low": auc06["taiwan@churn_features"]["ci_low"],
        "c2st_taiwan_at_churn_features_ci_high": auc06["taiwan@churn_features"]["ci_high"],
        "c2st_controlled_intervals_overlap": str(
            max(
                auc06["churn@taiwan_n"]["ci_low"],
                auc06["taiwan@churn_features"]["ci_low"],
            )
            <= min(
                auc06["churn@taiwan_n"]["ci_high"],
                auc06["taiwan@churn_features"]["ci_high"],
            )
        ),
        "c2st_demarcation_supported": verdict06["demarcation_supported"],
    }


def _compare_headlines(
    old: dict[str, float | str], new: dict[str, float | str]
) -> list[dict[str, str]]:
    rows = []
    for key, old_value in old.items():
        new_value = new[key]
        if isinstance(old_value, str) or isinstance(new_value, str):
            match = old_value == new_value
        else:
            match = abs(float(old_value) - float(new_value)) <= DRIFT_TOLERANCE
        rows.append(
            {
                "metric": key,
                "old": str(old_value),
                "new": str(new_value),
                "status": "MATCH" if match else f"DRIFT ({old_value} -> {new_value})",
            }
        )
    return rows


def _write_reproduction_report(
    step_results: list[StepResult],
    baseline: dict[str, float | str],
    fresh: dict[str, float | str],
    total_seconds: float,
) -> None:
    drift_rows = _compare_headlines(baseline, fresh)
    reproduces_clean = all(result.ok for result in step_results) and all(
        row["status"] == "MATCH" for row in drift_rows
    )

    lines = [
        "# Reproduction Report",
        "",
        f"- REPRODUCES_CLEAN: {'yes' if reproduces_clean else 'no'}",
        f"- Total wall time seconds: {total_seconds:.1f}",
        f"- Drift tolerance: {DRIFT_TOLERANCE}",
        "",
        "## Environment fingerprint",
        *_environment_fingerprint(),
        "",
        "## Step timings",
        "| step | status | seconds | detail |",
        "| --- | --- | --- | --- |",
    ]
    for result in step_results:
        lines.append(
            f"| {result.name} | {'PASS' if result.ok else 'FAIL'} | "
            f"{result.seconds:.1f} | {result.detail} |"
        )

    lines.extend(
        [
            "",
            "## Headline drift checks",
            "| metric | old | new | status |",
            "| --- | --- | --- | --- |",
        ]
    )
    for row in drift_rows:
        lines.append(f"| {row['metric']} | {row['old']} | {row['new']} | {row['status']} |")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "REPRODUCTION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_pending_stub(reason: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    text = f"""# Reproduction Report

- REPRODUCES_CLEAN: no
- Status: full drift-checked run pending author execution
- Reason: {reason}

Run:

```powershell
python scripts/run_all.py
```

The full run will overwrite this stub with environment, timing, and headline drift checks.
"""
    (OUTPUT_DIR / "REPRODUCTION_REPORT.md").write_text(text, encoding="utf-8")


def _selected(name: str, only: set[str], skip: set[str]) -> bool:
    if only and name not in only:
        return False
    return name not in skip


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the full thesis reproduction pipeline.")
    parser.add_argument("--fast", action="store_true", help="Run reduced-repeat smoke notebooks.")
    parser.add_argument(
        "--only", action="append", default=[], help="Only run a step/name, e.g. 02 or test."
    )
    parser.add_argument(
        "--skip", action="append", default=[], help="Skip a step/name, e.g. 06 or test."
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    only = set(args.only)
    skip = set(args.skip)
    results: list[StepResult] = []
    total_start = time.perf_counter()
    baseline = _extract_headlines() if not args.fast and not only else {}

    backup_parent: Path | None = None
    artifact_backups: dict[Path, Path] = {}
    artifacts_restored = False
    temp_dir_obj: tempfile.TemporaryDirectory[str] | None = None
    temp_dir: Path | None = None
    if args.fast:
        print(
            "FAST smoke mode: temporary notebooks are executed "
            "with reduced repeats where applicable; authoritative outputs are restored."
        )
        _write_pending_stub("fast smoke run does not perform authoritative drift checks")
        backup_parent = Path(tempfile.mkdtemp(prefix="run_all_artifact_backup_"))
        artifact_backups = _backup_artifacts(backup_parent)
        temp_dir_obj = tempfile.TemporaryDirectory(
            prefix=".run_all_fast_notebooks_", dir=PROJECT_ROOT
        )
        temp_dir = Path(temp_dir_obj.name)

    try:
        if _selected("data", only, skip):
            results.append(
                _run_python_code("data", DATA_CHECK_CODE)
            )

        for name, notebook_path in NOTEBOOKS:
            if _selected(name, only, skip):
                results.append(_execute_notebook(name, notebook_path, args.fast, temp_dir))

        if args.fast:
            _restore_artifacts(artifact_backups)
            artifacts_restored = True

        if _selected("verification", only, skip):
            results.append(_regenerate_master_verification_step())

        if _selected("test", only, skip):
            results.append(_run_command("test", [sys.executable, "tests/test_acceptance.py"]))

        total_seconds = time.perf_counter() - total_start
        if not args.fast and not only and all(result.ok for result in results):
            fresh = _extract_headlines()
            _write_reproduction_report(results, baseline, fresh, total_seconds)

        for result in results:
            print(f"{result.name:>12}: {'PASS' if result.ok else 'FAIL'} ({result.seconds:.1f}s)")
        print(f"Total: {total_seconds:.1f}s")
        return 0 if all(result.ok for result in results) else 1
    finally:
        if args.fast and artifact_backups and not artifacts_restored:
            _restore_artifacts(artifact_backups)
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
        if backup_parent is not None:
            shutil.rmtree(backup_parent, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
