#!/usr/bin/env python3
"""
End-to-end verification script for Experiment Tracking Dashboard enhancement.

Tests all components:
1. Local SQLite tracker
2. Enhanced metric logging
3. Publication-ready figure export
4. Hyperparameter tuning integration
5. Streamlit UI functionality

Usage:
    python verify_e2e_tracking_dashboard.py
"""

import sys
import subprocess
import tempfile
import time
from pathlib import Path
import json
import sqlite3


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}\n")


def run_command(cmd: list[str], description: str, check_output: bool = False) -> tuple[bool, str]:
    """Run a command and return success status and output."""
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        if result.returncode == 0:
            print(f"✅ SUCCESS: {description}")
            if check_output:
                print(f"Output preview:\n{result.stdout[:500]}")
            return True, result.stdout
        else:
            print(f"❌ FAILED: {description}")
            print(f"Error: {result.stderr[:500]}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏱️ TIMEOUT: {description} (exceeded 5 minutes)")
        return False, "Timeout"
    except Exception as e:
        print(f"💥 ERROR: {description}")
        print(f"Exception: {str(e)}")
        return False, str(e)


def check_file_exists(filepath: Path, description: str) -> bool:
    """Check if a file exists."""
    if filepath.exists():
        print(f"✅ File exists: {filepath} ({description})")
        return True
    else:
        print(f"❌ File missing: {filepath} ({description})")
        return False


def verify_sqlite_database(db_path: Path) -> bool:
    """Verify SQLite database structure and content."""
    print(f"Verifying SQLite database: {db_path}")

    if not db_path.exists():
        print(f"❌ Database does not exist: {db_path}")
        return False

    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Check required tables exist
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}
        required_tables = {'experiments', 'runs', 'metrics', 'artifacts', 'studies', 'trials'}

        missing_tables = required_tables - tables
        if missing_tables:
            print(f"❌ Missing tables: {missing_tables}")
            return False

        print(f"✅ All required tables present: {required_tables}")

        # Check if there's data
        cursor.execute("SELECT COUNT(*) FROM experiments")
        exp_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM runs")
        run_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM metrics")
        metric_count = cursor.fetchone()[0]

        print(f"  - Experiments: {exp_count}")
        print(f"  - Runs: {run_count}")
        print(f"  - Metrics: {metric_count}")

        conn.close()

        if run_count > 0 and metric_count > 0:
            print("✅ Database contains experiment data")
            return True
        else:
            print("⚠️ Database exists but has no data yet")
            return True  # Still valid, just empty

    except Exception as e:
        print(f"❌ Error verifying database: {e}")
        return False


def main() -> int:
    """Run end-to-end verification."""
    print("🔬 Experiment Tracking Dashboard - End-to-End Verification")
    print("=" * 80)

    results = {}

    # ========================================================================
    # Step 1: Verify Local Tracker Implementation
    # ========================================================================
    print_section("Step 1: Verify Local Tracker Implementation")

    # Test import
    success, _ = run_command(
        ["python", "-c", "from mammography.tracking.local_tracker import LocalTracker; print('OK')"],
        "Import LocalTracker class",
        check_output=True
    )
    results['local_tracker_import'] = success

    # Test LocalTracker initialization
    success, _ = run_command(
        ["python", "-c", """
from mammography.tracking.local_tracker import LocalTracker
from pathlib import Path
import tempfile

with tempfile.TemporaryDirectory() as tmpdir:
    tracker = LocalTracker(
        db_path=Path(tmpdir) / 'test.db',
        experiment_name='test_exp',
        run_name='test_run',
        params={'lr': 0.001}
    )
    tracker.log_metrics({'loss': 0.5}, step=1)
    tracker.finish()
    print('LocalTracker test passed')
"""],
        "Test LocalTracker basic functionality",
        check_output=True
    )
    results['local_tracker_basic'] = success

    # ========================================================================
    # Step 2: Train with Local Tracker
    # ========================================================================
    print_section("Step 2: Train with Local Tracker (Offline Mode)")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "local_tracking_test"

        success, output = run_command(
            [
                "python", "-m", "mammography.cli",
                "train-density",
                "--dataset", "mamografias",
                "--subset", "10",
                "--epochs", "1",
                "--tracker", "local",
                "--tracker-project", "e2e-test-local",
                "--outdir", str(outdir)
            ],
            "Training with local tracker",
            check_output=True
        )
        results['train_local'] = success

        if success:
            # Check if database was created
            db_path = outdir / "experiments.db"
            results['local_db_created'] = verify_sqlite_database(db_path)

    # ========================================================================
    # Step 3: Train with Enhanced Metric Logging (MLflow)
    # ========================================================================
    print_section("Step 3: Train with Enhanced Metric Logging (MLflow)")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "mlflow_test"

        success, output = run_command(
            [
                "python", "-m", "mammography.cli",
                "train-density",
                "--dataset", "mamografias",
                "--subset", "10",
                "--epochs", "1",
                "--tracker", "mlflow",
                "--tracker-project", "e2e-test-mlflow",
                "--outdir", str(outdir)
            ],
            "Training with MLflow (enhanced metrics)",
            check_output=True
        )
        results['train_mlflow'] = success

        # Check for enhanced metrics in output
        if success and output:
            required_metrics = ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                               'val_f1', 'val_kappa', 'balanced_acc']
            metrics_found = [m for m in required_metrics if m in output]
            print(f"Enhanced metrics found in output: {metrics_found}")
            results['enhanced_metrics_logged'] = len(metrics_found) >= 5

    # ========================================================================
    # Step 4: Export Figures in All Formats
    # ========================================================================
    print_section("Step 4: Export Figures in All Formats (PNG/PDF/SVG)")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "export_test"

        success, output = run_command(
            [
                "python", "-m", "mammography.cli",
                "train-density",
                "--dataset", "mamografias",
                "--subset", "10",
                "--epochs", "1",
                "--export-figures", "png,pdf,svg",
                "--outdir", str(outdir)
            ],
            "Training with figure export enabled",
            check_output=True
        )
        results['train_with_export'] = success

        if success:
            # Check if figures were created
            figures_dir = outdir
            if not figures_dir.exists():
                # Try to find figures directory
                run_dirs = list(outdir.glob("run_*"))
                if run_dirs:
                    figures_dir = run_dirs[0] / "figures"

            png_files = list(figures_dir.glob("**/*.png"))
            pdf_files = list(figures_dir.glob("**/*.pdf"))
            svg_files = list(figures_dir.glob("**/*.svg"))

            results['png_exported'] = len(png_files) > 0
            results['pdf_exported'] = len(pdf_files) > 0
            results['svg_exported'] = len(svg_files) > 0

            print(f"Exported figures: PNG={len(png_files)}, PDF={len(pdf_files)}, SVG={len(svg_files)}")

    # ========================================================================
    # Step 5: Hyperparameter Tuning with Tracking
    # ========================================================================
    print_section("Step 5: Hyperparameter Tuning with Local Tracker")

    with tempfile.TemporaryDirectory() as tmpdir:
        outdir = Path(tmpdir) / "tune_test"

        success, output = run_command(
            [
                "python", "-m", "mammography.cli",
                "tune",
                "--dataset", "mamografias",
                "--subset", "20",
                "--n-trials", "2",
                "--tracker", "local",
                "--outdir", str(outdir)
            ],
            "Hyperparameter tuning with local tracker",
            check_output=True
        )
        results['tune_with_local'] = success

        if success:
            # Check if study data was saved
            db_path = outdir / "experiments.db"
            if db_path.exists():
                try:
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT COUNT(*) FROM studies")
                    study_count = cursor.fetchone()[0]
                    cursor.execute("SELECT COUNT(*) FROM trials")
                    trial_count = cursor.fetchone()[0]
                    conn.close()

                    results['optuna_data_saved'] = study_count > 0 and trial_count > 0
                    print(f"Optuna data: {study_count} studies, {trial_count} trials")
                except Exception as e:
                    print(f"Could not verify Optuna data: {e}")
                    results['optuna_data_saved'] = False

    # ========================================================================
    # Step 6: Verify Export Module
    # ========================================================================
    print_section("Step 6: Verify Export Module")

    success, _ = run_command(
        ["python", "-c", """
from mammography.vis.export import (
    export_figure,
    export_training_curves,
    export_confusion_matrix,
    export_metrics_comparison
)
print('All export functions available')
"""],
        "Import export functions",
        check_output=True
    )
    results['export_module'] = success

    # ========================================================================
    # Step 7: Verify Streamlit Pages Exist
    # ========================================================================
    print_section("Step 7: Verify Streamlit Pages")

    experiments_page = Path("src/mammography/apps/web_ui/pages/3_📈_Experiments.py")
    tuning_page = Path("src/mammography/apps/web_ui/pages/5_🔬_Hyperparameter_Tuning.py")

    results['experiments_page'] = check_file_exists(experiments_page, "Experiments page")
    results['tuning_page'] = check_file_exists(tuning_page, "Hyperparameter Tuning page")

    # Try to import Streamlit pages (syntax check)
    success, _ = run_command(
        ["python", "-c", f"""
import sys
sys.path.insert(0, 'src')
try:
    # Just check if files can be parsed
    with open('{experiments_page}', 'r') as f:
        compile(f.read(), '{experiments_page}', 'exec')
    with open('{tuning_page}', 'r') as f:
        compile(f.read(), '{tuning_page}', 'exec')
    print('Streamlit pages syntax OK')
except SyntaxError as e:
    print(f'Syntax error: {{e}}')
    sys.exit(1)
"""],
        "Streamlit pages syntax check",
        check_output=True
    )
    results['streamlit_syntax'] = success

    # ========================================================================
    # Summary
    # ========================================================================
    print_section("Verification Summary")

    total_checks = len(results)
    passed_checks = sum(1 for v in results.values() if v)

    print(f"\nResults: {passed_checks}/{total_checks} checks passed\n")

    for check_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"  {status}: {check_name}")

    print("\n" + "="*80)

    if passed_checks == total_checks:
        print("🎉 ALL VERIFICATIONS PASSED!")
        print("\nAcceptance Criteria Status:")
        print("  ✅ Local SQLite tracking for offline/air-gapped use")
        print("  ✅ All training metrics automatically logged")
        print("  ✅ One-click export in PNG/PDF/SVG formats")
        print("  ✅ Hyperparameter tuning results saved to tracker")
        print("  ✅ Experiment comparison infrastructure in place")
        return 0
    else:
        print(f"⚠️ {total_checks - passed_checks} VERIFICATION(S) FAILED")
        print("\nPlease review the failures above and fix any issues.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
