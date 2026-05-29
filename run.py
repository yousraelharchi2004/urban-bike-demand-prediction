#!/usr/bin/env python3
"""
Bike Rental Demand Prediction System
Urban shared bike rental demand prediction and analysis by multimodal features.
"""

import os
import secrets
import sys
import threading
import webbrowser
from importlib import metadata
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent

# Load environment variables from project root .env
load_dotenv(dotenv_path=ROOT / '.env')


def ensure_secret_key():
    """Ensure SECRET_KEY exists (generate a temporary one for local runs if needed)."""
    if os.environ.get("SECRET_KEY"):
        return
    temp_key = secrets.token_urlsafe(32)
    os.environ["SECRET_KEY"] = temp_key
    print("[INFO] SECRET_KEY was not set; generated a temporary key for this run (set persistently in .env or the env for production).")


ensure_secret_key()


def open_browser(url: str):
    """Open the application URL in the default browser on startup."""

    def _open():
        try:
            webbrowser.open_new(url)
        except Exception as exc:
            print(f"[WARN] Could not open browser automatically: {exc}")

    threading.Timer(1.0, _open).start()


def check_dependencies():
    """Check if the application structure and core artifacts exist."""
    print("[CHECK] Checking project structure...")

    required_folders = [
        ROOT / 'app',
        ROOT / 'app' / 'blueprints',
        ROOT / 'app' / 'services',
        ROOT / 'app' / 'templates',
        ROOT / 'app' / 'static',
        ROOT / 'models',
        ROOT / 'data' / 'raw',
        ROOT / 'generated',
    ]

    all_good = True
    for folder in required_folders:
        if folder.exists():
            print(f"  [OK] {folder.relative_to(ROOT)}")
        else:
            print(f"  [MISSING] {folder.relative_to(ROOT)} - Missing!")
            all_good = False

    required_files = {
        'Application factory': ROOT / 'app' / '__init__.py',
        'Pages blueprint': ROOT / 'app' / 'blueprints' / 'pages.py',
        'Prediction blueprint': ROOT / 'app' / 'blueprints' / 'prediction.py',
        'Analytics blueprint': ROOT / 'app' / 'blueprints' / 'analytics.py',
        'Export blueprint': ROOT / 'app' / 'blueprints' / 'export.py',
        'Features blueprint': ROOT / 'app' / 'blueprints' / 'features.py',
        'Prediction service': ROOT / 'app' / 'services' / 'prediction_service.py',
        'Dataset': ROOT / 'data' / 'raw' / 'seoul_bike.csv',
        'Main template': ROOT / 'app' / 'templates' / 'index.html',
        'Utils': ROOT / 'app' / 'utils.py',
        'Unified artifact manifest': ROOT / 'models' / 'unified_pipeline' / 'artifact_manifest.json',
    }

    for desc, filepath in required_files.items():
        if filepath.exists():
            print(f"  [OK] {desc}: {filepath}")
        else:
            print(f"  [MISSING] {desc}: {filepath} - Missing!")
            all_good = False

    return all_good


def _print_package_status(package_name: str, label: str | None = None):
    """Print whether a Python package is available and report its version."""
    display = label or package_name
    try:
        version = metadata.version(package_name)
        print(f"  [OK] {display}: {version}")
    except metadata.PackageNotFoundError:
        print(f"  [MISSING] {display} not installed")
    except Exception as exc:
        print(f"  [WARN] Could not inspect {display}: {exc}")


def print_system_info():
    """Print system and project information."""
    print("\n" + "=" * 70)
    print("[BIKE] URBAN BIKE RENTAL DEMAND PREDICTION SYSTEM")
    print("[GRAD] Graduation Project - Multimodal Feature Analysis")
    print("=" * 70)

    print("\n" + "=" * 70)
    print("[STATS] PROJECT INFORMATION")
    print("=" * 70)
    print(f"[DIR] Project root: {ROOT}")
    print(f"[PY] Python version: {sys.version.split()[0]}")

    print("\n[PKG] Checking Python packages...")
    for package_name, label in (
        ("flask", "Flask"),
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("scikit-learn", "Scikit-learn"),
        ("joblib", "joblib"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
    ):
        _print_package_status(package_name, label)

    try:
        from app.config import get_runtime_environment_report

        report = get_runtime_environment_report()
        print("\n[ARTIFACT] Unified artifact runtime contract:")
        for package, version in sorted(report['runtime_dependencies'].items()):
            print(f"  [OK] {package}: {version}")
        if report['artifact_dependency_error']:
            print(f"  [MISMATCH] {report['artifact_dependency_error']}")
        else:
            print("  [OK] Artifact dependency versions match the deployed runtime")
    except Exception as exc:
        print(f"  [WARN] Could not validate artifact dependency versions: {exc}")


def show_app_model_overview():
    """Display app-level model availability and key metrics."""
    print("\n" + "=" * 70)
    print("[MODEL] MODEL INFORMATION")
    print("=" * 70)

    try:
        from app.model_metrics import calculate_model_metrics, get_model_catalog
        from app.models.loader import get_state
        from app.utils import load_unified_pipeline

        state = get_state()
        catalog = get_model_catalog()
        metrics = calculate_model_metrics()
        unified_loaded = load_unified_pipeline() is not None
        state_flags = {
            'original': state.get('original_model') is not None,
            'improved': state.get('improved_model') is not None,
            'advanced': state.get('advanced_model') is not None,
            'xgboost': state.get('xgboost_model') is not None,
            'lightgbm': state.get('lightgbm_model') is not None,
            'stacking': state.get('stacking_model') is not None,
            'unified': unified_loaded,
        }

        print("[INFO] Application model inventory:")
        for model_key in ('original', 'improved', 'advanced', 'xgboost', 'lightgbm', 'stacking', 'unified'):
            entry = catalog.get(model_key, {})
            if state_flags.get(model_key):
                status = "LOADED"
            elif entry.get('model_available'):
                status = "ARTIFACT ONLY"
            else:
                status = "UNAVAILABLE"

            print(
                f"  - {model_key:<9} {status:<12}"
                f" R2={entry.get('r2', 0.0):.3f}"
                f" RMSE={entry.get('rmse', 0.0):.1f}"
                f" Features={entry.get('features', 0)}"
            )

        best_model_key = metrics.get('best_model', 'original')
        best_entry = catalog.get(best_model_key, {})
        print("\n[INFO] Best available metrics summary:")
        print(f"  - Best model: {best_entry.get('name', best_model_key)} ({best_model_key})")
        print(f"  - R2 score:   {best_entry.get('r2', 0.0):.3f}")
        print(f"  - MAE:        {best_entry.get('mae', 0.0):.1f} bikes")
        print(f"  - RMSE:       {best_entry.get('rmse', 0.0):.1f} bikes")
        print(f"  - Source:     {best_entry.get('artifact_path', 'N/A')}")

        print("\n[INFO] App focus:")
        print("  - Multiple model families are exposed through the web application")
        print("  - Unified pipeline artifacts are supported, but not the only app path")
        print("  - Supported training entrypoint: train_unified_pipeline.py")
    except ImportError as exc:
        print(f"[STATS] Model metrics module not available: {exc}")
    except Exception as exc:
        print(f"[STATS] Could not load app model overview: {exc}")


def print_available_endpoints():
    """Print the primary pages and API routes exposed by the app."""
    print("\n[INFO] Available pages:")
    for label, url in (
        ("Homepage (v2)", "http://127.0.0.1:5000/v2"),
        ("Test Page", "http://127.0.0.1:5000/v2/test"),
        ("Status Page", "http://127.0.0.1:5000/v2/status"),
        ("Analytics Dashboard", "http://127.0.0.1:5000/v2/analytics"),
        ("Feature Engineering", "http://127.0.0.1:5000/v2/feature-engineering"),
        ("Legacy homepage", "http://127.0.0.1:5000/"),
    ):
        print(f"   - {label:<20} {url}")

    print("\n[INFO] Available API endpoints:")
    for label, url in (
        ("Status JSON", "http://127.0.0.1:5000/v2/api/status"),
        ("Models Catalog", "http://127.0.0.1:5000/v2/api/models"),
        ("Model Info", "http://127.0.0.1:5000/v2/model-info"),
        ("Metrics", "http://127.0.0.1:5000/v2/metrics"),
        ("Predict", "http://127.0.0.1:5000/v2/predict"),
        ("Compare Predictions", "http://127.0.0.1:5000/v2/predict/compare"),
        ("Prediction History", "http://127.0.0.1:5000/v2/prediction-history"),
        ("Dashboard KPIs", "http://127.0.0.1:5000/v2/api/analytics/kpis"),
        ("Hourly Patterns", "http://127.0.0.1:5000/v2/api/analytics/hourly-patterns"),
        ("Feature List", "http://127.0.0.1:5000/v2/api/features/advanced?model_type=advanced"),
        ("Feature Importance", "http://127.0.0.1:5000/v2/api/features/importance?model_type=xgboost"),
    ):
        print(f"   - {label:<20} {url}")


def print_quick_test_command():
    """Print a sample request aligned with the current prediction API."""
    print("\n[NOTE] Quick test command:")
    print('   curl -X POST "http://127.0.0.1:5000/v2/predict?model_type=xgboost" \\')
    print('     -H "Content-Type: application/json" \\')
    print("     -d '{")
    print('       "Hour": 14,')
    print('       "Temperature(°C)": 22.5,')
    print('       "Humidity(%)": 60,')
    print('       "Wind speed (m/s)": 2.5,')
    print('       "Visibility (10m)": 2000,')
    print('       "Dew point temperature(°C)": 10,')
    print('       "Solar Radiation (MJ/m2)": 1.5,')
    print('       "Rainfall(mm)": 0,')
    print('       "Snowfall (cm)": 0,')
    print('       "Seasons_Spring": 0,')
    print('       "Seasons_Summer": 1,')
    print('       "Seasons_Winter": 0,')
    print('       "Holiday_No Holiday": 1,')
    print('       "Functioning Day_Yes": 1')
    print("     }'")


def main():
    """Main entry point for the application."""
    if not check_dependencies():
        print("\n[FAIL] Required files are missing. Startup aborted.")
        print("   Please verify:")
        print("   1. The app blueprints, services, templates, and static files exist")
        print("   2. The dataset is present in data/raw/")
        print("   3. The model artifacts required by your demo are present in models/")
        sys.exit(1)

    print_system_info()
    show_app_model_overview()

    try:
        from app import create_app
        from app.config import validate_runtime_environment

        validate_runtime_environment()
        app = create_app()

        print("\n" + "=" * 70)
        print("[OK] APPLICATION READY")
        print("=" * 70)
        debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')

        print("[WEB] Server will listen on: http://127.0.0.1:5000")
        print_available_endpoints()

        print("\n[TOOLS] Development tools:")
        print(f"   - Debug mode:      {'ON' if debug_mode else 'OFF'}")
        print(f"   - Auto-reload:     {'ON' if debug_mode else 'OFF'}")
        print("   - Console output:  Detailed logs")

        print_quick_test_command()

        print("\n" + "=" * 70)
        print("[START] Starting Flask application...")
        print("=" * 70)
        print("Press Ctrl+C to stop the server\n")

        if (not debug_mode) or os.environ.get('WERKZEUG_RUN_MAIN') == 'true':
            open_browser("http://127.0.0.1:5000/v2")
        app.run(debug=debug_mode, host='0.0.0.0', port=5000, use_reloader=debug_mode)

    except Exception as exc:
        print(f"\n[FAIL] Failed to start application: {exc}")
        print("\n[TOOLS] Troubleshooting steps:")
        print("   1. Check if all dependencies are installed")
        print("   2. Verify the model artifacts required by the app exist")
        print("   3. Check app/__init__.py and the blueprint modules for syntax errors")
        import traceback
        traceback.print_exc()
        sys.exit(1)


from app import create_app

app = create_app()


if __name__ == '__main__':
    main()
