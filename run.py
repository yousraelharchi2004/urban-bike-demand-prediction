#!/usr/bin/env python3
"""
Bike Rental Demand Prediction System
Urban shared bike rental demand prediction and analysis by multimodal features
"""

import os
import secrets
import sys
import webbrowser
from pathlib import Path
from threading import Timer
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


def check_dependencies():
    """Check if all required files and folders exist"""
    print("[CHECK] Checking project structure...")

    required_folders = [
        ROOT / 'app',
        ROOT / 'app/templates',
        ROOT / 'app/static',
        ROOT / 'models',
        ROOT / 'data/raw',
    ]
    for folder in required_folders:
        if folder.exists():
            print(f"  [OK] {folder.relative_to(ROOT)}")
        else:
            print(f"  [MISSING] {folder.relative_to(ROOT)} - Missing!")
            return False
    
    # Check required files
    required_files = {
        'Supported artifact manifest': ROOT / 'models' / 'unified_pipeline' / 'artifact_manifest.json',
        'Dataset': ROOT / 'data' / 'raw' / 'seoul_bike.csv',
        'Main template': ROOT / 'app' / 'templates' / 'index.html',
        'Application factory': ROOT / 'app' / '__init__.py',
        'Prediction blueprint': ROOT / 'app' / 'blueprints' / 'prediction.py',
        'Utils': ROOT / 'app' / 'utils.py',
    }
    
    all_good = True
    for desc, filepath in required_files.items():
        if os.path.exists(filepath):
            print(f"  [OK] {desc}: {filepath}")
        else:
            print(f"  [MISSING] {desc}: {filepath} - Missing!")
            all_good = False
    
    return all_good

def open_browser():
    """Open the default browser to the main page (development only)."""
    try:
        Timer(1.5, lambda: webbrowser.open('http://127.0.0.1:5000')).start()
        print("[WEB] Opening browser automatically...")
    except Exception:
        print("[INFO] Please open the browser manually to: http://127.0.0.1:5000")

def print_system_info():
    """Print system and project information"""
    print("\n" + "="*70)
    print("[BIKE] URBAN BIKE RENTAL DEMAND PREDICTION SYSTEM")
    print("[GRAD] Graduation Project - Multimodal Feature Analysis")
    print("="*70)
    
    print("\n" + "="*70)
    print("[STATS] PROJECT INFORMATION")
    print("="*70)
    print(f"[DIR] Project root: {os.path.abspath('.')}")
    print(f"[PY] Python version: {sys.version.split()[0]}")
    
    # Check Python packages
    print("\n[PKG] Checking Python packages...")
    try:
        import importlib.metadata
        flask_version = importlib.metadata.version("flask")
        print(f"  [OK] Flask: {flask_version}")
    except Exception:
        print("  [MISSING] Flask not installed")
    
    try:
        import importlib.metadata
        pandas_version = importlib.metadata.version("pandas")
        print(f"  [OK] Pandas: {pandas_version}")
    except Exception:
        print("  [MISSING] Pandas not installed")
    
    try:
        import importlib.metadata
        sklearn_version = importlib.metadata.version("scikit-learn")
        print(f"  [OK] Scikit-learn: {sklearn_version}")
    except Exception:
        print("  [MISSING] Scikit-learn not installed")

    try:
        from app.config import get_runtime_environment_report

        report = get_runtime_environment_report()
        print("\n[ARTIFACT] Supported artifact runtime contract:")
        for package, version in sorted(report['runtime_dependencies'].items()):
            print(f"  [OK] {package}: {version}")
        if report['artifact_dependency_error']:
            print(f"  [MISMATCH] {report['artifact_dependency_error']}")
        else:
            print("  [OK] Artifact dependency versions match the deployed runtime")
    except Exception as e:
        print(f"  [WARN] Could not validate artifact dependency versions: {e}")

def show_model_metrics():
    """Display model metrics without loading the model"""
    print("\n" + "="*70)
    print("[MODEL] MODEL INFORMATION")
    print("="*70)
    
    # Just show basic model info without loading
    print("[INFO] Supported artifact contract: models/unified_pipeline/artifact_manifest.json")
    print("[INFO] Supported training entrypoint: train_unified_pipeline.py")
    
    # Try to load and display metrics from the model_metrics module
    try:
        from app.model_metrics import generate_performance_report
        print(generate_performance_report())
    except ImportError as e:
        print(f"[STATS] Model metrics module not available: {e}")
    except Exception as e:
        print(f"[STATS] Could not load performance metrics: {e}")

def main():
    """Main entry point for the application"""
    
    # Check project structure
    if not check_dependencies():
        print("\n[FAIL] Required files are missing. Startup aborted.")
        print("   Please verify:")
        print("   1. Trained the supported artifacts into models/")
        print("   2. Placed the dataset in data/raw/")
        print("   3. Created the required templates")
        sys.exit(1)
    
    # Print system info
    print_system_info()
    
    # Show model metrics (without loading model)
    show_model_metrics()
    
    # Create and configure the app
    try:
        from app import create_app
        from app.config import validate_runtime_environment

        validate_runtime_environment()
        app = create_app()
        
        print("\n" + "="*70)
        print("[OK] APPLICATION READY")
        print("="*70)
        print("[WEB] Server will listen on: http://127.0.0.1:5000")
        print("\n[INFO] Available endpoints:")
        print("   - Homepage:        http://127.0.0.1:5000/")
        print("   - API Status:      http://127.0.0.1:5000/status")
        print("   - Model Info:      http://127.0.0.1:5000/model-info")
        print("   - Model Metrics:   http://127.0.0.1:5000/metrics")
        print("   - Prediction History: http://127.0.0.1:5000/prediction-history")
        print("   - Test Page:       http://127.0.0.1:5000/test")
        print("   - API Prediction:  http://127.0.0.1:5000/predict (POST)")
        
        print("\n[TOOLS] Development tools:")
        print("   - Debug mode:      ON")
        print("   - Auto-reload:     ON")
        print("   - Console output:  Detailed logs")
        
        print("\n[NOTE] Quick test command:")
        print('   curl -X POST http://127.0.0.1:5000/predict \\')
        print('     -H "Content-Type: application/json" \\')
        print('     -d \'{')
        print('       "Hour": 14,')
        print('       "Temperature(C)": 22.5,')
        print('       "Humidity(%)": 60,')
        print('       "Wind speed (m/s)": 2.5,')
        print('       "Visibility (10m)": 2000,')
        print('       "Dew point temperature(C)": 10,')
        print('       "Solar Radiation (MJ/m2)": 1.5,')
        print('       "Rainfall(mm)": 0,')
        print('       "Snowfall (cm)": 0,')
        print('       "Seasons_Spring": 0,')
        print('       "Seasons_Summer": 1,')
        print('       "Seasons_Winter": 0,')
        print('       "Holiday_No Holiday": 1,')
        print('       "Functioning Day_Yes": 1')
        print('     }\'')
        
        print("\n" + "="*70)
        print("[START] Starting Flask application...")
        print("="*70)
        print("Press Ctrl+C to stop the server\n")

        open_browser()
        # Run the app
        debug_mode = os.environ.get('FLASK_DEBUG', 'False').lower() in ('true', '1', 'yes')
        app.run(debug=debug_mode, host='0.0.0.0', port=5000, use_reloader=debug_mode)
        
    except Exception as e:
        print(f"\n[FAIL] Failed to start application: {e}")
        print("\n[TOOLS] Troubleshooting steps:")
        print("   1. Check if all dependencies are installed")
        print("   2. Verify the model file exists")
        print("   3. Check app/__init__.py for syntax errors")
        import traceback
        traceback.print_exc()
        sys.exit(1)

# Create app instance for gunicorn: gunicorn run:app
from app import create_app
app = create_app()

if __name__ == '__main__':
    main()
