# Project Folder Documentation

Last updated: 2026-04-21
Project root: `C:\Users\PC\Desktop\PFE\bike_demand_project`

## Scope
This document summarizes the current repository structure, main components, and artifact locations.

Note: Three directories returned access errors during scan and were not inspectable:
- `.pytest_cache`
- `tmpqd_r_zcc`
- `tmpx49g_bu2`

## Top-Level Structure
- `.github/`: CI workflow configuration.
- `.ipynb_checkpoints/`: Jupyter notebook checkpoint artifacts.
- `.pytest_cache/`: pytest cache data (access denied during scan).
- `app/`: Flask application code (blueprints, services, templates, static assets).
- `data/`: raw and sample data files.
- `generated/`: runtime-generated outputs (for example prediction history).
- `models/`: trained model artifacts and metrics.
- `notebooks/`: exploration notebooks and exported notebook outputs.
- `Reports/`: thesis and project reporting materials (PDF/DOCX/PPTX/images).
- `scripts/`: operational scripts for startup/deployment.
- `tests/`: unit and integration tests.
- `tmpqd_r_zcc/`: temporary directory (access denied).
- `tmpx49g_bu2/`: temporary directory (access denied).
- `training/`: legacy training scripts kept for archival/reference.
- `__pycache__/`: Python bytecode cache.

## File Count by Main Directory
- `.github`: 1 file
- `app`: 63 files
- `data`: 7 files
- `generated`: 1 file
- `models`: 35 files
- `notebooks`: 8 files
- `Reports`: 10 files
- `scripts`: 1 file
- `tests`: 26 files
- `training`: 9 files

## Repository File-Type Snapshot
- `.py`: 52
- `.pyc`: 43
- `.json`: 23
- `.pkl`: 15
- `.csv`: 8
- `.html`: 6
- `.png`: 5
- `.js`: 4
- `.pdf`: 3
- `.ipynb`: 3
- Other supporting types: `.css`, `.yml`, `.docx`, `.pptx`, `.sh`, `.ico`, `.ini`, `.md`

## Key Application Areas
### 1) Web App (`app/`)
- `app/__init__.py`: app factory and initialization.
- `app/blueprints/`: route modules (`prediction`, `analytics`, `features`, `export`, `pages`, legacy compatibility).
- `app/services/`: domain logic (`prediction_service`, `input_normalization`, `export_service`).
- `app/models/loader.py`: model artifact loading layer.
- `app/templates/`: HTML views (`index`, `analytics`, `status`, `feature_engineering`, `test`).
- `app/static/`: frontend assets (`style.css`, JS modules, favicon).

### 2) Model Artifacts (`models/`)
- Multiple standalone model files (`random_forest`, `xgboost`, `lightgbm`, `stacking`, `timeseries`) and metrics JSONs.
- `models/unified_pipeline/`: canonical deployment artifacts including:
  - `artifact_manifest.json`
  - model binaries (`*.pkl`)
  - feature/scaler/support files (`feature_names.json`, `scaler.pkl`, `lag_medians.json`)
  - `metrics/all_models_comparison.json`

### 3) Data (`data/`)
- `data/raw/seoul_bike.csv`: primary raw dataset.
- sample prediction input/output files in CSV and JSON for testing/demo.
- model comparison exports (`model_comparison.csv`, `model_comparison.png`).

### 4) Runtime/Operations
- `run.py`: local runtime entrypoint.
- `train_unified_pipeline.py`: supported unified training pipeline.
- `Dockerfile` and `docker-compose.yml`: containerized deployment.
- `scripts/start.sh`: startup entry script.

### 5) Quality and Testing
- `tests/unit/`: unit coverage for preprocessing, services, artifact contract, runtime entrypoint, feature engineering, prediction history.
- `tests/integration/`: API/page integration checks.
- `tests/test_environment.py`: environment/package consistency checks.

### 6) Documentation and Research Assets
- `README.md`: primary project usage guide.
- `Reports/`: thesis proposal docs, final report assets, screenshots, presentation file.
- `notebooks/`: exploratory workflow artifacts.

## Main Root Files
- `README.md`
- `requirements.txt`
- `pytest.ini`
- `Dockerfile`
- `docker-compose.yml`
- `run.py`
- `train_unified_pipeline.py`
- `create_presentation.py`

## Suggested Maintenance Actions
- Remove or ignore stale caches (`__pycache__`, notebook checkpoints) in shared branches if not needed.
- Keep only active artifacts under `models/unified_pipeline/` for production deployment clarity.
- Decide whether root test scripts (`test_api.py`, `test_new_models.py`) should move fully under `tests/`.
- Confirm whether temporary directories (`tmpqd_r_zcc`, `tmpx49g_bu2`) can be cleaned safely.
