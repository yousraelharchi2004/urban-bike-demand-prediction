# Bike Rental Demand Prediction System

Single-source architecture centered on modular Flask blueprints (`app/blueprints/`), a services layer (`app/services/`), and the supported `train_unified_pipeline.py`. Legacy training scripts and debug utilities were removed to keep the repo focused on deployable artifacts.

## Directory reality

```
bike_demand_project/
├── app/
│   ├── __init__.py               # Application factory (requires SECRET_KEY)
│   ├── routes.py                 # Legacy compatibility shim exposing modular blueprints
│   ├── blueprints/               # Production endpoints (prediction, analytics, export, features)
│   ├── services/                 # Business logic (prediction_use, export helpers)
│   ├── utils.py                  # Legacy preprocessing, history, unified pipeline loader
│   ├── email_service.py          # Email with safe failure when config missing
│   └── models/loader.py          # Lazy artifact loader
├── models/
│   ├── unified_pipeline/         # Canonical artifacts + `artifact_manifest.json`
│   └── random_forest_model.pkl   # Legacy pickles retained for compatibility
├── scripts/
│   └── start.sh                  # Docker-friendly entrypoint verifying artifacts
├── train_unified_pipeline.py     # Supported training pipeline (use this)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── run.py                        # Browser-opening entrypoint that checks artifact/secret hygiene
└── README.md
```

## Key hygiene points
- **Secrets**: `app/__init__.py` refuses to run without `SECRET_KEY`. `.env` is ignored by git and should only live locally; use `.env.example` as a template.
- **Email**: `app/email_service.py` logs a warning when SMTP credentials are missing and returns a controlled error instead of attempting a send.
- **Artifacts**: `models/unified_pipeline/artifact_manifest.json` drives what the app loads. The Docker entrypoint (`scripts/start.sh`) verifies the manifest and only then starts Gunicorn.
- **Generated state**: Prediction history and exports now live under `generated/` (ignored). Set `PREDICTION_HISTORY_FILE` if you want to relocate the file.
- **Working directory**: `run.py` and `app/utils.py` resolve all paths relative to their own files, not the current working directory, so you can launch the app from anywhere.

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Copy `.env.example` → `.env`, set `SECRET_KEY`, SMTP credentials, and keep that file local (it's ignored by git).
3. Ensure `models/unified_pipeline/artifact_manifest.json` lists the same `joblib`, `numpy`, and `scikit-learn` versions that are installed in your environment.
4. Run the app:
   ```bash
   python run.py
   ```
   The browser opens automatically to `http://127.0.0.1:5000/v2` and the UI/API endpoints live under `/v2` (for example `/v2/status`, `/v2/predict`, `/v2/api/analytics/kpis`).

## Docker deployment

- Build and run with `docker-compose up --build`. The Dockerfile now creates a non-root `appuser`, writes runtime state under `/app/generated`, and executes the verified `scripts/start.sh` entrypoint.
- `docker-compose.yml` includes a `/v2/status` healthcheck, so orchestrators know when Gunicorn is ready.
- Set `STRICT_ARTIFACT_CHECK=1` to fail startup when the manifest dependencies no longer match the runtime; otherwise, the validator logs a warning but keeps running.

## Training

The only supported script is `train_unified_pipeline.py`. Discarded legacy training scripts (the `training/legacy/` folder and the removed top-level `train_*` helpers) remain only for archival reference.

## Testing

```bash
pytest tests/
```
The suite now includes `tests/test_environment.py`, which validates that the active Python packages (scikit-learn, pandas, numpy, joblib, xgboost, lightgbm) match the pinned versions in `requirements.txt`. Re-run `pip install -r requirements.txt` if the test reports drift before continuing.
