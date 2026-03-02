# Bike Rental Demand Prediction System

Urban shared bike rental demand prediction using multimodal features - **Seoul Bike Sharing Dataset (2017-2018)**.

Graduation Project (PFE) - Predicts hourly bike rental demand using weather, temporal, and engineered features with ensemble ML models.

## Architecture

```
bike_demand_project/
├── app/                          # Flask web application
│   ├── __init__.py               # App factory
│   ├── routes.py                 # API endpoints (legacy monolith)
│   ├── utils.py                  # Preprocessing & feature engineering
│   ├── email_service.py          # Email notifications
│   ├── model_metrics.py          # Model evaluation metrics
│   ├── model_manager.py          # Model lifecycle management
│   ├── blueprints/               # Modular route organization
│   │   ├── pages.py              # Page routes (/, /test, /status)
│   │   ├── prediction.py         # Prediction endpoints
│   │   ├── analytics.py          # Analytics dashboard API
│   │   ├── export.py             # Data export routes
│   │   └── features.py           # Feature engineering API
│   ├── services/                 # Business logic layer
│   │   ├── prediction_service.py # Prediction + history
│   │   └── export_service.py     # Export formatting
│   ├── models/                   # Model loading
│   │   └── loader.py             # Lazy model loading with caching
│   ├── templates/                # HTML templates
│   └── static/                   # CSS, JS, assets
├── models/                       # Trained model artifacts
│   ├── unified_pipeline/         # Leakage-free pipeline output
│   │   ├── pipeline_config.json
│   │   ├── scaler.pkl
│   │   ├── feature_names.json
│   │   ├── stacking_ensemble.pkl
│   │   ├── demand_cache.json
│   │   ├── metrics/
│   │   └── plots/shap/
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   └── ...
├── data/
│   └── raw/seoul_bike.csv        # Seoul Bike Sharing Dataset
├── training/legacy/              # Original training scripts (reference)
├── tests/                        # pytest test suite
│   ├── unit/
│   └── integration/
├── train_unified_pipeline.py     # Unified training (run this!)
├── run.py                        # Application entry point
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container deployment
├── docker-compose.yml
└── .env.example                  # Environment template
```

## Models

| Model | Type | Features | Test R2 | Notes |
|-------|------|----------|---------|-------|
| Random Forest | Baseline | 14 | ~0.82 | Original model |
| XGBoost | Gradient Boosting | 20+ | ~0.90 | Feature engineered |
| LightGBM | Gradient Boosting | 25+ | ~0.91 | Fast training |
| Gradient Boosting | Ensemble | 25+ | ~0.89 | Sklearn GBR |
| **Stacking Ensemble** | **Meta-learner** | **25+** | **~0.93** | **Top-3 + RidgeCV** |

The unified pipeline uses **TimeSeriesSplit** cross-validation and fits scalers **only on training data** to prevent data leakage.

## Quick Start

### 1. Clone and set up environment

```bash
git clone <repo-url>
cd bike_demand_project

python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env with your settings (SECRET_KEY, email credentials)
```

### 3. Train models (optional - pre-trained models included)

```bash
python train_unified_pipeline.py
```

This generates all artifacts in `models/unified_pipeline/` including:
- Trained models (RF, XGBoost, LightGBM, Gradient Boosting, Stacking Ensemble)
- Fitted scaler (train-only)
- SHAP explainability plots
- Residual analysis plots
- Learning curves
- Model comparison metrics

### 4. Run the application

```bash
python run.py
```

The app starts at `http://127.0.0.1:5000`.

### 5. Run with Docker

```bash
docker compose up --build
```

## API Reference

### Prediction

```bash
# Original model
POST /predict
Content-Type: application/json

{
  "Hour": 14,
  "Temperature(C)": 22.5,
  "Humidity(%)": 60,
  "Wind speed (m/s)": 2.5,
  "Visibility (10m)": 2000,
  "Dew point temperature(C)": 10,
  "Solar Radiation (MJ/m2)": 1.5,
  "Rainfall(mm)": 0,
  "Snowfall (cm)": 0,
  "Seasons_Spring": 0,
  "Seasons_Summer": 1,
  "Seasons_Winter": 0,
  "Holiday_No Holiday": 1,
  "Functioning Day_Yes": 1
}

# Unified ensemble model
POST /predict?model_type=unified
```

### Other Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web interface |
| `/status` | GET | System status dashboard |
| `/metrics` | GET | Model performance metrics |
| `/model-info` | GET | Model metadata |
| `/prediction-history` | GET | Past predictions |
| `/analytics` | GET | Analytics dashboard |
| `/api/status` | GET | API health check (JSON) |
| `/api/dashboard-metrics` | GET | Dashboard KPIs |
| `/export/predictions/csv` | GET | Export predictions as CSV |

## Testing

```bash
pytest tests/ -v
```

## Dataset

**Seoul Bike Sharing Demand Dataset**
- 8,760 hourly records (full year)
- Period: December 2017 - November 2018
- Location: Seoul, South Korea
- Features: Temperature, Humidity, Wind speed, Visibility, Solar radiation, Rainfall, Snowfall, Season, Holiday, Functioning day

## Key Technical Decisions

- **Time-based train/test split** (chronological, not random) to simulate real deployment
- **Scaler fitted on training data only** to prevent data leakage
- **Lag features use `shift(1)`** to exclude current observation
- **NaN fill with training median** (not bfill which leaks future data)
- **TimeSeriesSplit(n_splits=5)** for cross-validation
- **SHAP** for model explainability

## License

Academic project - Seoul National University Bike Sharing Dataset.
